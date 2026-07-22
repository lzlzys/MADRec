#!/usr/bin/env python3
"""
水平三智能体辩论（顺序发言 + 从众/妥协 + Summarizer 集成），与 chat_RecAct_debate 共用 RecEnv / llm_chat。

代码结构对齐 chat_RecAct_debate.py：
  HorizontalDebateAgentRegistry + BaseHorizontalPeerAgent（注册各 peer / summarizer）
  HorizontalDebateSimulation：调度 Round1 A→B→C、Round2 A→B→C、Round3 Summarizer，产出 finish 内层文本。

Round 1 — 独立陈述（顺序 A → B → C）
Round 2 — 交叉评审与妥协（顺序 A → B → C）
Round 3 — Summarizer → finish[Final_List]

用法：
  python chat_RecAct_horizontal_debate.py
  python chat_RecAct_horizontal_debate.py --prefetch-topk 30 --prefetch-condition None
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from chat_api import llm_chat

import chat_recEnv
import chat_recWrappers
from utils import *

env = chat_recEnv.RecEnv()
env = chat_recWrappers.reActWrapper(env)
env = chat_recWrappers.LoggingWrapper(env)


def step_env(env, action):
    attemps = 0
    while attemps < 10:
        try:
            return env.step(action)
        except Exception as e:
            print("An error occurred:", str(e))
            attemps += 1


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Cycle start.")
parser.add_argument("--step_num", type=int, default=100, help="Cycle step number.")
parser.add_argument(
    "--horizontal-sleep",
    type=float,
    default=1.0,
    help="每次 LLM 调用后的休眠秒数，减轻限流。",
)
parser.add_argument(
    "--prefetch-topk",
    type=int,
    default=30,
    help="辩论前 Retrieve 的候选池大小；0 表示不预检索（不推荐）。",
)
parser.add_argument(
    "--prefetch-condition",
    type=str,
    default="None",
    help="预检索 retrieve[condition, K] 的 condition（如 None、genre）。",
)
parser.add_argument(
    "--max-debate-rounds",
    type=int,
    default=3,
    help="辩论轮数上限（仅用于日志/未来扩展；当前固定为 3 阶段）。",
)
args, _ = parser.parse_known_args()


def load_hit_target_topk_user_ids():
    path = checkpoint_path + "hit_target_in_topk_user_ids.txt"
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


hit_topk_uids = load_hit_target_topk_user_ids()

task_prompt = prompt_dict["think_sample"]
instruction = prompt_pattern["instruction"]
question_tail = prompt_pattern["task"]

# ---------------------------------------------------------------------------
# 候选池解析 & 排序抽取
# ---------------------------------------------------------------------------


def parse_retrieval_pool(obs: str) -> Tuple[Dict[str, str], List[str]]:
    """从 CRS retrieve 的 Observation 中解析 item_id -> 标题，以及池内顺序。"""
    s = (obs or "").strip().replace("\\n", "\n")
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    id_to_title: Dict[str, str] = {}
    ordered: List[str] = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(", ", 2)
        if len(parts) < 2:
            continue
        iid = parts[0].strip()
        title = parts[1].strip()
        if not iid:
            continue
        if iid not in item_token_id:
            continue
        if iid not in id_to_title:
            id_to_title[iid] = title
            ordered.append(iid)
    return id_to_title, ordered


def extract_ids_from_text(text: str, pool: Set[str], k: int = 10) -> List[str]:
    found: List[str] = []
    for m in re.finditer(r"\b(\d+)\b", text or ""):
        tid = m.group(1)
        if tid in pool and tid not in found:
            found.append(tid)
        if len(found) >= k:
            break
    return found


def extract_section_ranking(text: str, pool: Set[str], k: int = 10) -> List[str]:
    t = text or ""
    upper = t.upper()
    # Prefer final-aggregation headers so Summarizer Thought can mention MY_RANKING safely.
    final_markers = ("FINAL_TOP10", "FINAL_TOP_10", "CONSENSUS_TOP10")
    final_start = -1
    for mk in final_markers:
        pos = upper.find(mk)
        if pos >= 0 and (final_start < 0 or pos < final_start):
            final_start = pos
    if final_start >= 0:
        sub = t[final_start:]
        return extract_ids_from_text(sub, pool, k=k)
    markers = (
        "MY_RANKING",
        "PROPOSED_RANKING",
        "REVISED_RANKING",
        "ROUND2_STATEMENT",
    )
    start = -1
    for mk in markers:
        pos = upper.find(mk)
        if pos >= 0:
            start = max(start, pos)
    if start >= 0:
        sub = t[start:]
        return extract_ids_from_text(sub, pool, k=k)
    return extract_ids_from_text(t, pool, k=k)


def pad_ranking(ids: List[str], pool_order: List[str], k: int = 10) -> List[str]:
    out = list(ids)
    for pid in pool_order:
        if len(out) >= k:
            break
        if pid not in out:
            out.append(pid)
    return out[:k]


def build_finish_inner(ids: List[str], id_to_title: Dict[str, str]) -> str:
    lines = []
    for iid in ids:
        if iid not in item_token_id:
            continue
        name = id_to_title.get(iid) or itemID_name.get(iid, str(iid))
        lines.append(f"{iid}, {name}, 0.0")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgentVerse 风格：HorizontalDebateAgentRegistry + BaseHorizontalPeerAgent
# （与 chat_RecAct_debate.DebateAgentRegistry / BaseDebateAgent 同构）
# ---------------------------------------------------------------------------

_FALLBACK_HORIZONTAL_A = (
    "You are Agent A (Behavioral): ground recommendations in rating history and genre frequency."
)
_FALLBACK_HORIZONTAL_B = (
    "You are Agent B (Semantic): link candidates to high-rated items by theme, director, or depth."
)
_FALLBACK_HORIZONTAL_C = (
    "You are Agent C (Persona): align picks with user profile (age, occupation, maturity)."
)
_FALLBACK_HORIZONTAL_SUM = (
    "You are the Summarizer: merge peer views into a single FINAL_TOP10 from the candidate pool only."
)


class HorizontalDebateAgentRegistry:
    """与 DebateAgentRegistry 类似：按 agent_type 字符串实例化水平辩论角色。"""

    _entries: Dict[str, Type["BaseHorizontalPeerAgent"]] = {}

    @classmethod
    def register(cls, agent_type: str):
        def _decorator(agent_cls: Type["BaseHorizontalPeerAgent"]):
            cls._entries[agent_type] = agent_cls
            return agent_cls

        return _decorator

    @classmethod
    def build(cls, agent_type: str, **kwargs: Any) -> "BaseHorizontalPeerAgent":
        if agent_type not in cls._entries:
            raise ValueError(
                f'agent_type "{agent_type}" not registered. '
                f"Available: {list(cls._entries.keys())}"
            )
        return cls._entries[agent_type](**kwargs)


class BaseHorizontalPeerAgent:
    """水平辩论 LLM 角色：instruction + ICL，拼 User 消息前缀（对标 BaseDebateAgent）。"""

    agent_type: str = "base"

    def __init__(
        self,
        *,
        name: str,
        instruction: str,
        icl_body: str,
        icl_title: str,
    ):
        self.name = name
        self.instruction = instruction or ""
        self.icl_body = icl_body or ""
        self.icl_title = icl_title

    def system_prefix(self) -> str:
        return self.instruction + self._icl_block(self.icl_title, self.icl_body)

    @staticmethod
    def _icl_block(title: str, body: str) -> str:
        body = (body or "").strip()
        if not body:
            return ""
        return f"=== ICL example ({title}) ===\n{body}\n=== End ICL ===\n\n"


@HorizontalDebateAgentRegistry.register("behavioral_peer")
class BehavioralPeerAgent(BaseHorizontalPeerAgent):
    agent_type = "behavioral_peer"

    def __init__(self, **kwargs: Any):
        super().__init__(
            name=kwargs.get("name", "Agent A · Behavioral"),
            instruction=kwargs.get("instruction", ""),
            icl_body=kwargs.get("icl_body", ""),
            icl_title=kwargs.get("icl_title", "Agent A · Behavioral"),
        )


@HorizontalDebateAgentRegistry.register("semantic_peer")
class SemanticPeerAgent(BaseHorizontalPeerAgent):
    agent_type = "semantic_peer"

    def __init__(self, **kwargs: Any):
        super().__init__(
            name=kwargs.get("name", "Agent B · Semantic"),
            instruction=kwargs.get("instruction", ""),
            icl_body=kwargs.get("icl_body", ""),
            icl_title=kwargs.get("icl_title", "Agent B · Semantic"),
        )


@HorizontalDebateAgentRegistry.register("persona_peer")
class PersonaPeerAgent(BaseHorizontalPeerAgent):
    agent_type = "persona_peer"

    def __init__(self, **kwargs: Any):
        super().__init__(
            name=kwargs.get("name", "Agent C · Persona"),
            instruction=kwargs.get("instruction", ""),
            icl_body=kwargs.get("icl_body", ""),
            icl_title=kwargs.get("icl_title", "Agent C · Persona"),
        )


@HorizontalDebateAgentRegistry.register("horizontal_summarizer")
class HorizontalSummarizerAgent(BaseHorizontalPeerAgent):
    agent_type = "horizontal_summarizer"

    def __init__(self, **kwargs: Any):
        super().__init__(
            name=kwargs.get("name", "Summarizer"),
            instruction=kwargs.get("instruction", ""),
            icl_body=kwargs.get("icl_body", ""),
            icl_title=kwargs.get("icl_title", "Summarizer"),
        )


def load_horizontal_agents_from_pattern() -> Tuple[
    BehavioralPeerAgent,
    SemanticPeerAgent,
    PersonaPeerAgent,
    HorizontalSummarizerAgent,
]:
    """对标 load_debate_agents_from_pattern：从 pattern / prompt_dict 构造四个角色。"""
    inst_a = prompt_pattern.get("horizontal-behavioral-instruction") or _FALLBACK_HORIZONTAL_A
    inst_b = prompt_pattern.get("horizontal-semantic-instruction") or _FALLBACK_HORIZONTAL_B
    inst_c = prompt_pattern.get("horizontal-persona-instruction") or _FALLBACK_HORIZONTAL_C
    inst_s = prompt_pattern.get("horizontal-summarizer-instruction") or _FALLBACK_HORIZONTAL_SUM
    icl_a = prompt_dict.get("horizontal_behavioral_sample") or ""
    icl_b = prompt_dict.get("horizontal_semantic_sample") or ""
    icl_c = prompt_dict.get("horizontal_persona_sample") or ""
    icl_s = prompt_dict.get("horizontal_summarizer_sample") or ""

    agent_a = HorizontalDebateAgentRegistry.build(
        "behavioral_peer",
        instruction=inst_a,
        icl_body=icl_a,
    )
    agent_b = HorizontalDebateAgentRegistry.build(
        "semantic_peer",
        instruction=inst_b,
        icl_body=icl_b,
    )
    agent_c = HorizontalDebateAgentRegistry.build(
        "persona_peer",
        instruction=inst_c,
        icl_body=icl_c,
    )
    summarizer = HorizontalDebateAgentRegistry.build(
        "horizontal_summarizer",
        instruction=inst_s,
        icl_body=icl_s,
    )
    return agent_a, agent_b, agent_c, summarizer


# Round-specific suffixes (after system_prefix; same role as AGENT_A_FIRST_TURN in debate.py)
# =========================
# Shared constraint
# =========================
CRS_PRIOR_BLOCK = """
[CRS recommender prior — MUST follow]
In [Context], "Observation 0 (CRS candidates, ...)" is the output of the sequential recommender (trained CRS; same family as SASRec-style full-sort scores). Each line is: item id, title, confidence score; list order is descending relevance (higher score = stronger collaborative signal for this user).
- Treat CRS order and scores as the PRIMARY evidence alongside user history. Your MY_RANKING should stay broadly aligned with CRS: high-score / early-list items should usually appear in your upper ranks unless user history or (for B/C) clear semantic or persona reasons justify moving them down.
- Reordering is allowed, but do NOT push many top-CRS items to the bottom without explicit justification tied to history or profile.
- When choosing between two pool items, prefer the one with higher CRS score / earlier in Observation 0 unless contradicting evidence is strong.
- Fair-evidence balance: CRS/history-frequency is not automatically stronger than a peer's **anchored** semantic or persona link (specific history/profile ↔ pool ID with a checkable reason in [Context]). Weigh both; dismiss neither by label alone.
"""

POOL_TAIL = """
[Hard constraint]
The candidate pool contains ONLY the IDs listed below (and titles in [Context]).
- Do NOT invent or hallucinate any movie IDs.
- All rankings, comparisons, and examples MUST use pool IDs only.
""" + CRS_PRIOR_BLOCK + """
Output format requirement:
End your reply with:

MY_RANKING:
1. <movie_id> — short rationale
...
10. <movie_id> — short rationale
(Exactly 10 lines, no extra text after it)
"""

# =========================
# Round 1
# =========================

HORIZONTAL_ROUND1_A = """You are Agent A (Behavioral).
You are the FIRST speaker in Round 1 and have not seen any other agents.

Task:
- Build an initial ranking based on behavioral signals from user history:
  (genre frequency, repeated patterns, viewing stability)
- Prioritize consistency and preference reinforcement.

Use history and CRS as main drivers; avoid long speculative plots. If several pool items tie on behavior, ties may be broken by thematic consistency **when visible across multiple high-rated history titles** (still pool-IDs only).

Style:
Conversational, stable, history-driven.

""" + POOL_TAIL


HORIZONTAL_ROUND1_B = """You are Agent B (Semantic).
You are the SECOND speaker in Round 1 and have seen Agent A.

Task:
- Critique frequency-only reasoning.
- Use specific items from user history (titles/IDs) to show:
  user also values narrative depth, director style, themes, or structure.
- Compare concrete pool candidates when possible.
- Argue for semantically stronger alternatives even if less frequent.

You may explicitly disagree with A, but must ground arguments in history + pool items.

Do NOT lean on user demographics or "persona" arguments — Agent C owns profile/context fit.

""" + POOL_TAIL


HORIZONTAL_ROUND1_C = """You are Agent C (Persona).
You are the THIRD speaker in Round 1 and have seen A and B.

You must add a DISTINCT lens from Agent B:
- B reasons about themes, narrative, director style, and film semantics.
- YOU reason about real-world plausibility from the user profile line only:
  age, gender (if given), occupation, zip/lifestyle cues — e.g. mainstream vs niche tolerance,
  likely viewing context (comfort/attention), familiarity vs experimental picks.

Hard anti-mimicry:
- Do NOT copy Agent B's MY_RANKING order verbatim.
- If your first pass matches B's ID order position-for-position, change at least TWO ranks
  using persona tie-breaks (mainstream accessibility, avoid gratuitously grim/niche picks for
  the stated profile unless history strongly supports it, etc.).
- In your critique, cite the profile text at least once; do NOT re-use B's thematic wording
  sentence-for-sentence.

You may side with A or B on individual titles, but every rank must be justified by persona + history, not by repeating B's film-critic rationale.

Do NOT introduce new candidates unless necessary for clarification.

""" + POOL_TAIL


# =========================
# Round 2
# =========================

HORIZONTAL_ROUND2_A = """You are Agent A (Behavioral).
You speak FIRST in Round 2 and have seen Round 1 + B/C feedback.

Task:
- Update ranking by integrating semantic + persona signals as weak constraints.
- Maintain stability: keep top historical-frequency items if still valid.
- Only adjust when B/C evidence clearly contradicts behavioral signals.

Output:
REVISED_RANKING:
1. <movie_id> — reason
...
10. <movie_id> — reason

Optional:
CONFORM_NOTE: brief sentence explaining what changed from Round 1.
""" + POOL_TAIL


HORIZONTAL_ROUND2_B = """You are Agent B (Semantic).
You are SECOND in Round 2 and have seen A's revision.

Task:
- Evaluate whether semantic corrections were properly integrated.
- If A ignored strong narrative/style evidence, correct it with history-backed examples.
- If A integrated well, explicitly acknowledge convergence.
- You may propose one “dark horse” ONLY if strongly supported by history similarity.

Output:
MY_RANKING:
(10 lines, pool IDs only)
""" + POOL_TAIL


HORIZONTAL_ROUND2_C = """You are Agent C (Persona).
You are THIRD in Round 2.

Task:
- Check persona fit of the *process*: did A incorporate B without ignoring obvious profile cues?
- Even when A and B largely agree on semantics, you STILL apply a persona pass:
  tweak 1–3 ranks if needed for mainstream/niche balance, tone heaviness, or life-stage plausibility
  (use the user profile line + history, not film-studies language).

Anti-mimicry:
- Do NOT paste Agent B's Round 2 MY_RANKING as your own.
- Your MY_RANKING must differ from B's Round 2 order in at least TWO positions whenever B listed 10 pool IDs,
  unless you briefly justify in one sentence why persona forces identity (rare).

Output:
MY_RANKING:
(10 lines, pool IDs only)
""" + POOL_TAIL


# =========================
# Summarizer
# =========================

HORIZONTAL_ROUND3_SUMMARIZER_BODY = (
    """You are the Summarizer.

You are NOT a fourth peer. Do NOT invent new movie-level arguments.
You MUST fuse A (behavioral), B (semantic), and C (persona) into one order.
"""
    + CRS_PRIOR_BLOCK
    + """
Anti-copy rule (critical):
- Do NOT adopt any single agent's MY_RANKING / REVISED_RANKING verbatim.
- Especially do NOT default to Round 2 Agent C's list just because it appears last in the transcript (recency bias).
- Persona (C) is ONE signal with the same standing as A and B. It is NOT a "final filter" or veto that overrides A+B when they agree.

Before the list, write an explicit synthesis (do not skip it):

Thought:
- 3–8 short bullets covering: (1) where A/B/C agree or disagree, (2) which IDs are consensus vs exploratory,
  (3) how you merged (behavioral vs semantic vs persona), (4) any deliberate departures from a single agent's order.
- Reference Parsed ID hints when useful; stay grounded in the transcript.

Then output the machine-readable block:

FINAL_TOP10:
1. <movie_id>
...
10. <movie_id>

Rules for FINAL_TOP10 lines:
- Each line must start with the numeric pool movie_id (optional short title after em dash for logs).
- No extra blank lines inside the 10 rows.

CRS fusion:
- In Thought, briefly note how the final order respects Observation 0 tiers versus peer edits.
- Do not produce a FINAL_TOP10 that ignores CRS top-scoring items unless A/B/C jointly justify it in the transcript.
"""
)


def _ctx_block(base_context: str, pool_line: str) -> str:
    return (
        "\n[Context]\n"
        + base_context
        + "\n[Candidate IDs]\n"
        + pool_line
        + "\n"
    )


class HorizontalDebateSimulation:
    """
    对标 DebateSimulation：持有多个 Agent + llm_chat 调度，跑水平顺序辩论。
    """

    def __init__(
        self,
        *,
        agent_a: BehavioralPeerAgent,
        agent_b: SemanticPeerAgent,
        agent_c: PersonaPeerAgent,
        summarizer: HorizontalSummarizerAgent,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.summarizer = summarizer

    def run(
        self,
        *,
        base_context: str,
        pool_ids: List[str],
        id_to_title: Dict[str, str],
        debate_sleep: float,
    ) -> Tuple[str, str, int]:
        """
        返回 (完整日志块, finish_inner, LLM 调用次数)。
        """
        pool_set = set(pool_ids)
        pool_line = ", ".join(pool_ids)
        ctx = _ctx_block(base_context, pool_line)
        n_llm = 0
        log_parts: List[str] = [
            "\n\n===== Horizontal Debate · Round 1 · Sequential (A → B → C) =====\n"
        ]

        msg_a1 = self.agent_a.system_prefix() + HORIZONTAL_ROUND1_A + ctx
        out_a1 = llm_chat(User_message=msg_a1)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent A · Round 1]\n" + out_a1.strip() + "\n\n")

        msg_b1 = self.agent_b.system_prefix() + HORIZONTAL_ROUND1_B + ctx + "\n[Agent A · Round 1]\n" + out_a1.strip() + "\n"
        out_b1 = llm_chat(User_message=msg_b1)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent B · Round 1]\n" + out_b1.strip() + "\n\n")

        msg_c1 = (
            self.agent_c.system_prefix()
            + HORIZONTAL_ROUND1_C
            + ctx
            + "\n[Agent A · Round 1]\n"
            + out_a1.strip()
            + "\n[Agent B · Round 1]\n"
            + out_b1.strip()
            + "\n"
        )
        out_c1 = llm_chat(User_message=msg_c1)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent C · Round 1]\n" + out_c1.strip() + "\n\n")

        r_a1 = pad_ranking(extract_section_ranking(out_a1, pool_set, 10), pool_ids, 10)
        r_b1 = pad_ranking(extract_section_ranking(out_b1, pool_set, 10), pool_ids, 10)
        r_c1 = pad_ranking(extract_section_ranking(out_c1, pool_set, 10), pool_ids, 10)

        log_parts.append(
            "===== Round 2 · Cross-review & compromise (A → B → C) =====\n"
        )

        msg_a2 = (
            self.agent_a.system_prefix()
            + HORIZONTAL_ROUND2_A
            + ctx
            + "\n[Round 1 · Agent B]\n"
            + out_b1.strip()
            + "\n[Round 1 · Agent C]\n"
            + out_c1.strip()
            + "\n[Your Round 1]\n"
            + out_a1.strip()
            + "\n"
        )
        out_a2 = llm_chat(User_message=msg_a2)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent A · Round 2 · Conformity]\n" + out_a2.strip() + "\n\n")

        msg_b2 = (
            self.agent_b.system_prefix()
            + HORIZONTAL_ROUND2_B
            + ctx
            + "\n[Round 1 · A]\n"
            + out_a1.strip()
            + "\n[Round 1 · B]\n"
            + out_b1.strip()
            + "\n[Round 1 · C]\n"
            + out_c1.strip()
            + "\n[Round 2 · A]\n"
            + out_a2.strip()
            + "\n"
        )
        out_b2 = llm_chat(User_message=msg_b2)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent B · Round 2]\n" + out_b2.strip() + "\n\n")

        msg_c2 = (
            self.agent_c.system_prefix()
            + HORIZONTAL_ROUND2_C
            + ctx
            + "\n[Round 1]\n"
            + out_a1.strip()
            + "\n---\n"
            + out_b1.strip()
            + "\n---\n"
            + out_c1.strip()
            + "\n[Round 2 · A]\n"
            + out_a2.strip()
            + "\n[Round 2 · B]\n"
            + out_b2.strip()
            + "\n"
        )
        out_c2 = llm_chat(User_message=msg_c2)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append("[Agent C · Round 2]\n" + out_c2.strip() + "\n\n")

        r_a2 = pad_ranking(extract_section_ranking(out_a2, pool_set, 10), pool_ids, 10)
        r_b2 = pad_ranking(extract_section_ranking(out_b2, pool_set, 10), pool_ids, 10)
        r_c2 = pad_ranking(extract_section_ranking(out_c2, pool_set, 10), pool_ids, 10)

        log_parts.append("===== Round 3 · Summarizer → Finish[Final_List] =====\n")
        msg_syn = (
            self.summarizer.system_prefix()
            + HORIZONTAL_ROUND3_SUMMARIZER_BODY
            + ctx
            + "\n[Round 1 · A]\n"
            + out_a1.strip()
            + "\n[Round 1 · B]\n"
            + out_b1.strip()
            + "\n[Round 1 · C]\n"
            + out_c1.strip()
            + "\n[Round 2 · A]\n"
            + out_a2.strip()
            + "\n[Round 2 · B]\n"
            + out_b2.strip()
            + "\n[Round 2 · C]\n"
            + out_c2.strip()
            + "\n[Parsed ID hints — compare orders; do not copy one row verbatim]\n"
            + "A_R1: "
            + ", ".join(r_a1)
            + "\nB_R1: "
            + ", ".join(r_b1)
            + "\nC_R1: "
            + ", ".join(r_c1)
            + "\nA_R2: "
            + ", ".join(r_a2)
            + "\nB_R2: "
            + ", ".join(r_b2)
            + "\nC_R2: "
            + ", ".join(r_c2)
            + "\n"
        )
        syn = llm_chat(User_message=msg_syn)
        n_llm += 1
        time.sleep(debate_sleep)
        log_parts.append(
            "[Summarizer · FINAL_TOP10]\n" + syn.strip() + "\n\n===== End horizontal debate =====\n"
        )

        final_ids = extract_section_ranking(syn, pool_set, 10)
        final_ids = pad_ranking(final_ids, pool_ids, 10)
        finish_inner = build_finish_inner(final_ids, id_to_title)

        full_log = "".join(log_parts)
        return full_log, finish_inner, n_llm


def run_horizontal_debate(
    base_context: str,
    *,
    pool_ids: List[str],
    id_to_title: Dict[str, str],
    debate_sleep: float,
    simulation: Optional[HorizontalDebateSimulation] = None,
) -> Tuple[str, str, int]:
    """兼容入口：内部使用 HorizontalDebateSimulation。"""
    if simulation is None:
        a, b, c, s = load_horizontal_agents_from_pattern()
        simulation = HorizontalDebateSimulation(
            agent_a=a,
            agent_b=b,
            agent_c=c,
            summarizer=s,
        )
    return simulation.run(
        base_context=base_context,
        pool_ids=pool_ids,
        id_to_title=id_to_title,
        debate_sleep=debate_sleep,
    )


def task_customization_horizontal(
    userID,
    sys_role=instruction,
    prompt=task_prompt,
    to_print=True,
    *,
    debate_sleep: float,
    prefetch_topk: int = 30,
    prefetch_condition: str = "None",
    simulation: Optional[HorizontalDebateSimulation] = None,
):
    question = env.reset(userID=userID)
    if to_print:
        print(userID, question)

    question = question + question_tail
    if prefetch_topk <= 0:
        raise ValueError("horizontal debate 需要 prefetch_topk>0 以共享候选池")

    cond = prefetch_condition.strip()
    pre_action = f"retrieve[{cond}, {prefetch_topk}]"
    pre_ret = step_env(env, pre_action)
    if pre_ret is None:
        obs = "[Prefetch retrieve failed after retries.]\n"
        id_to_title, pool_ids = {}, []
    else:
        obs, _r, _d, _info = pre_ret
        obs = (obs or "").replace("\\n", "")
        id_to_title, pool_ids = parse_retrieval_pool(obs)

    if not pool_ids:
        if to_print:
            print("警告: 候选池为空，使用 finish[] 走环境 conclude。")
        fin_ret = step_env(env, "finish[]")
        if fin_ret is None:
            raise RuntimeError("finish step failed")
        _obs, r, _done, info = fin_ret
        info_extra = {
            "n_calls": 0,
            "n_badcalls": 0,
            "traj": "",
            "use_horizontal_debate": True,
            "n_horizontal_llm_calls": 0,
            "horizontal_prefetch_topk": prefetch_topk,
            "horizontal_prefetch_condition": prefetch_condition.strip(),
            "horizontal_empty_pool": True,
        }
        info.update(info_extra)
        return r, info

    bootstrap = f"Observation 0 (CRS candidates, Top-{prefetch_topk}): {obs}\n"
    question += bootstrap

    pool_line = ", ".join(pool_ids)
    base_context = question + (
        f"\n[Candidate pool size] {len(pool_ids)}. IDs: {pool_line}\n"
    )

    if simulation is None:
        a, b, c, s = load_horizontal_agents_from_pattern()
        simulation = HorizontalDebateSimulation(
            agent_a=a,
            agent_b=b,
            agent_c=c,
            summarizer=s,
        )

    debate_block, finish_inner, n_h = simulation.run(
        base_context=base_context,
        pool_ids=pool_ids,
        id_to_title=id_to_title,
        debate_sleep=debate_sleep,
    )
    question += debate_block
    if to_print:
        print(debate_block)

    fin_ret = step_env(env, f"finish[{finish_inner}]")
    if fin_ret is None:
        raise RuntimeError("finish step failed after retries")
    _obs, r, _done, info = fin_ret

    info_extra = {
        "n_calls": 0,
        "n_badcalls": 0,
        "traj": "",
        "use_horizontal_debate": True,
        "n_horizontal_llm_calls": n_h,
        "horizontal_prefetch_topk": prefetch_topk,
        "horizontal_prefetch_condition": prefetch_condition.strip(),
        "horizontal_sequential": True,
    }
    info.update(info_extra)
    return r, info


logging.basicConfig(filename="./trajs/828_horizontal_debate.log", level=logging.DEBUG)

failed_times = 0
u_num = 0

infos = []
for uid in uid_iid.keys():
    if str(uid) not in hit_topk_uids:
        continue

    u_num += 1

    if u_num < args.start:
        continue
    if u_num > args.start + args.step_num:
        break

    try:
        r, info = task_customization_horizontal(
            uid,
            to_print=True,
            debate_sleep=args.horizontal_sleep,
            prefetch_topk=args.prefetch_topk,
            prefetch_condition=args.prefetch_condition,
        )
        infos.append(info)
        print("steps, \t recsys_steps, \t llm_steps, \t answer")
        logging.info(
            "steps {step}, \t recsys_steps {recsys_steps}, \t llm_steps {llm_steps}, \n answer {answer} \n trajectory {traj}".format(
                step=info["steps"],
                recsys_steps=info["recsys_steps"],
                llm_steps=info["llm_steps"],
                answer=info["answer"],
                traj=info["rec_traj"],
            )
        )
        print("-----------")
    except Exception as e:
        failed_times += 1
        time.sleep(20)
        print("An error occurred:", str(e))
        print(
            "OHHHHHHHH... User {user} Failed. Failed_times is {fail}".format(
                user=uid, fail=failed_times
            )
        )

with open(
    "chat_his/{dataset}/start_{st}_horizontal.json".format(
        dataset=dataset_name, st=args.start
    ),
    "w",
) as f:
    json.dump(infos, f)
    print("====  Info Dump Ends (horizontal) ====")
