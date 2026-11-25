"""Two-agent debate, evidence score calibration, and weighted reranking."""

from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Callable

from .prompts import INTERACTION, JUDGE, SEMANTIC

Agent = Callable[[str, dict], dict]
Verifier = Callable[[list[tuple[str, str]]], list[float]]


def item_id(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("item_id must be a nonempty string or integer")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("item_id must be a nonempty string or integer")
    return normalized


def number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError("expected a finite JSON number")
    return float(value)


def probability(value) -> float:
    result = number(value)
    if not 0 <= result <= 1:
        raise ValueError("expected a number in [0, 1]")
    return result


def string(value) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a JSON string")
    return value.strip()


@dataclass(frozen=True)
class Evidence:
    raw_text: str
    semantic: str = ""
    functional: str = ""
    contextual: str = ""


def select_candidates(rows: list[dict], seen: set[str], limit: int) -> list[dict]:
    candidates = []
    ids = set()
    for row in rows:
        key = item_id(row["item_id"])
        if key in ids:
            raise ValueError(f"duplicate candidate {key!r}")
        ids.add(key)
        candidate = {"item_id": key, "title": string(row.get("title", ""))}
        if "retrieval_score" in row:
            candidate["retrieval_score"] = number(row["retrieval_score"])
        if key not in seen:
            candidates.append(candidate)
    if any("retrieval_score" in row for row in candidates):
        if not all("retrieval_score" in row for row in candidates):
            raise ValueError("supply retrieval_score for every eligible item or omit it")
        candidates.sort(key=lambda row: row["retrieval_score"], reverse=True)
    return candidates[:limit]


def parse_scores(response: dict, pool: list[str], *, semantic: bool) -> dict:
    scores = {}
    for row in response["items"]:
        key = item_id(row["item_id"])
        if key not in pool or key in scores:
            raise ValueError(f"unexpected or duplicate item_id {key!r}")
        record = {
            "item_id": key,
            "preference": probability(row["preference"]),
            "critique": string(row.get("critique", "")),
        }
        if semantic:
            record.update(
                claim=string(row["claim"]),
                quote=string(row["quote"]),
                confidence=probability(row["confidence"]),
            )
        else:
            record["rationale"] = string(row["rationale"])
        scores[key] = record
    missing = set(pool).difference(scores)
    if missing:
        raise ValueError(f"missing scores for {', '.join(sorted(missing))}")
    return {"summary": string(response["summary"]), "items": scores}


def calibrate_semantic(items: dict, evidence: dict[str, Evidence], verifier: Verifier) -> None:
    keys = []
    pairs = []
    for key, row in items.items():
        source = evidence.get(key)
        row["support"] = 0.0
        if source is None or not source.raw_text.strip():
            row["evidence_status"] = "missing_source"
        elif not row["claim"] or not row["quote"]:
            row["evidence_status"] = "missing_claim_or_quote"
        elif row["quote"] not in source.raw_text:
            row["evidence_status"] = "quote_not_in_source"
        else:
            row["evidence_status"] = "verified"
            keys.append(key)
            pairs.append((row["quote"], row["claim"]))

    if pairs:
        support = verifier(pairs)
        if len(support) != len(pairs):
            raise ValueError("ECV returned the wrong number of support probabilities")
        for key, value in zip(keys, support):
            items[key]["support"] = probability(value)

    for row in items.values():
        row["score"] = row["preference"] * row["confidence"] * row["support"]


def rank(pool: list[str], records: dict, field: str, k: int) -> list[dict]:
    ordered = sorted(pool, key=lambda key: records[key][field], reverse=True)
    return [records[key] for key in ordered[:k]]


def branch_feedback(branch: dict) -> dict:
    return {**branch, "top_k": [row["item_id"] for row in branch["top_k"]]}


def parse_judge(response: dict) -> dict:
    interaction = probability(response["interaction_weight"])
    semantic = probability(response["semantic_weight"])
    total = interaction + semantic
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=0.001):
        raise ValueError("Judge weights must sum to 1")
    return {
        "summary": string(response["summary"]),
        "interaction_weight": interaction / total,
        "semantic_weight": semantic / total,
    }


class MADRec:
    def __init__(
        self,
        agent: Agent,
        verifier: Verifier,
        *,
        rounds: int = 2,
        candidate_count: int = 30,
        top_k: int = 10,
    ):
        if rounds < 1 or not 1 <= top_k <= candidate_count:
            raise ValueError("require rounds >= 1 and 1 <= top_k <= candidate_count")
        self.agent = agent
        self.verifier = verifier
        self.rounds = rounds
        self.candidate_count = candidate_count
        self.top_k = top_k

    def rerank(self, user: dict, evidence: dict[str, Evidence]) -> dict:
        seen = {item_id(row["item_id"]) for row in user["history"]}
        candidates = select_candidates(user["candidates"], seen, self.candidate_count)
        pool = [row["item_id"] for row in candidates]
        result = {
            "user_id": user["user_id"],
            "candidate_ids": pool,
            "rounds": [],
            "judge": None,
            "ranking": [],
            "llm_calls": 0,
        }
        if not pool:
            return result

        context = {
            "history": [
                {key: row[key] for key in ("item_id", "title", "genres", "category", "rating", "timestamp") if key in row}
                for row in user["history"]
            ],
            "profile": string(user.get("profile", "")),
            "candidates": candidates,
        }
        candidate_evidence = {key: asdict(evidence[key]) for key in pool if key in evidence}
        debate = result["rounds"]
        feedback = []
        for turn in range(1, self.rounds + 1):
            common = {**context, "round": turn, "previous_rounds": list(feedback)}
            interaction = parse_scores(self.agent(INTERACTION, common), pool, semantic=False)
            interaction["top_k"] = rank(pool, interaction["items"], "preference", self.top_k)
            semantic_context = {
                **common,
                "interaction_this_round": branch_feedback(interaction),
                "evidence": candidate_evidence,
            }
            semantic = parse_scores(self.agent(SEMANTIC, semantic_context), pool, semantic=True)
            calibrate_semantic(semantic["items"], evidence, self.verifier)
            semantic["top_k"] = rank(pool, semantic["items"], "score", self.top_k)
            debate.append({"round": turn, "interaction": interaction, "semantic": semantic})
            feedback.append({
                "round": turn,
                "interaction": branch_feedback(interaction),
                "semantic": branch_feedback(semantic),
            })

        final = debate[-1]
        eligible = {row["item_id"] for branch in ("interaction", "semantic") for row in final[branch]["top_k"]}
        union = [key for key in pool if key in eligible]
        judge_context = {**context, "debate": feedback, "calibration_candidates": union}
        judge = parse_judge(self.agent(JUDGE, judge_context))
        fused = {}
        for key in union:
            interaction_score = final["interaction"]["items"][key]["preference"]
            semantic_score = final["semantic"]["items"][key]["score"]
            fused[key] = {
                "item_id": key,
                "interaction_score": interaction_score,
                "semantic_score": semantic_score,
                "score": judge["interaction_weight"] * interaction_score + judge["semantic_weight"] * semantic_score,
            }
        result["judge"] = {**judge, "candidate_ids": union, "scores": fused}
        result["ranking"] = rank(union, fused, "score", self.top_k)
        result["llm_calls"] = 2 * self.rounds + 1
        return result
