"""Command-line entry points for MADRec."""

import argparse
import json
from math import log2
from pathlib import Path

from . import train_ecv
from .client import ChatClient
from .demo import demo_agent, demo_verifier, example
from .io import load_evidence, read_jsonl, write_jsonl
from .pipeline import MADRec, item_id, string
from .prompts import EVIDENCE


def build_evidence(args):
    chat = ChatClient()

    def records():
        seen = set()
        for row in read_jsonl(args.input):
            key = item_id(row["item_id"])
            if key in seen:
                raise ValueError(f"duplicate input item {key!r}")
            seen.add(key)
            raw_text = string(row["raw_text"])
            extracted = chat(EVIDENCE, {"raw_text": raw_text}) if raw_text else {}
            yield {
                "item_id": key,
                "raw_text": raw_text,
                **{field: string(extracted.get(field, "")) for field in ("semantic", "functional", "contextual")},
            }

    write_jsonl(args.output, records())


def run_ranking(args):
    if args.command == "demo":
        users, evidence = example()
        chat, verifier = demo_agent, demo_verifier
    else:
        from .ecv import ECV

        users = read_jsonl(args.input)
        evidence = load_evidence(args.evidence)
        chat, verifier = ChatClient(), ECV(args.ecv_checkpoint, args.device)

    ranker = MADRec(
        chat, verifier, rounds=args.rounds, candidate_count=args.candidate_count, top_k=args.top_k,
    )
    total = labeled = retrieval_hits = hits = 0
    ndcg = 0.0

    def results():
        nonlocal total, labeled, retrieval_hits, hits, ndcg
        for user in users:
            try:
                target = item_id(user["target_id"]) if user.get("target_id") is not None else None
                result = ranker.rerank(user, evidence)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"user {user.get('user_id', '<unknown>')!r}: {error}") from error
            total += 1
            if target is not None:
                labeled += 1
                retrieved = target in result["candidate_ids"]
                position = next(
                    (index for index, row in enumerate(result["ranking"], 1) if row["item_id"] == target),
                    None,
                )
                gain = 1 / log2(position + 1) if position else 0.0
                retrieval_hits += retrieved
                hits += position is not None
                ndcg += gain
                result["evaluation"] = {
                    "target_id": target, "retrieval_hit": retrieved, "rank": position,
                    "recall": float(position is not None), "ndcg": gain,
                }
            yield result
        if total == 0:
            raise ValueError("no user records in input")

    if args.output:
        write_jsonl(args.output, results())
    else:
        for result in results():
            print(json.dumps(result, ensure_ascii=False))

    metrics = {"users": total, "labeled_users": labeled}
    if labeled:
        metrics.update({
            f"RetrievalRecall@{args.candidate_count}": retrieval_hits / labeled,
            f"Recall@{args.top_k}": hits / labeled,
            f"NDCG@{args.top_k}": ndcg / labeled,
        })
    print(json.dumps(metrics, ensure_ascii=False))


def ranking_args(parser):
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, help="save per-user results to JSONL")


def main():
    parser = argparse.ArgumentParser(description="MADRec evidence-grounded sequential reranking")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run offline with fixed example model responses")
    ranking_args(demo)
    recommend = commands.add_parser("recommend", help="rerank exported candidate pools")
    ranking_args(recommend)
    recommend.add_argument("--input", type=Path, required=True)
    recommend.add_argument("--evidence", type=Path, required=True)
    recommend.add_argument("--ecv-checkpoint", required=True)
    recommend.add_argument("--device", help="ECV PyTorch device, e.g. cpu or cuda")
    evidence = commands.add_parser("build-evidence", help="extract structured item evidence")
    evidence.add_argument("--input", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train-ecv", help="fine-tune a binary evidence verifier")
    train_ecv.add_arguments(training)
    args = parser.parse_args()
    if args.command == "build-evidence":
        build_evidence(args)
    elif args.command == "train-ecv":
        train_ecv.train(args)
    else:
        run_ranking(args)
