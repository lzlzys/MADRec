"""Deterministic interactions and model stubs for an offline smoke run."""

from .pipeline import Evidence
from .prompts import INTERACTION, JUDGE, SEMANTIC


def example():
    users = [
        {
            "user_id": "hit-example",
            "history": [{"item_id": "h1", "title": "Night Patrol", "rating": 5}],
            "profile": "Enjoys investigative mysteries.",
            "candidates": [{"item_id": key, "title": title} for key, title in [
                ("a", "Ninth Avenue"), ("b", "The Archivist"), ("c", "Coastal Inn"),
            ]],
            "target_id": "b",
        },
        {
            "user_id": "miss-example",
            "history": [],
            "candidates": [{"item_id": "c", "title": "Coastal Inn"}],
            "target_id": "outside-pool",
        },
    ]
    evidence = {
        "a": Evidence("A reporter follows a political campaign.", semantic="Politics"),
        "b": Evidence("An archivist investigates a decades-old disappearance.", semantic="Mystery"),
    }
    return users, evidence


def demo_agent(instruction, context):
    if instruction == JUDGE:
        return {"summary": "The mystery match is supported.", "interaction_weight": 0.4, "semantic_weight": 0.6}
    items = []
    for candidate in context["candidates"]:
        key = candidate["item_id"]
        if instruction == INTERACTION:
            preference = {"a": 0.9, "b": 0.65, "c": 0.3}[key]
            items.append({"item_id": key, "preference": preference, "rationale": "Example history signal."})
        elif instruction == SEMANTIC:
            quote = {"a": "A reporter follows a political campaign.",
                     "b": "An archivist investigates a decades-old disappearance.", "c": ""}[key]
            items.append({
                "item_id": key,
                "preference": {"a": 0.8, "b": 0.95, "c": 0.2}[key],
                "confidence": 0.9,
                "claim": quote,
                "quote": quote,
                "critique": "Example item-side judgment.",
            })
        else:
            raise ValueError("Unexpected demo prompt")
    return {"summary": "Example agent assessment.", "items": items}


def demo_verifier(pairs):
    return [0.4 if "reporter" in passage else 0.98 for passage, _ in pairs]
