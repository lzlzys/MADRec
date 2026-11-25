"""JSONL files shared across evidence building, inference, and ECV training."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

from .pipeline import Evidence, item_id, string


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_evidence(path: Path) -> dict[str, Evidence]:
    entries = {}
    for row in read_jsonl(path):
        key = item_id(row["item_id"])
        if key in entries:
            raise ValueError(f"duplicate evidence for item {key!r}")
        entries[key] = Evidence(
            raw_text=string(row["raw_text"]),
            semantic=string(row.get("semantic", "")),
            functional=string(row.get("functional", "")),
            contextual=string(row.get("contextual", "")),
        )
    return entries
