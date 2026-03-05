"""JSONL IO helpers for predictions, corrections, and queues."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# ---------------------------
# Path Helper
# ---------------------------
def ensure_parent(path: str | Path) -> Path:
    """Ensure parent directory exists and return normalized path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------
# Write Helpers
# ---------------------------
def append_jsonl(path: str | Path, row: dict) -> None:
    """Append one JSON-serializable row to a JSONL file."""
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Overwrite a JSONL file with an iterable of rows."""
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------
# Read Helper
# ---------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL rows; skip empty and malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
