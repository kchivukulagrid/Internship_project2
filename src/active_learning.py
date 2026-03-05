"""Active-learning scoring and cycle metadata helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.correction_io import append_jsonl, ensure_parent


# ---------------------------
# Scoring Helpers
# ---------------------------
def uncertainty_score(predicted: dict) -> float:
    """Simple heuristic: fewer entities => higher uncertainty."""
    entities = predicted.get("entities", []) if isinstance(predicted, dict) else []
    n = len(entities)
    if n == 0:
        return 1.0
    if n == 1:
        return 0.6
    if n == 2:
        return 0.35
    return 0.15


# ---------------------------
# Record Builders
# ---------------------------
def build_cycle_record(text: str, predicted: dict, corrected: dict) -> dict:
    """Create a cycle-level record for active learning."""
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "timestamp_utc": ts,
        "text": text,
        "predicted": predicted,
        "corrected": corrected,
        "accepted": predicted == corrected,
        "uncertainty": uncertainty_score(predicted),
    }


# ---------------------------
# Persistence Helpers
# ---------------------------
def append_cycle_record(path: str | Path, record: dict) -> None:
    append_jsonl(path, record)


def write_cycle_metadata(path: str | Path, data: dict) -> None:
    """Write cycle-level metadata as formatted JSON."""
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
