"""Metrics for Task 1 schema with span-based evaluation."""

from __future__ import annotations

import json
from typing import Any

from src.core.parsing import extract_json

# ---------------------------------------------------------------------
# Set Helpers
# ---------------------------------------------------------------------


def _to_set(entities: list[dict[str, Any]]) -> set[tuple[str, str, int, int]]:
    # Span-level identity: type/value and exact character offsets.
    return set(
        (e["type"], e["value"], e["start"], e["end"])
        for e in entities
        if isinstance(e, dict)
        and isinstance(e.get("type"), str)
        and isinstance(e.get("value"), str)
        and isinstance(e.get("start"), int)
        and isinstance(e.get("end"), int)
    )


# ---------------------------------------------------------------------
# Public Metric API
# ---------------------------------------------------------------------
def compute_metrics(file_path: str) -> dict[str, Any]:
    """Compute precision/recall/F1 and JSON validity from prediction JSONL."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    valid_json = 0
    total = 0

    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            total += 1

            # Ground truth is already serialized as JSON in the dataset.
            gt = extract_json(item.get("ground_truth", "")) or {"entities": []}
            pred_raw = item.get("prediction", "")

            # Parsing doubles as JSON validity check against schema.
            pred_json = extract_json(pred_raw)
            if pred_json is not None:
                valid_json += 1

            pred_entities = (pred_json or {}).get("entities", [])
            gt_entities = (gt or {}).get("entities", [])

            gt_set = _to_set(gt_entities)
            pred_set = _to_set(pred_entities)

            total_tp += len(gt_set & pred_set)
            total_fp += len(pred_set - gt_set)
            total_fn += len(gt_set - pred_set)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    validity = valid_json / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "validity": validity,
        "total_examples": total,
        "valid_json_count": valid_json,
    }


# ---------------------------------------------------------------------
# Per-Type Metrics
# ---------------------------------------------------------------------
def _to_set_by_type(entities: list[dict[str, Any]]) -> dict[str, set[tuple[str, int, int]]]:
    by_type: dict[str, set[tuple[str, int, int]]] = {}
    for e in entities:
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        value = e.get("value")
        start = e.get("start")
        end = e.get("end")
        if not isinstance(etype, str) or not isinstance(value, str):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        key = (value, start, end)
        by_type.setdefault(etype, set()).add(key)
    return by_type


def compute_per_type_counts(
    gt_entities: list[dict[str, Any]],
    pred_entities: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return per-type TP/FP/FN counts."""
    gt_by_type = _to_set_by_type(gt_entities)
    pred_by_type = _to_set_by_type(pred_entities)
    all_types = set(gt_by_type) | set(pred_by_type)

    counts: dict[str, dict[str, int]] = {}
    for etype in all_types:
        gt_set = gt_by_type.get(etype, set())
        pred_set = pred_by_type.get(etype, set())
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        counts[etype] = {"tp": tp, "fp": fp, "fn": fn}
    return counts


def finalize_per_type_f1(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Convert per-type counts to precision/recall/F1."""
    metrics: dict[str, dict[str, float]] = {}
    for etype, c in counts.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[etype] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics
