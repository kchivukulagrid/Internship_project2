"""Evaluate Task 4 adversarial robustness."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any

from src.core.parsing import extract_json


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "experiments/task4_adversarial/predictions_pre.jsonl"
RESULTS_FILE = "experiments/task4_adversarial/results_pre.csv"
SUMMARY_FILE = "experiments/task4_adversarial/metrics_pre.json"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 4 adversarial outputs.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--results_file", default=RESULTS_FILE)
    parser.add_argument("--summary_file", default=SUMMARY_FILE)
    parser.add_argument("--label", default="pre")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _entity_set(entities: list[dict[str, Any]]) -> set[tuple[str, str, int, int]]:
    out = set()
    for e in entities:
        if (
            isinstance(e, dict)
            and isinstance(e.get("type"), str)
            and isinstance(e.get("value"), str)
            and isinstance(e.get("start"), int)
            and isinstance(e.get("end"), int)
        ):
            out.add((e["type"], e["value"], e["start"], e["end"]))
    return out


def _span_set(entities: list[dict[str, Any]]) -> set[tuple[int, int]]:
    out = set()
    for e in entities:
        if isinstance(e, dict) and isinstance(e.get("start"), int) and isinstance(e.get("end"), int):
            out.add((e["start"], e["end"]))
    return out


def _accumulate(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    span_tp = span_fp = span_fn = 0
    valid = 0
    total = 0
    for item in items:
        total += 1
        gt = extract_json(item.get("ground_truth", "")) or {"entities": []}
        pred_raw = item.get("prediction", "")
        pred = extract_json(pred_raw)
        if pred is not None:
            valid += 1
        pred = pred or {"entities": []}

        gt_set = _entity_set(gt.get("entities", []))
        pred_set = _entity_set(pred.get("entities", []))
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

        gt_spans = _span_set(gt.get("entities", []))
        pred_spans = _span_set(pred.get("entities", []))
        span_tp += len(gt_spans & pred_spans)
        span_fp += len(pred_spans - gt_spans)
        span_fn += len(gt_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    span_precision = span_tp / (span_tp + span_fp) if (span_tp + span_fp) else 0.0
    span_recall = span_tp / (span_tp + span_fn) if (span_tp + span_fn) else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall)
        else 0.0
    )

    validity = valid / total if total else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_f1": span_f1,
        "validity": validity,
        "total": total,
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    with open(args.input_file, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "unknown")].append(row)

    summary = {}
    results_rows = []

    for category, items in sorted(by_category.items()):
        metrics = _accumulate(items)
        summary[category] = metrics
        results_rows.append(
            [
                "task4",
                args.label,
                category,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["boundary_f1"],
                metrics["validity"],
            ]
        )

    # Aggregate overall and adversarial-only.
    metrics_all = _accumulate(rows)
    summary["all"] = metrics_all
    results_rows.append(
        [
            "task4",
            args.label,
            "all",
            metrics_all["precision"],
            metrics_all["recall"],
            metrics_all["f1"],
            metrics_all["boundary_f1"],
            metrics_all["validity"],
        ]
    )
    adv_rows = [r for r in rows if r.get("category") != "original"]
    metrics_adv = _accumulate(adv_rows)
    summary["adversarial_all"] = metrics_adv
    results_rows.append(
        [
            "task4",
            args.label,
            "adversarial_all",
            metrics_adv["precision"],
            metrics_adv["recall"],
            metrics_adv["f1"],
            metrics_adv["boundary_f1"],
            metrics_adv["validity"],
        ]
    )

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "phase", "category", "precision", "recall", "f1", "boundary_f1", "validity"]
        )
        writer.writerows(results_rows)

    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Failure modes.
    worst = None
    for category, metrics in summary.items():
        if category in ("all", "adversarial_all", "original"):
            continue
        score = metrics["f1"]
        if worst is None or score < worst[1]:
            worst = (category, score)
    if worst:
        print(f"Worst category by F1: {worst[0]} ({worst[1]:.4f})")

    print(f"Saved results to {args.results_file}")
    print(f"Saved summary to {args.summary_file}")


if __name__ == "__main__":
    main()
