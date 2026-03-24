"""Evaluate boundary precision/recall for steering runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any

from src.core.parsing import extract_json


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_GLOB = "experiments/task3_steering/pred_layer*_scale*.jsonl"
RESULTS_FILE = "experiments/task3_steering/results.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 3 steering outputs.")
    parser.add_argument("--input_glob", default=INPUT_GLOB)
    parser.add_argument("--results_file", default=RESULTS_FILE)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _span_set(entities: list[dict[str, Any]]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        start = e.get("start")
        end = e.get("end")
        if isinstance(start, int) and isinstance(end, int):
            spans.add((start, end))
    return spans


def _boundary_metrics(path: str) -> tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            gt = extract_json(item.get("ground_truth", "")) or {"entities": []}
            pred = extract_json(item.get("prediction", "")) or {"entities": []}
            gt_spans = _span_set(gt.get("entities", []))
            pred_spans = _span_set(pred.get("entities", []))
            tp += len(gt_spans & pred_spans)
            fp += len(pred_spans - gt_spans)
            fn += len(gt_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("No prediction files found.")
        return

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "a", newline="") as f:
        writer = csv.writer(f)
        for path in files:
            precision, recall, f1 = _boundary_metrics(path)
            name = os.path.basename(path)
            # name like pred_layer12_scale0.5.jsonl
            parts = name.replace(".jsonl", "").split("_")
            layer = next((p.replace("layer", "") for p in parts if p.startswith("layer")), "na")
            scale = next((p.replace("scale", "") for p in parts if p.startswith("scale")), "na")
            writer.writerow(
                [
                    "task3",
                    layer,
                    scale,
                    precision,
                    recall,
                    f1,
                    f1,
                    "steering",
                ]
            )
    print(f"Appended results to {args.results_file}")


if __name__ == "__main__":
    main()
