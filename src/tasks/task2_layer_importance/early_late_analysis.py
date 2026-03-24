"""Early vs late layer emergence analysis for Task 2."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "experiments/task2_layer_importance/results_logit_lens.csv"
OUTPUT_FILE = "experiments/task2_layer_importance/results_early_late.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare early vs late layer emergence.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _layer_index(layer_name: str) -> int | None:
    if not layer_name.startswith("layer_"):
        return None
    try:
        return int(layer_name.split("_", 1)[1])
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    rows = []
    with open(args.input_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "task":
                continue
            if len(row) < 6:
                continue
            _, etype, layer, metric, value, notes = row
            if notes != "logit_lens" or metric != "emergence_score":
                continue
            idx = _layer_index(layer)
            if idx is None:
                continue
            try:
                score = float(value)
            except ValueError:
                continue
            rows.append((etype, idx, score))

    if not rows:
        print("No emergence_score rows found.")
        return

    max_idx = max(idx for _, idx, _ in rows)
    half = (max_idx + 1) // 2

    early_scores = defaultdict(list)
    late_scores = defaultdict(list)
    for etype, idx, score in rows:
        if idx < half:
            early_scores[etype].append(score)
        else:
            late_scores[etype].append(score)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "early_mean", "late_mean", "late_minus_early", "conclusion"])
        for etype in sorted(set([e for e, _, _ in rows])):
            early = early_scores.get(etype, [])
            late = late_scores.get(etype, [])
            early_mean = sum(early) / len(early) if early else 0.0
            late_mean = sum(late) / len(late) if late else 0.0
            diff = late_mean - early_mean
            if diff > 0:
                conclusion = "late>early"
            elif diff < 0:
                conclusion = "early>late"
            else:
                conclusion = "equal"
            writer.writerow([etype, early_mean, late_mean, diff, conclusion])

    print(f"Early/late analysis written to {args.output_file}")


if __name__ == "__main__":
    main()

