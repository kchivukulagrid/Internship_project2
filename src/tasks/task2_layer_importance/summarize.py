"""Summarize logit-lens outputs into per-layer emergence scores."""

from __future__ import annotations

# ---------------------------------------------------------------------
# Task 2 Logit Lens Summary
# ---------------------------------------------------------------------

import argparse
import csv
import json
import os


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "experiments/task2_layer_importance/logit_lens.jsonl"
OUTPUT_FILE = "experiments/task2_layer_importance/results.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize logit lens outputs.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    stats = {}
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels_present = row.get("labels_present", {})
            layer_probs = row.get("layer_probs", {})

            for layer, probs in layer_probs.items():
                for label, prob in probs.items():
                    key = (layer, label)
                    if key not in stats:
                        stats[key] = {"present": [], "absent": []}
                    bucket = "present" if labels_present.get(label) else "absent"
                    stats[key][bucket].append(prob)

    rows = []
    for (layer, label), buckets in stats.items():
        present = buckets["present"]
        absent = buckets["absent"]
        mean_present = sum(present) / len(present) if present else 0.0
        mean_absent = sum(absent) / len(absent) if absent else 0.0
        emergence = mean_present - mean_absent
        rows.append(("task2", label, layer, "mean_present", mean_present, "logit_lens"))
        rows.append(("task2", label, layer, "mean_absent", mean_absent, "logit_lens"))
        rows.append(("task2", label, layer, "emergence_score", emergence, "logit_lens"))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"Summary appended to {args.output_file}")


if __name__ == "__main__":
    main()
