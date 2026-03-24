"""Extract critical layers per entity type from ablation deltas."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "experiments/task2_layer_importance/results_ablation.csv"
OUTPUT_JSON = "experiments/task2_layer_importance/critical_layers.json"
OUTPUT_CSV = "experiments/task2_layer_importance/results_critical_layers.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract critical layers from ablation results.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    by_type: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with open(args.input_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "task":
                continue
            if len(row) < 6:
                continue
            _, etype, layer, metric, value, notes = row
            if notes != "ablation" or metric != "f1_delta":
                continue
            try:
                delta = float(value)
            except ValueError:
                continue
            by_type[etype].append((delta, layer))

    critical = {}
    for etype, items in by_type.items():
        # Most negative deltas indicate most important layers.
        ranked = sorted(items)[: args.top_k]
        critical[etype] = [layer for _delta, layer in ranked]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(critical, f, indent=2)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "critical_layers"])
        for etype, layers in critical.items():
            writer.writerow([etype, ";".join(layers)])

    print(f"Critical layers written to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()

