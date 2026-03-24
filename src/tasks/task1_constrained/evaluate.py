"""Evaluate Task 1 prediction JSONL outputs."""

from __future__ import annotations

import argparse
import json
import os

from src.core.metrics import compute_metrics

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 predictions.")
    parser.add_argument(
        "--input_file",
        default="data/processed/exports/task1_predictions.jsonl",
    )
    parser.add_argument(
        "--output_file",
        default="experiments/task1_constrained/metrics.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    # Compute and persist span-level metrics.
    results = compute_metrics(args.input_file)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(results)


if __name__ == "__main__":
    main()
