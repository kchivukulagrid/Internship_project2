"""Compare pre/post adversarial robustness results."""

from __future__ import annotations

import argparse
import csv
from typing import Any


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
BASELINE_FILE = "experiments/task4_adversarial/results_pre.csv"
NEW_FILE = "experiments/task4_adversarial/results_post.csv"
OUTPUT_FILE = "experiments/task4_adversarial/robustness_gains.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Task 4 results (pre vs post).")
    parser.add_argument("--baseline_file", default=BASELINE_FILE)
    parser.add_argument("--new_file", default=NEW_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_csv(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    out = {}
    for row in rows[1:]:
        rec = dict(zip(header, row))
        key = rec["category"]
        out[key] = rec
    return out


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    base = _read_csv(args.baseline_file)
    new = _read_csv(args.new_file)

    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "f1_gain", "boundary_f1_gain", "precision_gain", "recall_gain"]
        )
        for category, rec in base.items():
            if category not in new:
                continue
            def _f(x: str) -> float:
                try:
                    return float(x)
                except ValueError:
                    return 0.0
            f1_gain = _f(new[category]["f1"]) - _f(rec["f1"])
            b_gain = _f(new[category]["boundary_f1"]) - _f(rec["boundary_f1"])
            p_gain = _f(new[category]["precision"]) - _f(rec["precision"])
            r_gain = _f(new[category]["recall"]) - _f(rec["recall"])
            writer.writerow([category, f1_gain, b_gain, p_gain, r_gain])

    print(f"Saved gains to {args.output_file}")


if __name__ == "__main__":
    main()
