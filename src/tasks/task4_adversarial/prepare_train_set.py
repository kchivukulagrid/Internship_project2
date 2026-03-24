"""Create adversarial training mix for Task 4."""

from __future__ import annotations

import argparse
import json
import os
import random

from src.tasks.task4_adversarial.prepare_eval_set import _transform_row


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "data/processed/task1_train.jsonl"
OUTPUT_DIR = "data/processed/adversarial"
ADV_COUNT = 1000


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare adversarial training mix.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--adv_count", type=int, default=ADV_COUNT)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.input_file, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.adv_count > len(rows):
        args.adv_count = len(rows)

    # Deterministic sample.
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = [rows[i] for i in indices[: args.adv_count]]

    categories = ["nested", "abbrev", "misspell", "ambiguous", "multilingual"]
    adversarial_rows = []
    for i, row in enumerate(selected):
        category = categories[i % len(categories)]
        adversarial_rows.append(_transform_row(row, category, rng))

    original_rows = [
        {"prompt": r["prompt"], "output": r["output"], "category": "original"}
        for r in rows
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    adv_path = os.path.join(args.output_dir, "train_adversarial.jsonl")
    mixed_path = os.path.join(args.output_dir, "train_mixed.jsonl")

    with open(adv_path, "w") as f:
        for row in adversarial_rows:
            f.write(json.dumps(row) + "\n")

    with open(mixed_path, "w") as f:
        for row in original_rows + adversarial_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Adversarial train: {adv_path} ({len(adversarial_rows)})")
    print(f"Mixed train:       {mixed_path} ({len(original_rows) + len(adversarial_rows)})")


if __name__ == "__main__":
    main()
