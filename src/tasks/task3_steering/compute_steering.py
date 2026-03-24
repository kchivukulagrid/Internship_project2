"""Compute steering vectors from strict vs loose activations."""

from __future__ import annotations

import argparse
import json
import os


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "experiments/task3_steering/mean_activations.json"
OUTPUT_FILE = "experiments/task3_steering/steering_vectors.json"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute steering vectors for Task 3.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    with open(args.input_file, "r") as f:
        data = json.load(f)

    strict = data["strict"]
    loose = data["loose"]
    vectors = {}
    for layer, strict_vec in strict.items():
        loose_vec = loose[layer]
        vectors[layer] = [s - l for s, l in zip(strict_vec, loose_vec)]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(vectors, f)

    print(f"Saved steering vectors to {args.output_file}")


if __name__ == "__main__":
    main()

