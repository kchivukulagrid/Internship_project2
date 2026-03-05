"""Evaluate inference outputs and persist metrics JSON."""

from __future__ import annotations

import argparse
import json
import os

from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONL and write metrics.")
    parser.add_argument(
        "--input_file",
        default="data/processed/exports/qwen2_5_1_5B_masked_tuned_predictions.jsonl",
    )
    parser.add_argument(
        "--output_file",
        default="experiments/qwen2_5_1_5B_masked_tuned/metrics.json",
    )
    return parser.parse_args()


def _print_results(results: dict) -> None:
    """Render human-readable metric summary to stdout."""
    print("\n==============================")
    print("        EVALUATION RESULTS")
    print("==============================")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")
    print(f"Validity  : {results['validity']:.4f}")
    print(f"Validity% : {results['validity'] * 100:.2f}%")
    print(f"Valid JSON: {results['valid_json_count']}/{results['total_examples']}")
    print(f"Repaired  : {results['repaired_json_count']}")


def main() -> None:
    args = parse_args()
    results = compute_metrics(args.input_file)
    _print_results(results)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
