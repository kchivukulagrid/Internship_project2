"""Benchmark constrained vs unconstrained decoding for Task 1."""

from __future__ import annotations

import argparse
import csv
import os
import time

from src.core.metrics import compute_metrics
from src.tasks.task1_constrained.inference import main as run_inference

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 benchmarks.")
    parser.add_argument("--input_file", default="data/processed/task1_test.jsonl")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model_path", default="experiments/task1_constrained/adapter")
    parser.add_argument("--sample_count", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/task1_constrained")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Benchmark Helpers
# ---------------------------------------------------------------------
def _run_variant(mode: str, args: argparse.Namespace) -> tuple[str, float, dict]:
    output_file = os.path.join(args.output_dir, f"task1_{mode}_predictions.jsonl")
    start = time.perf_counter()

    run_inference_args = [
        "--model_name",
        args.model_name,
        "--model_path",
        args.model_path,
        "--input_file",
        args.input_file,
        "--output_file",
        output_file,
        "--sample_count",
        str(args.sample_count),
        "--generation_mode",
        mode,
        "--json_validate",
        "yes",
        "--temperature",
        "0.0",
    ]

    import sys

    # Reuse the inference entrypoint with controlled args for benchmarking.
    sys.argv = ["task1_inference"] + run_inference_args
    run_inference()
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(output_file)
    return output_file, elapsed, metrics


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for mode in ["free", "constrained"]:
        output_file, elapsed, metrics = _run_variant(mode, args)
        results.append(
            {
                "mode": mode,
                "output_file": output_file,
                "elapsed_sec": round(elapsed, 3),
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1": round(metrics["f1"], 6),
                "validity": round(metrics["validity"], 6),
            }
        )

    csv_path = os.path.join(args.output_dir, "task1_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "output_file",
                "elapsed_sec",
                "precision",
                "recall",
                "f1",
                "validity",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark complete -> {csv_path}")


if __name__ == "__main__":
    main()
