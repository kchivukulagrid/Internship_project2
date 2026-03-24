"""Benchmark concurrent extraction latency with llama.cpp."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
LLAMA_BIN = "./llama.cpp/build/bin/llama-completion"
MODEL_PATH = "models/gguf/model.Q4_K_M.gguf"
INPUT_FILE = "data/processed/task1_test.jsonl"
OUTPUT_FILE = "experiments/task5_production/concurrency.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrency benchmark with llama.cpp.")
    parser.add_argument("--llama_bin", default=LLAMA_BIN)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--sample_count", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="none", choices=["none", "metal"])
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--label", default="Q4_K_M")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_one(args, prompt: str) -> float:
    cmd = [
        args.llama_bin,
        "-m",
        args.model_path,
        "-n",
        str(args.max_tokens),
        "-t",
        str(args.threads),
        "-ngl",
        str(args.n_gpu_layers),
        "--device",
        args.device,
        "--no-conversation",
        "--single-turn",
        "--no-display-prompt",
        "--simple-io",
        "--temp",
        "0",
        "--top-p",
        "1.0",
        "--seed",
        "42",
        "--prompt",
        prompt,
    ]
    env = os.environ.copy()
    if args.device == "none":
        env["GGML_METAL"] = "0"
        env["LLAMA_METAL"] = "0"
        env["LLAMA_NO_METAL"] = "1"
    else:
        env.pop("GGML_METAL", None)
        env.pop("LLAMA_METAL", None)
        env.pop("LLAMA_NO_METAL", None)
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, env=env)
    return time.time() - start


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[k]


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    conc_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))
    prompts = [row["prompt"] for row in dataset]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_header = not os.path.exists(args.output_file)

    with open(args.output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "label",
                    "concurrency",
                    "sample_count",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "throughput_rps",
                ]
            )

        for conc in conc_levels:
            start = time.time()
            latencies = []
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futures = [ex.submit(_run_one, args, p) for p in prompts]
                for fut in as_completed(futures):
                    latencies.append(fut.result())
            elapsed = time.time() - start
            p50 = _percentile(latencies, 50) * 1000
            p95 = _percentile(latencies, 95) * 1000
            p99 = _percentile(latencies, 99) * 1000
            throughput = len(latencies) / elapsed if elapsed > 0 else 0.0
            writer.writerow([args.label, conc, len(latencies), p50, p95, p99, throughput])

            print(
                json.dumps(
                    {
                        "concurrency": conc,
                        "p50_ms": p50,
                        "p95_ms": p95,
                        "p99_ms": p99,
                        "throughput_rps": throughput,
                    }
                )
            )

    print(f"Saved concurrency results to {args.output_file}")


if __name__ == "__main__":
    main()
