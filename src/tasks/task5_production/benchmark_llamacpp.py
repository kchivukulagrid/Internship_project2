"""Benchmark llama.cpp quantized models and compute metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from typing import Any

from datasets import load_dataset

from src.core.metrics import compute_metrics
from src.core.parsing import extract_json
from src.core.schema import empty_output


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
LLAMA_BIN = "./llama.cpp/build/bin/llama-completion"
MODEL_PATH = "models/gguf/model.Q4_K_M.gguf"
INPUT_FILE = "data/processed/task1_test.jsonl"
OUTPUT_DIR = "experiments/task5_production"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp quantized model.")
    parser.add_argument("--llama_bin", default=LLAMA_BIN)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--sample_count", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="none", choices=["none", "metal"])
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--label", default="Q4_K_M")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_llama(
    llama_bin: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    threads: int,
    device: str,
    n_gpu_layers: int,
) -> str:
    cmd = [
        llama_bin,
        "-m",
        model_path,
        "-n",
        str(max_tokens),
        "-t",
        str(threads),
        "-ngl",
        str(n_gpu_layers),
        "--device",
        device,
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
    if device == "none":
        env["GGML_METAL"] = "0"
        env["LLAMA_METAL"] = "0"
        env["LLAMA_NO_METAL"] = "1"
    else:
        env.pop("GGML_METAL", None)
        env.pop("LLAMA_METAL", None)
        env.pop("LLAMA_NO_METAL", None)
    proc = subprocess.run(cmd, capture_output=True, check=False, env=env)
    stdout = proc.stdout.decode("utf-8", errors="ignore")
    return stdout


def _extract_completion(stdout: str) -> str:
    # llama.cpp prints the prompt + completion; take the tail after prompt marker if present.
    return stdout.strip()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"pred_{args.label}.jsonl")

    start = time.time()
    with open(pred_path, "w") as f:
        for row in dataset:
            prompt = row["prompt"]
            stdout = _run_llama(
                args.llama_bin,
                args.model_path,
                prompt,
                args.max_tokens,
                args.threads,
                args.device,
                args.n_gpu_layers,
            )
            completion = _extract_completion(stdout)
            parsed = extract_json(completion)
            if parsed is None:
                parsed = empty_output()
            prediction = json.dumps(parsed, ensure_ascii=False)
            f.write(
                json.dumps(
                    {"ground_truth": row["output"], "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )
    elapsed = time.time() - start

    metrics = compute_metrics(pred_path)

    bench_path = os.path.join(args.output_dir, "quant_benchmark.csv")
    write_header = not os.path.exists(bench_path)
    with open(bench_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "label",
                    "model_path",
                    "sample_count",
                    "elapsed_sec",
                    "per_sample_ms",
                    "precision",
                    "recall",
                    "f1",
                    "validity",
                ]
            )
        writer.writerow(
            [
                args.label,
                args.model_path,
                args.sample_count,
                elapsed,
                (elapsed / max(args.sample_count, 1)) * 1000,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["validity"],
            ]
        )

    print(f"Wrote predictions to {pred_path}")
    print(f"Appended benchmark to {bench_path}")


if __name__ == "__main__":
    main()
