"""Profile activation and parameter memory for Task 5."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import get_device


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
INPUT_FILE = "data/processed/task1_test.jsonl"
OUTPUT_FILE = "experiments/task5_production/memory_profile.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile activation memory per layer.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=384)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _bytes_to_mb(num_bytes: float) -> float:
    return num_bytes / (1024 * 1024)


def _param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))

    layer_bytes: dict[int, int] = {}
    total_examples = 0

    with torch.no_grad():
        for row in dataset:
            prompt = row["prompt"]
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            for idx, h in enumerate(hidden_states[1:]):  # skip embeddings
                bytes_used = h.numel() * h.element_size()
                layer_bytes[idx] = layer_bytes.get(idx, 0) + bytes_used
            total_examples += 1

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "mean_activation_bytes", "mean_activation_mb", "param_bytes", "param_mb"]
        )
        param_bytes = _param_bytes(model)
        for layer, total_bytes in sorted(layer_bytes.items()):
            mean_bytes = total_bytes / max(total_examples, 1)
            writer.writerow(
                [
                    layer,
                    int(mean_bytes),
                    _bytes_to_mb(mean_bytes),
                    param_bytes,
                    _bytes_to_mb(param_bytes),
                ]
            )

    summary = {
        "model_name": args.model_name,
        "total_examples": total_examples,
        "param_bytes": _param_bytes(model),
    }
    with open(os.path.splitext(args.output_file)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved memory profile to {args.output_file}")


if __name__ == "__main__":
    main()
