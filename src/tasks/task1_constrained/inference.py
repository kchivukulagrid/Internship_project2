"""Task 1 inference with optional constrained decoding."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.parsing import extract_json
from src.core.schema import empty_output
from src.model import get_device
from src.tasks.task1_constrained.decode import generate_constrained_json, generate_unconstrained

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/task1_constrained/adapter"
INPUT_FILE = "data/processed/task1_val.jsonl"
OUTPUT_FILE = "data/processed/exports/task1_predictions.jsonl"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 inference.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--sample_count", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.1, 0.2])
    parser.add_argument("--json_validate", default="yes", choices=["yes", "no"])
    parser.add_argument("--generation_mode", default="free", choices=["free", "constrained"])
    return parser.parse_args()


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def _load_model(model_name: str, model_path: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    dataset = load_dataset("json", data_files={"validation": args.input_file})
    val_data = dataset["validation"]
    if args.sample_count and args.sample_count > 0:
        val_data = val_data.select(range(min(args.sample_count, len(val_data))))

    model, tokenizer = _load_model(args.model_name, args.model_path, device)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        for i, example in enumerate(val_data):
            print(f"{i + 1}/{len(val_data)}")
            prompt = example["prompt"]

            if args.generation_mode == "constrained":
                # Schema-constrained JSON generation.
                decoded = generate_constrained_json(model, tokenizer, prompt, device=device)
            else:
                # Free-form decoding for baseline comparison.
                decoded = generate_unconstrained(
                    model,
                    tokenizer,
                    prompt,
                    temperature=args.temperature,
                )

            if args.json_validate == "yes":
                # Normalize to strict schema or fall back to empty output.
                parsed = extract_json(decoded)
                if parsed is None:
                    parsed = empty_output()
                prediction = json.dumps(parsed, ensure_ascii=False)
            else:
                prediction = decoded

            f.write(
                json.dumps(
                    {"ground_truth": example["output"], "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nTask 1 inference complete.")


if __name__ == "__main__":
    main()
