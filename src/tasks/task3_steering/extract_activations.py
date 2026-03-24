"""Extract mean activations for strict vs loose boundary sets."""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import get_device


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"
STRICT_FILE = "data/steering/strict.jsonl"
LOOSE_FILE = "data/steering/loose.jsonl"
OUTPUT_FILE = "experiments/task3_steering/mean_activations.json"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean activations for Task 3.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--strict_file", default=STRICT_FILE)
    parser.add_argument("--loose_file", default=LOOSE_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--layers", default="12,13,14,15,16")
    parser.add_argument("--use_output", default="yes", choices=["yes", "no"])
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _layer_indices(layer_str: str) -> list[int]:
    return [int(x.strip()) for x in layer_str.split(",") if x.strip()]

def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean_activations(model, tokenizer, dataset, layer_idxs: list[int], device: str, use_output: bool):
    sums = {idx: None for idx in layer_idxs}
    counts = 0
    for example in dataset:
        prompt = example["prompt"]
        if use_output:
            text = prompt + "\n\n" + example["output"]
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        for idx in layer_idxs:
            # hidden_states[0] is embeddings
            h = hidden_states[idx + 1][0]
            mean_vec = h.mean(dim=0)
            if sums[idx] is None:
                sums[idx] = mean_vec
            else:
                sums[idx] = sums[idx] + mean_vec
        counts += 1
    means = {idx: (sums[idx] / counts).cpu().tolist() for idx in layer_idxs}
    return means


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

    layer_idxs = _layer_indices(args.layers)

    strict_ds = _load_jsonl(args.strict_file)
    loose_ds = _load_jsonl(args.loose_file)

    use_output = args.use_output == "yes"
    strict_means = _mean_activations(model, tokenizer, strict_ds, layer_idxs, device, use_output)
    loose_means = _mean_activations(model, tokenizer, loose_ds, layer_idxs, device, use_output)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"strict": strict_means, "loose": loose_means}, f)

    print(f"Saved mean activations to {args.output_file}")


if __name__ == "__main__":
    main()
