"""Run inference with activation steering applied."""

from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.parsing import extract_json
from src.core.schema import empty_output
from src.model import get_device


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"
INPUT_FILE = "data/processed/task1_test.jsonl"
STEERING_FILE = "experiments/task3_steering/steering_vectors.json"
OUTPUT_DIR = "experiments/task3_steering"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 3 steering inference.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--steering_file", default=STEERING_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--layers", default="12,13,14,15,16")
    parser.add_argument("--scales", default="0.5,1.0,1.5")
    parser.add_argument("--sample_count", type=int, default=200)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _layer_indices(layer_str: str) -> list[int]:
    return [int(x.strip()) for x in layer_str.split(",") if x.strip()]


def _scales(scale_str: str) -> list[float]:
    return [float(x.strip()) for x in scale_str.split(",") if x.strip()]


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "model")
        and hasattr(model.base_model.model.model, "layers")
    ):
        return model.base_model.model.model.layers
    raise RuntimeError("Could not locate transformer layers.")


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

    layers = _get_layers(model)

    with open(args.steering_file, "r") as f:
        vectors = json.load(f)

    layer_idxs = _layer_indices(args.layers)
    scales = _scales(args.scales)

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)

    for layer_idx in layer_idxs:
        layer_key = f"{layer_idx}"
        if layer_key not in vectors:
            layer_key = f"layer_{layer_idx}"
        vec = torch.tensor(vectors[layer_key], device=device)

        for scale in scales:
            def hook_fn(_mod, _inp, out):
                if isinstance(out, tuple):
                    hidden = out[0] + (scale * vec)
                    return (hidden,) + out[1:]
                return out + (scale * vec)

            handle = layers[layer_idx].register_forward_hook(hook_fn)

            output_file = os.path.join(
                args.output_dir,
                f"pred_layer{layer_idx}_scale{scale}.jsonl",
            )
            with open(output_file, "w") as f:
                for example in dataset:
                    prompt = example["prompt"]
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            repetition_penalty=1.05,
                            num_beams=1,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen = outputs[0][inputs["input_ids"].shape[1] :]
                    decoded = tokenizer.decode(gen, skip_special_tokens=True)

                    parsed = extract_json(decoded) or empty_output()
                    prediction = json.dumps(parsed, ensure_ascii=False)
                    f.write(json.dumps({"ground_truth": example["output"], "prediction": prediction}) + "\n")

            handle.remove()
            print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
