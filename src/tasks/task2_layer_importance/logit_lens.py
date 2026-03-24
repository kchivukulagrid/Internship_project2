"""Logit-lens extraction for Task 2."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import get_device


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/task1_constrained/adapter"
INPUT_FILE = "data/processed/task1_val.jsonl"
OUTPUT_FILE = "experiments/task2_layer_importance/logit_lens.jsonl"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-layer logits for token classification.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--sample_count", type=int, default=200)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Tokenization Helpers
# ---------------------------------------------------------------------
def _label_token_ids(tokenizer: Any, label: str) -> list[int]:
    """Return candidate token IDs for a label, allowing multi-token encodings."""
    candidates: list[int] = []
    for variant in [label, f" {label}"]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        candidates.extend(ids)
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for tok_id in candidates:
        if tok_id in seen:
            continue
        seen.add(tok_id)
        unique.append(tok_id)
    return unique


def _label_presence_from_output(output_json: str) -> dict[str, bool]:
    """Return whether each entity type appears in the ground-truth output."""
    present = {"PER": False, "ORG": False, "LOC": False, "MISC": False}
    try:
        payload = json.loads(output_json)
        for ent in payload.get("entities", []):
            etype = ent.get("type")
            if isinstance(etype, str):
                etype = etype.strip().upper()
                if etype in present:
                    present[etype] = True
    except json.JSONDecodeError:
        pass
    return present


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = get_device()

    dataset = load_dataset("json", data_files={"validation": args.input_file})
    val_data = dataset["validation"].select(range(min(args.sample_count, len(dataset["validation"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(model, args.model_path)
    model.to(device)
    model.eval()

    label_ids = {lbl: _label_token_ids(tokenizer, lbl) for lbl in ["PER", "ORG", "LOC", "MISC"]}
    if any(len(v) == 0 for v in label_ids.values()):
        print("Warning: some labels did not map to tokens; probabilities may be zero.")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        for example in val_data:
            prompt = example["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            # Logit lens: apply LM head to each layer's last-token hidden state.
            row = {
                "prompt": prompt,
                "labels_present": _label_presence_from_output(example["output"]),
                "layer_probs": {},
            }
            hidden_states = outputs.hidden_states or []
            for idx, hidden in enumerate(hidden_states[1:], start=0):
                last_hidden = hidden[0, -1]
                logits = model.lm_head(last_hidden)
                probs = torch.softmax(logits, dim=-1)

                label_probs = {}
                for lbl, ids in label_ids.items():
                    label_probs[lbl] = float(probs[ids].max().item()) if ids else 0.0
                row["layer_probs"][f"layer_{idx}"] = label_probs
            f.write(json.dumps(row) + "\n")

    print(f"Saved logit lens outputs to {args.output_file}")


if __name__ == "__main__":
    main()
