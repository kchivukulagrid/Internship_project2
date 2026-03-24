"""Layer-wise LoRA ablation for Task 2."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.metrics import (
    compute_metrics,
    compute_per_type_counts,
    finalize_per_type_f1,
)
from src.core.parsing import extract_json
from src.model import get_device
from src.tasks.task1_constrained.inference import generate_unconstrained


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/task1_constrained/adapter"
INPUT_FILE = "data/processed/task1_val.jsonl"
OUTPUT_DIR = "experiments/task2_layer_importance/ablation"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate LoRA layers and measure F1 impact.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--sample_count", type=int, default=200)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_model(model_name: str, model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def _collect_transformer_layers(model) -> list[tuple[int, Any]]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(enumerate(model.model.layers))
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "model")
        and hasattr(model.base_model.model.model, "layers")
    ):
        return list(enumerate(model.base_model.model.model.layers))
    return []


def _set_layer_adapters(layer: Any, enabled: bool) -> None:
    modules = []
    if hasattr(layer, "self_attn"):
        if hasattr(layer.self_attn, "q_proj"):
            modules.append(layer.self_attn.q_proj)
        if hasattr(layer.self_attn, "v_proj"):
            modules.append(layer.self_attn.v_proj)
    for module in modules:
        if hasattr(module, "set_adapter"):
            module.set_adapter("default" if enabled else [])
        elif hasattr(module, "active_adapter"):
            module.active_adapter = module.active_adapter if enabled else []
        elif hasattr(module, "enable_adapters") and hasattr(module, "disable_adapters"):
            enable_fn = getattr(module, "enable_adapters")
            disable_fn = getattr(module, "disable_adapters")
            if callable(enable_fn) and callable(disable_fn):
                enable_fn() if enabled else disable_fn()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def _run_eval(
    model,
    tokenizer,
    val_data,
    tmp_path: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    total_counts: dict[str, dict[str, int]] = {}
    predictions = []
    for example in val_data:
        prompt = example["prompt"]
        decoded = generate_unconstrained(
            model,
            tokenizer,
            prompt,
            temperature=0.0,
        )
        predictions.append({"ground_truth": example["output"], "prediction": decoded})

        gt = extract_json(example["output"]) or {"entities": []}
        pred = extract_json(decoded) or {"entities": []}
        gt_entities = gt.get("entities", [])
        pred_entities = pred.get("entities", [])

        counts = compute_per_type_counts(gt_entities, pred_entities)
        for etype, c in counts.items():
            if etype not in total_counts:
                total_counts[etype] = {"tp": 0, "fp": 0, "fn": 0}
            total_counts[etype]["tp"] += c["tp"]
            total_counts[etype]["fp"] += c["fp"]
            total_counts[etype]["fn"] += c["fn"]

    with open(tmp_path, "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    overall = compute_metrics(tmp_path)
    per_type = finalize_per_type_f1(total_counts)
    return overall, per_type


def main() -> None:
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("json", data_files={"validation": args.input_file})
    val_data = dataset["validation"].select(range(min(args.sample_count, len(dataset["validation"]))))

    model, tokenizer = _load_model(args.model_name, args.model_path, device)
    layers = _collect_transformer_layers(model)
    if not layers:
        raise RuntimeError("No transformer layers found for ablation.")

    results_path = os.path.join(args.output_dir, "ablation_results.jsonl")
    tmp_eval_path = os.path.join(args.output_dir, "tmp_eval.jsonl")
    with open(results_path, "w") as f:
        base_overall, base_per_type = _run_eval(model, tokenizer, val_data, tmp_eval_path)
        f.write(json.dumps({"layer": "baseline", "metrics": base_overall, "per_type": base_per_type}) + "\n")

        for idx, layer in layers:
            _set_layer_adapters(layer, enabled=False)
            overall, per_type = _run_eval(model, tokenizer, val_data, tmp_eval_path)
            f.write(
                json.dumps(
                    {
                        "layer": f"layer_{idx}",
                        "metrics": overall,
                        "per_type": per_type,
                    }
                )
                + "\n"
            )
            _set_layer_adapters(layer, enabled=True)

    print(f"Ablation complete -> {results_path}")

    # Append summary rows to results.csv
    results_csv = "experiments/task2_layer_importance/results.csv"
    with open(results_path, "r") as rf, open(results_csv, "a", newline="") as out_csv:
        writer = csv.writer(out_csv)
        baseline = None
        ablated = []
        for line in rf:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("layer") == "baseline":
                baseline = row
            else:
                ablated.append(row)

        if baseline is None:
            return

        base_overall_f1 = baseline["metrics"]["f1"]
        base_per_type = baseline["per_type"]

        for row in ablated:
            layer = row["layer"]
            overall_f1 = row["metrics"]["f1"]
            writer.writerow(["task2", "ALL", layer, "f1_ablated", overall_f1, "ablation"])
            writer.writerow(["task2", "ALL", layer, "f1_delta", overall_f1 - base_overall_f1, "ablation"])

            for etype, metrics in row["per_type"].items():
                base_f1 = base_per_type.get(etype, {}).get("f1", 0.0)
                writer.writerow(["task2", etype, layer, "f1_ablated", metrics["f1"], "ablation"])
                writer.writerow(["task2", etype, layer, "f1_delta", metrics["f1"] - base_f1, "ablation"])


if __name__ == "__main__":
    main()
