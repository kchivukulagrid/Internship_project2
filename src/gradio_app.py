"""Lightweight Gradio correction interface for NER JSON extraction."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import extract_json
from src.model import get_device
from src.preprocess import build_prompt


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = "experiments/with_defs_qwen2_5_1_5B"
FALLBACK_ADAPTER_PATH = "experiments/qwen2_5_1_5B_masked_tuned"

DEFAULT_PREDICTION_EXPORT = "data/active_learning/predictions_export.jsonl"
DEFAULT_CORRECTION_EXPORT = "data/active_learning/corrections.jsonl"


# ---------------------------
# CLI + Path Helpers
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio correction interface for NER JSON extraction.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--prediction_export", default=DEFAULT_PREDICTION_EXPORT)
    parser.add_argument("--correction_export", default=DEFAULT_CORRECTION_EXPORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def resolve_adapter_path(path: str) -> str:
    if Path(path).exists():
        return path
    if Path(FALLBACK_ADAPTER_PATH).exists():
        return FALLBACK_ADAPTER_PATH
    raise FileNotFoundError(
        f"Adapter path not found: {path} (fallback also missing: {FALLBACK_ADAPTER_PATH})"
    )


# ---------------------------
# Parsing + IO Helpers
# ---------------------------
def normalize_prediction(text: str) -> dict:
    parsed = extract_json(text)
    if parsed is None:
        return {"entities": []}
    return parsed


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------
# Runtime Builders
# ---------------------------
def build_runtime(model_name: str, adapter_path: str):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


# ---------------------------
# UI Callback Factories
# ---------------------------
def make_predict_fn(model, tokenizer, device, prediction_export: str):
    def predict(text: str):
        text = (text or "").strip()
        if not text:
            empty = json.dumps({"entities": []}, ensure_ascii=False, indent=2)
            return empty, empty, "Enter text first."

        prompt = build_prompt(text, prompt_style="with_defs")
        prompt += '\nReturn ONLY valid JSON with schema: {"entities":[{"text":"...","label":"PER|ORG|LOC|MISC"}]}\nOutput:\n'
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generation_kwargs = {
            "max_new_tokens": 192,
            "do_sample": False,
            "temperature": 0.0,
            "repetition_penalty": 1.05,
            "num_beams": 1,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        prediction = normalize_prediction(decoded)

        timestamp = datetime.now(timezone.utc).isoformat()
        append_jsonl(
            prediction_export,
            {
                "timestamp_utc": timestamp,
                "text": text,
                "prompt": prompt,
                "raw_model_output": decoded,
                "prediction": prediction,
            },
        )

        pretty = json.dumps(prediction, ensure_ascii=False, indent=2)
        return pretty, pretty, f"Prediction generated and exported to {prediction_export}"

    return predict


def make_save_fn(correction_export: str):
    def save_correction(text: str, predicted_json: str, corrected_json: str):
        text = (text or "").strip()
        if not text:
            return "Cannot save: text is empty."

        try:
            predicted_obj = json.loads(predicted_json)
            corrected_obj = json.loads(corrected_json)
        except json.JSONDecodeError as e:
            return f"Cannot save: corrected JSON is invalid ({e})"

        timestamp = datetime.now(timezone.utc).isoformat()
        append_jsonl(
            correction_export,
            {
                "timestamp_utc": timestamp,
                "text": text,
                "predicted": predicted_obj,
                "corrected": corrected_obj,
                "accepted": predicted_obj == corrected_obj,
            },
        )
        return f"Saved to {correction_export}"

    return save_correction


# ---------------------------
# Main Entrypoint
# ---------------------------
def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_path(args.adapter_path)
    model, tokenizer, device = build_runtime(args.model_name, adapter_path)

    predict_fn = make_predict_fn(model, tokenizer, device, args.prediction_export)
    save_fn = make_save_fn(args.correction_export)

    with gr.Blocks(title="NER JSON Correction Interface") as demo:
        gr.Markdown("# NER to JSON Correction Interface")
        gr.Markdown(
            "1) Enter text, 2) Generate prediction, 3) Edit corrected JSON, 4) Save correction for active learning."
        )

        input_text = gr.Textbox(label="Input Text", lines=6, placeholder="Paste sentence/document text here...")
        with gr.Row():
            predict_btn = gr.Button("Generate Prediction", variant="primary")
            save_btn = gr.Button("Save Correction")

        predicted_json = gr.Code(label="Predicted JSON", language="json")
        corrected_json = gr.Code(label="Corrected JSON (Editable)", language="json")
        status = gr.Textbox(label="Status", interactive=False)

        predict_btn.click(
            fn=predict_fn,
            inputs=[input_text],
            outputs=[predicted_json, corrected_json, status],
        )
        save_btn.click(
            fn=save_fn,
            inputs=[input_text, predicted_json, corrected_json],
            outputs=[status],
        )

    print(f"Using adapter: {adapter_path}")
    print(f"Device: {device}")
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
