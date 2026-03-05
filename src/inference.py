"""Run inference for NER-to-JSON extraction and export JSONL predictions."""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import extract_json
from src.model import get_device


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/qwen2_5_1_5B_masked_tuned"
INPUT_FILE = "data/processed/val.jsonl"
OUTPUT_FILE = "data/processed/exports/qwen2_5_1_5B_masked_tuned_predictions.jsonl"
VALID_LABELS = {"PER", "ORG", "LOC", "MISC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for NER extraction.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.1, 0.2])
    parser.add_argument("--json_validate", default="yes", choices=["yes", "no"])
    parser.add_argument("--output_format", default="json", choices=["json", "xml", "plain"])
    parser.add_argument("--generation_mode", default="free", choices=["free", "constrained"])
    return parser.parse_args()


def _normalize_entities(entities: list[dict]) -> list[dict]:
    """Normalize entities and remove invalid/duplicate entries."""
    normalized = []
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        text = entity.get("text")
        label = entity.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            continue
        text = text.strip()
        label = label.strip().upper()
        if not text or label not in VALID_LABELS:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"text": text, "label": label})
    return normalized


def _build_prompt(base_prompt: str, output_format: str) -> str:
    """Build output-format-specific prompt."""
    if output_format == "json":
        return (
            base_prompt
            + '\nReturn ONLY valid JSON with schema: {"entities":[{"text":"...","label":"PER|ORG|LOC|MISC"}]}\n'
            + "Output:\n"
        )
    if output_format == "xml":
        return (
            base_prompt
            + "\nReturn ONLY XML with schema: "
            + "<entities><entity><text>...</text><label>PER|ORG|LOC|MISC</label></entity></entities>\n"
            + "Output:\n"
        )
    return (
        base_prompt
        + "\nReturn ONLY plain text with one entity per line using EXACT format: <text>\\t<label> "
        + "where label is one of PER|ORG|LOC|MISC.\n"
        + "Do not add numbering, bullets, explanations, or extra symbols.\n"
        + "If no entities, output exactly: NONE\n"
        + "Example:\nJohn Doe\tPER\nLondon\tLOC\n"
        + "Output:\n"
    )


def _extract_xml(decoded: str):
    entities = []
    for block in re.findall(r"<entity>(.*?)</entity>", decoded, flags=re.IGNORECASE | re.DOTALL):
        text_match = re.search(r"<text>(.*?)</text>", block, flags=re.IGNORECASE | re.DOTALL)
        label_match = re.search(r"<label>(.*?)</label>", block, flags=re.IGNORECASE | re.DOTALL)
        if not text_match or not label_match:
            continue
        entities.append(
            {
                "text": text_match.group(1).strip(),
                "label": label_match.group(1).strip().upper(),
            }
        )
    entities = _normalize_entities(entities)
    return {"entities": entities} if entities else None


def _extract_plain(decoded: str):
    entities = []
    for raw_line in decoded.splitlines():
        line = raw_line.strip()
        if not line or line.upper() == "NONE":
            continue

        # Remove common bullet/numbering prefixes from plain-text lists.
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)

        if "\t" in line:
            text, label = line.rsplit("\t", 1)
        elif " | " in line:
            text, label = line.rsplit(" | ", 1)
        elif " - " in line:
            text, label = line.rsplit(" - ", 1)
        elif ":" in line:
            left, right = line.split(":", 1)
            left = left.strip().upper()
            right = right.strip()
            if left in VALID_LABELS:
                text, label = right, left
            elif right.upper() in VALID_LABELS:
                text, label = left, right
            else:
                continue
        else:
            # Recover from lines like: "text (LABEL)" or "text [LABEL]"
            m = re.match(
                r"^(?P<text>.+?)\s*[\(\[]\s*(?P<label>PER|ORG|LOC|MISC)\s*[\)\]]\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                text, label = m.group("text"), m.group("label")
            else:
                continue

        # Final normalization/validation happens in _normalize_entities.
        entities.append({"text": text.strip(), "label": label.strip().upper()})

    # Backup regex recovery across the full decoded output when line parsing fails.
    if not entities:
        for m in re.finditer(
            r"(?P<label>PER|ORG|LOC|MISC)\s*[:|-]\s*(?P<text>[^\n]+)",
            decoded,
            flags=re.IGNORECASE,
        ):
            entities.append({"text": m.group("text").strip(), "label": m.group("label").strip().upper()})
        for m in re.finditer(
            r"(?P<text>[^\n:|\\-]+?)\s*[:|-]\s*(?P<label>PER|ORG|LOC|MISC)",
            decoded,
            flags=re.IGNORECASE,
        ):
            entities.append({"text": m.group("text").strip(), "label": m.group("label").strip().upper()})

    entities = _normalize_entities(entities)
    return {"entities": entities} if entities else None


def _parse_prediction(decoded: str, output_format: str):
    """
    Parse model output into canonical {"entities":[...]} with robust fallbacks.
    Primary parser is based on requested format, then alternate parsers are tried.
    """
    if output_format == "json":
        parsers = [extract_json, _extract_xml, _extract_plain]
    elif output_format == "xml":
        parsers = [_extract_xml, extract_json, _extract_plain]
    else:
        parsers = [_extract_plain, extract_json, _extract_xml]

    for parser in parsers:
        parsed = parser(decoded)
        if parsed is not None:
            return parsed
    return None


def main() -> None:
    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    args = parse_args()
    device = get_device()
    print("Using device:", device)
    print(
        f"Config -> temperature={args.temperature}, json_validate={args.json_validate}, "
        f"output_format={args.output_format}, "
        f"generation_mode={args.generation_mode}, "
        f"input_file={args.input_file}, "
        f"output_file={args.output_file}"
    )

    dataset = load_dataset("json", data_files={"validation": args.input_file})
    val_data = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(model, args.model_path)
    model.to(device)
    model.eval()

    do_sample = args.temperature > 0.0
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # -----------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------
    with open(args.output_file, "w") as f:
        for i, example in enumerate(val_data):
            print(f"{i + 1}/{len(val_data)}")

            prompt = _build_prompt(example["prompt"], args.output_format)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generation_kwargs = {
                "max_new_tokens": 256,
                "do_sample": do_sample,
                "repetition_penalty": 1.1,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": args.temperature if do_sample else 0.0,
            }
            if args.generation_mode == "constrained":
                # Use conservative decoding knobs to reduce malformed/verbose output.
                generation_kwargs["do_sample"] = False
                generation_kwargs["temperature"] = 0.0
                generation_kwargs["num_beams"] = 1
                generation_kwargs["max_new_tokens"] = 192
                generation_kwargs["repetition_penalty"] = 1.05

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if args.json_validate == "yes":
                parsed = _parse_prediction(decoded, args.output_format)
                if parsed is None:
                    parsed = {"entities": []}
                prediction = json.dumps(parsed, ensure_ascii=False)
            else:
                prediction = decoded

            f.write(
                json.dumps(
                    {
                        "ground_truth": example["output"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nInference complete.")


if __name__ == "__main__":
    main()
