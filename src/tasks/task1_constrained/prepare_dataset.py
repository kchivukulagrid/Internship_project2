"""Build Task 1 datasets with strict schema and offsets."""

from __future__ import annotations

import argparse
import json
import os

from src.core.data_utils import build_output, extract_entities_with_offsets
from src.core.prompts import build_prompt
from src.data_loader import load_conll2003_local

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Task 1 JSONL datasets.")
    parser.add_argument("--prompt_style", default="with_defs", choices=["with_defs", "no_defs"])
    parser.add_argument("--train_output", default="data/processed/task1_train.jsonl")
    parser.add_argument("--val_output", default="data/processed/task1_val.jsonl")
    parser.add_argument("--test_output", default="data/processed/task1_test.jsonl")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Conversion Helpers
# ---------------------------------------------------------------------
def _convert_example(example: dict, label_names: list[str], prompt_style: str) -> dict:
    tokens = example["tokens"]
    tags = example["ner_tags"]
    # Build canonical output with char offsets and confidence.
    entities = extract_entities_with_offsets(tokens, tags, label_names)
    text = " ".join(tokens)
    prompt = build_prompt(text, prompt_style=prompt_style)
    output = json.dumps(build_output(entities, confidence=1.0))
    return {"prompt": prompt, "output": output}


# ---------------------------------------------------------------------
# IO Helpers
# ---------------------------------------------------------------------
def _save_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    dataset = load_conll2003_local()
    label_names = dataset["train"].features["ner_tags"].feature.names

    train_data = [
        _convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["train"]
    ]
    val_data = [
        _convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["validation"]
    ]
    test_data = [
        _convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["test"]
    ]

    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    _save_jsonl(train_data, args.train_output)
    _save_jsonl(val_data, args.val_output)
    _save_jsonl(test_data, args.test_output)

    print("Task 1 dataset build complete.")
    print(f"Train -> {args.train_output} ({len(train_data)} samples)")
    print(f"Val   -> {args.val_output} ({len(val_data)} samples)")
    print(f"Test  -> {args.test_output} ({len(test_data)} samples)")


if __name__ == "__main__":
    main()
