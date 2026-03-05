"""Build processed train/validation/test JSONL files for NER-to-JSON training."""

import argparse
import json
import os

from src.data_loader import load_conll2003_local
from src.preprocess import augment_text_and_entities, build_prompt, convert_example


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed JSONL datasets.")
    parser.add_argument("--prompt_style", default="with_defs", choices=["with_defs", "no_defs"])
    parser.add_argument("--synonym_aug", default="no", choices=["yes", "no"])
    parser.add_argument("--train_output", default="data/processed/train.jsonl")
    parser.add_argument("--val_output", default="data/processed/val.jsonl")
    parser.add_argument("--test_output", default="data/processed/test.jsonl")
    return parser.parse_args()


def save_jsonl(data: list[dict], path: str) -> None:
    """Persist a list of dictionaries in JSONL format."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _extract_text_from_prompt(prompt: str) -> str:
    """Recover source text from the generated prompt."""
    marker = "\nText:\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    print("Loading raw dataset...")
    dataset = load_conll2003_local()
    label_names = dataset["train"].features["ner_tags"].feature.names

    # -----------------------------------------------------------------
    # Convert train split
    # -----------------------------------------------------------------
    print("Converting train split...")
    train_data = [
        convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["train"]
    ]

    if args.synonym_aug == "yes":
        print("Applying synonym augmentation to train split...")
        aug_examples = []
        for item in train_data:
            entities = json.loads(item["output"]).get("entities", [])
            if not entities:
                continue

            text = _extract_text_from_prompt(item["prompt"])
            aug_text, aug_entities = augment_text_and_entities(text, entities)
            aug_examples.append(
                {
                    "prompt": build_prompt(aug_text, prompt_style=args.prompt_style),
                    "output": json.dumps({"entities": aug_entities}),
                }
            )
        train_data.extend(aug_examples)
        print(f"Augmented examples added: {len(aug_examples)}")

    # -----------------------------------------------------------------
    # Convert validation + test splits
    # -----------------------------------------------------------------
    print("Converting validation split...")
    val_data = [
        convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["validation"]
    ]

    print("Converting test split...")
    test_data = [
        convert_example(example, label_names, prompt_style=args.prompt_style)
        for example in dataset["test"]
    ]

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)

    save_jsonl(train_data, args.train_output)
    save_jsonl(val_data, args.val_output)
    save_jsonl(test_data, args.test_output)

    print("Dataset build complete.")
    print(f"Train -> {args.train_output} ({len(train_data)} samples)")
    print(f"Val   -> {args.val_output} ({len(val_data)} samples)")
    print(f"Test  -> {args.test_output} ({len(test_data)} samples)")


if __name__ == "__main__":
    main()
