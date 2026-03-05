"""Train a LoRA adapter for the NER-to-JSON task."""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

from src.model import load_model

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "experiments/qwen2_5_1_5B_masked_tuned"
MAX_LENGTH = 384
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA model on processed JSONL data.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--val_file", default=VAL_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )
    print("\nTrain size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))

    # -----------------------------------------------------------------
    # Tokenizer + tokenization
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict) -> dict:
        prompt = example["prompt"]
        output = example["output"]

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

        full_tokens = tokenizer(
            prompt + "\n\n" + output,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()

        prompt_length = len(prompt_tokens["input_ids"])
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    # -----------------------------------------------------------------
    # Model + trainer
    # -----------------------------------------------------------------
    model, _ = load_model(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # -----------------------------------------------------------------
    # Train + save
    # -----------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)

    print("\nTraining complete.")
    print("Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
