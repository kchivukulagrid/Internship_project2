# =========================================================
# TRAINING SCRIPT (STEP 2 + STEP 3 COMBINED)
# =========================================================

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from src.model import load_model

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "experiments/qwen2_5_1_5B_masked_tuned"
MAX_LENGTH = 384

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl",
    },
)

print("\nTrain size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))

# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# TOKENIZATION WITH PROMPT MASKING
# ---------------------------------------------------------

def tokenize(example):

    prompt = example["prompt"]
    output = example["output"]

    # Tokenize prompt separately
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    # Tokenize full sequence
    full_tokens = tokenizer(
        prompt + "\n\n" + output,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    # 🔥 MASK PROMPT TOKENS
    prompt_length = len(prompt_tokens["input_ids"])

    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model, device = load_model(MODEL_NAME)

# ---------------------------------------------------------
# TRAINING ARGUMENTS (TUNED)
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    logging_steps=50,
    dataloader_pin_memory=False,
    report_to="none",
)

# ---------------------------------------------------------
# TRAINER
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

trainer.train()
trainer.save_model(OUTPUT_DIR)

print("\nTraining complete.")
print("Model saved to:", OUTPUT_DIR)
