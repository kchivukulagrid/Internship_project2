import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.model import get_device
from src.metrics import extract_json


MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/qwen2_5_1_5B_masked_tuned"
OUTPUT_FILE = "data/processed/exports/qwen2_5_1_5B_masked_tuned_predictions.jsonl"

device = get_device()
print("Using device:", device)

dataset = load_dataset(
    "json",
    data_files={"validation": "data/processed/val.jsonl"},
)

val_data = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, MODEL_PATH)

model.to(device)
model.eval()

with open(OUTPUT_FILE, "w") as f:
    for i, example in enumerate(val_data):
        print(f"{i+1}/{len(val_data)}")

        prompt = (
            example["prompt"]
            + '\nReturn ONLY valid JSON with schema: {"entities":[{"text":"...","label":"PER|ORG|LOC|MISC"}]}\n'
            + "Output:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parsed = extract_json(decoded)
        if parsed is None:
            parsed = {"entities": []}
        canonical_prediction = json.dumps(parsed, ensure_ascii=False)

        f.write(
            json.dumps(
                {
                    "ground_truth": example["output"],
                    "prediction": canonical_prediction,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print("\nInference complete (Safe constrained).")
