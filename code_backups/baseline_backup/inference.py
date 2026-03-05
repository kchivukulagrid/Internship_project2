import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import extract_json
from src.model import get_device


MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_PATH = "experiments/qwen2_5_1_5B_masked_tuned"
OUTPUT_FILE = "data/processed/exports/qwen2_5_1_5B_masked_tuned_predictions.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for NER JSON extraction.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.1, 0.2])
    parser.add_argument("--json_validate", default="yes", choices=["yes", "no"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)
    print(
        f"Config -> temperature={args.temperature}, json_validate={args.json_validate}, "
        f"output_file={args.output_file}"
    )

    dataset = load_dataset(
        "json",
        data_files={"validation": "data/processed/val.jsonl"},
    )
    val_data = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(model, args.model_path)
    model.to(device)
    model.eval()

    do_sample = args.temperature > 0.0
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        for i, example in enumerate(val_data):
            print(f"{i+1}/{len(val_data)}")

            prompt = (
                example["prompt"]
                + '\nReturn ONLY valid JSON with schema: {"entities":[{"text":"...","label":"PER|ORG|LOC|MISC"}]}\n'
                + "Output:\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generation_kwargs = {
                "max_new_tokens": 256,
                "do_sample": do_sample,
                "repetition_penalty": 1.1,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
            }
            generation_kwargs["temperature"] = args.temperature if do_sample else 0.0

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if args.json_validate == "yes":
                parsed = extract_json(decoded)
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
