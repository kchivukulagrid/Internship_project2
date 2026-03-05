import json
import os
from src.metrics import compute_metrics

# 🔥 Correct file path
INPUT_FILE = "data/processed/exports/qwen2_5_1_5B_masked_tuned_predictions.jsonl"
OUTPUT_FILE = "experiments/qwen2_5_1_5B_masked_tuned/metrics.json"

results = compute_metrics(INPUT_FILE)

print("\n==============================")
print("        EVALUATION RESULTS")
print("==============================")
print(f"Precision : {results['precision']:.4f}")
print(f"Recall    : {results['recall']:.4f}")
print(f"F1 Score  : {results['f1']:.4f}")
print(f"Validity  : {results['validity']:.4f}")
print(f"Validity% : {results['validity'] * 100:.2f}%")
print(f"Valid JSON: {results['valid_json_count']}/{results['total_examples']}")
print(f"Repaired  : {results['repaired_json_count']}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)
