"""Build a review queue from prediction exports for correction UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.correction_io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review queue JSONL.")
    parser.add_argument("--input_file", default="data/processed/exports/dp_wdefs_test_pred.jsonl")
    parser.add_argument("--output_file", default="data/processed/review_queue/review_queue.jsonl")
    parser.add_argument("--max_items", type=int, default=500)
    return parser.parse_args()


def queue_score(row: dict) -> tuple[int, int]:
    pred = row.get("prediction")
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            pred = {"entities": []}
    entities = pred.get("entities", []) if isinstance(pred, dict) else []
    return (0 if entities else 1, len(entities))


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_file)

    ranked = []
    for i, row in enumerate(rows):
        item = {
            "id": i,
            "text": row.get("text") or row.get("ground_truth") or "",
            "ground_truth": row.get("ground_truth", ""),
            "prediction": row.get("prediction", ""),
        }
        ranked.append((queue_score(item), item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    queue = [item for _, item in ranked[: args.max_items]]

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_file, queue)
    print(args.output_file)


if __name__ == "__main__":
    main()
