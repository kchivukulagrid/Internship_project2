"""Prepare strict vs loose boundary datasets for Task 3."""

from __future__ import annotations

import argparse
import json
import os
import random


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "data/processed/task1_val.jsonl"
STRICT_FILE = "data/steering/strict.jsonl"
LOOSE_FILE = "data/steering/loose.jsonl"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare strict/loose boundary sets.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--strict_file", default=STRICT_FILE)
    parser.add_argument("--loose_file", default=LOOSE_FILE)
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _extract_text(prompt: str) -> str:
    marker = "\nText:\n"
    if marker in prompt:
        after = prompt.split(marker, 1)[1]
        # Stop before the schema hint if present.
        stop = "\nReturn ONLY valid JSON"
        if stop in after:
            return after.split(stop, 1)[0].strip()
        return after.strip()
    return prompt.strip()


def _perturb_entity(entity: dict, text: str) -> dict:
    start = int(entity["start"])
    end = int(entity["end"])
    # Expand or contract by 1 character when possible.
    if start > 0 and random.random() < 0.5:
        start -= 1
    if end < len(text) and random.random() < 0.5:
        end += 1
    if start >= end:
        start = max(0, start - 1)
        end = min(len(text), end + 1)
    value = text[start:end]
    return {
        "type": entity["type"],
        "value": value,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    with open(args.input_file, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows = rows[: min(args.sample_count, len(rows))]

    strict_rows = []
    loose_rows = []
    for row in rows:
        prompt = row["prompt"]
        text = _extract_text(prompt)
        output = json.loads(row["output"])
        entities = output.get("entities", [])

        strict_rows.append(row)

        loose_entities = [_perturb_entity(e, text) for e in entities if e]
        loose_output = {"entities": loose_entities, "confidence": output.get("confidence", 1.0)}
        loose_rows.append({"prompt": prompt, "output": json.dumps(loose_output)})

    os.makedirs(os.path.dirname(args.strict_file), exist_ok=True)
    with open(args.strict_file, "w") as f:
        for row in strict_rows:
            f.write(json.dumps(row) + "\n")

    with open(args.loose_file, "w") as f:
        for row in loose_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Strict set -> {args.strict_file} ({len(strict_rows)})")
    print(f"Loose set  -> {args.loose_file} ({len(loose_rows)})")


if __name__ == "__main__":
    main()

