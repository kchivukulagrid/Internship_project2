"""Create adversarial eval sets for Task 4."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
INPUT_FILE = "data/processed/task1_val.jsonl"
OUTPUT_PREFIX = "data/processed/adversarial/eval"
SAMPLE_COUNT = 100
CATEGORIES = ["nested", "abbrev", "misspell", "ambiguous", "multilingual"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task 4 eval sets.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--sample_count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------
def _split_prompt(prompt: str) -> tuple[str, str, str]:
    marker = "\nText:\n"
    if marker not in prompt:
        return "", prompt, ""
    prefix, rest = prompt.split(marker, 1)
    suffix_marker = "\nReturn ONLY valid JSON"
    if suffix_marker in rest:
        text, suffix = rest.split(suffix_marker, 1)
        return prefix + marker, text, suffix_marker + suffix
    return prefix + marker, rest, ""


def _rebuild_prompt(prefix: str, text: str, suffix: str) -> str:
    return f"{prefix}{text}{suffix}"


# ---------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------
def _valid_entity(e: dict[str, Any]) -> bool:
    return (
        isinstance(e, dict)
        and isinstance(e.get("type"), str)
        and isinstance(e.get("value"), str)
        and isinstance(e.get("start"), int)
        and isinstance(e.get("end"), int)
    )


def _locate_entity(text: str, entity: dict[str, Any]) -> tuple[int, int] | None:
    start = entity.get("start")
    end = entity.get("end")
    value = entity.get("value")
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
        and isinstance(value, str)
        and text[start:end] == value
    ):
        return start, end
    if isinstance(value, str):
        idx = text.find(value)
        if idx != -1:
            return idx, idx + len(value)
    return None


def _shift_entities(
    entities: list[dict[str, Any]],
    target_idx: int,
    pivot_end: int,
    delta: int,
    target_start: int,
    target_end: int,
    target_value: str,
    extra_entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for i, ent in enumerate(entities):
        if not _valid_entity(ent):
            continue
        s = ent["start"]
        e = ent["end"]
        if i == target_idx:
            s = target_start
            e = target_end
            ent = dict(ent)
            ent["value"] = target_value
        elif s >= pivot_end:
            s = s + delta
            e = e + delta
        ent = dict(ent)
        ent["start"] = s
        ent["end"] = e
        updated.append(ent)
    if extra_entities:
        updated.extend(extra_entities)
    return updated


# ---------------------------------------------------------------------
# Adversarial transforms
# ---------------------------------------------------------------------
def _make_abbrev(value: str) -> str | None:
    parts = [p for p in value.replace("-", " ").split() if p]
    if len(parts) >= 2:
        abbr = "".join(p[0] for p in parts if p[0].isalpha()).upper()
        return abbr if len(abbr) >= 2 else None
    if len(value) >= 4:
        return value[:3].upper()
    return None


def _misspell(value: str) -> str | None:
    if len(value) < 4:
        return None
    i = len(value) // 2
    if i <= 0 or i >= len(value) - 1:
        return None
    return value[: i - 1] + value[i] + value[i - 1] + value[i + 1 :]


def _transform_row(row: dict[str, Any], category: str, rng: random.Random) -> dict[str, Any]:
    prompt = row["prompt"]
    output = json.loads(row["output"])
    entities = [e for e in output.get("entities", []) if _valid_entity(e)]

    prefix, text, suffix = _split_prompt(prompt)
    if not entities and category != "multilingual":
        category = "multilingual"

    if category == "multilingual":
        text = text.rstrip() + " Texto adicional en espanol."
        new_prompt = _rebuild_prompt(prefix, text, suffix)
        return {
            "prompt": new_prompt,
            "output": json.dumps(output),
            "category": "multilingual",
        }

    # Choose a target entity.
    target_idx = 0
    target = entities[target_idx]
    loc = _locate_entity(text, target)
    if loc is None:
        # Fallback to multilingual if spans are unreliable.
        text = text.rstrip() + " Texto adicional en espanol."
        new_prompt = _rebuild_prompt(prefix, text, suffix)
        return {
            "prompt": new_prompt,
            "output": json.dumps(output),
            "category": "multilingual",
        }
    start, end = loc
    value = target["value"]

    if category == "abbrev":
        abbr = _make_abbrev(value)
        if not abbr:
            category = "misspell"
        else:
            new_text = text[:start] + abbr + text[end:]
            delta = len(abbr) - (end - start)
            new_entities = _shift_entities(
                entities,
                target_idx,
                end,
                delta,
                start,
                start + len(abbr),
                abbr,
            )
            new_output = dict(output)
            new_output["entities"] = new_entities
            new_prompt = _rebuild_prompt(prefix, new_text, suffix)
            return {
                "prompt": new_prompt,
                "output": json.dumps(new_output),
                "category": "abbrev",
            }

    if category == "misspell":
        miss = _misspell(value)
        if not miss:
            category = "multilingual"
        else:
            new_text = text[:start] + miss + text[end:]
            delta = len(miss) - (end - start)
            new_entities = _shift_entities(
                entities,
                target_idx,
                end,
                delta,
                start,
                start + len(miss),
                miss,
            )
            new_output = dict(output)
            new_output["entities"] = new_entities
            new_prompt = _rebuild_prompt(prefix, new_text, suffix)
            return {
                "prompt": new_prompt,
                "output": json.dumps(new_output),
                "category": "misspell",
            }

    if category == "ambiguous":
        wrapped = f"\"{value}\""
        new_text = text[:start] + wrapped + text[end:]
        delta = len(wrapped) - (end - start)
        # Keep entity value the same, but shift spans to exclude quotes.
        new_entities = _shift_entities(
            entities,
            target_idx,
            end,
            delta,
            start + 1,
            start + 1 + len(value),
            value,
        )
        new_output = dict(output)
        new_output["entities"] = new_entities
        new_prompt = _rebuild_prompt(prefix, new_text, suffix)
        return {
            "prompt": new_prompt,
            "output": json.dumps(new_output),
            "category": "ambiguous",
        }

    if category == "nested":
        prefix_text = "International "
        new_value = prefix_text + value
        new_text = text[:start] + new_value + text[end:]
        delta = len(new_value) - (end - start)
        inner_start = start + len(prefix_text)
        inner_end = inner_start + len(value)
        outer_entity = dict(target)
        outer_entity["value"] = new_value
        outer_entity["start"] = start
        outer_entity["end"] = start + len(new_value)
        new_entities = _shift_entities(
            entities,
            target_idx,
            end,
            delta,
            inner_start,
            inner_end,
            value,
            extra_entities=[outer_entity],
        )
        new_output = dict(output)
        new_output["entities"] = new_entities
        new_prompt = _rebuild_prompt(prefix, new_text, suffix)
        return {
            "prompt": new_prompt,
            "output": json.dumps(new_output),
            "category": "nested",
        }

    # Fallback: multilingual noise.
    text = text.rstrip() + " Texto adicional en espanol."
    new_prompt = _rebuild_prompt(prefix, text, suffix)
    return {
        "prompt": new_prompt,
        "output": json.dumps(output),
        "category": "multilingual",
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.input_file, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows = rows[: min(args.sample_count, len(rows))]

    original_rows = [
        {"prompt": r["prompt"], "output": r["output"], "category": "original"}
        for r in rows
    ]

    adversarial_rows = []
    for i, row in enumerate(rows):
        category = CATEGORIES[i % len(CATEGORIES)]
        adversarial_rows.append(_transform_row(row, category, rng))

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    original_path = f"{args.output_prefix}_original.jsonl"
    adversarial_path = f"{args.output_prefix}_adversarial.jsonl"
    combined_path = f"{args.output_prefix}_combined.jsonl"

    with open(original_path, "w") as f:
        for row in original_rows:
            f.write(json.dumps(row) + "\n")

    with open(adversarial_path, "w") as f:
        for row in adversarial_rows:
            f.write(json.dumps(row) + "\n")

    with open(combined_path, "w") as f:
        for row in original_rows + adversarial_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Original:    {original_path} ({len(original_rows)})")
    print(f"Adversarial: {adversarial_path} ({len(adversarial_rows)})")
    print(f"Combined:    {combined_path} ({len(original_rows) + len(adversarial_rows)})")


if __name__ == "__main__":
    main()
