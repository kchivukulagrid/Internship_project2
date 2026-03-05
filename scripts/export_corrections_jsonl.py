"""Export normalized corrections into a training-ready JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.correction_io import read_jsonl, write_jsonl
from src.correction_schema import normalize_payload
from src.preprocess import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export corrections as prompt/output JSONL.")
    parser.add_argument("--input_file", default="data/processed/corrections/corrections.jsonl")
    parser.add_argument("--output_file", default="data/processed/active_learning/corrections_train.jsonl")
    parser.add_argument("--prompt_style", default="with_defs", choices=["with_defs", "no_defs"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_file)

    out_rows = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        corrected = normalize_payload(row.get("corrected", {}))
        out_rows.append(
            {
                "prompt": build_prompt(text, prompt_style=args.prompt_style),
                "output": json.dumps(corrected, ensure_ascii=False),
            }
        )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_file, out_rows)
    print(args.output_file)


if __name__ == "__main__":
    main()
