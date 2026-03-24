#!/usr/bin/env bash
set -euo pipefail

python -m src.tasks.task1_constrained.benchmark --input_file data/processed/task1_test.jsonl --sample_count 500
