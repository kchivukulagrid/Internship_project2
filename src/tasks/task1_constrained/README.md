# Task 1: Constrained Decoding

This task introduces strict JSON schema output with offsets and a confidence score, then benchmarks constrained vs free decoding.

Key scripts:
- `src/tasks/task1_constrained/prepare_dataset.py`
- `src/tasks/task1_constrained/train.py`
- `src/tasks/task1_constrained/inference.py`
- `src/tasks/task1_constrained/benchmark.py`
- `src/tasks/task1_constrained/evaluate.py`

Quick start:
```bash
python -m src.tasks.task1_constrained.prepare_dataset
python -m src.tasks.task1_constrained.train --num_train_epochs 1
python -m src.tasks.task1_constrained.inference --input_file data/processed/task1_val.jsonl
python -m src.tasks.task1_constrained.evaluate --input_file data/processed/exports/task1_predictions.jsonl
python -m src.tasks.task1_constrained.benchmark --input_file data/processed/task1_test.jsonl --sample_count 500
```
