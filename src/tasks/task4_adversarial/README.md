# Task 4: Adversarial Robustness

This task builds adversarial test cases and measures robustness before and after adversarial training.

## 1) Create eval sets (100 original + 100 adversarial)

```bash
python -m src.tasks.task4_adversarial.prepare_eval_set \
  --input_file data/processed/task1_val.jsonl \
  --output_prefix data/processed/adversarial/eval \
  --sample_count 100

python -m src.tasks.task4_adversarial.prepare_eval_set \
  --input_file data/processed/task1_test.jsonl \
  --output_prefix data/processed/adversarial/heldout \
  --sample_count 100
```

This produces:
- `data/processed/adversarial/eval_original.jsonl`
- `data/processed/adversarial/eval_adversarial.jsonl`
- `data/processed/adversarial/eval_combined.jsonl`
- `data/processed/adversarial/heldout_original.jsonl`
- `data/processed/adversarial/heldout_adversarial.jsonl`
- `data/processed/adversarial/heldout_combined.jsonl`

## 2) Baseline inference (pre-adversarial training)

```bash
python -m src.tasks.task4_adversarial.inference \
  --input_file data/processed/adversarial/eval_combined.jsonl \
  --output_file experiments/task4_adversarial/predictions_pre.jsonl \
  --model_path experiments/task1_constrained/adapter \
  --generation_mode free \
  --json_validate yes \
  --temperature 0.0

python -m src.tasks.task4_adversarial.evaluate \
  --input_file experiments/task4_adversarial/predictions_pre.jsonl \
  --results_file experiments/task4_adversarial/results_pre.csv \
  --summary_file experiments/task4_adversarial/metrics_pre.json \
  --label pre
```

## 3) Prepare adversarial training mix

```bash
python -m src.tasks.task4_adversarial.prepare_train_set \
  --input_file data/processed/task1_train.jsonl \
  --adv_count 1000
```

This writes:
- `data/processed/adversarial/train_adversarial.jsonl`
- `data/processed/adversarial/train_mixed.jsonl`

## 4) Adversarial training (2 epochs)

```bash
python -m src.tasks.task4_adversarial.train \
  --train_file data/processed/adversarial/train_mixed.jsonl \
  --val_file data/processed/task1_val.jsonl \
  --output_dir experiments/task4_adversarial/adapter \
  --num_train_epochs 2
```

## 5) Post-training evaluation on held-out adversarial set

```bash
python -m src.tasks.task4_adversarial.inference \
  --input_file data/processed/adversarial/heldout_combined.jsonl \
  --output_file experiments/task4_adversarial/predictions_post.jsonl \
  --model_path experiments/task4_adversarial/adapter \
  --generation_mode free \
  --json_validate yes \
  --temperature 0.0

python -m src.tasks.task4_adversarial.evaluate \
  --input_file experiments/task4_adversarial/predictions_post.jsonl \
  --results_file experiments/task4_adversarial/results_post.csv \
  --summary_file experiments/task4_adversarial/metrics_post.json \
  --label post
```

## 6) Quantify robustness gains

```bash
python -m src.tasks.task4_adversarial.compare_results \
  --baseline_file experiments/task4_adversarial/results_pre.csv \
  --new_file experiments/task4_adversarial/results_post.csv \
  --output_file experiments/task4_adversarial/robustness_gains.csv
```
