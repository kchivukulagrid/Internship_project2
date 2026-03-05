#!/usr/bin/env bash
set -euo pipefail

# Cloud entrypoint for Hugging Face Spaces / generic hosting.
# Uses environment variables where available.

HOST="0.0.0.0"
PORT="${PORT:-7860}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
ADAPTER_PATH="${ADAPTER_PATH:-experiments/with_defs_qwen2_5_1_5B}"
PRED_EXPORT="${PRED_EXPORT:-data/processed/active_learning/predictions_export.jsonl}"
CORR_EXPORT="${CORR_EXPORT:-data/processed/corrections/corrections.jsonl}"
AL_EXPORT="${AL_EXPORT:-data/processed/active_learning/cycle_records.jsonl}"

python -m src.gradio_correction_app \
  --host "${HOST}" \
  --port "${PORT}" \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --prediction_export "${PRED_EXPORT}" \
  --correction_export "${CORR_EXPORT}" \
  --active_learning_export "${AL_EXPORT}"
