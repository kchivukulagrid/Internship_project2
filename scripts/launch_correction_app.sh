#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT="7860"
MODEL_NAME="Qwen/Qwen2.5-1.5B"
ADAPTER_PATH="experiments/with_defs_qwen2_5_1_5B"
SHARE="false"

PRED_EXPORT="data/processed/active_learning/predictions_export.jsonl"
CORR_EXPORT="data/processed/corrections/corrections.jsonl"
AL_EXPORT="data/processed/active_learning/cycle_records.jsonl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --share)
      SHARE="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cmd=(
  python -m src.gradio_correction_app
  --host "${HOST}"
  --port "${PORT}"
  --model_name "${MODEL_NAME}"
  --adapter_path "${ADAPTER_PATH}"
  --prediction_export "${PRED_EXPORT}"
  --correction_export "${CORR_EXPORT}"
  --active_learning_export "${AL_EXPORT}"
)

if [[ "${SHARE}" == "true" ]]; then
  cmd+=(--share)
fi

"${cmd[@]}"
