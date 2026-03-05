#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Gradio Active-Learning Interface Runner
# -----------------------------------------------------------------------------
# Launches the lightweight correction interface (`src.gradio_app`) with
# configurable host/port/model/adapter and export destinations.
# -----------------------------------------------------------------------------

HOST="127.0.0.1"
PORT="7860"
MODEL_NAME="Qwen/Qwen2.5-1.5B"
ADAPTER_PATH="experiments/with_defs_qwen2_5_1_5B"
PRED_EXPORT="data/active_learning/predictions_export.jsonl"
CORR_EXPORT="data/active_learning/corrections.jsonl"

usage() {
  cat <<USAGE
Usage: bash scripts/run_gradio_active_learning_interface.sh [options]

Options:
  --host HOST             Server host (default: 127.0.0.1)
  --port PORT             Server port (default: 7860)
  --model_name NAME       Base model name (default: Qwen/Qwen2.5-1.5B)
  --adapter_path PATH     Adapter path (default: experiments/with_defs_qwen2_5_1_5B)
  --prediction_export P   Prediction JSONL path (default: data/active_learning/predictions_export.jsonl)
  --correction_export P   Correction JSONL path (default: data/active_learning/corrections.jsonl)

Example:
  bash scripts/run_gradio_active_learning_interface.sh --port 7861
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --adapter_path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --prediction_export)
      PRED_EXPORT="$2"
      shift 2
      ;;
    --correction_export)
      CORR_EXPORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "Launching Gradio interface..."
echo "Host              : ${HOST}"
echo "Port              : ${PORT}"
echo "Model             : ${MODEL_NAME}"
echo "Adapter           : ${ADAPTER_PATH}"
echo "Prediction export : ${PRED_EXPORT}"
echo "Correction export : ${CORR_EXPORT}"

python -m src.gradio_app \
  --host "${HOST}" \
  --port "${PORT}" \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --prediction_export "${PRED_EXPORT}" \
  --correction_export "${CORR_EXPORT}"
