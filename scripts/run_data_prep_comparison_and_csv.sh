#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Data Prep Comparison Runner
# -----------------------------------------------------------------------------
# Variants compared:
#   1) with_defs : prompt contains entity definitions
#   2) no_defs   : prompt excludes entity definitions
#   3) syn_aug   : with_defs + train-set synonym augmentation
#
# For each variant this script:
#   - builds variant-specific processed files
#   - trains a LoRA model
#   - runs validation inference
#   - runs evaluation
#   - appends one row to a summary CSV
# -----------------------------------------------------------------------------

BASE_DIR="experiments/data_prep_comparison"
DATA_DIR="data/processed/variants"
EXPORT_DIR="data/processed/exports"
MODEL_NAME="Qwen/Qwen2.5-1.5B"
EPOCHS="2"
LEARNING_RATE="2e-4"
MAX_LENGTH="384"
TEMPERATURE="0.0"
JSON_VALIDATE="yes"
OUTPUT_FORMAT="json"
GENERATION_MODE="constrained"

usage() {
  cat <<USAGE
Usage: bash scripts/run_data_prep_comparison_and_csv.sh [options]

Options:
  --epochs VALUE          Num train epochs (default: 2)
  --learning_rate VALUE   Learning rate (default: 2e-4)
  --temperature VALUE     Inference temperature (default: 0.0)
  --generation_mode MODE  free|constrained (default: constrained)
  --json_validate yes|no  Inference JSON validate flag (default: yes)

Example:
  bash scripts/run_data_prep_comparison_and_csv.sh --epochs 2 --generation_mode constrained
USAGE
}

# -----------------------------------------------------------------------------
# Parse CLI options
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --generation_mode)
      GENERATION_MODE="$2"
      shift 2
      ;;
    --json_validate)
      JSON_VALIDATE="$2"
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

# -----------------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------------
if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available in PATH." >&2
  exit 1
fi

if [[ "${JSON_VALIDATE}" != "yes" && "${JSON_VALIDATE}" != "no" ]]; then
  echo "Error: --json_validate must be yes or no (received: ${JSON_VALIDATE})" >&2
  exit 1
fi

if [[ "${GENERATION_MODE}" != "free" && "${GENERATION_MODE}" != "constrained" ]]; then
  echo "Error: --generation_mode must be free or constrained (received: ${GENERATION_MODE})" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}" "${DATA_DIR}" "${EXPORT_DIR}"

temp_tag="${TEMPERATURE/./p}"
csv_file="${BASE_DIR}/data_prep_comparison_temp_${temp_tag}_mode_${GENERATION_MODE}.csv"

echo "variant,prompt_style,synonym_aug,f1,validity,valid_json_count,total_examples,repaired_json_count,model_dir,metrics_file,predictions_file" > "${csv_file}"

echo "========================================================"
echo "Data Prep Comparison Configuration"
echo "========================================================"
echo "Epochs         : ${EPOCHS}"
echo "Learning rate  : ${LEARNING_RATE}"
echo "Inference mode : ${GENERATION_MODE}"
echo "Temperature    : ${TEMPERATURE}"
echo "JSON validate  : ${JSON_VALIDATE}"
echo "Output format  : ${OUTPUT_FORMAT}"
echo "Summary CSV    : ${csv_file}"
echo "========================================================"

# -----------------------------------------------------------------------------
# Run each data-prep variant end-to-end
# -----------------------------------------------------------------------------
for variant in with_defs no_defs syn_aug; do
  prompt_style="with_defs"
  synonym_aug="no"

  if [[ "${variant}" == "no_defs" ]]; then
    prompt_style="no_defs"
  elif [[ "${variant}" == "syn_aug" ]]; then
    synonym_aug="yes"
  fi

  train_file="${DATA_DIR}/${variant}_train.jsonl"
  val_file="${DATA_DIR}/${variant}_val.jsonl"
  test_file="${DATA_DIR}/${variant}_test.jsonl"

  model_dir="experiments/${variant}_qwen2_5_1_5B"
  pred_file="${EXPORT_DIR}/${variant}_val_predictions.jsonl"
  metrics_file="${BASE_DIR}/${variant}_val_metrics.json"

  echo ""
  echo "========================================================"
  echo "Variant: ${variant}"
  echo "Prompt style: ${prompt_style} | Synonym aug: ${synonym_aug}"
  echo "========================================================"

  # 1) Build variant dataset
  python -m src.build_dataset \
    --prompt_style "${prompt_style}" \
    --synonym_aug "${synonym_aug}" \
    --train_output "${train_file}" \
    --val_output "${val_file}" \
    --test_output "${test_file}"

  # 2) Train variant model
  python -m src.train \
    --model_name "${MODEL_NAME}" \
    --train_file "${train_file}" \
    --val_file "${val_file}" \
    --output_dir "${model_dir}" \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}"

  # 3) Validation inference
  python -m src.inference \
    --model_name "${MODEL_NAME}" \
    --model_path "${model_dir}" \
    --input_file "${val_file}" \
    --output_format "${OUTPUT_FORMAT}" \
    --generation_mode "${GENERATION_MODE}" \
    --json_validate "${JSON_VALIDATE}" \
    --temperature "${TEMPERATURE}" \
    --output_file "${pred_file}"

  # 4) Validation evaluation
  python -m src.evaluation \
    --input_file "${pred_file}" \
    --output_file "${metrics_file}"

  metrics_row=$(python - "${metrics_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as f:
    m = json.load(f)

fields = [
    f"{m.get('f1', 0.0):.6f}",
    f"{m.get('validity', 0.0):.6f}",
    str(m.get('valid_json_count', '')),
    str(m.get('total_examples', '')),
    str(m.get('repaired_json_count', '')),
]
print(','.join(fields))
PY
)

  echo "${variant},${prompt_style},${synonym_aug},${metrics_row},${model_dir},${metrics_file},${pred_file}" >> "${csv_file}"
done

echo ""
echo "Completed data prep comparison runs."
echo "Summary CSV: ${csv_file}"
