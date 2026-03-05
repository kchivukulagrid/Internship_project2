#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Generation Mode Comparison Runner
# -----------------------------------------------------------------------------
# Purpose:
#   Compare free vs constrained generation with the same model/split/settings.
#   For each mode, this script:
#     1) runs inference
#     2) runs evaluation
#     3) appends results to a single comparison CSV
# -----------------------------------------------------------------------------

# ----------------------------
# Default configuration
# ----------------------------
BASE_DIR="experiments/qwen2_5_1_5B_masked_tuned"
EXPORT_DIR="data/processed/exports"
INPUT_FILE="data/processed/val.jsonl"
TEMPERATURE="0.0"
JSON_VALIDATE="yes"
OUTPUT_FORMAT="json"
OUTPUT_PREFIX="gen_mode"

usage() {
  cat <<USAGE
Usage: bash scripts/run_generation_mode_comparison_and_csv.sh [options]

Options:
  --input_file PATH       Input split JSONL (default: data/processed/val.jsonl)
  --temperature VALUE     Decoding temperature (default: 0.0)
  --json_validate yes|no  Whether to normalize/validate output (default: yes)
  --output_format FORMAT  json|xml|plain (default: json)
  --output_prefix NAME    Output filename prefix (default: gen_mode)

Example:
  bash scripts/run_generation_mode_comparison_and_csv.sh --temperature 0.0 --json_validate yes --output_format json
USAGE
}

# ----------------------------
# Argument parsing
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --json_validate)
      JSON_VALIDATE="$2"
      shift 2
      ;;
    --output_format)
      OUTPUT_FORMAT="$2"
      shift 2
      ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"
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

# ----------------------------
# Validation
# ----------------------------
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Error: input file not found -> ${INPUT_FILE}" >&2
  exit 1
fi

if [[ "${JSON_VALIDATE}" != "yes" && "${JSON_VALIDATE}" != "no" ]]; then
  echo "Error: --json_validate must be yes or no (received: ${JSON_VALIDATE})" >&2
  exit 1
fi

if [[ "${OUTPUT_FORMAT}" != "json" && "${OUTPUT_FORMAT}" != "xml" && "${OUTPUT_FORMAT}" != "plain" ]]; then
  echo "Error: --output_format must be one of json|xml|plain (received: ${OUTPUT_FORMAT})" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available in PATH." >&2
  exit 1
fi

# ----------------------------
# Setup paths
# ----------------------------
mkdir -p "${BASE_DIR}" "${EXPORT_DIR}"

temp_tag="${TEMPERATURE/./p}"
csv_file="${BASE_DIR}/${OUTPUT_PREFIX}_comparison_temp_${temp_tag}_validate_${JSON_VALIDATE}_format_${OUTPUT_FORMAT}.csv"

echo "========================================================"
echo "Generation Mode Comparison Configuration"
echo "========================================================"
echo "Input file     : ${INPUT_FILE}"
echo "Temperature    : ${TEMPERATURE}"
echo "JSON validate  : ${JSON_VALIDATE}"
echo "Output format  : ${OUTPUT_FORMAT}"
echo "Output prefix  : ${OUTPUT_PREFIX}"
echo "Summary CSV    : ${csv_file}"
echo "========================================================"

# ----------------------------
# Initialize CSV
# ----------------------------
echo "mode,run_name,json_validate,temperature,output_format,f1,validity,valid_json_count,total_examples,repaired_json_count,metrics_file,predictions_file" > "${csv_file}"

# ----------------------------
# Main loop: free vs constrained
# ----------------------------
for mode in free constrained; do
  run_name="${OUTPUT_PREFIX}_${mode}_${JSON_VALIDATE}_${temp_tag}_${OUTPUT_FORMAT}"
  predictions_file="${EXPORT_DIR}/${run_name}.jsonl"
  metrics_file="${BASE_DIR}/${run_name}_metrics.json"

  echo ""
  echo "========================================================"
  echo "Run: ${run_name}"
  echo "Mode: ${mode} | Temp: ${TEMPERATURE} | Validate: ${JSON_VALIDATE} | Format: ${OUTPUT_FORMAT}"
  echo "========================================================"

  # 1) Inference
  python -m src.inference \
    --input_file "${INPUT_FILE}" \
    --output_format "${OUTPUT_FORMAT}" \
    --generation_mode "${mode}" \
    --json_validate "${JSON_VALIDATE}" \
    --temperature "${TEMPERATURE}" \
    --output_file "${predictions_file}"

  # 2) Evaluation
  python -m src.evaluation \
    --input_file "${predictions_file}" \
    --output_file "${metrics_file}"

  # 3) Extract compact metric row
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
print(",".join(fields))
PY
)

  echo "${mode},${run_name},${JSON_VALIDATE},${TEMPERATURE},${OUTPUT_FORMAT},${metrics_row},${metrics_file},${predictions_file}" >> "${csv_file}"
done

echo ""
echo "Completed generation-mode comparison runs."
echo "Summary CSV: ${csv_file}"
