#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Format Comparison Runner
# -----------------------------------------------------------------------------
# Purpose:
#   Run inference + evaluation for output formats: json, xml, plain
#   and generate one consolidated CSV summary.
#
# Default comparison policy (recommended for fair ablation):
#   - temperature=0.0   -> deterministic decoding
#   - json_validate=yes -> normalize outputs before metrics
# -----------------------------------------------------------------------------

BASE_DIR="experiments/qwen2_5_1_5B_masked_tuned"
EXPORT_DIR="data/processed/exports"
INPUT_FILE="data/processed/val.jsonl"
TEMPERATURE="0.0"
JSON_VALIDATE="yes"
OUTPUT_PREFIX="fmt"

usage() {
  cat <<USAGE
Usage: bash scripts/run_format_comparison_and_csv.sh [options]

Options:
  --input_file PATH       Input split JSONL (default: data/processed/val.jsonl)
  --temperature VALUE     Decoding temperature (default: 0.0)
  --json_validate yes|no  Whether to normalize/validate output (default: yes)
  --output_prefix NAME    Output filename prefix (default: fmt)

Example:
  bash scripts/run_format_comparison_and_csv.sh --temperature 0.0 --json_validate yes
USAGE
}

# -----------------------------------------------------------------------------
# Parse CLI arguments
# -----------------------------------------------------------------------------
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
    --output_prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Validate arguments
# -----------------------------------------------------------------------------
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Error: input file not found -> ${INPUT_FILE}" >&2
  exit 1
fi

if [[ "${JSON_VALIDATE}" != "yes" && "${JSON_VALIDATE}" != "no" ]]; then
  echo "Error: --json_validate must be 'yes' or 'no' (received: ${JSON_VALIDATE})" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}" "${EXPORT_DIR}"

temp_tag="${TEMPERATURE/./p}"
csv_file="${BASE_DIR}/${OUTPUT_PREFIX}_format_comparison_temp_${temp_tag}_validate_${JSON_VALIDATE}.csv"

# -----------------------------------------------------------------------------
# Startup summary + preflight checks
# -----------------------------------------------------------------------------
echo "========================================================"
echo "Format Comparison Configuration"
echo "========================================================"
echo "Input file     : ${INPUT_FILE}"
echo "Temperature    : ${TEMPERATURE}"
echo "JSON validate  : ${JSON_VALIDATE}"
echo "Output prefix  : ${OUTPUT_PREFIX}"
echo "Export dir     : ${EXPORT_DIR}"
echo "Metrics dir    : ${BASE_DIR}"
echo "Summary CSV    : ${csv_file}"
echo "========================================================"

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available in PATH." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Initialize summary CSV
# -----------------------------------------------------------------------------
echo "format,run_name,json_validate,temperature,f1,validity,valid_json_count,total_examples,repaired_json_count,metrics_file,predictions_file" > "${csv_file}"

# -----------------------------------------------------------------------------
# Run format experiments
# -----------------------------------------------------------------------------
for format in json xml plain; do
  short="j"
  if [[ "${format}" == "xml" ]]; then
    short="x"
  elif [[ "${format}" == "plain" ]]; then
    short="p"
  fi

  run_name="${OUTPUT_PREFIX}_${short}_${JSON_VALIDATE}_${temp_tag}"
  predictions_file="${EXPORT_DIR}/${run_name}.jsonl"
  metrics_file="${BASE_DIR}/${run_name}_metrics.json"

  echo ""
  echo "========================================================"
  echo "Run: ${run_name}"
  echo "Format: ${format} | Temp: ${TEMPERATURE} | Validate: ${JSON_VALIDATE}"
  echo "Input: ${INPUT_FILE}"
  echo "========================================================"

  # 1) Inference
  python -m src.inference \
    --input_file "${INPUT_FILE}" \
    --output_format "${format}" \
    --json_validate "${JSON_VALIDATE}" \
    --temperature "${TEMPERATURE}" \
    --output_file "${predictions_file}"

  # 2) Evaluation
  python -m src.evaluation \
    --input_file "${predictions_file}" \
    --output_file "${metrics_file}"

  # 3) Extract core metrics for the consolidated CSV row
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

  echo "${format},${run_name},${JSON_VALIDATE},${TEMPERATURE},${metrics_row},${metrics_file},${predictions_file}" >> "${csv_file}"
done

echo ""
echo "Completed all format comparison runs."
echo "Summary CSV: ${csv_file}"
