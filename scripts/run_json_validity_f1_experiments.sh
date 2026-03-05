#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# JSON-Validation vs Temperature Experiment Runner
# -----------------------------------------------------------------------------
# Runs the 2x3 matrix:
#   json_validate in {yes,no}
#   temperature in {0.0,0.1,0.2}
# and writes one consolidated CSV.
# -----------------------------------------------------------------------------

BASE_DIR="experiments/qwen2_5_1_5B_masked_tuned"
EXPORT_DIR="data/processed/exports"
CSV_FILE="${BASE_DIR}/json_validity_f1_experiment_results.csv"

mkdir -p "${BASE_DIR}" "${EXPORT_DIR}"

echo "run_name,json_validate,temperature,do_sample,f1,validity,valid_json_count,total_examples,repaired_json_count,metrics_file,predictions_file" > "${CSV_FILE}"

echo "=================================================="
echo "JSON Validity / Temperature Experiment"
echo "=================================================="
echo "Metrics dir : ${BASE_DIR}"
echo "Export dir  : ${EXPORT_DIR}"
echo "Output CSV  : ${CSV_FILE}"
echo "=================================================="

for json_validate in yes no; do
  for temperature in 0.0 0.1 0.2; do
    temp_tag="${temperature/./p}"
    run_name="json_${json_validate}_temp_${temp_tag}"

    predictions_file="${EXPORT_DIR}/${run_name}_predictions.jsonl"
    metrics_file="${BASE_DIR}/${run_name}_metrics.json"

    echo "=================================================="
    echo "Running: ${run_name}"
    echo "=================================================="

    python -m src.inference \
      --temperature "${temperature}" \
      --json_validate "${json_validate}" \
      --output_file "${predictions_file}"

    python -m src.evaluation \
      --input_file "${predictions_file}" \
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
    str(m.get("valid_json_count", "")),
    str(m.get("total_examples", "")),
    str(m.get("repaired_json_count", "")),
]
print(",".join(fields))
PY
)

    do_sample="false"
    if [[ "${temperature}" != "0.0" ]]; then
      do_sample="true"
    fi

    echo "${run_name},${json_validate},${temperature},${do_sample},${metrics_row},${metrics_file},${predictions_file}" >> "${CSV_FILE}"
  done
done

echo ""
echo "Completed all runs."
echo "Summary CSV: ${CSV_FILE}"
