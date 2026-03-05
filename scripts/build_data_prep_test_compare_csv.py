import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Input/Output configuration
# ---------------------------------------------------------------------
BASELINE_METRICS = Path("experiments/qwen2_5_1_5B_masked_tuned/final_test_no_0p2_metrics.json")
DATA_PREP_METRICS = Path("experiments/data_prep_comparison/dp_wdefs_test_metrics.json")
OUTPUT_CSV = Path("experiments/data_prep_comparison/data_prep_test_compare.csv")

ROWS = [
    {
        "run_name": "baseline_json_no_0p2_test",
        "source": "baseline",
        "metrics_file": BASELINE_METRICS,
        "model_dir": "experiments/qwen2_5_1_5B_masked_tuned",
        "predictions_file": "data/processed/exports/final_test_no_0p2.jsonl",
    },
    {
        "run_name": "data_prep_with_defs_test",
        "source": "data_prep",
        "metrics_file": DATA_PREP_METRICS,
        "model_dir": "experiments/with_defs_qwen2_5_1_5B",
        "predictions_file": "data/processed/exports/dp_wdefs_test_pred.jsonl",
    },
]

CSV_HEADER = [
    "run_name",
    "source",
    "precision",
    "recall",
    "f1",
    "validity",
    "valid_json_count",
    "total_examples",
    "repaired_json_count",
    "model_dir",
    "metrics_file",
    "predictions_file",
]


def load_metrics(path: Path) -> dict:
    """Load metrics JSON from disk."""
    with path.open("r") as f:
        return json.load(f)


def validate_inputs(rows: list[dict]) -> None:
    """Fail early if a required metrics file is missing."""
    missing = [str(row["metrics_file"]) for row in rows if not Path(row["metrics_file"]).exists()]
    if missing:
        joined = "\n - ".join(missing)
        raise FileNotFoundError(f"Missing metrics file(s):\n - {joined}")


def build_csv_row(row: dict, metrics: dict) -> list:
    """Assemble one comparison row in the target CSV format."""
    return [
        row["run_name"],
        row["source"],
        metrics.get("precision", ""),
        metrics.get("recall", ""),
        metrics.get("f1", ""),
        metrics.get("validity", ""),
        metrics.get("valid_json_count", ""),
        metrics.get("total_examples", ""),
        metrics.get("repaired_json_count", ""),
        row["model_dir"],
        str(row["metrics_file"]),
        row["predictions_file"],
    ]


def main() -> None:
    validate_inputs(ROWS)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for row in ROWS:
            metrics = load_metrics(Path(row["metrics_file"]))
            writer.writerow(build_csv_row(row, metrics))

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
