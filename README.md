---
title: NER_to_JSON_Project2
app_file: app.py
sdk: gradio
sdk_version: 6.6.0
---

# NER JSON Studio: Fine-Tuned Qwen for Structured Entity Extraction

<p align="center">
  <img alt="Model" src="https://img.shields.io/badge/Model-Qwen2.5--1.5B-2563eb?style=for-the-badge">
  <img alt="Method" src="https://img.shields.io/badge/Tuning-LoRA-7c3aed?style=for-the-badge">
  <img alt="Best F1" src="https://img.shields.io/badge/Best%20F1-0.9066-059669?style=for-the-badge">
  <img alt="JSON Validity" src="https://img.shields.io/badge/JSON%20Validity-100%25-f59e0b?style=for-the-badge">
</p>

> Production-safe setup: `json_validate=yes`, `temperature=0.1`, constrained generation, with_defs adapter.

## 📌 Project Overview
This project builds a complete **NER to Structured JSON** pipeline using **Qwen2.5-1.5B + LoRA fine-tuning** on CoNLL-style data.

The system is designed to:
- extract entities from raw text,
- output clean schema-constrained JSON,
- compare decoding and data-prep strategies,
- support human correction + active learning with Gradio.

Target quality achieved:
- **F1 > 85%**
- **JSON validity = 100%** on key runs

---

## 🚀 Features
- **LoRA Fine-Tuning** of `Qwen/Qwen2.5-1.5B`
- **Structured JSON extraction** with schema normalization
- **Experiment sweeps**:
  - JSON validation yes/no
  - temperature (0.0 / 0.1 / 0.2)
  - output format (JSON / XML / plain)
  - generation mode (free / constrained)
  - data prep variants (with_defs / no_defs / syn_aug)
- **Gradio UI** for:
  - extraction,
  - correction,
  - active-learning export,
  - analytics dashboard
- **Active learning artifacts** persisted to JSONL
- **Plot dashboard** for reviewer presentation

---

## 📁 Project Structure
```text
NER_to_JSON_Project2/
│── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   ├── exports/
│   │   ├── corrections/
│   │   ├── active_learning/
│   │   ├── review_queue/
│   │   └── variants/
│
│── experiments/
│   ├── qwen2_5_1_5B_masked_tuned/
│   └── data_prep_comparison/
│
│── plots/
│   ├── index.html
│   └── *.svg
│
│── scripts/
│   ├── run_json_validity_f1_experiments.sh
│   ├── run_format_comparison_and_csv.sh
│   ├── run_generation_mode_comparison_and_csv.sh
│   ├── run_data_prep_comparison_and_csv.sh
│   ├── build_plotly_dashboard.py
│   ├── build_data_prep_test_compare_csv.py
│   ├── build_review_queue.py
│   ├── export_corrections_jsonl.py
│   └── launch_correction_app.sh
│
│── src/
│   ├── preprocess.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── model.py
│   ├── correction_schema.py
│   ├── correction_io.py
│   ├── correction_state.py
│   ├── active_learning.py
│   └── gradio_correction_app.py
│
└── README.md
```

---

## ✅ Core Tasks

### 🔹 Task 1: Train and run structured NER inference
- Fine-tune Qwen2.5-1.5B with LoRA
- Generate JSON predictions on val/test
- Evaluate precision, recall, F1, and validity

### 🔹 Task 2: Run controlled experiment matrix
- Validation and test comparisons for:
  - decoding settings,
  - format variants,
  - generation constraints,
  - data prep strategies

### 🔹 Task 3: Build correction + active learning loop
- Human-in-the-loop correction UI
- Save corrected examples to JSONL
- Persist cycle metadata for retraining decisions

### 🔹 Task 4: Presentation layer
- Interactive Plotly dashboard (`plots/index.html`)
- Gradio interface for extraction + correction + dashboard

---

## 📊 Key Results Snapshot

### JSON validation × temperature (validation)
Source: `experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv`
- Peak validation F1 around: **0.9064**
- Production-safe stable config: **yes + 0.1** (0 repaired)

### Final test configuration comparison
Source: `experiments/qwen2_5_1_5B_masked_tuned/final_test_comparison.csv`
- `final_test_no_0p2`: F1 **0.8837** (higher, but repaired > 0)
- `final_test_yes_0p1`: F1 **0.8825** (cleaner, repaired = 0)

### Data prep impact on test
Source: `experiments/data_prep_comparison/data_prep_test_compare.csv`
- baseline test F1: **0.8837**
- with_defs test F1: **0.9066**

### Generation mode comparison (validation)
Source: `experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv`
- constrained generation outperforms free generation

---

## 🏗️ Installation & Setup

### Prerequisites
- Python 3.10+
- macOS/Linux (MPS/CUDA/CPU supported via PyTorch setup)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Activate venv (recommended)
```bash
source venv/bin/activate
```

---

## 🏋️ Training & Inference

### Train
```bash
python -m src.train
```

### Inference (example)
```bash
python -m src.inference \
  --input_file data/processed/test.jsonl \
  --json_validate yes \
  --temperature 0.1 \
  --output_format json \
  --output_file data/processed/exports/final_test_yes_0p1.jsonl
```

### Evaluate
```bash
python -m src.evaluation \
  --input_file data/processed/exports/final_test_yes_0p1.jsonl \
  --output_file experiments/qwen2_5_1_5B_masked_tuned/final_test_yes_0p1_metrics.json
```

---

## 🧪 Experiment Commands

### 1. Parameters Experiment (JSON Validate + Temperature)
Purpose: compare `json_validate` (`yes/no`) across temperatures `0.0`, `0.1`, `0.2`.

```bash
bash scripts/run_json_validity_f1_experiments.sh
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv`

### 2. Architecture Experiment: Output Format (JSON vs XML vs plain)
Purpose: compare output format behavior under one fixed decoding setup.
Recommended fixed setup: `--temperature 0.0 --json_validate yes`

```bash
bash scripts/run_format_comparison_and_csv.sh \
  --temperature 0.0 \
  --json_validate yes
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/fmt_format_comparison_temp_0p0_validate_yes.csv`

### 3. Architecture Experiment: Generation Mode (Constrained vs Free)
Purpose: compare decoding strategy while keeping format/config fixed.
Recommended fixed setup: `--temperature 0.0 --json_validate yes --output_format json`

```bash
bash scripts/run_generation_mode_comparison_and_csv.sh \
  --temperature 0.0 \
  --json_validate yes \
  --output_format json
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv`

### 4. Data Prep Experiment (with_defs vs no_defs vs syn_aug)
Purpose: compare prompt/data variants after short retraining + evaluation.

```bash
bash scripts/run_data_prep_comparison_and_csv.sh \
  --epochs 2 \
  --generation_mode constrained \
  --temperature 0.0 \
  --json_validate yes
```

Primary outputs:
- `experiments/data_prep_comparison/data_prep_comparison_temp_0p0_mode_constrained.csv`
- `experiments/data_prep_comparison/data_prep_test_compare.csv`

### Suggested Order (for reproducible reporting)
1. `run_json_validity_f1_experiments.sh`
2. `run_format_comparison_and_csv.sh`
3. `run_generation_mode_comparison_and_csv.sh`
4. `run_data_prep_comparison_and_csv.sh`

---

## 🖥️ Gradio UI (Inference + Correction + Active Learning)
Main UI file: `src/gradio_correction_app.py`

### Includes
- **Extract tab**: text -> structured JSON + highlighted entities
- **Correct & Save tab**: editable JSON + approval/save
- **Analytics Dashboard tab**: embedded interactive Plotly dashboard

### Output artifacts from UI
- Predictions: `data/processed/active_learning/predictions_export.jsonl`
- Corrections: `data/processed/corrections/corrections.jsonl`
- AL cycles: `data/processed/active_learning/cycle_records.jsonl`

### Launch
```bash
python -m src.gradio_correction_app
```

Public temporary link:
```bash
python -m src.gradio_correction_app --share
```

---

## 📈 Plot Dashboard

Generate/rebuild dashboard:
```bash
python scripts/build_plotly_dashboard.py
```

Open locally:
```bash
open plots/index.html
```

---

## 🎯 Recommended Production Config

Use this for stable deployment:
- model/adaptor: `experiments/with_defs_qwen2_5_1_5B`
- json_validate: `yes`
- temperature: `0.1`
- output format: `json`
- generation mode: `constrained`

---

## 📌 Future Improvements
- Add per-sample scoring in UI (when ground truth provided)
- Add model switcher in Gradio (baseline vs with_defs)
- Add review-queue prioritization by uncertainty
- Add hosted demo deployment pipeline

---

## ✨ Author
Devolped by Krishna Chaitanya.
