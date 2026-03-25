
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

## 🏗️ Installation & Setup

### Prerequisites
- Python `3.10-3.12` (recommended: `3.11`)
- macOS/Linux (MPS/CUDA/CPU supported via PyTorch setup)
- Git LFS (required to pull tracked model artifacts under `models/`)

### One-time Git LFS setup (per machine)
```bash
git lfs install
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Pull LFS-tracked model files (after clone)
```bash
git lfs pull
```

### Activate venv (recommended)
```bash
source venv/bin/activate
```

---

## ⚡ Run On Another Computer (Quick Start)

Use these exact commands on a fresh machine.

### 1. Clone and enter project
```bash
git clone https://github.com/kchivukulagrid/NER_to_JSON_Project2.git
cd NER_to_JSON_Project2
```

### 2. Create environment and install dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git lfs install
git lfs pull
```

If `outlines_core` fails to build (usually on Python `3.13+`/`3.14`), use Python `3.11` for this project.  
Alternative (if you must keep Python `3.14`): install Rust first, then reinstall deps.

### 3. Run UI directly (no retraining required)
```bash
python -m src.gradio_correction_app
```

### 4. Rebuild and open analytics dashboard
```bash
python scripts/build_plotly_dashboard.py
open plots/index.html
```

### Notes for new users
- If adapter files are present in `experiments/with_defs_qwen2_5_1_5B`, UI runs with your tuned model.
- If adapter is missing, update `--adapter_path` when launching UI or keep your local adapter in the same path.
- UI runtime exports are written to:
  - `data/processed/active_learning/predictions_export.jsonl`
  - `data/processed/corrections/corrections.jsonl`
  - `data/processed/active_learning/cycle_records.jsonl`

---


## 📁 Project Structure
```text
NER_to_JSON_Project2/
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── task1_train.jsonl
│   │   ├── task1_val.jsonl
│   │   ├── task1_test.jsonl
│   │   ├── exports/
│   │   ├── corrections/
│   │   ├── active_learning/
│   │   ├── adversarial/
│   │   └── variants/
│   └── steering/
├── experiments/
│   ├── qwen2_5_1_5B_masked_tuned/
│   ├── data_prep_comparison/
│   ├── with_defs_qwen2_5_1_5B/
│   ├── no_defs_qwen2_5_1_5B/
│   ├── syn_aug_qwen2_5_1_5B/
│   ├── task1_constrained/
│   ├── task2_layer_importance/
│   ├── task3_steering/
│   ├── task4_adversarial/
│   └── task5_production/
├── models/
│   ├── gguf/
│   └── hf/
├── llama.cpp/
├── plots/
│   └── index.html
├── scripts/
│   ├── run_json_validity_f1_experiments.sh
│   ├── run_format_comparison_and_csv.sh
│   ├── run_generation_mode_comparison_and_csv.sh
│   ├── run_data_prep_comparison_and_csv.sh
│   ├── task1_run_benchmarks.sh
│   ├── build_plotly_dashboard.py
│   ├── build_data_prep_test_compare_csv.py
│   ├── build_review_queue.py
│   ├── export_corrections_jsonl.py
│   └── launch_correction_app.sh
├── src/
│   ├── core/
│   ├── tasks/
│   │   ├── task1_constrained/
│   │   ├── task2_layer_importance/
│   │   ├── task3_steering/
│   │   ├── task4_adversarial/
│   │   └── task5_production/
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── gradio_correction_app.py
├── requirements.txt
└── README.md
```

---

## ✅ Project_2 Tasks

### 🔹 Task 1: Constrained Decoding for Structured NER
- Train LoRA for strict JSON schema output with offsets/confidence.
- Compare free vs constrained decoding quality and latency.

### 🔹 Task 2: Layer-wise Entity Type Importance
- Run logit-lens + per-layer ablation.
- Identify critical layers and evaluate selective LoRA placement.

### 🔹 Task 3: Boundary Steering (Gemma 2B)
- Extract activations on strict vs loose boundary sets (layers 12-16).
- Compute/apply steering vectors and evaluate boundary F1 tradeoffs.

### 🔹 Task 4: Adversarial Robustness
- Create adversarial categories (nested, abbrev, misspell, ambiguous, multilingual).
- Train with mixed original+adversarial data and compare pre/post robustness.

### 🔹 Task 5: Production Profiling & Quantization
- Profile inference memory.
- Benchmark Q4/Q5/Q8 quantizations for latency, memory, and F1.
- Measure concurrency (1/4/8/16) with p50/p95/p99 and throughput.

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

## 🧪 Project_3 Tasks Final Observations

### Task 1 - Constrained Decoding
- LoRA training for JSON schema with offsets/confidence was completed end-to-end.
- Benchmark (`experiments/task1_constrained/task1_benchmark.csv`, 500 samples):
  - Free: precision `0.514586`, recall `0.448627`, F1 `0.479348`, validity `1.0`, elapsed `709.354s`
  - Constrained: precision `0.514586`, recall `0.448627`, F1 `0.479348`, validity `1.0`, elapsed `1171.269s`
- Observation: no F1 gain at deterministic decoding, with about `65.1%` latency overhead for constrained mode.

### Task 2 - Layer-wise Importance
- Layer emergence and ablation were completed, with outputs in `experiments/task2_layer_importance/`.
- Selective LoRA vs full LoRA (`results_selective.csv`):
  - Selective F1 `0.3598`
  - Full F1 `0.3870`
- Observation: entity types show different layer sensitivity; selective placement is lighter but slightly lower F1.

### Task 3 - Boundary Steering
- Model used for this task: `google/gemma-2-2b` (2B), intentionally different from the Qwen base used in Tasks 1/2/4/5.
- Activation extraction, steering vector computation, intervention, and evaluation were completed.
- Best tested setup (`experiments/task3_steering/results.csv`):
  - Layer `13`, scale `1.5`, boundary F1 `0.0041322314`
- Observation: boundary control effect is modest at scales `0.5-1.5`; stronger tradeoff probing can use scales `2.0-3.0`.

### Task 4 - Adversarial Robustness
- Adversarial categories: nested, abbrev, misspell, ambiguous, multilingual.
- Pre/post adversarial training comparison completed (`experiments/task4_adversarial/robustness_gains.csv`).
- F1 gains:
  - multilingual `+0.5495`
  - ambiguous `+0.2813`
  - nested `+0.2444`
  - misspell `+0.1026`
  - abbrev `+0.0774`
  - adversarial_all `+0.2486`
- Observation: robustness improved strongly across all adversarial categories while maintaining JSON validity.

---

## 🧠 Project3 Task 5 Final Observations

Source files:
- `experiments/task5_production/memory_profile.csv`
- `experiments/task5_production/memory_profile.json`
- `experiments/task5_production/quant_benchmark.csv`
- `experiments/task5_production/quant_benchmark_with_memory.csv`
- `experiments/task5_production/concurrency.csv`
- `experiments/task5_production/production_recommendation.md`

### Quantization benchmark (full CoNLL2003 test set, 3453 samples)
- `Q4_K_M_LORA_full`: per-sample `1922.74 ms`, precision `0.3665`, recall `0.3484`, F1 `0.3573`, validity `1.0`
- `Q5_K_M_LORA_full`: per-sample `2646.65 ms`, precision `0.4201`, recall `0.4040`, F1 `0.4119`, validity `1.0`
- `Q8_0_LORA_full`: per-sample `2108.27 ms`, precision `0.4565`, recall `0.4391`, F1 `0.4476`, validity `1.0`

### Memory profiling summary
- Mean activation memory per profiled layer: about `0.3304 MB`
- Model parameter memory: `3,087,428,608 bytes` (`~2944.4 MB`)
- Conclusion: memory bottleneck is **model weights**, not activation memory.

### Concurrency benchmark summary
- `c=1`: p95 `579.9 ms`, p99 `632.7 ms`, throughput `1.76 req/s`
- `c=4`: p95 `613.2 ms`, p99 `619.7 ms`, throughput `6.88 req/s`
- `c=8`: p95 `642.1 ms`, p99 `645.0 ms`, throughput `12.62 req/s`
- `c=16`: p95 `752.5 ms`, p99 `1198.0 ms`, throughput `21.65 req/s`
- Tuned SLA run (`Q4_K_M_server_tuned`, 64 samples):
  - `c=1`: p95 `283.7 ms`, throughput `3.96 req/s`
  - `c=2`: p95 `427.0 ms`, throughput `5.65 req/s`

### Task 5 takeaway
- Best quality among tested quantizations: `Q8_0` (highest F1).
- Higher concurrency improves throughput but increases tail latency.
- Instruction 7 production recommendation (quantization + batch size + concurrency):
  - quality-first profile: quantization `Q8_0`, batch size `1`, concurrency `1`
  - SLA-first profile (`<500 ms` p95): quantization `Q4_K_M`, batch size `1`, concurrency `2`
- SLA status: `<500 ms` p95 target is **met** in tuned Q4 server runs.
- Short note on how target was reached: switched to tuned server-style inference, used smaller output budget (`max_tokens=64`), kept batch size `1`, and selected `Q4_K_M` with low concurrency (`1-2`) to control tail latency.

---

## 🧰 Project3 Task 5 Reproducibility Notes (llama.cpp + LFS)

### Why `llama.cpp`
`llama.cpp` is used in Task 5 to convert/quantize GGUF models and run low-level quantized inference benchmarks.

### Install/build `llama.cpp` (for new users)
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DGGML_METAL=ON
cmake --build build -j
```

### Git LFS setup (for model artifacts)
```bash
git lfs install
git lfs track "models/**/*.gguf"
git lfs track "models/**/*.safetensors"
git lfs track "models/**/*.bin"
git add .gitattributes
```

### Pull LFS files after clone
```bash
git lfs pull
```

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
