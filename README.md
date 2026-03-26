
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

## 🧪 Project 3 — Detailed Task Reports

### Task 1: Constrained Decoding for Structured NER

Explanation: This task checks whether schema-aware decoding improves structured NER outputs enough to justify the added inference cost.

Objective: train a LoRA adapter that emits strict JSON with character offsets and confidence, then compare constrained vs free decoding on quality, validity, and latency.

Implementation:
- Dataset preparation, LoRA training, constrained decoding, inference, and benchmarking are implemented in `src/tasks/task1_constrained/`.
- The constrained path uses Outlines schema enforcement on the same base generation configuration used for free decoding.

Results:
- Source: `experiments/task1_constrained/task1_benchmark.csv`, `experiments/task1_constrained/results.csv`

| Mode | Samples | Precision | Recall | F1 | Validity | Elapsed |
|------|---------|-----------|--------|----|----------|---------|
| Free | 500 | 0.5146 | 0.4486 | 0.4793 | 1.0 | 709.4 s |
| Constrained | 500 | 0.5146 | 0.4486 | 0.4793 | 1.0 | 1171.3 s |
| Free | 200 | 0.5770 | 0.5061 | 0.5393 | 1.0 | 288.1 s |
| Constrained | 200 | 0.5770 | 0.5061 | 0.5393 | 1.0 | 382.7 s |

Interpretation:
- At deterministic decoding, constrained and free runs produce the same predictions, so precision, recall, and F1 are unchanged.
- JSON validity is already `100%`, which shows the adapter learned the output format reliably.
- Constraint enforcement still adds about `65%` latency overhead, so the main value here is format safety rather than quality gain.

---

### Task 2: Layer-wise Entity Type Importance

Explanation: This task analyzes which transformer layers matter most for different entity types and whether those insights can reduce LoRA training cost.

Objective: identify which transformer layers matter most for each entity type and test whether selective LoRA can preserve most of the full-model performance.

Implementation:
- Logit-lens, early-vs-late analysis, per-layer ablation, critical-layer extraction, and selective LoRA training are implemented in `src/tasks/task2_layer_importance/`.
- Outputs are stored in `experiments/task2_layer_importance/`.

Critical layers by entity type:
- Source: `experiments/task2_layer_importance/critical_layers.json`

| Entity Type | Critical Layers (top-5 by ablation F1 drop) |
|-------------|----------------------------------------------|
| ALL (macro) | 11, 12, 20, 6, 22 |
| PER | 11, 12, 16, 17, 9 |
| ORG | 12, 3, 4, 6, 11 |
| LOC | 11, 8, 16, 18, 20 |
| MISC | 20, 1, 25, 4, 6 |

**Early vs Late Layer Emergence:**

Source: `experiments/task2_layer_importance/results_early_late.csv`

| Entity Type | Stronger Half | Interpretation |
|-------------|--------------|----------------|
| ORG | Early (0–13) | Organization features develop in lower layers |
| MISC | Early (0–13) | Miscellaneous entity signals emerge early |
| LOC | Late (14–27) | Location recognition relies on higher-level representations |
| PER | Late (14–27) | Person name detection peaks in upper layers |

Selective vs full LoRA:
- Source: `experiments/task2_layer_importance/results_selective.csv`

| Adapter | Precision | Recall | F1 | Validity | Layers Trained |
|---------|-----------|--------|----|----------|----------------|
| Full (all 28 layers) | 0.3952 | 0.3791 | 0.3870 | 1.0 | 28 |
| Selective (critical only) | 0.3690 | 0.3511 | 0.3598 | 1.0 | 14 |

Interpretation:
- Layers `11` and `12` are repeatedly important across entity types and dominate the macro-critical set.
- ORG and MISC signals emerge earlier, while PER and LOC depend more on later layers.
- Selective LoRA keeps about `93%` of the full-adapter F1 while training only half the layers, which supports the efficiency claim even though full-layer training remains stronger.

---

### Task 3: Boundary Steering via Activation Intervention

Explanation: This task tests whether entity-boundary behavior can be adjusted at inference time by steering internal activations instead of retraining the model.

Objective: test whether activation steering can push generation toward stricter entity boundaries without retraining.

Model choice:
- This task uses `google/gemma-2-2b`, not the Qwen model from Tasks 1, 2, 4, and 5.
- The intent is to probe whether boundary-control signals exist in activation space even without task-specific fine-tuning.

Implementation:
- Boundary set preparation, activation extraction, steering vector computation, steered inference, and boundary evaluation are implemented in `src/tasks/task3_steering/`.
- Source: `experiments/task3_steering/results.csv`

| Layer | Scale | Precision | Recall | Boundary F1 |
|-------|-------|-----------|--------|-------------|
| 12 | 0.5 | 0.00351 | 0.00491 | 0.00409 |
| 12 | 1.0 | 0.00348 | 0.00491 | 0.00408 |
| 12 | 1.5 | 0.00348 | 0.00491 | 0.00408 |
| **13** | **0.5** | **0.00350** | **0.00491** | **0.00409** |
| **13** | **1.0** | **0.00355** | **0.00491** | **0.00412** |
| **13** | **1.5** | **0.00357** | **0.00491** | **0.00413** |
| 14 | 0.5 | 0.00350 | 0.00491 | 0.00409 |
| 14 | 1.0 | 0.00350 | 0.00491 | 0.00409 |
| 14 | 1.5 | 0.00350 | 0.00491 | 0.00409 |
| 15 | 0.5 | 0.00347 | 0.00491 | 0.00407 |
| 15 | 1.0 | 0.00351 | 0.00491 | 0.00410 |
| 15 | 1.5 | 0.00348 | 0.00491 | 0.00407 |
| 16 | 0.5 | 0.00347 | 0.00491 | 0.00407 |
| 16 | 1.0 | 0.00348 | 0.00491 | 0.00407 |
| 16 | 1.5 | 0.00348 | 0.00491 | 0.00408 |

Interpretation:
- Best configuration: layer `13`, scale `1.5`, boundary F1 `0.00413`.
- Recall is effectively constant across runs, so steering mostly affects precision rather than detection coverage.
- The effect exists but is weak, mainly because the base Gemma model is not tuned for this NER JSON task.
- This completes the task instruction-wise, but the tradeoff evidence remains partial within the tested `0.5–1.5` range.

---

### Task 4: Adversarial Robustness

Explanation: This task evaluates how the extractor behaves under noisy or misleading inputs and whether adversarial fine-tuning improves reliability.

Objective: evaluate how the model behaves under adversarial entity perturbations and measure whether mixed adversarial training improves robustness.

Implementation:
- Adversarial dataset generation, training, evaluation, and result comparison are implemented in `src/tasks/task4_adversarial/`.
- Categories include nested entities, abbreviations, misspellings, ambiguous boundaries, and multilingual noise.
- Sources: `experiments/task4_adversarial/results_pre.csv`, `results_post.csv`, `robustness_gains.csv`

| Category | Pre F1 | Post F1 | F1 Gain | Pre Boundary F1 | Post Boundary F1 | Boundary F1 Gain |
|----------|--------|---------|---------|-----------------|-------------------|-----------------|
| Nested | 0.0889 | 0.3333 | **+0.2444** | 0.0889 | 0.3969 | +0.3079 |
| Abbrev | 0.2462 | 0.3235 | **+0.0774** | 0.2769 | 0.4412 | +0.1643 |
| Misspell | 0.3333 | 0.4359 | **+0.1026** | 0.3333 | 0.5897 | +0.2564 |
| Ambiguous | 0.1235 | 0.4048 | **+0.2813** | 0.1235 | 0.4048 | +0.2813 |
| Multilingual | 0.0992 | 0.6486 | **+0.5495** | 0.1322 | 0.6757 | +0.5434 |
| Original | 0.2916 | 0.5052 | +0.2136 | 0.3274 | 0.5103 | +0.1829 |
| All | 0.2267 | 0.4597 | +0.2329 | 0.2506 | 0.4988 | +0.2482 |
| Adversarial (all) | 0.1700 | 0.4186 | **+0.2486** | 0.1834 | 0.4884 | +0.3049 |

Interpretation:
- Multilingual, ambiguous, and nested examples improved the most after adversarial training.
- Overall adversarial F1 improved by about `+0.2486`, while JSON validity remained `100%`.
- Boundary F1 is consistently higher than full entity F1, which indicates boundary recovery is easier than full type+value+offset matching.

---

### Task 5: Production Profiling & Quantization

Explanation: This task converts the model from an experimental system into a deployment-oriented service by measuring memory, quality, latency, and concurrency tradeoffs.

Objective: quantify the tradeoff between model size, latency, quality, and concurrency, then produce a deployable configuration that satisfies a strict latency target.

Implementation:
- Memory profiling, quantized benchmarking, and concurrency testing are implemented in `src/tasks/task5_production/`.
- The evaluation combines HuggingFace profiling with `llama.cpp` GGUF inference.

Memory profile:
- Source: `experiments/task5_production/memory_profile.csv`, `experiments/task5_production/memory_profile.json`

| Component | Memory |
|-----------|--------|
| Model parameters (FP16) | **2944.4 MB** (3,087,428,608 bytes) |
| Activation memory per layer (mean) | **0.33 MB** (346,460 bytes) |
| Activation total (27 layers) | **~8.9 MB** |

Conclusion: the dominant bottleneck is model weights, not activation memory.

Quantization benchmark:
- Source: `experiments/task5_production/quant_benchmark_with_memory.csv`

| Quantization | File Size | Per-sample Latency | Precision | Recall | F1 | Validity |
|-------------|-----------|-------------------|-----------|--------|----|----------|
| Q4\_K\_M | 940 MB | 1922.7 ms | 0.3665 | 0.3484 | 0.3573 | 1.0 |
| Q5\_K\_M | 1073 MB | 2646.7 ms | 0.4201 | 0.4040 | 0.4119 | 1.0 |
| Q8\_0 | 1570 MB | 2108.3 ms | 0.4565 | 0.4391 | 0.4476 | 1.0 |

Quality-size tradeoff: Q8\_0 gives the best F1, Q4\_K\_M gives the smallest footprint, and Q5\_K\_M sits in the middle but is the slowest in this setup.

Concurrency scaling:
- Source: `experiments/task5_production/concurrency.csv`

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|-------------|----------|----------|----------|--------------------|
| 1 | 566.8 | 579.9 | 632.7 | 1.76 |
| 4 | 575.0 | 613.2 | 619.7 | 6.88 |
| 8 | 633.1 | 642.1 | 645.0 | 12.62 |
| 16 | 708.8 | 752.5 | 1198.0 | 21.65 |

Observation: throughput improves with concurrency, but tail latency degrades sharply at higher concurrency, especially at `16`.

SLA-tuned server runs:

| Config | Concurrency | p50 (ms) | p95 (ms) | Throughput (req/s) |
|--------|-------------|----------|----------|--------------------|
| Q4\_K\_M tuned | 1 | 268.0 | **283.7** | 3.96 |
| Q4\_K\_M tuned | 2 | 372.6 | **427.0** | 5.65 |
| Q5\_K\_M tuned | 1 | 348.6 | 363.7 | 3.05 |
| Q5\_K\_M tuned | 2 | 525.8 | 583.9 | 4.01 |
| Q8\_0 server | 1 | 634.3 | 662.4 | 1.83 |

Production recommendations:
- Source: `experiments/task5_production/production_recommendation.md`

| Profile | Quantization | Batch Size | Concurrency | p95 Latency | Throughput | F1 |
|---------|-------------|------------|-------------|-------------|------------|----|
| **Quality-first** | Q8\_0 | 1 | 1 | 662.4 ms | 1.83 req/s | 0.4476 |
| **SLA-first** (< 500ms p95) | Q4\_K\_M | 1 | 2 | **427.0 ms** | 5.65 req/s | 0.3573 |

Interpretation:
- The `<500 ms` p95 SLA is met with the tuned `Q4_K_M` server configuration.
- `Q8_0` is the best quality option but does not satisfy the same SLA in the tested setup.
- The most effective latency improvement came from reducing `max_tokens` and using low-concurrency server-style inference.

---

## 🧰 Project3 Task 5 Reproducibility Notes (llama.cpp + LFS)

### Why `llama.cpp`
`llama.cpp` is used in Task 5 to convert/quantize GGUF models and run low-level quantized inference benchmarks.

### Install/build `llama.cpp` (for new users)
If `cmake` is missing (`command not found: cmake`), install it first:
```bash
# macOS (Homebrew)
brew install cmake
cmake --version
```

For Linux, install `cmake` from your package manager before building.

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
