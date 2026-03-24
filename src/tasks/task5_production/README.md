# Task 5: Memory Profiling & Quantization

This task profiles memory, quantizes with llama.cpp, benchmarks latency/F1, and measures concurrency.

## 1) Profile activation memory per layer

```bash
python -m src.tasks.task5_production.profile_memory \
  --model_name Qwen/Qwen2.5-1.5B \
  --input_file data/processed/task1_test.jsonl \
  --sample_count 50 \
  --output_file experiments/task5_production/memory_profile.csv
```

Outputs:
- `experiments/task5_production/memory_profile.csv`
- `experiments/task5_production/memory_profile.json`

## 2) Quantize with llama.cpp

Clone and build llama.cpp (once):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build -j
```

Convert HF model to GGUF:

```bash
python ./llama.cpp/convert_hf_to_gguf.py \
  Qwen/Qwen2.5-1.5B \
  --outtype f16 \
  --outfile models/gguf/qwen2.5-1.5b-f16.gguf
```

Quantize:

```bash
./llama.cpp/quantize models/gguf/qwen2.5-1.5b-f16.gguf models/gguf/qwen2.5-1.5b-Q4_K_M.gguf Q4_K_M
./llama.cpp/quantize models/gguf/qwen2.5-1.5b-f16.gguf models/gguf/qwen2.5-1.5b-Q5_K_M.gguf Q5_K_M
./llama.cpp/quantize models/gguf/qwen2.5-1.5b-f16.gguf models/gguf/qwen2.5-1.5b-Q8_0.gguf   Q8_0
```

## 3) Benchmark quantized models (latency + F1)

```bash
python -m src.tasks.task5_production.benchmark_llamacpp \
  --llama_bin ./llama.cpp/llama-cli \
  --model_path models/gguf/qwen2.5-1.5b-Q4_K_M.gguf \
  --input_file data/processed/task1_test.jsonl \
  --sample_count 200 \
  --label Q4_K_M

python -m src.tasks.task5_production.benchmark_llamacpp \
  --llama_bin ./llama.cpp/llama-cli \
  --model_path models/gguf/qwen2.5-1.5b-Q5_K_M.gguf \
  --input_file data/processed/task1_test.jsonl \
  --sample_count 200 \
  --label Q5_K_M

python -m src.tasks.task5_production.benchmark_llamacpp \
  --llama_bin ./llama.cpp/llama-cli \
  --model_path models/gguf/qwen2.5-1.5b-Q8_0.gguf \
  --input_file data/processed/task1_test.jsonl \
  --sample_count 200 \
  --label Q8_0
```

Results:
- `experiments/task5_production/quant_benchmark.csv`

## 4) Concurrency benchmark (p50/p95/p99 + throughput)

```bash
python -m src.tasks.task5_production.concurrency_benchmark \
  --llama_bin ./llama.cpp/llama-cli \
  --model_path models/gguf/qwen2.5-1.5b-Q4_K_M.gguf \
  --input_file data/processed/task1_test.jsonl \
  --sample_count 64 \
  --concurrency 1,4,8,16 \
  --label Q4_K_M
```

Results:
- `experiments/task5_production/concurrency.csv`

## 5) Production recommendation

Use the combined results to select:
- Quantization level (memory vs quality)
- Batch size / concurrency target
- SLA tradeoffs (latency vs throughput)
