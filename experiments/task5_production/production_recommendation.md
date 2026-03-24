# Task 5 Production Recommendation

## Instruction 3 Completion (Latency + Memory + F1)
A combined benchmark with memory columns is available at:
- `experiments/task5_production/quant_benchmark_with_memory.csv`

Full-set (`sample_count=3453`) highlights:
- `Q4_K_M_LORA_full`: F1 `0.3573`, per-sample latency `1922.74 ms`, model size `955 MB`
- `Q5_K_M_LORA_full`: F1 `0.4119`, per-sample latency `2646.65 ms`, model size `1089 MB`
- `Q8_0_LORA_full`: F1 `0.4476`, per-sample latency `2108.27 ms`, model size `1586 MB`

## Memory Bottleneck
Evidence from:
- `experiments/task5_production/memory_profile.csv`
- `experiments/task5_production/quant_benchmark_with_memory.csv`

Findings:
- Mean activation memory per profiled layer is about `0.3304 MB`.
- Model weight memory is GB-scale (`955 MB` to `1586 MB` quantized, `~2944 MB` fp16 params in profiler output).
- Therefore, the dominant bottleneck is **model weights**, not per-layer activations.

## Concurrency Results (Q4/Q5/Q8)
Source:
- `experiments/task5_production/concurrency.csv`

Observed pattern:
- Throughput improves with higher concurrency.
- Tail latency (`p99`) grows at higher concurrency (notably at `16`).
- Best latency for all three quantizations is at concurrency `1`, with p95 around `578-584 ms`.

## Instruction 7 Completion (Production Config Recommendation)
Target SLA example: `<500 ms` latency.

Recommended production default (best quality under current tests):
- Quantization: `Q8_0`
- Batch size / sample strategy: keep single-request generation path (no batching in current script)
- Concurrency: `1` for stable latency, `4` when throughput is prioritized and slight tail increase is acceptable

Why this choice:
- Highest full-set F1 (`0.4476`) among tested quantizations.
- Validity is `1.0`.
- Similar single-request latency class as Q4/Q5 in concurrency run.

SLA note:
- Current measurements do **not** meet `<500 ms` p95 yet (best observed is around `~578 ms`).
- To move toward `<500 ms`, reduce output budget (`max_tokens`), shorten prompt template, and retest with concurrency `1`.
