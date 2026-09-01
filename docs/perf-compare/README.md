# llama.cpp vs Juno — local compare

Baselines on **medion-Precision-T3610** · **Intel Xeon E5-1650 v2** (12 threads) · **62.7 GiB RAM** · **NVIDIA GeForce GTX 1080 (8 GiB)**.

Workload for both backends: `n_prompt=128`, `n_gen=64`, `reps=1`, temperature 0, Juno `--vector 0` (scalar).

Juno metrics use **JFR by default** (`--jfr 30m`): `TokenProduced.tps` for decode tg; pp from `ForwardPass.prefill.total_ms` when present, else `(API latency − decode total_ms)`.

| Run | Backend | Juno metrics | Artifacts |
|-----|---------|--------------|-----------|
| [`20260831T230258Z`](20260831T230258Z/) | CPU (`-ngl 0` / `--cpu`) | JFR pp/tg | [INDEX](20260831T230258Z/INDEX.md) |
| [`20260831T231403Z`](20260831T231403Z/) | GPU (`-ngl 99` / `--gpu`) | JFR pp/tg | [INDEX](20260831T231403Z/INDEX.md) |
| [`20260901T032753Z`](20260901T032753Z/) | GPU + Tier 5 (`JUNO_GPU_LAYERS=auto`) | JFR pp/tg | [INDEX](20260901T032753Z/INDEX.md) |

Earlier runs (API wall-clock tg only, no JFR): [`20260831T214609Z`](20260831T214609Z/) (CPU), [`20260831T223850Z`](20260831T223850Z/) (GPU).

## CPU summary (JFR) — `20260831T230258Z`

| Model | llama.cpp pp | llama.cpp tg | Juno pp | Juno tg | Juno/llama tg |
|-------|-------------:|-------------:|--------:|--------:|--------------:|
| tinyllama-1.1b Q4_K_M | 32.6 | 0.61* | 5.2 | 3.12 | 5.14* |
| qwen2.5-3b Q4_K_M | 23.6 | 3.80 | 1.78 | 1.01 | 0.27 |
| Phi-3.5-mini Q4_K_M | 17.5 | 3.54 | 0.86 | 0.84 | 0.24 |
| mistral-7b Q4_K_M | 10.7 | 2.18 | 0.81 | 0.48 | 0.22 |
| Qwen3.5-0.8B Q4_K_M | 68.1 | 8.01 | — | — | Juno load failed |

\* TinyLlama llama.cpp tg (0.61 t/s) looks like a single-rep outlier — prior CPU baseline was ~6.8 t/s on the same host. Juno JFR tg (3.12 t/s) is in line with expectations.

## GPU summary (JFR) — `20260831T231403Z` · GTX 1080

| Model | llama.cpp pp | llama.cpp tg | Juno pp | Juno tg | Juno/llama tg |
|-------|-------------:|-------------:|--------:|--------:|--------------:|
| tinyllama-1.1b Q4_K_M | 3583 | 186 | 19.5 | 31.4 | 0.17 |
| qwen2.5-3b Q4_K_M | 1356 | 68.0 | 8.1 | 13.3 | 0.19 |
| Phi-3.5-mini Q4_K_M | 1096 | 57.8 | 11.7 | 12.6 | 0.22 |
| mistral-7b Q4_K_M | 610 | 35.2 | 0.82 | 0.48 | 0.01 |

JFR tg is **~1.6–1.7×** API wall-clock tg on GPU for models that fit in VRAM. Mistral-7B Juno GPU still matches CPU (~0.48 t/s JFR), indicating VRAM/residency fallback on 8 GiB.

## Tier 5 GPU offload (`JUNO_GPU_LAYERS=auto`) — `20260901T032753Z` · mistral-7b only

| Model | llama.cpp tg | Juno tg (JFR) | Juno/llama tg | Notes |
|-------|-------------:|--------------:|--------------:|-------|
| mistral-7b Q4_K_M | 35.5 | 0.94 | **0.026** | Hybrid MatVec: ~10.4k GPU fp16 + ~6.6k CPU quant ops |

Prior GPU baseline (`20260831T231403Z`): mistral Juno tg **0.48** t/s (**0.01×**). Tier 5 auto offload is **~2×** faster but still below the P0 gate (**≥0.15×** ≈ 5.3 t/s).

## Re-run

```bash
# CPU (5 models incl. Qwen3.5 — Juno expected to fail on Qwen3.5)
./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 0 --reps 1 \
  --models tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,Qwen3.5-0.8B.Q4_K_M.gguf,qwen2.5-3b-instruct-q4_k_m.gguf,Phi-3.5-mini-instruct-Q4_K_M.gguf,mistral-7b-instruct-v0.1-q4_k_m.gguf

# GPU (default 4-model set)
./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0 --reps 1

# Tier 5 mistral bake-off (partial GPU residency)
JUNO_GPU_LAYERS=auto ./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0 --reps 1 \
  --models mistral-7b-instruct-v0.1-q4_k_m.gguf
```

Use `--no-jfr` to revert to API latency tg only. Per-model artifacts: `*-llama-cpp.json`, `*-juno.json`, `*-juno-jfr.json`, `*-compare.json`.

Build CUDA llama-bench once:

```bash
cmake -S ../llama.cpp -B ../llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build-cuda --target llama-bench -j"$(nproc)"
```
