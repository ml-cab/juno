# llama.cpp vs Juno — local compare

Baselines on **medion-Precision-T3610** · **Intel Xeon E5-1650 v2** (12 threads) · **62.7 GiB RAM** · **NVIDIA GeForce GTX 1080 (8 GiB)**.

Workload for both backends: `n_prompt=128`, `n_gen=64`, `reps=1`, temperature 0, Juno `--vector 0` (scalar).

Juno metrics use **JFR by default** (`--jfr 30m`): `TokenProduced.tps` for decode tg; pp from `ForwardPass.prefill.total_ms` when present, else `(API latency − decode total_ms)`.

| Run | Backend | Juno metrics | Artifacts |
|-----|---------|--------------|-----------|
| [`20260831T230258Z`](20260831T230258Z/) | CPU (`-ngl 0` / `--cpu`) | JFR pp/tg | [INDEX](20260831T230258Z/INDEX.md) |
| [`20260831T231403Z`](20260831T231403Z/) | GPU (`-ngl 99` / `--gpu`) | JFR pp/tg | [INDEX](20260831T231403Z/INDEX.md) |
| [`20260901T032753Z`](20260901T032753Z/) | GPU + Tier 5 (`JUNO_GPU_LAYERS=auto`) | JFR pp/tg | [INDEX](20260901T032753Z/INDEX.md) |
| [`20260901T154735Z-parallel`](20260901T154735Z-parallel/) | GPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T154735Z-parallel/INDEX.md) |
| [`20260901T155136Z-parallel`](20260901T155136Z-parallel/) | CPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T155136Z-parallel/INDEX.md) |
| [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) | GPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T173121Z-parallel/INDEX.md) |
| [`20260901T234024Z-prefill`](20260901T234024Z-prefill/) | CPU prefill microbatch (`--prefill-batch` 1 vs 32) | JFR pp | [INDEX](20260901T234024Z-prefill/INDEX.md) |
| [`20260902T200210Z-lora`](20260902T200210Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260902T200210Z-lora/INDEX.md) |
| [`20260905T031520Z-lora`](20260905T031520Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260905T031520Z-lora/INDEX.md) |
| [`20260904T141315Z-vision`](20260904T141315Z-vision/) | GPU vision chat (`compare-vision.sh`, `47-vision`) | latency / decode tps | [INDEX](20260904T141315Z-vision/INDEX.md) |
| [`20260904T194612Z`](20260904T194612Z/) | CPU Vector SIMD (`--vector 0`) | JFR pp/tg | [INDEX](20260904T194612Z/INDEX.md) |
| [`20260904T195731Z`](20260904T195731Z/) | CPU Vector SIMD (`--vector 1`) | JFR pp/tg | [INDEX](20260904T195731Z/INDEX.md) |
| [`20260910T025804Z`](20260910T025804Z/) | GPU + fused Q4_K MMQ (`JUNO_MMQ=on` / `-DJUNO_MMQ=on`) | JFR pp/tg | [INDEX](20260910T025804Z/INDEX.md) |
| [`20260910T030058Z-lora`](20260910T030058Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260910T030058Z-lora/INDEX.md) |

Earlier runs (API wall-clock tg only, no JFR): [`20260831T214609Z`](20260831T214609Z/) (CPU), [`20260831T223850Z`](20260831T223850Z/) (GPU).

## LoRA train-qa regression — `20260910T030058Z-lora`

Scenario: TinyLlama Q4_K_M · `/train-qa` *What is your name?* → *My name is Juno* · loss target 1.2 · playback temperature 0.

| ref | commit | train total ms | ms/pass | passes | playback tps | recall |
|-----|--------|---------------:|--------:|-------:|-------------:|:------:|
| release-0.1.2 | 51a3b90 | 45,000 | 3,000 | 15 | 38.3 | ✓ |
| HEAD | 8b78382 | 44,000 | 2,933 | 15 | 38.3 | ✓ |

**Current vs release-0.1.2:** train wall **0.98×**; playback tps **1.00×** (≥0.80 gate). Status **ok**. Run: `./scripts/performance-tests/compare-lora.sh --gpu --baseline release-0.1.2`.

Earlier ok snapshot: [`20260905T031520Z-lora`](20260905T031520Z-lora/). Earlier failing snapshot: [`20260902T200210Z-lora`](20260902T200210Z-lora/).

## GPU fused Q4_K MMQ bake-off — `20260910T025804Z`

`compare-llama-cpp.sh --gpu --vector 0` with `-DJUNO_MMQ=on`. JFR proves `cuda_resident_q4k` on all four models (including Phi-3 fused path).

| Model | llama tg | Juno tg (MMQ on) | Juno/llama | q4k MatVec count |
|-------|---------:|-----------------:|-----------:|-----------------:|
| tinyllama-1.1b Q4_K_M | 195.4 | 27.3 | 0.14 | 12462 |
| qwen2.5-3b Q4_K_M | 71.8 | 10.7 | 0.15 | 12744 |
| Phi-3.5-mini Q4_K_M | 60.6 | 7.39 | 0.12 | 6720 |
| mistral-7b Q4_K_M | 37.1 | 5.34 | 0.14 | 15936 |

**Gates:** P0 Phi-3.5 ≥ **0.5×** llama — **unmet** (0.12×). Original Tier 13B ≥ **1.3×** vs FP16-resident — **amended / deferred** (MMQ ships as VRAM-fit; this run is slower than prior FP16 GPU baseline Phi-3.5 tg **12.6**). Mistral packed-Q4 ≈ **0.14×** llama supports the fit claim (near P0 **0.15×**).

## Vision chat regression — `compare-vision.sh`

Scenario: `moondream2-q5_k.llamafile` (embedded vision, no mmproj) · `POST /v1/vision/chat` · *What is in this image?* · max_tokens 32 · temperature 0 · `./juno local --jfr` · **`--prefill single`** (default in the script).

Fixed prefill window: **~741 tokens** (729 image patches + ~11 text). Local mode only — cluster does not register vision routes.

Default `--prefill single` matches the known-good sequential Phi2 path on `47-vision`. Batched Q5_K prefill on current inference branches can finish after the hang fix but still yields wrong captions; use `--prefill batched` only when intentionally measuring that path.

| Check | Threshold |
|-------|-----------|
| Quality | HTTP 200, non-empty reply |
| Latency | `current.latency_ms / baseline ≤ 1.25` |
| Decode tps | `current.tps / baseline ≥ 0.80` (JFR `TokenProduced.tps` when present) |

```bash
./scripts/performance-tests/compare-vision.sh --gpu --baseline 47-vision
./scripts/performance-tests/compare-vision.sh --gpu --no-publish   # single ref only
./scripts/performance-tests/compare-vision.sh --gpu --prefill batched --no-publish  # batched path only
```

Test image: `scripts/performance-tests/fixtures/vision-bench.jpg`. Override with `--image` or `VISION_TEST_IMAGE`.

### Known-good snapshot — `20260904T141315Z-vision` (`47-vision`)

| Field | Value |
|-------|-------|
| Status | success |
| Prompt tokens | 741 |
| Latency | ~503 s |
| Decode tps (JFR) | ~1.41 |
| Reply | non-empty (color squares) |

## Vector SIMD CPU bake-off — `--vector 0` vs `--vector 1`

Paired CPU runs on the default model set (`n_prompt=128`, `n_gen=64`, `reps=1`, JFR). Policy: Q4_K/Q5_K weight-stationary accumulate stays scalar; `--vector` only toggles `--add-modules jdk.incubator.vector` (Q8_0 dequant when probe passes). See [`../performance.md`](../performance.md) and [`../infra-plan/PLAN-Infra-Vector-SIMD.md`](../infra-plan/PLAN-Infra-Vector-SIMD.md).

| Model | Juno tg `--vector 0` | Juno tg `--vector 1` | v1/v0 |
|-------|---------------------:|---------------------:|------:|
| tinyllama-1.1b Q4_K_M | 2.89 | 2.98 | 1.03 |
| qwen2.5-3b Q4_K_M | 0.957 | 0.965 | 1.01 |
| Phi-3.5-mini Q4_K_M | 0.818 | 0.812 | 0.99 |
| mistral-7b Q4_K_M | 0.453 | 0.463 | 1.02 |

**Verdict:** near-parity (±3%) as expected under the scalar accumulate policy. Artifacts: [`20260904T194612Z`](20260904T194612Z/) / [`20260904T195731Z`](20260904T195731Z/).

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

## Multi-session static batch (`--parallel`) — TinyLlama Q4_K_M

Workload: 8 concurrent blocking `POST /v1/chat/completions`, `max_tokens=64`, temperature 0, `--nodes 1`, `--batch-window-ms 50` when `parallel>1`.

| Run | Backend | parallel=1 agg tg | parallel=8 agg tg | Speedup 8/1 | Notes |
|-----|---------|------------------:|------------------:|------------:|-------|
| [`20260901T154735Z-parallel`](20260901T154735Z-parallel/) | GPU | **28.8** t/s | 24.9 t/s | **0.87×** | Before multi-request decode batching |
| [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) | GPU | 28.8 t/s | **32.1** t/s | **1.11×** | `forwardMultiDecode` + batched CUDA GEMV |
| [`20260901T155136Z-parallel`](20260901T155136Z-parallel/) | CPU | 1.46 t/s | **2.22** t/s | **1.52×** | Clear aggregate uplift on CPU |

**GPU (post-fix):** `LocalInferencePipeline.forwardBatch` routes N decode steps through `ForwardPassHandler.forwardMultiDecode`. All supported handler families implement batched decode: **Llama**, **Phi-3**, **Phi-2**, **Qwen3 dense**, and **Qwen3 MoE** (attention batched; MoE FFN routed per stream). Linear projections and LM head use `cublasHSSgemvStridedBatched` / `GpuBlasOps` where GPU weights are resident (batch ≤ 8). Prefill windows stay serial on GPU.

| Handler family | `forwardMultiDecode` | Parity test |
|----------------|---------------------|-------------|
| Llama | Yes | `LlamaTransformerHandlerMultiDecodeTest` |
| Phi-3 | Yes | `Phi3TransformerHandlerMultiDecodeTest` |
| Phi-2 | Yes (CPU quant batched GEMV) | `Phi2TransformerHandlerMultiDecodeTest` |
| Qwen3 | Yes (+ `forwardBatch` prefill) | `Qwen3TransformerHandlerMultiDecodeTest` |
| Qwen3 MoE | Yes (MoE FFN per stream) | `Qwen3MoeTransformerHandlerMultiDecodeTest` |

**Phi-3 / Qwen3 GPU multi-session:** re-run `./scripts/performance-tests/compare-parallel.sh --gpu --sessions 8` with the target model when validating non-Llama speedup; Llama baseline is [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) (1.11× aggregate tg).

**GPU regression (0.87×, pre-fix):** static batching did not fuse multi-request decode on GPU.

1. **`LocalInferencePipeline` had no `forwardBatch` override** — N serial `forward()` per decode step.
2. **Prefill in `generateBatch()` is serial** — eight `prefillBatch()` calls in a loop before decode starts.
3. **Unfair baseline:** `--parallel 1` still launches each HTTP request on its own virtual thread (`dispatchSingle`), so eight clients overlap on the GPU lock. `--parallel 8` runs all eight in **one** `generateBatch()` on a single thread — fully serialized GPU work without batched kernels.

Handler `forwardBatch(BatchForwardRequest)` only batches **one request's prefill window** (W prompt tokens), not N concurrent decode streams.

CPU uplift (1.52×) likely comes from fewer contending threads and better cache locality despite the same serial decode path.

Re-run:

```bash
./scripts/performance-tests/compare-parallel.sh --gpu   # or --cpu
```

## Prefill microbatch (`--prefill-batch`) — TinyLlama Q4_K_M

Workload: long raw prompt (`n_prompt=256`), single blocking chat completion, JFR on. Script: `compare-prefill-batch.sh`.

| Run | Backend | batch=1 pp | batch=32 pp | Speedup 32/1 | `prefill.count` (1 / 32) |
|-----|---------|----------:|------------:|-------------:|-------------------------|
| [`20260901T234024Z-prefill`](20260901T234024Z-prefill/) | CPU | **2.30** t/s | **5.39** t/s | **2.35×** | 246 / 9 |

Default `--prefill-batch` is **32**. `--prefill-batch 1` matches per-token batched prefill (many small `PrefillBatch` JFR events). GPU re-run pending on reference SKU — expect ≥2× on long prompts when VRAM-resident.

```bash
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32
```

## Single-stream compare re-run

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
