# Performance notes

Measured baselines live in [`perf-compare/README.md`](perf-compare/README.md). This file records tier-specific regression notes and exit-gate evidence.

## Fused Q4_K GPU matmul (`--mmq`)

**Plan:** [`infra-plan/PLAN-Infra-Tier13.md`](infra-plan/PLAN-Infra-Tier13.md) Phase B (in progress); LoRA play wiring [`infra-plan/PLAN-Infra-LoRA-MMQ.md`](infra-plan/PLAN-Infra-LoRA-MMQ.md) Phase 1 (**complete**).

**What:** When `--mmq on` (or `auto` with CUDA + kernel load), Q4_K projection weights stay packed on the device (`DeviceQ4KMatrix` / `ResidentQ4KWeight`). Decode/prefill GEMV uses a PTX fused dequant+accumulate kernel (`q4k_gemv`) instead of host dequant → FP16-resident cuBLAS. Non-Q4_K tensors still use the FP16 path. Default remains `--mmq off`.

**Surfaces:** Base text inference on Llama-family, Phi-3 (fused QKV/gate_up = one GEMV + host slice), and Qwen3 dense; plus `--lora-play` (playback-only). LoRA training logs a one-shot warning and keeps FP16/FP32 frozen residency (no Q4 transpose kernel). Phi-2 / Qwen3-MoE: no GPU residency yet (follow-up).

**Shared helper:** `Q4KResidentUpload` routes Q4 packed vs FP16 half upload.

**JFR:** `juno.MatVec.backend.cuda-resident-q4k.*`.

**Parity:** `Q4KMmqParityTest`, `Phi3Q4KMmqParityTest`, `Qwen3Q4KMmqParityTest` (GPU group) — fused kernel vs CPU `matVec` within `1e-2`. Policy: `LoraMmqPolicyTest` / `Q4KResidentUploadTest`. Playback order: `LoraQ4KPlaybackParityTest` (Q4 GEMV then LoRA delta).

**LoRA MMQ Phase 1 smoke** ([`target/lora-mmq-smoke/20260905T031236Z/`](../target/lora-mmq-smoke/20260905T031236Z/SUMMARY.md), GTX 1080):

| Gate | Result |
|------|--------|
| `--lora-play --mmq on` recall | `My name is Juno`; REPL `Fused Q4_K MMQ enabled`; JFR `cuda_resident_q4k.count=4020` |
| Base `--mmq on` (no LoRA) | exit 0; JFR `cuda_resident_q4k.count=4154` |
| `juno lora` + `JUNO_MMQ=on` | ignore-mmq warn; FP32 upload; loss finite (~2.77) |
| `compare-lora.sh --gpu --baseline release-0.1.2` | **ok** — train **0.98×**, play tps **1.00×** ([`20260910T030058Z-lora`](perf-compare/20260910T030058Z-lora/); prior ok [`20260905T031520Z-lora`](perf-compare/20260905T031520Z-lora/)) |

**Bake-off ([`20260910T025804Z`](perf-compare/20260910T025804Z/), `--gpu --vector 0`, `-DJUNO_MMQ=on`):** JFR `cuda_resident_q4k` on TinyLlama / Qwen2.5 / **Phi-3.5** / Mistral. Juno/llama tg ≈ **0.12–0.15×** (Phi-3.5 **0.12×** — P0 **0.5×** unmet). Paired `--mmq off` uplift (≥1.3×) not run in this session; vs prior FP16 GPU baseline Phi-3.5 tg **12.6**, MMQ-on **7.39** — exit gate still open.

## Vector SIMD / CPU MatVec

**Plan:** [`infra-plan/PLAN-Infra-Vector-SIMD.md`](infra-plan/PLAN-Infra-Vector-SIMD.md) (P0 step 4).

**Policy** (`VectorQuantKernels.policySummary()`, logged at startup):

| Phase | Path |
|-------|------|
| Q4_K / Q5_K weight-stationary accumulate | **Scalar** (inline loop; JIT auto-vectorize) |
| Q4_K / Q5_K dequant | **Scalar** |
| Q8_0 weight-stationary dequant | **Vector** when `jdk.incubator.vector` loads and the Q8_0 self-probe passes; else scalar |
| Q8_0 weight-stationary accumulate | **Scalar** (same as Q4/Q5) |
| `VectorQuantKernels.dot` | Unit tests / future gated use only — **not** on the weight-stationary hot path |

**Why:** Calling Vector `dot` once per (block, batch-row) at vision-scale batch (B≈741) on hosts with 128-bit `SPECIES_PREFERRED` was measured ~37–260× slower than sequential `matVec`, hanging moondream `forwardBatch` prefill for hours. Nested dedicated-ForkJoinPool dispatch around `IntStream.parallel()` had the same class of failure. Fix: scalar accumulate + `SimdThreadPool.forEachRow` → `ForkJoinPool.commonPool()` parallel stream (same as `matVecQ*raw`).

**Regression net:** `Q5KWeightStationaryBenchTest` — `sgemmQ5KWeightStationary` at B=128 must stay within 8× of sequential `matVecInto` and match logits within FP tolerance.

**Vision gate:** [`scripts/performance-tests/compare-vision.sh`](../scripts/performance-tests/compare-vision.sh) vs `47-vision` / [`perf-compare/20260904T141315Z-vision/`](perf-compare/20260904T141315Z-vision/). Default `--prefill single`. Module flag `--vector 0|1` in compare scripts only controls whether `--add-modules jdk.incubator.vector` is passed; with the scalar accumulate policy, Q4_K_M CPU tg should be near-parity between the two.

**Bake-off:** [`perf-compare/20260904T194612Z`](perf-compare/20260904T194612Z/) (`--vector 0`) vs [`perf-compare/20260904T195731Z`](perf-compare/20260904T195731Z/) (`--vector 1`). Juno tg v1/v0 ≈ **0.99–1.03** on the default Q4_K_M set — near-parity under the scalar accumulate policy. Track **feature complete**.

## Prefill microbatching (`--prefill-batch`)

**Run:** [`perf-compare/20260901T234024Z-prefill/`](perf-compare/20260901T234024Z-prefill/)

| Setting | Backend | pp t/s (JFR) | `prefill.count` | Notes |
|---------|---------|-------------:|----------------:|-------|
| `--prefill-batch 1` | CPU TinyLlama Q4_K_M | **2.30** | 246 | Per-token batched prefill (≈ one `PrefillBatch` per token) |
| `--prefill-batch 32` (default) | CPU TinyLlama Q4_K_M | **5.39** | 9 | **2.35×** faster than batch=1 |

Workload: raw 256-token user prompt (`compare-prefill-batch.sh`), `max_tokens=8`, JFR on. API reported 273 prompt tokens (chat template).

**JFR:** `juno.PrefillBatch` events populate `ForwardPass.prefill.*` on the API path (replacing the prior `prefill.count=0` gap on batched prefill).

**Parity:** `LlamaTransformerHandlerPrefillChunkParityTest` — chunked `forwardBatch` prefill matches whole-window logits within `1e-4`. `GenerationLoopTest.prefill_chunk_sizes_produce_same_tokens_as_whole_window` — chunk sizes 1 / 32 / whole window produce identical greedy decode.

**GPU:** Re-run on reference SKU (GTX 1080):

```bash
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32
```

Expect larger uplift than CPU when `runLayersBatch` uses GPU GEMM.

## GPU layer offload (`--gpu-layers`)

See [`perf-compare/20260901T032753Z/`](perf-compare/20260901T032753Z/) (mistral-7b, `JUNO_GPU_LAYERS=auto`).

## Static micro-batching (`--parallel`)

See [`perf-compare/20260901T173121Z-parallel/`](perf-compare/20260901T173121Z-parallel/) (GPU 1.11× aggregate tg, parallel 8 vs 1).
