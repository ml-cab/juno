# Performance notes

Measured baselines live in [`perf-compare/README.md`](perf-compare/README.md). This file records tier-specific regression notes and exit-gate evidence.

## Quantized KV cache (`--cache-type-k/v`)

**Plan:** [`infra-plan/PLAN-Infra-Tier6.md`](infra-plan/PLAN-Infra-Tier6.md) (P1 step 1 — **in progress**).

**What:** `--cache-type-k` / `--cache-type-v` (`f16|q8_0`, default `f16`). CLI `f16` keeps the current float32 in-process path. `q8_0` stores per-token GGUF-style blocks (34 B / 32 elems) via `DenseKvTensor`; attention dequants to float scratch. Manager write-through (`NodeKVCacheAdapter`) carries typed payloads.

**Claim:** ≥2× persistent KV memory vs default float path (measured ~3.8× when `kvDim` aligns to 32). Not a throughput claim.

**Surfaces:** All text handlers + LoRA playback inference maps. LoRA **train** still uses ephemeral float KV inside teacher-forced forward (explicit no-op + WARNING when q8 flags set).

**Parity:** `Q8_0KvCodecTest`, `DenseKvTensorTest`, `LlamaTransformerHandlerCacheTypeParityTest`.

**Cross-feature smoke ([`target/cache-type-smoke/20260910T154500Z/`](../target/cache-type-smoke/20260910T154500Z/), CPU + GPU follow-up):**

| Gate | Result |
|------|--------|
| Base `f16` / `q8_0` short decode | exit 0; policy log names types |
| `--lora-play` + q8_0 | recall `My name is Juno` |
| `juno lora` + q8_0 | `train-loss=3.76` finite; ephemeral KV warn |
| `--parallel 2` + q8_0 | decode ok |
| Multi-decode unit + q8_0 | surefire 2/2 green |
| `--gpu-layers auto` + q8_0 | exit 0; `cache-type-k=q8_0`; resolved `gpu-layers=22` ([`20260910T163000Z`](../target/cache-type-smoke/20260910T163000Z/)) |
| Cluster CLI + node `-D` forward | launcher accepts flags; `ClusterHarness` forwards `JUNO_CACHE_TYPE_*` |

**Bake-off:** [`perf-compare/20260910T170557Z`](perf-compare/20260910T170557Z/) (`--gpu --vector 0`, default `f16` path). Failures=0; Juno/llama tg ≈ **0.19–0.24×** on TinyLlama/Qwen/Phi-3.5; mistral ≈ **0.015×** (fit gate unchanged).

**LoRA §2:** [`perf-compare/20260910T180703Z-lora`](perf-compare/20260910T180703Z-lora/) vs `release-0.1.2` — status **ok**. Train **1.00×**; wall playback tps **0.88×** (≥0.80). JFR `tps_jfr` ≈ **0.98×** on this re-run (prior false fail was a 622 ms GC pause on one decode step — see below).

**JFR playback_tps gap explained (20260910T171139Z):** not a steady-state MatVec/KV regression. HEAD TokenProduced timeline showed ~36–40 ms/token except one `ForwardPass` at `startPosition=25` lasting **665 ms**, coincident with a **`jdk.GCPhasePause` of 622 ms**. Other decode steps matched baseline (~37 ms p95). Wall REPL was **662 vs 554 ms** (~**0.84×** wall-implied tps). `TokenProduced.tps = count/(last−first)` on a 5-token span is dominated by that single GC spike → false **0.18×** JFR ratio.

**Gate fix:** `compare-lora.sh` now gates on **wall-clock** playback tps; JFR `TokenProduced.tps` is recorded as `playback.tps_jfr` only.

**Status:** feature complete (default `f16` bit-compatible; `q8_0` ≥2× persistent KV via codec).

## Fused Q4_K GPU matmul (`--mmq`)

**Plan:** [`infra-plan/PLAN-Infra-Tier13.md`](infra-plan/PLAN-Infra-Tier13.md) Phase B (**feature complete** as VRAM-fit); LoRA play [`infra-plan/PLAN-Infra-LoRA-MMQ.md`](infra-plan/PLAN-Infra-LoRA-MMQ.md) Phase 1 (**complete**).

**What:** When `--mmq on` (or `auto` with CUDA + kernel load), Q4_K projection weights stay packed on the device (`DeviceQ4KMatrix` / `ResidentQ4KWeight`). Decode/prefill GEMV uses a PTX fused dequant+accumulate kernel (`q4k_gemv`) instead of host dequant → FP16-resident cuBLAS. Non-Q4_K tensors still use the FP16 path. Default remains `--mmq off`.

**Claim (honest):** `--mmq` is for **VRAM fit** (more layers / larger models on a fixed GPU). It is **not** a decode-throughput win vs `--mmq off` on the current PTX (Phi-3.5 MMQ slower than FP16-resident on GTX 1080). Speed vs FP16 awaits a tile/`mul_mat_vec`-class kernel.

**Surfaces:** Base text inference on Llama-family, Phi-3 (fused QKV/gate_up = one GEMV + host slice), and Qwen3 dense; plus `--lora-play` (playback-only). LoRA training logs a one-shot warning and keeps FP16/FP32 frozen residency (no Q4 transpose kernel). Phi-2 / Qwen3-MoE: no GPU residency yet (follow-up).

**Shared helper:** `Q4KResidentUpload` routes Q4 packed vs FP16 half upload.

**Shared activations (Phase 1):** `MatVec.sgemvSameX` — Llama decode uploads each shared `x` once for Q/K/V and for gate/up (CUDA/ROCm), coalescing device sync. Does not keep hidden states on GPU across norm/attention/residual.

**JFR:** `juno.MatVec.backend.cuda-resident-q4k.*`.

**Parity:** `Q4KMmqParityTest`, `Phi3Q4KMmqParityTest`, `Qwen3Q4KMmqParityTest` (GPU group); `SgemvSameXParityTest`; policy / playback tests as before.

**LoRA MMQ Phase 1 smoke** ([`target/lora-mmq-smoke/20260905T031236Z/`](../target/lora-mmq-smoke/20260905T031236Z/SUMMARY.md), GTX 1080):

| Gate | Result |
|------|--------|
| `--lora-play --mmq on` recall | `My name is Juno`; REPL `Fused Q4_K MMQ enabled`; JFR `cuda_resident_q4k.count=4020` |
| Base `--mmq on` (no LoRA) | exit 0; JFR `cuda_resident_q4k.count=4154` |
| `juno lora` + `JUNO_MMQ=on` | ignore-mmq warn; FP32 upload; loss finite (~2.77) |
| `compare-lora.sh --gpu --baseline release-0.1.2` | **ok** — train **0.98×**, play tps **1.00×** ([`20260910T030058Z-lora`](perf-compare/20260910T030058Z-lora/); prior ok [`20260905T031520Z-lora`](perf-compare/20260905T031520Z-lora/)) |

**Bake-off ([`20260910T025804Z`](perf-compare/20260910T025804Z/), `--gpu --vector 0`, `-DJUNO_MMQ=on`):** JFR `cuda_resident_q4k` on TinyLlama / Qwen2.5 / **Phi-3.5** / Mistral. Juno/llama tg ≈ **0.12–0.15×** (Phi-3.5 **0.12×** — P0 **0.5×** unmet). Paired smoke: Phi-3.5 `--mmq off` ≈ **12.2** tg vs `--mmq on` ≈ **7.4** — speed gate deferred. Mistral packed-Q4 ≈ **0.14×** llama (near P0 **0.15×** fit).

**Kernel note:** Phi-3.5 `cuda_resident_q4k.p95` ≈ **2.3 ms** vs FP16 ≈ **0.3–0.45 ms**. Warp-per-row / shared-mem `x` tile PTX experiments did not beat the landed kernel on GTX 1080.

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
