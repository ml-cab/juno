# Performance notes

Measured baselines live in [`perf-compare/README.md`](perf-compare/README.md). This file records tier-specific regression notes and exit-gate evidence.

## OpenAI field parity (`stop` / `seed` / `presence_penalty`)

**Plan:** [`infra-plan/PLAN-Infra-Tier2.md`](infra-plan/PLAN-Infra-Tier2.md) (P2 step 1 — **feature complete**).

**What:** Chat Completions honors `stop` (string/array ≤4), `seed` (seeded sampler RNG), and
`presence_penalty` (−2..2). `logit_bias` / `user` remain ignored with docs honesty.
`response_format` types `json_object` / `json_schema` are honored by constrained
decoding (see below).

**§2 regression:** [`perf-compare/20260911T221215Z`](perf-compare/20260911T221215Z/) (`--cpu --vector 0`).
Failures=0. GPU compare deferred (driver unavailable on host this run).

**Cross-feature smoke** ([`target/tier2-smoke/20260911T223000Z/`](../target/tier2-smoke/20260911T223000Z/)):

| Gate | Result |
|------|--------|
| Same `seed` twice | identical completion text |
| `stop=["STOP"]` | truncates before stop; `finish_reason=stop` |
| `presence_penalty=1.5` | HTTP 200 |
| `response_format.type=json_object` | HTTP 400 (pre-grammar smoke; now honored, see below) |
| `logit_bias` + `user` | HTTP 200 (explicit no-op) |

**Status:** feature complete. Constrained decoding shipped next (see below).

## Constrained decoding (GBNF + JSON Schema)

**Plan:** [`infra-plan/PLAN-Infra-Tier3.md`](infra-plan/PLAN-Infra-Tier3.md) (P2 step 2 — **feature complete**).

**What:** GBNF + documented JSON Schema subset mask illegal tokens before sample.
OpenAI `json_object` / `json_schema`, `x_juno_grammar`, CLI `--grammar-file` /
`--json-schema-file`. Fixture eval: 20 schemas, ≥95% valid JSON (`GrammarEvalTest`).
CLI grammar applies to `/v1/chat/completions` when the request omits a grammar
(`response_format.type=text` stays unconstrained).

**§2 regression:** [`perf-compare/20260912T193402Z`](perf-compare/20260912T193402Z/) (`--cpu --vector 0`).
Failures=0. Juno decode tg is in line with the previous CPU bake-off
([`20260911T221215Z`](perf-compare/20260911T221215Z/)). GPU compare skipped
(grammar is sampler-side; CUDA/ROCm matrix cells are N/A).

**Cross-feature smoke** ([`target/grammar-smoke/20260912T193149Z/`](../target/grammar-smoke/20260912T193149Z/)):

| Gate | Result |
|------|--------|
| `response_format.type=json_schema` | HTTP 200; parseable JSON; JFR `GrammarConstrained.count=3` |
| `response_format.type=json_object` | HTTP 200; parseable JSON object |
| `x_juno_grammar` yes/no | HTTP 200; output `yes` or `no` |
| `x_juno_grammar` + `json_object` | HTTP 400 |
| unsupported schema `pattern` | HTTP 400 (API) and CLI fail-closed |
| `--grammar-file` / `--json-schema-file` | constrain API with no request grammar; JFR count≥1 |
| `--lora-play` + `--parallel 2` + `json_schema` | two parseable replies; JFR count=2 |
| LoRA train `--grammar-file` | launcher WARNING (explicit no-op) |
| Vision `/v1/vision/chat` | no mmproj GGUF; wired surface is chat completions |
| CUDA / ROCm | N/A (sampler path) |

**Status:** feature complete. Function calling shipped next (see below).

## Function calling (`tools` / `tool_choice`)

**Plan:** [`infra-plan/PLAN-Infra-Tier4.md`](infra-plan/PLAN-Infra-Tier4.md) (P2 step 3 — **feature complete**).

**What:** OpenAI `tools` / `tool_choice` on chat completions. Prompt inject for
llama3 / chatml / qwen3; parse `<tool_call>` into `message.tool_calls`.
`tool_choice=none` never emits tools. `required` / named choice uses GBNF.
Unsupported templates and grammar conflicts fail closed. `/v1/vision/chat` does
not honor `tools`.

**§2 regression:** [`perf-compare/20260913T032734Z`](perf-compare/20260913T032734Z/) (`--cpu --vector 0`).
Failures=0. Juno decode tg is in line with the previous CPU bake-off
([`20260912T193402Z`](perf-compare/20260912T193402Z/)). GPU compare skipped
(tools are prompt+parse; CUDA/ROCm matrix cells are N/A).

**Cross-feature smoke** ([`target/tools-smoke/20260913T025903Z/`](../target/tools-smoke/20260913T025903Z/)):

| Gate | Result |
|------|--------|
| Qwen2.5 `tool_choice=required` | HTTP 200; `tool_calls` `get_weather`; `finish_reason=tool_calls` |
| named `tool_choice` | HTTP 200; `get_weather` |
| `tool_choice=none` | HTTP 200; no `tool_calls` |
| multi-turn `role=tool` | HTTP 200; continues generation |
| `tools` + `json_object` / `x_juno_grammar` | HTTP 400 |
| SSE tools path | buffered after generation (3 chunks) |
| JFR `GrammarConstrained.count` | 3 (required + named) |
| TinyLlama + `tools` | HTTP 400 (unsupported template) |
| `--lora-play` + TinyLlama + `tools` | HTTP 400 (same template fail-closed; overlay loaded) |
| `--parallel 2` + required | two `get_weather` tool_calls |
| Vision `/v1/vision/chat` | no mmproj; explicit no-op (howto) |
| CUDA / ROCm | N/A (prompt+parse) |

**Status:** feature complete. P2 API path (fields → grammar → tools) is done.

## Embeddings API (`--embeddings` / `--pooling`)

**Plan:** [`infra-plan/PLAN-Infra-Tier11.md`](infra-plan/PLAN-Infra-Tier11.md) (P3 — **feature complete**).

**What:** `POST /v1/embeddings`, off by default (`--embeddings`). Extracts the RMS/LayerNorm-normalized
hidden state at every prompt position (`InferencePipeline.embedTokens`, default method that throws
`UnsupportedOperationException` — only `LocalInferencePipeline` overrides it) and reduces it with
`EmbeddingPooling.pool(hidden, PoolingMode)`: `mean` (default), `cls`, `last`. Runs on the request's
own thread, bypassing `RequestScheduler`'s queue-depth limit / 429 semantics (documented v1 scope
boundary, not a silent gap).

**§2 regression:** [`perf-compare/20260914T220204Z`](perf-compare/20260914T220204Z/)
(`--models tinyllama --cpu --vector 0 --no-jfr`). Failures=0. This tier does not touch `MatVec`,
`forward`/`forwardMultiDecode`, KV, or vision code, so per Execution rule §2's API-only-tier
carve-out this is a regression spot-check on the existing chat completions path (not a throughput
claim), and `compare-lora.sh` / `compare-vision.sh` were not run.

**Live smoke** (`./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --api-port
18081 --embeddings --pooling mean --cpu`, TinyLlama Q4_K_M, hiddenDim=2048):

| Gate | Result |
|------|--------|
| `POST /v1/embeddings {"input":"What is Java?"}` | HTTP 200; `data[0].embedding` length 2048 |
| Batch `input: [s1, s2]` | 2 embedding objects, `index` 0/1, `usage.prompt_tokens` = combined token count |
| Same input twice | byte-identical response body (deterministic) |
| `x_juno_pooling: "bogus"` | HTTP 400 |
| Missing `input` | HTTP 400 |
| `POST /v1/chat/completions` with `--embeddings` on | HTTP 200 — unaffected |
| `--lora-play models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora` + `--embeddings` | HTTP 200; embedding values differ from the no-LoRA run (overlay is applied) — **wired** |
| Server without `--embeddings` (`InferenceApiServer` 3/4-arg constructor) | `POST /v1/embeddings` → HTTP 400 `embeddings_disabled` (`EmbeddingsDisabledTest`) |
| Distributed pipeline (`StubInferencePipeline`, stands in for gRPC / TP / PP node clients) + `--embeddings` on | HTTP 400 `embeddings_unsupported` — **fail closed**, not a 500 or a silently wrong vector (`EmbeddingsUnsupportedPipelineTest`) |

**Cross-feature matrix:** see `PLAN-Infra-Tier11.md`. `cluster` / TP / PP fail closed (no
implementation, matches `--schedule continuous`'s local/single-shard scope); vision and `--parallel`
are explicit no-ops on the embeddings path itself (batch input is processed sequentially, one
`embedTokens` call per string) with no interaction to break.

**Status:** feature complete.

## Block KV / gather tax (`--schedule` / `--kv-page-size`)

**Plan:** [`infra-plan/PLAN-Infra-Tier14.md`](infra-plan/PLAN-Infra-Tier14.md) (P1 step 2 — **feature complete**).

**What:** Dual KV path — dense under `--schedule static` (default); paged pool + gather-to-workspace under `continuous`. Cap `DenseKvTensor.MAX_SEQ_LEN` raised to **32768** so long-context cells are measurable. F16 paged gather uses page-bulk `FloatBuffer` copies (mitigation after first matrix). CLI: `--schedule` / `--kv-page-size` (+ envs) via `ConsoleMain`, `run.sh` / `run.bat`, cluster node `-D` forward.

**Microbench:** [`scripts/performance-tests/gather-tax-microbench.sh`](../scripts/performance-tests/gather-tax-microbench.sh) → `GatherTaxMicrobench` (TinyLlama-like GQA: 32/4/64). Full matrix: [`perf-compare/20260910T214300Z-gather-tax.md`](perf-compare/20260910T214300Z-gather-tax.md).

| Cell | gather % of (gather+attn) | Notes |
|------|--------------------------:|-------|
| batch **8** / ctx **8k** / page **16** | **4.61%** | Gate ≤ ~15% — **PASS**; Tier 15 unblocked on gather tax |
| batch 8 / ctx 8k / page 64 | 4.43% | Larger pages help little after bulk gather |
| batch 8 / ctx 2k / page 16 | 6.90% | Still under budget |
| batch 8 / ctx 32k / page 16 | 4.82% | Stress column OK |

**Budget decision:** Proceed to continuous scheduler on gather tax. Dual path remains: static stays dense (no gather).

**Pre-bulk baseline** (per-token byte unpack): gate cell was **10.72%** ([`20260910T213121Z-gather-tax.md`](perf-compare/20260910T213121Z-gather-tax.md)); still PASS, then page-bulk F16 gather applied.

**Cross-feature smoke** ([`target/tier14-smoke/20260910T220100Z/`](../target/tier14-smoke/20260910T220100Z/)):

| Gate | Result |
|------|--------|
| `--lora-play` + static | recall `My name is Juno` |
| Base continuous + page 16 | exit 0; paged policy log |
| static + `--kv-page-size 64` | `kv-page-size ignored (dense)` |

**Bake-off:** [`perf-compare/20260910T222026Z`](perf-compare/20260910T222026Z/) (`--gpu --vector 0`, default static/dense). Failures=0; Juno/llama tg ≈ **0.16–0.23×** on TinyLlama/Qwen/Phi-3.5; mistral ≈ **0.015×**.

**LoRA §2:** [`perf-compare/20260910T221031Z-lora`](perf-compare/20260910T221031Z-lora/) vs `release-0.1.2` — status **ok**. Train **1.00×**; wall playback tps **0.88×** (≥0.80).

**Status:** feature complete.

## Continuous batching (`--schedule continuous`)

**Plan:** [`infra-plan/PLAN-Infra-Tier15.md`](infra-plan/PLAN-Infra-Tier15.md) (P1 step 3 — **feature complete**).

**What:** Local/in-process running-set engine; SSE and non-stream share `forwardBatch` steps (`juno.ContinuousStep`). Default remains `static` (dense KV). Cluster / TP / PP auto-fallback to static. Per-request `x_juno_loras` fail-closed; LoRA train treats continuous as no-op (REPL WARNING). Prefix reuse is session-scoped (`x_juno_session_id`); `GET /v1/cluster/health` exposes `prefixLookups` / `prefixHits` / `prefixHitRate`.

**Bake-off:** [`perf-compare/20260911T194430Z-continuous/`](perf-compare/20260911T194430Z-continuous/) via `scripts/performance-tests/compare-schedule.sh --gpu`.

| Workload | Result |
|----------|--------|
| Multi-session TPS (8×64, GPU) | continuous **25.6** agg t/s vs static **29.7** (**0.86×**) |
| Concurrent SSE | mean TTFT ≈ **6.4 s**, TPOT ≈ **208 ms**; `ContinuousStep` max_decode_batch=**8**, shared_steps=**64** (proof **PASS**) |
| Prefix (multi-turn session) | lookups=8, hits=7, hit rate **0.875**; trie survives across turns |

P1 phase gate “continuous SSE beats static”: **unmet** on synchronized arrival (honest).

## Mixed chunked prefill (continuous)

**Plan:** [`infra-plan/PLAN-Infra-Tier16.md`](infra-plan/PLAN-Infra-Tier16.md) (P1 step 4 — **feature complete**).

**What:** Under `--schedule continuous`, admit no longer blocks on full prompt eval.
Remaining prompt advances in `--prefill-batch` ubatch chunks mixed into the same
engine steps as decode; decode is preferred when the step slot budget
(`--parallel` cap) is full. Static / single-request `--prefill-batch` unchanged.
Bake-off baseline: `-Djuno.continuous.mixedPrefill=false` (admit-time full prefill).

**Bake-off:** [`perf-compare/20260911T204721Z-mixed-prefill/`](perf-compare/20260911T204721Z-mixed-prefill/)
via `scripts/performance-tests/compare-mixed-prefill.sh --gpu`
(TinyLlama Q4_K_M · n_prompt=256 · 3 shorts · prefill-batch=32 · parallel=8).

| Metric | mixed (default) | admit-time baseline | mixed/admit |
|--------|----------------:|--------------------:|------------:|
| Short mean TTFT ms | 4552 | 13929 | **0.327×** |
| Short mean TPOT ms | 929 | 189 | 4.9× (shared-step tradeoff) |
| Short max TTFT ms | 4562 | 13938 | — |
| JFR `prefill_chunks` | 12 | — | proof **PASS** |

**Short-decode latency bound (this SKU/recipe):** short TTFT max **4562 ms** under
mixed load; re-run gate ≤ **1.25×** that max (**≤ 5702 ms**).

**Cross-feature smoke:** [`target/tier16-smoke/20260911T205500Z/`](../target/tier16-smoke/20260911T205500Z/)
(`ContinuousStep.prefill_chunks=6`, `max_decode_batch=2`).

**§2 bake-off:** [`perf-compare/20260911T204900Z`](perf-compare/20260911T204900Z/) (`--gpu --vector 0`). Failures=0.

**LoRA §2:** [`perf-compare/20260911T205447Z-lora`](perf-compare/20260911T205447Z-lora/) vs `release-0.1.2` — status **ok**. Train **1.00×**; wall playback tps **0.86×** (≥0.80).

**Status:** feature complete.

## Continuous batching — prior smoke / §2

**Cross-feature smoke** ([`target/tier15-smoke/20260911T200500Z/`](../target/tier15-smoke/20260911T200500Z/)):

| Gate | Result |
|------|--------|
| continuous + `--parallel 2` SSE | `ContinuousStep.max_decode_batch=2` |
| static + `--kv-page-size 64` | `kv-page-size ignored (dense)` |
| `x_juno_loras` under continuous | HTTP 400 fail-closed |
| `juno lora` + continuous | REPL WARNING: continuous is a no-op for train |

**§2 bake-off (continuous landing):** [`perf-compare/20260911T195008Z`](perf-compare/20260911T195008Z/). Failures=0; TinyLlama/Qwen/Phi-3.5 ≈ **0.15–0.22×**; mistral ≈ **0.013×**.

**LoRA §2 (continuous landing):** [`perf-compare/20260911T195711Z-lora`](perf-compare/20260911T195711Z-lora/) vs `release-0.1.2` — status **ok**. Train **0.98×**; wall playback tps **0.88×** (≥0.80).

**Status:** feature complete (P1 SSE-beats-static gate unmet).

## Quantized KV cache (`--cache-type-k/v`)

**Plan:** [`infra-plan/PLAN-Infra-Tier6.md`](infra-plan/PLAN-Infra-Tier6.md) (P1 step 1 — **feature complete**).

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

**What:** When `--mmq on` (or `auto` with CUDA + kernel load), Q4_K / Q5_K / Q6_K projection weights stay packed on the device (`DeviceQ4KMatrix` / `ResidentQ4KWeight`). Decode/prefill GEMV quantizes the activation to Q8_1 and integer-dots packed weights (`quantize_q8_1` + `q4k_gemv` / `q5k_gemv` / `q6k_gemv`) instead of host dequant → FP16-resident cuBLAS. Non-K-quant tensors still use the FP16 path. Default remains `--mmq off`.

**Claim (honest):** `--mmq on` is both **VRAM fit** (packed Q4 residency) and a **measured decode-throughput win** vs `--mmq off` on CUDA (Q8_1 activation + `dp4a` integer-dot GEMV). Default remains **off**. P0 Phi-3.5 ≥ 0.5× peer is still **unmet**.

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
| `compare-lora.sh --gpu --baseline release-0.1.2` | **ok** — train **1.00×**, play tps **0.88×** ([`20260911T235455Z-lora`](perf-compare/20260911T235455Z-lora/); prior ok [`20260910T030058Z-lora`](perf-compare/20260910T030058Z-lora/)) |

**Bake-off ([`20260911T235203Z`](perf-compare/20260911T235203Z/), `--gpu --vector 0 --mmq on --gpu-layers auto`):** JFR `cuda_resident_q4k` on TinyLlama / Qwen2.5 / **Phi-3.5** / Mistral (cpu.count=0). Juno/peer tg: TinyLlama **0.21×**, Qwen2.5 **0.27×**, Phi-3.5 **0.33×** (P0 **0.5×** unmet), Mistral **0.43×** (P0 **0.15×** **met**). Paired [`20260911T235353Z`](perf-compare/20260911T235353Z/): Phi-3.5 `--mmq off` **12.83** tg vs `--mmq on` **19.34** — **1.51×** (tile-kernel ≥1.3× **met**). Prior PTX bake-off [`20260910T025804Z`](perf-compare/20260910T025804Z/) (Phi-3.5 MMQ **7.4** tg) is superseded.

**Kernel note:** Phi-3.5 `cuda_resident_q4k.p95` ≈ **0.32 ms** (was ≈ **2.3 ms** on the float-dequant PTX). GEMV microbench on GTX 1080: Q4 3072×3072 ≈ **0.40 ms** vs FP16 ≈ **0.62 ms**.

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

**GPU (Tier 17 — batched-prefill GEMM):** prefill windows above `HALF_SGEMM_BATCH_MAX` (8) now
route through a real tiled GEMM (`cublasGemmEx`, FP16-resident weights directly; Q4_K/Q5_K/Q6_K
weights dequantized once to an FP16 scratch buffer then the same GEMM) instead of one serial
`sgemv` call per prefill token. Before this fix, GPU prefill and decode throughput sat within
roughly 1x of each other on every model — the fingerprint of prefill never getting a batched
kernel at all.

`compare-llama-cpp.sh --gpu` (default prompt, GTX 1080), `--mmq off`
([`perf-compare/20260915T041705Z`](perf-compare/20260915T041705Z/)) and `--mmq on`
([`perf-compare/20260915T042207Z`](perf-compare/20260915T042207Z/)):

| Model | `--mmq off` pp | `--mmq off` tg | pp/tg | `--mmq on` pp | `--mmq on` tg | pp/tg |
|-------|---------------:|---------------:|------:|--------------:|--------------:|------:|
| tinyllama-1.1b Q4_K_M | 64.97 | 28.66 | 2.27x | 59.38 | 39.20 | 1.51x |
| qwen2.5-3b Q4_K_M | 34.54 | 13.22 | 2.61x | 30.99 | 18.75 | 1.65x |
| Phi-3.5-mini Q4_K_M | 35.48 | 12.74 | 2.78x | 35.04 | 20.00 | 1.75x |
| mistral-7b Q4_K_M | 0.93 | 0.54 | 1.72x (CPU fallback, OOM — unaffected by this tier, Tier 5's domain) | 21.57 | 16.06 | 1.34x |

Qualitative gate (pp materially greater than tg, not pp ~= tg) holds for every model that fits GPU
residency, under both `--mmq off` and `--mmq on`. The quantitative "≥3x today's pp" floor from the
Tier 17 plan doc does not hold at face value against these absolute numbers — that floor's "today"
baseline was measured with a token-count-matched raw prompt, while these bake-off numbers use the
compare script's default short API/chat-template prompt (~20-30 tokens); the two are not directly
comparable (same prompt-length/methodology gap Tier 8 already flagged for `pp`, tracked as a
compare-script parity follow-up). Treat the pp/tg ratio above, not the absolute pp value, as this
tier's real signal.

At very long prefill windows (`--prefill-batch 512`, raw 512-token prompt,
[`perf-compare/20260915T043143Z`](perf-compare/20260915T043143Z/)) pp *regresses* relative to the
32/128-token-window runs (TinyLlama 64.2 -> 31.8 t/s). JFR confirms the new batched-GEMM path does
fire correctly at that window size — the regression traces to attention/softmax/RoPE cost
(`O(seq^2)`, untouched by this tier and not yet separately instrumented in JFR), not the GEMM this
tier fixed. Follow-up: add a JFR span around attention specifically before publishing a
`--prefill-batch >= 512` number as a tier result.

Per-handler live JFR proof: the new `cuda-resident-fp16-gemm` / `cuda-resident-q4k-gemm` backend
labels fire on real TinyLlama and Phi-3.5-mini forward passes (Llama-family and Phi-3 handlers);
Qwen3 shares the identical `backend.sgemm(...)` call site by code inspection but has no loadable
non-MoE Qwen3 GGUF fixture in this session to live-verify against (pre-existing Model E2E gap, not
a Tier 17 regression). `RocmMatVec` has no batched-GEMM override for any residency type — named
follow-up, no ROCm hardware available to implement or validate.

Correctness: `CudaSgemmBatchedPrefillParityTest` (`DeviceHalfMatrix`/`DeviceQ4KMatrix`, batches
{1, 8, 9, 16, 32, 128}, non-tile-aligned shapes) and `Q4KDequantParityTest` (isolated dequant
kernels vs. `GgufKQuantCodec.decodeRows`) are green, plus
`CudaSgemmBatchedPrefillConcurrencyTest` (4 threads, distinct shapes/seeds per thread, large-batch
path) confirms no cross-thread corruption in the per-thread `Fp16Scratch` / `Q4KDequantScratch`
buffers.

```bash
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32
```

## GPU layer offload (`--gpu-layers`)

See [`perf-compare/20260901T032753Z/`](perf-compare/20260901T032753Z/) (mistral-7b, `JUNO_GPU_LAYERS=auto`).

## Static micro-batching (`--parallel`)

See [`perf-compare/20260901T173121Z-parallel/`](perf-compare/20260901T173121Z-parallel/) (GPU 1.11× aggregate tg, parallel 8 vs 1).
