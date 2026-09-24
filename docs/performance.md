# Performance notes

Measured baselines live in [`perf-compare/README.md`](perf-compare/README.md). This file records tier-specific regression notes and exit-gate evidence.

## Measurement boundary: one JFR configuration, and prompt-token parity

Two changes to how measurements are taken. Every entry in this file recorded before them is on the
other side of a boundary and is not strictly comparable with one recorded after; runs taken before
the change are kept and are still valid against each other.

**One recording configuration.** Juno starts JFR recordings from several places, and they used to
name different settings, so two runs could differ by their instrumentation overhead rather than by
the code under test. All of them now resolve `scripts/performance-tests/juno-perf.jfc` — see
[`howto.md`](howto.md) for how to point a run at a different file. The configuration also records
what a throughput number has to be read against: collection pauses, allocated bytes and their
attribution, hot methods, and monitor and park time. The 622 ms pause documented in the LoRA
playback section below, which read as a regression until it was root-caused by hand, is the failure
mode this makes visible by default rather than by investigation.

**Prompt-token parity.** `compare-llama-cpp.sh` used to prefill a fixed short sentence for Juno
while asking the reference tool for `n_prompt` tokens, so the two prefill figures described
different amounts of work — roughly 20 tokens against 128 in the runs on record. Parity is now the
default, each result records Juno's real `prompt_tokens`, and a prefill ratio whose deviation from
the requested count exceeds 10% is withheld with its reason stated rather than published. Prefill
ratios published before this change read as better than a like-for-like measurement supports;
generation ratios are unaffected.

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
fire correctly at that window size — the regression traces to attention cost
(`O(seq^2)`, untouched by this tier), not the GEMM this tier fixed.

**Attention JFR span (`juno.Attention`):** added to confirm this directly — a new event wraps each
`gqaInto`/`gqa` call (QK^T + softmax + attention-weighted V sum; scalar CPU in every handler
regardless of GPU layer offload, since `CudaMatVec`/`RocmMatVec` only accelerate the linear
projection GEMM/GEMV) separately from `juno.MatVec`, with `windowSize`/`startPosition`/
`contextLength` fields so cost-vs-context-length is directly queryable.

Re-run of `compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32`
([`perf-compare/20260916T003101Z-prefill`](perf-compare/20260916T003101Z-prefill/), TinyLlama
Q4_K_M, GTX 1080) confirms attention, not GEMM, dominates at this window size:

| Phase | `ForwardPass.*.total_ms` | `Attention.*.total_ms` | `Attention` share | `MatVec` (est.) share |
|-------|-------------------------:|-----------------------:|-------------------:|-----------------------:|
| Prefill (batch=32, 17 windows) | 16,708.5 | 13,088.7 | **78.3%** | ~9.9% |
| Decode (8 tokens, ctx grown to ~512+) | 674.3 | 432.6 | **64.2%** | ~35.9% |

`Attention.prefill.count` = 374 = 17 windows x 22 TinyLlama layers, `p95_ms` = 66.0 per per-layer
call (versus `Attention.decode.p95_ms` = 2.6 for a single query position at the same context
length) — the O(seq^2) shape is now directly visible instead of inferred from `ForwardPass` minus
`MatVec`. Tier 17's batched GEMM fix is not the remaining lever at long context; attention
(`gqaInto`'s scalar per-head, per-position QK^T/softmax/weighted-sum loop) is. This reopens the
case for the P5 FlashAttention-style work ahead of schedule for long-context workloads
specifically — P5 remains gated on Tier 8 baselines per the ROADMAP, this is evidence for that
gate, not a scope change on its own.

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

## Prefill GPU-residency fixes: pinned staging memory + adaptive chunk sizing

**Run:** [`perf-compare/20260918T153900Z-prefill-adaptive/`](perf-compare/20260918T153900Z-prefill-adaptive/)

Two independent fixes to the `static`-schedule prefill path, found by profiling rather than assumed:

1. **Pinned host-staging memory.** `CudaMatVec`'s batched-GEMM paths (`sgemmHalfBatched`,
   `sgemmHalfBatchedGemm`, `sgemmQ4KBatchedGemm`) and `CudaRmsNorm.normalizeBatch` staged their
   H2D/D2H buffers through plain `Arena.ofConfined()` (pageable host memory), which forces the CUDA
   driver to stage through its own internal pinned bounce buffer on every call. They now stage
   through `GpuBindings.hostMalloc`/`hostFree` (`cudaMallocHost`/`cudaFreeHost`; `hipHostMalloc`/
   `hipHostFree` on ROCm), grown-and-kept-max the same way the existing device-side scratch already
   is.
2. **Adaptive whole-prompt chunk sizing for `static` schedule.** `--prefill-batch` previously
   defaulted to a fixed 32-token chunk regardless of prompt length or free VRAM, re-dequantizing
   every resident Q4_K/Q5_K/Q6_K weight matrix on every chunk. The default now sizes the chunk to
   cover the whole prompt in one window whenever there is CUDA/ROCm headroom for it — live free VRAM
   is queried via `GpuContext.freeVramBytes()` (`cudaMemGetInfo`/`hipMemGetInfo`), and the chunk size
   is `floor(freeBytes * 0.5 / 65536)` tokens, floored at the old fixed default (32, so this can only
   grow the chunk relative to today's behavior, never shrink it) and capped at 65536. `continuous`
   schedule is unchanged (Tier 16 owns that chunking for decode-interleaving fairness). CPU-only runs
   keep the fixed 32-token default. `--prefill-batch N` still works as an explicit override on every
   surface.

Real GTX 1080, `mistral-7b-instruct-v0.1-q4_k_m.gguf`, `--gpu-layers auto --mmq auto`, a real 488-token
chat prompt, `max_tokens=16`:

| `--prefill-batch` | resolved chunk | `PrefillBatch` calls | prefill total (JFR) | MatVec calls | request wall |
|---:|---:|---:|---:|---:|---:|
| `32` (explicit, old fixed default) | 32 | 16 | 15710 ms | 7008 | 16971 ms |
| *(none — new adaptive default)* | 24889 | 1 | 11000 ms | 2289 | 12354 ms |

**-30.0% prefill time, -27.2% request wall time, 3.06x fewer MatVec launches** from consolidating 16
prefill windows into 1 for this prompt. Correctness: chunk-boundary numeric identity across chunk
sizes (any size, not just 32) is covered by the pre-existing
`LlamaTransformerHandlerPrefillChunkParityTest`, unaffected by which fixed value the resolver picks.

**Phase B checkpoint (no-go):** re-ran the GPU-resident-Rope/SwiGlu round-trip microbenchmark from the
GPU-resident-attention/elementwise-ops investigation under the new pinned-memory staging — still
1.56-2.18x slower than CPU scalar at prefill batch scale (worse than the original confounded
1.30-1.32x). Pinned memory does not close the gap; the per-launch cost of one ad-hoc GPU round trip
with no activation-residency chain remains the bottleneck regardless of memcpy speed. Does not proceed
to building dedicated `RopeKernel`/`SwiGluKernel` classes on this evidence.

**Note:** this session's `nsys` install fails on every invocation (`option is ambiguous`, reproduced
even on a bare `nsys profile -- echo hi` with no Juno-specific arguments), so the specific
`cudaMemcpyAsync`-collapse timeline verification could not be re-run quantitatively here; the
end-to-end wall-clock win and passing parity/regression tests are relied on instead. A future session
with a working `nsys` install should re-run that measurement directly.

```bash
./juno local --model-path models/mistral-7b-instruct-v0.1-q4_k_m.gguf --gpu-layers auto --mmq auto
# no --prefill-batch needed — sizes to the whole prompt automatically when VRAM allows
```

## GPU-resident attention (`--gpu-attention`)

**Run:** [`perf-compare/20260916T035952Z-prefill/`](perf-compare/20260916T035952Z-prefill/) (off) vs.
[`perf-compare/20260916T040113Z-prefill/`](perf-compare/20260916T040113Z-prefill/) (on)

Direct follow-on to the attention-share finding above: `--gpu-attention on|off|auto`
(`JUNO_GPU_ATTENTION`, default **off**, CUDA only) moves QK^T + softmax + weighted-V-sum onto the
GPU against a device-resident FP16 KV mirror (`DeviceKvCache` + `CudaGqaAttention` +
`gqa_attention.ptx`) instead of running `gqaInto`/`gqa` as scalar CPU Java. Wired for
`LlamaTransformerHandler` (Llama-family, Mistral, Qwen2) and vision (delegates to the same
handler); Phi-2/Phi-3/Qwen3/Qwen3-MoE keep the scalar path (**follow-up**, each owns a separate
attention implementation / KV map). LoRA train and `--lora-play` explicitly ignore the flag and warn
once (separate handler, own KV map/attention math), same pattern as `--mmq` under LoRA training.

`compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32` (TinyLlama Q4_K_M, GTX 1080):

| `--gpu-attention` | prefill-batch | pp t/s (JFR) | prefill ms | attention share of prefill |
|---|---:|---:|---:|---:|
| off (default) | 1 | 20.85 | 25,414.9 | 0.0% |
| off (default) | 32 | 31.04 | 17,076.8 | **78.7%** |
| on | 1 | 40.74 | 13,008.1 | 0.2% |
| on | 32 | **119.56** | **4,432.8** | **11.0%** |

At `prefill-batch=32` — the window size where attention is actually classified as a windowed
"prefill" JFR event rather than folded into per-token decode accounting — `--gpu-attention on`
takes pp throughput from 31.04 to 119.56 t/s (**3.85x**) and attention's share of prefill wall time
from 78.7% down to 11.0%. This is the honest before/after number this feature set out to produce:
attention was the dominant long-context cost (see the Tier 17 follow-on finding above), and moving
it to the GPU removes most of that cost rather than merely shifting it. The remaining ~11% share is
whatever stays on CPU around the batched kernel dispatch (RoPE, cache-write bookkeeping) plus the
kernel's own device time as measured by the same JFR span.

The `prefill-batch=1` row shows a smaller, real win (20.85 -> 40.74 t/s, **1.95x**) but its
`attention_share_pct` is not a meaningful before/after signal — at window size 1 each prompt token
is processed like a decode step, and JFR attributes `juno.Attention` events to the `decode` bucket
rather than `prefill` in that shape, not because attention cost disappeared.

**Standing regression gate** (`compare-llama-cpp.sh --gpu --vector 0`, default short prompt,
4-model set): [`perf-compare/20260916T034621Z/`](perf-compare/20260916T034621Z/) (off) vs.
[`perf-compare/20260916T035124Z/`](perf-compare/20260916T035124Z/) (on) — flat to modestly improved,
never regressed, at this short (~20-30 token) context length where attention's share of total cost
is naturally small: TinyLlama tg 28.85 -> 29.10 t/s, Qwen2.5-3B 13.39 -> 14.55 t/s, Phi-3.5-mini
12.87 -> 13.23 t/s, Mistral-7B tuned lane (`--mmq on --gpu-layers auto`) 16.30 -> 19.52 t/s
(**+20%**). This is expected: the feature's real leverage is long-context prefill/decode, not short
default-prompt throughput, matching why the dedicated `compare-prefill-batch.sh` repro above (not
the short-prompt regression gate) is this feature's real bake-off signal.

**LoRA regression gate** (`compare-lora.sh --gpu --baseline release-0.1.2`, flag stays off,
explicit no-op): [`perf-compare/20260916T035640Z-lora/`](perf-compare/20260916T035640Z-lora/) —
train_total_ms ratio 0.95x, ms/pass ratio 0.95x, playback tps ratio 0.89x vs. baseline, all within
gate (recall correct). Flat as expected — `LoraTrainableHandler` never reads
`GpuAttentionOptions`.

**Known limitation** (see `DeviceKvCache` javadoc): multi-token greedy-decode sequences can
occasionally diverge between `--gpu-attention on` and `off` on some prompts after 15+ tokens — FP16
KV rounding occasionally flips a close greedy decision, the same class of tradeoff already accepted
for `--mmq` and other reduced-precision paths in this codebase. Single-step logits match tightly
(parity test); this is not bit-identical-generation territory, here or anywhere else in Juno.

```bash
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32 --gpu-attention on
```

## GPU layer offload (`--gpu-layers`)

See [`perf-compare/20260901T032753Z/`](perf-compare/20260901T032753Z/) (mistral-7b, `JUNO_GPU_LAYERS=auto`).

## Static micro-batching (`--parallel`)

See [`perf-compare/20260901T173121Z-parallel/`](perf-compare/20260901T173121Z-parallel/) (GPU 1.11× aggregate tg, parallel 8 vs 1).

## Recommended flags (GPU)

`--mmq`, `--gpu-attention`, and `--gpu-layers` all default to `auto` (no flags needed). `auto`
resolves per-model to off/serial wherever the flag is not wired for that architecture or CUDA is
unavailable — it never forces an unsupported path. Each still carries its own correctness or
VRAM-fit caveat (see the caveats below, their own sections above, and `docs/howto.md`); pass
`off` explicitly to opt out of one, e.g. for a bit-identical CPU-parity baseline run.

```
--mmq auto --gpu-attention auto --gpu-layers auto   # already the default; shown for scripting clarity
```

**Run:** [`perf-compare/20260918T024641Z/`](perf-compare/20260918T024641Z/) (default 4-model GPU
set, each with a paired default row and a tuned row using the flags above)

| Model | tuned/default tg | notes |
|---|---:|---|
| TinyLlama-1.1B Q4_K_M | 1.63× | `--mmq` + `--gpu-attention` both wired |
| Qwen2.5-3B Q4_K_M | 1.48× | `--mmq` + `--gpu-attention` both wired |
| Phi-3.5-mini Q4_K_M | 1.60× | `--mmq` wired; `--gpu-attention` not yet wired for `phi3` (falls back to the existing scalar path, verified via 0 `juno.Attention` JFR events) — the whole 1.60× is from `--mmq` alone |
| Mistral-7B Q4_K_M | 35.6× | all three flags contribute; `--gpu-layers auto` is what gives this model GPU residency at all on an 8 GiB card |

Caveats to know before turning these on in production:

- **`--gpu-layers`**: VRAM headroom dependent — `auto` measures free VRAM and falls back toward
  CPU-resident layers when the model does not fit; see the GPU layer offload section above.
- **`--gpu-attention`**: occasional greedy-decode divergence at the bit level vs. the scalar CPU
  path (FP16 KV rounding can flip a close logits comparison) — same class of tradeoff already
  accepted for `--mmq` and other reduced-precision paths in this codebase; not bit-identical
  generation. Wired for Llama-family/Mistral/Qwen2 (and vision, which shares the same handler);
  Phi-2/Phi-3/Qwen3/Qwen3-MoE and ROCm remain a named follow-up. Its largest measured win (3.85× pp,
  see the GPU-resident attention section above) is at long `--prefill-batch` windows, not the short
  default prompt used in the sweep above.
- **`--mmq`**: packed Q4_K device GEMV; wired for every architecture in the default GPU set,
  including Phi-3.5-mini, independent of `--gpu-attention`'s narrower architecture coverage.
- None of these three flags are wired for LoRA train or `--lora-play` (`--mmq`/`--gpu-attention`
  explicitly no-op and warn there); see their own sections for the full interaction matrix.

## Multi-adapter LoRA playback + GGUF import (Tier 10)

`--lora-play` now accepts `path[:scale][,path[:scale]...]` — a bare path still defaults to scale
`1.0` (unchanged from before this tier). Multiple adapters combine as
`sum(scale_i * adapter_i)`, computed by rank-concatenating the adapters into one merged
`LoraAdapterSet` at load time (`LoraPlaybackMerge`) rather than threading a list through every
forward-pass call site — see its javadoc for the exact linear-algebra argument. The single-file,
scale-1.0 case is a pure identity return (the original `LoraAdapterSet` object, unchanged), so it
carries zero risk to the existing playback path.

`./juno lora-import --gguf adapter.gguf --out x.lora` converts a GGUF LoRA adapter into a Juno
`.lora` v2 checkpoint. Not verified against a real converter-produced GGUF-LoRA file this session
(no network access) — see `GgufLoraImporter`'s javadoc for the documented naming/layout convention
and the honest caveat; unrecognized tensors or a rank/shape mismatch fail the import closed.

As part of this tier, the pre-existing `x_juno_loras` (OpenAI API per-request adapter override)
silent-ignore gap under `--schedule static` was closed: it was previously fail-closed (HTTP 400)
only under `--schedule continuous`, silently ignored under `static`. It now fails closed on every
schedule until real per-request wiring is implemented (named follow-up, not this tier) — use
process-wide `--lora-play` with the multi-adapter syntax above instead.

**LoRA regression gate** (`compare-lora.sh --gpu --baseline release-0.1.2`, single-file
scale-1.0 scenario, unaffected by design): **ok** — train **1.00×**, playback wall tps **0.90×**
([`20260918T044915Z-lora`](perf-compare/20260918T044915Z-lora/)). CPU base-inference spot-check
(`compare-llama-cpp.sh --cpu --vector 0 --models tinyllama`, `--no-publish`): failures=0 — Tier 10
does not touch the base forward-pass/MatVec path.

Unit tests: `LoraPlaybackMergeTest` / `LoraPlaySpecTest` (`lora` module, reference-math and CLI
parsing incl. Windows drive-letter paths) and `GgufLoraImporterTest` (`node` module, successful
import plus fail-closed cases) — 24 cases total, all green.

## Ngram speculative decoding (`--spec-type`)

`--spec-type none|ngram-simple` (`--spec-ngram-n`, `--spec-ngram-m`) drafts tokens from an in-request
ngram cache (prompt + generated tokens, no second model), verifies the whole draft window against
this model in one batched pass (`ForwardPassHandler.forwardVerify` — same windowed one-GEMM-per-layer
path as `forwardBatch`, but keeps every position's logits instead of only the last), and emits the
target model's own prediction at the first mismatch. Sampling runs exactly once per position in
order and always emits its result, so rng/grammar state — and the emitted token — is byte-identical
to `--spec-type none` regardless of draft accuracy. Wired for `GenerationLoop.generate()`
(single-request decoding) only; `generateBatch` (static multi-request batching) does not draft/verify
yet, and the local REPL/API launcher warns at startup when both `--spec-type` and a `--parallel > 1`
batch config are configured together. `LlamaTransformerHandler` overrides `forwardVerify` (Llama /
Mistral / Qwen2 family); Phi-2/Phi-3/Qwen3/Qwen3-MoE and cluster/tensor-parallel pipelines fall back
to the correctness-preserving serial default (no speed benefit there yet).

Full live-smoke-test numbers, methodology, and the off-by-one correctness bug this smoke test caught
(a first cut fed drafted tokens directly as the verify window's input, corrupting KV at an
already-confirmed position — unit tests with a scripted, non-causal test double could not catch this):
`docs/perf-compare/README.md` → "Ngram speculative decoding — regression gate + live smoke test".
Headline: on a maximally-repetitive synthetic workload (TinyLlama Q4_K_M, GTX 1080, greedy decode),
draft acceptance reached **94.9%** and decode rounds dropped roughly 16× (903 single-token forwards
to 57 forward/verify calls), but wall-clock tg improved only **~7%** (59.1 -> 63.3 t/s) — `Attention`
JFR count/time roughly halved (the real saving, from batching multiple query positions into one
attention dispatch per round) while `MatVec` time was flat-to-slightly-higher (a batched-window GEMM
over several rows costs more per call than a single-row GEMV, even with far fewer calls) — consistent
with, not contradicting, this doc set's separate finding that per-launch host/FFI overhead rather than
kernel throughput is the current GPU decode ceiling. Regression gates: `compare-llama-cpp.sh --gpu
--models tinyllama` (default `--spec-type none`) failures=0; `compare-lora.sh` flat as expected
(LoRA never routes through `forwardVerify`).

Unit tests: `NgramDraftCacheTest` (insert/lookup/eviction), `GenerationLoopSpeculativeDecodeTest`
(token-identity vs plain decode, including a divergence case, using a position-indexed — not
call-count-indexed — test pipeline so a discarded/wasted verify position doesn't corrupt the
comparison), `LlamaTransformerHandlerVerifyParityTest` (batched verify vs serial `forward`, both
final-node and intermediate-node shapes), `JfrMetricsExtractorSpeculationTest` (`metrics` module).

## Draft-model speculative decoding (`--spec-type draft-simple`)

`--spec-type draft-simple --model-draft PATH` drafts tokens from a second, independently-loaded GGUF
model instead of an ngram cache, reusing the same `GenerationLoop.generate()` draft/verify loop and
`ForwardPassHandler.forwardVerify` batched-verify path as `ngram-simple` above — both strategies
implement a shared `DraftProposer` interface (`propose`/`observe`/`close`) so the loop itself does not
care which one is active. `DraftModelSession` drives the draft model through its own persistent KV
session with ordinary greedy `forward()` calls (one per drafted token, feeding its own prediction back
in — the same shape a plain non-speculative decode step already takes), then reconciles that tentative
continuation against ground truth after every round: it walks forward from the last position both
sides are known to agree on and, on the first disagreement, issues exactly one corrective `forward()`
call — no bulk resend of the drafted window, and no explicit KV-truncate API, since KV storage is
indexed by absolute position and a later real write simply overwrites a stale speculative one (the
same overwrite-in-place semantics `ngram-simple`'s verify window already relies on). `GenerationLoop`'s
constructor fails closed when `--spec-type draft-simple` is set without a loaded draft pipeline, or
when the draft and target `InferencePipeline.vocabSize()` differ — draft-proposed token ids are
compared directly against the target's own sampled ids, so a vocab mismatch would otherwise silently
compare incompatible id spaces. Wired for the local single-shard REPL only (`ConsoleMain.runLocalRepl()`
+ `loadDraftPipeline()`, sharing the target's `MatVec`/`GpuContext`); `--lora-play`, LoRA train, and
cluster/tensor-parallel launches fail closed at CLI-parse time with an explicit error rather than
silently ignoring `--model-draft`.

Full live-smoke-test numbers and methodology: `docs/perf-compare/README.md` → "Draft-model speculative
decoding — regression gate + live smoke test". Headline, on the same maximally-repetitive synthetic
workload as the `ngram-simple` entry above (TinyLlama Q4_K_M as `--model-draft`, Mistral-7B Q4_K_M as
the target, both sharing the same 32000-token Llama-family vocabulary, GTX 1080, greedy decode): output
was byte-identical to `--spec-type none` (token-identity exit gate met) and draft acceptance was decent
(**55.2%**, 53/96 drafted tokens), but wall-clock tg **regressed to 0.52×** (19.78 -> 10.35 t/s,
JFR) rather than improving. `juno.MatVec.count` nearly quadrupled (7,965 -> 31,058) because the draft
model's own decode/prefill/resync forward calls route through the same global `MatVec` span the target
uses — unlike `ngram-simple`'s free lookup-table proposals, `draft-simple`'s proposals cost a real
transformer forward pass per drafted token, and on this GPU that additional cost is not offset by the
verify-side savings. This directly compounds, rather than contradicts, this doc set's separate finding
that per-launch host/FFI overhead (not kernel throughput) is the current GPU decode ceiling: a smaller
draft model still issues thousands of its own tiny per-projection launches, and those are not free even
though the FLOPs they represent are small. Reported honestly as a negative result, not hidden — the
tier's own exit gate anticipates this ("TPS uplift documented when draft is small and acceptance is
high; failure cases documented"). Regression gates: `compare-llama-cpp.sh --gpu --models mistral`
(default `--spec-type none`) failures=0; `compare-lora.sh` flat as expected (LoRA never routes through
`forwardVerify` or touches `--model-draft`).

Unit tests: `DraftModelSessionTest` (propose/observe reconciliation: full acceptance needing no
resync, divergence triggering exactly one resync `forward()` call, continuing correctly after resync,
and a case that starves the session of an `observe()` call between rounds to prove it still self-heals),
`SpeculativeDecodeOptionsTest` (`draft-simple` parsing, fail-closed when `--model-draft` is missing),
and two new `GenerationLoopSpeculativeDecodeTest` cases (full agreement and a scripted divergence
between an independent draft pipeline and the target, plus dedicated cases for the missing-draft-pipeline
and vocab-mismatch fail-closed constructor checks).
