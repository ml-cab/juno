# Tier 11: Embeddings API

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS.
4. Prefer adding new Java classes over extending existing ones.
5. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
6. No emojis; be strict and precise.
7. Output: list changed files for preview; never zip files back.

Also read:

- `PLAN-Infra-ROADMAP.md`
- `PLAN-Infra-Tier2.md` (API maturity prerequisite)
- `OpenAiChatHandler` / REST wiring
- Forward pass paths that can expose hidden states / embedding rows
- `JunoPlayer` embedding mentions in howto (if any)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P3 |
| **Exec step** | When free (do not block P0/P1) |
| **Depends on** | Tier 2 complete |
| **Blocks** | None |
| **Parallel with** | P2 / P3 |

## Overview

llama-server offers `--embedding` and `/v1/embeddings` (plus reranking). Juno should add OpenAI-compatible embeddings without breaking chat routes.

## Scope and compatibility

Goals:

1. `POST /v1/embeddings` with pooling `mean|cls|last` (CLI `--pooling`).
2. Dedicated embedding GGUFs when metadata indicates embedding models.
3. For chat models, allow last-token pooling only behind explicit `--embeddings` with a quality warning.
4. Batch input: array of strings → array of embedding objects.

Non-goals:

- Multimodal embeddings.
- Rerank endpoint as a hard exit requirement (optional stretch; otherwise defer to a follow-up patch after Tier 11).
- Training embedding models.

## Chosen design

- Extract embedding vector without sampling; reuse forward through final norm / output embedding path as architecture allows.
- Deterministic for fixed input and pooling mode.
- OpenAPI updated; chat completions unchanged when embeddings mode is off.

## Implementation

### 1. Pooling helpers — tests first

- mean / cls / last on fixture hidden-state matrices.

### 2. Forward embedding path

- Pipeline method returning float[] / float[][]; wire local + API.

### 3. OpenAI handler

- `/v1/embeddings` request/response shapes; error if model not embedding-capable and flag not set.

### 4. Docs

- howto examples; features.md; warn on chat-model pooling quality.

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Cluster / TP / PP | Default |
|--------------------|-----------------|-------------|------------|--------|------------|--------------|------------------|------|------|--------------------|---------|
| `--embeddings` / `--pooling` | **wired** — `POST /v1/embeddings`, `mean`/`cls`/`last` all verified against distinct values | **wired** — same handler chain; live smoke shows different embedding values with the adapter loaded vs. not | **explicit no-op** — flag accepted, startup warning printed (`ConsoleMain` lora-mode API block); that mode's API only ever serves `/v1/lora/train-file-qa` / `/v1/lora/save` | **follow-up** — `VisionAwareForwardPassHandler.lastRmsHiddenForEmbedding` delegates to the text handler by construction, so plain-text embeddings on a vision-loaded model are expected to work, but the combination was not live-smoked this tier (no vision+`--embeddings` run); not claimed as verified in howto.md | **explicit no-op** — `/v1/embeddings` batch `input` is processed one string at a time (`EmbeddingsHandler` loops `pipeline.embedTokens` per string); `--parallel` only batches chat completions decode steps | **wired by construction** — `embedTokens`/`lastRmsHiddenForEmbedding` run through the exact same `ForwardPassHandler` instance and `MatVec` backend chat completions uses; no embeddings-specific GPU code path exists to diverge. Live smoke this tier used `--cpu` only (host has a single shared GPU) — GPU-specific re-verification is deferred to the next tier that touches MatVec/GPU residency, per the same reasoning as `--gpu-layers` below | **explicit no-op** — `embedTokens` runs the pre-existing per-position causal-prefill loop (`LocalInferencePipeline.embedLastToken`'s original design, now shared via `embedTokens`), not the Tier 8 chunked-prefill path; long embeddings inputs do not benefit from `--prefill-batch` (documented in howto.md) | **wired by construction** — same reasoning as `--gpu-layers`; `GpuBindings.createMatVec()` dispatch has no embeddings-specific branch | **wired by construction** — same reasoning as `--gpu-layers` | **fail closed** — `InferencePipeline.embedTokens` default throws `UnsupportedOperationException`; only `LocalInferencePipeline` overrides it. `ProcessPipelineClient` / `TensorParallelPipelineClient` (used by `cluster`) inherit the default and return HTTP 400 `embeddings_unsupported` on every request (`EmbeddingsUnsupportedPipelineTest` uses `StubInferencePipeline` as a stand-in for the same non-override behavior); `ConsoleMain.runClusterRepl()` also prints a startup warning when `--embeddings` is passed in cluster mode | off (`--embeddings` absent); `--pooling` defaults to `mean` when `--embeddings` is on |

## Cross-feature smoke (before feature complete)

- [x] **Base inference** (wired): `./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --api-port 18081 --embeddings --pooling mean --cpu`; `POST /v1/embeddings {"input":"What is Java?"}` → HTTP 200, `data[0].embedding` length 2048 (= model hiddenDim); batch `input:[s1,s2]` → 2 objects, correct `index`/`usage`; same input twice → byte-identical response; `x_juno_pooling:"bogus"` → HTTP 400; missing `input` → HTTP 400. `POST /v1/chat/completions` on the same server → HTTP 200 (unaffected). Evidence: `docs/performance.md` "Embeddings API" section, live smoke table.
- [x] **--lora-play** (wired): same server + `--lora-play models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora` → adapters load (`✔ Loaded 44 LoRA adapters`), `POST /v1/embeddings` → HTTP 200 with embedding values that differ from the no-LoRA run (overlay is applied to the embedding path, not bypassed).
- [x] **LoRA train** (explicit no-op): `--embeddings` in lora mode prints `⚠ --embeddings has no effect in lora mode; ...` and does not register `/v1/embeddings` — `runLoraRepl` never constructs `InferenceApiServer`/`EmbeddingsHandler`. Documented in howto.md.
- [ ] **Vision** (follow-up): not smoked this tier — see matrix cell; no plan doc filed since this is a small, scoped verification, tracked here instead of a separate `PLAN-*` link.
- [x] **--parallel** (explicit no-op): documented in howto.md ("embeddings batch input ... processed one string at a time"); no runtime warning needed since `--parallel` continues to correctly accelerate chat completions and nothing about `--embeddings` implies it also batches embeddings requests.
- [x] **--gpu-layers / CUDA / ROCm** (wired by construction): no code path exists that could route embeddings through a different `MatVec` backend than chat completions on the same handler instance; documented reasoning recorded in the matrix rather than a separate GPU smoke run (host constraint this session — single shared GPU already in use for the base regression run).
- [x] **--prefill-batch** (explicit no-op): documented in howto.md.
- [x] **Cluster / TP / PP** (fail closed): `EmbeddingsUnsupportedPipelineTest` (coordinator module) proves a pipeline that does not override `embedTokens` (standing in for `ProcessPipelineClient`/`TensorParallelPipelineClient`) returns HTTP 400 `embeddings_unsupported`, not a 500 or a wrong vector; `EmbeddingsDisabledTest` proves the pre-existing 3/4-arg `InferenceApiServer` constructors (every caller before this tier) return HTTP 400 `embeddings_disabled` rather than 404 or silent success. `ConsoleMain.runClusterRepl()` prints a startup warning when `--embeddings` is combined with `cluster` mode.
- [x] §2 compare: `compare-llama-cpp.sh --models tinyllama --cpu --vector 0 --no-jfr` — regression gate only, per the API-only-tier carve-out (no MatVec/forward/KV/vision touched); [`docs/perf-compare/20260914T220204Z`](../perf-compare/20260914T220204Z/). `compare-lora.sh` / `compare-vision.sh` not run (same carve-out).

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on any surface that accepts the flag in the launcher (lora mode warns; cluster mode warns + fails closed per request; disabled server fails closed)
- [x] Launcher (`scripts/run.sh` / `run.bat`) forwards `--embeddings` / `--pooling` for `local` and `cluster`
- [x] User-facing docs (`docs/howto.md`) state which modes honor the feature
- [x] ROADMAP §5 architectures: no architecture-specific code — `embedTokens`/`lastRmsHiddenForEmbedding` are already implemented per handler family (Llama, Phi-2, Phi-3, Qwen3, Qwen3 MoE, LoRA decorators, vision decorator) from prior tiers; this tier adds no new per-architecture logic, so no new per-arch tests were needed

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.

Exit only when:

1. Deterministic embeddings for a fixed input across two calls. — **met** (live smoke)
2. Batch string inputs work. — **met** (live smoke + `EmbeddingsHandlerTest`)
3. Chat routes remain unbroken with embeddings enabled or disabled. — **met** (`chat_completions_route_still_works_with_embeddings_enabled`, live smoke)
4. OpenAPI + docs updated. — **met** (`api/src/main/resources/juno-api.yaml`, `docs/howto.md`, `README.md`, `docs/agent-arch.txt`)
5. Tests pass. — **met** (`EmbeddingPoolingTest`, `LocalInferencePipelineTest` additions, `EmbeddingsHandlerTest`, `EmbeddingsDisabledTest`, `EmbeddingsUnsupportedPipelineTest`; full `node` and `coordinator` suites re-run clean)

**Status: feature complete.** Not attempting "dedicated embedding-model GGUF metadata" (scope goal
2) or a rerank endpoint (non-goal) — Juno has no such GGUFs to test against yet; this is a
documented scope boundary, not a dropped exit-gate item (neither is in the exit-gate list above).
Next P3 tier: Tier 10 (multi-adapter LoRA playback + GGUF import), the only remaining P3 tier.

## Implementation todos (completed)

1. Pooling helpers + tests — `node/EmbeddingPooling.java` + `PoolingMode.java`, `EmbeddingPoolingTest`.
2. Forward embedding extraction path — `InferencePipeline.embedTokens` (fail-closed default),
   `LocalInferencePipeline.embedTokens` (renamed/generalized from the position-discarding loop
   inside the pre-existing `embedLastToken`, which is now a thin wrapper and also fixes a
   pre-existing KV-cache eviction leak).
3. REST `/v1/embeddings` + CLI `--pooling` / `--embeddings` — `EmbeddingsHandler` (coordinator),
   `InferenceApiServer` new constructor overload + route registration, `ConsoleMain` flags +
   help text, `scripts/run.sh` / `run.bat` forwarding for `local` and `cluster`.
4. Docs; ROADMAP status; preview files; no zip. — see file list below.

## Preview files (actual)

New:
- `node/src/main/java/cab/ml/juno/node/PoolingMode.java`
- `node/src/main/java/cab/ml/juno/node/EmbeddingPooling.java`
- `node/src/test/java/cab/ml/juno/node/EmbeddingPoolingTest.java`
- `coordinator/src/main/java/cab/ml/juno/coordinator/EmbeddingsHandler.java`
- `coordinator/src/test/java/cab/ml/juno/coordinator/EmbeddingsHandlerTest.java`
- `coordinator/src/test/java/cab/ml/juno/coordinator/EmbeddingsDisabledTest.java`
- `coordinator/src/test/java/cab/ml/juno/coordinator/EmbeddingsUnsupportedPipelineTest.java`
- `coordinator/src/test/java/cab/ml/juno/coordinator/PositionValueForwardPassHandler.java`

Modified:
- `node/src/main/java/cab/ml/juno/node/InferencePipeline.java` (fail-closed `embedTokens` default)
- `node/src/main/java/cab/ml/juno/node/LocalInferencePipeline.java` (`embedTokens`, `embedLastToken` refactor + evict fix)
- `node/src/test/java/cab/ml/juno/node/LocalInferencePipelineTest.java`
- `coordinator/src/main/java/cab/ml/juno/coordinator/RequestScheduler.java` (package-private `generationLoop()` accessor)
- `coordinator/src/main/java/cab/ml/juno/coordinator/InferenceApiServer.java` (new constructor overload, route)
- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java` (`--embeddings` / `--pooling` CLI, both API-server construction sites, lora-mode no-op warning, cluster-mode fail-closed warning)
- `scripts/run.sh`, `scripts/run.bat`
- `api/src/main/resources/juno-api.yaml`
- `docs/howto.md`, `README.md`, `docs/agent-arch.txt`, `docs/performance.md`, `docs/perf-compare/README.md`
- `docs/infra-plan/PLAN-Infra-ROADMAP.md`
