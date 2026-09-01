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

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Deterministic embeddings for a fixed input across two calls.
2. Batch string inputs work.
3. Chat routes remain unbroken with embeddings enabled or disabled.
4. OpenAPI + docs updated.
5. Tests pass.

## Implementation todos

1. Pooling helpers + tests.
2. Forward embedding extraction path.
3. REST `/v1/embeddings` + CLI `--pooling` / `--embeddings`.
4. Docs; ROADMAP status; preview files; no zip.

## Preview files (expected)

New: pooling helper, embeddings handler (or section), tests

Modified: pipeline/handler forward APIs, OpenAPI yaml, CLI, docs, ROADMAP status
