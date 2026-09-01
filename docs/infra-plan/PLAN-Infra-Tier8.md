# Tier 8: Inference Prefill Microbatching

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
- `PLAN-Infra-Tier1.md` (hard prerequisite for clean serving integration)
- `InferencePipeline.forwardBatch` (and handler batch paths)
- `GpuBlasOps` / strided batched GEMV (reuse ideas; do not couple to LoRA training)
- `GenerationLoop` prefill path
- LoRA microbatch docs for patterns only (`PLAN-LoRA-Tier9.md`)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P0 |
| **Exec step** | 3 (after P0 step 2) |
| **Depends on** | Tier 1 complete |
| **Blocks** | Tier 16; Tier 9; Tier 13 FlashAttn (P5) |
| **Parallel with** | Vector SIMD bake-off (P0 step 4) |

**Performance evidence (2026-08-31):** JFR `ForwardPass.prefill.count` = 0 on API path. This tier owns prefill JFR instrumentation and compare-script parity (`--raw-prompt`, `--gpu-layers` passthrough after Tier 5). See [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).

## Overview

llama.cpp ubatch speeds prompt evaluation by processing multiple prompt tokens efficiently. Juno should chunk long prefills into GPU-friendly batches while keeping decode single-token (aside from Tier 1 multi-request batching).

## Scope and compatibility

Goals:

1. CLI `--prefill-batch N` (default **32** or **64**; pick one and document) and env `JUNO_PREFILL_BATCH`.
2. Prefill processes the prompt in chunks of N using `forwardBatch` where supported.
3. Sequential fallback when a handler cannot batch.
4. Measurable prefill latency improvement on GPU for long prompts with logit parity.
5. Compare-script parity and harness updates: `--raw-prompt`, prefill JFR on API path, `--gpu-layers` passthrough (after Tier 5).

Non-goals:

- Changing multi-request `BatchConfig` semantics (Tier 1).
- Speculative decoding (Tier 9).
- Training microbatch CLI (LoRA Tier 11).

## Chosen design

- GenerationLoop owns chunking; handlers expose batch forward where already sketched.
- Decode step remains one new token at a time for a single sequence.
- Parity: final prefill hidden/logits (or first decode logits) match sequential within tolerance.

## Implementation

### 1. Loop chunking — tests first

- Unit/IT: prompt length not multiple of N; N=1 ≡ sequential.

### 2. Handler `forwardBatch`

- Ensure LLaMA-family GPU path batches correctly; CPU correctness mandatory even if no speedup.

### 3. Perf evidence

- Benchmark 512–1024 token prompts with **matched token count** vs llama-bench; record `prefillMs` in JFR / performance.md.
- Wire `ForwardPass.prefill.*` JFR on API chat path (currently `prefill.count` = 0 in bake-off).
- Add compare-script `--raw-prompt` mode (see PERF-ANALYSIS).

### 4. CLI / docs

- Wire flags; document defaults and when speedup is expected.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Long-prompt prefill is faster on GPU with N>1 vs N=1 (recorded numbers).
2. End-of-prefill logits match sequential within declared tolerance.
3. CPU path remains correct.
4. `--prefill-batch 1` matches prior sequential behavior.
5. Relevant tests pass; docs updated.

## Implementation todos

1. GenerationLoop prefill chunking + tests.
2. Handler forwardBatch parity / GPU path.
3. JFR + performance evidence.
4. CLI/docs; ROADMAP status; preview files; no zip.

## Preview files (expected)

Modified: `GenerationLoop`, handler batch APIs, CLI/run scripts, JFR if needed, docs, ROADMAP status

New: small options helper if needed (`PrefillBatchOptions`)
