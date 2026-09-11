# Tier 2: OpenAI Field Parity

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
- `PLAN-Infra-Tier1.md` (recommended prerequisite)
- `coordinator/.../OpenAiChatHandler.java`
- `coordinator/.../OpenAiAdapter.java`
- `sampler/.../Sampler.java`, `SamplingParams`
- `docs/features.md`, `RELEASE_NOTES.md`
- `api/src/main/resources/juno-api.yaml`

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P2 |
| **Exec step** | 1 (after P0 step 2) |
| **Depends on** | Tier 1 complete |
| **Blocks** | Tiers 3, 7, 11 |
| **Parallel with** | P0 / P1 if staffed |
| **Status** | **Feature complete** (2026-09-11) — §2 CPU regression [`20260911T221215Z`](../perf-compare/20260911T221215Z/); smoke [`target/tier2-smoke/20260911T223000Z/`](../../target/tier2-smoke/20260911T223000Z/) |

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| OpenAI `stop` / `seed` / `presence_penalty` | **wired** (`OpenAiChatHandler` → `SamplingParams` / decode loop) | **wired** (same OpenAI path + playback consumer) | **explicit no-op** (train REPL not OpenAI chat) | **wired** via chat completions when fields present | **wired** (per-request params in static + continuous slots) | N/A | N/A | N/A | N/A | absent = prior defaults |
| `response_format` (pre–Tier 3) | **fail closed** (HTTP 400 unless absent or `type=text`) | same | N/A | same | same | N/A | N/A | N/A | N/A | absent |
| `logit_bias` / `user` | **explicit no-op** (docs honesty; still ignored) | same | N/A | same | same | N/A | N/A | N/A | N/A | ignored |

## Cross-feature smoke (before feature complete)

- [x] Each **wired** cell: command + expected JFR/log proof recorded ([`target/tier2-smoke/20260911T223000Z/`](../../target/tier2-smoke/20260911T223000Z/) — seed match; stop truncates; `presence_penalty` 200; `response_format` 400)
- [x] Each **explicit no-op** cell: warning string + howto note (`logit_bias` / `user` ignored honesty in howto/features; LoRA train not OpenAI chat)
- [x] Each **follow-up** cell: none (`response_format` types → Tier 3)
- [x] §2 compares run as required by change surface (API regression gate [`20260911T221215Z`](../perf-compare/20260911T221215Z/), failures=0; LoRA optional)

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on OpenAI fields this tier claims to honor
- [x] Launcher unchanged (no new CLI flags)
- [x] User-facing docs / OpenAPI state which fields are honored vs ignored
- [x] ROADMAP §5 architectures covered (sampler + decode loop are handler-agnostic)

## Overview

llama-server honors stop sequences, seed, presence penalty, and related OpenAI fields. Juno currently lists `stop`, `presence_penalty`, `logit_bias`, `user`, and `seed` as ignored for compatibility.

This tier implements the high-value fields clients actually need for drop-in parity.

## Scope and compatibility

Goals:

1. Honor `stop` (string or array).
2. Honor `seed` for deterministic sampling.
3. Honor `presence_penalty` alongside existing `frequency_penalty`.
4. Accept `response_format` / grammar-related fields in the request DTO for Tier 3, but reject unsupported types with HTTP 400 until Tier 3 ships.

Non-goals:

- GBNF enforcement (Tier 3).
- Tool calling (Tier 4).
- `logit_bias` (remain ignored unless trivial; do not expand scope).
- Changing default sampling when fields are absent.

## Chosen design

- `stop`: tokenize stop strings where possible; also match decoded text suffixes. Map finish reason to OpenAI `stop` when hit.
- `seed`: wire into sampler RNG; same seed + same params → identical token sequence.
- `presence_penalty`: standard OpenAI-style presence penalty on sampled history.
- `response_format`: if present and not yet supported by Tier 3, return 400 with a clear message (do not silently ignore).

## Implementation

### 1. SamplingParams + Sampler — tests first

- Extend `SamplingParams` with stop strings/ids, seed, presence penalty.
- Unit tests: seeded repeatability; presence penalty changes ranking; stop id / string halts.

### 2. OpenAI adapter / handler

- Parse `stop`, `seed`, `presence_penalty`, `response_format` in `OpenAiChatHandler` request DTO.
- Map through `OpenAiAdapter` into generation / sampling config.
- Tokenizer helper for stop strings → token ids; keep decoded-suffix fallback.

### 3. Generation finish reasons

- Ensure stop via stop-sequence sets OpenAI-compatible finish reason.

### 4. Docs and OpenAPI

- Update `juno-api.yaml`, `docs/features.md`, `RELEASE_NOTES.md` so fields are no longer listed as ignored.
- Keep `logit_bias` / `user` honesty if still ignored.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when: **done** (2026-09-11) — §2 [`20260911T221215Z`](../perf-compare/20260911T221215Z/); smoke [`target/tier2-smoke/20260911T223000Z/`](../../target/tier2-smoke/20260911T223000Z/).

1. ~~Stop strings halt generation; `finish_reason` reflects stop.~~
2. ~~Same seed with fixed temperature produces an identical token sequence across two runs.~~
3. ~~`presence_penalty` changes ranking vs baseline in a unit test.~~
4. ~~Unsupported `response_format` returns 400 (until Tier 3).~~
5. ~~Docs / OpenAPI no longer claim implemented fields are ignored.~~
6. ~~Relevant module tests pass (`sampler`, `coordinator`, API fixtures).~~

## Implementation todos

1. ~~SamplingParams + Sampler changes + unit tests.~~
2. ~~OpenAiChatHandler / OpenAiAdapter wiring + stop tokenization helper.~~
3. ~~OpenAPI + features + RELEASE_NOTES + agent-arch as needed.~~
4. ~~§2 inference compare + mark feature complete / ROADMAP status; preview files; no zip.~~

## Preview files (expected)

Modified: `SamplingParams`, `Sampler` (+ tests), `OpenAiChatHandler`, `OpenAiAdapter` (+ tests), `juno-api.yaml`, `docs/features.md`, `RELEASE_NOTES.md`, ROADMAP status
