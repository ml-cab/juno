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


Exit only when:

1. Stop strings halt generation; `finish_reason` reflects stop.
2. Same seed with fixed temperature produces an identical token sequence across two runs.
3. `presence_penalty` changes ranking vs baseline in a unit test.
4. Unsupported `response_format` returns 400 (until Tier 3).
5. Docs / OpenAPI no longer claim implemented fields are ignored.
6. Relevant module tests pass (`sampler`, `coordinator`, API fixtures).

## Implementation todos

1. SamplingParams + Sampler changes + unit tests.
2. OpenAiChatHandler / OpenAiAdapter wiring + stop tokenization helper.
3. OpenAPI + features + RELEASE_NOTES + agent-arch as needed.
4. List preview files; no zip.

## Preview files (expected)

Modified: `SamplingParams`, `Sampler` (+ tests), `OpenAiChatHandler`, `OpenAiAdapter` (+ tests), `juno-api.yaml`, `docs/features.md`, `RELEASE_NOTES.md`, ROADMAP status
