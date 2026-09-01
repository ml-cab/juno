# Tier 12: Draft-Model Speculative Decoding

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
- `PLAN-Infra-Tier9.md` (hard prerequisite: verify contract)
- Multi-model load / memory constraints in player and node
- Tier 5 offload knobs for draft device placement

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P4 |
| **Exec step** | 2 (after Tier 9) |
| **Depends on** | Tier 9 complete |
| **Blocks** | None |
| **Parallel with** | P3 if staffed |

## Overview

llama.cpp supports draft models (`-md`) and advanced draft types (EAGLE3, DFlash, MTP). Juno adopts **draft-simple** only: a smaller GGUF drafts tokens; the target verifies using the Tier 9 verify loop.

## Scope and compatibility

Goals:

1. `--spec-type draft-simple` and `--model-draft PATH` (env allowed).
2. Draft and target must share vocabulary; fail closed otherwise.
3. Reuse Tier 9 verify; never emit unverified tokens.
4. Document memory use and disable path.

Non-goals:

- EAGLE3, DFlash, DSpark, MTP.
- Cross-machine draft RPC.
- Training draft models.

## Chosen design

- Dual pipeline / session object holding draft + target.
- Draft runs with same device policy as target by default; optional later knobs can follow as patches.
- Draft proposes up to M tokens; target verifies in one or few forwards (match Tier 9 semantics).

## Implementation

### 1. Dual load

- Load draft GGUF; validate vocab/arch compatibility checks that are necessary for shared token ids.

### 2. Speculate loop

- Wire draft propose → Tier 9-style verify on target.
- Tests with tiny fixtures or mocks.

### 3. CLI + memory

- Fail clearly on OOM; `--spec-type none` disables draft load.

### 4. Benchmarks

- Compare vs ngram and vs no speculation; document when draft helps.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Token identity vs non-speculative greedy holds.
2. TPS uplift documented when draft is small and acceptance is high; failure cases documented.
3. Clean disable without loading draft weights.
4. Vocab mismatch fails closed.
5. Docs and tests updated.

## Implementation todos

1. Dual pipeline session + vocab checks.
2. Draft propose + reuse verify loop.
3. CLI/env + memory failure paths.
4. Perf docs vs Tier 9; ROADMAP status; preview files; no zip.

## Preview files (expected)

New: draft session/helper classes, tests

Modified: GenerationLoop / player load path, CLI/run scripts, docs, ROADMAP status
