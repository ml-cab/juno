# Tier 10: Multi-Adapter Scales + GGUF LoRA Interop

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
- `docs/LoRA.md`, `.lora` v2 format / `LoraAdapterSet`
- `--lora-play` wiring in ConsoleMain / nodes
- llama.cpp GGUF LoRA / HF GGUF-my-LoRA format (reference only)
- `docs/lora-plan/PLAN-LoRA-ROADMAP.md` (training remains separate)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P3 |
| **Exec step** | When free (do not block P0/P1) |
| **Depends on** | Tier 1 optional for per-request routing |
| **Blocks** | None |
| **Parallel with** | Any phase after Tier 1 |

## Overview

llama-server loads multiple LoRA adapters with scales and allows per-request overrides. The Hub also distributes GGUF LoRA adapters. Juno trains `.lora` natively; this tier bridges playback and import without moving training into llama.cpp.

## Scope and compatibility

Goals:

1. Load multiple `.lora` files with scales: `--lora-play a.lora:0.5,b.lora:1.0`.
2. Optional OpenAI extension `x_juno_loras: [{ "id": 0, "scale": 0.5 }, ...]`.
3. `./juno lora-import --gguf adapter.gguf --out x.lora` mapping tensors → logical projection keys.
4. Fail closed on architecture / rank / shape mismatch.

Non-goals:

- Changing LoRA training math or checkpoint training format ownership (lora-plan).
- Export to GGUF LoRA unless cheap enough to include; **import is the priority**.
- Tensor-parallel LoRA training.
- Per-request heterogeneous adapters inside one **continuous** batch (ROADMAP v1 policy: global `--lora-play` set only under `--schedule continuous`; document fail-closed or static/serial fallback for mismatched `x_juno_loras`).

## Chosen design

- Multi-adapter apply: effective delta is sum of `scale_i * adapter_i` (document exact formula vs alpha scaling already in adapters).
- Parity: two adapters at scale 1.0 matching a single merged reference is not required if merge is F32 bake; instead match sequential apply reference math in tests.
- Import reads GGUF adapter tensors, writes Juno `.lora` v2.
- Continuous schedule interaction: document and test the ROADMAP **global-adapter-only** v1 policy; do not claim vLLM-style per-request multi-LoRA continuous until a later tier.

## Implementation

### 1. Multi-scale playback — tests first

- Apply two adapters with scales; compare to hand-computed reference.

### 2. CLI / API

- Parse `--lora-play` scale syntax; optional `x_juno_loras`.
- Preserve single-file `--lora-play path` back-compat (implicit scale 1.0).

### 3. GGUF import

- `lora-import` subcommand; fixture GGUF adapter → playback smoke.
- Clear errors for unknown tensor names / arch mismatch.

### 4. Docs

- howto + LoRA.md playback section; cookbook note if applicable.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Multi-scale playback matches reference math in unit tests.
2. Import produces a playback-equivalent `.lora` on a fixture.
3. Invalid GGUF / mismatched arch fails closed with a clear message.
4. Single-adapter path remains backward compatible.
5. Docs state continuous-schedule multi-LoRA v1 policy (global set only; per-request heterogeneous continuous deferred).
6. Docs updated; tests pass.

## Implementation todos

1. Multi-scale adapter apply + tests.
2. CLI/API scale syntax.
3. GGUF LoRA import → `.lora` v2 + fixture test.
4. Docs; ROADMAP status; preview files; no zip.

## Preview files (expected)

New: import tool/classes, multi-scale helper, tests

Modified: lora playback load path, ConsoleMain / run scripts, OpenAI handler (optional), `docs/LoRA.md`, howto, ROADMAP status
