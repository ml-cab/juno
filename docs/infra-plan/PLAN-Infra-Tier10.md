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

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|---|---|---|---|---|---|---|---|---|---|---|
| `--lora-play` multi-scale (`a:0.5,b:1.0`) | wired (this *is* the `--lora-play` surface) | wired | explicit no-op (train REPL never reads `--lora-play`; unaffected) | follow-up (multi-adapter × vision combination not verified this tier — single-adapter `--lora-play` × vision was already an unverified follow-up per Tier 7/11's rows, and multi-adapter inherits that same gap) | wired (merge happens once at load time into a single `LoraAdapterSet`; `--parallel` batching is unaffected downstream) | wired (residency logic doesn't inspect adapter count) | wired (prefill chunking is independent of LoRA) | wired (unit-tested; GPU dispatch is unchanged, same `LoraAdapterSet` shape post-merge) | wired (same reasoning — vendor-neutral `MatVec`, no LoRA-specific ROCm path) | off (opt-in: only multi-entry or non-1.0-scale specs invoke the merge path; single bare-path stays byte-identical) |
| `lora-import` subcommand | n/a — offline `.lora`-file converter, does not run during inference | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (not a serving-time flag; produces a `.lora` file consumed later via `--lora-play`) |
| `x_juno_loras` (per-request override) | fail closed (HTTP 400) | fail closed | n/a (REST-only field; train REPL has no HTTP surface) | fail closed | fail closed | fail closed | fail closed | fail closed | fail closed | always on — was previously fail-closed only under `--schedule continuous`; static schedule silently ignored it (ROADMAP §6 gap, pre-existing before this tier), now closed uniformly since real per-request wiring is a named follow-up, not implemented this tier |

## Cross-feature smoke (before feature complete)

- [x] `--lora-play a:S,b:S` (**wired**): `LoraPlaybackMergeTest` proves `merged.forward(x) == sum(scale_i * adapter_i.forward(x))` against hand-computed reference math (mirrors `LoraMergeFormulaTest`'s pattern); `LoraPlaySpecTest` covers CLI syntax parsing incl. Windows drive-letter paths. No live GPU JFR run needed beyond the existing `--lora-play` GPU tests (`LoraQ4KPlaybackParityTest` etc.) since the merge produces an ordinary `LoraAdapterSet` — same object type those tests already exercise.
- [x] Vision (**follow-up**): not newly verified this tier; documented as inheriting the existing single-adapter `--lora-play` × vision gap (Tier 7/11 follow-up), not silently implied covered.
- [x] `x_juno_loras` (**fail closed**, all schedules): `ContinuousLoraPolicyTest` — `forbidden(true, static)` and `forbidden(true, null)` now both assert `true` (previously only `continuous` did).
- [x] `lora-import` (**n/a / offline**): `GgufLoraImporterTest` — 9 cases covering successful import, multi-key/multi-layer, `.weight`-suffixed tensor names, alpha override vs. GGUF metadata, and fail-closed paths (unrecognized tensor, unsupported projection, missing half, rank mismatch).

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on any surface that accepts the flag in the launcher — `x_juno_loras` gap (silently ignored under static) closed as part of this tier
- [x] Launcher (`scripts/run.sh` / `run.bat`) forwards new flags for every command mode that should honor them — `lora-import` wired into both dispatchers alongside `merge`; `--lora-play`'s CLI surface is unchanged (same flag, richer syntax) so no launcher wiring was needed there
- [x] User-facing docs state which modes honor the feature — `docs/howto.md`, `juno-documentation/part3/05-lora-mode.md`
- [x] ROADMAP §5 architectures covered or named follow-up — multi-scale merge operates purely in the `lora` module on `LoraAdapterSet`/`LoraAdapter`, below any handler-specific code, so it applies uniformly to every architecture that already supports `--lora-play` (Llama-family, Phi-3, Qwen3 per `PLAN-Infra-LoRA-MMQ.md`); no new per-architecture gap introduced

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
