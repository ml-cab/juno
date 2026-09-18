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

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| `--spec-type draft-simple` / `--model-draft` | wired (local single-shard REPL, Llama/Mistral/Qwen2 family via `LlamaTransformerHandler.forwardVerify`; Phi-2/Phi-3/Qwen3/Qwen3-MoE **follow-up** — correctness-preserving serial default, same gap Tier 9 already named) | fail closed (CLI parse time: `--spec-type draft-simple` combined with `--lora-play` exits with a clear error before any model load) | fail closed (same CLI-parse-time guard; `runLoraRepl()` never reached with draft-simple) | follow-up (untested; draft pipeline is plain-text-only, never wraps `VisionAwareForwardPassHandler` — combining with a vision-capable target model is unverified) | explicit no-op (reuses Tier 9's existing warning: `GenerationLoop.generateBatch` never drafts/verifies, `--spec-type` and `--parallel > 1` together log a startup WARNING) | wired (draft pipeline shares the target's `MatVec`/`GpuContext` — same device policy, no separate `--gpu-layers` knob for the draft model itself in v1) | wired (orthogonal — prefill is unaffected; speculation only touches decode) | wired (CUDA GEMM path exercised live on GTX 1080) | follow-up (not tested on ROCm hardware this session) | off (`none`) |

## Cross-feature smoke (before feature complete)

- [x] **wired** (base inference, Llama family, CUDA, local REPL): `./juno local --model-path models/mistral-7b-instruct-v0.1-q4_k_m.gguf --model-draft models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --spec-type draft-simple --gpu-layers auto --temperature 0 --spec-ngram-m 8 --jfr 30s` — see `docs/perf-compare/README.md` → "Draft-model speculative decoding — regression gate + live smoke test" for the exact command, JFR acceptance numbers, and byte-identity check against `--spec-type none`.
- [x] **fail closed** (`--lora-play` / LoRA train + `--spec-type draft-simple`): CLI-parse-time guard in `ConsoleMain.main()` exits with a clear error before any model load; `docs/howto.md`'s `--spec-type` row states the limitation.
- [x] **explicit no-op** (`--parallel > 1` + `--spec-type draft-simple` together): reuses Tier 9's existing `resolveBatchConfig()` warning (now correctly threads `--model-draft` through so the warning fires on the flag actually resolving to `draft-simple`, not just when `--model-draft` happens to be unset).
- [ ] **follow-up** cells (Vision, ROCm, Phi-2/Phi-3/Qwen3/Qwen3-MoE `forwardVerify` overrides, cluster/tensor-parallel): not yet linked to a dedicated follow-up doc — tracked here until one is opened, same as Tier 9's equivalent cells.
- [x] §2 compares run: `compare-llama-cpp.sh --gpu --models mistral` (failures=0, default `--spec-type none`) + `compare-lora.sh --gpu --baseline release-0.1.2` (flat, expected — LoRA never reaches `--spec-type draft-simple`, guarded fail-closed at CLI parse time).

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on any surface that accepts the flag in the launcher (fail-closed CLI-parse-time guard for LoRA/cluster; WARNING reused for `--parallel`)
- [x] Launcher (`scripts/run.sh` / `run.bat`) forwards `--model-draft` for `local` (the only mode that honors it, same scoping as `--spec-type`/`--spec-ngram-*`)
- [x] User-facing docs state which modes honor the feature (`docs/howto.md`, `README.md`)
- [x] ROADMAP §5 architectures: Llama-family wired; Phi-2/Phi-3/Qwen3/Qwen3-MoE named follow-up (same pre-existing gap Tier 9 already named — `forwardVerify` is shared machinery, not something this tier re-implements per architecture)

## Status (2026-09-18)

**Feature complete.** `DraftProposer` (new shared interface: `propose`/`observe`/`close`, both
`NgramDraftCache` and the new `DraftModelSession` implement it so `GenerationLoop.generate()`'s
draft/verify loop does not care which strategy is active) + `DraftModelSession` (drives a second,
independently-loaded single-shard `InferencePipeline` with its own persistent KV session, keyed
`kvKey + "#draft"`; `propose()` greedily self-decodes up to `--spec-ngram-m` tokens per round exactly
like a plain non-speculative decode step would; `observe()` reconciles that tentative continuation
against ground truth by walking forward from the last agreed position and issuing at most one
corrective `forward()` call per round — no bulk resend, no explicit KV truncation API needed, since
KV storage is indexed by absolute position and a later real write simply overwrites a stale
speculative one, the same overwrite-in-place semantics Tier 9's own verify window already relies on)
+ `SpeculativeDecodeOptions.SpecType.DRAFT_SIMPLE` / `--model-draft` (CLI + `JUNO_MODEL_DRAFT`,
fail-closed when set without a value) + `ConsoleMain.loadDraftPipeline()` (local single-shard load,
shares the target's `MatVec`/`GpuContext`, never carries LoRA adapters or vision wrapping).
`GenerationLoop`'s constructor fail-closes when `--spec-type draft-simple` is given without a loaded
draft pipeline, or when `draftPipeline.vocabSize() != pipeline.vocabSize()` — draft-proposed token ids
are compared directly against the target's own sampled ids, so a silent vocab mismatch would compare
incompatible id spaces. Because `GenerationLoop.generate()` always emits the target's own sampled
prediction (`emitted` equals `predicted` whether or not the draft matched, by construction — the same
invariant Tier 9 established), a bug in `DraftModelSession`'s reconciliation can only ever cost
acceptance rate, never emitted-token correctness — proven directly in
`DraftModelSessionTest` (propose/observe reconciliation, including a case that intentionally starves
the session of an `observe()` call between rounds) and `GenerationLoopSpeculativeDecodeTest` (two new
cases: full agreement and a scripted divergence between draft and target, both asserting token
identity against a non-speculative reference). Vocab-mismatch and missing-draft-pipeline fail-closed
paths are covered by dedicated `GenerationLoopSpeculativeDecodeTest` cases. Token-identity exit gate
(#1) additionally verified live: TinyLlama-1.1B (`--model-draft`) drafting for Mistral-7B (target,
same 32000-token Llama-family vocabulary) on a real GTX 1080, `--temperature 0`, produced
byte-identical output to `--spec-type none` for the same prompt — see `docs/perf-compare/README.md`
→ "Draft-model speculative decoding — regression gate + live smoke test" for the exact command.
Exit gate #2 ("TPS uplift documented when draft is small and acceptance is high; **failure cases
documented**") lands on its own explicitly-anticipated failure branch: acceptance was decent (55.2%,
53/96 drafted tokens) but wall-clock tg **regressed to 0.52×** `--spec-type none` (19.78 -> 10.35 t/s,
JFR) — `juno.MatVec.count` nearly quadrupled (7,965 -> 31,058) because `DraftModelSession.propose()`
drives the draft model through its own real transformer forward pass per drafted token, and on this
GPU those extra launches cost more than the verify-side savings, directly compounding the P0 gap
analysis's per-launch-overhead finding rather than contradicting it. Reported honestly per exit gate
#2's own wording, not hidden or spun — same standard Tier 9 held itself to for its more modest ~7%
result. The disable path (#3, `--model-draft` is never touched unless `--spec-type draft-simple` is
actually resolved) and vocab-mismatch fail-closed (#4) are both covered above. Docs and tests updated
(#5): `docs/howto.md`, `README.md`, `docs/agent-arch.txt`, `docs/performance.md` updated;
`scripts/run.sh`/`run.bat` forward `--model-draft`; `compare-llama-cpp.sh` forwards `JUNO_MODEL_DRAFT`
alongside the existing `JUNO_SPEC_TYPE`/`JUNO_SPEC_NGRAM_*` pass-through, per ROADMAP §2's "keep the
regression gate able to parse what it gates" rule.

**Named follow-ups** (not claimed as working, per ROADMAP §6): Phi-2/Phi-3/Qwen3/Qwen3-MoE
`forwardVerify` overrides (same pre-existing gap Tier 9 named — inherited, not reintroduced here);
vision combination untested; ROCm untested; cluster/tensor-parallel and LoRA train/play fail closed
by explicit CLI-parse-time error rather than silently ignoring `--model-draft`; the draft model's own
KV session is rebuilt from scratch on every `generate()` call, including repeated turns of the same
chat session, so it does not yet share the target's cross-turn prefix-cache reuse (a named perf cost
for long multi-turn sessions, not a correctness gap — each turn's draft session still self-heals
correctly via `observe()`'s reconciliation, it just re-primes instead of resuming); draft model device
placement always mirrors the target's shared `MatVec`/`GpuContext` — an independent `--gpu-layers`-style
knob for the draft model alone (e.g. forcing a large draft fully CPU while the target stays GPU-resident)
is not implemented in v1; the measured 0.52× wall-clock regression on the one live pair tested
(TinyLlama drafting for Mistral-7B) means `draft-simple` is not yet a net win on this hardware — closing
that gap is not scoped to this tier (it needs the same per-launch host/FFI overhead lever named in the
P0 decode-kernel deferral, not new draft-model logic) and a future session should not assume the flag
pays off without first re-measuring on the specific draft/target pair and hardware in question.
