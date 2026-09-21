# Juno gap-closure plan

Source material: [`../llama-cpp-gap-analysis.md`](../llama-cpp-gap-analysis.md) (2026-09-18 snapshot,
branch `67-inference`, HEAD `0c519f1`). That document is raw analysis; this tree turns it into a
sequenced, testable implementation plan. Re-verify any cited line number against current source
before acting on it — both documents are snapshots, not ground truth that stays accurate forever.

## Execution rules (binding for every tier)

1. **One tier at a time, to feature-complete.** Do not start Tier N+1 work until Tier N's exit
   criteria (bottom of its file) are all checked off. Partial, half-wired features are not
   acceptable stopping points between tiers.
2. **No surface left aside.** A tier is not complete until its change has been carried through
   every product surface it touches — see "Cross-surface compatibility checklist" below. If a
   surface can't reasonably support the new feature yet (e.g. continuous schedule doesn't support
   per-request LoRA), the tier must make that an explicit, documented, fail-closed rejection for
   that surface, not a silent gap. "Out of scope for this tier" is only acceptable when stated
   explicitly in the tier's Scope section and the surface fails closed rather than silently
   degrading.
3. **Tests first, and they must cover the real matrix.** For every tier: write or extend the unit
   and integration tests *before* writing the implementation. Before marking any tier complete, run
   the full smoke matrix (CPU, CUDA GPU, LoRA train+play, vision, static schedule, continuous
   schedule, single-node local, pipeline-parallel cluster, tensor-parallel cluster) across every
   architecture family Juno claims to support, using real model files, and confirm zero regressions
   against the prior tier's baseline. See "Test infrastructure" below for the concrete mechanics.
   When a tier needs a model file that isn't present under `models/`, stop and ask — see
   [`INVENTORY.md`](INVENTORY.md) for what's already on disk and what's missing.
4. **No competitor product names in Juno-facing docs.** Never write `llama.cpp`, `vLLM`,
   `llama-server`, or any other competitor product name into anything outside this
   `docs/gap-closure-plan/` tree — not `docs/howto.md`, `docs/agent-arch.txt`, `README.md`,
   `CHANGELOG.md`, `docs/performance.md`, code, comments, CLI help text, or error messages. This
   plan tree itself may name them (it exists to compare against them). Filenames/script names that
   already exist under `docs/perf-compare/` and `scripts/performance-tests/` (e.g.
   `compare-llama-cpp.sh`) are pre-existing and out of scope for renaming; don't add new ones.
5. **The plan tree is self-contained.** When a tier's work requires updating production docs
   (`docs/howto.md`, `docs/agent-arch.txt`, `README.md`, `docs/performance.md`), those docs must
   read standalone — do not add "see `docs/gap-closure-plan/TIER-NN-...`" pointers into them. Only
   `docs/perf-compare/` entries may point back into `docs/performance.md` (existing convention);
   nothing may point into this plan tree from outside it.
6. **Cross-feature / product-surface compatibility.** Every tier's exit criteria include a
   cross-feature smoke table (same format `docs/performance.md` already uses for prior tiers) that
   exercises the new feature *combined with* the other major surfaces (LoRA play, vision, grammar,
   tools, cluster/TP, static/continuous), not just in isolation. A feature that only works alone is
   not feature-complete.

## Tier index and ordering rationale

Ordering: correctness-and-consistency first, then the shared architectural root cause behind three
independent measured regressions, then outward through the feature surface in the order a request
actually flows (attention/context → KV → quantization → sampling → speculative decoding →
scheduling → model coverage → parallelism → backend breadth → vision → LoRA → server/cluster
surface), closing with a documentation hardening pass.

| Tier | Title | Gap analysis refs |
|---|---|---|
| [00](TIER-00-correctness-and-consistency.md) | Correctness & consistency audit | §2.1, §2.4, §2.5, §2.6, §2.7, §2.9 |
| [01](TIER-01-gpu-activation-residency.md) | GPU activation-residency redesign | §1.6, §2.8 |
| [02](TIER-02-attention-long-context.md) | Attention & long context | §1.2 |
| [03](TIER-03-kv-cache-maturity.md) | KV cache maturity | §1.3 |
| [04](TIER-04-quantization-coverage.md) | Quantization coverage | §1.1 |
| [05](TIER-05-sampling-grammar.md) | Sampling & grammar completeness | §1.4 |
| [06](TIER-06-speculative-decoding.md) | Speculative decoding expansion | §1.5 |
| [07](TIER-07-continuous-batching.md) | Continuous batching maturity | §1.3 (scheduling half) |
| [08](TIER-08-model-architecture-breadth.md) | Model architecture breadth | §1.9, real files in `models/` |
| [09](TIER-09-tensor-parallelism-multi-gpu.md) | Tensor parallelism & multi-GPU | §1.8, §2.2 |
| [10](TIER-10-gpu-backend-breadth-cpu-simd.md) | GPU backend breadth & CPU SIMD | §1.7 |
| [11](TIER-11-vision.md) | Vision | §1.10 |
| [12](TIER-12-lora.md) | LoRA | §1.11 |
| [13](TIER-13-server-surface-clustering.md) | Server surface & clustering | §1.12, §2.3, §2.6 |
| [14](TIER-14-documentation-hardening.md) | Documentation hardening | §3 |

Also see [`INVENTORY.md`](INVENTORY.md) for the model/hardware inventory referenced by every tier.

## Cross-surface compatibility checklist (the rubric every tier applies)

Every tier's exit criteria must state, for each row below, either **PASS** (verified, cite the
smoke run), **N/A** (the tier's change genuinely cannot touch this surface — state why), or
**FAIL-CLOSED** (the surface correctly rejects the new capability with an explicit error, and that
rejection is itself tested).

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference (scalar) | `--gpu-layers 0` |
| 2 | CUDA GPU inference | GTX 1080 available locally |
| 3 | ROCm GPU inference | no AMD hardware available locally — see [`INVENTORY.md`](INVENTORY.md) for how this is gated |
| 4 | Static schedule | default schedule, `--parallel` micro-batching |
| 5 | Continuous schedule | `--schedule continuous`, local mode only today |
| 6 | Single-node local mode | `./juno local` |
| 7 | Pipeline-parallel cluster | `./juno cluster --pType pipeline` |
| 8 | Tensor-parallel cluster | `./juno cluster --pType tensor` |
| 9 | LoRA training | `./juno lora` |
| 10 | LoRA playback | `--lora-play` |
| 11 | Vision | `--mmproj-path`, `/v1/vision/chat`, local mode only today |
| 12 | OpenAI-compatible REST surface | `/v1/chat/completions`, streaming, tools, grammar |
| 13 | Native REST surface | `/v1/inference`, `/v1/inference/stream` |
| 14 | CLI direct usage | `./juno local`/`cluster`/`lora`/`merge`/`lora-import`/`gguf-info`/`test` |

## Test infrastructure

Three layers, all of which get extended (never replaced) tier over tier:

1. **Unit tests** in the owning module (`mvn test -pl <module>`). Each tier's file names the exact
   test classes to add cases to or create.
2. **Integration tests**: `ModelLiveRunnerIT` (`juno-master/src/test/java/cab/ml/juno/master/`,
   run via `./juno test --model-path ...`, 8 checks today — 6 pipeline-parallel + 2 tensor-parallel)
   and the forked-JVM cluster ITs (`ThreeNodeClusterIT`, `TensorParallelClusterIT`, `mvn verify -pl
   juno-master`). Every tier that changes forward-pass, batching, or cluster behavior adds a new
   check to `ModelLiveRunnerIT` rather than only relying on unit tests — this is the one test class
   that runs against a real model file end to end.
3. **Bash smoke scripts** under `scripts/performance-tests/`, following the existing
   `smoke-grammar.sh` / `smoke-tools.sh` convention: one `smoke-tierNN-<short-name>.sh` per tier,
   written in bash, that drives `./juno local`/`cluster` against the real model matrix (see
   `INVENTORY.md`) and asserts on process exit code, HTTP status, and response shape — not just
   "does it crash," but "is the output the specific thing this tier promised." Each smoke script
   must be runnable standalone and re-run (unmodified) by every later tier as a regression check —
   later tiers only ever *add* a new `smoke-tierNN` script, never edit an earlier one except to fix
   a bug in the script itself.

Performance regression gates stay governed by the existing rule in the root `CLAUDE.md`
("Performance gates on hot-path changes"): any tier touching the forward pass, MatVec, GPU
residency, batching, or KV paths runs `scripts/performance-tests/compare-lora.sh` (and
`compare-vision.sh` if vision/CLIP is touched) against the last published baseline in
`docs/perf-compare/`, and publishes a new timestamped result directory there — that mechanism
already exists and this plan doesn't change it, it just says explicitly, per tier, when it applies.

## Model and hardware inventory

See [`INVENTORY.md`](INVENTORY.md). Summary: one NVIDIA GTX 1080 (CUDA) is the only GPU available
in this environment; there is no ROCm/AMD hardware. Tiers that touch ROCm code paths are
implemented and unit-tested to the extent possible without hardware, and their exit criteria
include an explicit `FAIL-CLOSED` or `NEEDS-AMD-HARDWARE` marker rather than a false `PASS` for row
3 of the compatibility checklist — flag this to the user when a tier reaches that point rather than
guessing at real-hardware behavior.

## Feature-complete definition

A tier is feature-complete only when **all** of the following hold, not just "the happy path
works":

- Every row of the cross-surface compatibility checklist for that tier's feature is PASS, N/A (with
  reason), or FAIL-CLOSED (tested).
- All new/extended tests from layer 1-3 above pass, and all pre-existing tests still pass
  (`mvn test` across all unit-test-bearing modules, `mvn verify -pl juno-master`).
- Any hot-path change has a published `docs/perf-compare/` entry per the existing performance-gate
  rule, with no unexplained regression.
- `CHANGELOG.md` gets an entry describing what shipped, in the project's existing style.
- `docs/agent-arch.txt`, `docs/howto.md`, and `README.md` are updated if the change is user-facing
  or architectural (existing `CLAUDE.md` rule), using Juno-native language only (rule 4 above).
- No new dead/dormant/unwired scaffolding is left behind without an explicit tracking note in that
  tier's file explaining why it's dormant and what would need to happen to wire it in (the pattern
  flagged in gap-analysis §2.8 — don't repeat it silently).
