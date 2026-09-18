# Tier 18: Tuned-Lane Bake-off Generalization + Recommended-Flags Docs

**Status: Feature complete (2026-09-18).** `run_mistral_tuned_lane` -> `run_tuned_lane` generalized
in `compare-llama-cpp.sh`; bake-off published at
[`docs/perf-compare/20260918T024641Z/`](../perf-compare/20260918T024641Z/); `docs/performance.md`
"Recommended flags" section and `docs/perf-compare/README.md` updated. See
`PLAN-Infra-ROADMAP.md`'s Tier 18 catalog row for the measured numbers.

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

- `PLAN-Infra-ROADMAP.md` — Execution rules §1-§6 (one tier in flight, perf-compare gate, no tier
  labels shipped, all-architectures rule, interaction-matrix rule)
- `PLAN-Infra-PERF-GAP-CLOSURE.md` — the session-level analysis this tier closes one item of
- `PLAN-Infra-Tier5.md` (`--gpu-layers auto`), `PLAN-Infra-Tier13.md` §§ `--mmq`, `--gpu-attention`
  (Phase C) — this tier composes their existing `auto` modes, does not change their semantics
- `scripts/performance-tests/compare-llama-cpp.sh` — `run_mistral_tuned_lane` (lines ~772-796),
  `MISTRAL_TUNED_LANE` flag (line 52), the per-model tuned-lane trigger (lines 973-974)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P0 step 7 (after Tier 17; reporting/docs, not a kernel change) |
| **Depends on** | Tier 5 (feature complete), Tier 13 Phase B/C (feature complete) — this tier only composes their already-shipped `auto` modes |
| **Blocks** | None |
| **Parallel with** | None — ROADMAP §1 allows only one Infra tier in flight; sequence after the active P0 decode-kernel work (`PROMPT-P0-Gate.md`) unless that work is paused |

## Overview

`compare-llama-cpp.sh --gpu` publishes its headline bake-off numbers with every optional speed flag
at its default (`--mmq` off, `--gpu-attention` off, `--gpu-layers` off) — see
`docs/perf-compare/20260916T043335Z/host.json`: `juno_mmq`, `juno_gpu_attention`,
`juno_gpu_layers` are all empty strings for the default rows. Those defaults are individually
correct and intentional (each flag has its own documented correctness/fit caveat — see Tier 5 and
Tier 13's interaction matrices), but the published default-vs-peer ratio (0.13-0.20x tg, 0.002-0.03x
pp in that run) materially understates what Juno already does when the shipped, tested `auto` modes
are used together: the same run's Mistral-7B **tuned lane** (`--mmq on --gpu-layers auto`, the one
model the script already auto-tunes) shows 0.41x tg vs 0.13x default, and the separately-published
`--gpu-attention on --prefill-batch 32` run shows pp 31.0 -> 119.6 t/s (3.85x) with attention's share
of prefill wall time dropping from 78.7% to 11.0% (`docs/performance.md:424-440`).

The script already has exactly this mechanism (`run_mistral_tuned_lane`) but it is hardcoded to one
model and two flags, added specifically to track the Tier 5 mistral-on-8GiB fit gate. This tier
generalizes that existing pattern rather than inventing a new one: every default-set model gets a
second, clearly-labeled row using the three flags' own `auto` mode together
(`--mmq auto --gpu-attention auto --gpu-layers auto`), and `docs/performance.md` gains a short
"recommended flags" section pointing at that exact combination in Juno's own terms.

**Explicitly not in scope:** flipping any flag's own default. `--mmq`, `--gpu-attention`, and
`--gpu-layers` keep their current `off`/none defaults. `auto` on each flag already resolves safely
per architecture/hardware (falls back to off/serial where unsupported, per Tier 5's and Tier 13's own
interaction matrices) — this tier only makes that combination visible and reproducible in the
standing regression gate and in user docs, it does not add a new code path.

## Scope and compatibility

Goals:

1. Generalize `run_mistral_tuned_lane` in `compare-llama-cpp.sh` into a model-agnostic
   `run_tuned_lane` that runs for every model in the default GPU set (TinyLlama, Qwen2.5-3B,
   Phi-3.5-mini, Mistral-7B), using `--mmq auto --gpu-attention auto --gpu-layers auto` — not the
   mistral-specific `--mmq on --gpu-layers auto`. Rename `--no-mistral-tuned-lane` to
   `--no-tuned-lane` (dev-only benchmarking flag, no shipped/public API — no back-compat shim per
   project convention); update the script's own `--help` text and `docs/perf-compare/README.md`'s
   references to the old name.
2. `INDEX.md`/`README.md` reporting shows the default row and the tuned row side by side per model,
   the same shape the mistral lane already produces, generalized to all four models.
3. `docs/performance.md` gains a "Recommended flags" subsection: state the measured tuned/default
   uplift per model from the published bake-off, name the three flags and their `auto` mode, and
   state the known caveats already documented per-flag (VRAM headroom for `--gpu-layers`, occasional
   greedy-decode divergence for `--gpu-attention`, architecture coverage gaps — Phi-3/Qwen3/ROCm
   follow-ups) in one place instead of scattered across tier docs. Juno terms only, no competitor
   names (ROADMAP §3).
4. Re-publish a full GPU bake-off under `docs/perf-compare/<timestamp>/` using the generalized
   script, confirming every model now has a default row and a tuned row, and that the tuned row's
   ratios for Phi-3.5-mini and TinyLlama/Qwen2.5-3B materially close the gap versus their default
   rows (directional confirmation of the same effect already proven for mistral and for
   `--gpu-attention` individually — this tier does not require a new perf floor, it requires the
   existing wins to be visible together, per model, in the standing artifact).

Non-goals:

- Any new CLI/env flag. No `--fast`/preset meta-flag — `--mmq auto --gpu-attention auto
  --gpu-layers auto` is already three existing, independently-tested flags; adding a fourth
  composing flag is unnecessary surface for what is fundamentally a reporting/docs change.
- Changing any flag's own default value or semantics.
- Any kernel or scheduling change (that is `PROMPT-P0-Gate.md`'s scope, tracked separately).
- CPU-side tuned lane (the tuned flags here are all GPU-only levers; CPU vector-path parity is
  `PROMPT-Vector-SIMD-Refresh.md`'s scope).

## Chosen design

Reuse `run_mistral_tuned_lane`'s existing shape (run Juno a second time with different flags under a
`-tuned` stem, reuse the already-fetched llama.cpp baseline JSON via `cp -a` rather than re-running
the peer benchmark, write a paired summary) — do not redesign the script's run/report plumbing.
Change only: (a) the flag set passed (`--mmq auto --gpu-attention auto --gpu-layers auto` instead of
`--mmq on --gpu-layers auto`), and (b) the trigger condition (every model in the default set, not
`base == mistral-7b*`).

`auto` resolution for `--gpu-attention` on Phi-3.5-mini (Phi-3 architecture) is a documented
**follow-up** in Tier 13's own interaction matrix — the tuned lane for that model will show `--mmq
auto` taking effect but `--gpu-attention` resolving to off (unsupported architecture), and that must
be stated plainly in the published row's notes, not silently implied as "fully tuned." This is the
same honesty standard Tier 17 already applied to its own long-window caveat.

## New/modified classes

None — this is a shell-script and docs change:

- **`scripts/performance-tests/compare-llama-cpp.sh`** (modify): rename/generalize
  `run_mistral_tuned_lane` -> `run_tuned_lane`; rename `MISTRAL_TUNED_LANE` -> `TUNED_LANE` and
  `--no-mistral-tuned-lane` -> `--no-tuned-lane`; change the trigger at the model-loop call site
  (lines ~973-974) from `"$base" == mistral-7b*` to "every model in `USE_GPU=1` mode"; change the
  flags passed from `--mmq on --gpu-layers auto` to `--mmq auto --gpu-attention auto --gpu-layers
  auto`.
- **`docs/performance.md`** (modify): new "Recommended flags" subsection.
- **`docs/perf-compare/README.md`** (modify): update the "Standing Mistral-7B tuned lane" section
  heading/prose to describe the generalized per-model tuned lane; add the new bake-off run row.

## Implementation

### 1. Generalize the script

Rename and widen `run_mistral_tuned_lane`/`MISTRAL_TUNED_LANE`/`--no-mistral-tuned-lane` per Scope
item 1. Keep the `cp -a` reuse-of-peer-baseline optimization (the peer engine's numbers do not change
with Juno's flags, no need to re-run `llama-bench`). Verify `--help` text and any other script
referencing the old flag/function names are updated together (grep the whole
`scripts/performance-tests/` tree, not just this file).

### 2. Re-run and publish

`compare-llama-cpp.sh --gpu` (default 4-model set) with the generalized tuned lane. Publish under
`docs/perf-compare/<timestamp>/`; update `docs/perf-compare/README.md` with the new run row and a
per-model default-vs-tuned summary table (mirroring the existing mistral-only table at
`docs/perf-compare/README.md:174-192`, generalized to all four models).

### 3. Docs

Add the "Recommended flags" subsection to `docs/performance.md` per Scope item 3, sourced from the
new bake-off's numbers. State the Phi-3.5/`--gpu-attention` architecture-coverage caveat explicitly.

## Feature × surface interaction matrix

No new CLI/env flag is introduced (Scope non-goal) — this tier only changes which existing,
already-matrixed flag combination the benchmark script exercises and reports. The relevant matrix
cells are unchanged from `PLAN-Infra-Tier5.md` (`--gpu-layers`) and `PLAN-Infra-Tier13.md`
(`--mmq`, `--gpu-attention`); this tier does not add rows. Recorded here per ROADMAP §6 to state
explicitly why a fresh matrix is not required, rather than leaving the section silently absent.

## Cross-feature smoke (before feature complete)

- [x] Generalized tuned lane runs and completes for all four default-set models on GPU without
      script errors (`failures=0`, `20260918T024641Z`).
- [x] Published `INDEX.md`/`README.md` rows clearly label which flags each tuned row used and note
      the Phi-3.5/`--gpu-attention` non-wired-architecture caveat for that model's row specifically
      (verified via JFR: 0 `juno.Attention` events for Phi-3.5-mini in both lanes).
- [x] `--no-tuned-lane` still skips the extra rows (renamed flag, same behavior) — trigger logic
      unchanged (`TUNED_LANE` gate), only the flag/var/function names and the per-model condition
      changed.
- [x] No LoRA/vision code path is touched by this tier (script/docs only) — §2 LoRA/vision compares
      are not required, recorded here as the explicit reasoning rather than a silent skip.

## Exit checklist (compatibility)

- [x] Interaction matrix section present with the "no new flag" reasoning (above), not left blank.
- [x] No silent flag ignore introduced — this tier reports on existing `auto` resolution, it does
      not change what any flag does.
- [x] `docs/perf-compare/README.md` and `docs/performance.md` updated per Implementation §2-3.
- [x] ROADMAP §5 architectures: tuned-lane row published for TinyLlama, Qwen2.5-3B (`qwen2`),
      Phi-3.5-mini (`phi3`), Mistral-7B (LLaMA-family) — the full default set, with Phi-3.5's
      `--gpu-attention` non-wired caveat stated, not silently implied covered.

## Verification and exit gate

**Global rules** (`PLAN-Infra-ROADMAP.md` -> Execution rules): only one Infra tier in flight at a
time; publish a `docs/perf-compare/` bake-off before marking this tier complete. LoRA/vision compares
are not required (no code path touched — script and docs only).

Exit only when:

1. Generalized tuned lane published for the full default 4-model GPU set, each row labeled with the
   exact flags used and any architecture-coverage caveats.
2. `docs/performance.md` "Recommended flags" section lands, sourced from the new bake-off, Juno terms
   only (ROADMAP §3), no Infra tier numbers (ROADMAP §4).
3. `docs/perf-compare/README.md` updated; old `--no-mistral-tuned-lane` references replaced.
4. ROADMAP catalog/phase-table status line for Tier 18 updated to feature complete.

## Implementation todos

1. Generalize `run_mistral_tuned_lane` -> `run_tuned_lane` in `compare-llama-cpp.sh`; rename flag.
2. Re-run `compare-llama-cpp.sh --gpu` (default set); publish under `docs/perf-compare/<timestamp>/`.
3. `docs/performance.md` "Recommended flags" section.
4. `docs/perf-compare/README.md` update; ROADMAP status; preview files listed, no zip.

## Preview files (expected)

Modified: `scripts/performance-tests/compare-llama-cpp.sh`, `docs/performance.md`,
`docs/perf-compare/README.md`, `docs/infra-plan/PLAN-Infra-ROADMAP.md`

New: `docs/perf-compare/<timestamp>/` bake-off artifacts
