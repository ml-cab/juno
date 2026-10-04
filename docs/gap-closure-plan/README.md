# Juno gap-closure plan

Source material: [`../llama-cpp-gap-analysis.md`](../llama-cpp-gap-analysis.md) (2026-09-18 snapshot,
branch `67-inference`, HEAD `0c519f1`). That document is raw analysis; this tree turns it into a
sequenced, testable implementation plan. Re-verify any cited line number against current source
before acting on it — both documents are snapshots, not ground truth that stays accurate forever.

## Execution rules (binding for every tier)

1. **One tier at a time, to feature-complete.** Do not start the next tier's work until the current
   tier's exit criteria (bottom of its file) are all checked off. "Next" means the next row of the
   tier index below, not the next integer — the index is the running order. Five rows carry a
   non-integer number or sit out of integer order: Tier 01B sits between 01 and 02, Tier 01C between
   01B and 02, Tier 04B between 04 and 05, Tier 04C between 04B and 05, and Tier 08 runs before
   Tier 06. Only the last of those actually breaks the order — 01B, 01C, 04B and 04C read in
   sequence — but all five are enumerated here because this list, not the numbering, is what a
   reader builds the running order from, and 04C and 01C were added to the tree after the first
   three were written. Partial, half-wired features are not acceptable stopping points between
   tiers.
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
7. **Performance thresholds are numeric, and anchored to llama.cpp where the tier is performance-
   relevant.** Every tier's perf-gate bullet states a concrete pass/fail number for whatever new
   metric it introduces — "measured, published, no unexplained regression" alone is not enough for a
   metric with no prior baseline to be implicitly anchored to. See "Test infrastructure" below for
   the specific llama.cpp-relative gate this adds on top of the existing `compare-lora.sh` rule.
   **This rule is machine-checked, not trusted, and the check does not wait for Tier 14.** It lives in
   `scripts/performance-tests/check-plan-thresholds.sh` (shipped 2026-09-27, recorded in Tier 01B's
   execution record), reads only this tree, and fails on any of:
   - a tier file mentioning a perf gate in any capitalization (`Perf gate`, `perf gate`, `Perf gate
     (required)` are all in use) without a `**Threshold` block carrying a numeral and a comparison
     operator. A tier with no gate says so with an explicit `**No perf gate**` declaration and its
     reason, which exempts it; a lowercase mention no longer slips past the match;
   - an exit criterion reading "no unexplained regression", the phrasing this rule replaced;
   - an intermediate-milestone row (see "Program target") whose threshold is not `>= number`, or an
     `active` milestone its reference reading already meets. A milestone met before its tier starts
     measures nothing; it is either raised or marked `retired`, and a retired row must really be met;
   - *(check 4, specified 2026-09-30, shipped 2026-10-01 by [Tier 01B](TIER-01B-prefill-throughput.md) item 9)*
     a **ticked** exit criterion (`- [x]`) that states a threshold — it contains `Threshold`, `perf
     gate`, `>=`, `<=` or `≥`/`≤` — but whose criterion text, including its indented notes, cites no
     `docs/perf-compare/<dir>` (or `../perf-compare/<dir>`) that exists on disk. The first three checks
     prove a number was written down before implementation; this one proves something was scored
     against it before the box was ticked. A criterion scored on a non-published reading (a unit test,
     an unpublished A/B) names that evidence with an explicit `**Evidence (not published):**` marker,
     which exempts it, so the exemption is visible rather than silent.

   Tier 14's own file is excluded from the first check, since it names the string to describe it. The
   script exits non-zero listing every failure; it passes against this tree and was shown failing
   against the tree as it stood before the fixes that made it pass (seven failures: Tiers 04, 04B, 04C
   and 05 and all three milestone rows).

   **Every tier from 01B on runs it as the first item of its own test list.** It was originally
   specified inside Tier 14's `smoke-doc-consistency.sh`, which meant a rule governing seventeen
   tiers was enforced by nothing until the last of them — the same deferral shape this rule exists to
   prevent. Tier 14 keeps the check as one input to its doc-consistency script rather than as its
   origin: that script calls this one instead of restating it, so there is one implementation and one
   place a spelling can drift. This paragraph first said the script would ship "in the next tier to
   execute"; Tier 01 closed without it, which is recorded here rather than smoothed over.

   The rule was added once and immediately drifted — at the time this enforcement was written only six
   of seventeen tier files stated a threshold at all, and Tiers 11 and 12 both declared a required perf
   gate with no number. It has since been brought into compliance by hand across every tier file, which
   is exactly the state a check is needed to hold rather than to establish.
8. **Doc/comment claim audits are repo-wide, not scoped to a named list of files.** When a tier's job
   is to fix a doc/code drift item (Tier 00's §2.x items, Tier 14's full audit), the fix is not
   complete until the same class of claim has been grepped for across `src/main` and `CHANGELOG.md`,
   not just in whichever file the gap analysis happened to cite. A rule stated once in `CLAUDE.md`
   (e.g. "no internal tier numbers in shipped code") is binding everywhere that rule applies, not
   just in the specific files a prior audit pass already looked at.
9. **Work that lands outside a tier is recorded against the active tier.** A defect found and fixed
   between tiers — or any change to the forward pass, MatVec, KV, batching, quantization or the
   launcher that is not in the active tier's scope — is recorded in that tier's execution record
   under an **Out-of-tier changes** heading, naming the commit, what it touched, whether it is a
   measurement boundary, and which published baselines it invalidates. Tier 14's scorecard lists
   them alongside the tiers. The point of this plan is that it knows what was measured, when, and
   against which build; an unrecorded hot-path commit is precisely the change that breaks that
   property. This rule exists because it was already violated twice. Commit `c91f879` retired a
   device KV mirror by closing it in place, gated attention on a written-prefix watermark, and added
   `DeviceScratchBudget`/`Q4KDequantScratch` work plus three published `compare-lora.sh` runs;
   commit `1f90b68` changed both launchers to derive the JVM heap from the model file size. Both
   landed while Tier 01 sat at eight unticked exit criteria, and neither was recorded in any tier
   file. Both are now in Tier 01's execution record. A third occurrence should not need a new rule.

## Program target

This plan's stated purpose is closing the gap with llama.cpp. That purpose needs a number, or no
tier can fail on it and the final scorecard in Tier 14 can only report a direction of travel.

**Program exit target**, measured by `scripts/performance-tests/compare-llama-cpp.sh` on the standing
four-model sweep (`tinyllama-1.1b`, `qwen2.5-3b`, `Phi-3.5-mini`, `mistral-7b`, all Q4_K_M) on the
`docs/perf-compare/README.md` baseline host, under the benchmark-parity preconditions below:

| Metric | Target at end of plan | Reading when this plan was written | Parity-corrected reading (2026-09-25) | After the CPU RoPE table (2026-09-27) | Reference 2026-09-27, late | Reference 2026-09-30 (pinned GPU re-baseline) | Reference 2026-10-01 (pinned CPU) | Reference 2026-10-03 (Tier 01B closing sweeps, pinned) | **Current reference** (2026-10-04: Tier 01C closing sweeps, pinned) |
|---|---|---|---|---|---|---|---|---|
| GPU tg, Phi-3.5-mini | >= **0.70x** (was 0.50x, met 2026-09-30) | 0.330x | 0.423x | 0.415x (0.414x tuned) | 0.416x (0.417x tuned) | 0.394x (0.396x tuned) | 0.394x (0.396x tuned; GPU unchanged) | 0.535x (0.537x tuned) | **0.545x** (0.547x tuned) |
| GPU tg, mistral-7b | >= **0.70x** (was 0.60x, met 2026-09-27) | 0.513x | 0.581x (0.631x tuned) | **0.646x** (0.638x tuned) | 0.639x (0.643x tuned) | 0.610x (0.610x tuned) | 0.610x (0.610x tuned) | 0.611x (0.607x tuned) | **0.627x** (0.619x tuned) |
| GPU pp, every sweep model | >= **0.25x** (was 0.15x) | see the caveat below — not 0.016x to 0.031x | 0.036x to 0.073x, median 0.045x | 0.039x to 0.091x, median 0.051x | 0.040x to 0.101x, median 0.064x (Phi-3.5-mini 0.040x binding) | 0.037x to 0.101x at `n_prompt=128`, median 0.067x; 0.011x to 0.098x at 512 (Phi-3.5-mini binding at both) | 0.037x to 0.101x at `n_prompt=128`; 0.011x to 0.098x at 512 (GPU unchanged) | 0.173x to 0.299x at `n_prompt=128`, median 0.289x; 0.146x to 0.285x at 512, median 0.268x (Phi-3.5-mini binding at both) | **0.303x to 0.729x** at `n_prompt=128`, median 0.570x; **0.226x to 0.468x** at 512, median 0.373x (Phi-3.5-mini binding at both); **0.099x to 0.180x** at 2048, median 0.138x (Phi-3.5-mini binding) |
| CPU tg, every sweep model | >= **0.25x** (placeholder; Tier 10 restates it against the bandwidth roofline, never lower) | 0.106x to 0.147x | 0.079x to 0.128x | 0.090x to 0.135x | 0.090x to 0.135x (CPU not re-measured) | 0.092x to 0.121x (mistral-7b binding; pp 0.049x to 0.103x, Phi-3.5-mini binding) | 0.092x to 0.121x (CPU not re-measured) | **0.092x to 0.121x** (CPU not re-measured) |

**Score every later tier against the right-most column.** Its GPU rows are Tier 01C's closing sweeps
(2026-10-04, owner run, clocks pinned, HEAD `59c53cc` plus Tier 01C's two out-of-tier fixes, jar
`5ab4c78c505a4cef`): `docs/perf-compare/20261004T113210Z/` (`n_prompt=128`) and `20261004T114812Z/`
(`n_prompt=512`), same method and harness as the column to its left, so ratios compare across the two
directly; every row scorable, every prefill at its `n_prompt`. Its CPU row is unchanged (no CPU path
changed). Its 2048 figures are from the 2048 re-run on the same code (jar `a90122b2855bb826`, which adds two
fixes that do not touch throughput: an out-of-heap request fails instead of hanging, and prompt encoding
is recorded so the harness's span check can subtract it): `docs/perf-compare/20261004T220015Z/`
(TinyLlama, Qwen2.5-3B, Mistral 7B at their fixed heaps) and `20261004T222758Z/` (Phi-3.5-mini at an
explicit 8 GiB heap, since its 2048-token live set of about 6.1 GB does not fit the fixed 6 GiB). The
interrupted first attempt, `20261004T120622Z-partial`, is superseded by them. Made the reference by
owner decision (Tier 01C decision 6).

**The column to its left** was the reference until then. Its GPU rows are Tier 01B's closing sweeps
(2026-10-03, owner run, clocks pinned, HEAD `1ac490a`): `docs/perf-compare/20261003T042440Z/`
(`n_prompt=128`) and `20261003T044014Z/` (`n_prompt=512`), same method and harness as the 2026-09-30 GPU
re-baseline, so ratios compare across the two directly; every row scorable, every prefill at its
`n_prompt`. Its CPU row is unchanged from the column to its left, described next, which was the reference
until then. **That column's** CPU row is Tier 01B step 3a's pinned
CPU sweep (2026-10-01, `docs/perf-compare/20261001T180241Z/`): clocks pinned, Juno's common-pool
parallelism set from `--threads` so its kernels run the reference tool's `-t 12`. Pinning is a
measurement boundary for CPU readings (both engines' absolute t/s fell; tinyllama's reference prefill
by 16%), so compare CPU ratios across it, never absolute t/s. Its GPU rows are unchanged from the
column to its left, Tier 01B's step 2 re-baseline (2026-09-30): GPU `docs/perf-compare/20260930T135225Z/` (`n_prompt=128`) and `20260930T141026Z/`
(`n_prompt=512`), both with pinned clocks, the operating-system JFR clock and the warm-up recording;
CPU unchanged from `20260927T094414Z/`. It is a measurement boundary against every column to its left
(turbo off lowers both engines' absolute throughput, and the harness changes of 2026-09-28 and
2026-09-29 apply), so compare ratios across it, never absolute t/s. Tier 01B's execution record derives
it. The column to its left, described next, was the reference until then.

The previous reference was from late 2026-09-27:
GPU `docs/perf-compare/20260927T232837Z/` with the `qwen2.5-3b` tuned and both `Phi-3.5-mini` rows
from its re-run `20260927T234659Z/`; CPU `20260927T094414Z/`, which the change behind it did not touch.
It follows the per-request device-memory fix and the extracted FP16 pack loop, a GPU-prefill
measurement boundary recorded in Tier 01's "Out-of-tier changes" (GPU pp +10% to +34% on tinyllama,
qwen2.5-3b and mistral-7b; generation within the gate). The column to its left was the reference for
most of that day and is now superseded; it is kept, not overwritten, for the same reason the older
columns are.

**Rule: the reference column moves in the same change that supersedes it.** Whenever a run is marked
superseded-as-reference in `docs/perf-compare/` (its `INDEX.md` banner and `docs/perf-compare/README.md`),
the same change adds the new column here and names it as the one to score against. The 2026-09-27
column went stale for a day because this rule did not exist: Tier 01's out-of-tier table superseded
its source while this table still told every later tier to score against it.

**The 2026-09-27 (RoPE table) column** was taken after the scalar CPU RoPE stopped
recomputing its angles for every head of every layer, which is a measurement boundary (Tier 01's
"Out-of-tier changes"). Sources: GPU `docs/perf-compare/20260927T091155Z/` with the `tinyllama`,
`qwen2.5-3b` and `Phi-3.5-mini` rows from its re-run `20260927T093054Z/`; CPU `20260927T094414Z/`.
Juno's GPU throughput rose where the rotation runs on the CPU path (tinyllama tg +26%, mistral-7b
+17%, qwen2.5-3b +12%; pp +22% to +35%) and held on Phi-3.5-mini, which has its own rotation; its
ratio moved only with the reference tool's own reading. CPU throughput did not move (the rotation was
never a visible share of a CPU forward pass); the CPU ratios moved with the reference tool's readings.
The mistral-7b GPU tg row met its end-of-plan target on the default flags from this column on.

**The 2026-09-25 column was the reference until the morning of 2026-09-27, and the two columns
before it are not comparable with it.**
Those readings come from `docs/perf-compare/20260925T172231Z/` (GPU) and `20260925T174146Z/` (CPU),
the first sweep taken with prompt-token parity, warm repeated measurement, a fixed heap, `min_tokens`
generation parity, and prefill and generation measured in two separate runs as the reference tool
measures them. Tier 01's execution record derives them and explains each correction. The generation
figures moved up mainly because the measurement stopped penalizing Juno, not because the engine got
faster — score a later tier against the right-hand column, never against the middle one.

**The pp row's starting reading is not yet known, and the target is provisional until it is.** The
0.016x-to-0.031x figures widely quoted from `docs/perf-compare/20260918T204809Z/` were taken with
`raw_prompt: 0`, meaning llama-bench prefilled 128 tokens while Juno prefilled 20 to 30 — they are
not a like-for-like prefill measurement (see "Benchmark parity preconditions" below). The only
parity-corrected prefill figures on record are worse and narrower: at a real 512-token prompt,
`20260915T043143Z` puts mistral-7b at 0.0166x and tinyllama at 0.0075x, and the best GPU prefill
number anywhere in this repository — TinyLlama at 119.56 t/s with `--gpu-attention on` and
`--prefill-batch 32`, `20260916T040113Z-prefill/` — is about **0.028x** of llama.cpp's 4225 t/s on
the same shape. Tier 01's re-baseline establishes the real starting reading under parity; **the
0.15x target is re-derived from that number in Tier 01's own file before any later tier is scored
against it.** A target anchored to a measurement that was never like-for-like is not a target.

**Intermediate milestones**, so the trend is checkable before Tier 14 rather than only at the end.
This table is machine-read by `scripts/performance-tests/check-plan-thresholds.sh` (execution rule 7):
every threshold is `>= number`, every reference is a number or `unmeasured`, and an `active` row must
ask for more than its reference reading. For an "every model" scope the reference is the binding
(lowest) model's reading, named in the cell.

| After | Metric | Scope | Threshold | Reference reading | Status |
|---|---|---|---|---|---|
| 01B | GPU pp ratio, `n_prompt=512` | every sweep model except Phi-3.5-mini | >= 0.10x | 0.254x (tinyllama, binding; qwen2.5-3b 0.281x, mistral-7b 0.285x; Tier 01B closing sweep `20261003T044014Z`; was 0.064x at the step 2 re-baseline) | retired: met by Tier 01B |
| 01B | GPU pp ratio, `n_prompt=512` | Phi-3.5-mini | >= 0.08x | 0.146x (Tier 01B closing sweep `20261003T044014Z`; was 0.011x at the step 2 re-baseline; threshold kept by owner decision, see below) | retired: met by Tier 01B |
| 01C | GPU pp ratio, `n_prompt=512` | every sweep model | >= 0.20x | 0.226x (Phi-3.5-mini, binding; tinyllama 0.317x, qwen2.5-3b 0.429x, mistral-7b 0.468x; Tier 01C closing sweep `20261004T114812Z`; was 0.146x at Tier 01B's close) | retired: met by Tier 01C |
| 02 | GPU pp ratio at 512 over ratio at 128 | every sweep model | >= 1.00 | 0.606 (tinyllama, binding; mistral-7b 0.642, qwen2.5-3b 0.697, Phi-3.5-mini 0.743; Tier 01C closing sweeps `20261004T113210Z`/`20261004T114812Z`; was 0.845 at Tier 01B's close: the packed matmul removed fixed per-token cost, so attention's share of a long window grew) | active |
| 02 | GPU pp ratio at 2048 over ratio at 512 | every sweep model | >= 0.90 | 0.329 (mistral-7b, binding; tinyllama 0.381, qwen2.5-3b 0.420, Phi-3.5-mini 0.441; Tier 01C's 2048 sweeps `20261004T220015Z`/`20261004T222758Z` over its 512 sweep `20261004T114812Z`; Phi-3.5-mini at an explicit 8 GiB heap, see Tier 01C's record) | active |
| 04 | GPU tg ratio | Phi-3.5-mini | >= 0.40x | 0.416x | retired: met on arrival |
| 10 | CPU tg ratio | every sweep model | >= 0.20x | 0.092x (mistral-7b, `20261001T180241Z`; was 0.090x) | active |
| 10 | CPU pp ratio, `n_prompt=128` | every sweep model | >= 0.10x | 0.049x (Phi-3.5-mini, `20261001T180241Z`; the others 0.075x to 0.103x; was 0.048x at `20260927T094414Z`) | active |

**Why the 512-over-128 row moved from Tier 01B to Tier 02 (2026-10-01, owner decision).** Tier 01B's
step 5 decomposition showed that no item in that tier can meet it. The term that grows with prompt length
is attention: kernel, copies and host part go from 5.6% to 8.8% of a 128-token window to 11.4% to 17.3% of a
512-token one (`docs/perf-compare/20261001T225929Z/`, `20261001T224724Z/`). Tier 01B's items remove
costs that are fixed per token, which raises attention's share of what remains, so the row is projected
to get worse as they land (Phi-3.5-mini 0.916 to about 0.87). The mechanism that moves it is the
tiled long-context kernel, which is Tier 02's, and Tier 02's 2048-over-512 row measures the same cause
one step further out. Moved rather than kept as a known Tier 01B miss, so Tier 01B is scored on what its
own items can move. Threshold unchanged.

**Why the CPU pp row exists (added 2026-09-30).** CPU prefill is 0.048x to 0.090x of the reference
tool and was owned by nothing: Tier 01B scopes it out to "re-measure at the end of Tier 10", and Tier
10 had no prefill number. 0.10x is roughly a 2x move on the binding model (Phi-3.5-mini, whose CPU
attention is scalar and quadratic), and about 1.1x to 1.3x on the other three. It sits on Tier 10
because the costs it turns on — allocation per matmul, the common-pool dispatch, and the SIMD kernel
at prefill batch widths — are that tier's items 3 to 5.

**Why the Tier 01B milestone changed shape (2026-09-27).** It read "GPU pp >= 0.10x on mistral-7b".
Under the current reference mistral-7b already reads 0.101x at `n_prompt=128` — the out-of-tier
memory fix and the extracted pack loop moved it there — so a mistral-only milestone would have been
credited to Tier 01B for work that landed before the tier began. The binding model is Phi-3.5-mini at
0.040x, which runs a handler the GPU attention kernel does not reach today; Tier 01B item 0 is what
changes that. So the milestone now covers every sweep model: 0.10x for the three whose handler already
runs the kernel (a 1.6x move for the binding one, tinyllama at 0.062x), and 0.08x for Phi-3.5-mini (a
2.0x move). The references are `n_prompt=128` readings because no parity-corrected 512 reading exists
yet; Tier 01B's step 2 re-baseline takes it, and if the 512 figures sit materially below the 128 ones
(the degradation this milestone's third row is about), Tier 01B restates the first two rows against
them in its own file and here.

*Re-read 2026-09-30 against the step 2 re-baseline; owner decision: neither row restated.* The non-Phi
512 reading (0.064x, qwen2.5-3b) is not materially below 0.062x. Phi-3.5-mini's (0.011x) is a 3.6x
collapse from scalar attention, which Tier 01B item 0 removes; restating it at 2.0x (0.022x) would be
met by that fix alone. So 0.08x stands, now a 7.2x move. The restatement rule applies to a modest dip,
not to a collapse that an item in the same tier removes.

**Why the Tier 04 milestone is retired.** GPU tg on Phi-3.5-mini reads 0.416x before Tier 04 starts,
and nothing in Tier 04 touches that model's decode path (its Q4_K_M weights already take the packed
MMQ route). A milestone met before its tier begins, by work its tier does not contain, measures
nothing. It is kept as a retired row so the history stays visible. Phi-3.5-mini's decode is still the
binding constraint on the 0.50x end-of-plan target; giving that a milestone belongs with whichever tier
takes non-Llama decode residency, which no tier owns yet. *(2026-09-30: Tier 01B item 0 put GPU
attention on Phi-3's decode path and its tg ratio reached 0.534x, meeting that 0.50x target; the target
is now 0.70x, see "End-of-plan targets raised" below.)*

*2026-09-30: kept retired (owner decision).* The pinned re-baseline (`20260930T135225Z`) reads Phi-3.5-mini
GPU tg at 0.394x, below 0.40x. No Phi-3.5-mini decode code changed between the two readings, the 5% move
sits inside the 15% noise floor, and pinning (turbo off) changes both engines' clocks. The reference cell
keeps the 0.416x the row was retired on. Reactivating it would gate on noise, in a tier whose work cannot
move the number.

The GPU pp milestone is still the least certain number in this table. It is stated at that level
deliberately because prefill is the dominant gap, not because it is known to be reachable. Missing it
and reporting the miss plainly is an acceptable outcome for this plan; not having a number to miss is
not.

**End-of-plan targets raised (2026-09-30, owner decision after the plan review).**

| Row | Was | Now | Why |
|---|---|---|---|
| GPU tg, Phi-3.5-mini | >= 0.50x | **>= 0.70x** | Met at 0.534x by Tier 01B item 0 (`20260930T211637Z`). By this plan's own rule a target met before the work that is meant to reach it measures nothing |
| GPU tg, mistral-7b | >= 0.60x | **>= 0.70x** | Met since the 2026-09-27 column and 0.610x at the current reference; same reason |
| GPU pp, every sweep model | >= 0.15x | **>= 0.25x** | 0.15x was set before anyone knew where prefill time goes. The roofline below puts the FP32-compute ceiling at roughly 0.5x to 0.7x on mistral-7b, and Tier 01B item 6 plus Tier 01C are the two items aimed at it. 0.25x is still well under that ceiling |

The tg levers for 0.70x are the ones already in the plan, now with a number to be read against: attention
inside the decode residency region and CUDA graph replay (Tier 02 items 4 and 5), the `--gpu-residency`
default (Tier 02 item 6), and the residency region reaching the Phi-3 and Qwen3 handlers (no tier owns
that yet; Tier 02 records whether its items reach 0.70x without it, and names the gap if not). The pp
path is the milestone ladder above: 0.10x after Tier 01B, 0.20x after Tier 01C, and no fall-off from
128 to 512 or from 512 to 2048 tokens after Tier 02. The previous targets stay recorded in the table's first column so
Tier 14 can report both.

**Program objective: the layer runs on the device (added 2026-09-30).** Every GPU tier in this plan
builds a piece of one design, and it is stated here so the pieces are read against it: on a GPU run, a
transformer layer executes on the device at both decode and prefill width, and the activation crosses
to the host only where something that is not on the device needs it — the LM head and sampling, and
the gRPC boundary between cluster nodes. The pieces and their owners: the decode region through RoPE
(Tier 01, done), the prefill window's operands and elementwise work on the device (Tier 01B items 2 and
6), the packed GEMM at prefill width (Tier 01C), attention and then the rest of the layer inside the
decode region, on the LLaMA family, Phi-3 and Qwen3 (Tier 02 items 4 and 8), and the handlers that have
no device path today (Tier 08 item 6). The measurable form is copy counts per generated token and per
prefill window, read off `juno.DeviceStaging`: Tier 02's exit criterion holds decode to one upload and
one download per layer and a prefill window to none inside it; one crossing per forward pass is the
objective. Tier 14's scorecard reports the copy counts per handler at the end of the plan next to the
ratios, so how close the plan got to the objective is a number, like everything else in it.

**CPU tg: the memory-bandwidth roofline (added 2026-09-30).** CPU decode reads every weight once per
token, so its ceiling is memory bandwidth divided by the bytes read per token. This host runs four
populated DDR3 channels (eight 8 GiB DIMMs, two per channel, read from the EDAC controller): about 51 to
60 GB/s peak depending on the DIMM speed, which is not recorded; take about 40 GB/s as attainable until
Tier 10 measures it. Using the file size as the bytes read per token (an overestimate: not every
embedding row is read):

| Model | Weights per token | Roofline tg at 40 GB/s | Reference tool tg (`20260927T094414Z`) | Juno tg | Juno's share of the roofline |
|---|---|---|---|---|---|
| tinyllama-1.1b | 0.67 GB | about 60 t/s | 25.2 (about 42%) | 3.39 | about 6% |
| qwen2.5-3b | 2.10 GB | about 19 t/s | 12.1 (about 64%) | 1.14 | about 6% |
| Phi-3.5-mini | 2.39 GB | about 17 t/s | 9.8 (about 59%) | 0.93 | about 6% |
| mistral-7b | 4.37 GB | about 9 t/s | 5.9 (about 64%) | 0.53 | about 6% |

Juno reads its weights at about 6% of attainable bandwidth on every model; the reference tool at 42% to
64%. That uniform 6% says the CPU gap is a per-byte cost in Juno's kernels and dispatch, not a
model-specific one, and that a JVM engine attaining even a third of the roofline would read about 0.4x
to 0.7x. So 0.25x is a placeholder, not an ambition. [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md)
measures the host's attainable bandwidth, publishes Juno's attained weight bandwidth (bytes per token x
tg) beside every CPU tg reading, and restates this row after its step-2 breakdown, at the level the
breakdown supports and never below 0.25x.

**Post-plan anchor: GPU pp >= 0.40x (added 2026-09-30; not a gate of this plan).** The end-of-plan
`>= 0.25x` still leaves Juno about four times slower at prompt processing, and nothing about the JVM
explains a gap of that size on the GPU path: prefill is a handful of device kernels per layer, and the
kernels are CUDA whichever language launches them. So the 0.25x is a milestone on the way, not the
destination, and the plan records where the destination is so Tier 14's scorecard can say how far
short of it the plan ended. The figure comes from a roofline estimate on this host, not a measurement:

| Model, 512-token window | Matmul work | FP32-compute GEMM floor, GTX 1080 (about 6.5 to 8.9 TFLOPS) | Juno today (`20260930T141026Z`) | Reference tool |
|---|---|---|---|---|
| tinyllama-1.1b | about 1.0 TFLOP | about 0.11 to 0.15 s | 2.16 s | about 0.14 s |
| mistral-7b | about 7.2 TFLOP | about 0.8 to 1.1 s | 7.52 s | about 0.74 s |

Read three ways. About 85% of Juno's prefill time today is outside the GEMM compute, so removing host
work and copies (Tier 01B item 6) is the first and largest lever. A path whose GEMMs run at the
FP32-compute ceiling with no host work in between reaches roughly 0.5x to 0.7x on mistral-7b and
higher on the small models. Above that, only an integer `dp4a` tiled matmul over packed weights (Tier
01C) beats FP32 compute on this card, which is how the reference tool gets there. 0.40x is set below
the FP32 ceiling deliberately: it is reachable without 01C's kernel beating cuBLAS, and it is not
reachable without both 01B item 6 and 01C landing. Score it in Tier 14 as reported, never as a gate,
and re-derive it on any host with FP16 or int8 tensor cores, where the ceiling moves.

**Relationship to `docs/infra-plan/`.** This plan tree replaces it. The infra tree grew to 36
documents across four numbering schemes (Infra tiers 1-20, phases P0-P3, standalone `PROMPT-*`
prompts, and a routing document that points between them), and tracking what owned what became more
expensive than the planning was worth. `docs/gap-closure-plan/` is the deliberate clean start: this
tree is the live plan, and no tier here defers to, routes into, or waits on anything in
`docs/infra-plan/`.

The infra tree stays on disk as **evidence, not governance**. It holds the root-cause JFR analysis,
the dated record of what was measured and when, and the reasoning behind numbers this plan starts
from — all of which is worth citing. Cite it for measurements and history; do not cite it for
ownership, sequencing, or gates. In particular, the table above is this plan's own target and does
not need reconciling with that tree's P0 gate; it happens to adopt the same Phi-3.5 figure because
that figure was derived from real measurement on this host, not out of deference.

Two mechanical consequences. Tier numbers are not shared between the trees, so cite an infra
document by filename and never by its tier number. And the measurement thresholds this plan relies
on (the `compare-lora.sh` train/playback ratios, the `compare-vision.sh` latency/tps ratios, the
rule that a new CLI flag is added to `compare-llama-cpp.sh`'s pass-through set in the same change
that ships it) are restated where this plan uses them rather than incorporated by reference, so no
tier here has to go read the infra tree to know what gate it is being held to.

## Tier index and ordering rationale

Ordering: correctness-and-consistency first, then the shared architectural root cause behind three
independent measured regressions, then the largest measured gap in the repository (prompt
processing), then outward through the feature surface in the order a request actually flows
(attention/context → KV → quantization → tokenization → sampling → model coverage → speculative
decoding → scheduling → parallelism → backend breadth → vision → LoRA → server/cluster surface),
closing with a documentation hardening pass.

Two rows do not sit where their integer would put them, and the table below — not the numbering —
is the running order (execution rule 1). **Tier 01B** sits between 01 and 02, and **Tier 04B**
between 04 and 05. **Tier 08 runs before Tier 06**: Tier 06 adds a `forwardVerify` override per
architecture, so running it first would mean adding verify support to four handlers and then having
Tier 08 introduce four more that either need the same work again or silently lack it. Tier 08 first
means Tier 06 covers every handler in one pass. Tier 08's own file previously argued the opposite
ordering while sitting after 06; that rationale is corrected there.

**Tier 01C sits between 01B and 02 (added 2026-09-30).** It is the Q4_K/Q5_K/Q6_K half of what was
Tier 04C: the tiled integer matmul over packed weights for prefill-width batches. It was split out and
moved forward because prompt processing is the largest gap in the program (about 15x against about
2.5x for generation), Tier 01B's own step 2 record shows that removing staging and dequantization is
worth at most about 1.13x to 1.18x, and every sweep model is Q4_K_M, so the kernel needs nothing Tier
04 adds. Left at 04C's position it would have run after four tiers that each publish a pp ratio the
plan was not working on. What stays in 04C is the part that does depend on Tier 04: widening the
packed path and residency to Tier 04's new formats, the explicit FP16 fallback, and the `--mmq`
documentation audit.

| Tier | Title | Gap analysis refs |
|---|---|---|
| [00](TIER-00-correctness-and-consistency.md) | Correctness & consistency audit | §2.1, §2.4, §2.5, §2.6, §2.7, §2.9 |
| [01](TIER-01-gpu-activation-residency.md) | GPU activation-residency redesign | §1.6, §2.8 |
| [01B](TIER-01B-prefill-throughput.md) | Prefill throughput | none — see that file's "Why this tier, why now" |
| [01C](TIER-01C-packed-kquant-matmul.md) | Packed K-quant prefill matmul (tiled `dp4a` GEMM for Q4_K/Q5_K/Q6_K; split out of 04C) | none — adjacent to §1.1; see that file's "Why this tier, why now" |
| [02](TIER-02-attention-long-context.md) | Attention & long context | §1.2 |
| [03](TIER-03-kv-cache-maturity.md) | KV cache maturity | §1.3 |
| [04](TIER-04-quantization-coverage.md) | Quantization coverage (mapped weight loading first) | §1.1 |
| [04B](TIER-04B-tokenizer-fidelity.md) | Tokenizer fidelity (per-family splits and cross-engine parity; the key read and fail-closed path moved to Tier 01 as parity precondition 7) | none — see that file's "Why this tier, why now" |
| [04C](TIER-04C-packed-weight-matmul.md) | Packed-weight coverage (Tier 01C's packed matmul and residency widened to Tier 04's formats; explicit FP16 fallback) | none — adjacent to §1.1; see that file's "Why this tier, why now" |
| [05](TIER-05-sampling-grammar.md) | Sampling & grammar completeness | §1.4 |
| [08](TIER-08-model-architecture-breadth.md) | Model architecture breadth | §1.9, real files in `models/` |
| [06](TIER-06-speculative-decoding.md) | Speculative decoding expansion | §1.5 |
| [07](TIER-07-continuous-batching.md) | Continuous batching maturity | §1.3 (scheduling half) |
| [09](TIER-09-tensor-parallelism-multi-gpu.md) | Tensor parallelism & multi-GPU (and batched prefill on every cluster path, added 2026-10-02) | §1.8, §2.2 |
| [10](TIER-10-gpu-backend-breadth-cpu-simd.md) | GPU backend breadth & CPU hot path (SIMD, allocation, threading) | §1.7 |
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
| 15 | JVM embedding facade | `JunoPlayer`, `LoraTrainer`, `JunoHttpClient` (`juno-player`) — an embedder cannot reach a new capability through CLI or REST alone |

**Row 1 caveat (CPU inference) before Tier 10 lands:** every tier from 00 through 09, Tier 01B
included, uses row 1 as a *correctness* oracle only — `CpuMatVec`'s scalar path, not a SIMD path,
since Tier 10 is what makes CPU inference actually fast. A tier's row-1 "PASS" before Tier 10 means
"correct," not "final-performance-path verified." Tier 10 re-verifies, rather than assumes, that its
CPU changes preserve every earlier tier's row-1 correctness result — vectorized float accumulation
can legitimately reorder floating-point sums vs. the scalar path, and so can a different thread count
or work-splitting strategy, which Tier 10 now also changes — see that tier's own exit criteria.

**Row 15 (JVM embedding facade) was added late — read this before marking it N/A.** `JunoPlayer`,
`LoraTrainer` and `JunoHttpClient` in `juno-player` are a public surface `CLAUDE.md` names, and a
capability wired through the CLI and both REST surfaces is still unreachable to an embedder if the
facade does not expose it. Before this row existed the facade appeared in this tree exactly once, in
Tier 00's execution record. Tiers 00 and 01 were written and executed without it and do not carry
it. Tier 01B's scope is internal throughput with no new embedder-invocable capability, so it is N/A
there. From Tier 02 onward the row is resolved like any other — PASS, N/A with a reason, or
FAIL-CLOSED. Its table row is already present in the three tiers where it is most clearly
load-bearing: Tier 02 (context-shift opt-in), Tier 05 (sampler-chain selection) and Tier 12
(per-request LoRA selection). Any other tier from 02 onward that adds a user-invocable capability
adds the row to its own table rather than leaving it off.

## Test infrastructure

Three layers, all of which get extended (never replaced) tier over tier:

1. **Unit tests** in the owning module (`mvn test -pl <module>`). Each tier's file names the exact
   test classes to add cases to or create. Note that `CLAUDE.md`'s own quick-reference `mvn test -pl
   tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player` command omits two
   real modules with their own test suites — `vision` (relevant from Tier 08 onward, primary from
   Tier 11) and `metrics` (the JFR-extractor tests every perf-gate change ultimately depends on).
   Any tier touching either module must explicitly add `-pl vision` / `-pl metrics` to its own test
   commands rather than assuming the documented command covers them; Tier 00 corrects the documented
   command itself.
2. **Integration tests**: `ModelLiveRunnerIT` (`juno-master/src/test/java/cab/ml/juno/master/`,
   run via `mvn verify -pl juno-master -Pintegration -DMODELS=...` for several models, or `./juno test
   --model-path ...` for one; 9 checks — 6 pipeline-parallel, 2 tensor-parallel and, since Tier 01B, an
   in-process 512-token prefill on both schedules; both run the same `ModelLiveChecks`) and the
   forked-JVM cluster ITs (`ThreeNodeClusterIT`, `TensorParallelClusterIT`, `mvn verify -pl
   juno-master`). Every tier that changes forward-pass, batching, or cluster behavior adds a new
   check to `ModelLiveRunnerIT` rather than only relying on unit tests — this is the one test class
   that runs against a real model file end to end.
   *Corrected 2026-10-03 (Tier 01B step 8): this item said the IT runs via `./juno test`. It does not,
   and that command runs no check at all: commit `55122e8` removed the `ModelLiveRunner` main class, and
   `./juno test` had started `CoordinatorMain`, which exits 1 without `JUNO_NODE_ADDRESSES`. Restored
   2026-10-03 (owner decision): `ModelLiveRunner` runs the checks again, sharing `ModelLiveChecks` with the IT.*
3. **Bash smoke scripts** under `scripts/performance-tests/`, following the existing
   `smoke-grammar.sh` / `smoke-tools.sh` convention: one `smoke-<short-name>.sh` per tier, named for what
   it checks and never for the tier (`CLAUDE.md` keeps tier numbers out of shipped files and docs, and the
   script is named in `docs/howto.md`),
   written in bash, that drives `./juno local`/`cluster` against the real model matrix (see
   `INVENTORY.md`) and asserts on process exit code, HTTP status, and response shape — not just
   "does it crash," but "is the output the specific thing this tier promised." Each smoke script
   must be runnable standalone and re-run (unmodified) by every later tier as a regression check —
   later tiers only ever *add* a new smoke script, never edit an earlier one except to fix
   a bug in the script itself.
   *Corrected 2026-10-03 (owner decision): this item prescribed `smoke-tierNN-<short-name>.sh`. Tier 01
   already had to rename its script for this reason (`smoke-gpu-residency.sh`); Tier 00's and Tier 01B's were
   renamed to `smoke-consistency.sh` and `smoke-long-prompt-prefill.sh`, and every later tier's planned
   script name drops its tier number.*

Performance regression gates stay governed by the existing rule in the root `CLAUDE.md`
("Performance gates on hot-path changes"): any tier touching the forward pass, MatVec, GPU
residency, batching, or KV paths runs `scripts/performance-tests/compare-lora.sh` (and
`compare-vision.sh` if vision/CLIP is touched) against the last published baseline in
`docs/perf-compare/`, and publishes a new timestamped result directory there — that mechanism
already exists and this plan doesn't change it, it just says explicitly, per tier, when it applies.

**llama.cpp-relative gate (applies from Tier 01 onward).** `compare-lora.sh`/`compare-vision.sh`
only ever compare Juno against its own prior baseline — neither tells you whether the actual stated
goal of this plan (closing the gap with llama.cpp) is moving. Every tier whose scope includes the
forward pass, MatVec, GPU residency, batching, quantization, or KV paths (at minimum: Tiers 01, 01B, 01C,
02, 03, 04, 04B, 04C, 06, 07, 08, 09, 10) additionally re-runs
`scripts/performance-tests/compare-llama-cpp.sh` on the same host/model/quant/flags as the last
published run under `docs/perf-compare/`, and records the resulting Juno/llama.cpp tg and pp ratios
in that tier's own file (not just in `docs/performance.md`, so the trend across tiers is visible
from the plan tree itself), **against the program target and the intermediate milestone for that
tier** (see "Program target" above). The script itself is not a new mechanism — it already exists
with a multi-month history of published runs — it just has to actually be invoked per tier instead
of only opportunistically, and now has a number to be read against. Tier 14's exit criteria add a
final consolidated scorecard summarizing this trend across every tier (see that file).

**Benchmark parity preconditions (blocking — must land before any tier publishes a llama.cpp
ratio).** As `compare-llama-cpp.sh` stands today, the two engines are not measured comparably, and
every ratio this plan is about to collect would inherit the difference:

| Asymmetry, as of this writing | Effect on the ratio |
|---|---|
| **llama-bench prefills `-p ${N_PROMPT}` tokens; Juno prefills a fixed 57-character sentence unless `--raw-prompt` is passed, and `RAW_PROMPT` defaults to `0`** | **the largest asymmetry, and it lands on the metric this plan cares most about. In the headline run (`docs/perf-compare/20260918T204809Z/`, `raw_prompt: 0`) llama-bench prefilled 128 tokens while Juno prefilled 20, 20, 21 and 30. The published pp ratio is a batch-width mismatch plus a cold-start penalty as much as a throughput gap** |
| llama-bench runs `-r 3` (plus its own internal warmup); Juno is measured from a **single** `/v1/chat/completions` request of `n_gen` tokens | C2 compilation of the entire forward pass sits inside Juno's measurement window; the reported Juno number is a cold-JVM number |
| llama-bench gets `-t ${N_THREADS}`; Juno has **no** thread-count control that reaches the hot path (`-Djuno.simd.pool.size` is built but the kernels dispatch on `ForkJoinPool.commonPool()`) | the two engines run at different parallelism — commonPool defaults to one fewer than `availableProcessors()` |
| `COMPARE_HEAP` is derived automatically from model size | heap, and therefore GC behaviour, changes between models and between runs of the same model as code changes |
| Four JFR creation sites disagree: `run.sh`'s `cmd_test` uses `settings=profile`, `ClusterHarness` uses `settings=default`, and `ConsoleMain` builds two recordings programmatically from `Configuration.getConfiguration("default")` (`startLocalJfr` and the cluster-coordinator recording) | measurements carry different instrumentation overhead and are not directly comparable. Note which site matters: `compare-llama-cpp.sh` passes `--jfr` as an *app* argument, so **every published Juno ratio was taken under `ConsoleMain`'s programmatic `default`** — not under either of the two configurations named in a `settings=` string |
| no CPU governor, turbo, or GPU clock state is recorded | a thermally throttled run is indistinguishable from a regression |
| no JDK event is consumed anywhere in this repository — `JfrMetricsExtractor` declares only `juno.*` names, and `compare-llama-cpp.sh`'s `jfr_summary_json` reads only `juno.ForwardPass.*`/`juno.TokenProduced.*` | GC pauses, allocation rate, hot methods and lock/park time cannot be read off any gate run, so the noise-control rule below and Tier 10's cost breakdown are unenforceable until the extractor is widened |
| **the two engines tokenize the same prompt differently.** `GgufTokenizer` never reads `tokenizer.ggml.pre` — a repo-wide grep finds no reference to that key anywhere — so it applies one BPE strategy whatever the file declares, while the reference tool applies the pre-tokenizer split the file asks for | `prompt_tokens` is the denominator of every pp figure and the context depth every tg figure is taken at, so a token-count difference is absorbed silently into both ratios. Precondition 1's 10% tolerance detects a large divergence but cannot correct one, and cannot see a divergence that happens to land inside 10% |

Before Tier 01 publishes its gate, the harness and the engine gain the following, in this order of
importance. Preconditions 1 to 6 are `compare-llama-cpp.sh` changes; precondition 7 is an engine
change, and it is here rather than in a later tier because it moves the same number precondition 1
moves:

1. **Prompt-token parity.** `RAW_PROMPT` defaults to `1`, so Juno prefills approximately the same
   token count llama-bench is given. Every result JSON records Juno's actual `prompt_tokens`
   alongside `n_prompt`, and a run whose Juno `prompt_tokens` differs from `n_prompt` by more than
   10% is **not publishable as a ratio** — publish it as a Juno-only measurement or re-run it. This
   is first because it is the only asymmetry that changes the metric the program target is written
   against, and because the two runs this plan quotes as evidence that prefill degrades with prompt
   length sit on opposite sides of it (see Tier 01B).
2. `--juno-warmup N` (default 2) discarded requests before the measured one.
3. `--juno-reps N` (default 3) with the median reported and min/max recorded. Reuse
   `compare-lora.sh`'s existing `--reps`/median implementation rather than writing a second one, and
   note that `compare-lora.sh` itself defaults to `REPS=1` — every gate run in this plan passes
   `--reps 3` explicitly to both scripts.
4. A Juno thread-count control that actually reaches the hot path, set equal to llama-bench's `-t`
   (this requires the hot-path threading work in Tier 10 item 5 to expose one — until it does,
   record the effective Juno parallelism in the run metadata and state the mismatch in every
   published INDEX rather than leaving it implicit).
   *Amended 2026-09-30: the interim does not have to wait for Tier 10.* Every CPU hot-path kernel
   (`SimdThreadPool.forEachRow` and the `matVecQ*raw` family) dispatches through
   `IntStream.parallel()` on `ForkJoinPool.commonPool()`, whose size the JVM already takes from
   `-Djava.util.concurrent.ForkJoinPool.common.parallelism`. The calling thread joins the work as one
   more worker, so passing `N_THREADS - 1` gives `N_THREADS` threads on the hot path, matching the
   reference tool's `-t`. [Tier 01B](TIER-01B-prefill-throughput.md) item 7 wires that into
   `compare-llama-cpp.sh` and records `juno_threads` in `host.json`. It is a harness setting, not
   the product control: Tier 10 item 5 still ships `--threads` and may replace the common pool, at
   which point the harness switches to the flag. Changing the effective parallelism is a measurement
   boundary for CPU readings (GPU readings are unaffected unless a CPU-side term is material), so the
   first run with it is re-baselined against itself.
   *Harness half landed 2026-10-01 (Tier 01B step 3a).* Finding: at the harness default `--threads`
   (`nproc`, 12 on the baseline host) the property is 11, which is the JVM's own default, so every
   default-thread run already ran Juno's kernels on 12 threads; the old INDEX note's "effective
   parallelism of 11" left out the calling thread. The change therefore moves no default-thread
   reading and is a boundary only for runs given an explicit `--threads`. The CPU reference was re-taken
   with it on 2026-10-01, pinned (`docs/perf-compare/20261001T180241Z/`), and is the current CPU column.
5. A fixed per-model `COMPARE_HEAP` rather than a derived one.
6. CPU governor plus `nvidia-smi -q -d CLOCK` captured into the run metadata. *Extended 2026-09-27:
   captured is not enough for a gate tighter than the noise floor. `--pin-clocks` pins them for the run,
   and the run metadata now names the build it measured — see "Clocks are pinned for gate runs" below.*
7. **Pre-tokenizer parity.** `tokenizer.ggml.pre` is read and dispatched on for every pre-type the
   sweep models declare, so the token count the ratio divides by is the count the model was trained
   against, and a file declaring a pre-type Juno does not implement is rejected at load with an error
   naming it. A ratio published before this lands carries a tokenization boundary as well as a
   prompt-length one. This is last in the ordering because it is the only one of the seven that is a
   change to `src/main` rather than to the harness, not because it matters least: preconditions 1 and
   7 move the same figure, and 1 without 7 is a tolerance around an error rather than its absence.
   The work is [Tier 04B](TIER-04B-tokenizer-fidelity.md)'s items 1 and 3, hoisted into Tier 01 for
   this reason; that tier keeps the per-family splits and the cross-engine parity corpus, which
   deepen the guarantee but do not gate a ratio.

   **Landed 2026-09-26**, in [Tier 01](TIER-01-gpu-activation-residency.md) — see its execution
   record for the enumeration of declared types across `models/` and the measured divergence. The
   finding that matters for this list: **it moved no sweep model's benchmark `prompt_tokens`, so the
   2026-09-25 reference sweeps are not superseded** and the ratios in the target table above do not
   carry a tokenization boundary after all. Three of the four sweep models are SentencePiece, which
   takes no pre-tokenizer split at all, and the fourth's benchmark prompt contains none of the
   constructs the split moves.

**Why precondition 7 is not simply left to Tier 04B where it was written.** Tier 04B sits eighth in
the running order, after Tiers 01, 01B, 02, 03 and 04 have each published a `compare-llama-cpp.sh`
ratio. `prompt_tokens` is the denominator of every pp figure and sets the context depth of every tg
figure, so landing pre-tokenizer dispatch at that point would make Tier 04B a second measurement
boundary crossing five tiers of published gates — including [Tier 01B](TIER-01B-prefill-throughput.md)'s
`pp >= 0.10x` milestone, which is the largest single performance ask in this plan and the one it can
least afford to have scored against a denominator that later moves. Tier 04B's own text makes the
argument for this: it says precondition 1's tolerance "is a guard against the symptom. This tier
removes the cause." A cause of a measurement error belongs with the measurement corrections, and the
measurement corrections are Tier 01's.

The JFR configuration mismatch is resolved in the same change, and the fix has to name all four
creation sites, not the two that carry a literal `settings=` string. Add
`scripts/performance-tests/juno-perf.jfc` — derived from `default`, enabling every `juno.*` event
plus `jdk.GCPhasePause`, `jdk.ObjectAllocationSample` (throttle 300/s),
`jdk.ThreadAllocationStatistics` (period 500 ms), `jdk.ExecutionSample` (period 10 ms),
`jdk.JavaMonitorEnter` and `jdk.ThreadPark` (threshold 10 ms) — and point all four at it:
`run.sh`'s `cmd_test`, `ClusterHarness`'s forked-node flag, and **both**
`Configuration.getConfiguration("default")` calls in `ConsoleMain` (`startLocalJfr` and the
cluster-coordinator recording). The last two are the ones that matter most: `compare-llama-cpp.sh`
launches Juno with `--jfr` as an app argument, so that is the configuration every published ratio
was actually taken under.

Because this changes the harness, it also breaks strict comparability with the runs already
published under `docs/perf-compare/`. That is accepted: the first corrected run is re-baselined
against itself and labelled as the new reference, the pre-correction numbers are kept and marked as
pre-parity-correction rather than deleted, and no tier's gate is scored against a baseline taken on
the other side of the change. Tier 01 carries this work as a precondition, not as scope creep — it
is the measurement that ten tiers' exit criteria depend on.

**Numeric thresholds, not "no unexplained regression."** Per execution rule 7 above, every tier's
perf-gate bullet under "Tests to write/upgrade before implementation" states a concrete pass/fail
number for whatever new metric that tier introduces (a memory-reduction percentage, a throughput
ratio, a latency ceiling) — the same way Tiers 01/03/06/07 already do for the specific historical
regression each one is re-measuring ("no longer ~7x slower," "gather tax at zero," "faster than
`--spec-type none`," "beats static"). Pick the number before implementation starts, the same way the
tests-before-implementation rule already asks for tests before code.

**Regression-noise control: GC pauses and allocation rate are tracked by default, not discovered ad
hoc.** `docs/performance.md`'s own history includes a real false-positive regression signal caused by
a single 622ms GC pause contaminating a short JFR-window `tps` measurement (the LoRA playback gate
incident that led to switching that one gate to wall-clock timing). Every perf-gate run from Tier 01
onward records `jdk.GCPhasePause` count/max and an allocation-rate figure alongside its primary
metric, and a run containing an outlier GC pause is re-run rather than scored — this generalizes the
fix already applied once to LoRA, instead of waiting to rediscover the same failure mode per tier.

**"Outlier" is a number, not a judgement call.** A run is re-run rather than scored when either
condition fires:

- ~~any single `jdk.GCPhasePause` exceeds **200 ms**, or **5%** of the measurement window, whichever
  is smaller.~~ **Replaced, on measurement, by a dispersion rule: a row is not scorable when its own
  repetitions disagree by more than 15% of their median.** Applied by `compare-llama-cpp.sh`, which
  emits a `noise` object per comparison and a `Scorable` column in `INDEX.md` naming the condition.
  The pause rule was implemented first and then found to gate on a number that does not measure
  stopped time on this host: `tinyllama` tuned produced 64 tokens across token spans of 1110, 1120 and
  1107 ms while its three pause readings were 633, 5 and 4 ms, and a 633 ms stop-the-world inside a
  1110 ms span would imply 134 t/s, twice what the model reaches. It rejected three rows whose readings
  agreed to within 1% and passed one whose readings spanned 31%; dispersion gets all four right, and a
  pause that does cost time appears as one slow repetition. Pause figures are still published, now
  including the share overlapping the measured token span
  (`jdk.GCPhasePause.in_token_span.*`), for reading rather than gating. *Cause found 2026-09-28 (Tier
  01B step 0): the 633 ms "pause" was not a pause. On this host CPU0's timestamp counter reads 633 ms
  ahead of the other cores, and JFR stamped events with the raw counter, so any span crossing CPU0 read
  633 ms short or long. The harness now makes JFR read the operating-system clock on such a host and
  withholds any repetition whose spans do not account for its own request; see Tier 01B's execution
  record. The dispersion rule stays the gate;*
- ~~`jdk.JavaMonitorEnter.total_ms` plus `jdk.ThreadPark.total_ms` exceeds **10%** of wall time.~~
  **Withdrawn as a gate — measurement showed it cannot discriminate.** The park figure is a sum
  across every thread, so an idle worker pool parks for longer than the run takes however healthy
  the run is. On a clean GPU run during Tier 01's re-baseline, `qwen2.5-3b` park time was 11,458 ms
  against a 4,241 ms request — about 2.7x wall time, monitor time zero, on all three repetitions of
  a run whose readings agreed to within 1%. As written the condition fires on every run. Both
  figures are still published in every result JSON, for reading rather than gating. A usable version
  of this condition would have to normalize per thread, or count only threads that did work.

`compare-llama-cpp.sh` prints a `NOISY` marker in `INDEX.md` naming which condition fired, so a
reader can tell a discarded run from a missing one. Neither threshold is hypothetical: the
historical LoRA incident above was a 622 ms pause, and the first run taken under Tier 01's new JFR
configuration produced a 657 ms pause on `./juno local --cpu --jfr 2m` against tinyllama. Both fire
this rule. (The 657 ms reading sits within a few tens of milliseconds of the 633 ms timestamp offset
described above and was most likely the same artifact; the 622 ms LoRA incident predates any recording
of this host's counter offset and is not attributable either way.) A tier that scores a run despite a fired marker states why in its own file rather than
leaving the marker unexplained.

**That rule needs tooling that does not exist yet, and Tier 01 builds it.** `JfrMetricsExtractor`
declares twenty `juno.*` event names and no `jdk.*` ones; a repo-wide grep for `GCPhasePause`,
`ObjectAllocationSample`, `ExecutionSample`, `JavaMonitorEnter`, `ThreadPark` and
`ThreadAllocationStatistics` across every `.java` and `.sh` file returns nothing outside `docs/`. So
as things stand, no gate run in this plan can report a GC pause or an allocation rate, and Tier 10's
cost breakdown has nothing to read. Alongside the `.jfc` above, Tier 01 extends
`JfrMetricsExtractor` with a `jdk.*` bucket emitting `jdk.GCPhasePause.count`, `.max_ms`,
`.total_ms`, `jdk.ThreadAllocationStatistics.bytes_total` (this, not the sampled
`jdk.ObjectAllocationSample`, is what a bytes-per-token ceiling is read from — the sampled event
gives attribution, not a rate), `jdk.ObjectAllocationSample.top_sites`,
`jdk.ExecutionSample.top_methods`, `jdk.JavaMonitorEnter.total_ms` and `jdk.ThreadPark.total_ms`,
with tests in the `metrics` module. `compare-llama-cpp.sh`'s `jfr_summary_json` surfaces the GC and
allocation figures into every published result JSON. Until this lands, "records GC pauses and
allocation rate" is an instruction nobody can follow.

**Noise floor: no gate may be stated tighter than the harness can resolve.** Two CPU sweeps taken
eight minutes apart on this host with identical flags — `docs/perf-compare/20260918T031702Z/` and
`20260918T032455Z/`, differing only in `juno_use_vector` — moved llama.cpp's own TinyLlama tg from
25.98 to 22.45 t/s (**-14%**) and its pp from 61.17 to 71.07 t/s (**+16%**), with llama-bench's `-r 3`
already applied. That is the measurement floor. No tier may state a pass/fail threshold inside
±15% unless it is a median of at least three runs with min/max published. Where a tier's threshold is
necessarily tighter than that (Tier 01B's "tg within 0.95x", Tier 04's "within 15% of the Q4_K MMQ
kernel"), the median-of-three discipline is mandatory, not optional, and the min/max spread is
published next to the median so a reader can see whether the result cleared the floor.

**No-regression gates tighter than the floor are Juno-against-Juno, same hour, pinned clocks — never a
ratio.** Median-of-three does not remove drift between two sessions, and a llama.cpp-relative ratio
carries the reference tool's drift as well as Juno's: the -14%/+16% above was llama.cpp alone. A
"tg ratio within 0.95x of the baseline" gate read across two sweeps taken hours apart therefore cannot
tell a 5% regression from the host. Tier 01's close-out is the evidence in the other direction: a 7% to
9% prefill regression that two sweeps could not attribute was found by a same-hour A/B. So:

- **Any no-regression gate at 0.90x or tighter** is scored on **Juno absolute t/s** (or ms), from a
  **same-hour interleaved A/B**: the baseline build's jar and the candidate, alternating A B A B A B,
  each invocation `compare-llama-cpp.sh --juno-jar <jar> --juno-reps 1 --juno-warmup 2 --reps 1
  --no-publish` (the candidate's own build may be passed the same way), with **`--pin-clocks`** on
  every invocation. The gate reads the median of the three candidate readings against the median of
  the three baseline readings, and publishes all six. A run without pinned clocks is not scorable
  against such a gate. A gate comparing two configurations of one build (a flag on against off)
  alternates the flag the same way instead of the jar. `compare-lora.sh --baseline <ref>` already
  builds and runs both sides in one session, which satisfies the same-hour half; until it gains its
  own pinning flag, pin the governor and turbo by hand (`sudo cpupower frequency-set -g performance`,
  turbo off) for a gate run and say so in the tier's record.
- **llama.cpp-relative ratios are reported, never gated below the 15% floor.** They remain the
  program's scoreboard (milestones, the end-of-plan target, Tier 14's scorecard), all of which sit well
  outside the floor.
- Gates looser than 0.90x (for example the vision gate's latency `<= 1.25x`, decode tps `>= 0.80x`, or
  the LoRA playback gate's `>= 0.80x`) may still be read across published runs, median of three.

**Clocks are pinned for gate runs, not only recorded.** Recording the governor, turbo state and GPU
clocks lets a reader see that a run was taken at a moving clock; it does not stop the clock moving,
and this host runs `schedutil` with turbo on by default. `compare-llama-cpp.sh --pin-clocks` sets the
performance governor and turns turbo off for the run and restores both on exit, and locks the GPU
graphics clock where the driver allows (`--pin-gpu-mhz`, default the card's maximum). It needs
prompt-free sudo (`sudo -v` first) and refuses to start if it cannot pin the CPU. GPU clock locking is
refused by the driver on some cards, including consumer Pascal parts like this host's; the run then says
the GPU clock was recorded, not fixed. Every run also records what it measured: the Juno commit and
whether the tree was dirty, the jar's hash, the JDK build, the JVM flags (the heap is now fixed with
`-Xms` equal to `-Xmx`), the GPU driver, and the reference tool's build commit. **A change of reference
build is a measurement boundary** — this host already runs two (one per backend), and a ratio against
one is not comparable with a ratio against the other.

*Superseded 2026-09-29: the owner adopted GPU-free CI at Tier 01B's revisit
(`.github/workflows/ci.yml`: script checks on every push; `mvn -B clean verify` on pull requests and
`main`). Real-model and performance gates, and so tier gating, stay procedural. The paragraph below is
kept as the record of the earlier decision.*

*Checked 2026-09-30: the decision is recorded but CI does not yet run.* `.github/workflows/ci.yml`
exists in the working tree only: `git status` reports `.github/` as untracked and `git ls-files
.github` is empty, so no push has ever carried the workflow and no job has ever run. Until it is
committed and a first run is green, everything in this plan that cites CI as the enforcement (rule 7's
check on every push in particular) is enforced by nothing. Committing it is the owner's action (git
writes are the owner's in this repository); [Tier 01B](TIER-01B-prefill-throughput.md) carries the
exit criterion, with the first green run's URL recorded there.

*Decided 2026-10-01 (owner): CI is not an exit criterion.* Tier 01B's item 10 is withdrawn, and no tier
waits on a CI run. The workflow file stays uncommitted in the working tree. Every check this plan
attributes to CI (rule 7's `check-plan-thresholds.sh`, `compare-llama-cpp.sh --selftest`,
`selftest-engine-stdin.sh`, `mvn -B clean verify`) remains a step the executor runs by hand and records,
exactly as before CI was adopted. Rule 7 says "machine-checked": that means the script, not where it
runs.

**No CI exists in this repository today** (confirmed: `.github/` holds only a `modernize/`
directory, no workflows) — tier-gating (execution rule 1: don't start Tier N+1 until Tier N's exit
criteria are all checked) is enforced procedurally, by whoever executes the plan re-reading the
checklist, not automatically. This is an accepted, explicit trade-off rather than a silent gap; if a
CI pipeline is added during this plan's execution, wiring `mvn test` plus the relevant
`compare-*.sh` gate into it per tier is in scope for whichever tier is active at that point.

**Owner and revisit trigger:** whoever executes the plan owns this decision, and it is re-examined
at Tier 07 or at the first point a tier's full smoke matrix exceeds thirty minutes of hands-on
execution, whichever comes first. Record the outcome of that re-examination in the then-active
tier's file, so "we decided not to" stays a decision with a date on it rather than an omission.

**The thirty-minute condition fired during Tier 01 and was not recorded there.** Its eleven-module
unit reactor ran 25 to 27 minutes per pass, the residency smoke on llama-1-30b took 3 h 11 min, and the
RoPE-pairing diagnosis ran 36 minutes. The trigger and the evidence are now recorded in
[Tier 01B](TIER-01B-prefill-throughput.md)'s execution record, the active tier, with the decision itself
left to the owner and due before Tier 01B's step 2 re-baseline.

**That trigger now has somewhere to fire.** Naming Tier 07 here and nowhere else meant the revisit
lived only in this paragraph, and a reader working through Tier 07's own scope and exit criteria would
never learn it was due — an omission of exactly the shape this paragraph exists to prevent.
[Tier 07](TIER-07-continuous-batching.md) now carries it as a scope item and an exit criterion. The
thirty-minute condition stays a standing trip-wire for every tier: a tier whose smoke matrix crosses
it records the re-examination in its own file even if Tier 07 has already been closed out, because the
condition is about how expensive the matrix has become, and that only grows.

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
  rule, against a concrete numeric threshold stated in that tier's own file (execution rule 7 — not
  just "no unexplained regression"), and, for Tiers 01, 01B, 01C, 02, 03, 04, 04B, 04C, 06, 07, 08, 09, 10, an
  accompanying `compare-llama-cpp.sh` run recording the current Juno/llama.cpp ratio and reading it
  against the program target and that tier's intermediate milestone, if it has one.
- **The published API contract is updated in the same change as the code.**
  `api/src/main/resources/openapi.yaml`, `api/src/main/resources/juno-api.yaml` and
  `api/src/main/proto/inference.proto` are the contract, and no tier may add an endpoint, a request
  or response field, or an RPC without updating them. This is load-bearing for Tiers 02 (context-shift
  opt-in), 03 (session save/restore), 05 (`x_juno_samplers`, widened `json_schema`), 12
  (`x_juno_loras`) and 13 (`/v1/rerank`), and for Tiers 06 and 09 where the gRPC semantics change
  rather than the shape. Before this was written, the `api` module appeared in this plan exactly
  once — Tier 00 removing `RegistryService` — while six tiers added surface on top of it.
- `CHANGELOG.md` gets an entry describing what shipped, in the project's existing style.
- `docs/agent-arch.txt`, `docs/howto.md`, and `README.md` are updated if the change is user-facing
  or architectural (existing `CLAUDE.md` rule), using Juno-native language only (rule 4 above).
- No new dead/dormant/unwired scaffolding is left behind without an explicit tracking note in that
  tier's file explaining why it's dormant and what would need to happen to wire it in (the pattern
  flagged in gap-analysis §2.8 — don't repeat it silently).
