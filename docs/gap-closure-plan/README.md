# Juno gap-closure plan

Source material: [`../llama-cpp-gap-analysis.md`](../llama-cpp-gap-analysis.md) (2026-09-18 snapshot,
branch `67-inference`, HEAD `0c519f1`). That document is raw analysis; this tree turns it into a
sequenced, testable implementation plan. Re-verify any cited line number against current source
before acting on it — both documents are snapshots, not ground truth that stays accurate forever.

## Execution rules (binding for every tier)

1. **One tier at a time, to feature-complete.** Do not start the next tier's work until the current
   tier's exit criteria (bottom of its file) are all checked off. "Next" means the next row of the
   tier index below, not the next integer — the index is the running order. Eleven rows carry a
   non-integer number or sit out of integer order: Tier 01B sits between 01 and 02, Tier 01C between
   01B and 02, Tiers 02B, 02C, 02D, 02E, 02F and then 02G between 02 and 03, Tier 04B between 04 and 05,
   Tier 04C between 04B and 05, and Tier 08 runs before Tier 06. Only the last of those actually breaks
   the order — the lettered tiers read in sequence — but all eleven are enumerated here because this
   list, not the numbering, is what a reader builds the running order from, and several were added to
   the tree after the first three were written (02B and 02C on 2026-10-04, split out of Tiers 02 and 10;
   02D on 2026-10-08, by owner decision when Tier 02 closed; 02E on 2026-10-11, plan review, to own the
   CPU kernel work the only active end-of-plan row turns on; 02F and 02G on 2026-10-13, by owner
   decision, for the hardware validation kit and CUDA on Windows).
   Partial, half-wired features are not acceptable stopping points between tiers.
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

   - *(check 5, added 2026-10-04 by the plan review)* a tier whose `Status` line does not say
     `complete` and whose file never names `check-plan-thresholds.sh`, or one that has a perf gate (no
     `**No perf gate` declaration) and never names the **Standing throughput and allocation gate** (see "Test
     infrastructure"). Rules stated only in this README were skipped twice (this check's own
     invocation, and the CI revisit Tier 07 now carries); the tier file is what an executor reads;
   - *(check 6, added 2026-10-04)* a `perf-compare/` path containing a tier number (`-tier04b-...`) that
     does not exist on disk, which can only be a prescription for a new tier-numbered directory.
     `docs/performance.md` is shipped and links into `docs/perf-compare/`, so a tier number in a
     directory name ends up in a shipped doc. Existing directories may still be cited;
   - *(check 7, added 2026-10-04)* a row of the machine-read end-of-plan target table (see "Program target") that is
     malformed, or `active` while its reference reading already meets it — the milestone rule applied to
     end-of-plan targets, which it did not cover before; the unqualified GPU pp target was met at
     `n_prompt=128` on 2026-10-04 without the check noticing;
   - *(check 8, added 2026-10-11, plan review)* an exit criterion of an open tier, read with its indented
     notes, that lets a threshold close as "met or reported missed" / "reported as missed" / "the miss
     reported plainly" / "documents precisely why not" without citing an `**Owner decision`. Checks 1 to 4 prove a number exists and was scored; a criterion that may close
     on a written-down miss turns the number back into a report. Tiers 04, 04C and 07 carried that wording and
     were corrected: a miss keeps the tier open, and closing on one is the owner's recorded decision;
   - *(check 9, added 2026-10-11, plan review)* a paragraph of this README or of an open tier file that says
     "no tier owns" or "unowned" without linking a tier file: the tier that owns the work, or the tier
     record where the owner decided to leave it unowned. The only active end-of-plan row sat for a tier with
     its levers described as owned by no tier, and Tier 02D's greedy-softmax handoff to Tier 05 existed only
     in Tier 02D's record; both are now linked ([Tier 02E](TIER-02E-cpu-decode-kernels.md) and [Tier 05](TIER-05-sampling-grammar.md)).

   Tier 14's own file is excluded from the first check, since it names the string to describe it. The
   script exits non-zero listing every failure; it passes against this tree and was shown failing
   against the tree as it stood before the fixes that made it pass (seven failures: Tiers 04, 04B, 04C
   and 05 and all three milestone rows). Checks 5 and 6 and the end-of-plan table check were likewise
   shown failing before the tier files, directory names and table were corrected.

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

| Metric | Target at end of plan | Current reference (GPU: Tier 02D closing sweeps, 2026-10-12; CPU: Tier 02E closing sweeps, 2026-10-27) |
|---|---|---|
| GPU tg, Phi-3.5-mini | >= **0.70x** (was 0.50x, met 2026-09-30) | **0.973x** (0.972x tuned) |
| GPU tg, mistral-7b | >= **0.70x** (was 0.60x, met 2026-09-27) | 1.066x (1.061x tuned) |
| GPU pp, every sweep model | retired 2026-10-04: met at `n_prompt=128` (was >= 0.25x unqualified, before that 0.15x); replaced by the per-length rows of the end-of-plan target table below | **0.681x to 0.785x** at `n_prompt=128`, median 0.728x (TinyLlama binding); **0.726x to 0.833x** at 512, median 0.798x; **0.602x to 0.730x** at 2048, median 0.697x (Phi-3.5-mini binding at 512 and 2048) |
| CPU tg, every sweep model | >= **0.25x** (restated again 2026-10-11 from Tier 02C's closing sweep; was 0.40x from 2026-10-10, and a 0.25x placeholder before that); retired 2026-10-27: met by Tier 02E | **0.259x to 0.337x** (Phi-3.5-mini binding; pp at 128 0.431x to 0.516x) |
| CPU pp, `n_prompt=512`, every sweep model | >= **0.30x** (added 2026-10-11); retired 2026-10-27: met by Tier 02E | **0.422x to 0.512x** (Qwen2.5-3B binding; 0.378x to 0.411x at 2048, TinyLlama binding) |

The full history of readings, one column per reference since the plan was written, and the provenance of
every superseded column, are in [`TARGET-HISTORY.md`](TARGET-HISTORY.md) (moved 2026-10-11, plan review: the table
had grown to thirteen columns). Compare ratios across a measurement boundary named there, never absolute t/s.

The table above is the current reading against each target. The table below is what Tier 14 scores and what
`check-plan-thresholds.sh` reads (execution rule 7): every threshold is `>= number`, an `active` row
must ask for more than its reference reading, a retired row must really be met, and for an "every sweep
model" scope the reference is the binding (lowest) model's reading.

**End-of-plan targets, machine-read** (added 2026-10-04, plan review)

| Kind | Metric | Scope | Threshold | Reference reading | Status |
|---|---|---|---|---|---|
| end | GPU tg | Phi-3.5-mini | >= 0.70x | 0.858x (Tier 02 closing sweep `20261008T173153Z`; was 0.545x at `20261004T113210Z`) | retired: met by Tier 02 (decode residency region by default, 2026-10-08) |
| end | GPU tg | mistral-7b | >= 0.70x | 1.030x (Tier 02 closing sweep `20261008T173153Z`; was 0.627x at `20261004T113210Z`) | retired: met by Tier 02 (2026-10-08) |
| end | GPU pp, `n_prompt=512` | every sweep model | >= 0.40x | 0.594x (Phi-3.5-mini, binding; Tier 02 closing sweep `20261008T174605Z`; was 0.226x at `20261004T114812Z`) | retired: met by Tier 02 (2026-10-08) |
| end | GPU pp, `n_prompt=2048` | every sweep model | >= 0.25x | 0.540x (Phi-3.5-mini, binding; Tier 02 closing sweeps `20261008T180014Z`/`20261008T181417Z`; was 0.099x at `20261004T222758Z`) | retired: met by Tier 02 (2026-10-08) |
| end | GPU pp, prompt length unstated | every sweep model | >= 0.25x | 0.303x (Phi-3.5-mini, binding, at `n_prompt=128`; `20261004T113210Z`) | retired: met at `n_prompt=128` on 2026-10-04 |
| end | GPU tg, raised 2026-10-08 | every sweep model | >= 0.70x | 0.776x (TinyLlama, binding; Qwen2.5-3B 0.828x, Phi-3.5-mini 0.973x, Mistral 7B 1.066x; Tier 02D closing sweep `20261012T013642Z`; was 0.416x, Qwen2.5-3B, at `20261008T173153Z`) | retired: met by Tier 02D (Qwen2 in the decode region, 2026-10-12) |
| end | GPU tg, raised 2026-10-08 | Phi-3.5-mini | >= 0.90x | 0.973x (Tier 02D closing sweep `20261012T013642Z`, pinned; projected 0.967x on arrival from the Tier 02C pinned A/B `20261011T052951Z-cpu-hot-path-close-gate`) | retired: met on arrival at Tier 02D (owner decision 1, 2026-10-11), confirmed by the Tier 02D pinned closing sweep |
| end | GPU pp, `n_prompt=512`, raised 2026-10-08 | every sweep model | >= 0.70x | 0.726x (Phi-3.5-mini, binding; TinyLlama 0.782x, Mistral 7B 0.813x, Qwen2.5-3B 0.833x; Tier 02D closing sweep `20261012T015002Z`; was 0.594x at `20261008T174605Z`) | retired: met by Tier 02D (host KV reuse and direct K/V write, 2026-10-12) |
| end | GPU pp, `n_prompt=2048`, raised 2026-10-08 | every sweep model | >= 0.60x | 0.602x (Phi-3.5-mini, binding, 0.3% above the row; Qwen2.5-3B 0.680x, TinyLlama 0.713x, Mistral 7B 0.730x; Tier 02D closing sweeps `20261012T021800Z`/`20261012T020410Z`; was 0.540x at `20261008T181417Z`) | retired: met by Tier 02D (2026-10-12) |
| end | CPU tg, restated 2026-10-11 | every sweep model | >= 0.25x | 0.259x (Phi-3.5-mini, binding; Qwen2.5-3B 0.275x, Mistral 7B 0.309x, TinyLlama 0.337x; Tier 02E closing sweep `20261027T154748Z`, pinned; was 0.215x, Qwen2.5-3B, at `20261011T013059Z`) | retired: met by Tier 02E (CPU decode kernels, 2026-10-27) |
| end | CPU pp, `n_prompt=512`, added 2026-10-11 | every sweep model | >= 0.30x | 0.422x (Qwen2.5-3B, binding; TinyLlama 0.429x, Mistral 7B 0.436x, Phi-3.5-mini 0.512x; Tier 02E closing sweep `20261027T161927Z`, pinned; at 2048 0.378x to 0.411x, `20261027T172646Z`/`20261027T202604Z`; was 0.165x, TinyLlama, at the Tier 02E step 2 sweep `20261012T045931Z`, before any kernel change) | retired: met by Tier 02E (2026-10-27) |

**Why the GPU pp target became two rows (2026-10-04, plan review).** The target read "GPU pp, every sweep
model >= 0.25x" and named no prompt length. At `n_prompt=128` Tier 01C's closing sweep already meets it on
every model (0.303x to 0.729x), so by this plan's own rule (a target met before the work meant to reach
it measures nothing) it no longer asks anything. It is retired rather than deleted, so Tier 14 can report
it. Its replacements name their length. At 512 the target is the old post-plan anchor, 0.40x, which the
roofline below puts under the FP32-compute ceiling and which Tier 01C's packed GEMM now makes reachable
(three of four models already read 0.317x to 0.468x; Phi-3.5-mini binds at 0.226x). At 2048 it is 0.25x,
where prefill reads 0.099x to 0.180x today and attention is the term that grows; Tier 02's tiled kernel and
its attention-speedup milestone at 2048 are the lever.

**GPU targets raised again (2026-10-08, owner decision, Tier 02 decision 15).** Tier 02's closing sweeps met all
four GPU end-of-plan rows (tg 0.858x on Phi-3.5-mini and 1.030x on Mistral 7B against 0.70x; pp 0.594x at 512 and
0.540x at 2048 against 0.40x and 0.25x, Phi-3.5-mini binding), so they are retired above and replaced by four active
rows. Generation gains an every-model row at the old 0.70x, which Qwen2.5-3B (0.416x, outside the decode region)
binds; Phi-3.5-mini's own tg row rises to 0.90x; Mistral 7B, at 1.030x, has no separate row. Prefill rises to 0.70x
at 512 and 0.60x at 2048. All four are [Tier 02D](TIER-02D-gpu-host-overhead.md)'s, which decomposes them first and
restates any row the decomposition puts out of reach. The post-plan anchor below (GPU pp 0.40x at 2048) was passed by
the same sweeps (0.540x); it stays recorded as written and Tier 14 reports it as passed.

**Phi-3.5-mini GPU tg row retired on arrival (2026-10-11, owner decision, Tier 02D decision 1).** Tier 02D's decomposition (`docs/perf-compare/20261011T055415Z-gpu-host-overhead-decomposition/`) found the row met before the tier's work: Tier 02C's pinned GPU A/B read Phi-3.5-mini tg at 1.127x the build the 0.858x reference was taken on (0.967x projected), and two unpinned same-session runs read 0.977x and 0.955x. The step is 93.7% device kernels, so a raised row would be scored on kernel work no tier owns (left unowned by owner decision 4 in [Tier 02D](TIER-02D-gpu-host-overhead.md)'s closing record). Retired, not raised; Tier 02D's pinned closing sweep reports the measured ratio. *Confirmed 2026-10-12: the pinned closing sweep reads 0.973x (`docs/perf-compare/20261012T013642Z/`).*

**Every GPU end-of-plan row is met (2026-10-12, Tier 02D close).** Tier 02D's pinned closing sweeps met the three
remaining GPU rows: tg 0.776x to 1.066x on every sweep model against 0.70x (TinyLlama binding; Qwen2.5-3B 0.416x to
0.828x), pp 0.726x at 512 against 0.70x and 0.602x at 2048 against 0.60x (Phi-3.5-mini binding at both; the 2048 margin
is 0.3%). All three are retired. The 512-over-128 milestone is met on every model (1.036 to 1.149) and retired. The only
active end-of-plan row is CPU tg. Whether to raise the GPU rows again, as Tier 02 decision 15 did, is the owner's
decision, recorded in Tier 02D's closing record. *2026-10-11 (plan review): a CPU pp row at `n_prompt=512` is added, since
CPU prefill had no end-of-plan target once its milestone retired; both CPU rows are [Tier 02E](TIER-02E-cpu-decode-kernels.md)'s.*

**Every end-of-plan row is met (2026-10-27, Tier 02E close).** Tier 02E's pinned closing sweeps met both CPU rows: tg
0.259x to 0.337x on every sweep model against 0.25x (Phi-3.5-mini binding; Qwen2.5-3B, which bound at 0.215x, reads
0.275x), and pp at 512 0.422x to 0.512x against 0.30x (Qwen2.5-3B binding; TinyLlama, which bound at 0.165x, reads 0.429x).
Both are retired, kept at the values
written (owner decision 9 in [Tier 02E](TIER-02E-cpu-decode-kernels.md)'s record): no later tier owns CPU kernel work and no
further Q4_K candidate is in hand, and Tier 14 reports the distances. The tg margin is 3.6% and sits inside the reference
tool's own movement between pinned sessions: the same jar read 0.301x on Phi-3.5-mini hours earlier, when the reference
tool's CPU decode read 12% to 25% slower (`host-specific`; the closing gate's record has both readings). The decode kernels
reached 0.93 (Q4_K) and 2.13 (Q6_K) core cycles per weight on one thread, 1.10 and 2.49 per core at 12 threads, against
1.52 and 5.41 before the tier.

**Score every later tier against the current reference** (the right-most column of the history table in
[`TARGET-HISTORY.md`](TARGET-HISTORY.md)). Its CPU rows are Tier 02E's closing sweeps (2026-10-27, owner run, clocks
pinned, jar `4e480ab9131887eb`: `c2633a68` plus the in-recording allocation field; the vector Q6_K decode kernel, the
split Q4_K kernel, pooled vector prefill attention and the padded prefill tile): `docs/perf-compare/20261027T154748Z/`
(`n_prompt=128`), `20261027T161927Z/` (512), `20261027T172646Z/` (2048; TinyLlama, Qwen2.5-3B, Mistral 7B) and
`20261027T202604Z/` (2048, Phi-3.5-mini at an explicit 8 GiB heap); `--threads 12` against `-t 12`, prompt parity exact,
every gated reading scorable (the Phi-3.5-mini 2048 sweep's generation lane reads `NOISY`, 2.52 to 2.96 t/s; its prefill
lane is within 3.1%). It is a measurement boundary for CPU decode and prefill. Its GPU rows are unchanged from the column
to its left, Tier 02D's closing sweeps (`docs/perf-compare/20261012T013642Z/`, `20261012T015002Z/`, `20261012T020410Z/`,
`20261012T021800Z/`): Tier 02E's pinned GPU A/B read tg 0.986x to 1.013x and pp 0.969x to 1.008x the pre-tier build
(`docs/perf-compare/20261013T230239Z-cpu-kernels-close-gate/`). The reference tool's CPU decode reading moves by up to a
third between pinned sessions on this host (`docs/perf-compare/20261027T154748Z-cpu-kernels-close-gate-2/`), so a CPU tg
ratio taken in another session is compared with that in mind (`host-specific`).

**The 2026-10-12 column** was the reference until then. Its GPU rows are Tier 02D's closing sweeps (2026-10-12, owner
run, clocks pinned, jar `921ab742ba3f362a`: `198467ad` plus Qwen2 in the decode region, host KV reuse and the direct K/V
write): `docs/perf-compare/20261012T013642Z/` (`n_prompt=128`), `20261012T015002Z/` (512), `20261012T020410Z/` (2048;
TinyLlama, Qwen2.5-3B, Mistral 7B) and `20261012T021800Z/` (2048, Phi-3.5-mini at an explicit 8 GiB heap); every row
scorable, prompt parity exact, same harness as the column to its left. It is a measurement boundary for Qwen2.5-3B
generation (it now decodes in the device region) and for GPU prefill on every handler that reuses host KV. Its CPU row is
unchanged: the Tier 02D pinned CPU A/B read 0.983x to 1.030x the pre-tier build
(`docs/perf-compare/20261012T002824Z-gpu-host-overhead-close-gate/`).

**Rule: the reference column moves in the same change that supersedes it.** Whenever a run is marked
superseded-as-reference in `docs/perf-compare/` (its `INDEX.md` banner and `docs/perf-compare/README.md`),
the same change adds the new column to [`TARGET-HISTORY.md`](TARGET-HISTORY.md), updates the
current-reference table at the top of this section and the "Score every later tier" paragraph, and moves
the superseded column's provenance paragraph into `TARGET-HISTORY.md`. The 2026-09-27 column went stale
for a day because this rule did not exist: Tier 01's out-of-tier table superseded its source while this
table still told every later tier to score against it.

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
| 02 | GPU pp ratio at 512 over ratio at 128 | every sweep model | >= 1.00 | 1.036 (mistral-7b, binding; tinyllama 1.149, qwen2.5-3b 1.097, Phi-3.5-mini 1.044; Tier 02D closing sweeps `20261012T013642Z`/`20261012T015002Z`; 0.998 at the Tier 02 close, missed and reported by owner decision 13; was 0.606 at the Tier 01C close) | retired: met at the Tier 02D close |
| 02 | Attention speedup at `n_prompt=2048`: the pre-tier build's attention time per prefill window over the candidate's, clock-normalised (GPU ms x in-window median SM MHz), same-session A/B | every sweep model | >= 6.0x | 10.40x (qwen2.5-3b, binding; tinyllama 14.06x, Phi-3.5-mini 17.12x, mistral-7b 14.08x; Tier 02 closing A/B `20261008T054413Z-attention-close-gate`; the pre-tier build reads 1.0x by definition) | retired: met by Tier 02 |
| 04 | GPU tg ratio | Phi-3.5-mini | >= 0.40x | 0.416x | retired: met on arrival |
| 02C | CPU tg ratio | every sweep model | >= 0.20x | 0.215x (qwen2.5-3b, binding; tinyllama 0.238x, Phi-3.5-mini 0.222x, mistral-7b 0.224x; Tier 02C closing sweep `20261011T013059Z`, pinned; was 0.092x) | retired: met by Tier 02C |
| 02C | CPU pp ratio, `n_prompt=128` | every sweep model | >= 0.10x | 0.224x (tinyllama, binding; qwen2.5-3b 0.230x, Phi-3.5-mini 0.256x, mistral-7b 0.226x; Tier 02C closing sweep `20261011T013059Z`, pinned; was 0.049x) | retired: met by Tier 02C |

**Why the 512-over-128 row moved from Tier 01B to Tier 02 (2026-10-01, owner decision).** Tier 01B's
step 5 decomposition showed that no item in that tier can meet it. The term that grows with prompt length
is attention: kernel, copies and host part go from 5.6% to 8.8% of a 128-token window to 11.4% to 17.3% of a
512-token one (`docs/perf-compare/20261001T225929Z/`, `20261001T224724Z/`). Tier 01B's items remove
costs that are fixed per token, which raises attention's share of what remains, so the row is projected
to get worse as they land (Phi-3.5-mini 0.916 to about 0.87). The mechanism that moves it is the
tiled long-context kernel, which is Tier 02's, and Tier 02's 2048-over-512 row measures the same cause
one step further out. Moved rather than kept as a known Tier 01B miss, so Tier 01B is scored on what its
own items can move. Threshold unchanged. *Kept as written 2026-10-06 (owner decision, Tier 02 decision 6):* unlike the 2048 row it was not restated,
because the tiled kernel meets it on the current build, unpinned, on every sweep model (1.015 to 1.171,
`docs/perf-compare/20261006T045710Z-pp-length-scaling`; Phi-3.5-mini binding). The reference cell keeps the pre-tier
reading; Tier 02's pinned closing sweeps score the row.

**Why the 2048-over-512 row became an attention-speedup row (2026-10-05, owner decision).** It read "GPU pp
ratio at 2048 over ratio at 512 >= 0.90" (reference 0.329, mistral-7b binding). Tier 02's step 2 decomposition
(`docs/perf-compare/20261005T003146Z/milestone-decomposition.md`) showed that a ratio of two ratios does not
score the kernel meant to move it. The tiled kernel speeds attention at 512 as well, which raises the
denominator: with the same attention speedup at both lengths, 0.90 needs 6.4x (Phi-3.5-mini) to 84x
(mistral-7b), and a 10x kernel that lifts mistral-7b's pp ratio at 2048 from 0.14x to about 0.57x would still
read 0.69. The row also falls when anything else gets faster (Tier 01C's packed GEMM moved the 512-over-128 row
from 0.845 to 0.606) and moves with the reference tool's own scaling. The replacement is Juno against Juno and
moves only with the kernel. It is clock-normalised because a 2048-token window heats this host's card into
thermal slowdown (1607 MHz against about 1845 MHz at 512; `docs/perf-compare/20261005T012348Z/gpu-clocks.md`),
so a faster, shorter window would otherwise be credited with the clock it no longer loses. 6.0x is where the
binding model, Phi-3.5-mini, reaches the `>= 0.25x` end-of-plan row at 2048 (about 5x by the same
decomposition), and still about 25 times above mistral-7b's FP32 compute floor for exact attention (about 1.1
TFLOP per 2048-token window). The outcome against the reference tool stays scored by the end-of-plan rows at 512
and 2048. The 2048-over-512 ratio is still published by every tier that sweeps both lengths, as a reading,
not a gate.

**Why the CPU pp row exists (added 2026-09-30).** CPU prefill is 0.048x to 0.090x of the reference
tool and was owned by nothing: Tier 01B scopes it out to "re-measure at the end of Tier 10", and Tier
10 had no prefill number. 0.10x is roughly a 2x move on the binding model (Phi-3.5-mini, whose CPU
attention is scalar and quadratic), and about 1.1x to 1.3x on the other three. It sat on Tier 10
because the costs it turns on — allocation per matmul, the common-pool dispatch, and the kernel at
prefill batch widths — were that tier's items 3 to 5. *Moved 2026-10-04 with those items to
[Tier 02C](TIER-02C-cpu-hot-path.md) (items 1 to 3 there), and so are both CPU milestone rows.*

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
takes non-Llama decode residency, which no tier owns yet. *(2026-09-30: [Tier 01B](TIER-01B-prefill-throughput.md) item 0 put GPU
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
default (Tier 02 item 6), and the residency region reaching the Phi-3 and Qwen3 handlers (no tier owned
that when this was written; [Tier 02](TIER-02-attention-long-context.md) item 8 and [Tier 02D](TIER-02D-gpu-host-overhead.md) later took it; Tier 02 records whether its items reach 0.70x without it, and names the gap if not). The pp
path is the milestone ladder above: 0.10x after Tier 01B, 0.20x after Tier 01C, and no fall-off from
128 to 512 or from 512 to 2048 tokens after Tier 02 (*2026-10-05: the 512-to-2048 step is now an attention
speedup of >= 6.0x at 2048; see "Why the 2048-over-512 row became an attention-speedup row"*). The previous targets stay recorded in the table's first column so
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

**Why the CPU work runs third, not tenth (2026-10-04, plan review).** CPU reads about 0.1x on both tg and
pp, the largest gap in the program, and as Tier 10 it ran after nine tiers none of which moves a CPU
number. The CPU items moved to [Tier 02C](TIER-02C-cpu-hot-path.md), after Tier 02B and before Tier 03,
and the lever was restated: the decode kernel accumulates each row into one serial float, so the fix is
integer dot products over a once-quantized activation with split accumulators, not a wider float dot.
It also runs before Tier 04, whose mapped weight loading threads a `MemorySegment` accessor through the
same kernels and so should convert the new ones, not the old.

**CPU tg: the memory-bandwidth roofline (added 2026-09-30).** CPU decode reads every weight once per
token, so its ceiling is memory bandwidth divided by the bytes read per token. This host runs four
populated DDR3 channels (eight 8 GiB DIMMs, two per channel, read from the EDAC controller): about 51 to
60 GB/s peak depending on the DIMM speed, which is not recorded; take about 40 GB/s as attainable until
Tier 02C measures it. Using the file size as the bytes read per token (an overestimate: not every
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
to 0.7x. So 0.25x is a placeholder, not an ambition. [Tier 02C](TIER-02C-cpu-hot-path.md)
measures the host's attainable bandwidth, publishes Juno's attained weight bandwidth (bytes per token x
tg) beside every CPU tg reading, and restates this row after its step-2 breakdown, at the level the
breakdown supports and never below 0.25x.

**Restated 2026-10-10 (Tier 02C step 4): CPU tg >= 0.40x on every sweep model.** Two inputs. The host's
measured attainable read bandwidth is 42.4 GB/s (`docs/perf-compare/20261009T220314Z-host-read-bandwidth/`).
The CPU cost breakdown (`docs/perf-compare/20261010T032108Z-cpu-cost-breakdown/`) shows that about 99% of CPU
decode time is in the quantized matmul kernels. Those kernels are bound by instructions per weight (about 4 to
5 core cycles each), not by memory and not by the float add chain (splitting it recovers about 3.5%), and they
scale 5.9x from 1 to 12 threads. At the pinned CPU reference (`20261001T180241Z`), file size as bytes per token:

| Model | Reference tool share of 42.4 GB/s | Juno share | Juno tg ratio at a 25% share |
|---|---|---|---|
| tinyllama-1.1b | 40.8% | 5.0% | 0.61x |
| qwen2.5-3b | 54.3% | 5.4% | 0.46x |
| Phi-3.5-mini | 50.4% | 5.1% | 0.50x |
| mistral-7b | 57.3% | 5.3% | **0.44x** (binding) |

A tg ratio is Juno's bandwidth share over the reference tool's, so the model the reference tool runs closest
to the roofline binds. **25% is the share items 1 to 3 can plausibly reach.** It needs about 1.1 core cycles
per weight across six cores at the pinned 3.5 GHz. Integer dots over a once-quantized activation at this
host's 128-bit integer width should cost roughly 0.4 to 0.7, if C2 compiles them to packed multiply-adds;
Tier 02C item 1 measures that. Above it, the reference tool's 57% on the same host is the ceiling, and
reaching it needs native-quality code, which nothing measured yet supports. At 25% the binding model reads 0.44x. The
non-kernel time (about 1% today) grows to about 4% once the kernels are 4.6x faster, and adding
per-dispatch cost puts the reading near 0.41x. Hence 0.40x, not 0.44x. The restated row is at least 0.40x,
so no holding term is owed under the restatement rule. The term that would hold it lower is the instruction
count per weight at 128-bit integer width. If item 1's measurement shows the kernel cannot reach about 1.1
cycles per weight, the row is restated again (never below 0.25x) with that term named. `host-specific`: an
AVX2 host doubles the integer width and changes the cycle budget.

**Restated again 2026-10-11 (Tier 02C closing sweep, owner decision 4): CPU tg >= 0.25x on every sweep model.** The 0.40x
row assumed items 1 to 3 could reach a 25% bandwidth share, about 1.1 core cycles per weight. They reached 12.2% to 13.3%
(`docs/perf-compare/20261011T013059Z/`, pinned): the shipped Q4_K decode kernel runs at about 1.5 cycles per weight on one
thread and the Q6_K scalar kernel at about 2.7 (*corrected 2026-10-12: about 5.1 measured on one thread, `20261010T085329Z-cpu-integer-kernels`; see Tier 02E*), and those two kernels are about 93% of Mistral 7B decode. So, as this row's
own rule says, it is restated, to its floor: 0.25x, against 0.215x today (Qwen2.5-3B binding). **The term that holds it
below 0.40x is per-weight instruction cost in the Q4_K vector and Q6_K scalar decode kernels** on a host with 128-bit integer
SIMD and 16 vector registers. A faster Q6_K decode kernel and a Q4_K kernel nearer one cycle per weight are the levers.
*2026-10-11 (plan review): no tier owned them when this was written; [Tier 02E](TIER-02E-cpu-decode-kernels.md) now
does, and restates this row from its measured cycles per weight.* Tier 14 reports the row's distance. `host-specific`: an AVX2 or AVX-512/VNNI host widens the
integer lanes and moves the budget. *2026-10-27 (Tier 02E close, owner decision 9 there): met and retired at 0.25x, not
raised. Tier 02E's kernels reached 0.93 (Q4_K) and 2.13 (Q6_K) cycles per weight on one thread, 1.10 and 2.49 per core
at 12 threads; its pinned closing sweep reads 0.259x to 0.337x (Phi-3.5-mini binding), and Juno's decode attains 17.4% to 22.3%
of the host's 42.4 GB/s read bandwidth (file size as bytes per token; Mistral 7B 2.17 t/s x 4.37 GB = 9.5 GB/s; 12.2% to
13.3% at Tier 02C's close).
The term that would hold a higher row is still the Q4_K and Q6_K decode kernels' instruction cost at 128-bit integer
width; no tier owns more of it.*

**Post-plan anchor: GPU pp >= 0.40x (added 2026-09-30; not a gate of this plan).** *Promoted
2026-10-04 (plan review): 0.40x at `n_prompt=512` is now an end-of-plan target (see "End-of-plan targets,
machine-read" above), since Tier 01C's packed GEMM made it reachable and the unqualified 0.25x was met at
`n_prompt=128`. The post-plan anchor is restated as GPU pp >= 0.40x at `n_prompt=2048`, reported in Tier
14 as a distance, never as met or missed. The rest of this paragraph is the 2026-09-30 reasoning, kept as
the record.* The end-of-plan
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
processing) and the largest remaining one (CPU, Tier 02C), then outward through the feature surface in the order a request actually flows
(attention/context → KV → quantization → tokenization → sampling → model coverage → speculative
decoding → scheduling → parallelism → backend breadth → vision → LoRA → server/cluster surface),
closing with a documentation hardening pass.

Eleven rows do not sit where their integer would put them, and the table below — not the numbering —
is the running order (execution rule 1). **Tiers 01B and 01C** sit between 01 and 02, **Tiers 02B,
02C, 02D, 02E, 02F and 02G** between 02 and 03, and **Tiers 04B and 04C** between 04 and 05. **Tier 08 runs before Tier 06**: Tier 06 adds a `forwardVerify` override per
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

**Tiers 02B and 02C sit between 02 and 03 (added 2026-10-04, plan review).** Tier 02 held nine items,
three of which (context shifting, sliding windows, the Phi-3.5 LongRoPE remainder) carry no throughput
target, and under execution rule 1 every remaining GPU throughput lever waited on them. They became
**Tier 02B**, which runs after Tier 02 so that windowing is a parameter of Tier 02's tiled kernel, and
before Tier 03 because they settle what the KV cache holds. **Tier 02C** is Tier 10's CPU half (kernels,
allocation, threading, the bandwidth roofline), moved forward because CPU is now the largest gap in the
program (see "Why the CPU work runs third, not tenth" above). Tier 10 keeps ROCm and the backend
decision.

**Tier 02D sits between 02C and 03 (added 2026-10-08, owner decision, Tier 02 decision 15).** Tier 02's
closing sweeps met every GPU end-of-plan target, which were then raised (see "GPU targets raised again"
below). The work that has to reach the raised rows had no owner: Qwen2 decode outside the device region,
the host work around Phi-3's prefill window, and Phi-3's remaining decode cost. Tier 02D owns them. It
runs after Tier 02C because CPU is still the largest gap, and before Tier 03 because Tier 03 changes the
KV layout the prefill window's host KV write sits on.

**Tier 02E sits between 02D and 03 (added 2026-10-11, plan review).** Tier 02D closed with every GPU
end-of-plan row met and retired. The only active row left is CPU tg (>= 0.25x, 0.215x today), and this README
named its levers, a vector Q6_K decode kernel and a Q4_K kernel nearer one cycle per weight, while saying no
tier owned them. Every row after Tier 02D is feature breadth that cannot move a CPU number, so the plan would
have closed on its own largest gap without working on it. [Tier 02E](TIER-02E-cpu-decode-kernels.md) owns the
two kernels, the CPU integer kernels' fixed 128-bit SIMD width, and CPU prefill at 512 and 2048 tokens, which
had never been measured. It runs before Tier 03 for the reason Tier 02C did: Tier 04's mapped weight loading
threads a `MemorySegment` accessor through these kernels and should convert the ones this tier leaves.

**Tiers 02F and 02G sit between 02E and 03 (added 2026-10-13, owner decision).** This host has no AVX2,
no AMD GPU and no Windows CUDA install, so 18 tier files close hardware-dependent rows as
`NEEDS-AMD-HARDWARE` and Tier 02E closes its width row as `NEEDS-AVX2-HARDWARE`, and the bash harness that
would measure them does not run on Windows. [Tier 02F](TIER-02F-hardware-validation-kit.md) builds a Java
kit that a tester with that hardware runs with one command on Windows, Linux or macOS, returning one zip of
scrubbed results. [Tier 02G](TIER-02G-cuda-on-windows.md) makes Juno's CUDA backend run on Windows, and is
verified through the kit. Both run before Tier 03 so that every later tier adds the kit probes for its own
feature as it lands (Tier 02F's scope item 13), instead of having them retrofitted at Tier 14.

| Tier | Title | Gap analysis refs |
|---|---|---|
| [00](TIER-00-correctness-and-consistency.md) | Correctness & consistency audit | §2.1, §2.4, §2.5, §2.6, §2.7, §2.9 |
| [01](TIER-01-gpu-activation-residency.md) | GPU activation-residency redesign | §1.6, §2.8 |
| [01B](TIER-01B-prefill-throughput.md) | Prefill throughput | none — see that file's "Why this tier, why now" |
| [01C](TIER-01C-packed-kquant-matmul.md) | Packed K-quant prefill matmul (tiled `dp4a` GEMM for Q4_K/Q5_K/Q6_K; split out of 04C) | none — adjacent to §1.1; see that file's "Why this tier, why now" |
| [02](TIER-02-attention-long-context.md) | Attention & long context (tiled kernel, decode region; context policy split out to 02B) | §1.2 |
| [02B](TIER-02B-context-policy.md) | Context policy (context shifting, sliding windows, Phi-3.5 LongRoPE remainder; split out of 02) | §1.2 |
| [02C](TIER-02C-cpu-hot-path.md) | CPU hot path (integer kernels, allocation, threading; split out of 10) | §1.7 |
| [02D](TIER-02D-gpu-host-overhead.md) | GPU host overhead and decode-region coverage (Qwen2 in the region, Phi-3's prefill host work and decode cost; added 2026-10-08) | none — see that file's "Why this tier, why now" |
| [02E](TIER-02E-cpu-decode-kernels.md) | CPU decode and prefill kernels (vector Q6_K, Q4_K instruction cost, host SIMD width, CPU prefill length profile; added 2026-10-11) | §1.7 (CPU half, continued) |
| [02F](TIER-02F-hardware-validation-kit.md) | Hardware validation kit (one command on Windows, Linux or macOS, run by testers on hardware this host lacks; added 2026-10-13) | none — see that file's "Why this tier, why now" |
| [02G](TIER-02G-cuda-on-windows.md) | CUDA on Windows (library resolution per OS, display-driver behaviour, Windows launchers; added 2026-10-13) | none — see that file's "Why this tier, why now" |
| [03](TIER-03-kv-cache-maturity.md) | KV cache maturity | §1.3 |
| [04](TIER-04-quantization-coverage.md) | Quantization coverage (mapped weight loading first) | §1.1 |
| [04B](TIER-04B-tokenizer-fidelity.md) | Tokenizer fidelity (per-family splits and cross-engine parity; the key read and fail-closed path moved to Tier 01 as parity precondition 7) | none — see that file's "Why this tier, why now" |
| [04C](TIER-04C-packed-weight-matmul.md) | Packed-weight coverage (Tier 01C's packed matmul and residency widened to Tier 04's formats; explicit FP16 fallback) | none — adjacent to §1.1; see that file's "Why this tier, why now" |
| [05](TIER-05-sampling-grammar.md) | Sampling & grammar completeness | §1.4 |
| [08](TIER-08-model-architecture-breadth.md) | Model architecture breadth | §1.9, real files in `models/` |
| [06](TIER-06-speculative-decoding.md) | Speculative decoding expansion | §1.5 |
| [07](TIER-07-continuous-batching.md) | Continuous batching maturity | §1.3 (scheduling half) |
| [09](TIER-09-tensor-parallelism-multi-gpu.md) | Tensor parallelism & multi-GPU (and batched prefill on every cluster path, added 2026-10-02) | §1.8, §2.2 |
| [10](TIER-10-gpu-backend-breadth-cpu-simd.md) | GPU backend breadth (ROCm parity, backend decision; CPU half moved to 02C) | §1.7 |
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

**Row 1 caveat (CPU inference) before Tier 02C lands:** every tier from 00 through 02B, Tiers 01B and
01C included, uses row 1 as a *correctness* oracle only — `CpuMatVec`'s scalar path, since Tier 02C
(moved forward from Tier 10 on 2026-10-04) is what makes CPU inference actually fast. A tier's row-1
"PASS" before Tier 02C means "correct," not "final-performance-path verified." Tier 02C re-verifies,
rather than assumes, that its CPU changes preserve every earlier tier's row-1 correctness result —
integer accumulation over a quantized activation changes the numerics outright, and a different thread
count or work-splitting strategy can reorder a reduction — see that tier's own exit criteria. From Tier
02C on, row 1 is the final CPU path, and the standing CPU gate (see "Test infrastructure") holds its
throughput.

**Row 15 (JVM embedding facade) was added late — read this before marking it N/A.** `JunoPlayer`,
`LoraTrainer` and `JunoHttpClient` in `juno-player` are a public surface `CLAUDE.md` names, and a
capability wired through the CLI and both REST surfaces is still unreachable to an embedder if the
facade does not expose it. Before this row existed the facade appeared in this tree exactly once, in
Tier 00's execution record. Tiers 00 and 01 were written and executed without it and do not carry
it. Tier 01B's scope is internal throughput with no new embedder-invocable capability, so it is N/A
there. From Tier 02 onward the row is resolved like any other — PASS, N/A with a reason, or
FAIL-CLOSED. Its table row is already present in the three tiers where it is most clearly
load-bearing: Tier 02B (context-shift opt-in, moved from Tier 02 on 2026-10-04), Tier 05
(sampler-chain selection) and Tier 12 (per-request LoRA selection); Tier 02C carries it for `--threads`. Any other tier from 02 onward that adds a user-invocable capability
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

**Standing throughput and allocation gate (added 2026-10-04 as the "Standing CPU and allocation gate",
plan review; GPU lane and the binding CPU models added 2026-10-11, plan review; every tier with a perf
gate from Tier 02 on).** Closed tiers' records cite it by its earlier name. Three kinds of regression had
no gate. GPU throughput: once every GPU end-of-plan row was retired (Tier 02D), llama.cpp-relative ratios
are reported and never gated, and only four of the then-open tiers (04B, 08, 09, 10) held GPU t/s to a
number, so a GPU regression in Tier 03 (whose own gate sits on the paged path, while every sweep reads
`static` dense KV) or Tier 04 would close cleanly. CPU throughput: row 1 was "correctness only" until
the CPU tier, and every per-tier no-regression A/B read the GPU lane, so a CPU regression introduced in
Tier 03 or 04 would first be seen when the CPU tier measured it. And host allocation and GC: every result
JSON publishes `jfr.allocated_bytes_per_token`, `jfr.gc_pause_total_ms_in_token_span` and
`jfr.top_methods`, and no tier held them to a number. Tiers 03 (KV storage), 04 (mapped loading, which
threads a `MemorySegment` accessor through every CPU kernel; a `MemorySegment` read in a scalar loop is
not free for C2) and 07 (the scheduler) all change one or the other. So every such tier, before it
closes, runs this gate against the pre-tier jar and records it in its own file:

- **CPU lane, same-hour interleaved A/B**: the pre-tier jar and the candidate, alternating A B A B A B,
  each invocation `compare-llama-cpp.sh --cpu --pin-clocks --juno-jar <jar> --juno-reps 1 --juno-warmup 2
  --reps 1 --no-publish`, on all four sweep models at `n_prompt=128`. **Threshold**: Juno CPU tg and
  pp t/s **>= 0.95x** the pre-tier medians. *(2026-10-11: was tinyllama and mistral-7b only. Qwen2.5-3B
  binds the CPU tg row and Phi-3.5-mini runs its own handler, so a regression on either passed.)*
- **GPU lane, same-hour interleaved A/B** *(added 2026-10-11)*: the same alternation, each invocation
  `compare-llama-cpp.sh --gpu --pin-clocks --juno-jar <jar> --juno-reps 1 --juno-warmup 2 --reps 1
  --no-publish`, on all four sweep models at `n_prompt=512` (the length the GPU pp rows were scored at, and
  long enough that a prefill-window regression is not lost in the first decode steps). **Threshold**: Juno
  GPU tg and pp t/s **>= 0.95x** the pre-tier medians. A tier whose own threshold already holds GPU t/s
  on every sweep model to at least this bound from the same A/B (Tiers 04B, 08, 09, 10) scores it once.
- **Allocation and GC, both lanes**: from those CPU runs and from the tier's own GPU A/B, per model and
  lane, medians of three: `jfr.allocated_bytes_per_token_in_recording` **<= 1.10x** the pre-tier reading
  (a side without a reading, or a pre-tier reading of 0, is not scorable and fails), and
  `jfr.gc_pause_total_ms_in_token_span` **<= 1.25x** the pre-tier reading (where the pre-tier reading is
  under 5 ms, the bound is **<= 5 ms**, since a ratio of near-zero pause totals measures nothing).
- **Hot methods, both lanes**: the top 10 of `jfr.top_methods` (`jdk.ExecutionSample`) for both jars,
  side by side in the tier's record, so a shift in where time goes is visible even where throughput
  holds; on the GPU lane also the top 10 of `jfr.top_native_methods` (`jdk.NativeMethodSample`, added
  2026-10-11). `ExecutionSample` does not sample a thread inside a native call, and on a GPU run most of
  the forward pass is driver downcalls, so the Java histogram alone describes only the host work between
  them. The native histogram also counts threads idle in a native wait (socket polls, the gRPC event
  loop), so compare the two jars' rows by name rather than reading the top entry as the cost. Reported,
  not gated.

A tier whose change is scoped to the CPU path (Tier 02C) scores its CPU half under its own, stricter
thresholds; the allocation half still applies. `check-plan-thresholds.sh` check 5 fails an open tier with
a perf gate that never names this gate.

*Allocation field changed 2026-10-26 (owner decision 6, [Tier 02E](TIER-02E-cpu-decode-kernels.md)).* The gate read
`jfr.allocated_bytes_per_token`, which sums each thread's running total from the thread's start; the harness records
only the measured request, but its long-lived server and worker threads carry model load and both warmup requests
into that total, so a build that allocates more while compiling (new vector kernels before C2 has compiled them)
failed the gate with a measured request that allocated what the pre-tier build's did. The gate now reads each
thread's growth inside the recording. The old field is still published beside it, and both sides of an A/B are
extracted by the harness tree's own jar, so a pre-tier jar older than the field still gets it. Re-read on Tier 02E's
first closing run, the new field is the stricter of the two where the request itself allocates (generation lane: 325x
to 482x where the old field read 5.3x to 7.0x; [`20261026T204605Z-cpu-kernels-allocation-reread`](../perf-compare/20261026T204605Z-cpu-kernels-allocation-reread/INDEX.md)).
Its known limit: threads are sampled every 500 ms, so a request shorter than about a second undercounts; both
sides of an A/B are read the same way.

**llama.cpp-relative gate (applies from Tier 01 onward).** `compare-lora.sh`/`compare-vision.sh`
only ever compare Juno against its own prior baseline — neither tells you whether the actual stated
goal of this plan (closing the gap with llama.cpp) is moving. Every tier whose scope includes the
forward pass, MatVec, GPU residency, batching, quantization, or KV paths (at minimum: Tiers 01, 01B, 01C,
02, 02B, 02C, 02D, 02E, 03, 04, 04B, 04C, 06, 07, 08, 09, 10) additionally re-runs
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
   (this requires the hot-path threading work in Tier 02C item 3, formerly Tier 10 item 5, to expose one — until it does,
   record the effective Juno parallelism in the run metadata and state the mismatch in every
   published INDEX rather than leaving it implicit).
   *Amended 2026-09-30: the interim does not have to wait for Tier 10.* Every CPU hot-path kernel
   (`SimdThreadPool.forEachRow` and the `matVecQ*raw` family) dispatches through
   `IntStream.parallel()` on `ForkJoinPool.commonPool()`, whose size the JVM already takes from
   `-Djava.util.concurrent.ForkJoinPool.common.parallelism`. The calling thread joins the work as one
   more worker, so passing `N_THREADS - 1` gives `N_THREADS` threads on the hot path, matching the
   reference tool's `-t`. [Tier 01B](TIER-01B-prefill-throughput.md) item 7 wires that into
   `compare-llama-cpp.sh` and records `juno_threads` in `host.json`. It is a harness setting, not
   the product control: Tier 02C item 3 (formerly Tier 10 item 5) still ships `--threads` and may replace the common pool, at
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

*Decided 2026-10-13 (owner, recorded in [Tier 02E](TIER-02E-cpu-decode-kernels.md)'s record and
[Tier 02F](TIER-02F-hardware-validation-kit.md)'s scope item 14): a workflow for the hardware validation kit
on free hosted runners is approved, as part of Tier 02F.* It runs the kit's `quick` CPU suite on Linux x64
and arm64, Windows and macOS on manual dispatch and weekly, and the kit probes' self-tests on pull requests
that touch the kit. The 2026-10-01 decision stands for `ci.yml` itself, and CI is still not an exit
criterion for any other tier. Committing the workflow file is the owner's action.

**No CI runs in this repository today.** `.github/workflows/ci.yml` exists in the working tree but is
untracked, so it has never run (see the 2026-09-30 and 2026-10-01 notes above; *corrected 2026-10-11, plan
review: this paragraph said `.github/` held only a `modernize/` directory*) — tier-gating (execution rule 1: don't start Tier N+1 until Tier N's exit
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
in this environment; there is no ROCm/AMD hardware. *(2026-10-13: hardware this host lacks is reached
through [Tier 02F](TIER-02F-hardware-validation-kit.md)'s kit, run by testers; their hosts are recorded in
`INVENTORY.md`'s "Borrowed hardware" table once the first results arrive.)* Tiers that touch ROCm code paths are
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
  just "no unexplained regression"), and, for Tiers 01, 01B, 01C, 02, 02B, 02C, 02D, 02E, 03, 04, 04B, 04C, 06, 07, 08, 09, 10, an
  accompanying `compare-llama-cpp.sh` run recording the current Juno/llama.cpp ratio and reading it
  against the program target and that tier's intermediate milestone, if it has one.
- **The published API contract is updated in the same change as the code.**
  `api/src/main/resources/openapi.yaml`, `api/src/main/resources/juno-api.yaml` and
  `api/src/main/proto/inference.proto` are the contract, and no tier may add an endpoint, a request
  or response field, or an RPC without updating them. This is load-bearing for Tiers 02B (context-shift
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
