# Tier 14: Documentation hardening

Status: not started
Gap analysis refs: §3 (the "how Juno was built" synthesis)

## Objective

Close out the plan with a systematic pass confirming that every architectural claim in
`docs/agent-arch.txt`, `CLAUDE.md`, `docs/howto.md`, and `README.md` has a real, current call site
— the same kind of audit Tier 00 did for a handful of known items, now applied comprehensively
across the whole doc set now that thirteen tiers of real feature work have landed and inevitably
shifted some of what used to be true. Also establish the lightweight "known limitations" ledger
per major subsystem that the gap analysis recommended, so this kind of drift is caught earlier next
time instead of accumulating silently across another dozen tiers.

## Why this tier, why now

Every prior tier's exit criteria already require its own doc updates — this tier is not "do all the
doc work at the end instead of incrementally" (that would violate the spirit of rule 2, cover
everything as you go). It exists because a full-codebase claim-vs-reality audit, done once at the
end of the whole plan, catches cross-tier drift that no single tier's own doc update would think to
check (e.g., a claim in `docs/agent-arch.txt` written before Tier 03 that nothing since then
happened to touch, but that Tier 03's changes quietly invalidated).

## Scope

### In scope

1. **Claim-vs-reality audit**: for every load-bearing architectural claim in `docs/agent-arch.txt`,
   `CLAUDE.md`, and `CHANGELOG.md` (not every sentence — the ones that describe what a component
   *does*, the same category of claim that turned out to be wrong for `TensorShardContext`,
   `RegistryService`, the Hazelcast-backed leader/standby election claim in `CHANGELOG.md`'s
   "Actors — Design Decisions" section if Tier 00 didn't already catch it, and the CPU SIMD
   description before this plan started), verify it against an actual current call site or test. Fix
   or remove any claim that doesn't check out. `CHANGELOG.md` is included here specifically because
   it is excluded from item 3's "full doc consistency pass" below (a changelog's older entries are
   expected to be historical, but a claim describing a component's *design* rather than a dated
   event — like the Actors section — is exactly the kind of claim this audit exists to catch,
   regardless of which file it's in).
2. **Known-limitations ledger**: add a short, consistently-formatted "Known limitations" section
   next to the code it describes (a class-level javadoc block, or a short markdown file colocated
   with the module) for each major subsystem (`node`, `coordinator`, `kvcache`, `sampler`, `lora`,
   `vision`, `tokenizer`, `api`), stating current real limitations in one place instead of scattered across javadoc,
   `CHANGELOG.md`, and `docs/howto.md` caveats. This is a discoverability aid, not new analysis —
   it should mostly restate what's already true post-Tier-13, gathered into one findable spot per
   subsystem.
3. **Full doc consistency pass**: `docs/howto.md`, `docs/performance.md`, `README.md`,
   `docs/agent-arch.txt` all get read end-to-end against current source and corrected where drifted.
4. **Final llama.cpp-relative scorecard**: pull every `compare-llama-cpp.sh` run published per tier
   under README's "llama.cpp-relative gate" (Tiers 01, 01B, 02, 03, 04, 04B, 04C, 06, 07, 08, 09, 10
   — this list must match the one in [`README.md`](README.md)'s llama.cpp-relative gate paragraph
   verbatim; 04C was missing from it once already, which would have dropped the tier that changes the
   GEMM operand for every later pp figure out of the plan's single final answer)
   into one table in this tier's own file — model, quant, hardware, Juno tg/pp, llama.cpp tg/pp, ratio — so
   the plan's actual stated goal (closing the gap with llama.cpp) has a single, final, honest answer
   instead of being implied by eleven separate tier-local perf-compare directories nobody has
   aggregated. Report the trend (did the ratio improve, tier over tier, or not) as plainly as
   `docs/performance.md` already reports individual negative results (0.52x draft-model, 0.86x
   continuous) — an unchanged or worsened ratio on some workload is a valid, reportable outcome here,
   not a reason to withhold the table. **Score the final row against the program target table in
   [`README.md`](README.md)**, per metric, as met or missed with the actual number — including the
   intermediate milestones assigned to Tiers 01B, 04 and 10, so a milestone that was missed mid-plan
   and never recovered is visible rather than averaged away. Mark clearly which runs were taken
   before the benchmark-parity preconditions landed and which after; do not compare across that
   boundary in the same column. The same applies to Tier 04B: mark which runs were taken before the
   tokenizer fidelity work changed token counts, since `prompt_tokens` is the denominator of every
   pp and tg figure in the table.

   **List the out-of-tier changes alongside the tiers.** Execution rule 9 requires any hot-path or
   launcher change landing outside a tier's scope to be recorded in the then-active tier's execution
   record. Pull those entries into this scorecard as their own rows — commit, what it touched, and
   whether it was a measurement boundary — so a reader can tell which ratio movements belong to a
   tier's work and which to a change nobody planned. Two already exist (`c91f879`, `1f90b68`, both
   recorded against Tier 01); if the scorecard's rows do not account for every published
   `docs/perf-compare/` directory in the plan's date range, a change went unrecorded and that is
   itself a finding for this tier to report.

   **Score the pp row against Tier 01's re-derived target, not the README's original `>= 0.15x`.**
   That figure was set against readings taken with `raw_prompt: 0`, where llama-bench prefilled 128
   tokens and Juno prefilled 20 to 30 — never a like-for-like measurement. Tier 01 re-derives the
   target from its parity-corrected re-baseline and records the new number in its own file; this
   scorecard scores against that one and states plainly that the original was retired and why. A
   scorecard that scores a parity-corrected result against a pre-parity target would report a
   regression that is really a measurement correction, which is the opposite of what this table is
   for.

### Out of scope

- Any new feature work — this tier is purely documentation and light structural cleanup (e.g.
  deleting a stale comment), not code behavior changes. If the audit turns up a real bug (code does
  something other than what any doc claims, and the actual behavior is wrong), file it as a new,
  separately-scoped tier rather than fixing it inline here — this tier's job is to make claims
  match reality, not to also silently change reality.

## Cross-surface compatibility checklist

Not applicable in the usual sense — this tier makes no behavioral changes, so there's no
cross-surface regression risk. The equivalent check for this tier is: does every one of the 15
compatibility-checklist surfaces have accurate, current documentation describing its actual
capabilities as of the end of Tier 13? Confirm this explicitly for each surface rather than
skipping the checklist format entirely.

| # | Surface | What "accurate documentation" means here |
|---|---|---|
| 1-15 | (all surfaces, per the shared checklist in [`README.md`](README.md)) | Each surface's current, real capability (as landed by Tiers 00-13) is stated correctly in at least one production doc, with no stale claim contradicting it elsewhere. |

Row 15 (the JVM embedding facade — `JunoPlayer`, `LoraTrainer`, `JunoHttpClient`) deserves
particular attention here, because it was added to the shared checklist late and Tiers 00, 01 and
01B do not carry it. Whatever capability those three landed is reachable from the CLI and from both
REST surfaces; this tier confirms the facade's own documentation says what an embedder can and
cannot call.

## Implementation steps

1. Enumerate every load-bearing architectural claim across `docs/agent-arch.txt`, `CLAUDE.md` and
   the published API contract (`api/src/main/resources/openapi.yaml`, `juno-api.yaml`,
   `api/src/main/proto/inference.proto`)
   (a claim is "load-bearing" if a reader — human or agent — would make a different decision
   believing it true vs. false; skip purely descriptive/cosmetic text).
2. For each claim, find and cite the current call site/test that verifies it, or mark it for
   correction.
3. Fix every claim that doesn't check out.
4. Write the known-limitations ledger per subsystem.
5. Full read-through of `docs/howto.md`, `docs/performance.md`, `README.md` against current
   source; fix drift.
6. Confirm rule 4 (no competitor names outside this plan tree) still holds across every file
   touched — do a final grep pass for `llama.cpp`/`vLLM`/`llama-server` across everything outside
   `docs/gap-closure-plan/`, `docs/perf-compare/`, and `docs/infra-plan/` (pre-existing, out of
   scope) before closing this tier. In the same pass, re-run the `Tier [0-9]+`/`Infra tier` grep
   Tier 00 first cleaned up (item 9 there) across every `src/main` tree, to confirm none of Tiers
   01-13 reintroduced the pattern while citing their own tier number in a new comment.

## Tests to write/upgrade before implementation

This tier is documentation-only, so "tests" here means verification tooling rather than JUnit
cases:

- **New bash script**: `scripts/performance-tests/smoke-tier14-doc-consistency.sh` (naming kept
  consistent with the rest of the plan's smoke-script convention even though this one checks docs,
  not runtime behavior). It must fail, not warn, on each of:
  - a competitor product name outside the allowed directories;
  - `Tier [0-9]+`/`Infra tier` outside the allowed directories;
  - **any tier file under `docs/gap-closure-plan/` containing `Perf gate` but no `**Threshold`
    block carrying a numeral and a comparison operator** — this is what makes execution rule 7
    self-enforcing rather than a rule everyone agrees with and half the tiers ignore. Match on
    `Perf gate`, not `Perf gate (required)`: three spellings of that heading are in use. Exclude
    this tier's own file from the scan — it names the string in order to describe the check, and a
    check that fails on its own specification is noise;
  - **a mismatch between the tier list in this file's llama.cpp-relative scorecard and the tier list
    in [`README.md`](README.md)'s llama.cpp-relative gate paragraph.** Extract both comma-separated
    lists and fail on any difference. The two drifted apart once already, when Tier 04C was added to
    the README's list in two places and not to this one, so this check is a regression guard for a
    defect that actually happened rather than a hypothetical one;
  - any known-stale phrase identified during the audit, so a fixed claim cannot quietly return.

  Run it against this tree *before* starting the audit as well as after, so the rule-7 check is a
  finding this tier reports rather than something it silently fixes.
- No unit/integration test changes expected, since no code behavior changes in this tier — if the
  audit reveals a real behavioral bug, that goes into a newly-filed tier's test plan instead, not
  here.

## Models needed

None — this tier is documentation and static analysis only.

## Exit criteria

- [ ] Every load-bearing architectural claim in `docs/agent-arch.txt`/`CLAUDE.md`/`CHANGELOG.md` has
      a verified current call site, or has been corrected/removed.
- [ ] Known-limitations ledger exists for `node`, `coordinator`, `kvcache`, `sampler`, `lora`,
      `vision`, `tokenizer` and `api` — the last two added because nothing in Tiers 00-13 audited
      them by default: the `tokenizer` module had no owning tier until Tier 04B, and the published
      API contract (`openapi.yaml`, `juno-api.yaml`, `inference.proto`) is the one artifact a
      reader is most likely to trust without checking.
- [ ] `api/src/main/resources/openapi.yaml`, `juno-api.yaml` and `api/src/main/proto/inference.proto`
      audited against the implemented surface: every endpoint, field and RPC the code serves is
      declared, and nothing declared is unimplemented. Six tiers added surface here; this is the
      backstop for any that skipped the feature-complete rule.
- [ ] `smoke-tier14-doc-consistency.sh`'s rule-7 check passes — every tier file with a `Perf gate`
      has a numeric threshold.
- [ ] `smoke-tier14-doc-consistency.sh`'s tier-list check passes — this file's scorecard tier list
      and [`README.md`](README.md)'s llama.cpp-relative gate list are identical.
- [ ] `docs/howto.md`, `docs/performance.md`, `README.md` read-through complete, drift corrected.
- [ ] Competitor-product-name grep *and* `Tier [0-9]+`/`Infra tier` grep both pass clean outside the
      allowed directories.
- [ ] Final llama.cpp-relative scorecard published in this tier's file, aggregating every
      `compare-llama-cpp.sh` run from Tiers 01-13, with an honest statement of whether the ratio
      improved, and every row of the program target table scored met or missed with its actual
      number.
- [ ] The scorecard's out-of-tier rows (execution rule 9) account for every published
      `docs/perf-compare/` directory in the plan's date range; any directory with no owning tier and
      no out-of-tier row is reported as an unrecorded change rather than left unexplained.
- [ ] `CHANGELOG.md` entry added summarizing the audit and what was corrected.
- [ ] This plan tree (`docs/gap-closure-plan/`) is itself left in place as a historical record —
      not deleted — since it documents the reasoning behind thirteen tiers of real change; consider
      whether to add a final "plan complete" note to this README once Tier 14 closes.
