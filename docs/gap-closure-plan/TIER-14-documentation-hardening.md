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
   `vision`), stating current real limitations in one place instead of scattered across javadoc,
   `CHANGELOG.md`, and `docs/howto.md` caveats. This is a discoverability aid, not new analysis —
   it should mostly restate what's already true post-Tier-13, gathered into one findable spot per
   subsystem.
3. **Full doc consistency pass**: `docs/howto.md`, `docs/performance.md`, `README.md`,
   `docs/agent-arch.txt` all get read end-to-end against current source and corrected where drifted.
4. **Final llama.cpp-relative scorecard**: pull every `compare-llama-cpp.sh` run published per tier
   under README's "llama.cpp-relative gate" (Tiers 01, 02, 03, 04, 06, 07, 08, 09, 10) into one table
   in this tier's own file — model, quant, hardware, Juno tg/pp, llama.cpp tg/pp, ratio — so the
   plan's actual stated goal (closing the gap with llama.cpp) has a single, final, honest answer
   instead of being implied by eight separate tier-local perf-compare directories nobody has
   aggregated. Report the trend (did the ratio improve, tier over tier, or not) as plainly as
   `docs/performance.md` already reports individual negative results (0.52x draft-model, 0.86x
   continuous) — an unchanged or worsened ratio on some workload is a valid, reportable outcome here,
   not a reason to withhold the table.

### Out of scope

- Any new feature work — this tier is purely documentation and light structural cleanup (e.g.
  deleting a stale comment), not code behavior changes. If the audit turns up a real bug (code does
  something other than what any doc claims, and the actual behavior is wrong), file it as a new,
  separately-scoped tier rather than fixing it inline here — this tier's job is to make claims
  match reality, not to also silently change reality.

## Cross-surface compatibility checklist

Not applicable in the usual sense — this tier makes no behavioral changes, so there's no
cross-surface regression risk. The equivalent check for this tier is: does every one of the 14
compatibility-checklist surfaces have accurate, current documentation describing its actual
capabilities as of the end of Tier 13? Confirm this explicitly for each surface rather than
skipping the checklist format entirely.

| # | Surface | What "accurate documentation" means here |
|---|---|---|
| 1-14 | (all surfaces, per the shared checklist in [`README.md`](README.md)) | Each surface's current, real capability (as landed by Tiers 00-13) is stated correctly in at least one production doc, with no stale claim contradicting it elsewhere. |

## Implementation steps

1. Enumerate every load-bearing architectural claim across `docs/agent-arch.txt` and `CLAUDE.md`
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
  not runtime behavior) — greps for competitor product names *and* for `Tier [0-9]+`/`Infra tier`
  outside the allowed directories, and fails if any are found; optionally checks for a minimal set
  of known-stale phrases if any were identified during the audit, so the check can catch a
  regression if someone reintroduces one of the fixed claims later.
- No unit/integration test changes expected, since no code behavior changes in this tier — if the
  audit reveals a real behavioral bug, that goes into a newly-filed tier's test plan instead, not
  here.

## Models needed

None — this tier is documentation and static analysis only.

## Exit criteria

- [ ] Every load-bearing architectural claim in `docs/agent-arch.txt`/`CLAUDE.md`/`CHANGELOG.md` has
      a verified current call site, or has been corrected/removed.
- [ ] Known-limitations ledger exists for `node`, `coordinator`, `kvcache`, `sampler`, `lora`,
      `vision`.
- [ ] `docs/howto.md`, `docs/performance.md`, `README.md` read-through complete, drift corrected.
- [ ] Competitor-product-name grep *and* `Tier [0-9]+`/`Infra tier` grep both pass clean outside the
      allowed directories.
- [ ] Final llama.cpp-relative scorecard published in this tier's file, aggregating every
      `compare-llama-cpp.sh` run from Tiers 01-13, with an honest statement of whether the ratio
      improved.
- [ ] `CHANGELOG.md` entry added summarizing the audit and what was corrected.
- [ ] This plan tree (`docs/gap-closure-plan/`) is itself left in place as a historical record —
      not deleted — since it documents the reasoning behind thirteen tiers of real change; consider
      whether to add a final "plan complete" note to this README once Tier 14 closes.
