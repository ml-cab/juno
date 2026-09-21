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

1. **Claim-vs-reality audit**: for every load-bearing architectural claim in `docs/agent-arch.txt`
   and `CLAUDE.md` (not every sentence — the ones that describe what a component *does*, the same
   category of claim that turned out to be wrong for `TensorShardContext`, `RegistryService`, and
   the CPU SIMD description before this plan started), verify it against an actual current call
   site or test. Fix or remove any claim that doesn't check out.
2. **Known-limitations ledger**: add a short, consistently-formatted "Known limitations" section
   next to the code it describes (a class-level javadoc block, or a short markdown file colocated
   with the module) for each major subsystem (`node`, `coordinator`, `kvcache`, `sampler`, `lora`,
   `vision`), stating current real limitations in one place instead of scattered across javadoc,
   `CHANGELOG.md`, and `docs/howto.md` caveats. This is a discoverability aid, not new analysis —
   it should mostly restate what's already true post-Tier-13, gathered into one findable spot per
   subsystem.
3. **Full doc consistency pass**: `docs/howto.md`, `docs/performance.md`, `README.md`,
   `docs/agent-arch.txt` all get read end-to-end against current source and corrected where drifted.

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
   scope) before closing this tier.

## Tests to write/upgrade before implementation

This tier is documentation-only, so "tests" here means verification tooling rather than JUnit
cases:

- **New bash script**: `scripts/performance-tests/smoke-tier14-doc-consistency.sh` (naming kept
  consistent with the rest of the plan's smoke-script convention even though this one checks docs,
  not runtime behavior) — greps for competitor product names outside the allowed directories and
  fails if any are found; optionally checks for a minimal set of known-stale phrases if any were
  identified during the audit, so the check can catch a regression if someone reintroduces one of
  the fixed claims later.
- No unit/integration test changes expected, since no code behavior changes in this tier — if the
  audit reveals a real behavioral bug, that goes into a newly-filed tier's test plan instead, not
  here.

## Models needed

None — this tier is documentation and static analysis only.

## Exit criteria

- [ ] Every load-bearing architectural claim in `docs/agent-arch.txt`/`CLAUDE.md` has a verified
      current call site, or has been corrected/removed.
- [ ] Known-limitations ledger exists for `node`, `coordinator`, `kvcache`, `sampler`, `lora`,
      `vision`.
- [ ] `docs/howto.md`, `docs/performance.md`, `README.md` read-through complete, drift corrected.
- [ ] Competitor-product-name grep passes clean outside the allowed directories.
- [ ] `CHANGELOG.md` entry added summarizing the audit and what was corrected.
- [ ] This plan tree (`docs/gap-closure-plan/`) is itself left in place as a historical record —
      not deleted — since it documents the reasoning behind thirteen tiers of real change; consider
      whether to add a final "plan complete" note to this README once Tier 14 closes.
