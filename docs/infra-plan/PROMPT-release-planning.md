# Release planning: the next release (this week) and the one after (in a month) (written 2026-11-18)

Copy everything below the line into a new agent session.

---

## Task

The owner will cut a Juno release at the end of this week or the start of next (Fri 2026-11-20 to Tue 2026-11-24),
and another about a month later (around 2026-12-18 to 2026-12-22). Analyse `docs/gap-closure-plan/` and the code, then
write one release plan per release into `docs/gap-closure-plan/`, each placed in the running order directly after the
tier it follows. Each plan says:

1. **Where the cut is**: after which tier (or which commit), and why that point is sound. For the first release this
   decides what has to happen before the date. For the second it decides which tiers to aim at.
2. **What must be done before release**: ordered work items with estimates, each with a done-condition.
3. **How to do the release**: the procedure, gates and artifacts for this repo, following standard practice for a
   Maven multi-module JVM project published to Maven Central and GitHub.

You are planning, not implementing. You write planning docs only. You change no code, build files, shipped docs or
git state (see "Constraints").

## Why this exists

The gap-closure tree sequences tiers but has no release points. The last release is `v0.1.1` (2026-08-24). Since then,
102 commits of performance and KV work have landed on a feature branch that main does not have. Execution rule 1 says
partial, half-wired features are not acceptable stopping points between tiers. A release is that kind of stopping
point, so where it is cut has to be defended against that rule.

## Facts already established (verified 2026-11-18; re-check anything you rely on)

**Calendar.** Today is Wednesday 2026-11-18. Release 1 is 2 to 6 days away. Release 2 is about 30 days away and runs
into the year-end holidays.

**Git.**
- Branch `67-gap-inference`, HEAD `c607e926`, 106 commits ahead of `main`. `main` ends at `9c2da4c5`, the post-0.1.1
  plugin bump.
- Tags: `v0.1.0` (2026-04-29, made by `maven-release-plugin`) and `v0.1.1` (2026-08-24).
- The working tree has about 69 modified, uncommitted files. They are Tier 03's item 6, in progress in another session
  (owner decisions 10 (a), 11 (a), 12 (a), recorded 2026-11-18):
  - a request-length hint passed to every stage, including a new optional field on the gRPC `ForwardRequest`;
  - `f16` changes to store real IEEE halves, and float32 storage moves to a new `f32` type. This changes what an
    existing flag value means, so it is an upgrade-note item if it ships.
- **Do not touch, stash, build over or commit this work.** Analyse committed history with `git show`, `git log`, and
  `git archive <sha> | tar -x -C <scratchpad>/...`.

**Version and release tooling (root `pom.xml`).**
- The version is `0.1.2`, with no `-SNAPSHOT`, and there is no `v0.1.2` tag. `<scm><tag>` still says `v0.1.0`.
  `maven-release-plugin` expects a `-SNAPSHOT` version, so the release procedure has to resolve this.
- `maven-release-plugin` 3.1.1: `tagNameFormat v@{project.version}`, `pushChanges=false`, `localCheckout=true`,
  arguments `-DskipTests -DskipITs`, goals `install`.
- Profile `release-sign`: gpg signing at the install phase.
- Profile `central-publish`: `central-publishing-maven-plugin` 0.9.0, server id `central`, `juno-kit` excluded.
- `juno-bom` aligns versions. `juno-master` and `juno-node` are shaded executables. `juno-player` emits `*-shaded.jar`.
  `dist/juno-0.1.0/` exists (gitignored). Check what it holds and how 0.1.0 artifacts were staged.

**Release-facing files.**
- `RELEASE_NOTES.md` has user-facing 0.1.1 and 0.1.0 sections. Its style is a headline, one measured table, then
  "What else shipped".
- `CHANGELOG.md` is a session log ("Session 150 ..."), not a per-version changelog.
- `juno-documentation/` is the MyST book published at ml.cab/juno-documentation.
- Also relevant: `SECURITY.md`, `GOVERNANCE.md`, `CLA.md`, `LICENSE`, `Legal.md`.


**Tier state** (`Status` line of each file; the running order is the README index, not the numbers).
- Complete: 00, 01, 01B, 01C, 02, 02B, 02C, 02D, 02E, 02H.
- In progress:
  - 02F, the hardware kit. Its remaining criteria wait on testers' Windows and macOS zips.
  - 03, KV cache maturity. Begun 2026-11-09. Step 6 is under way. Read its exit criteria for the count met.
- Postponed: 02G, CUDA on Windows. It needs the Windows evaluation on this host. Its Linux-side work is done.
- Not started, in running order: 04, 04B, 04C, 05, 08, 06, 07, 09, 10, 11, 12, 13, 14.
- Pace, from Status lines:
  - 02D: 2026-10-08 to 10-12
  - 02E: 2026-10-11 to 10-27
  - 02H: 2026-10-31 to 11-09
  - 03: 2026-11-09 to open

  Work out the real velocity from the execution-record dates. Size tiers 04, 04B, 04C and 05 from their scope and
  implementation-step lists, not their line counts.

**Plan-tree tooling.** `scripts/performance-tests/check-plan-thresholds.sh` scans only `README.md` and `TIER-*.md` in
the tree. A file named `RELEASE-*.md` is not checked by it. The README rows you add are checked.

## Read first

1. `CLAUDE.md`, and the memory index (`MEMORY.md`). These memories bear directly on this task:
   - `juno-planning-trees-route-dont-fork`
   - `no-tier-numbers-in-file-names`
   - `owner-gate-ready-file-waiter`
   - `never-rebuild-while-tests-run`
   - `github-org-billing-state`
   - `tester-folder-unscrubbed`
   - `hardware-kit-and-windows-cuda`
2. `docs/gap-closure-plan/README.md` in full. Pay most attention to:
   - execution rules 1 to 10, especially 1, 2, 3, 6, 7 and 9;
   - "Program target";
   - "Tier index and ordering rationale";
   - "Cross-surface compatibility checklist";
   - "Test infrastructure", which holds the smoke matrix and the standing throughput and allocation gate;
   - "Feature-complete definition".
3. `TIER-03-kv-cache-maturity.md`: Scope, Exit criteria, and the execution record from 2026-11-09 on. Note which items
   are committed, whether each is wired on every surface or fails closed, and what is still open.
4. `TIER-02F-...md` and `TIER-02G-...md`: what is open, and what a release has to say about Windows and macOS.
5. Scope and exit criteria of the next tiers: `TIER-04`, `TIER-04B`, `TIER-04C`, `TIER-05`, then `TIER-08`.
6. `INVENTORY.md` (models on disk) and `TARGET-HISTORY.md`.
7. `RELEASE_NOTES.md`, the head of `CHANGELOG.md`, the root `pom.xml` release config, both workflows, `SECURITY.md`,
   and `docs/perf-compare/README.md` (baseline host, directory naming).
8. `git log --oneline v0.1.1..c607e926`, to build the user-visible change inventory since 0.1.1:
   - new or changed flags and defaults;
   - new model or KV types;
   - API and proto changes;
   - launcher changes, such as heap derived from model size;
   - performance changes, with published numbers.

## Steps

1. **Change inventory since v0.1.1.** List every user-visible change on HEAD, with its commit and the shipped doc that
   describes it (`docs/howto.md`, `README.md`, `docs/performance.md`). Flag:
   - every changed default and every flag whose meaning changed (upgrade notes);
   - every wire-format change (proto, `.lora` checkpoint, serialized KV) and whether it stays backward compatible;
   - every feature that is committed but not complete on every surface, and not failing closed. Use the README
     checklist. These block a release cut at that commit.
2. **Release 1: choose the cut.** Weigh at least these options, recommend one and say why:
   - (a) HEAD `c607e926` or another committed point after it, shipping Tier 03's committed items. Valid only if each
     one passes step 1's surface check.
   - (b) Finish Tier 03 item 6 first, then cut. Judge whether that fits the date from the item's remaining scope and
     the gates it carries.
   - (c) An earlier point, such as Tier 02H's close. Weigh what it drops against how easily it can be cut, given that
     Tier 03 commits follow it on the same branch.

   Then lay out a dated schedule from today to the release day:
   - freeze;
   - the merge of `67-gap-inference` to `main` by pull request, so `ci.yml` `build-and-test` runs;
   - release-branch creation, if you recommend one;
   - the version fix (`0.1.2` without `-SNAPSHOT`, stale `<scm><tag>`), and the recommended version number;
   - the release-candidate build;
   - the gates, run on the release-candidate jars;
   - docs and release notes;
   - publish;
   - the post-release bump.

   Name who does each step: the owner for anything needing sudo, git writes, gpg, Central credentials or Windows, and
   the agent for the rest. Say how Tier 03 work continues during the freeze without blocking the release or being
   shipped half done.
3. **Release 2: choose the target tier.** From the velocity and the sizes of the next tiers, name the tier after which
   release 2 sits. Name a fallback tier if that one slips, and the date by which the owner decides between them. The
   two releases need not follow the same tier. Explain what release 2 adds for users, and what it means for 02F and
   02G being open: does the release state Windows CUDA as unsupported and failing closed, or wait for it?
4. **Release practice, tailored to this repo.** For each item, either say how it is done here, with exact commands, or
   say why it is skipped:
   - versioning (pre-1.0 semver: minor for new features or behaviour changes, patch for fixes only);
   - branch and tag strategy, and the hotfix path (`0.x.y` from the release branch);
   - building a release candidate from a clean clone, with jar hashes recorded (the READY-file pattern);
   - gates on the candidate:
     - README rule 3's smoke matrix on real models;
     - `./juno test`;
     - `mvn verify` including the `juno-master` integration tests;
     - the GPU suites;
     - `compare-lora.sh` and `compare-vision.sh` against the last published baseline;
     - the standing four-model `compare-llama-cpp.sh` sweep, for the release-notes numbers;
     - the kit quick suite (`kit.yml`) on the four OS runners;
     - `check-plan-thresholds.sh`;
     - `CoverageTest`;
   - every gate states a numeric pass bar, as rule 7 requires of tiers;
   - published evidence: a `docs/perf-compare/<timestamp>-release-...` directory, with no tier number in the name;
   - release notes and upgrade notes;
   - the docs site update;
   - signing;
   - Maven Central: deploys are immutable and cannot be unpublished, so the candidate is verified before `deploy`;
   - a GitHub Release with the shaded jars, checksums and signatures;
   - optionally, an SBOM and a dependency vulnerability scan;
   - `SECURITY.md` supported-versions table;
   - repo hygiene before tagging:
     - `*.jfr` in the root, `juno-kit-results/` and `dist/` are all gitignored, so confirm the release is built from
       a clean clone, not this working tree;
     - tester folders are unscrubbed and stay out of commits;
   - post-release: bump to the next `-SNAPSHOT`, announce, and triage issues.
5. **Write the plans into the tree** (deliverables below). Run `check-plan-thresholds.sh` and confirm it passes.
6. **Ask the owner, don't assume.** Use AskUserQuestion, with your recommendation first, for:
   - the version numbers;
   - release 1's cut point;
   - release 2's target and fallback tier;
   - whether to publish to Maven Central, a GitHub Release, or both;
   - the merge-to-`main` strategy;
   - whether tier work pauses during the freeze.

   Record each answer in the plan as `*(Decided 2026-MM-DD by the owner: ...)*`, matching the tree's style.

## Deliverables

1. **One file per release in `docs/gap-closure-plan/`**, named by version, for example `RELEASE-0.2.0.md`, once the
   owner confirms the number. Put no tier number in the file name. Each file has these sections:
   - Status
   - Target date
   - Cut point (tier or commit) and why
   - What ships
   - What does not ship, and how each piece fails closed
   - Pre-release work items, ordered, with owner, estimate and done-condition
   - Gates, with numeric bars and the evidence location
   - Release procedure, with exact commands
   - Release-notes outline and upgrade notes
   - Known limitations, with Windows, macOS, AMD and AVX2 stated plainly
   - Rollback and hotfix
   - Post-release steps
   - Exit criteria (`- [ ]`)
   - Execution record, left empty
2. **`docs/gap-closure-plan/README.md` changes:**
   - a row for each release in the tier index table, directly after the tier it follows;
   - an addition to execution rule 1's enumeration, defining a release row as a checkpoint, not a tier: it adds no
     implementation scope, and its exit criteria gate the tag;
   - whether and how a release freeze may overlap the active tier, as an owner decision;
   - one paragraph in "Tier index and ordering rationale" for each release placement.

   Keep the README's existing voice: dated, precise, no emojis.
3. **A line in the preceding tier's file** naming the release that follows it, so an executor closing that tier sees
   the release is next.
4. **Your reply to the owner:**
   - the two recommended cut points, with one-line reasons;
   - the release-1 schedule as a dated list;
   - the decisions taken or still open;
   - the result of `check-plan-thresholds.sh`.

## Constraints (carried over; binding)

- **Planning docs only.** Write nothing to `src/`, `pom.xml`, workflows, `RELEASE_NOTES.md`, `CHANGELOG.md`,
  `docs/howto.md`, `README.md` (root) or `juno-documentation/`. The plans say what will change there and when.
- **Git is read-only.** No `add`, `commit`, `tag`, `push`, `stash`, `checkout` or `reset`. Write the owner's git, gpg
  and deploy commands into the plan for the owner to run.
- **Builds.** Do not run `mvn` in the main tree. Another session's item 6 work and its test runs may be live: check
  `pgrep -af "mvn|surefire|run-gate|compare-"`. If you must build to verify something, use a `git archive` copy in
  your scratchpad and never run `install` from it.
- **Shipped-text rules apply to what the plans prescribe for shipped files.** Release notes, CHANGELOG, howto,
  README, the docs site, git tag messages and GitHub Release text carry:
  - no competitor product names (README rule 4; express comparisons in Juno-native terms or against earlier Juno
    releases, as `RELEASE_NOTES.md` does);
  - no tier numbers;
  - no pointers into `docs/gap-closure-plan/` (rule 5).

  Inside the plan tree, naming tiers and competitors is fine.
- `docs/gap-closure-plan/` governs. `docs/infra-plan/` is history and evidence only.
- No Claude or Anthropic attribution anywhere. No emojis.
- Re-verify every line number or commit you cite. Both trees are snapshots.

## Exit checklist

- [ ] Change inventory since v0.1.1 built. Every half-wired feature at each candidate cut is listed with its surface.
- [ ] Release 1's cut point recommended, with options (a) to (c) weighed. Dated schedule written, each step with an
      owner.
- [ ] Release 2's target tier and fallback tier named, with the velocity evidence and a decision date.
- [ ] Both `RELEASE-<version>.md` files written with every section above. Gates carry numeric bars.
- [ ] README index rows, rule 1 wording and ordering paragraphs added. A line added to each preceding tier's file.
- [ ] `check-plan-thresholds.sh` passes.
- [ ] Owner decisions asked and recorded, or listed as open in the reply.
- [ ] No code, build, shipped-doc or git-state change made. `git status` shows only the new and edited plan files on
      top of the pre-existing item 6 work.
