# Handoff: owner decisions of 2026-10-30, then Tier 02F step 6

Read first: `CLAUDE.md`, `docs/gap-closure-plan/TIER-02F-hardware-validation-kit.md` (step 6 = scope items 11 to 13,
"Tests to write", exit criteria), `docs/infra-plan/PROMPT-kit-step5-rest.md` (how step 5 ended). Rules as before: git
read-only, no attribution, no tier numbers in shipped files, tests first, never build while a run is in progress.

## Owner decisions 2026-10-30 ("1. yes 2. yes after that start step 6")

1. Shorten the CPU context-policy probe: done in code. `ProbeCatalog` passes `turns=3`, `notesPerTurn=85` to both
   context-policy entries; `ProbeCatalogTest.contextPolicyEntriesGrowTheConversationInThreeLargeTurns` failed first.
   Owed: one kit run of both entries to measure the new cost (CPU estimate about 3 h), then set the CPU entry's cost
   and timeout (3x). Run it from the jar copy `<scratch>/kitrun/juno-kit.jar` (sha256 first 16 `b246151a5958cb6b`,
   has the change) so repo builds cannot touch it:
   `java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -Xmx1g
   -Djuno.kit.root=/home/medion/Repo/juno -jar <scratch>/kitrun/juno-kit.jar --suite custom
   --probes context-policy.phi-3.5-mini,gpu.context-policy.phi-3.5-mini --models-dir /home/medion/Repo/juno/models --yes`
2. Collector holds the batch open for `--batch-window-ms`: done. `RequestScheduler.collectBatch` polls with the time
   left until `batchWindowMs` after the first request, or until full (the contract `BatchConfig`'s javadoc already
   stated). Tests: `RequestSchedulerBatchTest.a_request_arriving_inside_the_window_joins_the_first_requests_batch`
   failed first; `a_full_batch_dispatches_without_waiting_out_the_window` is a regression test (passed first run).
   Coordinator: 400 unit tests pass; `TinyLlamaStaticBatchLiveTest` fails with `-pl coordinator` alone because
   `~/.m2` holds a node jar from 2026-10-28 (before the decode-batch fix); it passes with `-am`.
   LoRA gate not run: the collector only runs at `--parallel` > 1 and `compare-lora.sh` never sets it; the change has
   no MatVec/forward/KV/vision change (the roadmap's optional case).
   - GPU reading, fixed collector (`<scratch>/gpuwin/`, jar sha256 first 16 `dc12d31ed75320af`, 20261031T040116Z):
     TinyLlama P=1 108.66, P=8 81.23 t/s (0.75x; was 0.90x); Qwen2.5-3B 47.90 vs 32.36 (0.68x; was 0.87x). Every
     P=8 run now forms one batch of 8 (8 identical replies, the batched reply `484052` / `85d15b`; the old reading
     mixed single and batched replies). Cause: batched GPU decode (`forwardMultiDecode` -> `runLayersMultiDecode`)
     is host-driven per layer, not the device-resident decode region the single-request path uses (Tier 01 record:
     "batched decode ... not wired"). To publish as `docs/perf-compare/20261031T040116Z-gpu-parallel-window/`.
   - CPU reading, fixed collector: running (`<scratch>/cpuwin/`, same driver as `20261030T040305Z`).
   - Owner decision to ask: keep the fix (recommended; docs stop recommending 8) or revert it; and whether to plan a
     device-resident batched GPU decode.
3. Docs wording (`docs/howto.md` `--parallel` row "recommend `8` for API servers", `--batch-window-ms` row): to change
   after the CPU reading.

## Step 6 progress

- Coverage file: `juno-kit/src/main/resources/cab/ml/juno/kit/coverage.tsv`, `Coverage` (parse, `problems`),
  `CoverageTest` (7 tests; five detection tests failed first against a stub). Node gained
  `ForwardPassHandlerLoader.supportedArchitectures()` and `LlamaFamilyArchitectures.verified()`.
  Gap recorded in the file: native `/v1/inference/stream` is not probed (only consistency's blocking
  `/v1/inference`); the tier's cross-surface table said "same" as grammar/tools. Decide: add a native streaming
  check to `ConsistencyProbe`, or leave the gap recorded.
- Native streaming gap closed in code: `ConsistencyProbe` now checks `/v1/inference/stream` joins to the native
  blocking answer (self-test expectation added first; not yet run). `coverage.tsv` row 13 updated.
- `ingest` written, NOT yet compiled or run (the CPU reading holds the machine): `RunResults` (zip or dir; zip-slip
  refused; newer schema refused), `ReferenceComparison` (NLL <= 1.01x, argmax agreement drop <= 0.02, checks and
  greedy replies equal), `Ingest` (draft `<UTC>-remote-<host label>/` or `<UTC>-kit-reference/` with
  `--reference-run`; finds the newest `*-kit-reference` at the same commit), `IngestReport` (INDEX.md, README-row.md),
  `KitOptions` (`ingest <zip>`, `--reference`, `--reference-run`; both refused outside ingest; `--out` defaults to
  `docs/perf-compare` for ingest), `KitMain` dispatch, `KitBootstrap` reuses a built jar for ingest. Tests written
  first: `IngestTest` (8), `KitOptionsTest` (+2). Not seen failing first (written before the code, not compiled in
  between).
- No probe records logit checksums (scope item 11 names them); ingest compares NLL, argmax agreement, checks and
  replies. Report as a gap.
- Reference run: `dist/kit-reference/run-reference.sh` written (refuses modified tracked files; runs `--suite all`;
  then `ingest --reference-run`). Owed to the owner after a commit.
- Risk to check: `ConsistencyProbe` uses `--parallel 2 --batch-window-ms 300`; with the held window its batches now
  fill, and GPU batches of 2 differed from single replies at 64 tokens. Run both consistency probes on the new build
  (in the same kit run as the context-policy entries).
- CPU reading early rows: TinyLlama P=8 11.92 vs P=1 10.15 t/s (full batches now gain on the CPU).

## Status 2026-10-31 ~05:30 UTC

- Readings published: `docs/perf-compare/20261031T040116Z-gpu-parallel-window/` (GPU 0.748x / 0.676x) and
  `20261031T041246Z-cpu-parallel-window/` (CPU 1.174x / 1.084x, 96/96 replies equal); README rows added, old rows
  marked superseded; the 2026-10-30 readings' JDK corrected to 25.0.4.1 (upgraded 2026-10-29 20:44 UTC).
- Docs: `docs/howto.md` `--parallel` (8 on CPU, 1 on GPU) and `--batch-window-ms` rows, example comment;
  `ConsoleMain` help; `docs/agent-arch.txt` (RequestScheduler, SseTokenConsumer). Not yet compiled: ConsoleMain help,
  IngestReport RAM rounding.
- SSE fix (out of tier): `SseTokenConsumer.escape` writes control characters as unicode escapes; test failed first.
- Tests: juno-kit 116/116; coordinator 401, 0 failures, 1 skipped; live batch test passes. `mvn install` done, so
  `~/.m2` jars are current.
- ingest run end to end on run 2 (as reference) and run 3 zips in `<scratch>/ingest-out/`.
- Tier file: step 6 record written; probe-catalog exit criterion ticked; status line updated. Placeholder left:
  `KITRUN_RESULT` (run 5).
- Run 5 (background, `<scratch>/run5.log`, jar `c661c3166b219db7`): consistency.tinyllama, gpu.consistency.tinyllama,
  context-policy.phi-3.5-mini, gpu.context-policy.phi-3.5-mini.

## Status 2026-10-31 ~09:10 UTC: step 6 done except the owner's reference run

- Run 5 (20261031T052330Z): all 4 PASS. CPU context-policy 3:07:34 (run 4: 5:38:40); GPU 4:19, 5,036 MiB;
  consistency.tinyllama 1:42 and gpu.consistency.tinyllama 0:13, batched replies equal single ones, native stream
  joins to the blocking answer. Zip: `juno-kit-results/juno-kit-results-linux-amd64-20261031T052330Z.zip`.
- Catalog costs set: CPU context-policy 11,250 s (timeout 33,800 s), GPU 260 s. `--plan` here: quick 24-34 min,
  gpu 57-81, cpu 629-888, all 686-968.
- Tier file: `KITRUN_RESULT` filled. `dist/kit-reference/run-reference.sh` time text: about 11.5 to 16 hours.
- Tests: juno-kit 116/116; juno-player 127, 0 failures, 2 skipped; `check-plan-thresholds.sh` ok.

Owner decision 2026-10-31: "2. do it": `docs/gap-closure-plan/TIER-02H-batched-gpu-decode.md` created (after 02G,
before 03); README rule 1, index paragraph and row; Tier 07 linked; Tier 02F record updated; plan check ok (26 files;
a copy with 02H's Threshold block and plan-check mention removed fails both checks). Decision 1 (keep or revert the
collector) still open: the owner asked for details first. Reference run timing: after steps 7 and 8 are committed,
at the commit testers build (ingest matches the reference by commit).

Owner decision 2026-10-31 on decision 1: "Option: B", keep the collector change as it is.

## Step 7 done 2026-10-31

- `check-plan-thresholds.sh` check 10 (open tier needs `## Kit probes` with a list item): 17 failures before the
  sections, ok after; an emptied-section copy fails once. README rule 10 and rule 7's check 10 entry.
- `## Kit probes` (Keep passing / Add) in all 17 open tier files; exit criterion added to the 14 without one.
- `INVENTORY.md` "Borrowed hardware" table (empty). Tier 02F: criterion ticked, Status line fixed (it said no
  criterion ticked), step 7 record written.

Next: step 8, the CI workflow (decision 8; the owner commits it; first green run recorded); then the owner's
reference run at that commit (`dist/kit-reference/run-reference.sh`), before testers' runs (step 9).

## Step 8 written 2026-10-31

- `.github/workflows/kit.yml` (untracked; `ci.yml` untouched): `quick-suite` on dispatch and weekly (Mon 04:23 UTC) on
  ubuntu-24.04, ubuntu-24.04-arm, windows-2025, macos-15 through `./juno-kit.sh` / `juno-kit.bat --suite quick --yes`;
  zip uploaded as `juno-kit-results-<runner>`; last step fails the job on FAIL/ERROR/TIMEOUT/NOT RUN, a missing
  SUMMARY.txt, or an unrecognized summary format. `self-tests` on PRs touching the kit:
  `mvn -B -ntp package -pl juno-kit -am -Dtest='cab.ml.juno.kit.**' -Dsurefire.failIfNoSpecifiedTests=false`.
- Checked here: YAML parses; gate step exit 0 on run 5's summary, 1 on FAIL/ERROR/NOT RUN/missing/format; the
  self-test command BUILD SUCCESS 5:01, juno-kit 116/0/0/0 (launcher self-test ran). macOS runner (7 GB) runs only
  env, calibration, throughput 128/512, launcher.
- Tier file: Status line and step 8 record; README "Test infrastructure" notes kit.yml.

Next: owner commits, opens a PR to main (self-tests job runs), dispatches `gh workflow run kit.yml --ref main`;
record both run URLs in the tier file. Reference run at main's HEAD (ingest matches by commit; a merge commit
changes the hash). Then step 9: ingest the CI zips, fix defects, testers' runs.
