# Review Fix List — Post `release-0.1.2` Glue-Layer Cleanup

**Execution status (2026-09-15):** items 1-10, 12 implemented and retested this session; only item
11 (porting 14+ CHANGELOG sessions into `juno-documentation`'s prose style) remains open — that one
needs substantial narrative content authoring, not a mechanical fix, and was deliberately not
rushed.

Retested: `mvn test -pl node,coordinator,juno-player` green (no regressions from items 1-6, 12).
Item 7 (`compare-lora.sh --reps N`, median aggregation): verified against a real 2-rep GPU LoRA
train+playback run — both reps succeeded, median math checked by hand against the raw per-rep
JSONs (44000ms/45000ms → 44500ms; 11.804/11.111 tps → 11.458 tps). Environmental-noise
investigation published to `docs/perf-compare/README.md` (GPU persistence mode disabled, CPU
`schedutil` governor, background load ~4-5 — all verified live on this host, not guessed). Item 8
(`compare-llama-cpp.sh`'s standing Mistral-7B tuned lane): verified with a full published 4-model
GPU bake-off (`n_prompt=128`, `n_gen=64`, `reps=3`) at `docs/perf-compare/20260915T223032Z/` —
default lane 0.479 t/s (ratio 0.0136×) vs. tuned lane 15.65 t/s (ratio 0.444×, clears the P0 0.15×
gate), ~32.7× apart, both numbers now in `docs/perf-compare/README.md`.

Source: [`Sonet5-max-inference-review.md`](Sonet5-max-inference-review.md) (review of the 29 commits
between `51a3b90`/`release-0.1.2` and `3f4a340`). That review's own conclusion: the algorithms are
sound (concurrency design, kernel work, feature-flag discipline); the problems are in the glue layer
— measurement rigor, doc propagation, launcher parity, config-resolution consistency. This doc turns
that into an ordered, actionable punch list.

**Not an Infra Tier.** Nothing here adds a new CLI/env flag, MatVec/residency path, or scheduler mode
(Execution rule §6), so it does not need a feature × surface interaction matrix and does not consume
the "one Infra tier in flight" slot ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) §1). It can run
alongside whatever Infra tier is active. Items 4 and 6 touch the decode hot path / benchmark
instrumentation, so per root `CLAUDE.md`'s "Performance gates on hot-path changes" rule, run
`compare-lora.sh` (and publish under `docs/perf-compare/`) after those two specifically.

## Ordering rationale

Cheap, unambiguous, zero-risk fixes first (1-2). Then correctness/perf fixes with a clear mechanical
change (3-5). Then the measurement-integrity items, because every ratio the team publishes after this
point should be trustworthy (6-9) — including item 9, since a regression gate that can't parse a
tier's own flags is a measurement-integrity problem, not a feature gap. Then the one item that isn't a
code fix at all but a decision the team is implicitly deferring by not making it (10). Cleanup last
(11-12).

## 1. Untrack the committed JVM crash dump **[Done]**

`hs_err_pid8816.log` is tracked in git (`8b78382`, "Finish LoRA play MMQ"), timestamped ~15 minutes
before that commit. It is a genuine JVM/JFR bug — `ShouldNotReachHere()` in
`JfrSamplerThread::sample_java_thread` → `SuspendedThreadTask::internal_do_task()`
(`signals_posix.cpp:1793`), OpenJDK 25.0.3+9 — triggered while `--gpu --mmq on --lora-play ...` were
all active, i.e. the JFR profile-sampler thread suspending a thread mid Panama-FFI native downcall
into CUDA. It has no business in the repo regardless of cause.

- `git rm hs_err_pid8816.log`.
- Add `hs_err_pid*.log` and `core.*` to `.gitignore` (currently only `*.jfr` is ignored, which is
  exactly why this slipped through — loose `.jfr` files from local runs already sit gitignored in the
  working tree today, confirming the pattern works for that extension).

**Exit:** file untracked, `.gitignore` patterns added, `git status` clean of stray crash artifacts.

## 2. `run.bat` flag parity with `run.sh` **[Done]**

`run.bat` is missing four flags `run.sh` has: `--gpu-layers`, `--parallel`, `--batch-window-ms`,
`--prefill-batch` (confirmed via diff of long-flag tokens in both scripts). Every later session's
flags (`--mmq`, `--schedule`, `--kv-page-size`, `--cache-type-k/v`, `--embeddings`, `--hf`,
`--grammar-file`, tool-related) made it into both scripts — only this pre-69, unlabeled P0 block
(`GpuLayerOffload`, `--parallel`, prefill microbatching) was missed. Windows users currently cannot
reach GPU-layer offload or batching from the launcher at all.

**Exit:** `run.bat` forwards all four flags identically to `run.sh`; smoke-run `run.bat` locally (or
have a Windows-available contributor confirm) with each flag set once.

## 3. Standardize `JUNO_*` env/property resolution **[Done]**

Six option classes resolve their env var three different ways:

| Class | Resolution today |
|---|---|
| `node/.../MmqOptions.java` | `System.getProperty` only |
| `node/.../GpuLayerOffload.java` | `System.getProperty` only |
| `coordinator/.../ServeBatchOptions.java` | `System.getenv` only |
| `kvcache/.../CacheTypeOptions.java` | `firstNonBlank(getProperty, getenv)` |
| `kvcache/.../KvPageSizeOptions.java` | `firstNonBlank(getProperty, getenv)` |
| `kvcache/.../ServeScheduleOptions.java` | `firstNonBlank(getProperty, getenv)` |

`export JUNO_MMQ=on` or `export JUNO_GPU_LAYERS=auto` is silently ignored today — the worst kind of
misconfiguration for a performance flag, since the process starts successfully with the wrong
behavior instead of failing loudly. Three of six classes already use the correct pattern.

- Change `MmqOptions`/`GpuLayerOffload` to `firstNonBlank(System.getProperty(ENV_PROPERTY),
  System.getenv(ENV_PROPERTY), default)`, matching `CacheTypeOptions`/`KvPageSizeOptions`/
  `ServeScheduleOptions`.
- Change `ServeBatchOptions` to also check `System.getProperty` alongside `System.getenv`.

**Exit:** all six classes use the same `firstNonBlank(getProperty, getenv, default)` shape; existing
unit tests (`MmqOptionsTest`, `GpuLayerOffloadTest`, `ServeSchedulePolicyTest`, etc.) extended to cover
the env-var path for the four classes that didn't have it.

## 4. Fix O(n²) history reconstruction in the decode hot path **[Done — measured, no win at this scale]**

`ContinuousBatchEngine.runDecode` (`coordinator/.../ContinuousBatchEngine.java:288`) rebuilds
`historyArr` via `s.generated.stream().mapToInt(Integer::intValue).toArray()` every decode step, for
every active slot, from a `List<Integer>` — a full boxed-list traversal plus array allocation per
token per slot, compounding over a generation. The static `GenerationLoop` has the same pattern. This
sits directly on top of the gather-tax work already fought hard to shrink (10.7% → 4.6%, Tier 14) and
is worth profiling before treating the continuous-batching 0.86x-vs-static number (Tier 15 bake-off)
as a structural, unfixable tradeoff — part of it may be avoidable JVM churn rather than real algorithm
cost.

- Replace the boxed `List<Integer> generated` (or its per-step `.toArray()` call site) with a
  primitive growable `int[]` buffer maintained incrementally (append on each accepted token, no
  rebuild).
- Apply the same fix to `GenerationLoop`'s equivalent history-array construction.

**Exit:** unit test asserting sampler-visible history is correct after N decode steps; JFR
before/after comparison of `ForwardPass`/decode step timing at a representative slot count; per root
`CLAUDE.md`'s hot-path rule, publish `compare-lora.sh` (and ideally `compare-schedule.sh`, since this
is squarely `ContinuousBatchEngine`) results under `docs/perf-compare/`.

**Measured (2026-09-15), reported honestly:** ran a real controlled before/after using
`compare-schedule.sh --cpu --mode tps --sessions 8 --max-tokens 64` — `git stash` isolated exactly
the 3 files this fix touches, rebuilt, benchmarked, restored, rebuilt, benchmarked again, same
command both times (see `docs/perf-compare/README.md` → "historyArr fix measurement" for the raw
numbers). Result: **no measurable win at this scale** — static 3.731→3.661 t/s, continuous
3.686→3.674 t/s, both deltas smaller than the run-to-run noise this same session's item 7
investigation found on this host. The fix is still correct and worth keeping (removes real
per-step boxing/unboxing and stream-pipeline overhead, cleaner code, no behavior change, unit
tests green) — it just doesn't move the needle at 64 generated tokens × 8 sessions on a 1.1B CPU
model. The original hypothesis ("worth profiling... may be avoidable JVM churn") was reasonable to
raise but is **not confirmed** by this measurement; a longer-generation, higher-slot-count
workload would be needed to see the O(n) reconstruction cost actually dominate, and that wasn't
tested here.

## 5. CAS the non-atomic `closed` guards **[Done]**

`DeviceQ4KMatrix` (`node/.../DeviceQ4KMatrix.java`) uses `volatile boolean closed` with a
checked-then-set pattern (`if (!closed) { closed = true; ... }`, lines ~152-153); `ResidentQ4KWeight`
(`node/.../ResidentQ4KWeight.java`) uses a plain (non-volatile) `boolean closed` with the same
checked-then-set shape (lines ~64-66). Neither is atomic. Low risk today because LoRA fails closed
under continuous scheduling and single-threaded close paths are the norm, but a latent double-free if
that constraint is ever relaxed.

- Replace both with `AtomicBoolean` + `compareAndSet(false, true)` guarding the actual close/free
  call.

**Exit:** existing close/lifecycle tests still green; no behavior change under single-threaded use
(this is a defensive hardening fix, not a functional one).

## 6. Reduce JFR benchmark overhead and crash surface **[Done]**

Both the local REPL path (`ConsoleMain.startProgrammaticJfr`) and the forked cluster-node path
(`ClusterHarness`, `-XX:StartFlightRecording=...,settings=profile,dumponexit=true`) run JFR under the
heavyweight built-in `"profile"` configuration — aggressive native-thread stack sampling, not the
lighter `"default"`. This is the exact subsystem that crashed in item 1's `hs_err_pid8816.log`
(`JfrSamplerThread` suspending a thread mid GPU native downcall), and heavier sampling is itself a
plausible contributor to timing perturbation during bake-offs. Separately, `compare-lora.sh` already
documents that its JFR-derived `tps_jfr` should not gate results for short playback runs — an implicit
acknowledgment that JFR-derived throughput is less trustworthy than wall-clock, which is inconsistent
with treating other JFR-derived numbers elsewhere in the review (gather-tax %, kernel p95) as reliable
without the same caveat.

- Switch benchmark JFR recordings (both `ConsoleMain` and `ClusterHarness`) from `settings=profile` to
  `settings=default`, or a minimal custom `.jfc` enabling only the `juno.*` events actually consumed
  (`TokenProduced`, `GrammarConstrained`, `MatVec`, `ForwardPass`) plus whatever stdlib events those
  depend on — keep `"profile"` available as an opt-in for one-off deep-dive debugging sessions only,
  not the bake-off default.
- Decide explicitly, and document in `docs/perf-compare/README.md`, whether any JFR-derived number is
  allowed to gate a P0/P1 threshold. If not, label every JFR-derived figure "diagnostic only"
  consistently, not just in `compare-lora.sh`.

**Exit:** a bake-off run with `settings=default` published alongside the equivalent `"profile"` run to
confirm no metrics regression from the lighter config; `docs/perf-compare/README.md` states the JFR
gating policy explicitly.

## 7. Stabilize the measurement harness **[Done — compare-lora.sh; environmental note published]**

The llama.cpp reference number itself swung 9x for the identical tinyllama Q4_K_M CPU config — 46.0
tg (session 79, 2026-09-12 19:34) vs. 5.04 tg (session 80, 2026-09-13 03:27), ~13 hours apart, with
nothing Juno-side changed. Notably this happened *despite* `compare-llama-cpp.sh` already averaging
3 reps internally via `llama-bench -r 3` (`REPS=3`, `scripts/performance-tests/compare-llama-cpp.sh`)
— so the swing isn't a missing-reps problem on the reference side, more likely environmental (shared,
unisolated dev box: thermal state, other processes, GPU driver availability, which session 77 itself
notes was unavailable that run). Separately, the Juno-invoked side of most other compare scripts
(`compare-lora.sh`, `compare-schedule.sh`, `compare-parallel.sh`, `compare-prefill-batch.sh`,
`compare-mixed-prefill.sh`) launches the Juno process exactly once per benchmark — no reps at all.

- Add a `REPS`-style repeat option to the Juno-invoked side of the compare scripts listed above,
  taking a median (not mean, to resist the kind of outlier seen in sessions 79/80) across reps for the
  headline number.
- For the llama.cpp reference number specifically, since 3 reps already didn't prevent a 9x swing,
  investigate the environmental cause (thermal throttling, background load, GPU driver flakiness)
  rather than only adding more reps — reps alone did not fix this once.

**Exit:** at least one compare script (`compare-lora.sh` or `compare-schedule.sh`, since both already
appear in the review's own worst-noise examples) updated with reps + median reporting; a short note in
`docs/perf-compare/README.md` on the investigation into the reference-side swing.

## 8. Add a standing tuned lane for Mistral-7B **[Done — published bake-off `20260915T223032Z`]**

`--mmq`/`--gpu-layers` both default off. Mistral-7B on the review's 8 GB card sits at 0.42-0.49 tg
across the default-path checks in sessions 72, 73, 75, 76, 77 — worse than Juno's own CPU numbers for
smaller models. The one time the team ran `--mmq on --gpu-layers auto` deliberately (session 78's
dedicated bake-off), the same model hit 15.3 tg, roughly 30x better. The regression gate that's
supposed to protect this model is, most sessions, measuring a configuration nobody would run in
production.

- Add a second, explicitly "tuned" lane (`--mmq on --gpu-layers auto`) for Mistral-7B to the regular
  `compare-llama-cpp.sh --gpu` regression sweep, run alongside the existing vanilla-default lane, not
  instead of it.
- Separately consider (design decision, not blocking this item): should `--mmq`/`--gpu-layers` default
  to `auto`-equivalent behavior when the loader detects a model won't fit resident in FP16, rather than
  requiring the user to know the flags exist? That's a larger UX/default-behavior change and belongs
  in its own tier doc if pursued — flagging here so it isn't lost.

**Exit:** next published GPU bake-off includes both lanes for Mistral-7B, both numbers visible in
`docs/perf-compare/README.md`.

## 9. Stop the standing regression script from falling behind shipped tier flags **[Done — compare-llama-cpp.sh; ROADMAP rule added]**

`compare-llama-cpp.sh` — the script Execution rule §2 step 1 names as the regression gate every tier
must run before being marked feature complete — has not been modified since `f168895` (Tier 13B,
which added `--mmq`). Its pass-through option set is `--gpu-layers`, `--mmq`, `--prefill-batch` only.
It has **no support at all** for `--schedule` (Tier 15 continuous scheduler, Tier 16 mixed chunked
prefill), `--cache-type-k/v` (Tier 6 quantized KV), or `--kv-page-size` (Tier 14 block KV allocator) —
not defaulted-off, but unparseable by the script. Every tier from 14 onward that ran the mandated
"regression gate" per §2 step 1 was therefore structurally blind to its own predecessors' shipped
flags, not merely exercising them at default values (that narrower problem is item 8, above, and only
covers `--mmq`/`--gpu-layers`).

This is broader than one script. The AWS-style harness (`scripts/performance-tests/scenarios.yaml` /
`matrix.tsv`) has not been touched since `a582457` (2026-06-01) — it predates every tier in this
review's window and is fully abandoned. Every tier-specific script (`compare-schedule.sh`,
`compare-parallel.sh`, `compare-prefill-batch.sh`, `compare-mixed-prefill.sh`, `compare-lora.sh`,
`compare-vision.sh`) is written once at that tier's own completion and then frozen — e.g.
`compare-vision.sh` has not moved since session 47 (`d3449cc`, 2026-09-04), despite four tiers (6, 14,
15, 16) shipping afterward that touch MatVec/KV/residency paths vision's underlying text handlers
share. Nothing currently re-runs an older tier's dedicated script against a newer tier's change to
confirm continued compatibility, and nothing folds a new tier's flag back into the one script every
other tier treats as its baseline regression gate.

- Extend `compare-llama-cpp.sh`'s pass-through options to cover `--schedule`, `--cache-type-k/v`, and
  `--kv-page-size` (mirroring how `--gpu-layers`/`--mmq`/`--prefill-batch` already work), so the
  standing regression gate can at least be pointed at any shipped flag combination, current or future.
- Add a step to `PLAN-Infra-ROADMAP.md`'s Execution rule §2 (perf compare at tier completion): when a
  tier ships a new CLI/env flag, that flag must be added to `compare-llama-cpp.sh`'s pass-through
  option set in the same change, not left to live only in a bespoke one-off script. The bespoke script
  remains appropriate for the tier's own specialized bake-off (e.g. `compare-schedule.sh`'s
  aggregate-tps-under-load measurement), but the general regression gate should not need a different
  script per tier to exercise that tier's flag at all.
- Decide whether `scenarios.yaml`/`matrix.tsv` are still load-bearing for anything (the AWS
  `juno-deploy.sh` path references a `perf-lib.sh` pattern per the review, but this repo's actual
  `scripts/performance-tests/perf-lib.sh` has no `REPS` variable and is not the AWS one) — either
  update them to reflect the current flag surface or remove them so they stop reading as current
  guidance.

**Exit:** `compare-llama-cpp.sh --help` lists `--schedule`/`--cache-type-k/v`/`--kv-page-size`
alongside the existing pass-through flags; `PLAN-Infra-ROADMAP.md` Execution rule §2 states the
new-flag-goes-into-compare-llama-cpp.sh requirement explicitly; `scenarios.yaml`/`matrix.tsv` either
updated or removed with a one-line note on the decision.

## 10. Decide: resume or formally defer the P0 decode-kernel push **[Done — formally deferred]**

Per `PLAN-Infra-ROADMAP.md`, P0 (kernel path) gates all "peer" claims and requires Phi-3.5 GPU tg
≥0.5x llama.cpp. Session 78 (Tier 13B, `--mmq on --gpu-layers auto`) got Phi-3.5 to 0.33x via the
Q8_1/`dp4a` kernel — the project's own historical note: pre-window Juno GPU decode was 0.17-0.22x with
`juno.MatVec` at 93-96% of decode time, so 0.33x is real progress, but the 0.5x gate remains unmet.
Sessions 79-82 (and Tier 17, GPU batched-prefill GEMM, landed since this review was written) pivoted to
API-surface and prefill-side work — permitted by the roadmap since those aren't P0-gated, but it means
the core single-stream decode gap has sat unattended for several sessions in a row.

This is not a code change — it's a decision the team should make explicitly rather than let drift by
default while feature/adjacent-perf sessions continue to look attractive. Options as the review framed
them: (a) resume the P0 kernel push (per `PLAN-Infra-PERF-ANALYSIS.md`'s own suggestion #3 — GEMV
occupancy/tiling, not quantization scheme, is the more likely remaining lever now that the 1.3x
tile-kernel target was already cleared at 1.51x); or (b) formally record in `PLAN-Infra-ROADMAP.md`
that P0 is deferred for N sessions with a stated reason, so "P0 still open" doesn't silently keep
meaning "nobody decided."

**Exit:** either a new/updated tier doc picking up the P0 decode-kernel gap (successor to
`PROMPT-P0-Gate.md`), or an explicit deferral note added to `PLAN-Infra-ROADMAP.md`'s P0 phase row.
Not both silence and both options.

## 11. Sync `juno-documentation` alongside `CHANGELOG.md` **[Open]**

`juno-documentation/part11/02-changelog.md` was last touched at `3835bd3` (2026-09-02), then never
again, despite root `CLAUDE.md`'s "update docs alongside code" convention. `CHANGELOG.md` and
`docs/agent-arch.txt` are both current through `3f4a340` — the team clearly can keep docs in sync when
the habit is followed, it just isn't propagating to this tree (only 2 of ~62 files there have moved
since `3835bd3`).

- Add updating `juno-documentation/part11/02-changelog.md` (or its successor entries) to the same
  per-session doc-update step that already covers `CHANGELOG.md`/`docs/agent-arch.txt`.

**Exit:** `juno-documentation`'s changelog entry current through the same commit as `CHANGELOG.md` at
time of this fix; going forward, both move together.

## 12. Small cleanup pass **[Done]**

Bundle into whichever session is already touching these files — not worth a dedicated session:

- **Unreachable defensive branch**: `ContinuousBatchEngine.runDecode`'s `if (s.generated.size() >=
  s.params.maxTokens()) continue;` guard (line ~270) appears unreachable given the current control
  flow — a slot hitting `maxTokens` is already caught and retired earlier in the same method (line
  ~321, via `justFinished`/`retireFinished`), so it should never re-enter `runDecode`'s input batch:
  `retireFinished` removes a finished slot from `running` in the same `engineStep()` that finishes
  it, and `plan.decode()`'s input batch is derived from `running`, so a later call cannot see that
  slot again. **Resolved**: kept the guard (a full scheduling-harness test to prove the negative
  wasn't worth building for an O(1) check) and added a comment at the call site explaining the
  invariant and why it's kept as defense-in-depth rather than an assertion.
- ~~**`CHANGELOG.md` em-dash violations**~~ — **retracted**. The source review cited "CLAUDE.md rule
  10" against em-dashes; root `CLAUDE.md`'s actual "Project conventions" section has no such rule
  (checked directly), and `CLAUDE.md` itself uses em-dashes as ordinary prose punctuation throughout
  (e.g. its own line 7). `CHANGELOG.md`'s 252 em-dashes are the same normal usage — "Session 82 —
  Embeddings API", etc. — not a literal separator convention being violated. This was an unverified
  claim carried over from the source review into the original version of this plan; no fix needed.

**Exit:** unreachable-branch comment landed; no CHANGELOG.md change made (false positive retracted).
