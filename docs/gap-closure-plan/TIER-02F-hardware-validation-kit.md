# Tier 02F: Hardware validation kit (one command on Windows, Linux or macOS; results returned as one zip)

Status: in progress (added 2026-10-13 by owner decision; runs after Tier 02E and before Tier 02G). Steps 1 to 4 done
2026-10-28: catalog pinned; module, bootstrap, launchers and runner built; the `quick` CPU probes and the CUDA probes run
end to end on this host. Step 5 done 2026-10-30: part 1 2026-10-28 (probe groups from unit tests to throughput),
part 2 2026-10-30 (schedules, LoRA, vision, launcher, and the real-model runs). Step 6 done 2026-10-31 (coverage
file and test, `ingest`) except the reference run, which is the owner's, at a commit. Step 7 done 2026-10-31 (README
rule 10, check 10, the **Kit probes** sections, the "Borrowed hardware" table). Step 8 written 2026-10-31
(`.github/workflows/kit.yml`, checked here); not committed (the owner's commit `a1fe7166` left `.github/` out), and its
runs are deferred by owner decision 2026-11-03 ("for now skip github ci"). Step 10 done 2026-11-03, ahead of step 9,
whose testers need its guide; the same day's `quick` run corrected three catalog costs and found the CPU live
probe's wide prefill windows about 5.5x slower inside the kit than outside it (cause open; the engine is not slower).
The CPU batch dependence found in part 1 is
explained (not a KV leak; see "CPU batch dependence investigated") and fixed in the engine 2026-10-29 by owner
go-ahead, an out-of-tier change (see "CPU decode batch made bit for bit the single-request decode"); its pinned
throughput gate is met (owner run, 2026-10-29). Three exit criteria are ticked: the probe catalog (step 6), rule 10
with check 10 (step 7) and the docs (step 10).
Gap analysis refs: none — see "Why this tier, why now"

**Added 2026-10-13 (owner decisions, recorded in the execution record below).** This host has one CPU
without AVX2, one Pascal GPU, no AMD GPU and no Windows CUDA install. Work that depends on other hardware
is closed with markers instead of results, and the harness that would measure it is bash that a person
with that hardware, on Windows in particular, cannot run. This tier builds the tool such a person runs.

## Objective

A tool in the build, the `juno-kit` module, that a person with hardware this host lacks runs with one
command on Windows, Linux or macOS, given only JDK 25 and Maven. It builds the Juno copy it ships in,
detects what the machine can run, lets the tester choose what to run, estimates the time, says what the
tester may and may not do meanwhile, shows progress, runs Juno's correctness checks and same-run
comparative readings across every feature surface, and packs the results, scrubbed of personal data,
into one zip the tester sends back. On the owner's side, `ingest` turns a returned zip into a draft
`docs/perf-compare/` entry.

## Why this tier, why now

- **Hardware-gated results accumulate as markers.** 18 tier files carry `NEEDS-AMD-HARDWARE` rows (every
  tier from 01 to 13 that touches a GPU path), and [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md)'s
  ROCm parity has no device to run on. [Tier 02E](TIER-02E-cpu-decode-kernels.md)'s 256-bit width row is
  `NEEDS-AVX2-HARDWARE`, and its owner decision 4 keeps `auto` at 128 bits until a 256-bit host is
  measured. [Tier 02G](TIER-02G-cuda-on-windows.md) (CUDA on Windows) needs a Windows NVIDIA machine to be
  verified on.
- **The harness cannot run where that hardware is.** `scripts/performance-tests/` holds about 11,400
  lines of bash (`compare-llama-cpp.sh` alone 2,839) driving Juno through `curl` and `jq`. None of it runs
  on Windows, and a tester should not have to assemble it on Linux either.
- **The Java pieces exist.** `ModelLiveChecks` (`juno-master`, ten live checks including both cluster
  types, long-prompt prefill and context shift, behind `./juno test`), `ClusterHarness`, the
  `JunoPlayer`/`LoraTrainer`/`JunoHttpClient` embedding facade, `HfGgufFetcher` (resumable Hugging Face
  downloads with caching), `JfrMetricsExtractor` and `JfrSpanBreakdown`, `AnsiSupport`.
- **Sequenced after Tier 02E and before Tier 02G** (owner decision 1): Tier 02G cannot be verified without
  it, and every later tier adds the kit probes for its own feature as it lands (scope item 13), which
  costs less than retrofitting them at Tier 14.

## Scope

### In scope

1. **Module and launchers.** A new Maven module, `juno-kit` (owner decision 2), in the reactor. It depends
   on the modules its probes call (`juno-player`, `coordinator`, `node`, `vision`, `lora`, `registry`,
   `tokenizer`, `sampler`, `metrics`, and `ModelLiveChecks`; if depending on the `juno-master` deploy
   module for that one class proves awkward, the class moves to a shared module, recorded here). It
   builds a shaded runnable jar that is not deployed and is not in `juno-bom`. Launchers at the repository
   root, `juno-kit.sh` (sh) and `juno-kit.bat`, start `KitBootstrap`, a
   source file in the module that uses only JDK classes (a unit test holds it to that), so the JDK can run
   it directly before anything is built. *Corrected 2026-10-28 (step 3): this item named the sh launcher
   `juno-kit`, matching `juno`. That path is the module's directory, so a file cannot be created there; the
   module directory keeps its artifactId's name, as every module's does, and the sh launcher is
   `juno-kit.sh`. Windows is unaffected (`juno-kit.bat`, also run as `juno-kit`).*
2. **Bootstrap, before any build.** It checks the JDK (feature version 25 or later), finds Maven (`mvn` or
   `mvn.cmd` on `PATH`, or `MAVEN_HOME`), and checks free disk and the reachability of Maven Central and
   huggingface.co. It then builds (`mvn -B package -DskipTests -pl juno-kit -am`), showing reactor
   progress, and starts the kit jar in a fresh JVM with Juno's runtime flags. A failed build still
   produces the results zip, holding the build log and the environment: a build failure on a tester's OS
   is a result.
3. **Detection.**
   - OS, version and architecture; CPU model, cores and threads; RAM; free disk.
   - The JVM's preferred vector width per element type, and the OS's CPU feature flags (`/proc/cpuinfo`,
     `sysctl` on macOS, the processor identity on Windows).
   - Power source and battery state.
   - GPUs from `nvidia-smi` or `rocm-smi` where present.
   - Juno's own backend probe in a child JVM (`CudaAvailability`, `RocmAvailability`, device name, free
     and total memory). This one decides the GPU suites.

   Each suite is listed as runnable or not, with the reason ("ROCm: no AMD GPU found").
4. **Choice.** An interactive menu when a console is attached. Suites are `quick` (about 30 to 60
   minutes), `cpu`, `gpu`, `all` and `custom`. Flags:
   - `--suite` and `--probes` select what runs.
   - `--budget <duration>` runs the highest-priority probes that fit. Priority is set per probe in the
     catalog, with probes that unblock plan rows first.
   - `--yes` skips prompts.
   - `--plan` shows what would run, the time, the downloads and the disk needed, with no side effects.
   - `--list` and `--models-dir`.

   Subcommands: `resume`, `pack`, and `ingest` (item 11).
5. **Estimate and progress.**
   - A calibration probe of one to two minutes scales the catalog's reference cost per probe, which the
     owner's reference run (item 12) supplies.
   - Download time comes from the measured transfer rate; build time depends on whether Maven's local
     repository is filled.
   - The estimate is shown as a range before the confirmation prompt and refined as probes finish.
   - Progress shows an overall bar, the current probe, and elapsed and remaining time. Output falls back
     to plain lines when it is not a terminal, and a full log file is always kept.
6. **Warnings and machine state.**
   - At each stage the kit says whether the tester may use the computer (downloads, build) or should
     leave it idle (measurements).
   - It warns on battery power, on a metered connection (with the total download size), and, for GPU
     suites, to close 3D and GPU-heavy programs.
   - It keeps the machine awake without admin rights: `SetThreadExecutionState` through the FFM API on
     Windows, `caffeinate -i -w <pid>` on macOS, and `systemd-inhibit` on Linux where present (else a
     warning).
   - During readings it samples CPU use by other processes and GPU use by others. A reading taken under
     load, or an outlier by the README's rule ("'Outlier' is a number"), is re-run up to twice, and flagged
     in the results if it is still noisy.
7. **Models.** A catalog of ungated models, each with its Hugging Face source, size and SHA-256.
   - Download goes through `HfGgufFetcher` with resume, into a cache directory the tester can change. Each
     file is verified by hash and reused when present.
   - The hashes are those of this host's files: computed from `models/` without reading the files into
     context, and matched against Hugging Face's published LFS SHA-256, so nothing has to be downloaded to
     check them.
   - `quick` needs TinyLlama 1.1B Q4_K_M only. `all` adds the other three sweep models, Qwen3-1.7B (the
     `qwen3` handler) and moondream2 (vision).
   - The LoRA probes train their own small adapter in the run, so no adapter is downloaded.
   - A probe whose model the catalog cannot fetch is listed as not runnable, with the reason.
8. **Probe runner.**
   - A probe has an id and a kind: a **check** (pass or fail) or a **reading** (numbers with min and
     max).
   - It also has requirements (OS, CPU feature, backend, memory, models), a reference cost, a priority and
     a child-JVM entry point.
   - Every probe runs in a fresh JVM with Juno's runtime flags and a timeout. A failure, crash or timeout
     is recorded with its log, and the run continues.
   - A checkpoint is written after every probe, so an interrupted run resumes, and `pack` builds the zip
     from whatever exists.
9. **Probe catalog: the README's smoke matrix (execution rule 3) and every existing smoke and compare
   script.** Each probe ports the Juno-side checks or readings of what it names and calls Juno through
   its Java APIs, so a refactor that breaks a probe breaks the build:

   | Probe group | Ports or reuses | Kind | Runs where |
   |---|---|---|---|
   | Environment | new | reading | every host |
   | Unit tests | `mvn test` on the unit-test modules; the `gpu` and `rocm` test groups | check | every host; GPU groups where the backend loads |
   | Live checks | `ModelLiveChecks` 1 to 10 per model family, CPU and GPU | check | every host |
   | Consistency, grammar, tools | `smoke-consistency.sh`, `smoke-grammar.sh`, `smoke-tools.sh` (OpenAI and native REST through `JunoHttpClient`) | check | every host |
   | Context policy, long prompt | `smoke-context-policy.sh`, `smoke-long-prompt-prefill.sh` | check | every host |
   | GPU residency, packed K-quant | `smoke-gpu-residency.sh`, `smoke-packed-kquant-matmul.sh` | check | CUDA |
   | CPU kernels | `smoke-cpu-hot-path.sh`, Tier 02E's `smoke-cpu-kernels.sh`, and the kernel microbenchmark at 128 bits against forced 256 bits on 1 thread and all threads, interleaved (`cpu-decode-kernel-variants.sh`'s production variant) | check, reading | every host; the width comparison on 256-bit integer hosts |
   | Quality | `ModelQualityNllLiveTest`'s mean NLL per model | reading, compared at ingest | every host |
   | Throughput | `compare-llama-cpp.sh`'s two Juno lanes (prefill at `n_prompt` 128, 512 and 2048; generation) with JFR allocation and GC figures | reading | CPU; GPU where present |
   | Schedules | `compare-schedule.sh`, `compare-parallel.sh`, `compare-mixed-prefill.sh`, `compare-prefill-batch.sh` | reading | every host |
   | LoRA | `compare-lora.sh` (training step time, playback t/s); merge into a GGUF and reload | check, reading | every host; GPU playback where present |
   | Vision | `compare-vision.sh` (latency, decode t/s) on moondream2 | reading | every host |
   | Launcher | `juno` or `juno.bat` starts local mode with `--api-port`, answers one REST request, exits cleanly | check | every host (Windows exercises `juno.bat`) |

   Reference-tool comparisons are not in the catalog (owner decision 5, README rule 4). Juno-against-Juno
   comparisons inside one run are (two widths, a flag on and off, two schedules), and a remote reading
   never takes a ratio against this host.
10. **Packing and privacy.**
    - One zip per run, `juno-kit-results-<os>-<arch>-<UTC>.zip`, holding:
      - a manifest: schema version, kit version, the Juno commit (from git, or from the folder name of a
        GitHub commit archive, else `unknown`), the kit jar's SHA-256, the suites run, the hardware;
      - a JSON file and a log per probe, plus the test reports;
      - a readable `SUMMARY.txt` and an HTML report.
    - The user name, home directory, host name, and IP and MAC addresses are replaced with placeholders
      everywhere, logs and test reports included. Hardware model names are kept.
    - Nothing is uploaded: the kit's only network use is the Maven build and the model downloads.
11. **Owner side: `ingest`.** `juno-kit ingest <zip>` validates the manifest and drafts
    `docs/perf-compare/<UTC>-remote-<host-label>/INDEX.md` (tables with median, min and max, noise flags,
    hashes) plus a row for `docs/perf-compare/README.md`. It compares the correctness readings (NLL,
    greedy text, logit checksums within a tolerance) against the owner's reference run at the same
    commit. The owner reviews and publishes. `INVENTORY.md`'s "Borrowed hardware" table records each
    tester host by its hardware, not by the person.
12. **Reference run.** The owner runs the kit's `all` suite on this host at the same commit as a tester's
    build. It supplies the correctness references `ingest` compares against and the cost table the
    estimates scale, and is published once per kit release under `docs/perf-compare/<UTC>-kit-reference/`.
13. **Kept current with Juno.**
    - The module is in the reactor, so `mvn test` compiles and tests it.
    - Every probe has a self-test that runs in `mvn test` with no model and no GPU (a synthetic model the
      test writes, or a stub handler).
    - A coverage file maps every feature and cross-surface row to its probes. A unit test fails when a
      feature has no probe, or a probe names a feature that does not exist.
    - A new README execution rule 10 requires every tier that adds a feature, a flag or a threshold to name
      its kit probes. A new `check-plan-thresholds.sh` check 10 fails an open tier file that has no **Kit
      probes** section, and this tier adds that section to every open tier file.
    - New measurement logic goes into Java first. A bash script that is changed hands its logic to the
      probe and becomes a wrapper.
14. **CI on free hosted runners** *(approved by the owner 2026-10-13, decision 8)*. A
    workflow runs the kit's `quick` CPU suite on Linux x64 and arm64, Windows and macOS, on manual dispatch
    and weekly, and the probes' self-tests on pull requests that touch the kit. Each results zip is
    uploaded as a run artifact. It is a workflow file of its own: the untracked `ci.yml` and the owner's
    2026-10-01 decision about it are unchanged. Committing the file is the owner's action; its first green
    run's URL is recorded here.
15. **Docs.** `juno-kit/README.md` (the tester's guide: one page, standalone), a `docs/howto.md` section,
    `docs/agent-arch.txt`, `README.md` and `CHANGELOG.md`; Juno-native language only, and no pointer into
    this plan tree.

### Out of scope

- Reference-tool comparisons (owner decision 5; README rule 4 forbids competitor names in code).
  `compare-llama-cpp.sh` stays the only reference path, run on this host.
- Anything needing admin rights on the tester's machine: clock pinning, driver or toolkit installation,
  firewall changes. The kit detects what is missing and says how to install it.
- Scoring this plan's pinned gates on a remote host. Remote readings are same-run comparisons on that
  host. A threshold is scored on a named remote host only by an **Owner decision** recorded in the tier
  that owns the threshold.
- CUDA on Windows itself ([Tier 02G](TIER-02G-cuda-on-windows.md)). Until it lands, the kit reports a
  Windows NVIDIA GPU as not usable by Juno.
- Paid GPU runners in CI, and uploading results anywhere.

## Cross-surface compatibility checklist

The kit adds no engine feature; each row says how the kit exercises that surface.

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | every host: throughput, kernels, quality, live checks |
| 2 | CUDA GPU inference | Linux CUDA hosts now, Windows after Tier 02G: the `gpu` test group, residency, packed K-quant, throughput |
| 3 | ROCm GPU inference | Linux ROCm hosts: the `rocm` test group and the GPU probes. This host: self-tests only, NEEDS-AMD-HARDWARE until a tester's run |
| 4 | Static schedule | the default for every throughput probe |
| 5 | Continuous schedule | schedule probes |
| 6 | Single-node local mode | every probe's default |
| 7 | Pipeline-parallel cluster | live checks 1 to 6 (local forked JVMs) |
| 8 | Tensor-parallel cluster | live checks 7 and 8 |
| 9 | LoRA training | the LoRA probe trains its adapter |
| 10 | LoRA playback | the LoRA probe plays it back; the merge check |
| 11 | Vision | the vision probe |
| 12 | OpenAI REST surface | grammar, tools and launcher probes |
| 13 | Native REST surface | same |
| 14 | CLI | the launcher probe (`juno`, `juno.bat`), and the kit's own CLI |
| 15 | JVM embedding facade | probes call `JunoPlayer`, `LoraTrainer`, `JunoHttpClient` |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Pin the model catalog: SHA-256 of this host's files, matched to Hugging Face sources.
3. Build the module skeleton, bootstrap and launchers. Then detection, selection, estimate, warnings,
   runner, checkpoint, packing and scrubbing, with the environment probe and the `quick` CPU probes
   (throughput at 128 and 512, the CPU kernel width comparison, live checks on TinyLlama). Run end to end
   on this host.
4. CUDA probes, run on this host.
5. The remaining probe groups in the catalog table's order, each one's self-test first.
6. The coverage file and its test; `ingest`; the reference run on this host, published.
7. Plan wiring: README rule 10, and `check-plan-thresholds.sh` check 10, shown failing on the tree before
   the **Kit probes** sections exist and then passing. Add those sections to every open tier file, and the
   "Borrowed hardware" table to `INVENTORY.md`.
8. The CI workflow (approved, decision 8); the owner commits it, and its first green run is recorded.
9. First runs on Windows and macOS (the CI runners, or testers), each zip ingested and any defect
   fixed. First run on a 256-bit integer host, its reading recorded against Tier 02E's width row.
10. Docs and `CHANGELOG.md`.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other test
  or code in this tier (README execution rule 7).
- **Selection**: suites to probes; requirements against detected hardware; `--budget` fitting by priority;
  `--plan` performing no writes, downloads or builds.
- **Estimate**: calibration scaling and ranges from a recorded cost table, and refinement as probes finish.
- **Detection parsers** on recorded outputs from Linux, Windows and macOS (`/proc/cpuinfo`, `sysctl`, the
  Windows processor and battery queries, `nvidia-smi`, `rocm-smi`), including absent tools.
- **Scrubbing** on fixtures from each OS: no user name, home path, host name, or IP or MAC address remains,
  and nothing else changes.
- **Manifest** round trip, and `ingest` of a zip from an older schema version.
- **Checkpoint and resume**: a run interrupted after any probe resumes without repeating it; `pack` works on
  a partial run.
- **Noise rule**: a reading the README's outlier rule flags is re-run and, if still noisy, marked.
- **Bootstrap**: a failed build produces a zip holding the build log; `KitBootstrap` imports JDK classes
  only.
- **Probe self-tests**: every probe runs in `mvn test` with no model and no GPU, and the coverage test fails
  on a feature with no probe.
- **End to end on this host**: `juno-kit.sh --suite quick --yes` completes, and its zip passes `ingest`.

**No perf gate**: this tier adds a tool and changes no inference path (forward pass, MatVec, KV, batching,
quantization). The kit's own acceptance numbers are in the exit criteria.

## Kit probes

README execution rule 10.

- Keep passing: every probe's self-test and `CoverageTest`, in `mvn test -pl juno-kit`.
- Add: this tier builds the kit: the catalog (74 probes, `./juno-kit.sh --list`), `coverage.tsv` and its test,
  `ingest`, and the reference run on this host. Later tiers' sections name the probes they keep and add.

## Models needed

All present here (`INVENTORY.md`): TinyLlama 1.1B Q4_K_M, Qwen2.5-3B, Phi-3.5-mini and Mistral 7B (the
sweep models), Qwen3-1.7B, moondream2. Testers download them through the catalog (step 2). The hardware
this tier exists for is what this host lacks: a 256-bit integer CPU, an AMD GPU, Windows with an NVIDIA
GPU, macOS.

## Exit criteria

- [ ] `juno-kit` is in the reactor with the `juno-kit.sh` and `juno-kit.bat` launchers; the bootstrap checks
      prerequisites before building, and a failed build still produces a results zip.
- [ ] Detection, the suites, `--budget`, `--plan`, `--yes`, `--list`, `resume` and `pack` work as
      specified, each unit-tested.
- [ ] The estimate is shown before the run and refined during it. On this host, the first estimate for the
      `quick` suite is `>= 0.65x` and `<= 1.35x` the actual time.
      *(2026-11-03: read 0.741x, point 1738 s against 2345 s, on the `a1fe7166` catalog
      (`docs/perf-compare/20261103T205950Z-kit-quick-estimate/`). Not ticked: three catalog costs were corrected after
      that reading, and `cpu.live.tinyllama` ran about 200 s long for a reason inside the kit that is still open. Owed by
      the executor, no owner run: a fresh `quick` run on the corrected catalog once that finding is explained.)*
- [ ] Stage warnings, keep-awake on each OS, and the noise rule's re-runs work as specified.
- [ ] The model catalog is pinned by SHA-256, downloads resume, and hashes are verified.
- [x] The probe catalog covers every row of the README's smoke matrix and the Juno-side checks and readings
      of every smoke and compare script. Every probe has a self-test in `mvn test`, and the coverage test
      passes. *(2026-10-31, step 6: `CoverageTest` passes against the catalog, the scripts on disk and the
      supported architectures; 21 probe classes run in `ProbeSelfTests`, and `UnitTestsProbe`'s self-test is
      `SurefireReportsTest`, its command and report parsing, since running Maven inside `mvn test` would recurse.)*
- [ ] Results are scrubbed (fixture tests), the zip holds `SUMMARY.txt` and the HTML report, and nothing in
      the kit uploads.
- [ ] `ingest` drafts the `docs/perf-compare/` entry and compares against the owner's reference run; the
      reference run on this host is published.
- [ ] The kit has completed at least once each on Windows, macOS and Linux (CI runner or tester), each zip
      ingested.
- [x] Rule 10 and check 10 are in place (check 10 seen failing first); every open tier file has its **Kit
      probes** section; `INVENTORY.md` has the "Borrowed hardware" table.
      *(2026-10-31, step 7: check 10 failed on all 17 open tier files, then passed once the sections were
      written; see that day's step 7 record.)*
- [ ] Every module's unit tests pass, `juno-kit` included, and `mvn verify -pl juno-master` passes.
- [ ] Cross-surface checklist fully resolved.
- [x] Docs (`juno-kit/README.md`, `docs/howto.md`, `docs/agent-arch.txt`, `README.md`) updated, in
      Juno-native language only; `CHANGELOG.md` entry added.
      *(2026-11-03, step 10: all five written; a grep of the added text finds no competitor product name, tier number,
      emoji or pointer into this plan tree. A step 9 fix that changes what a tester sees updates them in the same change.)*

## Execution record

### 2026-10-13: tier created (owner decisions)

Created at the owner's request: hardware this host lacks is to be tested by other people, each running one
command and sending back one file. Owner decisions, 2026-10-13:

1. Runs after Tier 02E and before [Tier 02G](TIER-02G-cuda-on-windows.md) (CUDA on Windows), which needs it
   for verification.
2. The module is named `juno-kit`.
3. Written in Java (no bash or PowerShell); builds the Juno copy it ships in; compares inside one run
   instead of needing pinned clocks; protects the tester's privacy; records failures and continues.
4. The repository is public, so testers clone it or download a commit archive.
5. The reference tool is not part of the kit (README rule 4).
6. The kit detects what it can run and lets the tester choose (all, gpu, cpu, custom). It estimates the
   time, warns what the tester may and may not do meanwhile, and shows progress.
7. The kit covers every needed measure for every Juno feature, and is kept current with Juno (scope
   item 13).

**Findings at creation (code read, 2026-10-13).**
- The bash harness is about 11,400 lines and uses `curl` and `jq`, so none of it runs on Windows.
- `ModelLiveChecks.run(modelPath, suites, progress)` is public and returns per-check results, so the live
  checks are reused, not ported.
- `HfGgufFetcher` already resumes and caches downloads.
- The free hosted CI runners for public repositories (Linux x64 and arm64, Windows x64, macOS on Apple
  silicon) have no GPU.

**Decision 8 (owner, 2026-10-13): CI on free hosted runners is approved** (scope item 14), as a workflow
for the kit only. It gives the kit its first Windows, macOS and arm64 runs without a tester, and a first,
noisy 256-bit integer reading on x64 runners. Raised as a reversal of the owner's 2026-10-01 decision
(README, "Test infrastructure"); that decision stands for `ci.yml`, and the README records the change.

### 2026-10-28: steps 1 and 2 (plan check; model catalog pinned)

**Plan against code, verified at HEAD `06d6d22d`** (Tier 02E closed; tree clean apart from the untracked `.github/`).
`scripts/performance-tests/check-plan-thresholds.sh` passes (25 tier files). Every class the tier names exists:
`ModelLiveChecks` (`juno-master`, public `run(modelPath, suites, progress)`), `ClusterHarness`, `JunoPlayer`, `LoraTrainer`,
`JunoHttpClient`, `HfGgufFetcher`, `AnsiSupport` (`juno-player`), `JfrMetricsExtractor` (`metrics`), `CudaAvailability`,
`RocmAvailability` (`node`), `ModelQualityNllLiveTest` (`node` tests); `JfrSpanBreakdown` is a source-launched script under
`scripts/performance-tests/`, not a module class. Every smoke and compare script in the probe catalog table exists, as do
`juno` and `juno.bat`. All six catalog models are under `models/`.

**Drift found (scope item 7), changes step 3's work, not its scope.** `HfGgufFetcher` resumes (`Range`) and caches (ETag)
as stated, but (a) `download(repo, asset)` is package-private, so a class outside `cab.ml.juno.player` cannot fetch a
named file; (b) `listGgufAssets` keeps only `.gguf` siblings, so moondream2 (a `.llamafile`) cannot be fetched through
`resolve`; (c) it verifies no hash; (d) it always fetches `main`, so a re-upload changes the bytes under a fixed name; and
(e) its class comment says the cache directory joins org and repo with `__`, while `sanitizeRepoDirName` uses `_`. Step 3
gives the kit a fetch of a named file at a pinned commit with SHA-256 verification, reusing the fetcher's resume.

**Step 2: the catalog, pinned.** SHA-256 of this host's files (`sha256sum`, digests only), matched against the LFS SHA-256
that the Hugging Face tree API publishes (`/api/models/<repo>/tree/main`), then confirmed on the pinned-commit download
URL itself (`HEAD /<repo>/resolve/<commit>/<file>`: `302`, `X-Linked-Etag` and `X-Linked-Size` equal to this host's file).
Nothing was downloaded. Every repo reports `gated=false`.

| Model | Repo | Commit | File | Bytes | SHA-256 |
|---|---|---|---|---|---|
| TinyLlama 1.1B Chat Q4_K_M | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `52e7645ba7c3` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 668,788,096 | `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` |
| Qwen2.5-3B Instruct Q4_K_M | `Qwen/Qwen2.5-3B-Instruct-GGUF` | `7dabda4d13d5` | `qwen2.5-3b-instruct-q4_k_m.gguf` | 2,104,932,768 | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |
| Phi-3.5-mini Instruct Q4_K_M | `bartowski/Phi-3.5-mini-instruct-GGUF` | `6d70da17e749` | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | 2,393,232,672 | `e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5` |
| Mistral 7B Instruct v0.1 Q4_K_M | `itlwas/Mistral-7B-Instruct-v0.1-Q4_K_M-GGUF` | `1a037a56dcd9` | `mistral-7b-instruct-v0.1-q4_k_m.gguf` | 4,368,440,864 | `1af6bf966c4113f0cb5de7af42f38572075c9798bc16e6c72faee051bd7d6d33` |
| Qwen3-1.7B Q4_K_M | `unsloth/Qwen3-1.7B-GGUF` | `d7f544eead69` | `Qwen3-1.7B-Q4_K_M.gguf` | 1,107,409,472 | `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` |
| moondream2 Q5_K (llamafile) | `cjpais/moondream2-llamafile` | `8abde7e79de0` | `moondream2-q5_k.llamafile` | 1,977,619,136 | `df6cf6bc9799946435704dafdf925c65295af5aa4e48740c1bcff57856607564` |

Full commit ids: `52e7645ba7c309695bec7ac98f4f005b139cf465`, `7dabda4d13d513e3e842b20f0d435c732f172cbe`,
`6d70da17e749a471ccb62ade694486011a75cda3`, `1a037a56dcd9c6ad39ff8459dec3f47f287dcaf4`,
`d7f544eead698dbd1f15126ef60b45a1e1933222`, `8abde7e79de0c3bbbfeb29827ef594f36bc5fe67`.

- **Mistral 7B is not TheBloke's file.** `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`'s Q4_K_M is 4,368,438,944 bytes (SHA-256
  `14466f9d...`), 1,920 bytes shorter than this host's. A search of 106 Mistral 7B v0.1 GGUF repos found two files of this
  host's size, and only `itlwas/...`'s hash matches (`NonMiFrega/...`'s is `5e0c164e...`). It is a single-user conversion
  repo; if it disappears, the fallback is TheBloke's file, whose bytes differ, so the reference run and every remote
  reading of it would be re-taken on that file. Recorded as a risk, not raised: the plan pins this host's bytes.
- **Qwen3-1.7B has no official Q4_K_M** (`Qwen/Qwen3-1.7B-GGUF` publishes none); `lmstudio-community`'s Q4_K_M is a different
  file (1,282,439,328 bytes). The unsloth file is this host's.
- Downloads go to the pinned commit, never `main`, so a re-upload to `main` cannot change what a tester gets.

### 2026-10-28: step 3 (module, bootstrap, launchers, runner, the `quick` CPU probes; run end to end here)

**Built.** Module `juno-kit` (reactor, after `juno-node`; `maven.install.skip`/`maven.deploy.skip`, excluded from Central
publishing in the root pom's `central-publish` profile, not in `juno-bom`; shaded jar written beside the thin jar, main class
`KitMain`). Launchers `juno-kit.sh` and `juno-kit.bat` at the root (scope item 1 corrected above: `juno-kit` is the module
directory). `juno-kit-results/` is git-ignored. Classes, package `cab.ml.juno.kit`:
- `KitBootstrap` (JDK classes and Java 17 syntax only, run from source): JDK 25 check, Maven discovery (`MAVEN_HOME`, `M2_HOME`,
  `PATH`; `cmd.exe /c mvn.cmd` on Windows), free disk, Maven Central and huggingface.co reachability, the build
  (`mvn -B package -DskipTests -pl juno-kit -am`, reactor steps echoed, full log kept), the kit JVM launch. A failed build packs
  `build.log`, `environment.txt` and a `build-failure` manifest into the results zip. It also holds `Privacy` (scrubbing) and
  `zip`, which the kit reuses, so each has one implementation. `resume`, `pack` and `--plan` reuse a built jar, so a resumed run
  keeps its jar's hash; `--plan` without a built jar says so and builds nothing.
- Detection: `HostDetector`, `DetectionParsers` (`/proc/cpuinfo`, `sysctl`, `Win32_Processor`, `nvidia-smi`, `rocm-smi`, Linux
  power supplies, `pmset`, `Win32_Battery`, `nmcli`), and `EnvProbe` in a child JVM (vector widths, `UseAVX`, `CpuThreads`,
  `CudaAvailability`, `RocmAvailability`), which reports on stdout so detection writes no file.
- Choice and estimate: `KitOptions` (`--suite`, `--probes`, `--budget`, `--yes`, `--plan`, `--list`, `--models-dir`,
  `--cache-dir`, `--out`; `resume`, `pack`), an interactive menu when a console is attached, `Selection`, `Estimator`
  (reference cost x calibration scale + download bytes / measured rate, shown as 0.85x to 1.20x; after 60 reference seconds
  have run, the observed pace replaces the calibrated scale).
- Running: `ProbeRunner` (fresh JVM per probe, Juno's runtime flags, `-Xms`=`-Xmx` and pre-touch for readings, JFR on the OS
  clock where the kernel rejected the TSC (the harness's rule), timeout, crash and timeout recorded, process tree killed on
  timeout and on kit shutdown), `LoadSampler` (other processes' CPU share, a process that exits keeps its last CPU time),
  `NoiseRule` (README rule: a scored reading whose repetitions spread over 15% of the median, or one taken while other
  processes used over 5% of the CPU, is re-run up to twice, then marked `NOISY` with the reason; GC pause and allocation
  readings are published, not scored), `RunSession` + `Checkpoint` (written after every probe), `Progress`, `KeepAwake`
  (`SetThreadExecutionState` through FFM on Windows, `caffeinate -i -w` on macOS, `systemd-inhibit ... tail --pid` on Linux).
- Packing: `Manifest` (schema 1), `ReportWriter` (`SUMMARY.txt`, `report.html`), `ResultsPacker`; only text files are packed
  (JFR recordings, which carry host data, never are), every one scrubbed.
- Models: `CatalogModel` (step 2's table), `ModelStore` (name-and-size lookup in `--models-dir`, read only, then the cache;
  SHA-256 verified once per path, size and mtime, the record kept in the cache dir; download at the pinned commit).
- Probes (`ProbeMain` dispatches by a switch, so a renamed probe breaks the build): `env`, `calibration` (shipped Q4_K decode
  kernel, 4096x4096, all threads), `cpu.kernel-width` (128 against forced 256 bits, 1 and all threads, Mistral 7B FFN shapes,
  one JVM per timing, interleaved; needs a 256-bit integer JVM), `live.tinyllama` (`ModelLiveChecks` 1 to 10),
  `cpu.throughput.tinyllama.128` and `.512` (the harness's two Juno lanes in-process through `JunoPlayer`: word-count prompt
  calibration, 2 warmups, 3 reps, each request in its own JFR recording read by `JfrMetricsCli`; prefill t/s off the prefill
  spans, generation t/s off the decode spans, the harness's span check withholding a rep whose spans do not fit its request).

**Changes in other modules (in this tier's scope; no inference path touched, not a measurement boundary).**
`HfGgufFetcher.downloadPinned(repo, revision, filename, dir)` (public; any extension; resume shared with `download` through
one private `transfer`); its class comment's cache-directory separator corrected (`_`, as the code does).
`CpuDecodeKernelVariantMicrobench.productionMsPerCall(...)` (public; times only the shipped kernel, which the probes call).
Root `pom.xml` (module, publishing exclusion), `.gitignore`.

**Tests, written before the code they test.** `SelectionTest` 8, `EstimatorTest` 5, `NoiseRuleTest` 5, `DetectionParsersTest`
10, `PrivacyTest` 5, `ManifestTest` 4, `RunSessionTest` 2, `KitBootstrapTest` 4 were run against stubs first: **37 of 43
failed** (assertions, or the stubs' missing behaviour); the 6 that passed are ones whose stub default is the expected value
(both JDK-only compile checks, "checks are never re-run", "nothing else changes", the 256-bit host's selection, the peripheral
battery). All 43 passed after the implementation. `HfGgufFetcherTest` (2 new) failed to compile first, then 18 of 18.
Written after the code, passed on first run (regression tests, not seen failing): `ProbeRunnerTest` 3 (real child JVMs: works,
crash, timeout), `KitOptionsTest` 3, `KeepAwakeTest` 1, `PlanTest` 2 (`--plan` and `--list` create nothing; every catalog
model faked as a sparse file of its size, so no network), `ModelStoreTest` 3 (local server: a wrong-hash local copy replaced by
a verified download, a tampered download refused and deleted). `ProbeSelfTests` 5 (every probe in-process on a synthetic
three-layer LLaMA GGUF the test writes, no model file, no GPU: the live-checks self-test runs the tensor suite, whose nodes run
on the CPU) failed first on two setup defects, fixed: the kit pom lacked `juno-node` (`NodeMain`), and two layers cannot be
split across three pipeline shards. `NoiseRuleTest` gained the unscored-reading case. **`juno-kit`: 62 tests, 0 failures.**
The full eleven-module unit suite and `mvn verify -pl juno-master` were not run in this step (owed at the tier's close).

**End to end on this host** (`./juno-kit.sh --suite quick --yes --models-dir models`, unpinned; TinyLlama read from `models/`,
hash verified, nothing downloaded):

| Run | Build (Maven total, local repository filled) | Calibration median (ms/call) | live.tinyllama | prefill t/s at 128 / 512 (spread) | generation t/s at 128 / 512 (spread) | First estimate | Actual |
|---|---|---|---|---|---|---|---|
| 1 (`20261028T155213Z`) | 38 s | 0.9075 (2 attempts) | 10 of 10 PASS, 96 s | 35.82 (3.4%) / 33.14 (2.6%) | 12.78 (1.0%) / 12.63 (1.3%) | 683 to 964 s (placeholder costs) | 282 s |
| 2 (`20261028T160128Z`) | 37 s | 0.8757 | 10 of 10 PASS, 95 s | 35.24 (5.5%) / 34.45 (2.2%) | 12.85 (0.6%) / 12.57 (1.3%) | 242 to 342 s (point 285 s) | 279 s |

- Run 1 set the catalog's reference costs and `REFERENCE_CALIBRATION_MS = 0.9075`; run 2's first estimate is **1.02x** the
  actual time, inside exit criterion 3's band. Not ticked: it was read on the host the costs came from, and steps 4 and 5 add
  probes to `quick` (the scope's 30 to 60 minutes), so the criterion is scored on the finished suite.
- Calibration's first attempt in run 1 was re-run by the noise rule (its values spread 10%, under 15%, so other processes'
  share was over 5%, just after the Maven build). The runner now keeps each re-run's reason in the outcome's note.
- `cpu.kernel-width` was listed as not runnable ("256-bit integer vectors: this JVM prefers 128-bit"), as expected here.
- Interrupt and resume (run `20261028T160659Z-custom`): `TERM` to the kit JVM during the throughput probe stopped the probe JVM
  with it (shutdown hook); the checkpoint held the two finished probes; `juno-kit.sh resume` ran only the third; `pack` repacked
  the finished run.
- Privacy, checked on run 1's zip: no user name, host name or home path; 115 home paths replaced. It also scrubbed
  `com.google.android:annotations:jar:4.1.1.4` as an address. Fixed test first (the case added to "nothing else changes", and
  IPv4-mapped IPv6 and net-tools `addr:` cases added; seen failing, then passing): a dotted quad right after `:` or `-` is a
  version, except after `addr:`.
- Generation allocation in the recording reads about 0.40 MB per token in-process through `JunoPlayer`; a reading, not a gate.

**Findings, recorded for the steps that own them.**
- `ModelLiveChecks` has no backend choice: checks 9 and 10 take CUDA whenever it is present (`ContextShiftCheck`,
  `longPromptPrefill`), checks 1 to 8 always run on the CPU. So the step-3 probe is `live.tinyllama`, not a CPU probe; step 4's
  CPU and GPU split needs a backend parameter there.
- The macOS, Windows and ROCm detection fixtures follow documented output formats; they are not recordings. The Linux CPU,
  power-supply and NVIDIA fixtures are this host's. Recorded outputs arrive with step 9's first runs.
- `KeepAwake`'s Windows (FFM) and macOS paths, and `juno-kit.bat`, have not run anywhere yet (step 9). On Linux the inhibitor
  was started; whether it held was not observed.
- The kit's throughput lanes were not cross-read against `compare-llama-cpp.sh`'s on the same build; the reference run (step 6)
  is the natural place.

**Exit criteria after step 3:** none ticked. Owed: Windows and macOS runs of the launchers, keep-awake and detection (step 9);
the GPU probes and the remaining groups (steps 4, 5); the coverage file, `ingest` and the reference run (step 6, the owner's
run); rule 10 and check 10 (step 7); CI (step 8, committed by the owner); docs and `CHANGELOG.md` (step 10); the full unit suite
and `mvn verify -pl juno-master` at the close.

**Next: step 4**, CUDA probes on this host: a backend choice for `ModelLiveChecks` (GPU live checks), GPU throughput at 512
and 2048 (`gpu.throughput.*`), the `smoke-gpu-residency.sh` and `smoke-packed-kquant-matmul.sh` checks, the `gpu` test group,
and GPU use by other processes in the load sampler.

### 2026-10-28: step 4 (CUDA probes; run on this host)

**Plan against code, verified.** `check-plan-thresholds.sh` passes; tree as step 3 left it; the GPU idle apart from the
desktop (10% utilization, no compute process). `smoke-gpu-residency.sh` and `smoke-packed-kquant-matmul.sh` judge the
device paths by engine log lines (`GPU-resident decode region active`, `requested, but`, `Tiled K-quant GEMM kernels
loaded`, `Fused Q4_K MMQ enabled`, `Tiled K-quant GEMM kernel unavailable`) and per-process device memory from
`nvidia-smi`; no Java accessor reports either, so the probes read the same lines through a `java.util.logging` handler
(`NoticeCapture`) and the same `nvidia-smi` query. `JunoPlayer` reads the schedule (`ServeScheduleOptions.ENV`) and
residency (`JUNO_GPU_RESIDENCY`) properties when it builds, so a probe JVM selects both. The `gpu` JUnit tag marks 75 test
classes in `node`; no module excludes the tag, so on a CUDA host the default `mvn test` runs them too.

**Drift found: the live checks' cluster nodes take CUDA wherever it is present** (correcting step 3's finding, which said
checks 1 to 8 always run on the CPU, and the step-3 probe's title, "clusters on the CPU"). `NodeMain` reads
`JUNO_USE_GPU` (default true) and `ClusterHarness` never passed it, so forked nodes used the GPU: a tensor-parallel
run of checks 7 and 8 from the kit jar showed three `NodeMain` JVMs holding about 916 MiB each. Tier 02E's closing record
claimed `ModelLiveRunnerIT` checks 1 to 8 ran on the CPU and scored its cross-surface rows 7 and 8 on that; corrected there
(dated notes, no box changed) after the CPU re-verification below. The step-3 self-test of the live-checks probe ran its
nodes on this host's GPU inside `mvn test`; it now runs them on the CPU.

**Implementation, tests first.**
- `ModelLiveChecks.run(modelPath, suites, Backend, progress)`: `AUTO` (the old call, unchanged), `CPU` (every check and
  node on the CPU), `CUDA` (all on the GPU; without a usable one every selected check fails closed with the reason). Checks
  9 and 10 (`longPromptPrefill`, `ContextShiftCheck.run`) take the choice instead of probing CUDA themselves; the cluster
  suites set `JUNO_USE_GPU` for their lifetime, restored after. `ClusterHarness` forwards `JUNO_USE_GPU` to the nodes it
  forks, as it forwards the KV, schedule and residency settings. `ModelLiveChecksBackendTest` (3) failed to compile first,
  then passed.
- Kit: `Requirements.gpu` (CUDA or ROCm) and `.rocm`; `GpuLoad` (`nvidia-smi` compute processes split into the probe's own
  and others, utilization); `LoadSampler` samples them every other second and records the probe's own peak device memory;
  before a GPU probe the runner samples utilization three times a second apart and calls the GPU busy if even the quietest
  sample is over 20% (graphics programs are not compute processes, so this stands in for them); `NoiseRule.verdict` re-runs a
  reading taken with the GPU busy, as it does one taken under CPU load. `ProbeOutcome` carries the probe's peak device
  memory and the busy flag, and `SUMMARY.txt` prints both. `GpuLoadTest` (3), `SurefireReportsTest` (3), one new
  `NoiseRuleTest` case and one new `SelectionTest` case failed to compile first, then passed.
- Probes: `cpu.live.tinyllama` (backend CPU; replaces `live.tinyllama`), `gpu.live.tinyllama` (backend CUDA),
  `gpu.throughput.tinyllama.512` and `.2048` (the throughput probe with `gpu=true`), `gpu.residency.tinyllama`
  (`GpuResidencyProbe`: the residency smoke's local checks, region off then on, each mode in a JVM of its own as each runs
  in its own server in the script; 4 requests, 32 and 8 tokens; growth limit 8 MiB per request over the second half, and
  no more on than off; output identical on and off), `gpu.packed-kquant.tinyllama.static` and `.continuous`
  (`PackedKquantProbe`: about 512 prompt tokens within 10%, exactly 512 generated, one stream piece per token, streamed equal
  to unstreamed, the packed-path log lines present and the fallback absent; time to first token and peak device memory
  recorded), `gpu.unit-tests` and `rocm.unit-tests` (`UnitTestsProbe`: `mvn -B test -Dgroups=<tag> -pl node -am` in the
  checkout, each failed test a check, Maven's output in `<id>.maven.log`). Not ported: the residency smoke's cluster part
  (forked nodes with the region on must answer and match the local output); `gpu.live.tinyllama` covers the clusters on CUDA
  with the region at its default, without the output comparison. Owed with step 5.
- Self-tests, no GPU: the live-checks self-test runs with the CPU backend; the residency and packed probes run their
  mechanics on the synthetic model with the GPU off, where their device-only checks must report themselves not met (the
  packed-path check fails with its reason; the memory check says there is no reading). Both passed on their first run
  (regression tests). **`juno-kit`: 72 tests, 0 failures.**

**End to end on this host** (`./juno-kit.sh --suite <s> --yes --models-dir models`, unpinned):

| Run | Probes | Result | First estimate | Actual (after the estimate) |
|---|---|---|---|---|
| `gpu` (`20261028T170703Z`) | 9 run, `rocm.unit-tests` not runnable (no AMD GPU) | every probe PASS or DONE | 882 to 1245 s | 1664 s (0.62x; the unit-test group's cost was a guess, 600 s against 1434 s) |
| `quick` (`20261028T173815Z`) | 11 run, `cpu.kernel-width` not runnable (128-bit) | every probe PASS or DONE | 724 to 1022 s (point 852 s) | 662 s (estimate 1.29x actual; the CPU live checks' guess was 420 s against 244 s) |

- `gpu.live.tinyllama`: 10 of 10 PASS, 99 s, 2,748 MiB of device memory across the probe and its nodes.
  `cpu.live.tinyllama`: 10 of 10 PASS, 244 s, **0 MiB**: no node touched the GPU. Every CPU probe read 0 MiB.
- GPU throughput, TinyLlama (`quick` run): prefill 2,768 t/s at 512 and 2,322 t/s at 2048, generation 165.4 and 167.2 t/s,
  spreads 1.7% to 5.1%; the `gpu` run read within 1.5%. Generation allocation inside the recording reads 0 B per token on
  the GPU: 64 tokens take about 0.4 s, under the 500 ms sampling period of `jdk.ThreadAllocationStatistics` (the limit the
  README records for the harness), so the GPU allocation reading is not informative at this length.
- `gpu.residency.tinyllama`: region active on 20 of 20 layers (whole layer on all 20), output identical over 32 tokens,
  device memory flat at 938 MiB over all four requests in both modes.
- `gpu.packed-kquant.tinyllama.*`: every check PASS on both schedules; time to first token and peak memory recorded.
- `gpu.unit-tests`: 367 tests in the `gpu` group (`node` and what it builds), 0 failures, 0 errors, 7 skipped, 23 min 54 s;
  `PrefillRegionHandlerParityTest` (636 s) and `SlidingWindowGpuTest` (365 s) are most of it. The device-wide free-memory
  checks passed in this run.
- No probe saw another compute process or a busy GPU; other processes' CPU share stayed at 0 to 4.2%.
- Reference costs set from these runs (CPU live checks 250 s, GPU unit tests 1,450 s, the GPU probes 15 to 100 s), so the
  next estimate is scored on measured costs. Exit criterion 3 stays open until the `quick` suite is complete (step 5).

**CPU re-verification of Tier 02E's cluster rows.** With the backend choice in place, `LiveChecksProbe` with `backend=cpu`
was run from the kit jar (`666458f684aa5868`) on the five models Tier 02E's IT named, `nvidia-smi` sampled every 5 s
throughout (no compute process in any sample):

| Model | Heap | Checks 1 to 8 (clusters, CPU nodes) | Checks 9, 10 | Time |
|---|---|---|---|---|
| TinyLlama 1.1B | 4 GiB | PASS | PASS | 390 s |
| Qwen2.5-3B | 5 GiB, then 16 GiB for 9 and 10 | PASS | `OutOfMemoryError` at 5 GiB; PASS at 16 GiB | 448 s + 395 s |
| Phi-3.5-mini | 8 GiB | PASS | PASS | 850 s |
| Mistral 7B | 10 GiB, then 24 GiB for 9 and 10 | PASS | `OutOfMemoryError` at 10 GiB; PASS at 24 GiB | 922 s + 824 s |
| Qwen3-1.7B | 4 GiB, then 16 GiB for 9 and 10 | PASS | `OutOfMemoryError` at 4 GiB; PASS at 16 GiB | 256 s + 521 s |

On the CPU, checks 9 and 10 hold the weights on the heap with a dense KV sized to the model's context (32k for Qwen2.5-3B and
Mistral 7B, 40k for Qwen3-1.7B); the IT's runs took CUDA with failsafe's default heap of a quarter of RAM. Per-family CPU
live probes (step 5) size their heaps from this table.

**Changes in other modules (in this tier's scope; not a measurement boundary).** `juno-master`: `ModelLiveChecks` backend
choice, `ContextShiftCheck.run` takes it, an unused import dropped. `juno-player`: `ClusterHarness` forwards
`JUNO_USE_GPU`. No forward-pass, MatVec, KV or batching code changed. `juno-player` and `juno-master` unit tests:
`mvn test -pl juno-player,juno-master`:
**127 and 13 tests, 0 failures** (2 skipped), including `HfGgufFetcherTest`, `ContextShiftCheckTest` and the new backend test.
The full eleven-module suite and `mvn verify -pl juno-master` remain owed at the close.

**Exit criteria after step 4:** none ticked. Owed as listed under step 3, less the CUDA probes; the `rocm` probes and
detection are `NEEDS-AMD-HARDWARE` until a tester's run.

**Next: step 5**, the remaining probe groups in the catalog table's order, each self-test first: the full unit-test
group, per-family live checks (CPU and GPU), consistency, grammar and tools through `JunoHttpClient`, context policy and
long prompt, the CPU kernel smokes, quality (NLL), schedules, LoRA, vision and the launcher; and the residency smoke's
cluster part.

### 2026-10-28: step 5, part 1 (probe groups from unit tests to throughput)

Stopped at a context limit after the throughput group; the handoff for part 2 is
`docs/infra-plan/PROMPT-kit-step5-rest.md`.

**Plan against code.** As step 4 left it. Two smokes need what the kit cannot download: the consistency smoke's
architecture audit uses four large unsupported-architecture files, and Juno checks the architecture before it reads
anything else, so the probe writes header-only GGUF files declaring those architectures (`HeaderOnlyGguf`). The REST
smokes run the player jar's launcher, whose server is wired differently from `JunoPlayer.startApiServer` (its own
registry, prefill sizing, batch config and vision routes), so the probes start the launcher itself in a child JVM
(`JunoProcess`), stopped by typing `exit` so its shutdown hooks (the JFR recording) run on Windows too.

**Findings.**
- **Constrained decoding switched alternatives mid-sequence (engine defect, fixed).** The grammar probe's self-test got
  `yno` under `root ::= "yes" | "no"`. `GbnfGrammar` opened the next alternative from any position of a sequence, not only
  from its start, so after `y` the state still accepted `n`; every grammar with alternatives (JSON Schema, tool calls,
  `x_juno_grammar`) was affected, though real vocabularies mostly hide it behind whole-word tokens. Fix in `sampler`: an
  alternative is entered only at its first element (`nextAlternative`). Two `GbnfGrammarTest` cases failed first (`yn`
  and `ab` accepted as prefixes), then passed; `sampler` 89 tests and `coordinator` (tool-call grammar, OpenAI handler)
  399 tests, 0 failures. Not a measurement boundary: the mask runs
  per token but its cost is unchanged.
- **Intermittent batch dependence on the CPU static schedule (cause found the same day, not a KV leak: see "CPU batch
  dependence investigated" below; fixed in the engine 2026-10-29, see "CPU decode batch made bit for bit the
  single-request decode").** `consistency.tinyllama`
  failed in 2 of 4 runs: with `--parallel 2`, TinyLlama's greedy answer to "How many legs does a spider have?" was "A
  spider has six legs." alone and "A spider has eight legs." in some batches (an OpenAI round in one run, a native round in
  another). `gpu.consistency.tinyllama` passed; the unmodified `smoke-consistency.sh --no-gpu --skip-audit` passed its one
  run at this commit (19 PASS). The cause is not determined: a KV leak, which is what the smoke exists to catch, or
  rounding that depends on the batch's composition flipping a near tie. The check stays strict.
- `JunoHttpClient.blockingInference` sends no temperature, so a native request through the facade uses the server's
  default sampling: the probe's first native rounds were not greedy. The consistency probe sends the smoke's native body
  (`sampling.temperature 0`) itself; OpenAI requests go through the facade.
- The context-policy smoke's limit error is Phi-3's alone (`Phi3RopeConfig`, the original training context); every
  other family's limit is the KV cache's 32,768 positions. Its probes therefore run on Phi-3.5-mini, CPU and GPU.

**Probes added** (catalog order; self-tests in `ProbeSelfTests` unless noted):

| Group | Probes | Notes |
|---|---|---|
| Unit tests | `cpu.unit-tests` | the documented unit-test modules, tags `gpu` and `rocm` excluded (their own probes); command in `SurefireReportsTest` |
| Live checks | `cpu.live.` and `gpu.live.` + qwen2.5-3b, phi-3.5-mini, mistral-7b, qwen3-1.7b | heaps from step 4's CPU table; `Requirements.withDeviceMemory` added (largest GPU's memory; an unread GPU does not block), minimums to be set from measured peaks |
| Consistency, grammar, tools | `consistency.tinyllama`, `gpu.consistency.tinyllama`, `grammar.tinyllama`, `tools.qwen2.5-3b` | `ConsistencyProbe`, `GrammarProbe`, `ToolsProbe`; LoRA overlays trained in the run for one step (`KitAdapters`); constrained-request counts from the server's JFR recording (`JfrEvents`); not ported: the grammar smoke's source-text check that `run.sh` forwards grammar flags, and the LoRA-train warning, which `run.sh` prints (owed to the launcher probe) |
| Context policy, long prompt | `context-policy.phi-3.5-mini`, `gpu.context-policy.phi-3.5-mini`, `long-prompt.tinyllama`, `long-prompt.mistral-7b`, `gpu.long-prompt.tinyllama`, `gpu.long-prompt.mistral-7b` | `ContextPolicyProbe`, `LongPromptProbe`, shared `LedgerNotes`; the long-prompt smoke's baseline-jar comparison becomes `ingest`'s comparison of the recorded greedy texts |
| GPU residency | `gpu.residency.tinyllama` gains the cluster part | pipeline and tensor with the region on: the nodes answer, none left; equality with the local run recorded, as the smoke reports it |
| CPU kernels | `cpu.hot-path.` + the four sweep models | `CpuHotPathProbe`; the two-build kernel smoke becomes `ingest`'s comparison of the recorded replies; the vision case moves to the vision group |
| Quality | `quality.` + the four sweep models | `QualityProbe` with one `QualityRunMain` JVM per kernel setting (read when the JVM starts); the text is copied into the kit and a test holds it equal to the NLL test's; the kit tokenizes it as the test's id files were made |
| Throughput | the remaining sweep models and lengths, CPU and GPU | catalog entries on the existing probe |

Also: `ProbeRunner` passes every model as `model.<catalog id>` (test first); `JvmLauncher` makes class-path entries
absolute (children run in their own directories). Tests first throughout, except: `CpuHotPathProbe`'s self-test was
written after the class, and the residency-cluster and quality self-tests were written first but not run before their
code (a probe run was holding the build); all three passed on their first run. The consistency, grammar, tools,
context-policy and long-prompt self-tests failed first (compilation), then exposed the two defects above and the
facade's temperature, then passed. **`juno-kit`: 84 tests, 0 failures.**

**Real-model runs on this host** (through `./juno-kit.sh --suite custom`):

| Probe | Result | Time |
|---|---|---|
| `consistency.tinyllama` | FAIL (the batch dependence above; architecture refusals, known-good load, native rounds, sessions, prefix hits, continuous all PASS) | 1:40 |
| `gpu.consistency.tinyllama` | PASS | 0:11 |
| `grammar.tinyllama` | PASS (every check, every JFR count) | 1:40 |
| `tools.qwen2.5-3b` | PASS | 3:20 |
| `gpu.context-policy.phi-3.5-mini` | PASS (both schedules: opt-in turns, decode-time shift, the limit error; `--context-shift on` default and explicit false) | 7:30 |
| `long-prompt.tinyllama` | PASS (both schedules, 128, 512, 2048) | 10:41 |
| `gpu.long-prompt.tinyllama` | PASS | 0:44 |
| `gpu.residency.tinyllama` (with the cluster part) | PASS: both clusters answer with the region on and leave no node JVM; their output differs from the local run's (recorded, not required, as in the smoke) | 0:47 |
| `cpu.hot-path.tinyllama-1.1b` | PASS at 4 and 12 threads, replies identical | 1:42 |
| `quality.tinyllama-1.1b` | PASS: NLL 2.45609 integer, 2.45566 float, ratio 1.00017, argmax agreement 99.0%: identical to the published reading (`docs/perf-compare/20261027T025856Z-cpu-kernels-quality/`), so the kit's tokenization reproduces the test's ids | 7:48 |

Not yet run on real models (owed in part 2, with reference costs set from the measured times): `cpu.unit-tests`, the
per-family live probes, `context-policy.phi-3.5-mini` (CPU), the Mistral long-prompt probes, the other three hot-path
and quality probes, and the new throughput entries. The run zips are in `juno-kit-results/` (git-ignored).

**Exit criteria after step 5 part 1:** none ticked.

**Next: step 5 part 2** (`docs/infra-plan/PROMPT-kit-step5-rest.md`): schedules, LoRA, vision, launcher; the
real-model runs above; then step 6.

### 2026-10-28: CPU batch dependence investigated (owner decision: investigate before step 5 part 2)

Readings: `docs/perf-compare/20261029T045221Z-cpu-batch-consistency/` (two single-file programs against the kit jar, in one
JVM, through the server's `GenerationLoop` and handler; not pinned, nothing is a throughput figure).

- **Not a KV leak.** Of the 9 ordered pairs of the probe's prompts run as sequential static batches, only [colour,
  spider] and [spider, colour] change the spider answer, at token 4 ("six" to "eight"); 18 rows of batches run
  concurrently, as the scheduler runs them, all match. The colour answer ends after 3 tokens, so from decode step 4 the
  spider request is the batch's only active row, and a one-row batch goes through `forward`, the single-request path.
  Before that its decode steps ran through `forwardMultiDecode`, whose matmul is `CpuIntegerGemm`; the single path's is
  `CpuIntegerKernels`. A single request whose first 1 to 4 decode steps are sent through the batched path answers
  "eight"; all steps or none, "six". A batched row does not depend on the other rows (`CpuIntegerGemmTest` pins that
  bit for bit).
- **Why the kernels disagree.** They agree to about 1e-5 of the largest product per matmul, not bit for bit. Every
  layer re-quantizes its input to 8 bits, so a last-bit difference sometimes crosses a rounding step: at the logits the
  two differ by about 0.03 to 0.05, and " six" leads " eight" by 0.009 on the single path. With the float-activation
  kernels (`-Djuno.cpu.integerKernels=false`) the two paths differ by about 1e-5 at the logits, and all 9 pairs match.
- **Consequences.** On the CPU's default kernels, a greedy answer at a near tie depends on how many requests are active
  at each of its decode steps, so on what else the server serves (static schedule; the continuous schedule's decode
  batch also changes size, not measured). The consistency check as written (batched equals single) therefore fails a
  correct engine whenever such a tie meets a kernel switch, so it is not a sound leak detector on the CPU; the shell
  smoke has the same exposure, and its pass at this commit was a pairing that did not switch kernels at the tie.
  `TinyLlamaStaticBatchLiveTest` (`coordinator`) asserts the same property but runs without the Vector API module, so
  on the float-activation kernels only.
- **Remedy: owner decision pending.** (a) In the kit: the CPU static-batch and continuous checks run the server with
  the float-activation kernels, where equality does detect a leak (a leak is kernel-independent); the default-kernel run
  is kept as a recorded, unscored reading. (b) In the engine: make the integer kernels batch-invariant (one row's
  result bit for bit the same in `CpuIntegerKernels` and `CpuIntegerGemm`, both lane widths): hot-path kernel work with
  the performance gates and a pinned owner run, a task of its own. (c) In the engine, cheaper: compute each row of a
  decode batch with the single-row kernel; bit-identical, but every weight is read once per row per step, so CPU
  `--parallel` throughput falls toward one stream's. First recommended: (a) now, (b) later as its own item.
- **Recommendation revised the same day: (b), scoped to the decode batch.** Batched equals single is a property the
  engine states (`TinyLlamaStaticBatchLiveTest`, the consistency smoke), and the default integer kernels broke it
  without a test noticing, so it is a regression to fix, not a tolerance; (a) alone would only stop the check seeing it.
  Everything else in the batched decode layer is already the single path's code (RMS norm, RoPE, SwiGLU, residuals,
  the output projection; attention through `GqaMath.attendPooled`, which `GqaMathPooledTest` pins bit for bit to the
  serial form), and the continuous engine decodes through the same `forwardBatch`. Only the matmul differs.
- **Design, corrected the same day.** Computing each batch row with the single-row kernel's row function would be
  bit-identical, but CPU decode on this host is bound by instructions per weight, not by memory (Tier 02C: 4 to 5
  cycles per weight, 13% of host bandwidth), so repeating that work per row forfeits most of what batching buys: close
  to remedy (c). The design that keeps it: a new decode-batch kernel derived from the single-row kernel, per format
  (Q4_K, Q5_K, Q6_K, Q8_0) and lane width (128, 256), that loads and unpacks each weight sub-block once and then runs
  the single-row kernel's own integer and float sequence for each of four activation rows (zero rows padding the last
  group), each with its own accumulator. The integer sums are exact, so equal float sequences give equal bits; a unit
  test holds every row bit for bit against `CpuIntegerKernels`. It replaces `CpuIntegerGemm` in the five handlers'
  multi-decode paths only (Llama for the Llama family and Qwen2, Phi-2, Phi-3, Qwen3, Qwen3 MoE; vision decorates them,
  the continuous schedule and both cluster types reach them). The single-row kernel and the prefill gemm do not change,
  so the pinned single-stream CPU figures are untouched. Its throughput lies between per-row and today's gemm, and is
  the cost to measure.
- **Plan alignment.** Execution rule 9: recorded below under **Out-of-tier changes**, naming what it touched; a
  measurement boundary for CPU batched decode (static `--parallel`, continuous), not for single-stream tg and pp.
  Execution rule 3: the defect passed because the static-batch live test runs without the Vector API module, so the
  fix adds the [colour, spider] pairing to it and runs it on the default kernels, and the handlers' multi-decode tests
  (now within 1e-4) hold the CPU integer path bit for bit; Tier 00's "batched against single" premise is re-measured.
  Execution rule 7: a threshold for the new cost, proposed CPU aggregate t/s at `--parallel` 2, 4 and 8 **>= 0.90x**
  today's gemm on TinyLlama and Qwen2.5-3B; a miss is an owner decision (determinism against throughput). Execution
  rule 2: the GPU is out of scope, stated: its consistency check passes, and its single and batched paths are not
  examined here. `CLAUDE.md`: `compare-lora.sh` against the last baseline, since batching changes.
  Owner go-ahead given 2026-10-29; outcome in the next section.

### 2026-10-29: CPU decode batch made bit for bit the single-request decode (owner go-ahead; out-of-tier change)

Owner decision 2026-10-29: "Go ahead with this design". The throughput threshold was proposed with the design and not
answered separately; it is applied as proposed. Readings: `docs/perf-compare/20261029T182639Z-cpu-decode-batch/`
(unpinned).

- **What was built.**
  - `CpuIntegerDecodeBatch` (`node`, new class): the decode-batch matmul for Q4_K, Q5_K, Q6_K and Q8_0. Each
    activation row is quantized once, as the single-row kernel quantizes it. The worker pool splits the weight rows,
    and a task runs the single-row kernel's own row function (`CpuIntegerKernels`, or `CpuIntegerKernels256` at the
    256-bit width) against each activation row in turn, so a weight row is read from memory once and from cache after.
  - The five handlers' multi-decode paths call it wherever the single path takes the integer kernel
    (`CpuKernelSelection.integerMatVec`): `LlamaTransformerHandler` (Llama family and Qwen2),
    `Phi2TransformerHandler` (its batched output head too, which ran the gemm), `Phi3TransformerHandler`,
    `Qwen3TransformerHandler` and `Qwen3MoeTransformerHandler`. Prefill and device paths are unchanged.
  - Batched decode attention takes `GqaMath.attendPooledAsSerial` (the pooled form with the serial head) in the four
    handlers that used `attendPooled`. Phi-2's attention was already the same on both paths.
- **Deviation from the approved design.** The design above unpacks each weight sub-block once and shares it across
  four rows. What landed is the simpler form: no shared unpack; each row reruns the single-row function on a weight
  row held in cache.
  - Reason: most of the single-row kernel's work is per activation row (multiplies, widening, float scaling), so
    sharing the unpack was estimated to recover little: about 0.77x of the gemm at batch 8, in microbenchmark terms.
  - The end-to-end A/B, the measurement the threshold names, already reads the simple form above the gemm.
  - The shared-unpack kernel was not built (KISS). It remains an option if the pinned run misses.
  - **Owner decision 2026-10-29: keep the simpler form.** Recorded after the alternatives were laid out: the
    shared-unpack kernel; the single-request kernel taking the gemm's arithmetic, which needs a pinned re-gate of the
    single-request figures; a revert with a tolerant kit check; and an opt-in switch back to the gemm. The pinned gate
    met the threshold the same day.
- **Corrections to the investigation above.**
  - Attention. The investigation said the batched decode's attention was already the single path's code, bit for
    bit. That holds only without the Vector API module. With the module, `attendPooled` runs `GqaVectorHead`, while
    the single path runs the serial `GqaMath.attend`. The new live test found it (logits still differed after the
    matmul change); the serial-head form fixes it.
  - Phi-2. Its batched output head also ran the gemm (a residual difference of 3.3e-6 after the matmul change). It
    now takes the decode-batch matmul.
- **Tests, written first; each failed for the right reason, then passed.**
  - `CpuIntegerDecodeBatchTest`, 31 cases: every row bit for bit against the single-row kernel, per format, at batch
    sizes 1 to 9 and both lane widths; also row ranges, scratch reuse and concurrent callers.
  - The handlers' multi-decode tests. Llama on random Q4_K, Q5_K, Q6_K and Q8_0 weights: logits bit for bit. Phi-2,
    Phi-3, Qwen3 and Qwen3-MoE: call counters show the decode-batch matmul ran and the gemm did not.
  - `CpuDecodeBatchLiveTest`, on real weights: TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Qwen3-1.7B, and moondream2's
    Phi-2 backbone, three requests over three steps. Before the change, every case failed at its first step (logit
    differences up to 0.12 to 0.24); after it, every case is bit for bit equal. The Qwen3-Coder-30B-A3B case is
    opt-in and was not run.
  - `TinyLlamaStaticBatchLiveTest` (`coordinator`) gains the [colour, spider] round and a surefire execution with the
    Vector API module (`static-batch-integer-kernels`). That is why the defect passed before: the test ran on the
    float-activation kernels only. Against the pre-change `node` jar the new round got " eight" (9475) where the
    single request gives " six" (4832); with the change it gets the single reply.
  - Full unit suite: 0 failures, every module of `CLAUDE.md`'s list (`node` 1242 run, 48 skipped; `coordinator` 398
    plus 1 in the new execution).
- **Throughput, unpinned A/B** (`06d6d22d` against this tree, A B / B A, three repetitions, median, two warmup rounds
  per server). CPU `--parallel` aggregate t/s B/A:
  - TinyLlama: 1.017 at P=2, 1.247 at P=4, 1.114 at P=8.
  - Qwen2.5-3B: 1.064 at P=2, 1.250 at P=4, 1.134 at P=8.
  - P=1, which both builds run on the single path: 0.985 and 1.006.

  The threshold (**>= 0.90x**) holds in this reading. It is a 0.90x no-regression gate, so the plan scores it only
  pinned. That run is the owner's: `dist/cpu-decode-batch-gate/`, the same A/B with clocks pinned by hand.
- **Pinned gate, owner run 2026-10-29: met.** 20:30 to 22:37 UTC, governor `performance` and turbo off before all 48
  runs, A B A B A B. Readings: `docs/perf-compare/20261029T203047Z-cpu-decode-batch-gate/`. CPU `--parallel`
  aggregate t/s B/A, median of three, unrounded against `>= 0.90`:
  - TinyLlama: 1.029 at P=2, 1.148 at P=4, 1.214 at P=8.
  - Qwen2.5-3B: 1.009 at P=2, 1.212 at P=4, 1.039 at P=8.
  - P=1 control: 0.981 and 1.004.
  - Batched replies differing from the single reply: baseline 23 of 84, candidate 0 of 84.
  - TinyLlama's readings spread by up to 24% within a side even pinned; Qwen2.5-3B's within 9%.
- **Replies.** Each server's batched replies against its single reply, on the A/B's greedy prompt: the baseline
  differs in 27 of 84 batched replies (half at P=4, a third at P=8, none at P=2). The candidate differs in 0 of 84.
  *(Clarified 2026-11-03: 27 of 84 is the unpinned A/B, `20261029T182639Z-cpu-decode-batch/`; the pinned gate's count
  is the 23 of 84 in the bullet above, `20261029T203047Z-cpu-decode-batch-gate/`. `CHANGELOG.md` cites the pinned one.)*
- **Finding for the owner, not acted on.** CPU batching buys no aggregate throughput on this host, in either build.
  The candidate at P=8 reads 1.02x to 1.04x its own P=1; the baseline reads 0.90x to 0.92x, and 0.82x to 0.83x at
  P=4. Why the gemm loses end to end, though it wins in isolation, was not investigated. The comparison the
  `--parallel` advice in `docs/howto.md` rests on was not measured either. P sessions at `--parallel P` were compared
  with one session alone, but 8 clients at `--parallel 1` were not; there the scheduler decodes each request on its
  own thread, it does not queue them.
  - **Owner decision 2026-10-29: measure that comparison, then decide the docs.** See "CPU `--parallel` against
    concurrent single requests" below.
- **LoRA gate** (`CLAUDE.md`, batching changed; each tree's own `compare-lora.sh`, A B / B A x3, GPU and CPU lanes,
  unpinned, gates train time at most 1.25x and playback at least 0.80x). **All four met.**
  - GPU: train time 1.000, playback 1.037.
  - CPU: train time 1.009, playback 0.993.
  - Every run trains in 15 passes and recalls "My name is Juno". Neither training nor playback reaches the changed
    code.
- **Scripts.** `compare-parallel.sh` takes `--juno-jar` and `--warmup-rounds`, and now loads the Vector API module;
  before this, its CPU readings ran the float-activation kernels. New: `cpu-decode-batch-kernels.sh` with
  `CpuDecodeBatchMicrobench`. `compare-schedule.sh`, `compare-mixed-prefill.sh` and `compare-prefill-batch.sh` still
  start `java` without the module (carried to step 5 part 2).
- **Docs.** `docs/agent-arch.txt` (the class and the attention entry point), `docs/howto.md` (the `--parallel` row:
  CPU batch invariance, GPU caveat), `CLAUDE.md` (the CPU path and the new surefire execution).

**Out-of-tier changes** (rule 9).

- **Commit:** none yet; uncommitted, on `06d6d22d`.
- **Touched:** batching (the CPU decode batch's matmul and attention in five handlers), `GqaMath` (a new entry point),
  `compare-parallel.sh`, and `coordinator/pom.xml` (one surefire execution).
- **Measurement boundary:** yes for CPU batched decode, that is the CPU static schedule at `--parallel` 2 or more and
  the continuous schedule's decode batches; readings from before and after are not comparable. Not for single-stream
  CPU tg and pp: the single-row kernel and the prefill gemm are unchanged, and a one-row batch already took the single
  path. Not for the GPU: device paths are unchanged, and GPU-resident batched decode was not examined.
- **Invalidates:** `docs/perf-compare/20260901T155136Z-parallel/` (CPU `--parallel`, already stale, taken before the
  integer kernels), and the step 5 part 1 FAIL of `consistency.tinyllama`, to be re-run on a kit jar with the fix. No
  pinned reference moves.
- **Tier 00's batched-against-single premise:** on the CPU's default integer path, batched decode is now bit for bit
  the single decode. Before, it was about 3e-5 on the float-activation kernels and 0.03 to 0.05 at the logits on the
  integer kernels.

### 2026-10-30: CPU `--parallel` against concurrent single requests (owner decision: measure, then decide the docs)

Owner decisions 2026-10-29, after the alternatives were laid out (with how another serving engine handles batch
invariance: off by default, an opt-in mode that fixes every kernel's reduction order at a throughput cost):

1. keep the simpler decode-batch form (recorded in the section above);
2. measure 8 concurrent clients at `--parallel 1` against `--parallel 8` on the CPU, then decide what `docs/howto.md`
   says.

Readings: `docs/perf-compare/20261030T040305Z-cpu-parallel-concurrency/`. One jar, the pinned gate's candidate
(`1c181de29e52d249`); unpinned, the flag alternated, median of three, two warmup rounds; 2026-10-30, 03:06 to 04:02
UTC.

- **`--parallel 8` over `--parallel 1`, 8 concurrent clients.** Aggregate t/s 0.993 on TinyLlama (10.74 against
  10.81) and 1.003 on Qwen2.5-3B (3.94 against 3.93). Mean latency per session is equal within 1% (TinyLlama 47.5
  against 47.2 s; Qwen2.5-3B 129.7 against 130.0 s). On this host's CPU the setting changes neither throughput nor
  latency.
- **Concurrent requests share one stream's rate either way.** 8 unbatched clients read 10.81 t/s on TinyLlama; one
  client alone read 10.44 to 10.60 the day before. CPU decode here is bound by arithmetic, so batching, which saves
  only memory traffic, gains nothing.
- **Replies.** All 96, at both settings, equal the single-request reply.
- **Scope.** One host: a CPU without AVX2, decode limited by arithmetic. A CPU limited by memory bandwidth should gain
  from batching, which reads a weight row once per batch; the kit's throughput probes on volunteer hardware are where
  that would show. The GPU's last reading of this workload, 1.11x, is from 2026-09-01, before the current GPU decode
  paths.
- **Docs: owner decision pending.** `docs/howto.md`'s `--parallel` row says "recommend `8` for API servers" without
  naming a backend; on this host's CPU that advice neither helps nor hurts.
- **GPU, measured the same day** (`docs/perf-compare/20261030T062550Z-gpu-parallel-concurrency/`). Prompted by the
  kit's `gpu.parallel.tinyllama`, which read 0.74x to 0.92x and was noisy. Same method with `--gpu`, player jar
  `8fa44602112f19c4`, GTX 1080, 06:25 to 06:34 UTC:
  - **`--parallel 8` over `--parallel 1`:** aggregate t/s 0.901 on TinyLlama (100.91 against 111.96) and 0.866 on
    Qwen2.5-3B (42.46 against 49.01). Mean latency 0.862 and 0.926.
  - **Cause: the 8 requests rarely make one batch.** `RequestScheduler.collectBatch` waits up to `--batch-window-ms`
    for a first request, then takes only what is already queued; it does not hold the batch open. In five of six runs,
    2 or 3 requests formed a second, small batch that finished after the first. The one run with all 8 in one batch
    read 1.04x. `docs/howto.md` and `--help` call the flag a "batch collect window", which the collector does not do.
  - **Replies.** Batches of 2 and 3 give a reply that differs from the single-request reply, also on a fresh server,
    so no state leaks between batches. Batches of 5 to 8 equal it. GPU batching is batch-variant, as is usual for GPU
    serving; the CPU is not.
  - So on this host neither backend supports "recommend `8`". The GPU gain when the batch fills is small (1.04x), and
    the collector's behavior costs 10% to 13% in practice. Owner decisions pending: the docs wording, and whether to
    make the collector hold the batch open for the window.

**Exit criteria:** none ticked (this was out-of-tier work).

### 2026-10-30: step 5, part 2 (schedules, LoRA, vision, launcher; the real-model runs)

Handoff used: `docs/infra-plan/PROMPT-kit-step5-rest.md`.

**Plan against code.** As part 1 left it, plus the out-of-tier CPU decode-batch change above. The schedule scripts'
ports start the server through the launcher with the kit's runtime flags, Vector API module included, so their CPU
readings take the integer kernels. `compare-schedule.sh`, `compare-mixed-prefill.sh` and `compare-prefill-batch.sh`
start `java` without that module and read the float-activation kernels.

**Probes added** (self-tests in `ProbeSelfTests`):

| Group | Probes | Notes |
|---|---|---|
| Schedules | `parallel.tinyllama`, `gpu.parallel.tinyllama` (readings) | `ParallelProbe`: `--parallel 1` and `8`, 8 concurrent blocking chats of 64 tokens, two untimed warmup rounds, three alternating repetitions. The ratio is host data, not a gain gate |
| | `schedule.tinyllama`, `gpu.schedule.tinyllama` | `ScheduleProbe`: static and continuous at `--parallel 8`. Readings: blocking t/s; mean TTFT and TPOT of 8 SSE clients; first against later turn latency of an 8-turn `x_juno_session_id` session, with prefix hits from the health endpoint. Checks: each server runs its schedule, and continuous shares decode steps (JFR) |
| | `mixed-prefill.tinyllama`, `gpu.mixed-prefill.tinyllama` | `MixedPrefillProbe`: `-Djuno.continuous.mixedPrefill` true and false; 1 long and 3 short SSE together; short TTFT and TPOT, long TTFT. Check: in mixed mode the long prompt is prefilled in chunks inside decode steps (JFR) |
| | `prefill-batch.tinyllama`, `gpu.prefill-batch.tinyllama` (readings) | `PrefillBatchProbe`: `--prefill-batch 1` and `32`, prefill t/s off the server's JFR summary, one fresh server per value and repetition |
| LoRA | `lora.tinyllama`, `gpu.lora.tinyllama` | `LoraProbe`: trains through `LoraTrainer` to the REPL's loss target, plays back with `--lora-play`, merges with `LoraMerge` and serves the merged GGUF. Checks: the trained reply both ways. Readings: time per pass, playback t/s both ways |
| Vision | `vision.moondream2`, `gpu.vision.moondream2` | `VisionProbe`: `POST /v1/vision/chat` with the benchmark image, the script's prompt and the smoke's one-sentence prompt. CPU runs at two thread counts and requires identical captions |
| Launcher | `launcher.tinyllama` (quick suite) | `LauncherProbe`: `juno` (or `juno.bat` through `cmd.exe`) local mode with `--api-port` answers, then exits with code 0 on `exit`, no process left; `juno lora --grammar-file X --help` prints the no-op warning |

Helpers: `ConcurrentChats` and `StreamTiming`. `JunoProcess` gained `serve` with JVM properties, `serveCommand`
(a launcher script), `postImage` (multipart), `metrics()` (the launcher's `--jfr` summary), `tree()` and `stop()`
(the exit code). Tests first, except:
- The LoRA self-test was written before the probe but not run in between.
- The vision self-test runs on a text-only synthetic model, the only synthetic model there is, and requires every
  caption check to fail with HTTP 404.
- The launcher self-test skips, saying why, without the player's shaded jar. It ran in every suite after a build.

`VisionProbeTest` and `ProbeCatalogTest` failed first (compilation; then three catalog assertions) and then passed.

**Real-model runs on this host** (`./juno-kit.sh --suite custom`; zips in `juno-kit-results/`, git-ignored):

| Run | Probes | Result | Time |
|---|---|---|---|
| 1, `20261030T045842Z` | the 13 new entries, both consistency probes | all PASS or DONE; `gpu.parallel.tinyllama` NOISY (`--parallel 8` spread 17% to 61%); `consistency.tinyllama` PASS on the fixed engine | 1 h 16 min |
| 2, `20261030T063627Z` | everything else owed from part 1, `consistency.tinyllama` again, `gpu.vision.moondream2` | 34 of 40 PASS or DONE; failures below | 9 h 42 min |
| 3, `20261030T164322Z` | the four Phi-3.5-mini failures, fixed | `gpu.live.phi-3.5-mini` PASS; both 2048-token throughput entries complete, NOISY (desktop applications used 7% to 14% of the CPU); `context-policy.phi-3.5-mini` ERROR (request limit) | 5 h 10 min |
| 4, `20261030T220016Z` | `context-policy.phi-3.5-mini` | PASS, all 7 checks; the crossing turn is 7 on both schedules, as on the GPU. Phases: static 1 h 46 min, continuous 3 h 36 min, static with shifting on 17 min | 5 h 39 min |

Passing on real models in run 2:
- `cpu.unit-tests`: 2098 tests, 0 failures, 25 skipped.
- The four `cpu.live.*`, and `gpu.live.*` on Qwen2.5-3B, Mistral 7B and Qwen3-1.7B.
- The Mistral long-prompt probes: both schedules, 128 to 2048 tokens, streamed text equal to non-streamed.
- The three hot-path probes.
- The three quality probes, whose NLL is identical to the published reading
  (`docs/perf-compare/20261027T025856Z-cpu-kernels-quality/`): 1.69656, 1.36110 and 1.53013 on the integer
  kernels.
- Every throughput entry except Phi-3.5-mini at 2048.

The kit's LoRA readings match `compare-lora.sh`: CPU 48.9 s per pass and 0.90 t/s playback, GPU 2.8 s and 12.3 t/s.

**Findings and fixes.**
- **Vision readings (kit defect, fixed after run 1, test first).** "decode t/s" was completion tokens over the
  server's decode total. With single-token prefill every prompt token runs through that same pass, so the figure read
  0.19 t/s where the pass rate is 5.45. It is now the mean single-token pass rate. "prompt t/s" is dropped: the
  prompt runs through the same pass, and the two prefill events the summary counts are not the prompt.
- **Vision on the GPU measures the CPU.** moondream2's phi2 has no device path, and the launcher warns about it;
  `gpu.vision` read 118 MiB of device memory and the CPU's latency. A GPU run whose matmuls all ran on the host now
  gives no readings and records `gpu.deviceMatVecs`. Run 2 confirmed this: 0 device matmuls, no readings, checks
  PASS.
- **GPU `--parallel 8` below `--parallel 1`** (out of tier; see the GPU bullet of the 2026-10-30 section above and
  `docs/perf-compare/20261030T062550Z-gpu-parallel-concurrency/`). The static collector does not hold the batch open
  for `--batch-window-ms`.
- **Phi-3.5-mini's long prompts outgrow the kit's heaps (kit sizing, fixed).** `gpu.live.phi-3.5-mini` check 9
  (8 GiB) and both `*.throughput.phi-3.5-mini.2048` entries (6 GiB) failed with `OutOfMemoryError` in
  `DenseKvTensor.grow`. The `f16` dense KV is held as `float[]`, and 2048 + 64 positions double the capacity to 4096.
  Phi-3.5-mini has no grouped-query attention and K and V are 3072 wide, so that is about 3 GB, beside 2.3 GB of
  weights. The live set after GC, measured at 12 GiB, was 7.5 GB on the GPU and 7.2 GB on the CPU, so these three
  entries now get 12 GiB. The CPU live probe passes at 8 GiB, because its check 9 skips the 2048-token leg on the CPU.
- **Timeouts.** CPU `context-policy.phi-3.5-mini` hit its 2-hour limit in its continuous phase. It was slow, not hung:
  about 80k prompt tokens in all, at 6 to 10 t/s prefill on this CPU. Several CPU probes also ran above a third of
  their limit (quality on Phi-3.5-mini: 70 min against 120). Timeouts are fixed seconds, and this host is the
  reference, so a slower tester's host would time out on sound probes. New rule, held by `ProbeCatalogTest`: every
  timeout is at least three times its reference cost.
- **Request limit.** In run 3, one request of the context-policy probe took over `JunoProcess`'s 20-minute HTTP
  limit: the continuous schedule's near-limit prompt of about 4,000 tokens with 200 decode tokens. The limit is now
  2 hours; the probe's own timeout bounds the run. Not unit-tested: a constant, checked by run 4, which passed.
- **The CPU context-policy probe is over a third of the `cpu` suite.** It took 5 h 39 min in run 4 (reference cost
  now 20,300 s, timeout 60,900 s); the `cpu` suite's other probes cost about 9 h together. The probe needs Phi-3.5-mini,
  the only family whose 4,096-position limit a conversation can reach, and its conversation of 10 turns of 25 notes
  prefills each turn's whole history again. Its continuous phase took twice its static phase for the same requests;
  the cause is not traced. Owner decision 2026-10-30: shorten its conversation (below).

**Catalog after the runs.**
- Reference costs are the measured wall times. The CPU context-policy probe is priority 3, so a budgeted run picks
  it last.
- Timeouts are at least 3x the cost: the quality and throughput helpers take `max(base, 3 * cost)`; long-prompt
  Mistral is 11,600 s and the unit tests 6,600 s.
- The Phi-3.5-mini entries above have 12 GiB heaps.
- Device-memory minimums come from each GPU probe's own measured peak (`MEASURED_DEVICE_PEAK_MIB`), rounded up to
  512 MiB: 512 MiB (vision) to 7,680 MiB (Mistral 7B and Phi-3.5-mini live).

**`juno-kit`: 98 tests, 0 failures.**

**Exit criteria after step 5:** none ticked. The coverage test, which the probe-catalog criterion names, is step 6.

**Next: step 6**: the coverage file and its test; `ingest`; the reference run on this host, published.

### 2026-10-31: owner decisions of 2026-10-30 carried out; step 6

Owner decisions 2026-10-30, after step 5's report: "1. yes 2. yes after that start step 6": shorten the CPU
context-policy probe; make the static collector hold a batch open for `--batch-window-ms` (out of tier), measure the
GPU again, then word the `--parallel` docs from the reading.

**Context-policy probe, shorter conversation.** Both entries pass `turns=3`, `notesPerTurn=85`: the conversation
crosses Phi-3.5-mini's 4,096 positions at turn 2 and ends at about 6,700 prompt tokens, the longest prompt of ten turns
of 25 notes; every check stays. `ProbeCatalogTest.contextPolicyEntriesGrowTheConversationInThreeLargeTurns` failed
first. Measured in kit run 5 (20261031T052330Z, this branch's uncommitted tree, calibration 1.02x):
`context-policy.phi-3.5-mini` PASS in 3:07:34, against run 4's 5:38:40 (0.55x), with all seven checks, the
held-open request reaching 4,239 positions against the 4,096 limit; `gpu.context-policy.phi-3.5-mini` PASS in 4:19,
GPU 5,036 MiB. Catalog costs set from these: 11,250 s (timeout 33,800 s, 3x) and 260 s. `--plan` on this host at
these costs: `quick` 24 to 34 min, `gpu` 57 to 81 min, `cpu` 629 to 888 min, `all` 686 to 968 min. The same run checked the
collector change and native streaming on real weights: `consistency.tinyllama` PASS in 1:42 and
`gpu.consistency.tinyllama` PASS in 0:13, every batched reply equal to its single-request reply at the probe's 8
tokens on both backends (its `--parallel 2` batches now fill), and the native stream joined to the blocking answer.

**Static collector (out of tier).** `RequestScheduler.collectBatch` took the requests already queued when the first
arrived and dispatched at once; it now polls for the time left until `batchWindowMs` after the first request, or until
the batch is full, the contract `BatchConfig`'s javadoc states. `RequestSchedulerBatchTest`:
`a_request_arriving_inside_the_window_joins_the_first_requests_batch` failed first (no batch of 2 formed);
`a_full_batch_dispatches_without_waiting_out_the_window` is a regression test and passed on its first run. Coordinator
unit tests: 400, 0 failures. `TinyLlamaStaticBatchLiveTest` fails under `mvn test -pl coordinator` alone, because
`~/.m2` holds a `node` jar from 2026-10-28, before the decode-batch fix; with `-am` it passes on both executions.
LoRA gate not run: the collector runs only at `--parallel` above 1, which `compare-lora.sh` never sets, and the change
touches no MatVec, forward, KV or vision path, the roadmap's optional case.

- **GPU** (`docs/perf-compare/20261031T040116Z-gpu-parallel-window/`, unpinned, median of three): `--parallel 8`
  over `--parallel 1` is 0.748x (TinyLlama) and 0.676x (Qwen2.5-3B), against 0.901x and 0.866x before; mean latency
  1.38x and 1.50x. Every P = 8 run is now one batch of 8. Batched GPU decode (`forwardMultiDecode`) runs each layer's
  matmuls from the host, while a single request's decode uses the device-resident region (Tier 01's record: batched
  decode "not wired"), so 8 unbatched concurrent requests outrun one batch of 8 on this card.
- **CPU** (`docs/perf-compare/20261031T041246Z-cpu-parallel-window/`, same jar and method): 1.174x (TinyLlama) and
  1.084x (Qwen2.5-3B), against 0.993x and 1.003x before; mean latency 0.853x and 0.924x. Every P = 8 run is one batch
  of 8, and all 96 replies equal the single-request reply.
- **Docs**, from these readings: `docs/howto.md` and the launcher's `--help` no longer recommend `--parallel 8` for every
  API server: `8` on the CPU, `1` on the GPU; `--batch-window-ms` now says what the collector does.
- **Owner decision pending**: keep the collector change (CPU gains, GPU loses) or revert it; and whether to plan a
  device-resident batched GPU decode, without which static batching cannot gain on the GPU.
  **Owner decision 2026-10-31** on the second: plan it as its own entry after Tier 02G, not started now.
  [Tier 02H](TIER-02H-batched-gpu-decode.md) created, with the README's running order and index.
  **Owner decision 2026-10-31** on the first, after the details (who it affects, before and after, the
  consequences, alternatives: revert, a GPU startup warning, idle-aware dispatch, a smaller default window):
  option B, keep the collector change as it is, with no warning and the default window of 50 ms.
- **Native SSE escaping (out of tier, found by the kit).** The new native-streaming check failed on its first run in
  the synthetic-model self-test: `SseTokenConsumer` passed control characters other than newline, carriage return and
  tab raw into the event's JSON string, which JSON forbids; a byte-fallback token decodes to one, so a real model can
  emit it too. It now writes them as unicode escapes. `SseTokenConsumerTest.control_characters_are_escaped_as_unicode`
  failed first. The OpenAI stream serializes through Jackson and was not affected. Coordinator: 401 unit tests, 0
  failures, 1 skipped (as before); `TinyLlamaStaticBatchLiveTest` passes.

**Step 6.**
- **Coverage file** (`juno-kit/src/main/resources/cab/ml/juno/kit/coverage.tsv`, `Coverage`, `CoverageTest`): every
  cross-surface row, supported `general.architecture`, smoke and compare script and feature, with its probes. The test
  fails when an item has no probe or names a probe, script, architecture or row that does not exist, and when a
  catalog probe, a smoke or compare script, a supported architecture or a cross-surface row has no line. The
  architectures come from `ForwardPassHandlerLoader.supportedArchitectures()` (new, with
  `LlamaFamilyArchitectures.verified()`), so a new handler fails the test until it has a line. Five detection tests
  failed first against a stub; the file passes. What the file records honestly:
  - `qwen3moe` is covered by its handler's synthetic-model unit tests only: the one real file is 18.6 GB.
  - No kit model declares `mistral`, `tinyllama` or `qwen2.5`; those lines name the probes of the models that load
    through the same handler under `llama` and `qwen2`.
  - Native streaming (`/v1/inference/stream`) was not probed, though this tier's checklist said row 13 was covered
    like the OpenAI surface. `ConsistencyProbe` now joins a native stream and compares it with the native blocking
    answer (self-test expectation added first).
- **`ingest`**: `juno-kit.sh ingest <zip> [--reference <dir or zip>] [--reference-run] [--out <dir>]`. It reads the
  zip (an entry that would land outside the extraction folder is refused; a manifest of a newer schema is refused
  with an update message; unknown fields are ignored), and drafts `docs/perf-compare/<UTC>-remote-<host label>/`
  with `INDEX.md` (host, build, probes, readings with median, min, max and noise status, correctness, coverage on
  that host), `README-row.md`, and the zip's manifest, checkpoint and probe reports. Correctness is compared with the
  newest `*-kit-reference` entry at the same Juno commit: mean NLL at most 1.01x the reference's (the CPU-kernel
  quality reading's bound), argmax agreement at most 0.02 lower, checks and greedy replies equal. Throughput is never
  compared with the reference host. `--reference-run` drafts the owner's own run as `<UTC>-kit-reference/`.
  `IngestTest` (8) and two `KitOptionsTest` cases were written before the code but not run in between, so they were
  not seen failing. Run end to end on this host's own zips: run 2's zip drafted with `--reference-run`, then run 3's
  compared with it (10 items; the one difference listed is `gpu.live.phi-3.5-mini` check 9, which run 2 failed out of
  memory and run 3 passed).
- **Reference run**: owed to the owner. It must be at a commit, since a tester builds a commit and the tree here is
  uncommitted. `dist/kit-reference/run-reference.sh` refuses modified tracked files, runs `--suite all`, then
  `ingest --reference-run`.
- **Gap**: no probe records a logit checksum, which scope item 11 lists among `ingest`'s comparisons; it compares NLL,
  argmax agreement, checks and replies.

**Exit criteria after step 6:** the probe-catalog criterion is ticked (above). The `ingest` criterion waits for
the reference run to be published.

**Next:** step 7: README rule 10, `check-plan-thresholds.sh` check 10 seen failing first, the **Kit probes**
sections, and the "Borrowed hardware" table in `INVENTORY.md`. The owner's reference run
(`dist/kit-reference/run-reference.sh`) follows steps 7 and 8, at the commit testers build for step 9: `ingest`
matches a tester's zip to the reference at the same Juno commit, and steps 7 and 8 each make a new commit.

### 2026-10-31: step 7 (plan wiring)

Owner decision 2026-10-31 on the collector, after the details: "Option: B", keep it as it is (recorded in the step
6 section above). With no decision open and the reference run due after steps 7 and 8, step 7 followed.

**Plan against code, verified.** `check-plan-thresholds.sh` passed first (26 tier files). Step 6's code is as recorded:
juno-kit 116 tests and juno-player 127 (0 failures, 2 skipped) passed on this build. Drift found and corrected: the
Status line still said "No exit criterion is ticked yet" after step 6 ticked the probe-catalog criterion.

- **Check 10** (`scripts/performance-tests/check-plan-thresholds.sh`): an open tier file needs a `## Kit probes`
  heading with at least one list item before the next heading. Written first, and run on the tree before any
  section existed: 17 failures, one per open tier file (02F, 02G, 02H, 03, 04, 04B, 04C, 05, 06, 07, 08, 09, 10, 11,
  12, 13, 14), none for a complete tier. With the sections written it passes. A copy of the plan whose Tier 11
  section keeps its heading but loses its list fails with one failure. The script's header lists check 10, and
  `--help` prints the whole header (it cut off the last usage line before).
- **README rule 10**, and check 10 in rule 7's list. Every open tier carries **Keep passing** (catalog probes
  covering what it changes) and **Add** (a probe or check per feature, flag or threshold it introduces, each with a
  self-test and a `coverage.tsv` line). A probe whose expectation the tier changes is listed as a change. The check
  sees only that the section exists and lists something; whether the probes named are the right ones is each
  tier's review, and catalog-against-coverage agreement is `CoverageTest`'s.
- **The sections.** One per open tier file, before "Models needed". The 14 tier files without one gain an exit
  criterion: the **Add** list is in the catalog with self-tests and coverage lines, and the **Keep passing** list
  passes on the tier's build. Tier 02G and 02H already had one; this tier's own is the exit criterion above.
  Two existing checks will have to change, and are listed as changes:
  - Tier 03: cross-session prefix sharing makes the consistency probes' "no prefix-cache hit for stateless
    traffic" wrong by design.
  - Tier 08: each newly supported architecture leaves `consistency.tinyllama`'s refusal list, and needs its
    `family:` line, which `CoverageTest` already enforces through `ForwardPassHandlerLoader`.
- **`INVENTORY.md`, "Borrowed hardware"**: host label, OS, CPU, GPU and backend, ingested runs, rows it scores;
  recorded by hardware, never by person. Empty until the first tester's zip is ingested.

**Tests and checks.** `check-plan-thresholds.sh`: 17 failures before the sections, ok after. Plan files only, and a
bash check script: no Java changed, so no unit test was rerun for this step.

**Exit criteria after step 7:** rule 10 and check 10 ticked (above); two of the tier's criteria are now ticked.

**Next:** step 8, the CI workflow (approved, decision 8): written here; the owner commits it and its first green run
is recorded. Then the owner's reference run at that commit (`dist/kit-reference/run-reference.sh`), before
testers' runs (step 9).

### 2026-10-31: step 8 (CI workflow written; the owner commits it)

Owner request 2026-10-31: "start step 8".

**Plan against code, verified.** `check-plan-thresholds.sh` passed first (26 tier files). Scope item 14 and decision
8 match the README's 2026-10-13 paragraph. The kit's code is as step 6 left it. One finding for this step: the kit
exits 0 whatever its probes found. Its own runs and the testers' rely on that ("records failures and continues"),
so the workflow reads the summary instead of changing the exit code. No drift in the plan text.

**The workflow: `.github/workflows/kit.yml`**, a file of its own. `ci.yml` is untouched and stays untracked under the
owner's 2026-10-01 decision. Workflow permissions are read-only (`contents: read`).
- **`quick-suite`**, on manual dispatch and weekly (Mondays 04:23 UTC). It runs on four runners:
  - `ubuntu-24.04`: Linux x64.
  - `ubuntu-24.04-arm`: Linux arm64.
  - `windows-2025`: Windows x64.
  - `macos-15`: macOS on Apple silicon.

  The labels are fixed rather than `-latest`, so an image change shows as a change to this file. Each runner runs
  the tester's command:
  - `./juno-kit.sh --suite quick --yes` on Linux and macOS, which also tests that the launcher is committed
    executable;
  - `juno-kit.bat --suite quick --yes` under `cmd` on Windows.

  Temurin 25 comes from `actions/setup-java` with its Maven cache. Maven is on all four runner images: the arm64
  image's software list names it.

  Each runner's `juno-kit-results/*.zip` is uploaded as artifact `juno-kit-results-<runner>`, even when an earlier
  step failed. The upload fails when there is no zip. A last step reads the run's `SUMMARY.txt` into the job summary
  and fails the job in any of three cases:
  - a probe line reads FAIL, ERROR, TIMEOUT or NOT RUN;
  - the file is missing;
  - no line has the summary's status format. A future format change then makes the job fail rather than pass
    without checking anything.

  The job timeout is 300 minutes.
- **`self-tests`**, on pull requests that touch the kit. "Touch" covers `juno-kit/**`, both launchers, the
  workflow, and two inputs of `CoverageTest`: `scripts/performance-tests/**` and the two node files that define
  the supported architectures. The job runs
  `mvn -B -ntp package -pl juno-kit -am -Dtest='cab.ml.juno.kit.**' -Dsurefire.failIfNoSpecifiedTests=false`.
  - `package` rather than `test`, because the launcher self-test needs juno-player's shaded jar and skips without
    it.
  - The filter keeps the upstream modules' suites out of this job. Those suites are `ci.yml`'s.
- **What the hosted runners can run.** They have no GPU, so every GPU probe is listed as not run.
  - **Linux and Windows** (4 vCPU, 16 GB): every quick CPU probe fits.
  - **x64 runners**: the kernel-width probe runs where the JVM prefers 256-bit integer vectors. That is the
    first, noisy 256-bit reading decision 8 named.
  - **Linux arm64** (128-bit NEON): the kernel-width probe is skipped.
  - **macOS** (3 M1 cores, 7 GB): five probes fit: env, calibration, CPU throughput at 128 and 512, and the
    launcher check. Live checks, consistency, grammar, long prompts, hot path and quality need 8 GiB or more and
    are listed as not run. The tier's macOS run therefore still comes from a tester with more memory, or covers
    only those five probes.

  On these runners the noise rule's re-runs (at most three attempts per reading) bound the extra time. A reading
  under load is marked NOISY, which the last step does not count as a failure.

**Checks run here.** There is no workflow linter on this host (no actionlint, Go or Docker), so:
- **YAML structure:** the file parses (PyYAML). Its triggers, permissions, jobs and four-runner matrix are as
  above, and every step has exactly one of `uses` or `run`.
- **The last step,** extracted from the parsed file and run under `bash -e` against six summaries:
  - the real summary of run 5 (20261031T052330Z, every probe PASS): exit 0;
  - copies with one probe line changed to FAIL, ERROR and NOT RUN: exit 1, each naming the probe;
  - a missing file: exit 1;
  - a summary whose status column was rewritten: exit 1, "the summary format changed".
- **The self-test job's Maven command** on this tree: BUILD SUCCESS in 5:01 with a warm local Maven cache.
  - juno-kit: 116 tests, 0 failures, 0 errors, 0 skipped. The launcher self-test ran, because `package` built the
    shaded jar.
  - Every upstream module compiled and ran surefire with no tests selected.
- **The actions** are each one's current major: `actions/checkout@v7`, `actions/setup-java@v6`,
  `actions/upload-artifact@v7`, all on Node 24. The inputs used (`distribution`, `java-version`, `cache`, `name`,
  `path`, `if-no-files-found`) are in each one's `action.yml` at those tags.

No Java changed, and no test was written first. The workflow is configuration, and the one piece of logic in it
(the last step) was run against passing and failing inputs as listed.

**Not yet run on GitHub.** That happens after the owner's commit. GitHub runs a dispatched or scheduled workflow
only from the default branch. The order:
1. Commit the tree.
2. Open a pull request to `main` from the branch: its `self-tests` job is the first run.
3. Once on `main`, dispatch `quick-suite` (`gh workflow run kit.yml --ref main`).
4. Record both runs' URLs here.

A failing probe on a runner is a step 9 defect, fixed there.

**Reference run and commit matching.** `ingest` compares a zip with the reference run at the same commit. If `main`
gets a merge commit, its hash differs from the branch's, and the CI zips find no reference unless `ingest` is given
`--reference`. Running `dist/kit-reference/run-reference.sh` on `main`'s HEAD, the commit the CI runs, avoids that.

**Exit criteria after step 8:** unchanged (two ticked). The workflow is not an exit criterion of its own. Its runs
feed the criterion "completed at least once each on Windows, macOS and Linux".

**Next:** the owner commits the tree and starts the workflow, and its first runs are recorded. The reference run at
`main`'s HEAD follows. Then step 9: ingest the CI zips and fix any defect they show, then testers' runs.

### 2026-11-03: step 10 (docs and `CHANGELOG.md`), ahead of step 9; GitHub CI deferred (owner decision)

**Owner decision 2026-11-03: "for now skip github ci".** Step 8's runs (the `self-tests` job on a pull request, the
`quick-suite` dispatch) and the CI half of step 9 are deferred. `.github/workflows/kit.yml` stays as written and
uncommitted. Until CI is taken up again, the criterion "completed at least once each on Windows, macOS and Linux" can
close only on testers' runs. With no decision open and nothing else owed but owner runs, step 10 was taken before
step 9: step 9's testers need the guide it writes, and they should build a commit that contains it.

**Plan against code, verified at HEAD `a1fe7166`.** `check-plan-thresholds.sh` passed first (26 tier files). No gate,
sweep or build was running. Findings:
- The owner's commit `a1fe7166` (2026-10-31) holds the kit, the engine fixes and the plan wiring. It does not hold
  `.github/` (`ci.yml` and `kit.yml` are still untracked) or the `.gitignore` line for `juno-kit-results/`, which is
  still a working-tree modification. So `dist/kit-reference/run-reference.sh` refuses to start until a commit holds
  that line, and on this host the kit names its build `a1fe7166 (modified tree)`.
- The kit jars (2026-10-31 11:47) were newer than every `src/main` file, so this step's `--skip-build` runs read
  `a1fe7166`'s code.
- `--list`: 74 probes, 72 runnable here (`cpu.kernel-width`: 128-bit JVM; `rocm.unit-tests`: no AMD GPU). `--plan`:
  step 6's figures. `--plan` and `--list` left `git status` unchanged.
- The step 5 decode-batch record put the unpinned A/B's 27 of 84 under the pinned bullet; clarified in place.

**Docs written** (scope item 15):
- `juno-kit/README.md` (new): the tester's guide, one standalone page. It covers prerequisites, getting the commit
  (git or a commit archive), running, the suites with reference-host times, what the tester may do meanwhile,
  keep-awake, `resume` and `pack`, the zip and its contents, privacy, the six models, options, common problems and
  exit codes. The prerequisites include RAM per probe and GPU backends per OS: CUDA resolves only Linux library names
  and ROCm is Linux only, so the GPU probes are not runnable on Windows and macOS. Problems are quoted with the kit's
  own messages. Exit codes, checked in `KitBootstrap` and `KitMain`: 0 ran, 1 build failure or unexpected error,
  2 prerequisite, disk or option.
- `docs/howto.md`:
  - a section "Hardware validation kit (`juno-kit`)" before "Build and Test": commands, suites, probe mechanics,
    estimate, noise rule, models, zip, privacy, the maintainer side (`ingest`), coverage;
  - the kit's test command in "Build and Test", both shells. It is step 8's `package -pl juno-kit -am` form, because
    `test -pl juno-kit` alone takes the upstream modules from `~/.m2`, and the launcher self-test needs the player's
    shaded jar.
- `docs/agent-arch.txt`: a `juno-kit` module entry (classes by role), `ModelLiveChecks`' backend and `ClusterHarness`'s
  forwarding of `JUNO_USE_GPU` under `juno-master`, `HfGgufFetcher.downloadPinned`, and the dependency-graph line.
- `README.md`: "Hardware validation kit" under "What's new", and a module-table row.
- `CHANGELOG.md`, Session 128: the kit, and the changes that shipped with it in `a1fe7166` without an entry:
  - the CPU decode batch, with the pinned gate's figures;
  - the collector;
  - the grammar alternatives fix;
  - native SSE escaping;
  - `./juno cluster --cpu` keeping its forked nodes on the CPU (`ConsoleMain.main` sets `JUNO_USE_GPU` before
    `runClusterRepl`, and `ClusterHarness` now forwards it);
  - `ModelLiveChecks`' backend.
- A grep of the added text and the new file finds no competitor product name, tier number, emoji or pointer into this
  plan tree. `CLAUDE.md` is not in scope item 15; by owner decision (2026-11-03) its module table gains one
  `juno-kit` row.

**The `quick` suite, scored against exit criterion 3** (`docs/perf-compare/20261103T205950Z-kit-quick-estimate/`,
unpinned; kit jar `8ef5098cf3034632`):
- 18 probes PASS or DONE; calibration 1.03x.
- First estimate 1478 to 2086 s, point 1738 s. Actual 2345 s, so point over actual is **0.741x**: inside the band,
  above the range's top.
- Not ticked, for two reasons.
- **Three catalog costs matched no measurement:** `long-prompt.tinyllama` 240 s (measured 641 and 626),
  `quality.tinyllama-1.1b` 120 s (468 and 458) and `grammar.tinyllama` 300 s (100 and 97). Step 5's catalog note said
  the costs were the measured times; these three were not updated from part 1's runs. They are corrected in
  `ProbeCatalog` (630, 460, 100), not test first: they are data, and `ProbeCatalogTest`'s rule (timeout at least 3x
  cost) still holds. `juno-kit`: 116 tests, 0 failures, 0 errors (step 8's command, rebuilt jar `df00680b5bbc0f29`).
  `--plan` now reads `quick` 32 to 45 min, `cpu` 636 to 898, `gpu` 57 to 81, `all` 693 to 979. The docs and
  `run-reference.sh`'s comment carry these figures. With them, this run's point would have been about 0.97x.
- **`cpu.live.tinyllama` took 451 s against 243 s on 2026-10-28.** Two windows are about 5.5x slower in this probe
  only: check 9's 573-token batched window (16.5 s, now 92.7 s) and check 10's (5.1 s, now 28.6 s). Per layer the
  matmuls are about 6x slower and attention is unchanged.
  - Re-run alone on an idle machine (other processes 3.9%): 89.7 s and 28.2 s.
  - Outside the kit, from the same jar: 18.0 to 18.8 s. That holds for check 9 alone at 4 and 16 GiB, for checks 1
    to 9 in one JVM, for the probe's exact JVM flags, and with `nvidia-smi` polled every 2 s beside it.
  - The engine's CPU prefill is unchanged elsewhere the same day: the server (`long-prompt.tinyllama`, 512 tokens
    15.2 s static and 16.1 s continuous, against 15.3 and 16.2 on 2026-10-29) and the throughput probe (33.6 t/s at
    512).
  - So this is a kit-side finding, not an engine regression, and not a measurement boundary. Its cause inside the
    kit's runner is not found. Ruled out: heap, GC, JVM flags, the `hpet` clock source, GPU polling, the probe's
    entry class and other load. It stays open for step 9, which owns kit defects. The probe's checks are unaffected.

**Tests and checks.** `juno-kit` as above. `check-plan-thresholds.sh` passes after the edits. No engine code
changed. The full unit suite and `mvn verify -pl juno-master` are owed at the close, as before.

**Out-of-tier changes:** none. `ProbeCatalog` is this tier's code.

**Exit criteria after step 10:** docs ticked. Three of thirteen are ticked. Criterion 3 has a dated note.

**Owed to the owner, in order:**
1. Commit the tree: this step's docs and the catalog fix, the `.gitignore` line, and this record. `.github/` can stay
   out while CI is deferred.
2. The reference run at that commit, `dist/kit-reference/run-reference.sh`: about 11.5 to 16.5 h, the machine idle
   during the timed probes. Then publish its draft and add its `README-row.md` to `docs/perf-compare/README.md`.
3. Testers' runs at that commit (Windows, macOS, a 256-bit integer host), each zip ingested (step 9).

**Next for the executor:** find why the CPU live probe's wide windows run slow under the kit's runner, then
re-score criterion 3 on a fresh `quick` run. Then step 9 as zips arrive, and the close: the full unit suite,
`mvn verify -pl juno-master`, and the cross-surface table.
