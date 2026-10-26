# Tier 02F: Hardware validation kit (one command on Windows, Linux or macOS; results returned as one zip)

Status: not started (added 2026-10-13 by owner decision; runs after Tier 02E closes and before Tier 02G)
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
   root, `juno-kit` (sh) and `juno-kit.bat` (matching `juno` and `juno.bat`), start `KitBootstrap`, a
   source file in the module that uses only JDK classes (a unit test holds it to that), so the JDK can run
   it directly before anything is built.
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
- **End to end on this host**: `juno-kit --suite quick --yes` completes, and its zip passes `ingest`.

**No perf gate**: this tier adds a tool and changes no inference path (forward pass, MatVec, KV, batching,
quantization). The kit's own acceptance numbers are in the exit criteria.

## Models needed

All present here (`INVENTORY.md`): TinyLlama 1.1B Q4_K_M, Qwen2.5-3B, Phi-3.5-mini and Mistral 7B (the
sweep models), Qwen3-1.7B, moondream2. Testers download them through the catalog (step 2). The hardware
this tier exists for is what this host lacks: a 256-bit integer CPU, an AMD GPU, Windows with an NVIDIA
GPU, macOS.

## Exit criteria

- [ ] `juno-kit` is in the reactor with the `juno-kit` and `juno-kit.bat` launchers; the bootstrap checks
      prerequisites before building, and a failed build still produces a results zip.
- [ ] Detection, the suites, `--budget`, `--plan`, `--yes`, `--list`, `resume` and `pack` work as
      specified, each unit-tested.
- [ ] The estimate is shown before the run and refined during it. On this host, the first estimate for the
      `quick` suite is `>= 0.65x` and `<= 1.35x` the actual time.
- [ ] Stage warnings, keep-awake on each OS, and the noise rule's re-runs work as specified.
- [ ] The model catalog is pinned by SHA-256, downloads resume, and hashes are verified.
- [ ] The probe catalog covers every row of the README's smoke matrix and the Juno-side checks and readings
      of every smoke and compare script. Every probe has a self-test in `mvn test`, and the coverage test
      passes.
- [ ] Results are scrubbed (fixture tests), the zip holds `SUMMARY.txt` and the HTML report, and nothing in
      the kit uploads.
- [ ] `ingest` drafts the `docs/perf-compare/` entry and compares against the owner's reference run; the
      reference run on this host is published.
- [ ] The kit has completed at least once each on Windows, macOS and Linux (CI runner or tester), each zip
      ingested.
- [ ] Rule 10 and check 10 are in place (check 10 seen failing first); every open tier file has its **Kit
      probes** section; `INVENTORY.md` has the "Borrowed hardware" table.
- [ ] Every module's unit tests pass, `juno-kit` included, and `mvn verify -pl juno-master` passes.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`juno-kit/README.md`, `docs/howto.md`, `docs/agent-arch.txt`, `README.md`) updated, in
      Juno-native language only; `CHANGELOG.md` entry added.

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
