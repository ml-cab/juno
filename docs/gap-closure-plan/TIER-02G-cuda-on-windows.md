# Tier 02G: CUDA on Windows

Status: postponed 2026-11-07 by owner decision until it is run on Windows; Tier 02H runs meanwhile (README execution
rule 1). In progress before that (added 2026-10-13 by owner decision; runs after Tier 02F). Begun 2026-11-05 while Tier 02F stays
open, by owner decision (README execution rule 1): Tier 02F's open criteria all wait on testers' runs. Step 4 (items 1, 2
and 6) done 2026-11-05 on Linux, tests first: `CudaLibraryLocator` and `GpuPlatform`, the binding audit, and the
fail-closed refusals. The kit probes this tier adds done 2026-11-06 on Linux, tests first: `gpu.kernel-timing` (slowest
single launch at a 2048-token window 37.37 ms here; attention reaches 596.51 ms at 8192 tokens, a finding raised for item
3), the CUDA libraries in the `env` reading, the launcher's `[DBG]` and Windows registry checks. Item 4's audit done 2026-11-07 on Linux, tests first: a local
cluster's nodes listen on loopback and stop through their shutdown hooks, `.gitattributes` checks text out with LF on every
OS, `juno merge` refuses to overwrite its input, `/train-file` reads CR LF as LF. Item 5 (launchers) done 2026-11-07 on
Linux by owner decision (b), `run.bat` mirrors `run.sh` flag by flag, tests first: a unit test reads both launchers and
holds them to the same subcommands, flags, forwarded options and jars; no `[DBG]` line, no registry write, help text under
labels of its own (`cmd.exe` ends a block at an unescaped `)`); `--health-port` now sets the sidecar's port. Read, not
run: no `cmd.exe` runs here. Nothing has run on Windows yet; steps 2 and 3 (the evaluation install and the pre-tier record) follow the Linux-side work, as amended under
"Implementation steps".
Gap analysis refs: none — see "Why this tier, why now"

**Added 2026-10-13 (owner decision).** Juno's CUDA backend loads only Linux library names, so on Windows
an NVIDIA GPU is never used and every run is CPU-only. This tier makes every CUDA path Juno has on Linux
run on Windows.

## Objective

Every CUDA path Juno has on Linux runs on Windows x64 with an NVIDIA GPU, through the same code, with
Windows launchers and documentation. CUDA on macOS and ROCm on Windows fail closed with an explicit
message.

## Why this tier, why now

- **What already works across OSes.**
  - The custom kernels are checked-in PTX (`node/src/main/resources/cab/ml/juno/node/*.ptx`, ISA 8.0,
    target sm_61), which the driver loads and compiles (`cuModuleLoadData`). A Windows machine therefore
    needs the driver and the CUDA runtime libraries, not a compiler.
  - The FFM descriptors in `CudaBindings` pass sizes and strides as `size_t` or `long long`, which are
    64-bit on Windows as on Linux. That is a first pass; item 2 completes the audit.
- **What is missing.**
  - `CudaBindings` (lines 158 and 159) and `CudaDriverBindings` (line 76) try only `libcudart.so.12`,
    `libcublas.so.12`, `libcuda.so.1` and their unversioned names.
  - Error text tells the user to set `LD_LIBRARY_PATH`.
  - `scripts/run.bat` (1,089 lines against `run.sh`'s 1,604) prints `[DBG]` lines, and writes
    `HKCU\Console\VirtualTerminalLevel` to the user's registry on every run.
  - `scripts/set_cuda_env.bat` refers to a `jnicudart.dll` that no longer exists.
- **Windows' display driver model behaves differently from Linux**, and Juno has never been checked
  against it:
  - a 2-second kernel watchdog (TDR) on a GPU that drives a display;
  - higher kernel-launch overhead;
  - its own accounting of free device memory, which `--gpu-layers auto`, the prefill reserve and the
    device-wide free-memory tests read.
- **Library versions differ by GPU generation.** CUDA 13 removed Maxwell, Pascal and Volta support. This
  host's GTX 1080 (Pascal) therefore needs the CUDA 12.x runtime, while Turing and newer GPUs may use 12.x
  or 13.x. The two name and lay out their Windows libraries differently: 13.x keeps them under `bin\x64`.
- **Sequenced right after [Tier 02F](TIER-02F-hardware-validation-kit.md)**, whose kit carries the Windows
  verification: the GPU test group, the live checks, the smoke equivalents and the throughput readings,
  none of which the bash harness can run on Windows.

## Scope

### In scope

1. **Library resolution per OS.** A new class, `CudaLibraryLocator`, returns the ordered candidates per
   OS, and `CudaBindings` and `CudaDriverBindings` use it:
   - Linux: unchanged.
   - Windows: `nvcuda.dll`, `cudart64_12.dll` or `cudart64_13.dll`, and `cublas64_12.dll` or
     `cublas64_13.dll`, searched on `PATH`, `%CUDA_PATH%\bin` and `%CUDA_PATH%\bin\x64`.
   - macOS: CUDA unavailable, with an explicit message.

   Error text names the fix for the OS it runs on.
2. **ABI audit** of every FFM descriptor in `CudaBindings`, `CudaDriverBindings` and the kernel launch
   wrappers, against the C prototypes, for Windows' 32-bit `long`.
3. **Display-driver behaviour.**
   - Time the longest single kernel at shipped shapes on Windows, against the watchdog: prefill attention
     at 2048 tokens, the tiled K-quant GEMM at the largest sweep shape, and the LM head.
   - Read launch overhead through the throughput threshold.
   - Check Windows' free-memory accounting against `--gpu-layers auto`, the prefill reserve and the
     device-wide free-memory tests. Each is made correct on Windows, or documented with a fail-closed
     limit.

   *(2026-11-10, owner decision: option (a) of the 2026-11-06 finding. Every launch's work is bounded on every OS: a prefill
   window's attention splits into launches whose rows times keys stay under a budget, and the GEMM into launches of at most so
   many rows, with `gpu.kernel-timing` extended to the longest window the device holds. It lands in
   [Tier 03](TIER-03-kv-cache-maturity.md) before that tier's step 6, recorded there under README rule 9; the timing against
   the watchdog on Windows stays here. See the 2026-11-10 entry.)*
4. **The rest of Juno on Windows**, as the kit's CPU suites find it. Following README rule 8, the audit is
   repo-wide:
   - file paths, and file locking on open model files;
   - line endings in fixtures and templates;
   - ports and the firewall prompt for local clusters (bind to loopback);
   - every `/proc` or other Linux-only read reachable on Windows (`HealthReporter`'s `/proc/meminfo`),
     which must fail soft or closed.
5. **Launchers.**
   - Bring `juno.bat` and `scripts/run.bat` to `run.sh`'s subcommands and flags.
   - Fold in the CUDA path setup from `set_cuda_env.bat`.
   - Remove the `[DBG]` lines.
   - Replace the registry write by enabling virtual-terminal output in-process (`AnsiSupport`, through FFM
     `SetConsoleMode`), so nothing on the user's system changes.
6. **Fail closed elsewhere.** ROCm on Windows (`RocmBindings` loads `libamdhip64.so`) and CUDA on macOS
   refuse with one-line explicit messages, unit-tested.
7. **Docs.** A Windows section in `docs/howto.md` (NVIDIA driver, CUDA 12.x or 13.x runtime by GPU
   generation, JDK 25, Maven to build), the platform support in `README.md`, `docs/agent-arch.txt`, and
   `CHANGELOG.md`.

### Out of scope

- GPU support on macOS (Metal), and ROCm on Windows: both belong to
  [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md)'s backend decision. Here they fail closed (item 6).
- WSL2: it runs the Linux build and libraries; it is verified only if a tester runs the kit there.
- Kernel changes: the PTX is the Linux PTX.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | unchanged; the kit's CPU suites run on Windows to find OS defects (item 4) |
| 2 | CUDA GPU inference | primary target: every CUDA path on Windows |
| 3 | ROCm GPU inference | Linux unchanged (NEEDS-AMD-HARDWARE as before); Windows fails closed (item 6) |
| 4 | Static schedule | GPU throughput probes on Windows |
| 5 | Continuous schedule | schedule probes on Windows |
| 6 | Single-node local mode | primary surface |
| 7 | Pipeline-parallel cluster | local forked JVMs on Windows with a GPU node (live checks 1 to 6) |
| 8 | Tensor-parallel cluster | same (live checks 7 and 8) |
| 9 | LoRA training | on Windows as on Linux (the frozen forward stays on the float kernels) |
| 10 | LoRA playback | GPU playback on Windows |
| 11 | Vision | moondream2 on Windows, GPU decode |
| 12 | OpenAI REST surface | on Windows, through the kit's grammar, tools and launcher probes |
| 13 | Native REST surface | same |
| 14 | CLI | `juno.bat` and `scripts/run.bat` at parity with `run.sh` |
| 15 | JVM embedding facade | `JunoPlayer` with the GPU on Windows |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Install the Windows 11 Enterprise evaluation on this host, alongside Linux, with the NVIDIA driver and
   the CUDA 12.x runtime (owner decision, 2026-10-13). Install it at this step, not earlier, so that its
   90 days cover the tier's work.
3. Before any change, run the kit's `cpu` and `gpu` suites on that Windows machine and publish them as the
   pre-tier record. The `gpu` suite is expected to report CUDA unavailable.
4. Items 1, 2 and 6, tests first.
5. Item 5 (launchers), then item 4 as the kit finds defects.
6. Item 3, on the Windows machine.
7. The thresholds below, the standing gate on this host, then the docs.

*Amended 2026-11-05 (owner decision, recorded under README execution rule 1 and in the execution record):* the tier began
while Tier 02F waits on testers, so the work this host can do runs first. Step 4 ran before steps 2 and 3. Steps 2 and 3
follow the remaining Linux-side work (the kit probes this tier adds, item 4's repo-wide audit as far as it can be read
here, and item 5's launcher edits), so the evaluation's 90 days go to work that needs Windows. Step 3's "before any
change" is kept by building `80ec949e` on the Windows machine: no change of this tier is in it, and it is the commit
Tier 02F's testers build and its reference run was taken at, so one kit run there is both this tier's pre-tier record
and Tier 02F's Windows run.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other test
  or code in this tier (README execution rule 7).
- **Library resolution**, as a pure function of the OS, the environment and which files exist: Windows
  with the CUDA 12.x and 13.x layouts, `CUDA_PATH` unset, Linux unchanged, macOS refused; the error text
  per OS.
- **Fail-closed**: ROCm on Windows and CUDA on macOS refuse with their messages.
- **Launcher**: the kit's launcher probe on Windows (`juno.bat` local mode, one REST request, exit code 0,
  no registry change).
- **Kit probes on Windows**: the `gpu` test group, live checks, residency, packed K-quant, grammar and
  tools, LoRA, vision, local clusters and throughput. A new kit probe times the longest kernels with CUDA
  events.
- **Standing throughput and allocation gate** (README, "Test infrastructure"), on this host's Linux,
  against the pre-tier jar. Library loading and the launchers are not the inference path, but the GPU lane
  holds Linux unchanged.
- **Perf gate (required)**: the Windows-against-Linux readings on the same machine and GPU, the kernel
  timings on Windows, and the standing gate on this host. Published under `docs/perf-compare/` with
  descriptive directory names.

  **Threshold.**
  - CUDA unit tests (`-Dgroups=gpu`) on Windows: failures **<= 0**, over every test Linux runs.
  - Windows GPU tg and pp t/s **>= 0.85x** Linux on the same machine and GPU, every sweep model, at
    `n_prompt=512`. Each OS is read as a median of three runs in one session, unpinned, with the CPU
    governor or power plan and the GPU clocks recorded. *Restated 2026-10-13 by owner decision: the bar was
    0.90x (decision 7), which the README scores only from a same-hour pinned A/B, and that cannot cross a
    reboot. At 0.85x the README's rule for gates looser than 0.90x applies.*
  - Model quality: mean NLL on Windows **<= 1.01x** Linux's on the same GPU, every sweep model. Greedy text
    must be identical to Linux's on the smoke prompts on the same GPU; a difference keeps the tier open
    unless an **Owner decision** records why it is acceptable.
  - Longest single kernel at shipped shapes on Windows: **<= 0.5 s**, a 4x margin under the 2-second
    watchdog default.
  - Standing throughput and allocation gate on Linux: GPU tg and pp **>= 0.95x** the pre-tier build,
    allocation per token **<= 1.10x**, in-span GC pause total **<= 1.25x**.
  - Every throughput number is a median of at least three runs, with min and max published.

## Kit probes

README execution rule 10.

- Keep passing, on Windows and on this host's Linux: the `gpu` suite (`gpu.unit-tests`, `gpu.live.*`,
  `gpu.residency.tinyllama`, `gpu.packed-kquant.*`, `gpu.throughput.*`, `gpu.consistency.tinyllama`,
  `gpu.long-prompt.*`, `gpu.schedule.tinyllama`, `gpu.lora.tinyllama`, `gpu.vision.moondream2`) and
  `launcher.tinyllama`. *(2026-11-07, item 4:)* also `cpu.unit-tests`, which runs item 4's tests (loopback binding, the node stop,
  the merge refusal, CR LF training and grammar files) on the tester's host; `quality.*`, which scores the quality text as checked
  out; and `cpu.live.*`, which runs local clusters on the CPU.
- Add: the kernel-timing probe (the longest single kernel at shipped shapes, CUDA events, against the
  watchdog); the launcher probe's Windows checks (`juno.bat` local mode, one REST request, exit code 0, no
  `[DBG]` output, no registry write); the CUDA libraries resolved and where from, in the `env` reading.
  *(2026-11-06: all three added, with `coverage.tsv` lines and self-tests: `gpu.kernel-timing` (`quick` and `gpu`
  suites), `env`'s `cudaLibraries` and `cudaUnavailableReason`, and `launcher.tinyllama`'s "prints no [DBG] lines" (every
  system) and "writes nothing to the registry" (Windows only). Run on Linux here; the Windows checks have not run.)*
  *(2026-11-07: item 4 adds no probe. Its behaviour is unit-tested and reaches a tester's host through `cpu.unit-tests`;
  `coverage.tsv` carries `feature:line-endings` and `feature:local-cluster-lifecycle`.)*
  *(2026-11-07, item 5: no probe added. The launcher parity and batch-rule tests read files and run in `cpu.unit-tests`;
  `launcher.tinyllama` runs `juno.bat`; `coverage.tsv` carries `feature:windows-launcher-parity`.)*

## Models needed

The four sweep models, all present here; the kit downloads them on Windows. Hardware: this host's GTX
1080 under the Windows 11 Enterprise evaluation (owner decision, 2026-10-13), installed at step 2, with the
CUDA 12.x runtime (Pascal). A tester's run on a Turing or newer GPU (an RTX 3070, for example) would also
exercise the CUDA 13.x layout; it is welcome, not required.

## Exit criteria

- [ ] Every CUDA path runs on Windows through `CudaLibraryLocator`; the CUDA 12.x and 13.x layouts both
      resolve; Linux is unchanged.
- [ ] The ABI audit is recorded, with every descriptor checked.
- [ ] Every threshold above is met, on the Windows machine and, for the standing gate, on this host. A miss
      keeps the tier open; closing on one takes an explicit **Owner decision** recorded here with the
      measured number.
- [ ] Watchdog, launch overhead and free-memory behaviour recorded; each device-wide test is correct on
      Windows or fails closed with a stated limit.
- [x] `juno.bat` and `scripts/run.bat` are at parity with `run.sh`, with no `[DBG]` output and no registry
      write.
      *(2026-11-07: holds as read, by `LauncherParityTest` and `WindowsLauncherScriptTest` (execution record of that day,
      item 5). Owed: the launcher probe on Windows.)*
      *(2026-11-09: the launcher probe ran on a tester's Windows host (no NVIDIA GPU) and failed: `juno.bat local` refused
      `--temperature`. `run.bat` had no space before `(` on that flag in `local` and `cluster` and on `--mmproj-path` in `local`,
      and `LauncherParityTest` reads flag text, so it passed. Fixed, with `WindowsLauncherScriptTest`'s glued-comparison check.
      The same run passed the other three Windows launcher checks (LoRA grammar warning, no `[DBG]`, registry untouched).
      Owed: the probe's re-run on Windows, for local mode's request and clean exit. See Tier 02F's record of that day.)*
      *(2026-11-10: met. The launcher probe passed 5 of 5 in both of the tester's Windows re-runs, at `f234e90c` (the same code as
      `e16748ad`). `juno.bat local --api-port` with `--temperature 0` answered a chat request and exited with code 0, leaving none
      of its processes running. It printed no `[DBG]` line, and `HKCU\Console`'s last-write time was unchanged. See Tier 02F's
      record of that day.)*
- [ ] ROCm on Windows and CUDA on macOS fail closed, tested.
- [ ] Every Linux-only assumption reachable on Windows is fixed or fails closed (repo-wide, README rule 8).
      *(2026-11-07: the repo-wide audit as far as this host reads it is recorded, with five fixes (execution record of that
      day). Owed: the kit's CPU suites on the Windows machine, which find what reading cannot.)*
      *(2026-11-09: a tester's Windows run of the `quick` CPU suite found two: the `run.bat` flags above, and probe logs in
      the machine's code page, which stopped the kit's packing. Both fixed. It also confirmed that a checkout without
      `.gitattributes` writes CR LF text: its quality reading equals this host's CR LF reading to all 16 digits. Still owed:
      the `cpu` and `all` suites, which `quick` does not cover.)*
- [x] Kit probes for this tier's behaviour are added: the kernel-timing probe, and the launcher probe on
      Windows.
      *(2026-11-06: both added and run on Linux (`docs/perf-compare/20261106T045718Z-gpu-kernel-timing/`; the launcher's
      `[DBG]` check passes on `juno`). Owed: the launcher probe's Windows checks have run nowhere; ticked when a Windows
      run shows them, which on the current `juno.bat` should fail both (its 21 `[DBG]` lines and the `reg add`) until
      item 5's edits land.)*
      *(2026-11-10: met. A Windows run shows the launcher probe's checks: all five passed in both of a tester's re-runs
      (`docs/perf-compare/20261110T162530Z-...` and `20261110T162533Z-remote-windows-amd64-12th-gen-core-i5-1240p/`). The
      kernel-timing probe needs an NVIDIA GPU and has run on Linux. Its Windows reading belongs to item 3's watchdog row.)*
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (the `docs/howto.md` Windows section, `README.md`, `docs/agent-arch.txt`) updated, in
      Juno-native language only; `CHANGELOG.md` entry added.

## Execution record

### 2026-10-13: tier created (owner decisions)

- **Owner decisions, 2026-10-13.** Decision 1: this tier runs after Tier 02F. Decision 7: Windows GPU
  throughput must be >= 0.90x Linux on the same machine and GPU.
- **Findings at creation**, from the code: as listed under "Why this tier, why now". In addition,
  `RocmBindings` loads only `libamdhip64.so` and `librocblas.so`, and the PTX's sm_61 target compiles on
  every GPU of compute capability 6.1 or newer.

**Raised with the owner, 2026-10-13: the Windows machine.** This host has a Windows install that is not
activated.
- Activation does not block drivers or CUDA. On an unactivated copy, programs and updates install
  normally; what is locked is personalization, plus a watermark. Technically, that install can run this
  tier.
- Whether to use it unlicensed is the owner's call. Microsoft's Windows 11 Enterprise 90-day evaluation (no
  product key, sign-in with a Microsoft account) is a licensed route for testing.
- Without a Windows install on this host, the same-machine threshold has no reading. It would then need a
  different basis, such as a tester's machine measured against itself on Linux, or a fixed bar.

**Raised with the owner, 2026-10-13: how to score the 0.90x gate.** The README ("No-regression gates
tighter than the floor are Juno-against-Juno, same hour, pinned clocks") scores any gate at 0.90x or
tighter from a same-hour interleaved A/B with pinned clocks. Across two OSes that cannot be done as
written: the A/B alternates by rebooting, Windows offers no clock pinning without admin changes, and this
host's Pascal GPU refuses clock locking on either OS. Options:
- (i) Keep 0.90x, scored by alternating Linux and Windows across six boots in one session (L W L W L W),
  with the CPU pinned on Linux, Windows on its high-performance power plan, and GPU clocks recorded.
- (ii) 0.85x, read from a median-of-three session per OS, unpinned, with clocks recorded. This is the
  README's rule for gates looser than 0.90x.

Recommendation: (ii). The quantity measured is Windows' own overhead, and a 10% band sits inside the
drift between sessions that the README's noise floor describes.

**Owner decisions, 2026-10-13 (later).**
- **The Windows machine:** this host, under Microsoft's Windows 11 Enterprise 90-day evaluation, installed
  when this tier starts (step 2) so that its 90 days cover the work. Not the unactivated install.
- **The scoring method:** option (ii). The Windows-against-Linux bar is restated to `>= 0.85x`, read from a
  median-of-three session per OS, unpinned, with clocks recorded (Threshold block above).

### 2026-11-05: tier begun while Tier 02F stays open (owner decision); step 4, items 1, 2 and 6 on Linux

**Owner decision 2026-11-05.** Tier 02F's six open criteria all wait on testers' Windows and macOS runs, and no executor work on
this host is left in it (its record of the same day). Options put to the owner: start this tier's work this host can do, keep
waiting, or run Tier 02H first. Decision: this tier begins now, Tier 02F's criteria staying open (README execution rule 1 carries
the note). The step order is amended under "Implementation steps".

**Plan against code, verified at HEAD `d8012b11`.** `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep or
build was running. Every "Why this tier" claim holds: `CudaBindings.java:158-159` load `libcudart.so.12`/`libcudart.so` and
`libcublas.so.12`/`libcublas.so`, `CudaDriverBindings.java:76` loads `libcuda.so.1`/`libcuda.so`, the error text names
`LD_LIBRARY_PATH`, `RocmBindings` loads `libamdhip64.so` and `librocblas.so` with no OS check, `scripts/run.bat` (1,089 lines)
prints 21 `[DBG]` lines, and `scripts/set_cuda_env.bat` names `jnicudart.dll`. `CudaLibraryLocator` did not exist. The only Java
GPU bindings are `CudaBindings`, `CudaDriverBindings` and `RocmBindings` (`FunctionDescriptor` appears elsewhere only in
`AnsiSupport` and the kit's `KeepAwake`). The models this step needs: none (unit tests, and this host's CUDA libraries).

**Tests first** (`node`; each seen failing for the missing behaviour against stub classes, then passing):
- `GpuPlatformTest` (4): `os.name` to platform; CUDA refused on macOS and other systems with one line; ROCm refused outside
  Linux with one line. Against the stub, 3 failed (`expected: LINUX but was: OTHER`; the refusals `null`); the fourth, that
  every refusal is one line, passes trivially on a stub and is no evidence.
- `CudaLibraryLocatorTest` (13): library resolution as a pure function of platform, environment and files, Windows paths as
  strings so every case runs on any host. Linux names unchanged whatever the environment; Windows CUDA 12.x layout
  (`%CUDA_PATH%\bin`, cuBLAS with its `cublasLt` preloaded from the same folder); 13.x layout (`%CUDA_PATH%\bin\x64`);
  `CUDA_PATH` unset (quoted and empty `PATH` entries, a variable named `Path`); 12 preferred over 13; cuBLAS matched to the
  runtime's major version; macOS refused for every library; the error text per OS, one line each; and, on a CUDA host, the
  runtime and cuBLAS loaded through the locator (`libcudart.so.12`, `libcublas.so.12` here). Against the stub, 9 failed and 3
  errored; the one-line check passed trivially.
- `GpuBackendRefusalTest` (1): `-Djuno.gpu.backend=rocm` on Linux without ROCm. Failed with
  `"juno.gpu.backend=rocm but ROCm not available"`; now `"...: Could not load GPU library — tried: [libamdhip64.so]"`. On
  Windows the same path carries `GpuPlatform`'s refusal (one test covers the wiring; the refusal text is `GpuPlatformTest`'s).

**The change** (`node` only):
- `GpuPlatform` (new enum): LINUX, WINDOWS, MACOS, OTHER from `os.name`; `cudaRefusal()`/`rocmRefusal()` and the `require*`
  forms.
- `CudaLibraryLocator` (new): `candidates()` and `notFoundMessage()` (pure) and `load()`.
  - Linux: unchanged names.
  - Windows: `nvcuda.dll`; `cudart64_12.dll`, then `_13`, on `PATH`, in `%CUDA_PATH%\bin` and in `%CUDA_PATH%\bin\x64`, then
    the bare names for the system loader; cuBLAS likewise, with `cublasLt64_NN.dll` loaded first from a found file's folder.
    Windows' loader looks for a DLL's dependencies on the search path, not in the folder the DLL was loaded from, so a cuBLAS
    found only under `CUDA_PATH` would otherwise fail on `cublasLt`.
  - macOS and other systems: the refusal.
  - `load()` takes cuBLAS at the runtime's major version. When a file was found but did not load, the error names it.
- `CudaBindings` and `CudaDriverBindings` load through the locator. Their private `loadLibrary` helpers are removed, and
  their ready log lines name the files loaded.
- `CudaBindings` and `RocmBindings` gain `unavailableReason()`; `GpuContext.selectBindings()` appends it to a forced backend's
  error and to the no-backend error.
- `RocmBindings` calls `GpuPlatform.current().requireRocm()` before loading anything.

**Item 2, the binding audit**, against this host's CUDA 12.0 headers (`/usr/include`: `CUDART_VERSION 12000`, cuBLAS 12.0,
`CUDA_VERSION 12000`; libraries `libcudart.so.12.0.146`, `libcublas.so.12.0.2.224`). The Windows question is C `long`, 64-bit
on Linux (LP64) and 32-bit on Windows (LLP64). Every other type in these prototypes has one width on both: `int`,
`unsigned int` and enums 4 bytes; `size_t`, `long long` and pointers 8.

| Library | Symbol | C parameters (header) | Juno descriptor | Same on Windows |
|---|---|---|---|---|
| cudart | `cudaGetDeviceCount` | `int*` | `ADDRESS` | yes |
| cudart | `cudaGetDeviceProperties` | `cudaDeviceProp*, int` | `ADDRESS, JAVA_INT` | yes; see the note below |
| cudart | `cudaSetDevice` | `int` | `JAVA_INT` | yes |
| cudart | `cudaMalloc`, `cudaMallocHost` | `void**, size_t` | `ADDRESS, JAVA_LONG` | yes |
| cudart | `cudaFree`, `cudaFreeHost` | `void*` | `ADDRESS` | yes |
| cudart | `cudaMemGetInfo` | `size_t*, size_t*` (read as 8 bytes) | `ADDRESS, ADDRESS` | yes |
| cudart | `cudaMemcpy` | `void*, const void*, size_t, cudaMemcpyKind` | `ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT` | yes |
| cudart | `cudaMemcpyAsync` | the same, `cudaStream_t` | the same, `ADDRESS` | yes |
| cudart | `cudaStreamCreateWithFlags` | `cudaStream_t*, unsigned int` | `ADDRESS, JAVA_INT` | yes |
| cudart | `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaEventCreate`, `cudaEventDestroy` | one handle or handle pointer | `ADDRESS` | yes |
| cudart | `cudaEventRecord` | `cudaEvent_t, cudaStream_t` | `ADDRESS, ADDRESS` | yes |
| cudart | `cudaEventElapsedTime` | `float*, cudaEvent_t, cudaEvent_t` | `ADDRESS` x3 | yes |
| cuBLAS | `cublasCreate_v2`, `cublasDestroy_v2` | `cublasHandle_t*` / `cublasHandle_t` | `ADDRESS` | yes |
| cuBLAS | `cublasSetStream_v2` | `cublasHandle_t, cudaStream_t` | `ADDRESS, ADDRESS` | yes |
| cuBLAS | `cublasSetPointerMode_v2` | `cublasHandle_t, cublasPointerMode_t` | `ADDRESS, JAVA_INT` | yes |
| cuBLAS | `cublasSgemv_v2` | handle, op, `int m, n`, `float*` x2, `int lda`, `float*`, `int incx`, `float*` x2, `int incy` | the same order | yes |
| cuBLAS | `cublasSgemm_v2` | handle, op x2, `int m, n, k`, pointers and `int` leading dimensions | the same order | yes |
| cuBLAS | `cublasHSSgemvStridedBatched` | as Sgemv plus `long long` `strideA`, `stridex`, `stridey`, `int batchCount` | `JAVA_LONG` for the strides | yes (`long long` is 8 bytes on both) |
| cuBLAS | `cublasGemmEx` | handle, op x2, `int m, n, k`, `const void*` operands, `cudaDataType` x3, `int` leading dimensions, `cublasComputeType_t`, `cublasGemmAlgo_t` | `JAVA_INT` for every enum | yes |
| driver | `cuInit` | `unsigned int` | `JAVA_INT` | yes |
| driver | `cuCtxGetCurrent`, `cuModuleUnload` | `CUcontext*` / `CUmodule` | `ADDRESS` | yes |
| driver | `cuModuleLoadData` | `CUmodule*, const void*` | `ADDRESS, ADDRESS` | yes |
| driver | `cuModuleGetFunction` | `CUfunction*, CUmodule, const char*` | `ADDRESS` x3 | yes |
| driver | `cuLaunchKernel` | `CUfunction`, `unsigned int` x7, `CUstream`, `void**` x2 | `ADDRESS`, `JAVA_INT` x7, `ADDRESS` x3 | yes |

- **No descriptor passes a C `long`**, so all 31 are the same on Windows. `CUDARTAPI`, `CUBLASWINAPI` and `CUDAAPI` name a calling
  convention only on 32-bit Windows; x64 Windows has one, which `Linker.nativeLinker()` implements.
- **`cudaDeviceProp`**, compiled against the header here with gcc: 1,032 bytes, `name` at 0, `totalGlobalMem` at 288. Its fields
  are `char` arrays, `int`, `unsigned int`, `size_t` and `int` arrays, with no `long`, so the layout is the same under LLP64.
  Juno's 1,512-byte buffer covers it, and only those two fields are read.
- **One finding, not Windows-specific.** The 12.x header maps `cudaGetDeviceProperties` to `cudaGetDeviceProperties_v2`
  (`cuda_runtime_api.h:167`). Juno bound the unversioned export, which 12.x keeps for binaries built against 11.x and which
  fills the 11.x layout. Both layouts put `name` and `totalGlobalMem` at 0 and 288, so the readings were right. `CudaBindings`
  now binds the unversioned name where it is exported (unchanged here) and `_v2` where it is not. Whether a 13.x runtime still
  exports the unversioned name is not known here. No 13.x headers or libraries are on this host; a tester's Turing-or-newer
  run on 13.x would show it.
- **`cuLaunchKernel`**: the header maps it to `_ptsz` only under `CUDA_API_PER_THREAD_DEFAULT_STREAM`; Juno binds the plain
  symbol, on both systems.
- **Kernel arguments.** `KernelParams` passes pointers to 8-byte slots holding a device pointer, an `int`, a `long` or a
  `float`. The per-call launch paths (`Q4KMmqKernel`, `GqaAttentionKernel`, `RmsNormKernel`) pass pointers to slots of each
  argument's own Java layout (`ADDRESS`, `JAVA_INT`). Either way the widths are the PTX `.param` declarations' (the same
  checked-in PTX on both systems), not the host C ABI's.
- **Not audited:** `RocmBindings`, since ROCm now refuses on Windows before binding anything (item 6).

**Tests run.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` (first, and after the plan edits) | ok, 26 tier files |
| `mvn -q test -pl node -Dtest=GpuPlatformTest,CudaLibraryLocatorTest` on the stubs | `GpuPlatformTest` 3 of 4 failed; `CudaLibraryLocatorTest` 9 failed and 3 errored of 13, as above |
| `mvn -q test -pl node -Dtest=GpuBackendRefusalTest,GpuPlatformTest` before the wiring | the refusal test failed, as above; `GpuPlatformTest` 4 passed |
| `mvn -q test -pl node -Dtest=GpuPlatformTest,CudaLibraryLocatorTest,GpuBackendRefusalTest,CudaBindingsTest,CudaAvailabilityTest,GpuBindingsDelegationTest,GpuContextTest,RocmBindingsTest,RocmAvailabilityTest` | 79 tests, 0 failures, 0 errors, 31 skipped (the ROCm-present and CUDA-absent cases); the locator test loaded both libraries through the locator here |
| `mvn test -pl node` (every node test, the GPU groups on this host's GTX 1080) | 1,262 tests, 0 failures, 0 errors, 48 skipped; 45 min. The log reads "CudaBindings ready — Panama FFI (cudart from libcudart.so.12, cublas from libcublas.so.12)" and "CudaDriverBindings ready — Panama FFI (libcuda.so.1)", the files loaded before. The device-wide checks passed with the GPU idle: `PrefillRegionHandlerParityTest` 4, `GpuAttentionHandlerParityTest` 2, `PrefillReserveDeviceTest` 3 |

**Docs:** `docs/agent-arch.txt` (the module list and the GPU FFI section: `CudaLibraryLocator`, `GpuPlatform`, the audit outcome)
and `CHANGELOG.md` Session 131, which says that CUDA on Windows has not yet run on Windows. `docs/howto.md`'s Windows section and
`README.md`'s platform support wait for step 7, once the Windows readings exist; claiming Windows CUDA support before then would
be untested.

**Out-of-tier changes:** none; the work is this tier's items 1, 2 and 6. It changes no inference path: on Linux, the files loaded
and the symbols bound are the ones bound before, so it is no measurement boundary. The standing gate is scored at the tier's
close, against `d8012b11`.

**Exit criteria:** none ticked yet.
- Criterion 1 needs every CUDA path run on Windows.
- Criterion 2 (the audit) is recorded above. It stays unticked until a Windows run reads the same device name and VRAM through
  these bindings, which the pre-tier record and the kit's environment reading will show.
- Criterion 6 (the refusals) is unit-tested here, but its macOS half is only seen running on a macOS host (Tier 02F's tester
  run reports it) and its Windows half on the Windows machine. It is ticked when either run shows the refusal line.

**Next for the executor**, all on this host, in this order:
1. The kit probes this tier adds, as far as this host can run them: the environment reading names the CUDA libraries resolved
   and where from; the kernel-timing probe (CUDA events, the longest single kernel at shipped shapes, against 0.5 s); the
   launcher probe's Windows checks (no `[DBG]`, no registry write). Self-tests and `coverage.tsv` lines first.
2. Item 4's repo-wide audit of Linux-only assumptions reachable on Windows (`/proc` reads, paths, line endings, loopback
   binding), with the fixes this host can test.
3. Item 5's launcher edits (`juno.bat`, `scripts/run.bat`, the `set_cuda_env.bat` fold-in, `AnsiSupport`'s `SetConsoleMode`
   instead of the registry write).

Then step 2: the owner installs the Windows evaluation, and step 3's pre-tier kit run at `80ec949e` follows.

**Owed to the owner:** commit this change. No run is owed yet.

### 2026-11-06: the kit probes this tier adds (Linux side; next-for-executor item 1)

**Plan against code, verified at HEAD `d8012b11`** with the 2026-11-05 step 4 change uncommitted in the tree (the owner has
not committed it yet; it is built on, not redone). `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build
or test run was in progress; the GPU drove the desktop only (8%, no compute process). Every claim this item depends on holds:
`CudaLibraryLocator.Loaded` carries each library's location; `scripts/run.bat` prints 21 `[DBG]` lines and runs
`reg add "HKCU\Console" /v VirtualTerminalLevel ...` at line 38, and `juno.bat` calls it; `juno` and `scripts/run.sh` print no
`[DBG]` line; the catalog had no kernel-timing probe (`KernelTimingMain` is the CPU width probe's timing JVM). The shipped shapes
were read from the code and the files, not assumed:
- one prefill window per prompt on the GPU: the static schedule's adaptive window covers a whole prompt in one window, up to a
  device-memory ceiling and at most 65,536 rows (`PrefillBatchOptions.ADAPTIVE_CHUNK_CEILING`), and the device prefill region
  launches attention once per layer over the window (`PrefillWindowRegion.attendInside`, rows over one cache);
- the GEMM: `CudaMatVec.gemmOnStream` runs `KQuantGemmKernel` over the window's rows; gate and up are one matrix only for Phi-3,
  so the largest matrices of the sweep are Mistral 7B's 14336x4096 gate and up and its 4096x14336 down;
- the LM head: `outputProjectionBatch` takes one row per token through `CudaMatVec.sgemvInto` (`Q4KMmqKernel`'s packed GEMV);
- tensor types, from the four sweep models' GGUF headers (type fields only, no weights read): every output matrix is Q6_K
  (Qwen2.5-3B's 151936x2048 the largest), Mistral 7B's `ffn_gate` Q4_K and `ffn_down` Q6_K.

**Tests first.** Each was seen failing against a stub for the missing behaviour, then passing:
- `node`: `GpuKernelTimingsTest` (3; the two GPU cases tagged `gpu`): every kernel at every sweep shape, one positive reading per
  launch; the attention reading at 1024 rows over 8x the 64-row reading (it follows the work, not launch overhead); without
  CUDA, a refusal naming CUDA (skips here). Against the stub: one failure (no timings), one error (no attention reading).
  `CudaAvailabilityTest` (+2): the libraries loaded and where from (`libcudart.so.12`, `libcublas.so.12`, `libcuda.so.1` on
  Linux), and, without CUDA, none plus a one-line reason (skips here). Against the stub: one failure (empty map).
- `juno-kit`: `ProbeSelfTests` (one new, two extended): `gpu.kernel-timing` at a 64-token window (on a CUDA host every kernel
  read and the check met; elsewhere the check fails with the reason and nothing is read); `env` carries `cudaLibraries` (or
  `cudaUnavailableReason`); `launcher.tinyllama` adds "the launcher prints no [DBG] lines". `CoverageTest` failed with
  "probe gpu.kernel-timing: covers no line" once the catalog held the probe. All four failed before the code.

**The change.**
- `node`: `GpuKernelTimings` (new, public for the kit): the device time of single launches, a CUDA event pair around each
  (default stream, two untimed launches first), of prefill attention on the four sweep models' heads over a window of `tokens`
  rows (row `b` over keys `[0, b + 1)`, as the region launches it at a prompt's start), the tiled K-quant GEMM at Mistral 7B's
  FFN shapes over `tokens` rows, and the LM head GEMV at Qwen2.5-3B's output (one row). Only the timed kernel runs between the
  events (the Q8_1 packing runs before them). Seeded random weights with finite FP16 block scales. `CudaBindings` and
  `CudaDriverBindings` keep the location each library was loaded from; `CudaAvailability.loadedLibraries()` and
  `unavailableReason()` expose them (`CudaDriverBindings.unavailableReason()` added for symmetry).
- `juno-kit`: `GpuKernelTimingProbe` (`gpu.kernel-timing`: CHECK, `quick` and `gpu` suites, needs CUDA, priority 1, cost 10 s,
  timeout 900 s; arguments `tokens` 2048, `reps` 5, `limitMs` 500). One unscored reading per kernel; the check holds the
  slowest launch of all to 500 ms. `EnvProbe` adds `cudaLibraries` and `cudaUnavailableReason`. `LauncherProbe` adds the
  `[DBG]` check over both of its runs' output (every system) and, on Windows, `HKCU\Console`'s last-write time before and after
  both runs (`ConsoleRegistryKey`, FFM to `advapi32`, query access only). The time, not the value: an earlier `run.bat` run has
  already set `VirtualTerminalLevel` on any machine that ran it, so a value comparison would miss the rewrite.
  `coverage.tsv`: `gpu.kernel-timing` under surface 2, `feature:cuda-libraries`, `feature:gpu-kernel-watchdog`.

**Tests run.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` (first, and after the plan edits) | ok, 26 tier files |
| `mvn -q test -pl node -Dtest=GpuKernelTimingsTest,CudaAvailabilityTest` on the stubs, then the code | 3 and 1 failing as above; then 13 tests, 0 failures, 3 skipped (the CUDA-absent cases) |
| `mvn -q test -pl juno-kit -am -Dtest='ProbeSelfTests#envProbe*+gpuKernel*+launcherProbe*,CoverageTest,ProbeCatalogTest'` on the stub, then the code | 4 failing as above; then 15 tests, 0 failures |
| `mvn -q test -pl juno-kit -am -Dtest='cab/ml/juno/kit/**/*'` (every kit test, compiled against this tree's `node`) | 120 tests, 0 failures, 0 errors, 0 skipped |
| `mvn -q test -pl node -Dtest=` the step 4 binding classes, `GpuKernelTimingsTest`, `CudaAvailabilityTest`, `GqaAttentionKernelParityTest` | 11 classes, 86 tests, 0 failures, 0 errors, 33 skipped (ROCm-present and CUDA-absent cases) |
| `./juno-kit.sh --probes env,gpu.kernel-timing,launcher.tinyllama --yes --models-dir models` | every probe PASS or DONE; zip `juno-kit-results-linux-amd64-20261106T045718Z.zip`, no user name in it |

The full `node` suite was not re-run: the change adds a class and two read-only accessors and moves no code path (the step 4 run,
1,262 tests, stands for the rest). The full unit suite and `mvn verify -pl juno-master` run at the tier's close.

**Readings** (unpinned, GTX 1080 on Linux; `docs/perf-compare/20261106T045718Z-gpu-kernel-timing/`). Single-launch device
time, the slowest of each kernel's launches:

| Kernel and shape | 2048 tokens | 4096 tokens | 8192 tokens |
|---|---|---|---|
| Prefill attention, Mistral 7B (32/8/128), the slowest of the four head shapes | **37.37 ms** | 149.73 ms | **596.51 ms** |
| Prefill attention, Phi-3.5-mini (32/32/96) | 35.17 ms | 140.18 ms | 565.39 ms |
| Tiled K-quant GEMM, Mistral 7B `ffn_down` 4096x14336 Q6_K (the slower of the two) | 25.24 ms | 49.06 ms | 98.72 ms |
| LM head GEMV, Qwen2.5-3B 151936x2048 Q6_K | 1.93 ms | 1.95 ms | 2.01 ms |
| Check, slowest launch <= 500 ms | met | met | **missed** |

- The `env` reading names `libcudart.so.12`, `libcublas.so.12` and `libcuda.so.1`, the names the locator's Linux candidates load
  through the system loader.
- `gpu.kernel-timing` took 6 s here (catalog cost 10 s) and peaked at 152 MiB of device memory as sampled every other second,
  which can miss the LM head's phase (about 260 MiB of weights); both round up to the catalog's 512 MiB minimum.

**Finding, raised with the owner: the watchdog bound holds at 2048 tokens, not at long prompts.** Scope item 3 names "prefill
attention at 2048 tokens", and there the slowest kernel is 13x under the 0.5 s bar. But the GPU prefill window is the prompt, so
the attention launch grows with the square of the prompt length and the GEMM linearly: on this GPU Mistral 7B's attention reads
0.60 s at 8192 tokens, and extrapolates to about 2.4 s, past Windows' 2 s watchdog, near 16,000 tokens. An 8 GiB card holds a
Mistral 7B window of several thousand rows; a 24 GiB card holds windows up to the 65,536-row ceiling, where even a GPU four times
faster passes 2 s. A watchdog reset on Windows ends the CUDA context, so the request (and the process's GPU state) fails rather
than slows. Linux on a GPU that drives a display has a watchdog of its own. Options for item 3 (step 6), none needed before it:
- (a) Bound each launch's work on every OS: split a window's attention into launches whose rows times keys stay under a budget,
  and the GEMM into launches of at most so many rows, sized so the slowest launch stays under 0.5 s at the longest window the
  device can hold; extend `gpu.kernel-timing` to time that window. A launch-site change in the prefill paths, so the standing
  throughput gate applies to it.
- (b) A fail-closed limit: on Windows (or wherever CUDA reports a kernel time limit for the device), cap the prompt tokens per
  GPU prefill window and say so in the log and the docs. Smaller, but a long prompt's later windows still attend over every
  earlier key, so a row cap alone does not bound attention time.
- (c) Keep the plan's 2048-token reading as the threshold and document the long-prompt risk.
- **Recommendation: (a).** It is the only option that keeps every single kernel bounded at every prompt length; (b) caps the
  rows but not the keys, and (c) leaves Windows users with long prompts a context reset.

**Out-of-tier changes:** none. The kit probes are this tier's (its **Kit probes** section); `GpuKernelTimings` is a new class off
the inference path, and the binding classes only keep a string they already had. No measurement boundary.

**Exit criteria:** none ticked. Criterion 8 has its dated note (the launcher probe's Windows checks have run nowhere).

**Next for the executor**, on this host, as the 2026-11-05 entry ordered: item 4's repo-wide audit of Linux-only assumptions
reachable on Windows, then item 5's launcher edits. A suggestion for the owner, not a decision: committing this change before item
5's launcher edits lets a Windows run at that commit show the launcher probe's two Windows checks failing on the current `juno.bat`,
which is the only way to see the registry check detect a write.

**Owed to the owner:** commit the step 4 change and this one; the decision above before item 3. No run is owed.

### 2026-11-07: item 4, Linux-only assumptions reachable on Windows (Linux side; next-for-executor item 2)

**Plan against code, verified at HEAD `d8012b11`** with the 2026-11-05 and 2026-11-06 changes uncommitted in the tree (built on,
not redone). `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build or test run was in progress. No tester's
zip has arrived, so Tier 02F has no executor work. The claims this item depends on hold: `HealthReporter` reads `/proc/meminfo`;
`EmbeddedNodeServer` listens through `ServerBuilder.forPort` (every interface); `ClusterHarness.stop()` calls `Process.destroy()`;
the repository has no `.gitattributes`; `run.bat`, `juno.bat` and `set_cuda_env.bat` are stored with LF, `juno-kit.bat` and three
`.java` files with CR LF. No model is needed beyond TinyLlama (present).

**The audit, repo-wide** (README rule 8): every `src/main` tree except the old baseline trees under `dist/`, grepped for `/proc`,
`/sys`, `/dev`, external commands, path separators and string-built paths, `/tmp` and `HOME`, Linux library names, bind addresses,
timestamped file names, file opening, mapping, moving and deleting, text-file and resource reads, and signals and process
termination. The kit (Tier 02F's, built for three OSes from the start) was read only where it consumes repository text.

| Area | Finding | Outcome |
|---|---|---|
| `/proc`, `/sys` | product code: `HealthReporter` reads `/proc/meminfo` only, and falls back to `getFreeMemorySize()` when the file is missing (on Windows the available physical memory, the figure the Linux path reads) | fails soft; unchanged |
| External commands | product code starts `java` only (`ClusterHarness`, the path from `ProcessHandle`, the JVM's own class path) | none |
| Library names | CUDA and ROCm names are behind step 4's `CudaLibraryLocator` and `GpuPlatform` | done at step 4 |
| File names and paths | timestamps `yyyyMMdd-HHmmss` (no `:`); names from `Path.getFileName`; no splitting on `/`; temp files from `createTempFile` | none |
| Open model files | `GgufReader` and `LlamafileGgufIndex` open read-only channels and map nothing; NIO opens with read, write and delete sharing on Windows, so an open model locks nothing | none |
| Merge output | `LoraMerge` opens the output with truncation while it reads the model, so an output that is the model destroys it, on every OS; on Windows another letter case names the same file | **fixed**: refused when the output is the model or the checkpoint (`Files.isSameFile`) |
| Line ends, repository files | no `.gitattributes`: with `core.autocrlf=true` (Git for Windows' default) the fixtures, the PTX and the kit's quality text check out CR LF; with `false` the batch files check out LF, which `cmd.exe` misreads around labels | **fixed**: `.gitattributes`. CR LF PTX compiles to the same cubin (`ptxas -arch=sm_61`, all seven kernels); the quality text does not score the same (below) |
| Line ends, user files | `--grammar-file`: `\r` is whitespace to the GBNF parser and comments end at `\n`; JSON schema and QA files: Jackson; `/train-file`: tokenized as read, a CR token before every newline | `/train-file` **fixed**; grammar held by a new test |
| Ports and the firewall | a local cluster's forked nodes listened on every interface, which makes Windows ask the user about the firewall | **fixed**: loopback |
| REST server (`--api-port`) | listens on every interface on every OS, as before; on Windows the first run shows the firewall prompt, and requests from this host work whatever the answer | unchanged; the Windows section of `docs/howto.md` (step 7) says so |
| Stopping a node | `Process.destroy()` is SIGTERM on Linux and a forced termination on Windows, where the node's shutdown hooks (the gRPC drain, the `--jfr` recording's dump) never run | **fixed**: stdin closed first |
| Console text | `AnsiSupport` writes UTF-8 and enables virtual-terminal output in-process; `run.bat` sets code page 65001 | item 5's input: a thinner launcher keeps `chcp 65001` or moves it into `AnsiSupport` |

**The quality text, measured** (`docs/perf-compare/20261107T070530Z-nll-line-endings/`, TinyLlama 1.1B, CPU integer kernels, 2048
predictions, the kit jar of 2026-11-06): LF 2.45609, CR LF 2.39748 nats, 0.976x. `ingest`'s bound is one-sided (at most 1.01x),
so a CR LF run passes it while scoring another text. Tier 02F's record carries the reading note for testers' runs at `80ec949e`.

**Tests first** (each seen failing against a stub that kept today's behaviour, then passing):
- `node`: `NodeBindAddressTest` (4; 3 failed: `null` for a literal and for the loopback value, no refusal of an unresolvable
  name); `EmbeddedNodeServerBindTest` (2; the loopback case failed, the server listened on the wildcard address; the
  every-interface case passed, today's behaviour); `ParentPipeWatchTest` (2, both failed: the action never ran);
  `LoraMergeSameFileTest` (3, all failed with "Expecting code to raise a throwable": the merge ran over its own input).
- `juno-player`: `ClusterHarnessNodeStopTest` (3) forks `NodeMain` in stub mode the way the harness does. "Exits with code 0
  once its stdin closes" failed with exit code 143 (SIGTERM from `destroy()`), and the loopback-property case failed; the
  fallback case (a node ignoring its stdin is destroyed) passed, as it holds today's behaviour. `TrainingTextFileTest` (3; the
  CR LF and lone-CR cases failed, the LF case passed).
- `sampler`: `GbnfGrammarTest` gains "a CR LF grammar file parses like the LF file", a regression test that passed on its first
  run (the parser already treated CR as whitespace).

**The change.**
- `.gitattributes` (new): `* text=auto eol=lf`, `*.bat` and `*.cmd` `text eol=crlf`. `git cat-file --filters` (the checkout
  conversion, read only) shows the result for both settings: the quality text, `rope.ptx`, `juno.bat` and `run.bat` went from 77,
  985, 4 and 1,089 CR LF lines (`autocrlf=true`) and 0, 0, 0 and 0 (`false`) to 0, 0, 4 and 1,089 under both. Files already stored
  CR LF (`juno-kit.bat`, `AnsiSupport.java`, `Color.java`, `InferencePipeline.java`) are left as stored; `git status` shows no file
  changed by the attributes.
- `node`: `NodeBindAddress` (new: `node.bind`, unset for every interface, one-line refusal of a name that does not resolve);
  `EmbeddedNodeServer` takes a bind address (`NettyServerBuilder.forAddress`) and logs it; `ParentPipeWatch` (new: reads the
  standard input to its end on a daemon thread, then runs the action); `LoraMerge.refuseOverwrite` before anything is read.
- `juno-node`: `NodeMain` reads `node.bind` (exit status 1 with the one line on a bad value) and, with
  `-Djuno.node.exitWithParent=true`, exits through `System.exit(0)` when its standard input ends.
- `juno-player`: `ClusterHarness` gives every node `nodeLifecycleProperties()` (loopback, exit with parent); `stop()` closes every
  node's standard input, then `stopNode` waits up to `NODE_STOP_GRACE_MS` (10 s) for each and destroys only a node still running.
  `TrainingTextFile` (new) reads `/train-file` with CR LF as LF. `LoraMergeMain` prints the merge refusal as one line.
- `coverage.tsv`: `feature:line-endings` (`cpu.unit-tests`, `quality.*`) and `feature:local-cluster-lifecycle`
  (`cpu.unit-tests`, `cpu.live.*`, `gpu.live.*`). The kit's `cpu.unit-tests` probe runs the new tests on a tester's host, Windows
  included, at any commit that has them.

**Tests run.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` (first, and after the plan edits) | ok, 26 tier files |
| the new tests against the stubs (`mvn test -pl node,sampler,juno-player -am -Dtest=...`) | `node` 9 of 11 failed, `juno-player` 4 of 6 failed, as above; `GbnfGrammarTest` 11 passed |
| the same, against the change, with `EmbeddedNodeServerLoadFailureTest` | 33 tests, 0 failures |
| `./juno cluster --model-path <tinyllama> --cpu --jfr 2m`, one prompt, then stdin closed (scratch script) | the nodes listened on `[::ffff:127.0.0.1]:19092` to `19094` only; the cluster answered; the coordinator exited 0, no node JVM was left, and each node's recording was written with `jdk.Shutdown` reason "Shutdown requested from Java" (the standard-input path, not a signal) |
| `mvn install -DskipTests`, then `mvn verify -pl juno-master` (stub mode) | 13 unit tests and 20 ITs, 0 failures (`ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5, `UnsupportedArchitectureClusterIT` 1, `InProcessClusterIT` 6) |
| `mvn test -Dmaven.test.failure.ignore=true -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player,juno-kit` (the GPU groups on the GTX 1080; 1 h 04 min) | 2,652 tests, **1 failure**, 0 errors, 55 skipped. `node` 1,278 (1 failure, 50 skipped), `coordinator` 401, `juno-player` 133, `juno-kit` 120, `lora` 116, `tokenizer` 109, `kvcache` 105, `vision` 97, `registry` 93, `sampler` 90, `metrics` 86, `health` 23, every module but `node` without a failure. The failure is device-wide and not this change's, below |
| `PrefillReserveDeviceTest` alone, three times; then the same class from committed `d8012b11` built clean in a scratch tree (`git archive`, `mvn test -pl node -am`), twice | this tree: 83,165,184, 87,162,880 and 87,097,344 bytes reported free with the device full, against <= 67,108,864; `d8012b11`: 84,410,368 both times. Same failure without any of this tier's changes |
| `mvn test -pl juno-kit -am -Dtest=CoverageTest,ProbeCatalogTest` (the two new `coverage.tsv` lines) | 12 tests, 0 failures |

**The failing test, read.** `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` fills the device in
chunks down to 4 KiB and then reads `cudaMemGetInfo`: what the driver still reports free but will not allocate must stay within
`DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES` (64 MiB). It read 80 to 83 MiB on every run today, on this tree and on `d8012b11`
alike; it passed on 2026-11-05 with the GPU idle. The driver is unchanged (580.178.04); the host was rebooted on 2026-11-05 at
18:25 local and the GPU drives its desktop (about 560 MiB used, 7% to 34% busy, no compute process). So the holdback the driver
keeps moves with the desktop's state, by about 16 MiB here. The bound is not loosened. Free-memory accounting against the reserve
is already item 3's ("the device-wide free-memory tests"); this reading is its first input on Linux, and the Windows reading
comes with step 6.

**Out-of-tier changes:** one, doc only: `docs/howto.md` linked the LoRA guide as `docs/LoRA.md`, which commit `9e96c3b0` removed when
it split the guide into `juno-documentation/part4/`; the link now points there. No inference path changed: the local cluster's
nodes run the same forward pass, and only how they listen and stop changed. No measurement boundary. The compare and smoke
scripts drive `./juno local`, the kit's cluster probes and the ITs, which pass; the standing gate is scored at the tier's close.

**Found, not changed here:** `docs/howto.md`'s Windows note says every flag, environment variable and subcommand is identical across
platforms. It is not: `run.bat` has no `gguf-info` or `health` subcommand and lacks flags `run.sh` has. Making it true is item 5's,
and how is the decision below.

**Docs:** `docs/agent-arch.txt` (the node and juno-player class lists; the local cluster node lifecycle; the merge refusal),
`docs/howto.md` (the cluster section: loopback, the stop, `node.bind`; merge `--output`), `juno-documentation/part4/03-training-guide.md`
(`/train-file` line ends), `CHANGELOG.md` Session 133.

**Exit criteria:** none ticked.
- Criterion 7 (every Linux-only assumption reachable on Windows fixed or failing closed) stays open: the audit above is the half
  this host can read; the kit's CPU suites on Windows are the other half (scope item 4).

**Decision for the owner, before item 5: how `run.bat` reaches parity with `run.sh`.** `run.sh` parses every flag of every
subcommand and then passes the same flag on to the Java entry point, which parses it again and refuses an unknown one ("Unknown
option"). Most `JUNO_*` environment overrides exist only in `run.sh`'s mapping; `ConsoleMain` reads three itself. `run.bat` mirrors
`run.sh` by hand and has drifted: no `gguf-info`, no `health`, flags missing per subcommand. No `cmd.exe` runs on this host, so any
`run.bat` edit is first run by a Windows kit run.
- (a) **Thin `run.bat`**: it keeps the subcommand dispatch, Java discovery and the version check, the heap derived from the model
  file (a scan for `--model-path` and `--heap`), the JVM flags and code page 65001, and passes every other argument through to the
  Java entry point, which already parses and refuses. The environment overrides move into one Java table read when a flag is
  absent, so both launchers get them. Parity holds by construction for every flag, now and later; `run.sh` is unchanged.
- (b) **Mirror `run.sh` flag by flag** (the current design): about 40 flags per serving subcommand, every new flag added to both
  scripts in two languages, none of it runnable here.
- (c) **Move the launcher logic into Java** so both scripts become thin: the most code, and it changes the Linux launcher that
  every smoke and compare script drives.
- **Recommendation: (a).** Also, for item 5's CUDA setup: fold `set_cuda_env.bat`'s extra search places (`CUDA_HOME`, the
  toolkit's default folder) into `CudaLibraryLocator`, where every surface (the embedding facade included) gets them, and delete
  `set_cuda_env.bat`, which names a `jnicudart.dll` that no longer exists.

*Owner decision, 2026-11-07: (b), "as it works now".* Recorded under the next entry, which carries item 5 out on that basis:
`run.bat` keeps its own parsing, and `set_cuda_env.bat`'s search places are folded into `run.bat` (the launcher, as now), not
into `CudaLibraryLocator`; `set_cuda_env.bat` stays, its text corrected.

**Next for the executor:** item 5, once the owner has chosen. Then steps 2 and 3 (the evaluation install, the pre-tier kit run at
`80ec949e`). The suggestion of 2026-11-06 stands: committing before item 5's launcher edits lets a Windows run show the launcher
probe's two Windows checks failing on today's `juno.bat`.

**Owed to the owner:** the decision above; commit the step 4 change, the kit probes and this change. No run is owed.

### 2026-11-07: item 5, the launchers (Linux side; owner decision (b))

**Owner decision, 2026-11-07.** Of the three ways for `run.bat` to reach parity with `run.sh` (entry above), the owner chose
(b), "as it works now": `run.bat` keeps its own parsing and mirrors `run.sh` flag by flag. The CUDA search places of
`set_cuda_env.bat` go into the launcher, as the scope says, and `set_cuda_env.bat` stays.

**Plan against code.** The scope's four points held: `run.bat` lacked subcommands and flags, `set_cuda_env.bat`'s search
places were not in `run.bat`, 21 `[DBG]` lines, one `reg add "HKCU\Console" ... VirtualTerminalLevel`. `AnsiSupport` already
enabled virtual-terminal output through FFM `SetConsoleMode` in `ConsoleMain`, `LoraMergeMain` and `LoraImportMain`, so the
registry write was redundant for them; `GgufInfoMain`, `ModelLiveRunner` and the health server print no escape codes. A
comparison of the two launchers per subcommand (flags accepted, options passed to Java, environment overrides read, jars run)
found more than the scope names:

| Found | Where | Effect on Windows before this change |
|---|---|---|
| no `gguf-info`, no `health` subcommand | dispatch | the first word was taken as a `cluster` flag |
| flags missing: `--context-shift`, `--gpu-residency`, `--prefill`, `--health`, `--health-port` (cluster); `--context-shift`, `--gpu-attention`, `--gpu-residency`, `--health`, `--health-port` (local); `--health`, `--health-port` (lora) | parse loops | "Unknown flag", exit 1 |
| `local --lora-play PATH` read and never passed on | `local` launch line | no adapter applied, silently (`docs/howto.md` shows this command for Windows) |
| `test` ran `juno-master-0.1.2.jar`, the thin jar | `LIVE_JAR` | `NoClassDefFoundError: cab/ml/juno/node/ModelFileGate` at once (run here from that jar) |
| `USE_GPU` overwritten before it was read (`USE_GPU_ENV` read instead) | cluster, local, lora | `USE_GPU=false` ignored |
| `JUNO_EMBEDDINGS`, `JUNO_POOLING`, `PREFILL_MODE`, `JUNO_GPU_RESIDENCY`, `JUNO_GPU_ATTENTION`, `JUNO_CONTEXT_SHIFT`, `HEALTH_PORT`; `merge`'s `JUNO_USE_GPU` | prologues | overrides ignored |
| an `echo` inside `if ... ( ... )` whose `)` ends the block: 10 places | help blocks of cluster, local, lora, test; three one-line `--hf` notices; three warnings | after the first unescaped `)` the rest of a help block runs whatever the condition, so an unknown flag or a positional model path printed part of the help and exited 0; the one-line `--hf` notices leave text after their block's end on a line every run parses |
| a header note: "Do not re-add a `health` subcommand or `--health` / `--health-port` flags ... without explicit instruction" | `run.bat` lines 9 to 13 | contradicted by `run.sh`, which has both, and by this tier's scope (owner, 2026-10-13); removed |
| `--health-port` accepted by `run.sh` and never passed on; `ConsoleMain` had no such option | `run.sh`, `ConsoleMain` | the sidecar listened on 8081 while the launcher printed the chosen port, on every OS |

**Tests first.**
- `LauncherParityTest` (juno-player, new; reads both scripts, no `cmd.exe`): every `run.sh` subcommand dispatched by
  `run.bat`; every flag a `run.sh` subcommand accepts accepted by `run.bat` (case-insensitive, as `if /i`); every option a
  `run.sh` subcommand passes to Java passed by `run.bat`; every jar `run.bat` names one `run.sh` names. Before the change, 6
  tests (the two batch-rule tests were in this class then), 6 failures: the subcommands `gguf-info` and `health`; 12 flags; 9
  options; `juno-master-*.jar`; the `[DBG]` lines; the `reg add`.
- `WindowsLauncherScriptTest` (new): no `[DBG]`, no registry write (moved here), and no `echo` that ends its enclosing block
  early, with a fixture test showing that the check finds the help-text pattern and passes its fixes. On the pre-change
  `run.bat` the check reports the 10 places of the table above, the same 10 an independent reading in a scratch script found.
- `HealthPortOptionTest` (new): a port parses; 0, 65536, -1, a word and an empty value are refused with the flag named;
  `ConsoleMain` takes the sidecar port from `--health-port`. Before the wiring: the parse tests passed and the console test
  failed (`expected: 9191 but was: 8081`, and "Unknown option: --health-port" on stderr).

**Change.**
- `scripts/run.bat`: the `gguf-info` and `health` subcommands as `run.sh` has them; the missing flags, overrides and
  forwarded options above; `--context-shift` on `cluster` accepts only an off value and otherwise stops with `run.sh`'s
  message; `USE_GPU` read as `run.sh` reads it (`:use_gpu_from_env`; `USE_GPU_ENV` still read when `USE_GPU` is unset);
  `merge` reads `JUNO_USE_GPU`; `test` runs `juno-master.jar`, and `health` runs `juno-health.jar`; the 21 `[DBG]` lines and
  the `reg add` removed (`chcp 65001` kept: it sets this console's code page only); each `--help` text moved under its own
  label (`:cluster_help`, `:local_help`, `:lora_help`, `:test_help`, as `merge` already had), and the notices inside blocks
  escape their parentheses; `:prepend_cuda_path` takes `CUDA_PATH`, else `CUDA_HOME`, else the last version folder under
  `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`, and puts its `bin\x64` (13.x) and `bin` (12.x) on `PATH`, as
  `set_cuda_env.bat` did; usage lists the new subcommands and flags.
- `scripts/run.sh`: `--health` passes `--health-port N` with it (cluster, local, lora).
- `ConsoleMain`: `--health-port N` (new class `HealthPortOption`); its help drops four health-server flags it never parsed
  (`--port`, `--stale-ms`, `--warn`, `--critical`, which are `juno health`'s).
- `scripts/set_cuda_env.bat`: kept; its text named a `jnicudart.dll` and a `setenv.bat` that do not exist, and now says what
  it is for (running a jar with `java` directly; `run.bat` needs no call to it).
- `coverage.tsv`: `feature:windows-launcher-parity` (`cpu.unit-tests`, `launcher.tinyllama`). No probe added: the parity
  and batch-rule tests read files and run in `cpu.unit-tests` on every host; `launcher.tinyllama` runs `juno.bat` on Windows.

**Read, not run.** No `cmd.exe` runs on this host (no Windows, no emulator). The table's batch effects are read from
`cmd.exe`'s parsing rules, not observed; the fixes do not depend on them being right (help text under a label prints the
same whichever way a block parses, and an escaped parenthesis is plain text in and out of a block). Every changed `run.bat`
line is first run by a Windows kit run: `launcher.tinyllama` (`juno.bat` local mode, one REST request, exit 0, no `[DBG]`,
no registry write) and the probes that drive `juno.bat`.

**Found, not changed.**
- Paths with spaces: both launchers pass `--model-path`, `--lora-play`, `--grammar-file` and the like unquoted
  (`--model-path $model`, `--model-path %MODEL%`), so a path such as `C:\Users\First Last\models\x.gguf` splits into two
  arguments. Linux and Windows alike; more common on Windows, where profile folders carry spaces. A change to both
  launchers; raised, not made, since it changes `run.sh`'s argument handling that every smoke and compare script drives.
- `test --jfr` names its recording from `wmic os get localdatetime`; recent Windows 11 builds no longer ship `wmic`, and
  the file is then named `juno--.jfr`.
- `--health-port` without `--health` does nothing, in both launchers, as before.
- A value refused at `ConsoleMain`'s command line (`--gpu-residency`, `--health-port`) ends with a stack trace and exit 1.

**Docs.** `docs/howto.md`: the `health` subcommand row, `--health` and `--health-port` rows, and `gguf-info` on Windows
(the "Linux/macOS only" note removed; the parity claim of the Windows note on line 17 now holds). `docs/agent-arch.txt`:
`HealthPortOption`, and a "Launchers" paragraph (the two tests, the block rule, colours, the CUDA `PATH`). `CHANGELOG.md`:
Session 134.

**Perf gate.** Not run: no forward pass, MatVec, residency, batching or KV path changed (launcher scripts, one console
option).

**Tests run.**

| Command | Result |
|---|---|
| `mvn test -pl node,juno-player -Dtest=LauncherParityTest` (before the `run.bat` change) | 6 tests, 6 failures, as listed above |
| `mvn test -pl node,juno-player -Dtest=HealthPortOptionTest` (before the `ConsoleMain` wiring) | 3 tests, 1 failure (`expected: 9191 but was: 8081`) |
| the block check on the pre-change `run.bat` (scratch copy; the test's own method) | 10 places; 0 on the changed `run.bat` and on `juno.bat` |
| `mvn test -pl node,juno-player -Dtest='LauncherParityTest,WindowsLauncherScriptTest,HealthPortOptionTest'` | 11 tests, 0 failures |
| `mvn test -pl juno-player,juno-kit -am -Dtest='cab/ml/juno/player/**/*,cab/ml/juno/kit/**/*'` (2026-11-08 02:53Z to 03:12Z) | juno-player 144 tests, 0 failures, 2 skipped (`LoraTrainerTest`, as before); juno-kit 120 tests, 0 failures (`CoverageTest` 7, `ProbeCatalogTest` 5) |
| `./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --cpu --health --health-port 9191` (rebuilt shaded jar) | `GET http://127.0.0.1:9191/health` 200 `{"nodeCount":0,"nodes":[],"status":"HEALTHY"}`; nothing listens on 8081 |
| `bash -n scripts/run.sh` | passes |
| `scripts/performance-tests/check-plan-thresholds.sh` | ok, before and after |

**Exit criteria.** None ticked. "`juno.bat` and `scripts/run.bat` are at parity with `run.sh`, with no `[DBG]` output and no
registry write" holds as read (the two tests); owed: the launcher probe on Windows. "Kit probes ... the launcher probe on
Windows": unchanged, owed the same run.

**Next for the executor.** The Linux-side work this tier listed is done. Steps 2 and 3 are the owner's: install the Windows
11 Enterprise evaluation on this host with the NVIDIA driver and the CUDA 12.x runtime, then the kit's `cpu` and `gpu`
suites at `80ec949e` on it (Tier 02F's Windows run as well). Then this change's tree on Windows (the launcher probe and
item 4's unit tests), item 3, and the thresholds.

**Owed to the owner:** commit the step 4, kit-probe, item 4 and item 5 changes; steps 2 and 3.

### 2026-11-09: Tier 03 begins while this tier is postponed (owner decision)

Owner decision 2026-11-09, put in [Tier 02F](TIER-02F-hardware-validation-kit.md)'s record of that date and recorded under README
execution rule 1: [Tier 03](TIER-03-kv-cache-maturity.md) begins while this tier and Tier 02F stay open.
- This tier's item 3 choice (the 2026-11-06 finding, options (a) to (c)) is due before Tier 03's step 3, the block-table kernel.
  Option (a) splits each prefill window's attention into bounded launches, and Tier 03's items 1 and 7 change how the attention
  kernels read the KV, so whichever lands second carries the other.
- When this tier resumes, its standing gate names which build is its pre-tier jar if Tier 03's work is in the tree by then. Its
  pre-tier kit record at `80ec949e` is unaffected.
- Nothing else here changes.

### 2026-11-10: item 3 decided (owner decision); it lands in Tier 03

**Owner decision 2026-11-10: option (a)** of the 2026-11-06 finding: bound every launch's work on every OS. Recorded with Tier
03's decision 1 (option (d)), which makes Tier 03's block-table attention the CPU attention's read of the host pages, so Tier
03's step 3 changes no GPU kernel.
- The split (prefill attention in launches whose rows times keys stay under a budget, the GEMM in launches of bounded rows) lands
  in [Tier 03](TIER-03-kv-cache-maturity.md) before its step 6, item 7, the first step that changes how the GPU attention
  kernels read the KV. It is recorded there under README execution rule 9, with `gpu.kernel-timing` extended to the longest
  window and the standing throughput gate applied to it.
- This tier keeps the Windows half of item 3: the longest single kernel timed against the watchdog on Windows, launch overhead,
  and the free-memory accounting. Those wait on the Windows runs, as before.
- The 2026-11-09 note above ("due before Tier 03's step 3") is superseded: the choice is made, and the work is due before Tier
  03's step 6. README execution rule 1 carries the same note.
- Nothing else here changes.

### 2026-11-10: the launcher on Windows, from a tester's re-runs (recorded in Tier 02F)

A tester ran the kit's `quick` suite twice on Windows 10 at `f234e90c`, whose code is `e16748ad`'s, after the 2026-11-09
`run.bat` fix. The launcher probe passed 5 of 5 in both runs. [Tier 02F](TIER-02F-hardware-validation-kit.md)'s record of
this date has the details.
- Two criteria are ticked, with dated notes: launcher parity (`juno.bat local` answered and exited cleanly; no `[DBG]` line; the
  registry untouched) and the kit-probe criterion, whose launcher half waited for a Windows run.
- The same runs found one failure on a three-node tensor-parallel cluster on the CPU (Tier 02F's finding 1). It concerns item 4
  (the rest of Juno on Windows) only if its cause turns out to be Windows-specific. Nothing is recorded against an item here
  until that cause is known.
- **Exit criteria:** 2 of 10. Steps 2 and 3 are still the owner's.
