# Tier 02G: CUDA on Windows

Status: not started (added 2026-10-13 by owner decision; runs after Tier 02F)
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
- [ ] `juno.bat` and `scripts/run.bat` are at parity with `run.sh`, with no `[DBG]` output and no registry
      write.
- [ ] ROCm on Windows and CUDA on macOS fail closed, tested.
- [ ] Every Linux-only assumption reachable on Windows is fixed or fails closed (repo-wide, README rule 8).
- [ ] Kit probes for this tier's behaviour are added: the kernel-timing probe, and the launcher probe on
      Windows.
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
