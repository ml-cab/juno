# Remote kit run: windows-amd64-12th-gen-core-i5-1240p (20261110T073556Z)

**Written by `juno-kit ingest`; reviewed 2026-11-10 (notes below).** Unpinned. A remote run's readings are same-run
comparisons on that host; none is a throughput gate.

**Host.** Windows 10 10.0 amd64; 12th Gen Intel(R) Core(TM) i5-1240P, 12 cores, 16 threads, 15 GiB RAM; JVM vectors: integer 256-bit, float 256-bit; GPU: none

**Build.** Juno `f234e90cfb3c56c10527cdc75214cefca8f7c8dc` (modified tree); kit 0.1.2, kit jar sha256 `5c2c6155a0a993b40fe2d2051abe8fbe9318bfa01b02ce9963388e87068841c6`.

**Run.** Suite `quick`, 20261110T070210Z to 20261110T073556Z; 12 probes: 4 DONE, 1 NOISY, 7 PASS.

## Review notes (2026-11-10)

- **How this entry was made.** From the kit's own zip, as the tester returned it: 69 files, `SUMMARY.txt` and `report.html`
  among them. The kit scrubbed it on the tester's host: no user name, 299 home paths replaced, no host name, no JFR recording.
- **Commit: `f234e90c`, "modified tree".** The kit records git's flag (`git status --porcelain --untracked-files=no` not
  empty), not which files changed. `f234e90c` tracks a results zip at the repository root, and deleting or overwriting that
  file sets the flag. The zip cannot confirm that this is what happened. Every probe's result equals run `20261110T013914Z`'s
  and the reference's, so whatever changed did not change a result. The code otherwise is `e16748ad`'s, as in that entry.
- **The whole suite passed: 93 of 93 items equal the reference** (`80ec949e`, Linux, Xeon E5-1650 v2), every greedy reply
  byte-equal, across OS, CPU and commit. The live checks passed 10 of 10, check 8 (tensor-parallel greedy determinism)
  included, which failed in the run before this one. The launcher probe passed 5 of 5, and the kit log reads "Keeping this
  machine awake for the run (SetThreadExecutionState)."
- **On battery** (discharging, 95% at the start). Against the run before, on mains: calibration 0.588 ms against 0.478 ms,
  prefill 49.9 and 50.1 t/s against 55.5 and 53.2 at 128 and 512 tokens, generation 17.9 and 17.6 t/s against 17.8 and 18.0.
  Published, not compared.
- **Noise.** `cpu.kernel-width` read NOISY at 9%, 9% and 8% other CPU while every other probe read 3% to 7%: the sampler
  finding in that entry. `cpu.throughput.tinyllama.512` re-ran once for its own spread (generation 17%), then read DONE.
- **Time.** The first estimate was 1319 s (range 1121 to 1583); the run took 2026 s, 0.65x.
- **128 against 256 bits** (`cpu.kernel-width`, 128-bit time over 256-bit time; unpinned hybrid laptop CPU, on battery):

  | Shape, threads | 128-bit median (ms) | 256-bit median (ms) | Ratios of attempts 1, 2, 3 |
  |---|---|---|---|
  | Q4_K 14336x4096, 1 | 11.20 | 8.64 | 1.27x, 1.25x, 1.30x |
  | Q4_K 14336x4096, 16 | 1.80 | 1.47 | 1.23x, 1.20x, 1.23x |
  | Q6_K 4096x14336, 1 | 23.85 | 19.20 | 1.20x, 1.19x, 1.24x |
  | Q6_K 4096x14336, 16 | 4.17 | 3.36 | 1.24x, 1.23x, 1.24x |

  Only this timing ran at 256 bits; every live, quality and greedy probe ran at 128 bits.
- **Mean NLL 2.397478341905886**, as in the other two Windows runs: a CR LF checkout's reading (no `.gitattributes` at
  `f234e90c`).

## Probes

| Probe | Status | Time | Attempts | Other CPU | Note |
|---|---|---|---|---|---|
| `calibration` | DONE | 0:10 | 1 | 5% |  |
| `env` | DONE | 0:01 | 1 | 0% |  |
| `cpu.kernel-width` | NOISY | 5:46 | 3 | 8% | attempt 1 re-run: other processes used 9% of the CPU. attempt 2 re-run: other processes used 9% of the CPU. still: other... |
| `cpu.live.tinyllama` | PASS | 3:18 | 1 | 5% |  |
| `cpu.throughput.tinyllama.128` | DONE | 0:47 | 1 | 3% |  |
| `cpu.throughput.tinyllama.512` | DONE | 3:12 | 2 | 4% | attempt 1 re-run: generation t/s spread 17% > 15%; other processes used 5% of the CPU. |
| `consistency.tinyllama` | PASS | 1:39 | 1 | 6% |  |
| `grammar.tinyllama` | PASS | 1:58 | 1 | 7% |  |
| `long-prompt.tinyllama` | PASS | 10:20 | 1 | 6% |  |
| `cpu.hot-path.tinyllama-1.1b` | PASS | 1:16 | 1 | 5% |  |
| `quality.tinyllama-1.1b` | PASS | 5:01 | 1 | 4% |  |
| `launcher.tinyllama` | PASS | 0:18 | 1 | 5% |  |

## Readings

Each reading's repetitions on this host. They are not compared with the reference host.

| Probe | Reading | Unit | Median | Min | Max | n | Status |
|---|---|---|---|---|---|---|---|
| `calibration` | q4k_decode_ms_per_call | ms | 0.588238 | 0.55859 | 0.612797 | 5 | DONE |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 128-bit | ms/call | 11.1985 | 11.1444 | 11.6012 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 256-bit | ms/call | 8.63835 | 8.56579 | 9.54434 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 128-bit | ms/call | 1.80024 | 1.68607 | 1.94512 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 256-bit | ms/call | 1.46956 | 1.46441 | 1.48313 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 128-bit | ms/call | 23.8515 | 23.429 | 24.0351 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 256-bit | ms/call | 19.2038 | 18.7969 | 19.3505 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 128-bit | ms/call | 4.16953 | 4.12696 | 4.39532 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 256-bit | ms/call | 3.36099 | 3.32981 | 3.36673 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | 49.8934 | 46.2832 | 50.0372 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation t/s | t/s | 17.8626 | 17.4552 | 18.0029 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation allocation per token in recording | B/token | 402440 | 323582 | 427436 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill t/s | t/s | 50.0688 | 48.1693 | 50.1267 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation t/s | t/s | 17.6277 | 17.5679 | 17.8801 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | 402697 | 367778 | 421941 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `quality.tinyllama-1.1b` | mean NLL, integer kernels | nats | 2.39748 | 2.39748 | 2.39748 | 1 | PASS |
| `quality.tinyllama-1.1b` | mean NLL, float-activation kernels | nats | 2.39823 | 2.39823 | 2.39823 | 1 | PASS |
| `quality.tinyllama-1.1b` | argmax agreement, first 512 positions | % | 99.6094 | 99.6094 | 99.6094 | 1 | PASS |

## Correctness against the reference run

Reference: `20261104T173942Z-kit-reference` (Juno `80ec949e`). 93 items compared: 93 equal or within the bound.

## Coverage on this host

**GPU backend:** none. Juno found no usable GPU, so no GPU probe ran.

| Item | Title | Probes run here |
|---|---|---|
| surface:1 | CPU inference | 2 DONE, 1 NOISY, 3 PASS |
| surface:2 | CUDA GPU inference | none |
| surface:3 | ROCm GPU inference | none |
| surface:4 | Static schedule | 1 PASS |
| surface:5 | Continuous schedule | 1 PASS |
| surface:6 | Single-node local mode | 3 PASS |
| surface:7 | Pipeline-parallel cluster | 1 PASS |
| surface:8 | Tensor-parallel cluster | 1 PASS |
| surface:9 | LoRA training | none |
| surface:10 | LoRA playback | 1 PASS |
| surface:11 | Vision | none |
| surface:12 | OpenAI-compatible REST surface | 3 PASS |
| surface:13 | Native REST surface | 1 PASS |
| surface:14 | CLI direct usage | 2 PASS |
| surface:15 | JVM embedding facade | 2 DONE, 2 PASS |
| family:llama | Llama family | 2 PASS |
| family:mistral | Mistral (Llama-family handler) | none |
| family:tinyllama | TinyLlama (Llama-family handler) | 1 PASS |
| family:qwen2 | Qwen2 family | none |
| family:qwen2.5 | Qwen2.5 (Llama-family handler) | none |
| family:phi2 | Phi-2 | none |
| family:phi3 | Phi-3 | none |
| family:qwen3 | Qwen3 | none |
| family:qwen3moe | Qwen3 mixture of experts | none |
| script:compare-llama-cpp.sh | Throughput, Juno lanes | 2 DONE |
| script:compare-lora.sh | LoRA training and playback | none |
| script:compare-mixed-prefill.sh | Mixed prefill | none |
| script:compare-parallel.sh | Static micro-batching under load | none |
| script:compare-prefill-batch.sh | Prefill batch size | none |
| script:compare-schedule.sh | Static against continuous | none |
| script:compare-vision.sh | Vision latency and decode | none |
| script:smoke-consistency.sh | Batched equals single requests | 1 PASS |
| script:smoke-context-policy.sh | Context policy | none |
| script:smoke-cpu-hot-path.sh | CPU hot path | 1 PASS |
| script:smoke-cpu-kernels.sh | CPU kernels against the previous build | 1 PASS |
| script:smoke-gpu-residency.sh | GPU residency | none |
| script:smoke-grammar.sh | Constrained decoding | 1 PASS |
| script:smoke-long-prompt-prefill.sh | Long-prompt prefill | 1 PASS |
| script:smoke-packed-kquant-matmul.sh | Packed K-quant matmul | none |
| script:smoke-tools.sh | Tool calling | none |
| feature:grammar | Constrained decoding: response_format, x_juno_grammar, the CLI grammar file | 1 PASS |
| feature:tools | Tool calling, and its refusal on templates without tool support | none |
| feature:context-shift | Context shift opt-in, decode-time shift and the limit error | none |
| feature:long-prompt | Long prompts, streamed text equal to blocking | 1 PASS |
| feature:static-batching | --parallel micro-batching: batched replies equal single ones | 1 PASS |
| feature:session-reuse | x_juno_session_id prefix reuse | none |
| feature:mixed-prefill | Prefill chunks inside continuous decode steps | none |
| feature:prefill-batch | --prefill-batch | none |
| feature:gpu-residency | Device-resident decode | none |
| feature:batched-gpu-decode | A decode batch runs in the device region: replies equal single requests, the --parallel ratio read | none |
| feature:packed-kquant | Packed K-quant matmul | none |
| feature:cpu-integer-kernels | CPU integer kernels and their vector width | 1 NOISY, 1 PASS |
| feature:quality | Mean NLL per model | 1 PASS |
| feature:throughput | Prefill and generation throughput, allocation and GC | 2 DONE |
| feature:lora-merge | A trained adapter merged into a GGUF and served | none |
| feature:unit-tests | Every module's unit tests and the GPU test groups | none |
| feature:kit-environment | The host's description and the calibration that scales estimates | 2 DONE |
| feature:cuda-libraries | The CUDA driver, runtime and cuBLAS found per operating system, and where from | 1 DONE |
| feature:gpu-kernel-watchdog | The longest single GPU kernels at the sweep models' shapes, against the display-driver watchdog | none |
| feature:line-endings | Text reads the same on every OS: LF checkouts of fixtures, the quality text and the PTX; CR LF grammar and training file... | 1 PASS |
| feature:local-cluster-lifecycle | A local cluster's forked nodes listen on loopback and stop through their shutdown hooks | 1 PASS |
| feature:windows-launcher-parity | juno.bat and scripts/run.bat take every subcommand, flag and environment override run.sh takes and pass the same options... | 1 PASS |

## Files

- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.
- `probes/`: each probe's report.
