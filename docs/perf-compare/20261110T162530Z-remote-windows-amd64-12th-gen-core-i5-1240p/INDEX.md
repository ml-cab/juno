# Remote kit run: windows-amd64-12th-gen-core-i5-1240p (20261110T020641Z)

**Written by `juno-kit ingest`; reviewed 2026-11-10 (notes below).** Unpinned. A remote run's readings are same-run
comparisons on that host; none is a throughput gate.

**Host.** Windows 10 10.0 amd64; 12th Gen Intel(R) Core(TM) i5-1240P, 12 cores, 16 threads, 15 GiB RAM; JVM vectors: integer 256-bit, float 256-bit; GPU: none

**Build.** Juno `f234e90cfb3c56c10527cdc75214cefca8f7c8dc`; kit 0.1.2, kit jar sha256 `a1fab52698c7116fe535746a5844aaebdd0cd5b7dfe1357746b570710589c524`.

**Run.** Suite `quick`, 20261110T013914Z to 20261110T020641Z; 12 probes: 4 DONE, 1 FAIL, 1 NOISY, 6 PASS.

## Review notes (2026-11-10)

- **How this entry was made.** From the kit's own zip, as the tester returned it: 68 files, `SUMMARY.txt` and `report.html`
  among them. The kit scrubbed it on the tester's host: no user name, 281 home paths replaced, no JFR recording. The scrubber
  knew the host's name and replaced none, and no host name appears in the text.
- **Commit.** `f234e90c` on a clean tree. Its tree equals `e16748ad`'s except for a results zip at the repository root, which
  the owner removed from history afterwards, so the code is `e16748ad`'s. The reference run is at `80ec949e`, so the comparison
  crosses commits as well as OS and CPU.
- **The three defects of the first Windows run are fixed on Windows.** The packer finished on the code-page logs. The launcher
  probe passed 5 of 5: `juno.bat local --api-port` answered a chat request with `--temperature 0` and exited with code 0. The kit
  log reads "Keeping this machine awake for the run (SetThreadExecutionState).", so the call held.
- **The one row that differs: live check 8.** Its second request failed 3 s into prefill with `RuntimeException: Tensor-parallel
  forward pass failed: UNAVAILABLE: io exception`, which means the gRPC connection to a node closed. Check 7 and check 8's first
  request had answered on the same three-node cluster seconds earlier. The zip cannot say why: the forked node JVMs discard
  their stderr unless `JUNO_VERBOSE` is set, and a JVM's fatal-error file goes to the working directory, which is the tester's
  checkout, and the kit does not pack it. The probe can use up to 16 GiB of heap (the coordinator's 4 GiB and three nodes at
  4 GiB) on 15 GiB of RAM, and this run started with 3.1 GiB of free disk for the page file. The other two Windows runs, at
  5.3 and 30.3 GiB free, passed check 8. That is a correlation over three runs, not a cause.
- **The coverage table counts the whole live probe.** `surface:6`, `surface:7`, `surface:8` and `feature:local-cluster-lifecycle`
  show `1 FAIL` for check 8 alone. Checks 1 to 6, the pipeline-parallel cluster's, passed, as did checks 9 and 10.
- **Noise.** Calibration and both throughput probes read DONE on the first attempt, at 3% to 4% other CPU, so a Windows laptop
  can read under the 5% limit. `cpu.kernel-width` read NOISY at 8%, 9% and 16% while every other probe read 3% to 6%. The
  kit's sampler counts the last partial second of each child JVM that exits as other programs' CPU, and this probe starts 24
  multi-threaded child JVMs per attempt. That comes to about 6% here, which accounts for the excess (finding in Tier 02F's
  record of 2026-11-10).
- **Time.** The first estimate was 1071 s (range 911 to 1286); the run took 1647 s, 0.65x. `cpu.kernel-width`'s two re-runs
  account for about 220 s of the difference.
- **128 against 256 bits** (`cpu.kernel-width`, 128-bit time over 256-bit time; unpinned hybrid laptop CPU):

  | Shape, threads | 128-bit median (ms) | 256-bit median (ms) | Ratios of attempts 1, 2, 3 |
  |---|---|---|---|
  | Q4_K 14336x4096, 1 | 8.72 | 6.85 | 1.20x, 1.21x, 1.27x |
  | Q4_K 14336x4096, 16 | 2.48 | 2.05 | 1.21x, 1.25x, 1.21x |
  | Q6_K 4096x14336, 1 | 22.93 | 18.56 | 1.24x, 1.25x, 1.24x |
  | Q6_K 4096x14336, 16 | 4.57 | 3.67 | 1.25x, 1.25x, 1.24x |

  Only this timing ran at 256 bits. Every live, quality and greedy probe ran at 128 bits, the width `auto` selects.
- **Mean NLL 2.397478341905886**, the same 16 digits as the first Windows run and this host's CR LF reading
  (`20261107T070530Z-nll-line-endings`). `f234e90c` has no `.gitattributes`, so the checkout wrote the quality text with CR LF.

## Probes

| Probe | Status | Time | Attempts | Other CPU | Note |
|---|---|---|---|---|---|
| `calibration` | DONE | 0:09 | 1 | 3% |  |
| `env` | DONE | 0:01 | 1 | 0% |  |
| `cpu.kernel-width` | NOISY | 5:33 | 3 | 16% | attempt 1 re-run: Q4_K 14336x4096 threads 1 256-bit spread 18% > 15%; other processes used 8% of the CPU. attempt 2 re-r... |
| `cpu.live.tinyllama` | FAIL | 3:08 | 1 | 8% | failed: 8. tensor-parallel greedy determinism |
| `cpu.throughput.tinyllama.128` | DONE | 0:45 | 1 | 4% |  |
| `cpu.throughput.tinyllama.512` | DONE | 1:29 | 1 | 4% |  |
| `consistency.tinyllama` | PASS | 1:18 | 1 | 4% |  |
| `grammar.tinyllama` | PASS | 1:17 | 1 | 3% |  |
| `long-prompt.tinyllama` | PASS | 7:21 | 1 | 4% |  |
| `cpu.hot-path.tinyllama-1.1b` | PASS | 1:15 | 1 | 3% |  |
| `quality.tinyllama-1.1b` | PASS | 4:57 | 1 | 4% |  |
| `launcher.tinyllama` | PASS | 0:15 | 1 | 6% |  |

## Readings

Each reading's repetitions on this host. They are not compared with the reference host.

| Probe | Reading | Unit | Median | Min | Max | n | Status |
|---|---|---|---|---|---|---|---|
| `calibration` | q4k_decode_ms_per_call | ms | 0.477794 | 0.475598 | 0.482893 | 5 | DONE |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 128-bit | ms/call | 8.71943 | 8.62917 | 9.15064 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 256-bit | ms/call | 6.85065 | 6.52967 | 7.13647 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 128-bit | ms/call | 2.47823 | 2.26835 | 2.6291 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 256-bit | ms/call | 2.05054 | 1.83697 | 2.1708 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 128-bit | ms/call | 22.9342 | 20.1257 | 24.4871 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 256-bit | ms/call | 18.564 | 16.7572 | 18.9256 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 128-bit | ms/call | 4.56566 | 4.55502 | 4.57656 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 256-bit | ms/call | 3.67052 | 3.60765 | 3.69522 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | 55.5204 | 51.0672 | 56.0093 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation t/s | t/s | 17.8007 | 17.6292 | 18.0148 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation allocation per token in recording | B/token | 426573 | 282650 | 429977 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill t/s | t/s | 53.1903 | 51.4347 | 53.4642 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation t/s | t/s | 18.024 | 17.972 | 18.1264 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | 407897 | 294259 | 442698 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `quality.tinyllama-1.1b` | mean NLL, integer kernels | nats | 2.39748 | 2.39748 | 2.39748 | 1 | PASS |
| `quality.tinyllama-1.1b` | mean NLL, float-activation kernels | nats | 2.39823 | 2.39823 | 2.39823 | 1 | PASS |
| `quality.tinyllama-1.1b` | argmax agreement, first 512 positions | % | 99.6094 | 99.6094 | 99.6094 | 1 | PASS |

## Correctness against the reference run

Reference: `20261104T173942Z-kit-reference` (Juno `80ec949e`). 93 items compared: 92 equal or within the bound; listed below.

| Probe | Item | Here | Reference | Verdict |
|---|---|---|---|---|
| `cpu.live.tinyllama` | 8. tensor-parallel greedy determinism | FAIL | PASS | fails here, passes in the reference |

## Coverage on this host

**GPU backend:** none. Juno found no usable GPU, so no GPU probe ran.

| Item | Title | Probes run here |
|---|---|---|
| surface:1 | CPU inference | 2 DONE, 1 FAIL, 1 NOISY, 2 PASS |
| surface:2 | CUDA GPU inference | none |
| surface:3 | ROCm GPU inference | none |
| surface:4 | Static schedule | 1 PASS |
| surface:5 | Continuous schedule | 1 PASS |
| surface:6 | Single-node local mode | 1 FAIL, 2 PASS |
| surface:7 | Pipeline-parallel cluster | 1 FAIL |
| surface:8 | Tensor-parallel cluster | 1 FAIL |
| surface:9 | LoRA training | none |
| surface:10 | LoRA playback | 1 PASS |
| surface:11 | Vision | none |
| surface:12 | OpenAI-compatible REST surface | 3 PASS |
| surface:13 | Native REST surface | 1 PASS |
| surface:14 | CLI direct usage | 2 PASS |
| surface:15 | JVM embedding facade | 2 DONE, 2 PASS |
| family:llama | Llama family | 1 FAIL, 1 PASS |
| family:mistral | Mistral (Llama-family handler) | none |
| family:tinyllama | TinyLlama (Llama-family handler) | 1 FAIL |
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
| feature:local-cluster-lifecycle | A local cluster's forked nodes listen on loopback and stop through their shutdown hooks | 1 FAIL |
| feature:windows-launcher-parity | juno.bat and scripts/run.bat take every subcommand, flag and environment override run.sh takes and pass the same options... | 1 PASS |

## Files

- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.
- `probes/`: each probe's report.
