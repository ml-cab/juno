# Remote kit run: windows-amd64-12th-gen-core-i5-1240p (20261110T000353Z)

**Written by `juno-kit ingest`; reviewed 2026-11-09 (notes below).** Unpinned. A remote run's readings are same-run
comparisons on that host; none is a throughput gate.

**Host.** Windows 10 10.0 amd64; 12th Gen Intel(R) Core(TM) i5-1240P, 12 cores, 16 threads, 15 GiB RAM; JVM vectors: integer 256-bit, float 256-bit; GPU: none

**Build.** Juno `616f577fa8d63272dc195db6cc06113a04da023c`; kit 0.1.2, kit jar sha256 `4da1a843a5b5972c0bd5bd73ee934bd0451296feb2de79f7b724f5bf35c47353`.

**Run.** Suite `quick`, 20261109T232611Z to 20261110T000353Z; 12 probes: 1 DONE, 1 FAIL, 4 NOISY, 6 PASS.

## Review notes (2026-11-09)

- **How this entry was made.** The kit's packer stopped at the run's first log that was not UTF-8 (`cpu.live.tinyllama.log`,
  `MalformedInputException`). On Windows a probe JVM's redirected output is in the machine's code page, and the engine's log
  lines hold an em dash, byte 0x97. The zip the kit left holds 37 of the run's 74 text files and no `SUMMARY.txt` or report; in
  those 37 no user name remains and 158 home paths were replaced. The tester sent the results folder instead. This entry
  comes from that folder, packed on the reference host with the kit's own scrubber, given the tester's user name and home path,
  reading the probe logs as windows-1252: no user name in the 74 files, 367 home paths replaced. The scrubber did not know the
  host's name on this side. No host name appears in the text: a search for host-name shapes found none, and the kit's own run
  of the scrubber, which knew it, replaced none in its 37 files. The JFR recordings in the folder are not in this entry.
- **Commit.** The run is at `616f577f`, the reference run at `80ec949e`, so the comparison below crosses commits as well as OS
  and CPU. 80 of 82 checks and 8 of 8 greedy replies equal the reference.
- **The two rows that differ are one defect.** `scripts/run.bat` had no space between `"--temperature"` and `(` in its `local`
  and `cluster` parsers (and `"--mmproj-path"` in `local`), so `cmd.exe` never matched the flag and refused it as unknown. The
  launcher probe's server never started, so its other check could not pass. Fixed after this run; not yet re-run on Windows.
- **The mean NLL rows are not like for like.** The checkout wrote the quality text with CR LF line ends, so this run scored a
  different text from the reference's. Its 2.397478341905886 nats equals, to all 16 digits, the reference host's own CR LF reading
  (`20261107T070530Z-nll-line-endings`), which is the real comparison: the same integer-kernel NLL on Windows and Linux, on two
  CPU generations. "Within the bound" against the LF reading says nothing here.
- **Every timed probe is marked NOISY for other programs' CPU use**, 8% to 15% against the 5% limit on all three attempts, so
  the three-attempt re-runs ran as specified. The readings' own spreads are 2.5% to 4.7% for calibration and throughput, and
  1.8% to 13.2% for the kernel widths, except the 16-thread Q4_K reading at 256 bits (31.4%).
- **Time.** The first estimate was 1632 s (range 1387 to 1958); the run took 2262 s, 0.72x. The re-runs on four probes are most
  of the difference.
- **128 against 256 bits** (`cpu.kernel-width`, ms per call, 128-bit time over 256-bit time; every value on an unpinned hybrid
  laptop CPU with other programs running):

  | Shape, threads | 128-bit median | 256-bit median | Ratio, this attempt | Ratios of attempts 1, 2, 3 |
  |---|---|---|---|---|
  | Q4_K 14336x4096, 1 | 10.15 | 7.97 | 1.27x | 1.22x, 1.16x, 1.27x |
  | Q4_K 14336x4096, 16 | 2.23 | 1.95 | 1.14x | 1.20x, 1.22x, 1.14x |
  | Q6_K 4096x14336, 1 | 20.00 | 16.88 | 1.18x | 1.23x, 1.21x, 1.18x |
  | Q6_K 4096x14336, 16 | 5.05 | 3.92 | 1.29x | 1.26x, 1.25x, 1.29x |

  Only this timing ran at 256 bits. Every live, quality and greedy probe ran at 128 bits, the width `auto` selects.

## Probes

| Probe | Status | Time | Attempts | Other CPU | Note |
|---|---|---|---|---|---|
| `calibration` | NOISY | 0:28 | 3 | 9% | attempt 1 re-run: other processes used 11% of the CPU. attempt 2 re-run: other processes used 8% of the CPU. still: othe... |
| `env` | DONE | 0:01 | 1 | 0% |  |
| `cpu.kernel-width` | NOISY | 5:35 | 3 | 14% | attempt 1 re-run: Q4_K 14336x4096 threads 16 256-bit spread 17% > 15%; other processes used 13% of the CPU. attempt 2 re... |
| `cpu.live.tinyllama` | PASS | 3:33 | 1 | 11% |  |
| `cpu.throughput.tinyllama.128` | NOISY | 2:41 | 3 | 10% | attempt 1 re-run: other processes used 9% of the CPU. attempt 2 re-run: other processes used 9% of the CPU. still: other... |
| `cpu.throughput.tinyllama.512` | NOISY | 5:22 | 3 | 11% | attempt 1 re-run: other processes used 10% of the CPU. attempt 2 re-run: other processes used 11% of the CPU. still: oth... |
| `consistency.tinyllama` | PASS | 1:26 | 1 | 10% |  |
| `grammar.tinyllama` | PASS | 1:21 | 1 | 10% |  |
| `long-prompt.tinyllama` | PASS | 9:13 | 1 | 9% |  |
| `cpu.hot-path.tinyllama-1.1b` | PASS | 1:26 | 1 | 8% |  |
| `quality.tinyllama-1.1b` | PASS | 6:16 | 1 | 10% |  |
| `launcher.tinyllama` | FAIL | 0:02 | 1 | 19% | failed: local mode with --api-port answers a chat request; the launcher exits cleanly on exit |

## Readings

Each reading's repetitions on this host. They are not compared with the reference host.

| Probe | Reading | Unit | Median | Min | Max | n | Status |
|---|---|---|---|---|---|---|---|
| `calibration` | q4k_decode_ms_per_call | ms | 0.687901 | 0.681904 | 0.699892 | 5 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 128-bit | ms/call | 10.1453 | 10.0009 | 10.2178 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 256-bit | ms/call | 7.96971 | 7.47622 | 8.07521 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 128-bit | ms/call | 2.23156 | 2.20568 | 2.26073 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 16 256-bit | ms/call | 1.94942 | 1.81246 | 2.42544 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 128-bit | ms/call | 19.995 | 19.87 | 20.3491 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 256-bit | ms/call | 16.8826 | 15.9692 | 18.2021 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 128-bit | ms/call | 5.05343 | 4.98124 | 5.07455 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 16 256-bit | ms/call | 3.92094 | 3.91407 | 3.99453 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | 44.6249 | 42.5445 | 44.6617 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | generation t/s | t/s | 14.5806 | 14.5147 | 14.9579 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | generation allocation per token in recording | B/token | 395585 | 386121 | 447681 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | NOISY |
| `cpu.throughput.tinyllama.512` | prefill t/s | t/s | 42.1288 | 41.5338 | 42.5881 | 3 | NOISY |
| `cpu.throughput.tinyllama.512` | generation t/s | t/s | 14.9044 | 14.6014 | 15.188 | 3 | NOISY |
| `cpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | 389568 | 329359 | 447690 | 3 | NOISY |
| `cpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | NOISY |
| `cpu.throughput.tinyllama.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | NOISY |
| `quality.tinyllama-1.1b` | mean NLL, integer kernels | nats | 2.39748 | 2.39748 | 2.39748 | 1 | PASS |
| `quality.tinyllama-1.1b` | mean NLL, float-activation kernels | nats | 2.39823 | 2.39823 | 2.39823 | 1 | PASS |
| `quality.tinyllama-1.1b` | argmax agreement, first 512 positions | % | 99.6094 | 99.6094 | 99.6094 | 1 | PASS |

## Correctness against the reference run

Reference: `20261104T173942Z-kit-reference` (Juno `80ec949e`). 93 items compared: 91 equal or within the bound; listed below.

| Probe | Item | Here | Reference | Verdict |
|---|---|---|---|---|
| `launcher.tinyllama` | local mode with --api-port answers a chat request | FAIL | PASS | fails here, passes in the reference |
| `launcher.tinyllama` | the launcher exits cleanly on exit | FAIL | PASS | fails here, passes in the reference |

## Coverage on this host

**GPU backend:** none. Juno found no usable GPU, so no GPU probe ran.

| Item | Title | Probes run here |
|---|---|---|
| surface:1 | CPU inference | 3 NOISY, 3 PASS |
| surface:2 | CUDA GPU inference | none |
| surface:3 | ROCm GPU inference | none |
| surface:4 | Static schedule | 1 PASS |
| surface:5 | Continuous schedule | 1 PASS |
| surface:6 | Single-node local mode | 1 FAIL, 2 PASS |
| surface:7 | Pipeline-parallel cluster | 1 PASS |
| surface:8 | Tensor-parallel cluster | 1 PASS |
| surface:9 | LoRA training | none |
| surface:10 | LoRA playback | 1 PASS |
| surface:11 | Vision | none |
| surface:12 | OpenAI-compatible REST surface | 1 FAIL, 2 PASS |
| surface:13 | Native REST surface | 1 PASS |
| surface:14 | CLI direct usage | 1 FAIL, 1 PASS |
| surface:15 | JVM embedding facade | 2 NOISY, 2 PASS |
| family:llama | Llama family | 2 PASS |
| family:mistral | Mistral (Llama-family handler) | none |
| family:tinyllama | TinyLlama (Llama-family handler) | 1 PASS |
| family:qwen2 | Qwen2 family | none |
| family:qwen2.5 | Qwen2.5 (Llama-family handler) | none |
| family:phi2 | Phi-2 | none |
| family:phi3 | Phi-3 | none |
| family:qwen3 | Qwen3 | none |
| family:qwen3moe | Qwen3 mixture of experts | none |
| script:compare-llama-cpp.sh | Throughput, Juno lanes | 2 NOISY |
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
| feature:throughput | Prefill and generation throughput, allocation and GC | 2 NOISY |
| feature:lora-merge | A trained adapter merged into a GGUF and served | none |
| feature:unit-tests | Every module's unit tests and the GPU test groups | none |
| feature:kit-environment | The host's description and the calibration that scales estimates | 1 DONE, 1 NOISY |
| feature:cuda-libraries | The CUDA driver, runtime and cuBLAS found per operating system, and where from | 1 DONE |
| feature:gpu-kernel-watchdog | The longest single GPU kernels at the sweep models' shapes, against the display-driver watchdog | none |
| feature:line-endings | Text reads the same on every OS: LF checkouts of fixtures, the quality text and the PTX; CR LF grammar and training file... | 1 PASS |
| feature:local-cluster-lifecycle | A local cluster's forked nodes listen on loopback and stop through their shutdown hooks | 1 PASS |
| feature:windows-launcher-parity | juno.bat and scripts/run.bat take every subcommand, flag and environment override run.sh takes and pass the same options... | 1 FAIL |

## Files

- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.
- `probes/`: each probe's report.
