# Remote kit run: linux-amd64-core-i7-8550u-geforce-mx150 (20261110T144459Z)

**Written by `juno-kit ingest`; reviewed 2026-11-11 (notes below).** Unpinned. A remote run's readings are same-run
comparisons on that host; none is a throughput gate.

**Host.** Linux 6.17.0-19-generic amd64; Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 4 cores, 8 threads, 7 GiB RAM; JVM vectors: integer 256-bit, float 256-bit; GPU: NVIDIA GeForce MX150

**Build.** Juno `9dda5f7d9248da634effb83d4a14bdcccb8bd89f` (modified tree); kit 0.1.2, kit jar sha256 `3a84952eb367ce3c6c4c56e0460edc7e1e48df419276d026a4df11cd4be3755a`.

**Run.** Suite `quick`, 20261110T142832Z to 20261110T144459Z; 11 probes: 6 DONE, 1 NOISY, 4 PASS.

## Review notes (2026-11-11)

- **How this entry was made.** From the kit's own zip, which a second tester returned inside their whole results folder.
  The kit scrubbed the zip on the tester's host: no user name, no home path, no host name, no MAC or IP address, no JFR
  recording. The folder itself is not scrubbed and stays out of the tree.
- **Commit: `9dda5f7d`, "modified tree".** The kit records git's flag, not which files changed.
- **Correctness: 13 of 13 items equal the reference** (`80ec949e`, Linux, Xeon E5-1650 v2 and GTX 1080), the launcher's
  greedy reply among them. Packed K-quant passed on both schedules on this Pascal laptop GPU (sm_61): 512 greedy tokens,
  streamed text equal to blocking text, no dequantizing fallback.
- **Every throughput reading is empty, and the four `DONE`s say nothing.** In all 24 repetitions the JFR spans add up to
  more than the request's wall-clock time: prefill 1.1037 to 1.1066 times, generation 1.0826 to 1.1018. The kit withheld
  every repetition, as its guard requires, and still reported `DONE`. Cause: on this CPU, with an invariant TSC, the JVM
  converts JFR's counter ticks at the brand string's 1.80 GHz, and the counter runs at about 1.99 GHz. The kernel's
  clocksource is `tsc`, so the kit's rule kept JFR's fast clock. Recorded as finding 3 of
  [Tier 02F](../../gap-closure-plan/TIER-02F-hardware-validation-kit.md)'s 2026-11-10 record; the kit now checks JFR's
  clock once per host and marks a reading probe with no valid repetition `NOISY` with the reason (2026-11-11 record).
  Wall-clock times from the same facts, not readings: CPU 128-token prefill requests 6.3 to 6.5 s, 512-token 26.2 to
  27.3 s, 64 generated tokens 11.2 to 11.8 s; GPU 512-token prefill 1.30 to 1.33 s, 2048-token 7.13 to 7.23 s, 64
  generated tokens 2.03 to 2.04 s.
- **10 probes not run for memory.** The machine's 8 GB shows as 7.66 GiB, and `Requirements` compares bytes with whole
  GiB, so every probe that needs 8 GiB was skipped: the live checks, consistency, long prompts and grammar on both
  backends, residency, the hot path and quality.
- **GPU kernel timing passed:** the longest single kernel is 268.75 ms (Mistral 7B prefill attention over 2048 keys),
  under the 500 ms limit.
- **128 against 256 bits** (`cpu.kernel-width`, 128-bit time over 256-bit time; NOISY at 6% to 7% other CPU, on a kit
  that predates the sampler fix): Q4_K 14336x4096 1.42x on 1 thread and 1.58x on 8; Q6_K 4096x14336 1.22x and 1.41x.
  Recorded, not scored; `auto` stays at 128 bits by the owner decision of 2026-11-09.
- **Outside the kit:** the tester's own `mvn` build failed 2 of 1315 tests in `node`, `KQuantGemvRowsTest` for Q5_K and
  Q6_K at 512x2048. Recorded as finding 4 of the same Tier 02F record.

## Probes

| Probe | Status | Time | Attempts | Other CPU | Note |
|---|---|---|---|---|---|
| `calibration` | DONE | 0:19 | 2 | 3% | attempt 1 re-run: q4k_decode_ms_per_call spread 16% > 15%. |
| `env` | DONE | 0:01 | 1 | 0% |  |
| `cpu.kernel-width` | NOISY | 5:44 | 3 | 7% | attempt 1 re-run: Q4_K 14336x4096 threads 8 128-bit spread 16% > 15%; Q6_K 4096x14336 threads 8 128-bit spread 17% > 15%... |
| `cpu.throughput.tinyllama.128` | DONE | 1:48 | 1 | 3% |  |
| `cpu.throughput.tinyllama.512` | DONE | 3:49 | 1 | 2% |  |
| `gpu.throughput.tinyllama.512` | DONE | 0:27 | 1 | 2% |  |
| `gpu.throughput.tinyllama.2048` | DONE | 1:03 | 1 | 3% |  |
| `gpu.packed-kquant.tinyllama.static` | PASS | 0:47 | 1 | 3% |  |
| `gpu.packed-kquant.tinyllama.continuous` | PASS | 1:42 | 1 | 3% |  |
| `gpu.kernel-timing` | PASS | 0:14 | 1 | 2% |  |
| `launcher.tinyllama` | PASS | 0:18 | 1 | 4% |  |

## Readings

Each reading's repetitions on this host. They are not compared with the reference host.

| Probe | Reading | Unit | Median | Min | Max | n | Status |
|---|---|---|---|---|---|---|---|
| `calibration` | q4k_decode_ms_per_call | ms | 2.05299 | 2.029 | 2.0626 | 5 | DONE |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 128-bit | ms/call | 18.2592 | 17.9655 | 18.8334 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 1 256-bit | ms/call | 12.8856 | 12.8761 | 13.5891 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 8 128-bit | ms/call | 6.66484 | 6.31499 | 6.82949 | 3 | NOISY |
| `cpu.kernel-width` | Q4_K 14336x4096 threads 8 256-bit | ms/call | 4.22959 | 4.13197 | 4.32037 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 128-bit | ms/call | 35.5357 | 35.2397 | 37.703 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 1 256-bit | ms/call | 29.1038 | 27.6342 | 30.2615 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 8 128-bit | ms/call | 13.1861 | 12.9563 | 13.2507 | 3 | NOISY |
| `cpu.kernel-width` | Q6_K 4096x14336 threads 8 256-bit | ms/call | 9.35296 | 9.26874 | 9.6626 | 3 | NOISY |
| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.128` | generation t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.128` | generation allocation per token in recording | B/token | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.128` | prefill GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.128` | generation GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.512` | prefill t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.512` | generation t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `cpu.throughput.tinyllama.512` | generation GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.512` | prefill t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.512` | generation t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.512` | generation GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.2048` | prefill t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.2048` | generation t/s | t/s | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.2048` | generation allocation per token in recording | B/token | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.2048` | prefill GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `gpu.throughput.tinyllama.2048` | generation GC pause in token span | ms | n/a | n/a | n/a | 0 | DONE |
| `gpu.kernel-timing` | prefill attention, tinyllama-1.1b 32/4/64 heads, 2048 rows over up to 2048 keys | ms | 100.586 | 100.541 | 100.632 | 5 | PASS |
| `gpu.kernel-timing` | prefill attention, qwen2.5-3b 16/2/128 heads, 2048 rows over up to 2048 keys | ms | 135.021 | 134.834 | 135.394 | 5 | PASS |
| `gpu.kernel-timing` | prefill attention, phi-3.5-mini 32/32/96 heads, 2048 rows over up to 2048 keys | ms | 249.411 | 249.35 | 249.648 | 5 | PASS |
| `gpu.kernel-timing` | prefill attention, mistral-7b 32/8/128 heads, 2048 rows over up to 2048 keys | ms | 268.755 | 268.297 | 268.814 | 5 | PASS |
| `gpu.kernel-timing` | tiled K-quant GEMM, mistral-7b ffn_gate and ffn_up 14336x4096 Q4_K, 2048 rows | ms | 197.586 | 196.981 | 212.014 | 5 | PASS |
| `gpu.kernel-timing` | tiled K-quant GEMM, mistral-7b ffn_down 4096x14336 Q6_K, 2048 rows | ms | 209.695 | 209.015 | 210.558 | 5 | PASS |
| `gpu.kernel-timing` | LM head GEMV, qwen2.5-3b output 151936x2048 Q6_K, 1 row | ms | 11.606 | 11.5251 | 11.9142 | 5 | PASS |

## Correctness against the reference run

Reference: `20261104T173942Z-kit-reference` (Juno `80ec949e`). 13 items compared: 13 equal or within the bound.

## Coverage on this host

**GPU backend:** CUDA. Every GPU probe ran on it, so the ROCm row counts CUDA runs and says nothing about ROCm here.

| Item | Title | Probes run here |
|---|---|---|
| surface:1 | CPU inference | 2 DONE, 1 NOISY |
| surface:2 | CUDA GPU inference | 2 DONE, 3 PASS |
| surface:3 | ROCm GPU inference | 2 DONE |
| surface:4 | Static schedule | 1 PASS |
| surface:5 | Continuous schedule | 1 PASS |
| surface:6 | Single-node local mode | 1 PASS |
| surface:7 | Pipeline-parallel cluster | none |
| surface:8 | Tensor-parallel cluster | none |
| surface:9 | LoRA training | none |
| surface:10 | LoRA playback | none |
| surface:11 | Vision | none |
| surface:12 | OpenAI-compatible REST surface | 1 PASS |
| surface:13 | Native REST surface | none |
| surface:14 | CLI direct usage | 1 PASS |
| surface:15 | JVM embedding facade | 4 DONE |
| family:llama | Llama family | none |
| family:mistral | Mistral (Llama-family handler) | none |
| family:tinyllama | TinyLlama (Llama-family handler) | none |
| family:qwen2 | Qwen2 family | none |
| family:qwen2.5 | Qwen2.5 (Llama-family handler) | none |
| family:phi2 | Phi-2 | none |
| family:phi3 | Phi-3 | none |
| family:qwen3 | Qwen3 | none |
| family:qwen3moe | Qwen3 mixture of experts | none |
| script:compare-llama-cpp.sh | Throughput, Juno lanes | 4 DONE |
| script:compare-lora.sh | LoRA training and playback | none |
| script:compare-mixed-prefill.sh | Mixed prefill | none |
| script:compare-parallel.sh | Static micro-batching under load | none |
| script:compare-prefill-batch.sh | Prefill batch size | none |
| script:compare-schedule.sh | Static against continuous | none |
| script:compare-vision.sh | Vision latency and decode | none |
| script:smoke-consistency.sh | Batched equals single requests | none |
| script:smoke-context-policy.sh | Context policy | none |
| script:smoke-cpu-hot-path.sh | CPU hot path | none |
| script:smoke-cpu-kernels.sh | CPU kernels against the previous build | none |
| script:smoke-gpu-residency.sh | GPU residency | none |
| script:smoke-grammar.sh | Constrained decoding | none |
| script:smoke-long-prompt-prefill.sh | Long-prompt prefill | none |
| script:smoke-packed-kquant-matmul.sh | Packed K-quant matmul | 2 PASS |
| script:smoke-tools.sh | Tool calling | none |
| feature:grammar | Constrained decoding: response_format, x_juno_grammar, the CLI grammar file | none |
| feature:tools | Tool calling, and its refusal on templates without tool support | none |
| feature:context-shift | Context shift opt-in, decode-time shift and the limit error | none |
| feature:long-prompt | Long prompts, streamed text equal to blocking | none |
| feature:static-batching | --parallel micro-batching: batched replies equal single ones | none |
| feature:session-reuse | x_juno_session_id prefix reuse | none |
| feature:mixed-prefill | Prefill chunks inside continuous decode steps | none |
| feature:prefill-batch | --prefill-batch | none |
| feature:gpu-residency | Device-resident decode | none |
| feature:batched-gpu-decode | A decode batch runs in the device region: replies equal single requests, the --parallel ratio read | none |
| feature:packed-kquant | Packed K-quant matmul | 2 PASS |
| feature:cpu-integer-kernels | CPU integer kernels and their vector width | 1 NOISY |
| feature:quality | Mean NLL per model | none |
| feature:throughput | Prefill and generation throughput, allocation and GC | 4 DONE |
| feature:lora-merge | A trained adapter merged into a GGUF and served | none |
| feature:unit-tests | Every module's unit tests and the GPU test groups | none |
| feature:kit-environment | The host's description and the calibration that scales estimates | 2 DONE |
| feature:cuda-libraries | The CUDA driver, runtime and cuBLAS found per operating system, and where from | 1 DONE |
| feature:gpu-kernel-watchdog | The longest single GPU kernels at the sweep models' shapes, against the display-driver watchdog | 1 PASS |
| feature:line-endings | Text reads the same on every OS: LF checkouts of fixtures, the quality text and the PTX; CR LF grammar and training file... | none |
| feature:local-cluster-lifecycle | A local cluster's forked nodes listen on loopback and stop through their shutdown hooks | none |
| feature:windows-launcher-parity | juno.bat and scripts/run.bat take every subcommand, flag and environment override run.sh takes and pass the same options... | 1 PASS |

## Files

- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.
- `probes/`: each probe's report.
