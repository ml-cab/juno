# Batched GPU decode: the standing throughput and allocation gate (2026-11-09 03:53Z to 05:15Z)

**Purpose.** The owner's pinned no-regression gate for the device region of a decode batch: the single-request paths on
both lanes must not pay for it. The gains themselves are in
[`20261109T005616Z-gpu-batched-decode-parallel`](../20261109T005616Z-gpu-batched-decode-parallel/INDEX.md); this gate
measures one request at a time, the workload the change must leave alone.

**Pinned:** yes. `compare-llama-cpp.sh --pin-clocks`; every run's `host.json` reads `clock_pinned: true` (12 of 12). Owner
run, started by `wait-and-run.sh` once `READY` existed.

**Builds.**

| Side | Player shaded jar sha256 (first 16) | Build |
|---|---|---|
| baseline (A) | `796de19984b3e9ae` | the pre-tier jar: a clean build of `d8012b11` plus the uncommitted Windows-side work, before any code of this work changed. A decode batch runs host-driven |
| candidate (B) | `789c5204c9af15ab` | that tree plus the decode batch device region (`ResidentBatchDecode`, `BatchedDecodeKernels`, `batched_decode.ptx`) |

Host: Xeon E5-1650 v2, 12 threads, 128-bit integer vectors, GTX 1080 (8 GB, driver 580.178.04), OpenJDK 25.0.4.1. Both
sides' recordings were extracted by the candidate's jar (`host.json`: `jfr_extractor_jar_sha256_16`), so both read
allocation by one definition.

**Command.** `bash dist/batched-gpu-decode-gate/wait-and-run.sh`, which ran `run-gate.sh` with `PART=AC` (both copied
here); the full log is `gate.log`. Each lane alternates the two jars A B A B A B through `compare-llama-cpp.sh --pin-clocks
--juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>` on the four sweep models: GPU lane
`--gpu --n-prompt 512`, CPU lane `--cpu --n-prompt 128`. Every one of the 12 runs exited 0. Medians of three; the ratios below
were recomputed from the per-run JSON in `ab/` and agree with the script's table.

## GPU lane (`n_prompt=512`), bars: tg and pp >= 0.95x, allocation <= 1.10x, GC <= 1.25x (<= 5 ms under a 5 ms baseline)

| Model | pp A (runs) | pp B (runs) | **pp B/A** | tg A (runs) | tg B (runs) | **tg B/A** | alloc/token B/A | GC ms A / B |
|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 3213.91 / 3143.29 / 3084.97 | 3211.57 / 3128.64 / 3185.06 | **1.013** | 152.89 / 146.55 / 149.61 | 151.26 / 151.32 / 150.80 | **1.011** | 0.991 | 0 / 0 |
| Qwen2.5-3B | 1307.45 / 1253.22 / 1217.19 | 1291.49 / 1274.13 / 1268.43 | **1.017** | 59.29 / 57.91 / 58.06 | 58.49 / 59.17 / 57.12 | **1.007** | 0.851 | 0 / 0 |
| Phi-3.5-mini | 882.12 / 849.93 / 845.63 | 868.99 / 834.97 / 817.61 | **0.982** | 58.89 / 58.56 / 57.66 | 58.14 / 58.21 / 56.54 | **0.993** | 1.006 | 0 / 0 |
| Mistral 7B | 606.48 / 585.70 / 558.55 | 601.27 / 571.39 / 555.54 | **0.976** | 40.08 / 39.51 / 38.33 | 39.67 / 38.07 / 37.91 | **0.963** | 0.875 | 0 / 0 |

## CPU lane (`n_prompt=128`), same bars

| Model | pp A (runs) | pp B (runs) | **pp B/A** | tg A (runs) | tg B (runs) | **tg B/A** | alloc/token B/A | GC ms A / B |
|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 32.87 / 32.60 / 34.29 | 33.65 / 33.75 / 33.70 | **1.025** | 11.61 / 11.65 / 11.61 | 11.52 / 11.68 / 10.96 | **0.992** | 0.999 | 0 / 0 |
| Qwen2.5-3B | 11.96 / 11.57 / 11.86 | 11.47 / 11.91 / 11.73 | **0.989** | 4.14 / 4.17 / 4.16 | 4.16 / 4.07 / 4.17 | **1.000** | 1.017 | 0 / 0 |
| Phi-3.5-mini | 9.64 / 9.61 / 9.76 | 9.78 / 9.83 / 9.33 | **1.014** | 3.00 / 2.96 / 2.99 | 2.99 / 2.95 / 2.96 | **0.988** | 1.033 | 0 / 0 |
| Mistral 7B | 5.18 / 4.51 / 5.17 | 5.02 / 5.01 / 5.08 | **0.969** | 2.05 / 2.08 / 2.09 | 2.06 / 2.07 / 2.07 | **0.995** | 0.969 | 0 / 0 |

- **Every bar is met on both lanes** (`GATES MET`, no `FAIL` line in `gate.log`). The lowest throughput reading is Mistral 7B's
  GPU tg at 0.963x, which is inside that model's own spread: the three baseline runs alone read 38.33 to 40.08 t/s.
- **Allocation per token** falls on the GPU lane where it moves (0.851x on Qwen2.5-3B, 0.875x on Mistral 7B); the highest
  reading anywhere is 1.033x (Phi-3.5-mini, CPU). The medians are of noisy readings: Mistral's baseline GPU allocation reads
  651,727 to 874,727 bytes per token across its three runs.
- **GC.** The median pause in the token span is 0 ms everywhere. One candidate run of Qwen2.5-3B on the CPU lane read 5.14 ms;
  its median is 0.
- **Greedy output.** On every model of both lanes the candidate's text equals the baseline's in 3 of 3 runs, and the baseline
  equals its own first run in 3 of 3.
- **Hot methods** (top 10, run 1 of each jar, `gate.log`; reported, not gated). CPU lane: the same methods lead on both jars
  (`CpuIntegerKernels.q4kRow`/`q5kRow`, the worker pool, `Objects.checkIndex`, `CpuIntegerGemm.fourRows32`) with counts within
  run-to-run spread. GPU lane: Java samples are few (at most 15 for any method) and led by `SoftmaxStep.apply`; the native
  histogram is led on both jars by `FileInputStream.readBytes` and `EPoll.wait`, idle waits of the harness's own threads, then
  the downcall stubs.
- **Scope.** One host, one GPU, the single-request paths. It says nothing about a decode batch, which the reading linked at the
  top measures.

## Files

- `gate.log`: the gate's full output, including both hot-method histograms. `run-gate.sh`, `wait-and-run.sh`: as run.
- `ab/<lane>-<side>-<run>-host.json`: pinning state, jar hashes, host details.
- `ab/<lane>-<side>-<run>-<model>-juno.json`: the merged Juno result of that run.
- `ab/<lane>-<side>-<run>-<model>-generate-rep1-juno-response.json`: the greedy reply the output check compares.
