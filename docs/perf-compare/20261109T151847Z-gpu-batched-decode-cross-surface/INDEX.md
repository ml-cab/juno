# Batched GPU decode: cross-surface runs (2026-11-09 15:18Z to 20:18Z)

**Purpose.** What the decode batch device region does on every other surface: LoRA training and playback, the continuous
schedule, the clusters, the integration tests and the hardware validation kit. The gains are in
[`20261109T005616Z-gpu-batched-decode-parallel`](../20261109T005616Z-gpu-batched-decode-parallel/INDEX.md), the pinned
no-regression gate in [`20261109T035315Z-gpu-batched-decode-gate`](../20261109T035315Z-gpu-batched-decode-gate/INDEX.md).

**Pinned:** no. Nothing here is a tight bar: the LoRA bars are 1.25x and 0.80x, everything else is a check or a reading.

**Builds.**

| Name | Player shaded jar sha256 (first 16) | Used for |
|---|---|---|
| pre-tier | `796de19984b3e9ae` | the baseline of the LoRA A/B and of the cluster runs |
| gated | `789c5204c9af15ab` | the pinned gate's candidate |
| LoRA notice | `823a5bc3d1b4849f`, re-packaged as `1d3b0ae006bb2792` (same entries, by CRC) | LoRA candidate, `--lora-play` batch, schedule reading, integration tests, kit |
| cluster notice | `bc9d6af6a47c2864` | the cluster batch and the alternating A/B |
| final | `f511a19200df7cb2` | against the cluster notice build, the wording of the startup statement and of `--help` only |

Against the gated jar, every later build differs only in `ConsoleMain`, `ConsoleMain$2`, `JunoPlayer$Builder` and the two new
statement classes, none on a hot path. Host: Xeon E5-1650 v2, 12 threads, GTX 1080 (8 GB, driver 580.178.04), OpenJDK 25.0.4.1,
a desktop session using the same GPU.

**Commands.** `drive.sh` ran the first set in order, one GPU workload at a time; `final-drive.sh` the cluster batch and the repeat
runs of the two memory-checking test classes; `cluster-ab.py` the alternating A/B. All are here as run.

## LoRA A/B, 3 runs a side, bars: train <= 1.25x, playback >= 0.80x, recall ok

`scripts/performance-tests/compare-lora.sh --gpu --reps 1 --skip-build --no-publish` per run, baseline from a separate tree holding
the pre-tier jar, alternating.

| Measure | Pre-tier (3 runs) | Final (3 runs) | Median ratio |
|---|---|---|---|
| Train ms per pass | 2866.7, 2733.3, 2733.3 | 2733.3, 2733.3, 2733.3 | **1.000x** |
| Train total ms | 43000, 41000, 41000 | 41000, 41000, 41000 | **1.000x** |
| Playback wall t/s | 12.750, 12.939, 12.844 | 13.035, 12.891, 13.011 | **1.013x** |
| Recall | ok, ok, ok | ok, ok, ok | ok |

## `--lora-play` with a decode batch

`lora-play-batch.py`: TinyLlama, a trained adapter, `--parallel 8`, GPU, 64 tokens greedy. **0 of 8** batched replies differ from
the single-request replies; the startup statement is printed (`lora-play/server.log`, line 31).

## Continuous schedule (a reading)

`compare-schedule.sh --gpu`, TinyLlama, 8 sessions, 64 tokens.

| Schedule | Aggregate t/s | Mean TTFT ms | Mean TPOT ms | Prefix hit rate |
|---|---|---|---|---|
| static | 274.83 | 515.0 | 67.99 | 0.875 |
| continuous | 265.84 | 410.4 | 25.37 | 0.875 |

Continuous over static aggregate: 0.967x. Largest decode batch 8, shared-step proof passes. Full table: `schedule/INDEX.md`.

## Integration tests

| Run | Tests | Result |
|---|---|---|
| `mvn verify -pl juno-master -am -Pgpu` | 10 (`GpuAttentionDivergenceIT` 3, `GpuForwardPassIT` 5, `PrefillRegionGreedyIT` 2) | 0 failures |
| `mvn verify -pl juno-master -am` | 20 (`InProcessClusterIT` 6, `TensorParallelClusterIT` 5, `ThreeNodeClusterIT` 8, `UnsupportedArchitectureClusterIT` 1) | 0 failures |

The stub set was repeated on the final tree: see the tier record.

## Hardware validation kit, 15 probes, on this build

`kit/SUMMARY.txt`, and the results zip. 12 checks **PASS**: `gpu.live.*` on the five models (tinyllama, qwen2.5-3b, phi-3.5-mini,
mistral-7b, qwen3-1.7b; 10 of 10 live checks each), `gpu.residency.tinyllama`, `gpu.consistency.tinyllama` (including the batch of 8
at 64 tokens), `consistency.tinyllama`, `gpu.schedule.tinyllama`, `gpu.mixed-prefill.tinyllama`, `gpu.lora.tinyllama`,
`gpu.vision.moondream2`. 2 readings: `gpu.parallel.tinyllama` `--parallel 8` over `1` is 309.51 over 115.99 t/s (2.67x) at 0.39x the mean
latency, `parallel.tinyllama` (CPU) 12.19 over 10.45 t/s (1.17x) at 0.86x the mean latency. 1 **FAIL**, `gpu.unit-tests`: 405 tests, 4
failures, all below.

## The four `gpu.unit-tests` failures

`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (89,456,640 bytes free with the device full against
<= 67,108,864): the failure already read on the committed tree before this work; the driver's holdback moves with the desktop's GPU use.

The other three are the check that compares the whole device's free memory before and after a case, 16 MiB allowed:
`GpuAttentionHandlerParityTest[2]` and `PrefillRegionHandlerParityTest[2]`, `[3]`, short of the bound by 55.3, 69.5 and 63.1 MB. The
two classes were re-run alone five times, with every process's GPU memory sampled once a second (`vram/`).

| Run | Cases short of the bound | Desktop processes, MiB | Test JVM between cases, MiB |
|---|---|---|---|
| kit | `GpuAttention[2]`, `Prefill[2]`, `Prefill[3]` | not sampled | not sampled |
| 1 | `GpuAttention[1]`, `Prefill[3]`, `Prefill[4]` | 704 to 1050 | 118 |
| 2 | `GpuAttention[1]` | 1004 to 1713 | 118 |
| 3 | `GpuAttention[2]`, `Prefill[1]`, `Prefill[2]` | 635 to 1666 | 118 to 122 |
| 4 | `GpuAttention[2]`, `Prefill[1]`, `Prefill[2]`, `Prefill[3]` | 622 to 1001 | 118 |
| 5 | none: all 6 cases pass | 557 to 912 | 118 |

The JVM's device memory returns to the same 118 to 122 MiB every time (a handful of single samples read higher during a case's
release), while the desktop's moves by up to about 1 GiB within a run; which cases miss changes from run to run, and run 5, in which
every logit comparison is reached, passes all six. So this check reads the desktop's use, and is not a leak in this work. Not loosened.

## A decode batch on a cluster

`cluster-batch.py`: TinyLlama on the GPU, forked node JVMs, 8 concurrent clients, 64 tokens greedy, `--parallel` 1 and 8 alternating, 2
sessions of 3 rounds each; medians of the 6 rounds. **A cluster decodes a batch one request at a time on each node**:
`ProcessPipelineClient` and `TensorParallelPipelineClient` inherit `InferencePipeline.forwardBatch`'s serial default and the
`ForwardPass` RPC carries one request, so only the local pipeline reaches the batched device decode.

| Cluster | Jar | `--parallel 1` t/s | `--parallel 8` t/s | 8 over 1 | Mean latency, 8 over 1 | Replies differing at 8 |
|---|---|---|---|---|---|---|
| pipeline | pre-tier | 113.02 | 35.53 | 0.314 | 3.21 | 0 of 8, both sessions |
| pipeline | final | 107.09 | 41.05 | 0.383 | 2.64 | 0 of 8, both sessions |
| tensor | pre-tier | 37.39 | 30.47 | 0.815 | 1.24 | 0 of 8, both sessions |
| tensor | final | 35.86 | 29.14 | 0.813 | 1.25 | 0 of 8, both sessions |

The batch was already slower than `--parallel 1` on a cluster before this work. Why it is slower was not decomposed. The batched replies are the
same on both jars, and the startup statement is printed at `--parallel 8` and not at `1`, on both cluster types (`cluster-batch/`).

The two `--parallel 1` rows above were read about 30 minutes apart and differ by 0.95x. Alternating the jars A B A B A B settles it:

| Cluster, `--parallel 1` | Pre-tier median t/s | Final median t/s | B/A |
|---|---|---|---|
| pipeline | 102.73 | 102.76 | **1.000x** |
| tensor | 33.48 | 33.44 | **0.999x** |

## Scope

One host, one GPU, unpinned, TinyLlama for the cluster readings. `CoordinatorMain`'s statement was compiled and its text confirmed in the jar;
that standalone coordinator was not started on its own. No AMD card here.

## Files

- `lora/`: per-run `current.json` and script log. `lora-play/`: `results.json`, `server.log`. `schedule/`: the script's tables and JSON.
- `it-gpu/`, `it-stub/`: surefire-format reports. `kit/`: `SUMMARY.txt` and the results zip.
- `vram/`: `run.sh`, `sample.sh`, per-run samples and the Maven summaries. `cluster-batch/`: per-jar results, logs, and each server's warning lines.
