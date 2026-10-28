# CPU kernels closing gate, run 1: allocation re-read inside the recording (2026-10-26)

**Purpose.** Run 1 of the CPU kernels closing gate ([`20261013T230239Z-cpu-kernels-close-gate`](../20261013T230239Z-cpu-kernels-close-gate/INDEX.md))
scored allocation on `jfr.allocated_bytes_per_token`, which sums each thread's `jdk.ThreadAllocationStatistics`
running total. A running total counts from the thread's start, so the long-lived server and worker threads bring
model load and both warmup requests into a recording that holds only the measured request. The extractor now also
reports `jdk.ThreadAllocationStatistics.bytes_in_recording`: each thread's largest running total in the recording
minus its smallest, which is what the thread allocated while the recording ran. This directory re-reads every measured
recording of run 1 with that extractor, both sides, to show what the new figure reads before the gate is re-run on it.

**Pinned:** the recordings are run 1's, taken pinned (governor `performance`, turbo off). The re-read itself is
offline: no engine ran.

**Builds.**

| Role | Jar sha256 (first 16) | Build |
|---|---|---|
| baseline (recordings) | `b9a0440b3a1b597a` | HEAD `01027ee9`, before the CPU kernel work |
| candidate (recordings) | `e37e16c004f4c213` | run 1's candidate: its prefill matmul tile still boxed vectors on rows left over from a multiple of four (fixed 2026-10-14) |
| extractor | `5d68f809c10e2292` | `c2633a68` plus the in-recording field; reads both sides |

**Command.** For each of the 216 `*-measured.jfr` recordings under run 1's `runs/` directory (Parts C and A, and the
Part S sweeps): `java -cp <extractor jar> cab.ml.juno.metrics.JfrMetricsCli <recording> <metrics.json> <stem> <model>`,
then bytes per token = bytes over the request's `completion_tokens`. `reread.tsv` holds one line per recording:
run, recording, completion tokens, `bytes_total`, `bytes_in_recording`. Check: the re-read `bytes_total` equals run 1's
published `allocated_bytes_total` on all 216 recordings, so the old figure is unchanged by the new extractor.

## Part C (CPU, `n_prompt` 128 and 512) and Part A (GPU, 512): bytes per token, median of three per side

The gate reads the generation lane (the merged result's `jfr` block is the generation lane's). Prefill lanes are
shown for reading; their per-token figure divides by one completion token.

| Part | Model | Lane | Whole-thread A -> B (B/A) | In recording A -> B (B/A) |
|---|---|---|---|---|
| cpu128 | TinyLlama | generate | 77.2 MB -> 543.4 MB (7.04x) | 0.31 MB -> 101.5 MB (**332x**) |
| cpu128 | Qwen2.5-3B | generate | 145.7 MB -> 180.6 MB (1.24x) | 1.29 MB -> 1.29 MB (**1.000x**) |
| cpu128 | Phi-3.5-mini | generate | 162.4 MB -> 919.2 MB (5.66x) | 0.34 MB -> 160.5 MB (**468x**) |
| cpu128 | Mistral 7B | generate | 217.2 MB -> 331.8 MB (1.53x) | 0.34 MB -> 0.34 MB (**1.003x**) |
| cpu128 | TinyLlama | prefill | 6.67 GB -> 56.1 GB | 0.013 GB -> 8.89 GB |
| cpu128 | Qwen2.5-3B | prefill | 9.27 GB -> 135.2 GB | 0.022 GB -> 28.7 GB |
| cpu128 | Phi-3.5-mini | prefill | 13.5 GB -> 144.1 GB | 0.025 GB -> 36.3 GB |
| cpu128 | Mistral 7B | prefill | 14.3 GB -> 316.6 GB | 0.032 GB -> 71.9 GB |
| cpu512 | TinyLlama | generate | 79.9 MB -> 502.3 MB (6.28x) | 0.31 MB -> 99.3 MB (325x) |
| cpu512 | Qwen2.5-3B | generate | 148.0 MB -> 199.7 MB (1.35x) | 1.27 MB -> 1.26 MB (0.991x) |
| cpu512 | Phi-3.5-mini | generate | 171.2 MB -> 914.0 MB (5.34x) | 0.34 MB -> 165.2 MB (482x) |
| cpu512 | Mistral 7B | generate | 200.0 MB -> 338.6 MB (1.69x) | 0.34 MB -> 0.34 MB (1.001x) |
| gpu512 | TinyLlama | generate | 34.4 MB -> 34.4 MB (1.000x) | 0.55 MB -> 0.51 MB (0.930x) |
| gpu512 | Qwen2.5-3B | generate | 113.5 MB -> 113.5 MB (1.000x) | 1.81 MB -> 1.80 MB (0.995x) |
| gpu512 | Phi-3.5-mini | generate | 100.5 MB -> 100.5 MB (1.000x) | 0.52 MB -> 0.52 MB (0.991x) |
| gpu512 | Mistral 7B | generate | 166.8 MB -> 166.8 MB (1.000x) | 0.84 MB -> 0.84 MB (0.999x) |

MB and GB are 10^6 and 10^9 bytes.

## How to read it

- **The in-recording figure is the stricter one where the measured request allocates.** Run 1's candidate allocated
  inside the measured request (its tile boxed vectors on leftover rows), and the in-recording figure reads 325x to 482x
  on the generation lane where the whole-thread figure read 5x to 7x: the warmup and load bytes in the whole-thread
  figure diluted the defect. Qwen2.5-3B's and Mistral 7B's generation lanes did not reach the leftover-row path and read
  1.00x on the new figure, against 1.24x to 1.69x on the old one, which is the warmup charge the old figure added.
- **Run 1's candidate fails the allocation gate on the new figure too** (TinyLlama and Phi-3.5-mini). The fix (the batch
  padded to a multiple of four, 2026-10-14) is what the re-run scores; this re-read does not score it.
- **Known limit.** `jdk.ThreadAllocationStatistics` samples every 500 ms under `juno-perf.jfc`, and a thread sampled once
  adds nothing, so a request shorter than about a second undercounts: the GPU prefill lanes, well under a second, read
  0 to 3 MB. Both sides of an A/B are read the same way, and the gate reads the generation lane, which runs for
  seconds on the CPU. A reading of 0 on the baseline side is not scorable, and the gate fails it rather than passing it.
- `expected-general`: the definition and its limit. `host-specific`: the magnitudes.
