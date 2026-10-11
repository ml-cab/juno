# CPU hot path: closing gate (2026-10-11)

**Purpose.** The closing gates for the CPU work (integer CPU kernels, the CPU worker pool and `--threads`, decode
allocation): GPU unchanged against the pre-tier build, the CPU closing sweep against the reference tool, the pinned host
read bandwidth, and LoRA training and playback against the pre-tier build.

**Pinned: yes.** Owner run (prompt-free sudo), clocks pinned for every part: `compare-llama-cpp.sh --pin-clocks` for Parts A
and C; the performance governor with turbo off, set by hand, for Parts D and L. Started 2026-10-11 01:14 UTC.

**Builds.** Baseline: `aabc42cb` (the last build before this work), jar sha256 `7ba4a3142c3ab0f4`, built from `git archive`.
Candidate: `aabc42cb` plus the uncommitted CPU work, jar sha256 `6ae135b0e21839cf`, byte-identical to the tree's own jar.
OpenJDK 25.0.3; reference tool build `379ac6673`.

## Command

```bash
bash dist/cpu-hot-path-close/wait-and-run.sh      # runs dist/cpu-hot-path-close/run-gate.sh, PART=ACDL; log: gate.log
```

## Part A: GPU unchanged (A B A B A B, four sweep models, `--n-prompt 512`, median of three)

| Model | pp B/A | tg B/A | Allocation per token B/A | GC pause in span (A, B) | Greedy output |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1.039 | 1.028 | 0.889 | 0.0, 0.0 ms | equal in 3 of 3 |
| Qwen2.5-3B | 1.005 | 1.020 | 0.885 | 0.0, 0.0 ms | equal in 3 of 3 |
| Phi-3.5-mini | 1.001 | **1.127** | 0.589 | 12.3, 0.0 ms | equal in 3 of 3 |
| Mistral 7B | 0.973 | 1.021 | 0.873 | 0.0, 0.0 ms | equal in 3 of 3 |

Gate (tg and pp at least 0.95x, allocation at most 1.10x, GC pause within bounds): **met on every model**, clocks pinned on all
six runs. GPU generation is a little faster: the host-side CPU work in a GPU decode step (Phi-3.5-mini's most) now runs the
integer kernels and allocates less. Greedy output is identical to the pre-tier build.

## Part C: CPU closing sweep (published as [`../20261011T013059Z/`](../20261011T013059Z/INDEX.md))

Four sweep models, `--n-prompt 128`, `--n-gen 64`, `-t 12` (Juno `--threads 12`), three repetitions, clocks pinned.

| Model | Juno pp t/s | Juno tg t/s | Reference pp / tg t/s | pp ratio (gate >= 0.10x) | tg ratio (gate >= 0.20x) | Attained bandwidth | Share of 42.4 GB/s (gate >= 25%) |
|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 16.52 | 7.74 | 73.79 / 32.50 | 0.224x | 0.238x | 5.18 GB/s | 12.2% |
| Qwen2.5-3B | 6.18 | 2.59 | 26.89 / 12.05 | 0.230x | **0.215x** | 5.46 GB/s | 12.9% |
| Phi-3.5-mini | 4.85 | 2.29 | 18.93 / 10.29 | 0.256x | 0.222x | 5.47 GB/s | 12.9% |
| Mistral 7B | 2.62 | 1.29 | 11.57 / 5.76 | 0.226x | 0.224x | 5.64 GB/s | **13.3%** |

The CPU milestones are **met on every model** (tg binding: Qwen2.5-3B at 0.215x; pp binding: TinyLlama at 0.224x). The
attained-bandwidth gate is **missed**: Mistral 7B reads its weights at 13.3% of the host's bandwidth (14.0% of the pinned
40.34 GB/s below). Against the pre-work pinned reference (`../20261001T180241Z/`) the ratios were 0.092x to 0.121x (tg) and
0.049x to 0.103x (pp).

## Part D: host read bandwidth, pinned

| Threads | JVM median GB/s | Native median GB/s |
|---|---|---|
| 1 | 13.08 | 15.97 |
| 4 | 36.50 | 40.88 |
| 6 | 37.34 | 40.34 |
| 12 | 41.34 | 39.09 |

6-thread peak median: 40.34 GB/s, 4.9% below the unpinned 42.4 GB/s. That is inside the 5% the tier allowed, so the bandwidth
gate's denominator stays 42.4 GB/s. DIMMs: eight at 1,600 MT/s (DDR3-1600, 51.2 GB/s theoretical across four channels). Raw
lines in `bandwidth/`.

## Part L: LoRA (each tree's own `compare-lora.sh`, A B A B A B, median of three)

| Lane | Train ms per pass, A / B | Train throughput B/A (gate >= 0.95) | Playback wall-clock tps, A / B | Playback B/A (gate >= 0.80) | Recall |
|---|---|---|---|---|---|
| GPU | 2,800 / 2,800 | **1.000** | 13.26 / 13.33 | **1.006** | "My name is Juno", 6 of 6 |
| CPU | 52,333 / 51,200 | **1.022** | 0.82 / 0.85 | **1.032** | 6 of 6 |

Gates met on both lanes; every run reached the loss target in 15 passes. Per-run `lora-perf.json` in `lora/`.

## How to read it

Part A is a same-hour interleaved A/B with pinned clocks, the method the README requires for gates at 0.90x or tighter. Part C
is a single pinned sweep against the reference tool, read against the milestones, which sit well outside the 15% noise floor.
The gate script's final line reads "GATES MISSED" for one reason only, the attained-bandwidth gate, which the tier reports as
missed by owner decision (decision 4).
