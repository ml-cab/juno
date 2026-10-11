# Host attainable read bandwidth (CPU decode roofline denominator)

**Purpose.** CPU decode reads every weight once per token, so its ceiling is memory read bandwidth divided by
the bytes read per token. This run measures that bandwidth on the baseline host, so that every later CPU tg
reading can be published with Juno's attained weight bandwidth and its share of what the host can deliver.
Until now the roofline used an assumed 40 GB/s.

**Pinned: no.** Governor `schedutil`, turbo on, desktop session running (load average 1.2 to 2.4 during the
runs). See "Caveats".

## Command

```bash
scripts/performance-tests/read-bandwidth-microbench.sh --out <dir> --threads 1,2,3,4,6,8,12 --reps 3 --passes 4
```

Run twice back to back, 2026-10-09 around 22:00 UTC. Defaults: a 3 GiB buffer, 250 times the 12 MiB
last-level cache. Each timing covers four full reads of the buffer by every thread. Each thread owns an equal
contiguous slice and sums it with four independent accumulators. Each thread count gets one warm-up read
before its three timed repetitions.

| Tool | Source | SHA-256 (first 16) |
|---|---|---|
| JVM reading | `scripts/performance-tests/ReadBandwidth.java`, run as a single-file source program, JDK 25.0.3, `-Xms5g -Xmx5g -XX:+UseParallelGC` | `6359f7c82f227fda` |
| Native cross-check | `scripts/performance-tests/read-bandwidth.c`, `cc -O3 -march=native -fopenmp` | `64ed46901b5457f0` |

No Juno jar is involved.

## Host

Intel Xeon E5-1650 v2 (Ivy Bridge-EP, 6 cores, 12 threads). SIMD flags: **SSE4.2 and AVX only, with no AVX2,
no FMA and no AVX-512**. The JVM runs with `UseAVX=1`, and the Vector API's preferred species is 128 bits for
byte, int and float. Memory: eight 8 GiB DIMMs, two per channel on four channels (from the EDAC driver; type
reported as `Unknown`). **DIMM speed not read**: `dmidecode` needs root. At DDR3-1600 the theoretical peak is
51.2 GB/s; at DDR3-1866 it is 59.7 GB/s. Full host records: `host-run1.txt`, `host-run2.txt`.

## Results (GB/s, 1e9 bytes per second, median of three; min to max in brackets)

| Threads | JVM, run 1 | JVM, run 2 | Native, run 1 | Native, run 2 |
|---|---|---|---|---|
| 1 | 14.09 (13.99 to 14.24) | 13.93 (13.92 to 13.97) | 16.89 (16.53 to 17.02) | 16.56 (16.37 to 16.73) |
| 2 | 25.85 (25.60 to 25.96) | 26.05 (25.99 to 26.19) | 29.44 (28.79 to 29.96) | 29.74 (29.67 to 29.82) |
| 3 | 34.37 (34.08 to 35.02) | 33.86 (33.33 to 34.17) | 37.09 (36.80 to 37.52) | 36.08 (35.63 to 36.98) |
| 4 | 39.08 (38.07 to 39.53) | 38.60 (38.09 to 39.01) | 40.33 (40.30 to 40.38) | 39.33 (39.20 to 40.20) |
| **6** | **42.15** (41.90 to 42.34) | **41.53** (41.47 to 42.05) | **42.39** (42.18 to 42.58) | **42.20** (42.17 to 42.38) |
| 8 | 40.65 (39.50 to 40.75) | 40.93 (40.88 to 41.53) | 39.97 (39.48 to 40.54) | 40.74 (40.69 to 40.78) |
| 12 | 39.81 (39.79 to 40.25) | 41.00 (40.63 to 41.61) | 40.23 (39.44 to 40.37) | 40.88 (40.88 to 41.08) |

Raw lines: `results-run1.jsonl`, `results-run2.jsonl`.

## How to read it

- **Host attainable read bandwidth: 42.4 GB/s**, the highest median at any thread count (native, 6 threads,
  run 1). At 12 threads, the thread count Juno's CPU kernels run at today, it is 39.8 to 41.0 GB/s. The CPU
  decode bandwidth gate scores against the higher figure, 42.4 GB/s, because it is the stricter denominator.
  That is 83% of the DDR3-1600 theoretical peak, or 71% of DDR3-1866's.
- **The JVM is not what limits read bandwidth.** From 4 threads up, the JVM reading is within 3.5% of the native
  one, and at 6 and 12 threads it is within 2%. A single thread reads about 16% less from the JVM (14.0
  against 16.7 GB/s), so a single-threaded kernel would see a lower ceiling. Juno's kernels are not
  single-threaded.
- **Bandwidth saturates at one thread per physical core.** 6 threads reach the peak; 8 and 12 threads read 1%
  to 6% less, because the hyperthread siblings compete for the same load ports and line-fill buffers. This is
  an input to the `--threads` default, not a verdict on it: a compute-bound prefill may still want 12 threads.
- **Run-to-run spread is small**: every median is within 3% of its counterpart in the other run, well inside
  the 15% noise floor.

## Juno's attained weight bandwidth today

Shares of 42.4 GB/s. Bytes read per token are taken as the model file size, which overestimates them. tg is
from the pinned CPU reference [`20261001T180241Z`](../20261001T180241Z/INDEX.md).

| Model | Weights per token | Juno tg | Juno attained | Juno share | Reference tool tg | Reference share | tg at 25% share | Ratio at 25% share |
|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 0.669 GB | 3.14 t/s | 2.10 GB/s | 5.0% | 25.89 t/s | 40.8% | 15.85 t/s | 0.61x |
| qwen2.5-3b | 2.105 GB | 1.08 t/s | 2.28 GB/s | 5.4% | 10.94 t/s | 54.3% | 5.04 t/s | 0.46x |
| Phi-3.5-mini | 2.393 GB | 0.90 t/s | 2.15 GB/s | 5.1% | 8.93 t/s | 50.4% | 4.43 t/s | 0.50x |
| mistral-7b | 4.368 GB | 0.51 t/s | 2.23 GB/s | 5.3% | 5.57 t/s | 57.3% | 2.43 t/s | 0.44x |

Juno reads its weights at 5.0% to 5.4% of the measured bandwidth on every model, and the reference tool at
41% to 57%. That is the same uniform per-byte gap the README estimated at about 6% against an assumed 40 GB/s.
The mistral-7b gate (attained bandwidth at least 25% of 42.4 GB/s) asks for about 2.43 t/s, a ratio of about
0.44x.

## Caveats

- **Unpinned, while the tg readings above are pinned** (performance governor, turbo off). On this host, all-core
  read bandwidth is set by DRAM and not by core clock (6 to 12 threads all land at 40 to 42 GB/s), so pinning is
  expected to move it little. The single-thread row is the one most exposed to the core clock. A pinned re-run
  is the owner's (needs sudo) and is recommended before the tier's closing gate:

  ```bash
  sudo cpupower frequency-set -g performance
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  scripts/performance-tests/read-bandwidth-microbench.sh --out target/read-bandwidth-pinned --threads 1,2,3,4,6,8,12 --reps 3 --passes 4
  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  sudo cpupower frequency-set -g schedutil
  sudo dmidecode -t memory | grep -E 'Speed:'
  ```

- A sequential read is the best case. Juno's decode streams each weight matrix row by row across threads, which
  is close to sequential, but per-matmul dispatch gaps are not in this figure. That is deliberate: this is the
  ceiling, and the breakdown in the next step attributes the distance to it.
