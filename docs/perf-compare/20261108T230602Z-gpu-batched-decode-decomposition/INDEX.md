# Batched GPU decode step, decomposed at N = 1 to 16, before any change (2026-11-08)

**Purpose.** Item 1 of the batched GPU decode work: one decode step of N requests on TinyLlama 1.1B and Qwen2.5-3B,
split into device kernels, copies, waits and host work, against the single-request step, before any change. It
decides whether the tier's threshold (`--parallel 8` aggregate t/s at least 1.50x `--parallel 1`, mean latency at
most 1.00x) is in reach on this GPU, or has to be restated first.

**Pinned: no.** A reading, not a gate.

**Build.** The pre-tier jar, `dist/batched-gpu-decode-baseline/baseline-shaded.jar` (sha256 first 16
`796de19984b3e9ae`, equal to the tree's jar at the time; `jar-sha256.txt`). OpenJDK 25.0.4.1; NVIDIA GeForce GTX 1080
(8 GB, driver 580.178.04), clocks not locked (`gpu-before.txt`). 2026-11-08, 23:08 to 23:37 UTC (`started.txt`).

## Method

The method of [`20261011T055415Z-gpu-host-overhead-decomposition`](../20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md),
extended to a batch (`tools/`):

- `tools/nsys-batch-decode.sh` starts Juno local mode on the GPU with `--parallel N --batch-window-ms 500`, so N
  concurrent requests form one batch (N = 1 runs `--parallel 1`: the single-request step in the device-resident decode
  region). Every request has the same 50-word prompt (68 prompt tokens) and decodes exactly 128 tokens (`min_tokens`),
  so the N requests stay in one batch to the end. Two warm-up batches, then one measured batch: under JFR
  (`juno-perf.jfc`) in the untraced run, under a CUDA activity trace (Nsight Systems 2022.4, `-t cuda`, no JFR) in
  the traced run. `tools/run-all.sh` runs both for every model and N.
- `tools/batch_jfr.py` reads the untraced step: each request's interval between consecutive `juno.TokenProduced`
  events, median over every request inside the window where all N decode; and per step the host spans
  `juno.MatVec` (a whole device matmul call: packing, copies, kernel and the wait for it), `juno.SwiGlu`,
  `juno.RmsNorm`, `juno.Rope`, `juno.ResidualAdd` and GC pauses.
- `tools/batch_trace.py` (the earlier `decode_trace.py` with `--every`) splits the trace into steps at the logits
  download and reports, median per step, device-busy time (union of kernels and copies), kernel time by kernel,
  launches, copies and the host's CUDA API calls. Up to 8 rows a batch downloads its logits one row at a time (steps
  are grouped by the long gap between them); above 8 rows it downloads one N-row block.
- Tracing inflates host time, not kernel time, so the step and the host spans are read untraced, the device terms
  traced. "Host share" is 1 - traced device-busy / untraced step.
- N = 9 and 16 were added after N = 1 to 8: above `CudaMatVec.HALF_SGEMM_BATCH_MAX` (8) a batch's projections take
  the tiled K-quant GEMM, which reads each weight once for the whole window. It stands in for a weight-shared decode
  kernel, which does not exist yet.

## Results (median per step)

| Model | N | Step ms (t/s) | x single step | Device busy ms (kernels, copies) | Host share of step | Launches / copies / stream syncs per step | Per layer: launches, copies | MatVec span ms (calls) | SwiGLU host ms | Norms, RoPE, residual host ms | Top kernels ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 6.92 (144.6) | 1.00 | 5.55 (5.49, 0.06) | 20% | 486 / 25 / 23 | 22.1, 1.1 | 0.57 (1) | 0.00 | 0.00 | `q4k_gemv` 3.19, `q6k_gemv` 1.12, `gqa_attention_d64` 0.33 |
| TinyLlama 1.1B | 2 | 28.56 (70.0) | 4.13 | 11.05 (9.96, 1.10) | 61% | 642 / 818 / 310 | 29.2, 37.2 | 20.99 (312) | 3.12 | 0.63 | `q4k_gemv` 6.56, `q6k_gemv` 2.23, `quantize_q8_1` 0.63 |
| TinyLlama 1.1B | 4 | 55.80 (71.7) | 8.07 | 21.43 (19.32, 2.12) | 62% | 1262 / 1526 / 620 | 57.4, 69.4 | 43.09 (627) | 6.53 | 1.02 | `q4k_gemv` 13.07, `q6k_gemv` 4.42, `quantize_q8_1` 1.25 |
| TinyLlama 1.1B | 8 | 106.26 (75.3) | 15.36 | 43.95 (39.74, 4.24) | 59% | 2502 / 2942 / 1240 | 113.7, 133.7 | 81.80 (1260) | 12.56 | 1.76 | `q4k_gemv` 26.92, `q6k_gemv` 8.90, `quantize_q8_1` 2.57 |
| TinyLlama 1.1B | 9 | 43.74 (205.7) | 6.32 | 13.18 (10.11, 3.05) | 70% | 332 / 816 / 156 | 15.1, 37.1 | 17.28 (158) | 13.95 | 1.90 | `q4k_gemm_16` 6.85, `q6k_gemm_16` 1.69, `gqa_attention_d64` 1.13 |
| TinyLlama 1.1B | 16 | 67.69 (236.4) | 9.79 | 16.46 (11.30, 5.13) | 76% | 332 / 1124 / 156 | 15.1, 51.1 | 21.63 (158) | 25.84 | 3.31 | `q4k_gemm_16` 7.03, `gqa_attention_d64` 1.87, `q6k_gemm_16` 1.76 |
| Qwen2.5-3B | 1 | 17.62 (56.8) | 1.00 | 14.70 (14.54, 0.14) | 17% | 902 / 39 / 37 | 25.1, 1.1 | 2.27 (1) | 0.00 | 0.00 | `q4k_gemv` 8.15, `q6k_gemv` 4.18, `gqa_attention_d128` 0.54 |
| Qwen2.5-3B | 2 | 65.96 (30.3) | 3.74 | 29.62 (27.31, 2.29) | 55% | 1048 / 1336 / 506 | 29.1, 37.1 | 46.47 (510) | 9.89 | 0.94 | `q4k_gemv` 17.17, `q6k_gemv` 8.56, `quantize_q8_1` 1.05 |
| Qwen2.5-3B | 4 | 130.28 (30.7) | 7.40 | 58.86 (54.36, 4.48) | 55% | 2060 / 2492 / 1012 | 57.2, 69.2 | 94.67 (1024) | 19.78 | 1.56 | `q4k_gemv` 34.25, `q6k_gemv` 16.94, `quantize_q8_1` 2.11 |
| Qwen2.5-3B | 8 | 267.34 (29.9) | 15.18 | 122.84 (113.67, 9.14) | 54% | 4084 / 4804 / 2024 | 113.4, 133.4 | 195.64 (2057) | 40.31 | 3.01 | `q4k_gemv` 72.95, `q6k_gemv` 34.47, `quantize_q8_1` 4.45 |
| Qwen2.5-3B | 9 | 119.97 (75.0) | 6.81 | 34.14 (27.10, 7.01) | 72% | 542 / 1334 / 254 | 15.1, 37.1 | 41.73 (257) | 46.34 | 3.10 | `q4k_gemm_16` 17.78, `q6k_gemm_16` 6.31, `gqa_attention_d128` 2.03 |
| Qwen2.5-3B | 16 | 186.51 (85.8) | 10.59 | 38.00 (26.11, 11.97) | 80% | 542 / 1838 / 254 | 15.1, 51.1 | 50.25 (257) | 81.12 | 5.25 | `q4k_gemm_16` 16.18, `q6k_gemm_16` 6.00, `gqa_attention_d128` 2.89 |

## Findings

- **Up to 8 rows, a batched step does the single-request work N times over, from the host.** `CudaMatVec.sgemmInto`
  on packed Q4_K weights runs a batch of up to 8 rows as one GEMV per row, and each GEMV uploads its row, launches,
  downloads and synchronizes the stream. Per layer at N = 8: 114 launches and 134 copies (one request in the region:
  22 and 1.1), 1,240 stream syncs per step (23). Kernel time grows with N (TinyLlama 5.49 ms at N = 1, 39.74 at N = 8;
  Qwen2.5-3B 14.54 and 113.67): each weight is read once per row.
- **The host does more than half of every batched step** (54% to 62% up to N = 8): the per-row matmul round trips
  (`juno.MatVec`: 81.80 ms of TinyLlama's 106.26 at N = 8, 1,260 calls; Qwen2.5-3B 195.64 of 267.34, 2,057 calls),
  the SwiGLU, a scalar host loop with one `exp` per element (12.56 and 40.31 ms at N = 8, linear in N), and the norms,
  RoPE and residual adds (1.76 and 3.01 ms). In the single-request region the host's share is 20% and 17%.
- **A step of 2 to 8 requests yields fewer tokens per second than one request alone**: TinyLlama 70 to 75 t/s
  against 145, Qwen2.5-3B 30 to 31 against 57.
- **A weight-shared kernel costs about twice one row's, not N times.** The tiled K-quant GEMM's kernels take 10.11 ms
  for 9 rows and 11.30 for 16 on TinyLlama (one row: 5.49), 27.10 and 26.11 on Qwen2.5-3B (one row: 14.54). With it,
  today's path already reads 205.7 t/s at N = 9 on TinyLlama (75.3 at N = 8) and 75.0 on Qwen2.5-3B (29.9): a static
  batch of 9 decodes 2.7x and 2.5x faster than a batch of 8 on this build. Its host share is still 70% to 80%,
  mostly the host SwiGLU and the matmul round trips.

## The threshold: in reach, with both halves of item 2

The threshold's method (`compare-parallel.sh --gpu --sessions 8`, 64 tokens) reads aggregate t/s over the wall time
of 8 prefills and 63 decode steps. From the pre-tier reading
[`20261108T032429Z-gpu-parallel-baseline`](../20261108T032429Z-gpu-parallel-baseline/INDEX.md) (P = 1: 122.63 and 49.54
t/s) and this reading's single step, each request's prefill and turnaround is about 86 ms (TinyLlama) and 182 ms
(Qwen2.5-3B). Then 1.50x needs a step of 8 requests of at most **32.5 ms** (4.69x the single step) and **83.7 ms**
(4.75x). Estimates, with that 8-prefill term held fixed:

| Step of 8 requests | TinyLlama step ms, ratio | Qwen2.5-3B step ms, ratio |
|---|---|---|
| Today (measured above) | 106.26, 0.56x (the reading measured 0.652x) | 267.34, 0.55x (the reading measured 0.668x) |
| Today's per-row kernels with no host time at all (device-busy at N = 8) | 43.95, **1.19x** | 122.84, **1.10x** |
| The tiled GEMM on today's host-driven path (step at N = 9) | 43.74, **1.19x** | 119.97, **1.12x** |
| A weight-shared kernel in a device-resident batched step (device-busy at N = 9, plus the single step's host time) | 14.54, **2.52x** | 37.06, **2.65x** |

The model reads today's ratio about 15% below the measured one: this reading's requests are twice as long (128
tokens, so later steps attend over more positions) and its JFR recording adds an event per operation (1,260
`juno.MatVec` events per step at N = 8). Its estimates therefore err low.

Neither half alone reaches the bar: per-row kernels cap it near 1.1x to 1.2x even with the host removed, and the
weight-shared kernel on the host-driven path reaches about 1.2x. Together they read about 2.5x. The latency bar is
then met as well: a batch's sessions all finish with it, at about 1.66 s and 3.84 s against P = 1's mean of 4.09 and
10.12 s. These are estimates from measured terms, not a run. **No threshold moves.**

**A constraint for item 4 (batched equals single).** A weight-shared kernel sums in another order than the single-row
`q4k_gemv`, so its rows can differ from the single request's in the last bits, and greedy replies can part. Today
the P = 8 replies already differ on TinyLlama and Qwen2.5-3B (pre-tier reading). Item 2's kernel either reproduces the
single-row kernel's per-row arithmetic exactly, or the single step takes the same kernel.

## Files

- `tools/`: the driver, the two readers and `run-all.sh` (`NS="9 16"` for the wider batches).
- `runs/<model>-n<N>-step.json` / `-step.txt`: the untraced step and host spans. `runs/<model>-n<N>-trace.json` /
  `-trace.txt`: the traced step, kernels by name and CUDA API calls. `*.out`, `*.log`: each run's output and server
  log; `*-measured-<i>.json`: the measured replies.
- The raw traces and recordings (`*.nsys-rep`, `*.sqlite`, `*.jfr`, 751 MB) are kept outside the repository, in
  `dist/batched-gpu-decode-baseline/decomposition-raw/`.
- `progress.txt`: one line per configuration (its N = 9 and 16 trace columns predate the corrected split; the
  `-trace.json` files are the corrected ones).
