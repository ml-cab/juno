# GPU `--parallel` 8 against 1 on the four sweep models, before batched decode moves to the device (2026-11-08)

**Purpose.** The pre-tier reading of the batched GPU decode work: 8 concurrent clients against a server at
`--parallel 1` (each request decoded on its own, in the device-resident decode region) and at `--parallel 8` (one
decode batch, today's host-driven batched step), on every sweep model. The tier's threshold is read by this method:
`--parallel 8` aggregate t/s at least 1.50x `--parallel 1`, and mean latency per session at most 1.00x.
`../20261031T040116Z-gpu-parallel-window/` took the same reading on two models.

**Pinned: no.** The threshold's method is unpinned (same hour, the flag alternated, median of three).

**Build.** One jar, the pre-tier jar: `dist/batched-gpu-decode-baseline/baseline-shaded.jar`, sha256 first 16
`796de19984b3e9ae`, a clean build of `d8012b11` plus the uncommitted work of 2026-11-05 to 2026-11-07 (none of it in the
decode path). OpenJDK 25.0.4.1; NVIDIA GeForce GTX 1080 (8 GB, driver 580.178.04), clocks not locked (`gpu-before.txt`).

## Method

The earlier reading's driver with the two other sweep models added (`concurrency-parallel-gpu.sh`, run with
`JUNO_ROOT` set):

- `compare-parallel.sh --gpu --sessions 8 --parallel-values P`, P = 1 or 8;
- 64 tokens per session, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server;
- heap 4 GB (TinyLlama), 6 GB (Qwen2.5-3B, Phi-3.5-mini), 9 GB (Mistral 7B);
- three repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even;
- 2026-11-08, 03:24 to 03:45 UTC.

Every run reports success and no failed session. Each run has 512 completion tokens, except the Qwen2.5-3B runs at
P = 8 with 504, as in the earlier reading.

## Results

| Model | Aggregate t/s, P = 1 (runs) | P = 1 median | P = 8 (runs) | P = 8 median | **8 over 1** | Mean latency, P = 1 / P = 8 (medians) | Latency 8 over 1 |
|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 122.63 / 124.06 / 120.87 | 122.63 | 79.24 / 79.90 / 83.02 | 79.90 | **0.652** | 4.09 s / 6.38 s | 1.559 |
| Qwen2.5-3B | 54.50 / 49.54 / 47.53 | 49.54 | 33.36 / 33.08 / 31.29 | 33.08 | **0.668** | 10.12 s / 15.21 s | 1.502 |
| Phi-3.5-mini | 47.39 / 47.41 / 47.67 | 47.41 | 36.46 / 36.72 / 36.38 | 36.46 | **0.769** | 10.46 s / 14.00 s | 1.338 |
| Mistral 7B | 33.90 / 32.27 / 31.49 | 32.27 | 23.05 / 23.15 / 22.95 | 23.05 | **0.714** | 15.47 s / 22.18 s | 1.434 |

- **A batch of 8 decodes slower than 8 requests decoded on their own, on every sweep model**: 0.652x to 0.769x in
  aggregate t/s, mean latency 1.34x to 1.56x. The tier's bars are 1.50x and 1.00x.
- **Every P = 8 run is one batch of 8.** In each run the 8 requests' server latencies (`x_juno_latency_ms`) lie within
  19 ms of each other.
- **Replies.** At P = 1 every reply is one text per model, and at P = 8 one text per model, the same in all three runs.
  On Phi-3.5-mini and Mistral 7B the two texts are the same. On TinyLlama and Qwen2.5-3B the P = 8 text differs from
  the P = 1 text: the batched step's projections and attention run other kernels than the single-request region's.
  The tier's "batched equals single" bar is read on the consistency prompts; this prompt already shows 2 of 4 models
  differing.
- **Against the earlier reading** (another build, unpinned clocks): TinyLlama 0.748x then, 0.652x now, with P = 1
  higher (122.63 against 108.66 t/s) and P = 8 about the same (79.90 against 81.23); Qwen2.5-3B 0.676x then, 0.668x now.
- **Scope.** One host and one GPU, a 2016 consumer card; static schedule only.

## Files

- `concurrency-parallel-gpu.sh`: the driver. `progress.txt`: one line per run. `driver.log`: the driver's output.
- `<model>/p<P>-rep<N>/`: each run's `compare-parallel.sh` output (`parallel-<P>.json`, its log, the 8 replies).
- `gpu-before.txt`: the GPU's driver, maximum SM clock and power limit before the runs. `started.txt`: start and end.
