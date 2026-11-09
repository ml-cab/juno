# GPU `--parallel` 8 against 1 on the four sweep models, with a decode batch in the device region (2026-11-09)

**Purpose.** The batched GPU decode work's threshold reading: 8 concurrent clients against a server at `--parallel 1`
(each request decoded on its own, in the single-request device region) and at `--parallel 8` (one decode batch, now in
the batch device region), on every sweep model, by the method of the pre-tier reading
([`20261108T032429Z-gpu-parallel-baseline`](../20261108T032429Z-gpu-parallel-baseline/INDEX.md)). The bars are
`--parallel 8` aggregate t/s at least 1.50x `--parallel 1`, mean latency per session at most 1.00x, and no batched reply
differing from the single-request reply.

**Pinned: no.** The threshold's method is unpinned (same hour, the flag alternated, median of three).

**Build.** `dist/batched-gpu-decode-gate/candidate-shaded.jar` as it stood for this run, sha256 first 16
`edf0cea7f8e24b74`: the pre-tier tree (`796de19984b3e9ae`) plus the batch device region (`ResidentBatchDecode`,
`BatchedDecodeKernels`, `batched_decode.ptx`). The jar the gate runs on (`789c5204c9af15ab`) differs from it only in
`--help` and diagnostic text. OpenJDK 25.0.4.1; NVIDIA GeForce GTX 1080 (8 GB, driver 580.178.04), clocks not locked
(`gpu-before.txt`). 2026-11-09, 00:56 to 01:11 UTC.

## Method

The pre-tier reading's driver, unchanged (`concurrency-parallel-gpu.sh`): `compare-parallel.sh --gpu --sessions 8
--parallel-values P`, P = 1 or 8; 64 tokens per session, the script's greedy prompt, batch window 50 ms; two untimed
concurrent warmup rounds per server; heap 4 GB (TinyLlama), 6 GB (Qwen2.5-3B, Phi-3.5-mini), 9 GB (Mistral 7B); three
repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even. Every run succeeded with no failed session
and 512 completion tokens.

`rest-consistency/`: the batched-equals-single bar on the consistency prompts, through REST: a server at `--parallel 8`,
each of the three prompts asked alone at 64 tokens, then 8 concurrent requests cycling the prompts, each reply compared
with the same prompt's single-request reply.

## Results

| Model | Aggregate t/s, P = 1 (runs) | P = 1 median | P = 8 (runs) | P = 8 median | **8 over 1** (bar >= 1.50) | Mean latency, P = 1 / P = 8 (medians) | Latency 8 over 1 (bar <= 1.00) | Pre-tier 8 over 1 |
|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 135.0 / 136.0 / 135.3 | 135.27 | 311.6 / 313.3 / 318.4 | 313.34 | **2.316** | 3.70 s / 1.60 s | **0.433** | 0.652 |
| Qwen2.5-3B | 55.3 / 54.5 / 53.8 | 54.47 | 107.5 / 107.5 / 103.9 | 107.50 | **1.973** | 9.09 s / 4.73 s | **0.521** | 0.668 |
| Phi-3.5-mini | 50.1 / 49.9 / 50.1 | 50.08 | 96.0 / 95.2 / 97.6 | 96.01 | **1.917** | 9.93 s / 5.29 s | **0.533** | 0.769 |
| Mistral 7B | 34.6 / 33.2 / 32.1 | 33.20 | 58.6 / 59.3 / 56.2 | 58.65 | **1.767** | 15.00 s / 8.69 s | **0.580** | 0.714 |

- **Both throughput bars are met on every sweep model**: 1.77x to 2.32x aggregate (bar 1.50x), mean latency 0.43x to
  0.58x (bar 1.00x).
- **Replies.** Across all three repetitions every reply is one text per model at P = 1 and the same text at P = 8. In
  the pre-tier reading TinyLlama and Qwen2.5-3B differed between the two.
- **Consistency prompts through REST** (`rest-consistency/results.json`): 0 of 8 batched replies differ from the
  single-request reply, at 64 tokens, on every sweep model.
- **Against the pre-tier reading**, in aggregate t/s at P = 8: TinyLlama 79.90 to 313.34, Qwen2.5-3B 33.08 to 107.50,
  Phi-3.5-mini 36.46 to 96.01, Mistral 7B 23.05 to 58.65. The P = 1 lane is the unchanged single-request path.
- **Scope.** One host and one GPU, a 2016 consumer card; the static schedule only. The pinned no-regression gate on the
  single-request paths is a separate, owner-run reading.

## Files

- `concurrency-parallel-gpu.sh`: the driver. `progress.txt`: one line per run. `driver.log`: the driver's output.
- `<model>/p<P>-rep<N>/`: each run's `compare-parallel.sh` output (`parallel-<P>.json`, its log, the 8 replies).
- `rest-consistency/`: the driver (`rest-consistency.py`), each model's server log, `results.json`.
- `gpu-before.txt`: the GPU's driver, maximum SM clock and power limit before the runs. `started.txt`: start and end.
