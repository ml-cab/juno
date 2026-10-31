# CPU `--parallel`: 8 concurrent clients with the batch window held open (2026-10-31)

**Purpose.** The CPU reading of `../20261030T040305Z-cpu-parallel-concurrency/`, repeated after the static
collector was made to hold a batch open for `--batch-window-ms` (owner decision 2026-10-30), beside the GPU reading
of the same change (`../20261031T040116Z-gpu-parallel-window/`). The collector now dispatches a batch when
`batchWindowMs` has passed since its first request or when it is full, as `BatchConfig` documents; before, it took
only the requests already queued when the first one arrived.

**Pinned: no.** No gate reads this; it informs the `--parallel` docs and the owner's keep-or-revert decision. Same
hour, the flag alternated, median of three.

**Build.** The GPU reading's jar: `06d6d22d` plus the uncommitted work of this branch, the collector change included
(player shaded jar sha256 first 16 `dc12d31ed75320af`). OpenJDK 25.0.4.1; host: 12 CPU threads, governor
`schedutil`, turbo on, CPU integer kernels at 128-bit lanes.

## Method

The earlier reading's driver, unchanged (`concurrency-parallel.sh`, copied here, run with `JUNO_ROOT` set):

- `compare-parallel.sh --cpu --sessions 8 --parallel-values P`, P = 1 or 8;
- 64 tokens per session, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server;
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- three repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even;
- 2026-10-31, 04:12 to 05:07 UTC.

Every run reports success, no failed session, and 512 completion tokens.

## Results

| Model | `--parallel` | Aggregate t/s, runs | Median | Mean latency per session, runs (s) | Median |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 10.18 / 10.15 / 10.14 | 10.15 | 50.2 / 50.3 / 50.4 | 50.3 |
| TinyLlama 1.1B | 8 | 11.91 / 11.92 / 12.24 | 11.92 | 43.0 / 42.9 / 41.8 | 42.9 |
| Qwen2.5-3B | 1 | 3.94 / 3.97 / 3.93 | 3.94 | 129.6 / 128.8 / 129.9 | 129.6 |
| Qwen2.5-3B | 8 | 4.27 / 4.34 / 4.12 | 4.27 | 119.7 / 118.0 / 124.2 | 119.7 |

- **`--parallel 8` over `--parallel 1`: 1.174 (TinyLlama) and 1.084 (Qwen2.5-3B) in aggregate t/s; mean latency
  0.853x and 0.924x.** Before the change: 0.993 and 1.003, latency equal within 1%.
- **Every P = 8 run is one batch of 8.** In each run the 8 requests' server latencies (`x_juno_latency_ms` in
  `replies/`) lie within 11 ms of each other; at P = 1 they spread over 198 to 637 ms.
- **Replies.** All 96 replies, at both settings, are one text per model, the build's single-request reply: a CPU
  decode batch computes each request with the arithmetic it gets alone.
- **Against the earlier reading.** TinyLlama at P = 1 reads lower here than on 2026-10-30 (10.15 against 10.81 t/s);
  Qwen2.5-3B reads the same (3.94 against 3.93). The ratio is the reading: both settings ran within one hour,
  alternating. The earlier reading did not record how its P = 8 requests split into batches, so why a batch of 8
  gains where the old collector's batches did not is not separated here.
- **Scope.** One host: a 6-core CPU without AVX2. Static schedule only.

## Files

- `concurrency-parallel.sh`: the driver.
- `progress.txt`: every reading, in run order.
- `<model>/p<P>-rep<N>/parallel-<P>.json` and `replies/`: per-run results and replies, with local paths replaced by
  `<repo>`, `<run-dir>` and `<scratch>`.
