# CPU `--parallel`: 8 concurrent clients, batched against unbatched (2026-10-30)

**Purpose.** The comparison that the `--parallel` advice in `docs/howto.md` ("recommend `8` for API servers") rests
on, measured on the CPU for the first time since the integer kernels. The earlier readings compared P batched sessions
with one session alone (`../20261029T182639Z-cpu-decode-batch/`, `../20261029T203047Z-cpu-decode-batch-gate/`). This
one serves the same 8 concurrent clients both ways:

- at `--parallel 1` the scheduler decodes each request on its own thread, concurrently (it does not queue them);
- at `--parallel 8` it decodes them as one decode batch.

Owner decision 2026-10-29: measure this, then decide the docs.

**Pinned: no.** No gate reads this; it informs a docs decision. Same hour, the flag alternated, median of three.

**Build.** One jar: `06d6d22d` plus the uncommitted decode-batch work (player shaded jar sha256 first 16
`1c181de29e52d249`), the candidate of the pinned gate. OpenJDK 25.0.4.1; host: 12 CPU threads, governor `schedutil`,
turbo on, CPU integer kernels at 128-bit lanes.

## Method

`concurrency-parallel.sh` (copied here, run with `JUNO_ROOT` set) runs the tree's `compare-parallel.sh` against the
jar:

- `--cpu --sessions 8 --parallel-values P`, P = 1 or 8;
- 64 tokens per session, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server;
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- three repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even;
- 2026-10-30, 03:06 to 04:02 UTC, just after a host reboot.

Every run reports success, no failed session, and 512 completion tokens.

## Results

| Model | `--parallel` | Aggregate t/s, runs | Median | Mean latency per session, runs (s) | Median |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 9.80 / 10.89 / 10.81 | 10.81 | 52.1 / 46.9 / 47.2 | 47.2 |
| TinyLlama 1.1B | 8 | 10.33 / 10.74 / 11.02 | 10.74 | 48.4 / 47.5 / 45.7 | 47.5 |
| Qwen2.5-3B | 1 | 3.93 / 3.92 / 3.95 | 3.93 | 130.0 / 130.1 / 129.4 | 130.0 |
| Qwen2.5-3B | 8 | 3.97 / 3.94 / 3.94 | 3.94 | 128.5 / 129.7 / 129.8 | 129.7 |

- **`--parallel 8` over `--parallel 1`: 0.993 (TinyLlama) and 1.003 (Qwen2.5-3B) in aggregate t/s; mean latency
  equal within 1%.** On this host's CPU the setting changes neither throughput nor latency for 8 concurrent clients.
- **Concurrent requests share one stream's rate either way.** 8 unbatched clients read 10.81 t/s on TinyLlama; one
  client alone read 10.44 to 10.60 in the unpinned A/B the day before. CPU decode here is bound by arithmetic per
  weight, and the decode-batch matmul repeats the single-request kernel's arithmetic per row, so batching saves
  memory traffic only, which this host does not need.
- **Replies.** All 96 replies, at both settings, equal the build's single-request reply (P=1, one session, in
  `../20261029T182639Z-cpu-decode-batch/`). Concurrent unbatched requests do not change each other's output, and
  with the decode-batch matmul batched ones do not either.
- **Scope.** One host: a 6-core CPU without AVX2, decode limited by arithmetic, not memory bandwidth. On a CPU where
  decode is limited by memory bandwidth, batching should gain, because a weight row is read once per batch; not
  measured. The GPU is not measured here; its last reading of the same workload is 1.11x
  (`../20260901T173121Z-parallel/`), taken long before the current GPU decode paths. Measured the same day since:
  `../20261030T062550Z-gpu-parallel-concurrency/` (0.90x and 0.87x).

## Files

- `concurrency-parallel.sh`: the driver.
- `progress.txt`: every reading, in run order.
- `<model>/p<P>-rep<N>/parallel-<P>.json` and `replies/`: per-run results and replies, with local paths replaced by
  `<repo>`, `<run-dir>` and `<scratch>`.
