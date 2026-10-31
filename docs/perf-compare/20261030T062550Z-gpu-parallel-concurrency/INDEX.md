# GPU `--parallel`: 8 concurrent clients, batched against unbatched (2026-10-30)

**Purpose.** The GPU half of the comparison behind the `--parallel` advice in `docs/howto.md` ("recommend `8` for API
servers"), measured the way `../20261030T040305Z-cpu-parallel-concurrency/` measured the CPU. Here too the same 8
concurrent clients are served two ways:

- at `--parallel 1` the scheduler decodes each request on its own thread, concurrently, on the GPU;
- at `--parallel 8` it gathers them into static micro-batches.

The hardware validation kit's `gpu.parallel.tinyllama` probe prompted this reading: its first real-model run read
`--parallel 8` at 0.74x to 0.92x of `--parallel 1`, flagged as noisy. The last published GPU reading was 1.11x
(`../20260901T173121Z-parallel/`), taken before the current GPU decode paths.

**Pinned: no.** No gate reads this; it informs a docs decision. Same hour, the flag alternated, median of three.

**Build.** One jar: `06d6d22d` plus the uncommitted work of this branch (player shaded jar sha256 first 16
`8fa44602112f19c4`; the engine sources are those of `1c181de29e52d249`, the jar of the CPU reading). OpenJDK 25.0.4.1;
NVIDIA GeForce GTX 1080 (8 GB, driver 580.178.04), clocks not locked.

## Method

`concurrency-parallel-gpu.sh` (copied here, run with `JUNO_ROOT` set) is the CPU reading's driver with `--gpu`:

- `compare-parallel.sh --gpu --sessions 8 --parallel-values P`, P = 1 or 8;
- 64 tokens per session, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server;
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- three repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even;
- 2026-10-30, 06:25 to 06:34 UTC.

Every run reports success and no failed session. Each run has 512 completion tokens, except two Qwen2.5-3B runs at
P = 8 with 510.

Afterwards, one check of where the reply differences come from (below). Each case got a fresh server and TinyLlama at
`--parallel 8` with no warmup, and sent 2 concurrent requests, then 3 (`fresh/`).

## Results

| Model | `--parallel` | Aggregate t/s, runs | Median | Mean latency per session, runs (s) | Median |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 110.68 / 111.96 / 112.95 | 111.96 | 4.44 / 4.46 / 4.31 | 4.44 |
| TinyLlama 1.1B | 8 | 101.09 / 100.91 / 98.14 | 100.91 | 3.83 / 3.85 / 3.70 | 3.83 |
| Qwen2.5-3B | 1 | 49.21 / 49.01 / 47.73 | 49.01 | 10.09 / 9.97 / 10.49 | 10.09 |
| Qwen2.5-3B | 8 | 42.46 / 51.02 / 41.11 | 42.46 | 8.97 / 9.64 / 9.34 | 9.34 |

- **`--parallel 8` over `--parallel 1`: 0.901 (TinyLlama) and 0.866 (Qwen2.5-3B) in aggregate t/s. Mean latency is
  lower at P = 8: 0.862 and 0.926.**
- **The 8 requests rarely make one batch.** The static scheduler's collector (`RequestScheduler.collectBatch`) waits
  up to the batch window for a *first* request. It then takes only the requests already queued at that moment; it
  does not hold the batch open for the window. In five of the six P = 8 runs, 2 or 3 of the 8 near-simultaneous
  requests were queued a moment too late (TinyLlama: 2, 2, 3; Qwen2.5-3B: 2, 0, 2) and formed a second, small batch.
  Its replies match those of a batch of 2 or 3 on a fresh server (below), and it finished after the first batch.
  Per-request latency in each run (`x_juno_latency_ms` in `replies/`):
  - first batch: TinyLlama 2.7 to 3.5 s, Qwen2.5-3B 7.6 to 8.5 s;
  - late requests: TinyLlama 5.0 to 5.2 s, Qwen2.5-3B 12.0 to 12.4 s.

  The late batch sets the wall time, so aggregate t/s falls. The one run where all 8 made one batch (Qwen2.5-3B,
  repetition 2) read 51.02 t/s, **1.04x** its P = 1 median. `docs/howto.md` and the launcher's `--help` call
  `--batch-window-ms` a "batch collect window", which is not what the collector does.
- **Batching gains little on this GPU even when the batch fills.** A first batch of 5 or 6 TinyLlama requests runs at
  about the rate of 8 unbatched concurrent requests: 6 x 64 tokens in about 3.4 s is 113 t/s, against 112 t/s. Eight
  unbatched requests share one GPU and already keep it busy.
- **Replies.** Every P = 1 reply is the same text per model (24 replies per model across the three runs). At P = 8:
  - the requests decoded in the first batch (batches of 5, 6 and 8) give that same text;
  - every late request (batches of 2 and 3) gives one other text: 7 of 48 replies on TinyLlama, 4 of 48 on Qwen2.5-3B.

  On a fresh server with no earlier batch, a batch of 2 and a batch of 3 give that same other text (`fresh/`). So the
  difference comes from the batch size's arithmetic on the GPU, not from state left by an earlier batch. On the CPU,
  a batched request's greedy output does not depend on the batch (the CPU reading: 96 of 96 equal).
- **Scope.** One host and one GPU, a 2016 consumer card; static schedule only. The continuous schedule admits late
  requests into running decode steps; the kit's `gpu.schedule.tinyllama` read it at 75 t/s against 95 for static at
  `--parallel 8` (run of 2026-10-30, unpinned, one reading each).

## Files

- `concurrency-parallel-gpu.sh`: the driver.
- `gpu-before.txt`: the GPU's state before the first run (idle clocks).
- `progress.txt`: every reading, in run order.
- `<model>/p<P>-rep<N>/parallel-<P>.json` and `replies/`: per-run results and replies, with local paths replaced by
  `<repo>`, `<run-dir>` and `<scratch>`.
- `fresh/s<N>-w0/`: the fresh-server batches of 2 and 3.
