# GPU `--parallel`: 8 concurrent clients with the batch window held open (2026-10-31)

**Purpose.** The GPU reading of `../20261030T062550Z-gpu-parallel-concurrency/`, repeated after the static
collector was made to hold a batch open for `--batch-window-ms` (owner decision 2026-10-30). That reading found
that `RequestScheduler.collectBatch` took only the requests already queued when the first one arrived, so 8
near-simultaneous requests usually split into a batch of 5 or 6 and a late batch of 2 or 3. The collector now
dispatches a batch when `batchWindowMs` has passed since its first request or when it is full, as `BatchConfig`
documents.

**Pinned: no.** No gate reads this; it informs the `--parallel` docs and the owner's keep-or-revert decision. Same
hour, the flag alternated, median of three.

**Build.** One jar: `06d6d22d` plus the uncommitted work of this branch, the collector change included (player
shaded jar sha256 first 16 `dc12d31ed75320af`). OpenJDK 25.0.4.1; NVIDIA GeForce GTX 1080 (8 GB, driver 580.178.04),
clocks not locked.

## Method

The earlier reading's driver, unchanged (`concurrency-parallel-gpu.sh`, copied here, run with `JUNO_ROOT` set):

- `compare-parallel.sh --gpu --sessions 8 --parallel-values P`, P = 1 or 8;
- 64 tokens per session, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server;
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- three repetitions per model, P = 1 then 8 on odd repetitions and 8 then 1 on even;
- 2026-10-31, 04:01 to 04:09 UTC.

Every run reports success and no failed session. Each run has 512 completion tokens, except the Qwen2.5-3B runs at
P = 8 with 504.

## Results

| Model | `--parallel` | Aggregate t/s, runs | Median | Mean latency per session, runs (s) | Median |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 114.41 / 108.66 / 108.61 | 108.66 | 4.33 / 4.58 / 4.55 | 4.55 |
| TinyLlama 1.1B | 8 | 81.23 / 81.74 / 79.23 | 81.23 | 6.27 / 6.23 / 6.43 | 6.27 |
| Qwen2.5-3B | 1 | 47.90 / 48.90 / 47.63 | 47.90 | 10.46 / 10.18 / 10.40 | 10.40 |
| Qwen2.5-3B | 8 | 32.36 / 32.66 / 31.86 | 32.36 | 15.55 / 15.40 / 15.79 | 15.55 |

- **`--parallel 8` over `--parallel 1`: 0.748 (TinyLlama) and 0.676 (Qwen2.5-3B) in aggregate t/s**, against 0.901
  and 0.866 before the change. Mean latency is now higher at P = 8 (1.38x and 1.50x), where it was lower before.
- **Every P = 8 run is one batch of 8.** In each run the 8 requests' server latencies (`x_juno_latency_ms` in
  `replies/`) lie within 10 ms of each other (over the three runs: TinyLlama 6.23 to 6.44 s, Qwen2.5-3B 15.40 to
  15.79 s).
- **A batch of 8 decodes slower on this GPU than 8 unbatched concurrent requests.** Batched decode
  (`forwardMultiDecode`) runs each layer's matmuls from the host over the batch's activations; the device-resident
  decode region that a single request's decode uses does not serve batches. Eight unbatched requests each take the
  device-resident path and together keep the GPU busy, so they read at about one stream's rate in aggregate (108.66
  and 47.90 t/s). Before the change, most requests of a P = 8 run sat in a first batch that ran beside a late one, and
  the one run in which all 8 made one batch read 1.04x; the late batch was not the only cost, and this reading does not
  separate the two.
- **Replies.** At P = 1 every reply is one text per model, the same as before. At P = 8 every reply is one other text
  per model, the same in all three runs. That other text is the one the earlier reading got from its late batches of 2
  or 3 and from fresh-server batches of 2 and 3; its one full batch of 8 gave the P = 1 text. What decides between the
  two texts is not traced here. GPU-resident layers decode a batch with another kernel than a single request, and the
  results can differ in the last bits (`docs/howto.md`).
- **Scope.** One host and one GPU, a 2016 consumer card; static schedule only.

## Files

- `concurrency-parallel-gpu.sh`: the driver.
- `gpu-before.txt`: the GPU's state before the first run.
- `progress.txt`: every reading, in run order.
- `<model>/p<P>-rep<N>/parallel-<P>.json` and `replies/`: per-run results and replies, with local paths replaced by
  `<repo>`, `<run-dir>` and `<scratch>`.
