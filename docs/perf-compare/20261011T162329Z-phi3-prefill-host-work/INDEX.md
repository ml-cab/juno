# Phi-3.5-mini prefill window: host KV allocation and the host KV write removed - 20261011T162329Z (gpu)

**Purpose.** The prefill window after the two host terms the decomposition
([`20261011T055415Z-gpu-host-overhead-decomposition`](../20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md))
ranked first were removed, read the same way (`compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane`,
Phi-3.5-mini, `n_prompt` 512 and 2048 at `COMPARE_HEAP=8g`, then `prefill-breakdown.sh`):

1. **Host KV tensors reused across requests** (`kv-reuse/`, jar `f10fdb263810ca13`). An evicted request's dense host
   KV tensors go to the next request instead of being allocated again: the allocation (64 tensors of 6.3 MB at 512
   tokens, 25 MB at 2048) and the G1 humongous-allocation collections it triggered were the window's unattributed
   residue.
2. **K and V written straight into the host tensors** (`kv-direct-write/`, jar `921ab742ba3f362a`, which includes 1).
   The region's download of a window's K and V rows is copied once, in bulk, from the download buffer into the host
   KV tensor, instead of into per-row arrays and then again into the tensor.

**Pinned: no.** Unpinned (schedutil, turbo on), median of three, span recording on. Readings, not gates: the tier's
pinned closing sweeps score the rows.

**Builds.** Before: jar `6ae135b0e21839cf` (HEAD `198467ad`; the decomposition). After: the two jars above, both
`198467ad` plus this tier's uncommitted changes, Qwen2's decode region included (it does not touch Phi-3.5-mini).

## Window (ms, median of three)

| Term | 512 before | 512 reuse | 512 reuse + direct write | 2048 before | 2048 reuse | 2048 reuse + direct write |
|---|---|---|---|---|---|---|
| **Window** | 767.4 | 659.6 | **602.2 (-21.5%)** | 3628.2 | 3348.5 | **3096.5 (-14.7%)** |
| GEMM compute | 333.6 | 321.4 | 324.8 | 1376.5 | 1318.1 | 1324.7 |
| Region attention | 76.5 | | 73.7 | 1099.2 | | 1052.2 |
| Device elementwise | 70.1 | | 69.2 | 237.9 | | 234.5 |
| Region host side | 84.4 | 75.1 | 72.8 | 263.7 | 270.3 | 260.6 |
| Host KV write | 69.3 | 62.0 | **0.1** | 249.8 | 260.9 | **0.1** |
| Matmul staging | 42.3 | 43.8 | 44.0 | 164.1 | 163.7 | 164.3 |
| **Residue** | 84.3 (11.0%) | **15.0 (2.3%)** | | 226.4 (6.2%) | **50.8 (1.5%)** | |

Empty cells: see that column's `pp*-breakdown.md`. The residue is now under `prefill-breakdown.sh`'s 5% limit at both
lengths.

## Against the raised rows (projection)

The pinned reference reads Phi-3.5-mini's pp ratio 0.594x at 512 and 0.540x at 2048. With Juno's window 21.5% and
14.7% shorter, the projection is **about 0.757x** (row: >= 0.70x) and **about 0.633x** (row: >= 0.60x). The same-session
unpinned readings of these runs, span recording on, are 0.690x and 0.623x (`kv-direct-write/run*/RUN-INDEX.md`); the
pinned closing sweep, without span recording, scores the rows. Phi-3.5-mini decode in the same runs: 0.970x and 0.965x.

## Correctness

- `PrefillRegionHandlerParityTest` (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Qwen3-1.7B): logits bit-identical, region on
  against off, over two prefill windows, a continuation window and decode; and now every layer's host K and V tensors
  bit-identical as well, since the region writes them itself.
- `HostKvReuseLiveTest` (TinyLlama, Phi-3.5-mini, Qwen3-1.7B, CPU): the second request takes the evicted one's
  tensors, its logits equal those on new tensors bit for bit, and its window allocates at least the KV's bytes less
  (Phi-3.5-mini 59.9 MB to 9.6 MB for a 40-token window with 50.3 MB of KV).
- A reused tensor reads zero wherever the new request has not written (`SessionKvReuseTest`, `KvWriteRowsTest`).

## Files

`kv-reuse/pp*-breakdown.*`, `kv-direct-write/pp*-breakdown.*`, and `kv-direct-write/run512|run2048/` (the harness's JSON
and `RUN-INDEX.md`). Local paths removed.
