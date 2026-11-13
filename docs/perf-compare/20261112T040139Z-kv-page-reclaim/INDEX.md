# Idle KV pages released: pause, real-model check, same-session A/B

**Purpose.** Evidence for releasing idle paged-KV pages (`--schedule continuous`): a page every request has released stays
in the pool and is reused first; one left unused for a whole reclaim period gives its storage back to the heap, from a
background thread (`KvBlockPool.reclaimIdle`, `KvPageReclaimer`, `-Djuno.kv.reclaimPeriodMs`, default 60000). It checks the
pause budget (a pass adds at most 5 ms to any decode step at ctx 8192 / batch 8 / page 16), that pages are released and
replies unchanged on a real model, that `--lora-play` now releases a finished request's KV (it kept it before), and the
standing throughput and allocation bars on the paged lanes against the build before the change.

**Pinned: no.** Unpinned, same host, one run at a time, nothing else measuring (CPU governor `schedutil`, turbo on, GPU clocks
not pinned). Not the pinned standing-gate run, which is the owner's at the close of the KV work.

**Host.** Intel Xeon E5-1650 v2 (6 cores, 12 threads, AVX, F16C), 62.7 GiB; NVIDIA GTX 1080.

**Builds.**

| Side | Tree | Shaded jar (`juno-player-0.1.2-shaded.jar`, SHA-256, first 16) |
|---|---|---|
| A, baseline | `6ad5c728`, built from `git archive` in scratch (`mvn -q package -DskipTests`) | `03820803bf46dce6` |
| B, candidate | `6ad5c728` plus the change | `f9b39bf898e734e2` |

## Pause microbench

```bash
scripts/performance-tests/kv-reclaim-microbench.sh <out>
```

Each sweep model's K and V pools at ctx 8192, batch 8, page 16: eight live requests and eight finished ones, built interleaved
so the finished requests' idle pages sit among the live ones, then the finished ones released. Rows store 16 floats: the pass
drops the idle pages' storage and touches none of their bytes, so its cost is per page, and the model sets the page count
(layers x requests x 512 pages, K and V).

Run 2, the final code (`microbench/run2-kv-reclaim-20261112T040139Z.md`):

| model | layers | idle pages | pass on its own, median | pass on its own, max | steps overlapping a pass | longest overlapping step | median step alone | added_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tinyllama | 22 | 180,224 | 3.381 ms | 13.300 ms | 72 | 0.754 ms | 0.045 ms | **0.709** |
| qwen2.5-3b | 36 | 294,912 | 2.991 ms | 5.013 ms | 69 | 0.782 ms | 0.062 ms | **0.720** |
| phi-3.5-mini | 32 | 262,144 | 4.451 ms | 5.413 ms | 81 | 0.982 ms | 0.052 ms | **0.930** |
| mistral-7b | 32 | 262,144 | 4.498 ms | 4.867 ms | 77 | 1.008 ms | 0.051 ms | **0.957** |

**Gate met: at most 0.957 ms added to a decode step (budget <= 5 ms).** `added_ms` is the longest decode step's KV writes
(one row appended to every live table, the part of a step that takes the pools' lock) that overlapped a pass running on its
own thread, as the server runs it, less the median step alone. The pass timed on its own is what a step would wait if the pass
ran on the decode thread, which the server does not do; the 13.3 ms is TinyLlama's first, interpreted pass.

Run 1 (`microbench/run1-kv-reclaim-20261112T035917Z.md`) measured the first form of the code: free page ids in an
`ArrayDeque<Integer>` (each released id boxed again), lock holds of 1,024 pages, no yield between them. Its pass on its own read
4.6 to 5.6 ms median and 17.6 ms max, and with another thread allocating, freeing and reclaiming in the same pools a step's KV
writes reached 3.7 to 5.9 ms against 0.4 to 0.6 ms alone: the pool's lock is not fair, so the reclaimer took it again before a
waiting writer. The final form keeps ids as `int`s (`IntRing`), holds the lock for at most 256 pages and yields between holds;
run 1's contended column also counted that churn thread's own allocations, and run 2 replaces it with steps overlapping a real
pass.

## Real model: pages released, replies unchanged

```bash
real-model/kv-reclaim-check.sh <B jar> <out> cpu|gpu [lora]
```

TinyLlama 1.1B Q4_K_M, `--schedule continuous --parallel 8 --prefix-cache-tokens 0`, `-Djuno.kv.reclaimPeriodMs=1000` (one
second, so that the reading does not wait minutes). A greedy request and a session's first turn, a burst of eight concurrent
requests, three seconds idle, then the same request and the session's next turn. Health counters after each stage
(`real-model/result-*.txt`):

| stage | CPU live / idle / reclaimed / MB held | GTX 1080 live / idle / reclaimed / MB held | CPU `--lora-play` live / idle / reclaimed / MB held |
|---|---|---|---|
| after one request and a session turn | 1,188 / 0 / 1,108 / 19.5 | 1,188 / 0 / 44 / 19.5 | 1,188 / 0 / 1,176 / 19.5 |
| after the burst of eight | 1,188 / 9,504 / 1,108 / 175.2 | 1,188 / 9,504 / 44 / 175.2 | 1,188 / 4,752 / 5,928 / 97.3 |
| idle for over two periods | 1,188 / **0** / 10,612 / **19.5** | 1,188 / **0** / 9,548 / **19.5** | 1,188 / **0** / 10,680 / **19.5** |
| greedy reply after the reclaim | equal to the one before | equal to the one before | not reached (see below) |
| session's next turn | answered, own-prefix hit counted | answered, own-prefix hit counted | not reached |

- Live pages stay at the session's 1,188 throughout: a finished request's pages are released, and the session's are never
  touched by the reclaim (its next turn resumes from them, `prefixHits` 1).
- Under `--lora-play` the live count also stays at the session's 1,188 across the first request and the burst: the LoRA
  handlers now release a finished request's KV. Before this change they kept it, so every stateless request would have added
  its pages (about 1,176 here) to the live count for the life of the process. Under `--lora-play` a decode batch runs one
  request at a time, so the burst took most of the run and the script's 25-minute limit stopped it after the idle stage, before
  the two closing requests; `PagedKvReclaimHandlerTest` checks replies across a reclaim on the three LoRA handlers.
- `kvPageBytes` counts the live and idle pages' storage; on the CPU run the first request's 1,108 pages were already released
  while the session's turn ran, since the period was one second.

## Same-session A/B

```bash
ab/ab-run.sh   # A B A B A B; for each side:
scripts/performance-tests/compare-llama-cpp.sh --cpu --models tinyllama-1.1b-chat-v1.0.Q4_K_M --n-prompt 512 \
  --schedule continuous --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <side's jar> --out <dir>
scripts/performance-tests/compare-lora.sh --cpu|--gpu --skip-build --no-publish --out <dir>   # each side's own tree
```

Medians of three (`ab/`):

| Lane | A | B | B/A | Bar | Result |
|---|---|---|---:|---|---|
| TinyLlama `continuous` F16, `n_prompt` 512: pp t/s | 31.49 (31.92, 31.49, 31.33) | 31.52 (31.52, 31.41, 31.73) | 1.001 | >= 0.95 | met |
| same: tg t/s | 10.28 (10.35, 10.28, 10.14) | 10.29 (10.29, 10.46, 10.16) | 1.001 | >= 0.95 | met |
| same: allocation per token in the recording | 305,795 B | 312,530 B | 1.022 | <= 1.10 | met |
| same: in-span GC pause total | 0 ms (all three) | 0 ms (all three) | | <= 5 ms | met |
| same: greedy replies | | | equal, 6 of 6 | | |
| LoRA CPU, train ms per pass | 58,133 (57,733, 58,133, 58,200) | 58,133 (58,133, 58,133, 58,000) | 1.000 | <= 1.25 | met |
| LoRA CPU, playback t/s | 0.757 (0.759, 0.757, 0.756) | 0.763 (0.770, 0.762, 0.763) | 1.008 | >= 0.80 | met |
| LoRA GPU, train ms per pass | 3,000 (all three) | 3,000 (all three) | 1.000 | <= 1.25 | met |
| LoRA GPU, playback t/s | 11.44 (11.44, 11.48, 11.35) | 11.29 (11.29, 11.29, 10.85) | 0.987 | >= 0.80 | met |
| LoRA recall | 3 of 3, CPU and GPU | 3 of 3, CPU and GPU | | | |

- pp on the continuous lane is `prompt_tokens * 1000 / jfr.forward_pass_prefill_total_ms` and tg is
  `jfr.token_gen_tps_decode_derived`, as in `20261111T051528Z-paged-kv-in-place`: the harness withholds the lane JSON's own
  `prompt_eval_tps`/`token_gen_tps` on the continuous schedule, on both builds ("JFR timestamps disagree with the engine clock":
  about 0.79 s of the 17 s chunked prefill and 3.27 s of the 10 s generation fall outside the forward-pass spans).
- Allocation is `jfr.allocated_bytes_per_token_in_recording` from the generate lane, the standing gate's field.
- The `q8_0` continuous lane was not run: the pool change is the same code for both page types. A first attempt with
  `--models tinyllama` matched the Q2_K file as well and was stopped and restarted with the exact basename.
- Paths in the copied JSON are scrubbed to `<repo>`, `<scratch>`, `<baseline-tree>` and `<home>`.

## Regression smoke

`JUNO_JAR=<B jar> scripts/performance-tests/smoke-consistency.sh`, unmodified: 56 of 56 checks pass (`smoke-consistency.log`).

## How to read it

The pause gate is the microbench's `added_ms`, per model. The real-model table shows memory held at the peak and after idling,
from the server's own health counters. The A/B holds the paged lanes and LoRA to the standing bars; the pinned run at the close
of the KV work is the one that scores the standing gate.
