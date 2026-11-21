# KV cache tier close: the standing throughput and allocation gate, owner run, pinned

**Purpose.** The owner's pinned no-regression gate for the KV cache tier's final build against the build before it:
generation and prompt throughput, allocation per token and GC pause on the GPU and CPU lanes of the four sweep models,
and the tier's two microbench bars retaken pinned.

**Builds.**
- A, baseline: the pre-tier jar, `git archive e16748ad` built with `mvn package -DskipTests`; sha256 `21109a3a8fd317c8`.
- B, candidate: HEAD `71099a85` plus the tier close's uncommitted changes (live check 12, one reworded refusal message;
  neither on a measured path); sha256 `2ceed60ef9139b7a`, equal to the tree's own shaded jar (`run-gate.sh` checks it).
  `tree.txt`: the tree's diff hash at the start.

**Pinned.** CPU yes: `compare-llama-cpp.sh --pin-clocks` sets the governor to `performance` and turbo off for every run;
every run's `host.json` reads `clock_pinned: true` (12 of 12); Part M ran with the same settings (`micro-pin.txt`). GPU
clock recorded, not fixed: the harness requested a 1,911 MHz lock, which this card accepts and does not apply. The loaded
graphics clock (samples at 30% or more utilisation) had the same median on both jars for every model: TinyLlama
1,822 MHz, Qwen2.5-3B 1,822, Phi-3.5-mini 1,809, Mistral 7B 1,746.

**Method.** `run-gate.sh`, started by `wait-and-run.sh` once `READY` existed: each lane runs A B A B A B
(`compare-llama-cpp.sh --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane`), GPU lane `--n-prompt 512`, CPU lane
`--n-prompt 128`, 64 generated tokens; each metric is the median of three. Bars: prefill and generation t/s B/A >= 0.95;
allocation per token in the recording <= 1.10x; GC pause in the token span <= 1.25x (<= 5 ms under a 5 ms baseline).
The full terminal output is `gate.log`; the scorer's code is in `run-gate.sh`.

## 1. GPU lane, `--n-prompt 512`

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token A | alloc/token B | B/A | GC ms A / B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 3,107.91 | 3,177.21 | **1.022** | 148.88 | 162.86 | **1.094** | 547,378 | 427,105 | **0.780** | 0 / 0 |
| Qwen2.5-3B | 1,270.57 | 1,279.39 | **1.007** | 58.26 | 61.88 | **1.062** | 1,521,118 | 1,478,428 | **0.972** | 0 / 0 |
| Phi-3.5-mini | 857.21 | 915.07 | **1.067** | 57.95 | 59.09 | **1.020** | 515,467 | 426,009 | **0.826** | 0 / 0 |
| Mistral 7B | 561.96 | 584.13 | **1.039** | 38.89 | 39.66 | **1.020** | 755,489 | 536,713 | **0.710** | 0 / 0 |

Every row meets every bar.

## 2. CPU lane, `--n-prompt 128`

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token A | alloc/token B | B/A | GC ms A / B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 33.11 | 33.52 | **1.012** | 11.25 | 11.38 | **1.012** | 304,747 | 309,359 | **1.015** | 0 / 0 |
| Qwen2.5-3B | 11.79 | 11.88 | **1.008** | 3.97 | 4.02 | **1.013** | 1,269,360 | 1,289,163 | **1.016** | 0 / 0 |
| Phi-3.5-mini | 9.28 | 9.44 | **1.017** | 2.88 | 2.89 | **1.004** | 358,741 | 347,850 | **0.970** | 0 / 0 |
| Mistral 7B | 5.13 | 5.10 | **0.995** | withheld | 1.992 (2 of 3) | not scored | 347,919 | 347,984 | **1.000** | 0 / 0 |

`gate.log` ends `GATES MISSED`, on one line: `FAIL: mistral generation below 0.95x`. That row has no ratio to fail: the
harness withheld Mistral 7B's CPU generation reading (`token_gen_tps: null`) on all three baseline runs and on candidate
run 1, and the scorer counts a side with no reading as 0. The harness's span check withholds a run when the JFR spans
and the engine clock disagree by more than its bounds: the token span's residual read -26.2, -29.8 and -26.7 ms on the
baseline runs and -25.7 ms on candidate run 1, against a floor of -25 ms, in requests of about 34 seconds (0.08%).
Two other readings of the same six pinned runs, neither withheld:

| Reading, Mistral 7B CPU generation | A, median of 3 | B, median of 3 | B/A |
|---|---:|---:|---:|
| Request-timed `api_token_gen_tps` | 1.8601 | 1.8511 | **0.995** |
| Decode rate from the forward-pass totals (`jfr.token_gen_tps_decode_derived`: 64 tokens over the decode passes' total) | 1.9877 | 1.9800 | **0.996** |

Both are above 0.95x. How the row is scored is an owner decision, raised in the tier record.

**Scored by the re-run.** Owner decision 21 (b) re-ran this row alone, pinned, with the same jars, method and bars:
generation 1.004x, prefill 0.985x, allocation 1.013x, no GC pause, met
([`20261120T190622Z-kv-cache-close-mistral-cpu`](../20261120T190622Z-kv-cache-close-mistral-cpu/INDEX.md)).

## 3. The tier's microbench bars, pinned

| Bar | Reading | Result |
|---|---|---|
| Paged KV read in place: `paged_vs_dense%` <= 2% at ctx 8,192, batch 8, page 16 ([report](micro/gather-tax-20261120T151422Z.md)) | -3.43% | PASS |
| Idle page reclaim: a pass adds <= 5 ms to an overlapping decode step at ctx 8,192, batch 8, page 16 ([report](micro/kv-reclaim-20261120T152341Z.md)) | at most 3.691 ms (TinyLlama); Qwen2.5-3B 0.681, Phi-3.5-mini 0.829, Mistral 7B 0.386 | PASS |

## 4. Reported, not gated

- **Greedy output**, candidate against baseline: equal on Qwen2.5-3B and Phi-3.5-mini on both lanes and on Mistral 7B's
  CPU lane; different on TinyLlama (both lanes) and Mistral 7B (GPU lane) in all three runs. The tier changes the KV's
  rounding on purpose: the `f16` host KV stores IEEE halves where it stored float32, and the FP16 decode attention sums
  on the wide decode block. The candidate equals its own run 1 in 3 of 3 on every model and lane.
- **Against the reference tool**, candidate, same runs: GPU pp 0.781x to 0.880x, tg 0.852x to 1.102x; CPU pp 0.475x to
  0.561x, tg 0.435x to 0.633x (`gate.log`).
- **Hot methods**, CPU lane: the same integer kernels and worker loop lead on both jars for every model (`gate.log`).

Files: `gate.log`, `run-gate.sh`, `wait-and-run.sh`, `tree.txt`, `micro-pin.txt`,
`ab/<lane>-<baseline|candidate>-<n>-host.json`, `ab/...-<model>-juno.json` and `...-generate-rep1-juno-response.json`
per run, `micro/` (both microbench reports). Local paths are written `<repo>` and `<repos>`.
