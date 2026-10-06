# Decode region with attention inside: `--gpu-residency` off against on, pinned gate

> **Gate reading, clocks pinned** (`clock_pinned: true` in all six `host.json`). Same-hour A/B on one jar, the
> flag alternated off, on, off, on, off, on (`off-1` 03:37Z to `on-3` 03:50Z, 2026-10-06), one repetition per
> run, medians of three per side. Owner run.

**Purpose.** The threshold for attention inside the decode residency region (the KV append and attention moved
into the region; one download per layer): end-to-end tg with the region on **>= 1.00x** the region-off run on
every sweep model where the region runs, and **>= 0.95x** everywhere.

**Build.** HEAD `ffb9583` plus the uncommitted change (`juno_tree_dirty: true`), jar `ace9e5a64b1192eb`, the
same jar on both sides. GPU attention at its default (`auto`, on under CUDA); `n_prompt=128`; default lane only.

**Command** (from the repository root):

```
bash dist/gpu-residency-attention-ab/run-gate.sh
# each of the six invocations:
scripts/performance-tests/compare-llama-cpp.sh --gpu --pin-clocks --models <the four sweep models> \
  --n-prompt 128 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --gpu-residency off|on --juno-jar dist/gpu-residency-attention-ab/candidate-shaded.jar
```

**Result: met.** Juno t/s, all six readings and the medians:

| Model | Region | tg off (3) | tg on (3) | tg on/off | Bound | pp off median | pp on median | pp on/off |
|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | runs | 63.47, 61.08, 61.31 | 70.01, 70.23, 71.92 | **1.145** | >= 1.00 | 2434.39 | 2360.56 | 0.970 |
| mistral-7b-instruct-v0.1-q4_k_m | runs | 21.15, 20.96, 20.91 | 22.84, 22.98, 23.17 | **1.096** | >= 1.00 | 495.92 | 490.71 | 0.989 |
| qwen2.5-3b-instruct-q4_k_m | declined (split-half RoPE, Q/K/V biases) | 26.84, 26.84, 27.26 | 27.43, 27.46, 27.14 | 1.022 | >= 0.95 | 1027.02 | 1015.60 | 0.989 |
| Phi-3.5-mini-instruct-Q4_K_M | declined (other handler) | 29.73, 29.55, 29.18 | 29.38, 29.91, 29.60 | 1.001 | >= 0.95 | 349.86 | 346.62 | 0.991 |

Allocation and GC, generation lane, three readings per side (`jfr.allocated_bytes_per_token`,
`jfr.gc_pause_total_ms_in_token_span`):

| Model | Bytes per token, off | Bytes per token, on | GC ms in span, off | GC ms in span, on |
|---|---|---|---|---|
| tinyllama | 47.85M, 47.43M, 47.28M | 46.45M, 46.48M, 46.45M | 6.7, 7.3, 6.9 | 7.8, 7.3, 0.0 |
| mistral-7b | 222.9M, 224.3M, 224.3M | 218.3M, 218.3M, 217.7M | 14.3, 15.1, 13.7 | 14.2, 14.0, 14.0 |

**How to read it.** The region is decode-only, so tg is the gated figure; every repetition of both region models
reads higher on than any off repetition. Where the region is declined the flag changes nothing in the forward
pass, and on/off sits at 1.00 to 1.02. pp is recorded, not gated: 0.970 to 0.991, within the run-to-run spread of
the off side (TinyLlama 2393 to 2525). Allocation per token falls about 3% with the region on (the per-call
staging arenas of the KV append and attention are gone). Ratios against the reference tool are in each run's
`*-compare.json` and are not read here. Recordings and logs were not copied; local paths are scrubbed.
