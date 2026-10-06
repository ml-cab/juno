# Decode region with attention inside: `--gpu-residency` off against on, indicative

> **Indicative, not a gate reading.** **Clocks not pinned** (`clock_pinned: false` in every `host.json`), one
> repetition per run (`--juno-reps 1 --juno-warmup 2 --reps 1`). The scope-item threshold (tg region on >= 1.00x
> region off where the region runs, >= 0.95x everywhere) is scored only from the same script run with pinned
> clocks, which is the owner's (`dist/gpu-residency-attention-ab/run-gate.sh`, default `PIN=1`).

**Purpose.** First throughput reading of the decode residency region with the KV append and attention moved
inside it, one jar, the flag alternated.

**Build.** HEAD `ffb9583` plus the uncommitted change (`juno_tree_dirty: true`), jar `ace9e5a64b1192eb`. Default
GPU attention (`auto`, on under CUDA); `n_prompt=128`; default lane only.

**Command** (from the repository root; six invocations alternating off and on, `off-1`, `on-1`, ..., `on-3`):

```
PIN=0 bash dist/gpu-residency-attention-ab/run-gate.sh
# each invocation:
scripts/performance-tests/compare-llama-cpp.sh --gpu --models <the four sweep models> --n-prompt 128 \
  --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --gpu-residency off|on --juno-jar dist/gpu-residency-attention-ab/candidate-shaded.jar
```

Medians of three, Juno t/s:

| Model | Region runs? | pp off | pp on | pp on/off | tg off | tg on | tg on/off |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | yes | 2487.68 | 2247.11 | 0.903 | 65.82 | 73.83 | 1.122 |
| mistral-7b-instruct-v0.1-q4_k_m | yes | 520.07 | 513.55 | 0.987 | 22.19 | 24.34 | 1.097 |
| qwen2.5-3b-instruct-q4_k_m | no (split-half RoPE, Q/K/V biases) | 1081.53 | 1069.58 | 0.989 | 29.24 | 29.07 | 0.994 |
| Phi-3.5-mini-instruct-Q4_K_M | no (other handler) | 353.20 | 352.36 | 0.998 | 31.34 | 31.45 | 1.004 |

Per run (pp / tg), TinyLlama and Mistral 7B:

| Run | TinyLlama | Mistral 7B |
|---|---|---|
| off-1 | 2490 / 65.82 | 522 / 22.48 |
| off-2 | 2376 / 64.95 | 520 / 22.19 |
| off-3 | 2487 / 66.27 | 514 / 21.28 |
| on-1 | 2247 / 74.18 | 513 / 24.34 |
| on-2 | 2371 / 73.83 | 517 / 24.37 |
| on-3 | 2048 / 61.93 | 460 / 21.10 |

**How to read it.** The region is decode-only, so tg is the figure that moves: about +10% to +12% where it runs,
flat where it is declined. `on-3` is low on both lanes of both models at once, which points at the host during
that run rather than the region; with it, the TinyLlama prefill median reads 0.903, but prefill does not run
the region, and `on-2` (2371) sits next to `off-2` (2376). One repetition per run cannot separate that from
noise; the pinned gate records pp beside tg. Ratios against the reference tool are in each run's
`*-compare.json` and are not read here. Recordings and logs were not copied; local paths are scrubbed.
