# Whole decode layer in the residency region: `--gpu-residency` off against on, pinned gate

**Gate reading.** Clocks pinned on all six runs (`clock_pinned: true` in every `host.json`), owner run,
2026-10-06 08:02Z to 08:19Z. Threshold: generation (tg) with the region on **>= 1.00x** region off on every model
where the region runs, **>= 0.95x** everywhere. **Met.**

**Purpose.** Score the decode residency region running the whole LLaMA-family layer (output projection, residual
adds, FFN norm and SwiGLU FFN after attention) with the residual row kept on the device between layers, one jar,
the flag alternated.

**Build.** HEAD `d3ad6ee` plus the uncommitted change (`juno_tree_dirty: true`), jar
`dist/gpu-residency-whole-layer-ab/candidate-shaded.jar`, sha256 `ade0bb660c5e2065`, on both sides. Default GPU
attention (`auto`, on under CUDA); `n_prompt=128`; default lane only. Every run: 128 prompt tokens, 64 generated.

**Command** (from the repository root; six invocations alternating off and on, `off-1`, `on-1`, ..., `on-3`):

```
bash dist/gpu-residency-whole-layer-ab/run-gate.sh
# each invocation:
scripts/performance-tests/compare-llama-cpp.sh --gpu --pin-clocks --models <the four sweep models> --n-prompt 128 \
  --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --gpu-residency off|on --juno-jar dist/gpu-residency-whole-layer-ab/candidate-shaded.jar
```

Medians of three, Juno t/s, with the generate lane's allocation per token and GC pause inside the token span, and
the tg ratio against the reference tool (read, not gated):

| Model | Region | pp off | pp on | pp on/off | tg off | tg on | **tg on/off** | Bound | alloc/token off / on | GC pause in span, ms, off / on | tg ratio off / on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | whole layer | 2409.48 | 2169.78 | 0.901 | 55.24 | 121.95 | **2.208** | >= 1.00, met | 48.1M / 38.5M | 8.3 / 0.0 | 0.323x / 0.677x |
| mistral-7b-instruct-v0.1-q4_k_m | whole layer | 504.50 | 483.17 | 0.958 | 20.20 | 34.66 | **1.716** | >= 1.00, met | 223.0M / 193.1M | 15.8 / 0.0 | 0.546x / 1.008x |
| qwen2.5-3b-instruct-q4_k_m | declined (split-half RoPE, Q/K/V biases) | 1001.91 | 980.36 | 0.978 | 26.58 | 25.48 | **0.959** | >= 0.95, met | 143.2M / 143.1M | 0.0 / 0.0 | 0.392x / 0.377x |
| Phi-3.5-mini-instruct-Q4_K_M | declined (other handler) | 338.29 | 336.06 | 0.993 | 28.71 | 28.46 | **0.991** | >= 0.95, met | 210.3M / 209.3M | 14.8 / 13.2 | 0.515x / 0.498x |

Per run (pp / tg):

| Run | TinyLlama | Mistral 7B | Qwen2.5-3B | Phi-3.5-mini |
|---|---|---|---|---|
| off-1 | 2481 / 63.47 | 515 / 20.20 | 1002 / 27.09 | 352 / 28.71 |
| off-2 | 2163 / 55.24 | 505 / 18.92 | 1006 / 26.09 | 336 / 30.10 |
| off-3 | 2409 / 54.92 | 480 / 20.43 | 994 / 26.58 | 338 / 27.94 |
| on-1 | 2157 / 121.95 | 512 / 35.02 | 990 / 26.38 | 324 / 28.46 |
| on-2 | 2170 / 117.64 | 463 / 34.66 | 980 / 25.48 | 342 / 30.32 |
| on-3 | 2251 / 122.09 | 483 / 33.97 | 935 / 24.30 | 336 / 26.50 |

**How to read it.** On both region models every on repetition is far above every off repetition: generation 2.2x
(TinyLlama) and 1.7x (Mistral 7B), allocation per generated token 13% to 20% lower, no GC pause inside the token span.
Mistral 7B's generation reads 1.008x of the reference tool with the region on.

Two readings to note, neither gated:
- **Qwen2.5-3B, 0.959 against a 0.95 bound.** The model declines the region (the handler never builds it), so on
  and off run the same code; the unpinned reading was also 0.959, and the attention-only region's pinned gate read
  1.022 on the same model. Its on repetitions sit lowest in the last pair (`on-3` 24.30). Not attributed.
- **TinyLlama prefill on/off 0.901** (Mistral 7B 0.958, the declined models 0.978 and 0.993), although the
  prefill window does not run the decode region. In `on-1` against `off-1` the measured prefill forward pass reads
  59.4 against 51.6 ms while the whole request reads 79 against 89 ms. The off side spreads 13% on its own (2163 to
  2481). Not attributed; the published `--device-spans` run owed at the end of the item reads the prefill window.

Ratios against the reference tool are in each run's `*-compare.json`. Recordings, logs and response bodies were not
copied; local paths are scrubbed.
