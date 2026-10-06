# Whole decode layer in the residency region: `--gpu-residency` off against on, indicative

> **Scored by the pinned run [`20261006T080204Z-gpu-residency-whole-layer-ab`](../20261006T080204Z-gpu-residency-whole-layer-ab/INDEX.md) (gate met).**
>
> **Indicative, not a gate reading.** **Clocks not pinned** (`clock_pinned: false` in every `host.json`), one
> repetition per run (`--juno-reps 1 --juno-warmup 2 --reps 1`). The threshold (tg region on >= 1.00x region off
> where the region runs, >= 0.95x everywhere) is scored only from the same script run with pinned clocks, which is
> the owner's (`dist/gpu-residency-whole-layer-ab/run-gate.sh`, default `PIN=1`).

**Purpose.** First throughput reading of the decode residency region running the whole LLaMA-family layer
(output projection, residual adds, FFN norm and SwiGLU FFN added after attention) with the residual row kept on the
device between layers, one jar, the flag alternated.

**Build.** HEAD `d3ad6ee` plus the uncommitted change (`juno_tree_dirty: true`), jar `ade0bb660c5e2065`. Default GPU
attention (`auto`, on under CUDA); `n_prompt=128`; default lane only. Every run: 128 prompt tokens, 64 generated.

**Command** (from the repository root; six invocations alternating off and on, `off-1`, `on-1`, ..., `on-3`):

```
PIN=0 bash dist/gpu-residency-whole-layer-ab/run-gate.sh
# each invocation:
scripts/performance-tests/compare-llama-cpp.sh --gpu --models <the four sweep models> --n-prompt 128 \
  --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --gpu-residency off|on --juno-jar dist/gpu-residency-whole-layer-ab/candidate-shaded.jar
```

Medians of three, Juno t/s, with the generate lane's allocation per token and GC pause inside the token span, and
the tg ratio against the reference tool (read, not gated):

| Model | Region | pp off | pp on | pp on/off | tg off | tg on | **tg on/off** | alloc/token off / on | GC pause in span, ms, off / on | tg ratio off / on |
|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | whole layer | 2399.91 | 2216.27 | 0.923 | 60.33 | 116.91 | **1.938** | 47.7M / 38.1M | 8.0 / 0.0 | 0.311x / 0.649x |
| mistral-7b-instruct-v0.1-q4_k_m | whole layer | 484.86 | 481.92 | 0.994 | 21.90 | 36.36 | **1.660** | 223.7M / 193.6M | 14.1 / 0.0 | 0.587x / 0.990x |
| qwen2.5-3b-instruct-q4_k_m | declined (split-half RoPE, Q/K/V biases) | 1050.78 | 1026.60 | 0.977 | 27.42 | 26.30 | 0.959 | 143.3M / 143.6M | 0.0 / 0.0 | 0.404x / 0.378x |
| Phi-3.5-mini-instruct-Q4_K_M | declined (other handler) | 350.52 | 346.67 | 0.989 | 29.78 | 29.52 | 0.991 | 211.6M / 212.2M | 12.2 / 11.6 | 0.494x / 0.494x |

Per run (pp / tg), TinyLlama and Mistral 7B:

| Run | TinyLlama | Mistral 7B |
|---|---|---|
| off-1 | 2400 / 61.84 | 485 / 21.90 |
| off-2 | 2414 / 60.33 | 467 / 21.63 |
| off-3 | 2221 / 55.33 | 508 / 22.69 |
| on-1 | 2170 / 115.86 | 497 / 36.92 |
| on-2 | 2216 / 116.91 | 470 / 36.36 |
| on-3 | 2275 / 124.85 | 482 / 35.90 |

**Which layers ran whole.** These runs' console logs do not carry the region's activation line (it goes to the
engine log). `smoke-gpu-residency.sh` on the same build logged every layer of every node running whole on both
models (TinyLlama 8, 7 and 7 layers per node, Mistral 7B 11, 11 and 10), with no layer announced as leaving the region
after attention.

**How to read it.** The region is decode-only, so tg is the figure that moves, and every on repetition is far
above every off repetition on both region models. The decode layer that took about a dozen synchronous host round
trips now takes one upload per token and one download per layer, with one host wait per layer. Against the
attention-only region (`20261006T025445Z-gpu-residency-attention-ab-unpinned`, unpinned: TinyLlama 73.83, Mistral 7B
24.34 t/s with the region on) generation is about 1.6x and 1.5x faster. Mistral 7B's tg reads 0.990x of the
reference tool here, a single unpinned reading. The declined models move only inside the noise; Qwen2.5-3B's 0.959
is the closest to its 0.95 bound and is the row to watch in the pinned run. TinyLlama's pp on/off reads 0.923 again
(0.903 in the attention-only reading, 0.970 when pinned), although prefill does not run the region; the pinned run
records pp beside tg.

Greedy text: within each mode the generated text is identical across the three runs. On against off it parts after
about 20 to 30 tokens on both region models, as it already did with the attention-only region on this prompt; the
region's norms sum in a different order than the CPU norms of the flag-off path (bit-identity holds against the GPU
op-at-a-time path, by test). `smoke-gpu-residency.sh` finds greedy output identical on against off over its own
32-token prompts.

Ratios against the reference tool are in each run's `*-compare.json`. Recordings, logs and response bodies were not
copied; local paths are scrubbed.
