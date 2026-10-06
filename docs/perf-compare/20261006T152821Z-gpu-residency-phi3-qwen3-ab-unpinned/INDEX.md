# Decode residency region on Phi-3 and Qwen3: `--gpu-residency` off against on, indicative

> **Indicative, not a gate reading.** **Clocks not pinned** (`clock_pinned: false` in every `host.json`), one
> repetition per run (`--juno-reps 1 --juno-warmup 2 --reps 1`). The threshold (tg region on >= 1.00x region off
> where the region runs, >= 0.95x everywhere) is scored only from the same script run with pinned clocks, which is
> the owner's (`dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`, default `PIN=1`).

**Purpose.** First throughput reading of the decode residency region running the whole layer on the Phi-3 handler
(fused Q/K/V and gate/up read through row views, the extended RoPE with frequency factors and magnitude scale on the
device) and the dense Qwen3 handler (per-head Q/K norms on the device, split-half RoPE), alongside the LLaMA family,
one jar, the flag alternated.

**Build.** HEAD `3761abc` plus the uncommitted change (`juno_tree_dirty: true`), jar `9659ff62d527578b`. Default GPU
attention (`auto`, on under CUDA); `n_prompt=128`; default lane only. Every run: 128 prompt tokens, 64 generated.
Qwen3-1.7B is not a sweep model; its heap is derived from the file size (`heap_source` in each result).

**Command** (from the repository root; six invocations alternating off and on, `off-1`, `on-1`, ..., `on-3`):

```
PIN=0 bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh
# each invocation:
scripts/performance-tests/compare-llama-cpp.sh --gpu --models <the four sweep models>,Qwen3-1.7B-Q4_K_M \
  --n-prompt 128 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --gpu-residency off|on --juno-jar dist/gpu-residency-phi3-qwen3-ab/candidate-shaded.jar
```

Medians of three, Juno t/s, with the generate lane's allocation per token and GC pause inside the token span, and
the tg ratio against the reference tool (read, not gated):

| Model | Region | pp off | pp on | pp on/off | tg off | tg on | **tg on/off** | alloc/token off / on | GC pause in span, ms, off / on | tg ratio off / on |
|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | whole layer | 2183.52 | 2335.15 | 1.069 | 66.03 | 130.62 | **1.978** | 48.0M / 38.8M | 7.1 / 0.0 | 0.340x / 0.676x |
| mistral-7b-instruct-v0.1-q4_k_m | whole layer | 504.84 | 515.49 | 1.021 | 22.28 | 36.36 | **1.632** | 223.9M / 193.6M | 14.2 / 0.0 | 0.605x / 0.985x |
| Phi-3.5-mini-instruct-Q4_K_M | whole layer (new) | 355.34 | 352.60 | 0.992 | 31.34 | 48.20 | **1.538** | 207.5M / 174.1M | 11.5 / 13.2 | 0.524x / 0.805x |
| Qwen3-1.7B-Q4_K_M | whole layer (new) | 722.82 | 732.43 | 1.013 | 37.62 | 63.93 | **1.699** | 111.9M / 97.5M | 0.0 / 16.4 | 0.327x / 0.557x |
| qwen2.5-3b-instruct-q4_k_m | declined (split-half RoPE, Q/K/V biases) | 1067.07 | 1065.38 | 0.998 | 29.21 | 29.35 | 1.005 | 143.5M / 143.3M | 0.0 / 0.0 | 0.412x / 0.413x |

Per run (pp / tg), the two handlers this change adds:

| Run | Phi-3.5-mini | Qwen3-1.7B |
|---|---|---|
| off-1 | 367 / 31.52 | 754 / 36.76 |
| off-2 | 355 / 31.22 | 691 / 37.62 |
| off-3 | 353 / 31.34 | 722 / 37.91 |
| on-1 | 350 / 48.20 | 732 / 63.93 |
| on-2 | 352 / 48.32 | 749 / 63.57 |
| on-3 | 356 / 47.95 | 680 / 64.20 |

**Which layers ran whole.** These runs' console logs do not carry the region's activation line (it goes to the
engine log). `smoke-gpu-residency.sh` on the same jar logged every layer of every node running whole (Phi-3.5-mini
11 of 11 layers on the node it reports, Qwen3-1.7B 10 of 10), and `Phi3TransformerHandlerGpuResidencyTest` /
`Qwen3TransformerHandlerGpuResidencyTest` hold the whole-layer copy counts on a single node: 2 uploads and 33
downloads per token on Phi-3.5-mini's 32 layers, 2 and 29 on Qwen3-1.7B's 28.

**How to read it.** The region is decode-only, so tg is the figure that moves; every on repetition is far above every
off repetition on all four region models. Phi-3.5-mini's tg reads 0.805x of the reference tool with the region on,
above the 0.70x end-of-plan target, in a single unpinned reading with a flag that is off by default. Allocation per
generated token falls 13% to 19% on the region models. The GC pause in the token span rises on Qwen3-1.7B (0.0 to
15.6 to 17.7 ms on every repetition) and slightly on Phi-3.5-mini (11.5 to 13.2): on Qwen3-1.7B it is one young
collection now landing inside the 64-token span (`gc_pause_count_in_token_span` 1 against 0) in a 4 GiB heap while
total allocation falls, a threshold effect, recorded here and read again by the tier's standing allocation gate. The
`top_allocation_sites` lists are a throttled sample and differ run to run; they do not attribute the change.
Qwen2.5-3B, which declines the region, reads 1.005. pp moves within the spread on every model.
