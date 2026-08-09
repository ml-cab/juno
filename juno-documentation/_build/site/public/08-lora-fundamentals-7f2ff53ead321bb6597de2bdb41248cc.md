(ch-08)=
# 8. LoRA Fundamentals: The Math and the Architecture Support Matrix

Juno implements parameter-efficient fine-tuning for LLaMA-family and related models on a
**quantized GGUF base**, entirely in Java. This is not QLoRA: there is no NF4, no double
quantization, and no paged Adam. There is no Python, no PEFT library, and no separate training
process — training happens in the same JVM that would otherwise run `./juno local`.

## How it works

For each frozen weight matrix **W**, LoRA inserts two small trainable matrices **A**
(rank × inDim) and **B** (outDim × rank):

```
W_effective = W + scale × B × A
```

where `scale = alpha/rank` (standard) or `alpha/√rank` (rsLoRA via `--lora-scaling rslora`).
New adapters default to Kaiming-uniform A initialization (`--lora-init kaiming-uniform`); use
`legacy-normal` for the historical `N(0, 0.01)` path. **B** starts at zero, so the LoRA delta is
exactly zero before the first optimizer step.

Canonical DoRA (`--lora-mode dora`) further rescales each output row:
`y = (magnitude / ‖W+Δ‖) ⊙ (W·x + Δ)`, with the row norm detached from gradients. Checkpoints
are version 2 (version 1 still loads as standard scaling + legacy-normal init + plain LoRA).

Default targets are `wq` and `wv`. Use `--lora-targets all` for all seven dense linear
projections (`wq,wk,wv,wo,wgate,wup,wdown`) — all-linear training and F32 merge increase adapter
count and merged GGUF size substantially.

For `rank=8` on `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | no | yes |

## Architecture support matrix

LoRA training and `--lora-play` are routed by `LoraTrainingHandlerFactory` from GGUF
`general.architecture`, against an explicit allowlist. Unrecognized or unsupported
architectures fail loudly at load time rather than silently falling back to the LLaMA layout.

| Architecture | Handler | Notes |
|---|---|---|
| `llama`, `mistral`, `tinyllama` | `LoraTrainableHandler` | Separate Q/K/V/FFN tensors |
| `qwen2`, `qwen2.5` | `Qwen2LoraTrainableHandler` | Same dense layout + frozen QKV biases |
| `phi3` | `Phi3LoraTrainableHandler` | Fused `attn_qkv` / `ffn_up`; NeoX RoPE; fused-slice F32 merge |
| `qwen3` (dense) | `Qwen3LoraTrainableHandler` | Per-head Q/K RMSNorm; `qDim` may differ from `hiddenDim` |
| `qwen3moe`, `qwen35`, `gemma`, unknown | **Rejected** | Explicit allowlist error |

Checkpoint keys stay logical (`wq,wk,wv,wo,wgate,wup,wdown`); physical GGUF names and row slices
are resolved at load/merge time via `LoraModelLayout`. Qwen3 `/train-qa` text must include the
closed, empty `<think>` block to match inference (`ChatTrainingFormats` /
`ChatTemplate.qwen3`). For the full inference-side status of every architecture — including
ones not listed above, such as Qwen3-MoE and Gemma — see [Chapter 11](#ch-11).

## The training loop

Gradients are summed over chunks, then divided by total prediction tokens, optionally clipped
by global L2 norm (`--lora-max-grad-norm`), then AdamW steps once per accumulation group
(`--lora-gradient-accumulation`). Reported loss is token-weighted, not the last chunk's mean.
Optimizer updates use a scheduled learning rate — constant, or warmup-cosine.

Weight decay is decoupled AdamW on **A** only; **B** is never decayed, because B starts at zero
and applying decay to it would counteract learning from scratch. LoRA+ scales B's learning rate
by `--lora-plus-ratio` (default `1.0`, meaning ordinary LoRA).

Train-only deterministic dropout may mask the LoRA branch input; inference and validation never
apply dropout. With `--lora-validation-split` and `--lora-validation-patience`, complete units
(Q&A variants or text chunks) are held out; the best A/B weights are restored on exit and the
optimizer is reset.

JFR `juno.LoraTrainStep` fires once per optimizer update and includes A/B learning rate, LoRA+
ratio, dropout, and mode-identity fields (algorithm, scaling, initialization, architecture,
train device, rank, alpha, targets, group width). Held-out evals emit `juno.LoraValidation`.
DoRA also emits `juno.LoraNormRefresh`; merge emits `juno.LoraMerge`; `--lora-play` load emits
`juno.LoraPlayback`; save/load emit `juno.LoraCheckpoint`. Every LoRA mode — train, validate,
merge, playback, DoRA norm refresh, checkpoint I/O — is covered by this event catalog.

### Profiling with `--jfr`

```bash
./juno lora --model-path /path/to/model.gguf --jfr 1m --lora-mode dora
# train, then quit → prints JFR Metrics Summary and writes target/metrics/metrics.json
```

LoRA `--jfr` uses the same programmatic recording as `./juno local --jfr` (not a JVM
`-XX:StartFlightRecording` flag). Metrics JSON keys include `juno.LoraTrainStep.*.p95`,
`juno.LoraTrainStep.by_algorithm.<algo>.*`, `juno.LoraValidation.*`, `juno.LoraMerge.rmse.last`,
`juno.LoraNormRefresh.count`, and `juno.LoraPlayback.load_ms.p95`. Missing series are `0`, never
`NaN`. Projected-merge RMSE and delta-retention (see [Chapter 10](#ch-10)) describe approximate
requantization quality, not an exact closure proof. GPU-specific timing subfields
(`frozen_forward_ms`, `frozen_transpose_ms`, …) may be zero on CPU-only runs.

## Architecture: files and routing

| File | Role |
|---|---|
| `LoraAdapter.java` | Core math: A/B matrices, forward delta, backward gradient accumulation |
| `LoraAdapterSet.java` | Collection indexed by (layer, projection), binary checkpoint format |
| `LoraAdamOptimizer.java` | Per-adapter Adam with bias correction; weight decay on A only |
| `LoraTrainableHandler.java` | Full training handler: frozen inference + training backward pass |
| `ForwardPassHandlerLoader.java` | `load(..., LoraAdapterSet)` overload for inference-only adapter application |

**How `--lora-play` routes through the stack:**

```
ConsoleMain (--lora-play PATH)
    |
    +-- local mode: LoraAdapterSet.load(path)
    |                    +-- ForwardPassHandlerLoader.load(model, ctx, backend, adapters)
    |                              +-- LoraTrainableHandler (inference-only, no optimizer)
    |
    +-- cluster mode: ClusterHarness.withLoraPlay(path)
                           +-- launchNode(): -Djuno.lora.play.path=PATH injected per JVM
                                    +-- EmbeddedNodeServer.loadShard()
                                             +-- LoraAdapterSet.load(Path.of(property))
                                             +-- ForwardPassHandlerLoader.load(..., adapters)
```

## Rank selection

| rank | Parameters (TinyLlama qv) | When to use |
|---|---|---|
| 4 | ~360K | Quick experiments |
| 8 | ~720K | General fine-tuning (recommended) |
| 16 | ~1.4M | Complex style/domain adaptation |

## Training decisions

**Truncated BPTT.** Gradients do not flow backward through KV-cache entries from earlier
positions. This avoids O(seqLen²) backward work with negligible effect on LoRA quality.

**Frozen weights in backward.** When a GPU backend is active and resident weights fit,
`LoraTrainableHandler` reuses the same device matrices for forward `W*x` and transpose
`W^T*g` (`GpuMatVec.sgemvTranspose`). Otherwise the transpose matVec dequantizes frozen weights
one row at a time on CPU: O(hiddenDim) peak extra allocation per layer, not O(model). GPU
residency accelerates the frozen forward/transpose paths; it does not yet imply a fully
GPU-resident optimizer step — see [Chapter 13](#ch-13) for how this shows up in measured
throughput.

**Weight decay.** Applied only to A, never B, for the reason noted above.

**Microbatching.** `--lora-gradient-accumulation` groups multiple training chunks into a single
optimizer update: gradients from each chunk are summed and token-weighted before the AdamW step
fires, rather than one step per chunk. This keeps a single update numerically equivalent to
training on the concatenated chunk regardless of how many chunks a Q&A or text unit is split
into.

**Test coverage.** Each supported architecture (LLaMA family, Qwen2/2.5, Phi-3, dense Qwen3) has
its own adjointness tests (`dot(A·x, v) == dot(Aᵀ·v, x)`), finite-difference gradient checks, and
zero-adapter parity tests against the corresponding non-LoRA transformer handler, so a freshly
initialized adapter is provably a no-op before any training occurs.

Practical training and inference workflows built on top of these mechanics are in
[Chapter 9](#ch-09); producing a standalone merged model is in [Chapter 10](#ch-10).

---

[← Chapter 7: AWS Deployment](#ch-07) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 9: Training and Inference Workflows →](#ch-09)
