# LoRA Fine-Tuning in Juno

Parameter-efficient fine-tuning for GGUF-based models, implemented entirely in Java.
Training runs on a quantized GGUF base model. This is not QLoRA: Juno does not implement
NF4, double-quantization, compute-dtype, or paged Adam. QA-LoRA is a separate grouped-adapter
algorithm documented in its own section.

No Python, no PEFT library, no separate training process.

See also [features.md](features.md) for the feature overview and [legal.md](legal.md)
if you plan to merge or redistribute adapters.

---

## How it works

For each frozen weight matrix **W**, LoRA inserts two small trainable matrices **A** (rank x inDim)
and **B** (outDim x rank):

```
W_effective = W + scale * B * A
```

**Scaling modes** (set at adapter creation; authoritative in the checkpoint):

| Mode | Formula | Flag |
|------|---------|------|
| Standard (default) | `scale = alpha / rank` | `--lora-scaling standard` |
| rsLoRA | `scale = alpha / sqrt(rank)` | `--lora-scaling rslora` |

**Initialization:**

| Mode | A init | B init | Flag |
|------|--------|--------|------|
| `kaiming-uniform` (default) | `U(-1/sqrt(inDim), +1/sqrt(inDim))` matching PyTorch `kaiming_uniform_(a=sqrt(5))` | zeros | `--lora-init kaiming-uniform` |
| `legacy-normal` | `N(0, 0.01)` | zeros | `--lora-init legacy-normal` |

Use `legacy-normal` only to reproduce historical runs. Newly created adapters default to
Kaiming-uniform.

**DoRA** (`--lora-mode dora`) adds per-row magnitude rescaling on top of the LoRA delta:

```
direction = W + scale * B * A
output    = (magnitude / norm(direction)) * (W*x + scale*B*A*x)
```

Row norms are detached from gradients (canonical PEFT-style DoRA). Magnitude is a separate
AdamW parameter group with decay off. DoRA is correctness-complete: train, save, playback,
and F32 merge are fully tested. Norm refresh is not production-perf-gated. Prefer standard
LoRA or rsLoRA for large all-linear jobs until a measured refresh budget is published.

**QA-LoRA** (`--lora-mode qa-lora`) uses sum-pooled grouped A:

```
pooled[group] = sum(input[groupStart : groupEnd])
delta         = scale * B * A * pooled
```

A is shaped `[rank x groupCount]` rather than `[rank x inDim]`. Group width is auto-detected
from the tensor GGML type: 32 for Q4_K / Q5_K, 16 for Q6_K. See the Merge section for merge
capability policies.

**Default targets** are `wq` and `wv`. Use `--lora-targets all` for all seven dense linear
projections (`wq,wk,wv,wo,wgate,wup,wdown`). Targets are stored in the checkpoint and resolved
at load time.

For `rank=8` on `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | no | yes |

---

## Architecture support

LoRA training, `--lora-play` inference overlay, and `./juno merge` are all routed by
`LoraTrainingHandlerFactory` using the GGUF `general.architecture` field:

| `general.architecture` | Handler | Support | Notes |
|---|---|---|---|
| `llama`, `mistral`, `tinyllama` | `LoraTrainableHandler` | Full | Separate Q/K/V/FFN tensors, adjacent-pair RoPE |
| `qwen2`, `qwen2.5` | `Qwen2LoraTrainableHandler` | Full | Same dense layout with frozen QKV biases |
| `phi3` | `Phi3LoraTrainableHandler` | Full | Fused `attn_qkv` / `ffn_up`; NeoX RoPE adjoint; fused-slice F32 merge |
| `qwen3` (dense) | `Qwen3LoraTrainableHandler` | Full | Per-head Q/K RMSNorm; `qDim` may differ from `hiddenDim` |
| `qwen3moe`, `qwen35`, `gemma`, unknown | **Rejected** | None | Explicit allowlist error; no silent routing |

Checkpoint keys remain logical (`wq,wk,wv,wo,wgate,wup,wdown`). Physical GGUF tensor names
and row-slice offsets are resolved at load and merge time via `LoraModelLayout` and
`LoraProjectionBinding`. Phi-3 fused-slice merge patches `attn_qkv` at Q/K/V row ranges and
`ffn_up` at gate/up row ranges; multiple logical adapters map to one physical tensor without
overwriting each other.

Qwen3 `/train-qa` text must include the closed empty `<think>` block to match inference
formatting (`ChatTrainingFormats` / `ChatTemplate.qwen3`).

---

## Training loop

**Gradient accumulation.** Gradients are summed over `--lora-gradient-accumulation` chunks,
divided by total prediction tokens, then clipped globally by L2 norm before the optimizer step.
Reported loss is token-weighted across all chunks in the accumulation group, not the last
chunk's mean.

**Truncated BPTT.** The backward window is `--lora-chunk-tokens` (default **32** for
reproducibility with historical runs; recommend **128** for `/train-file`). Gradients do not
flow backward through KV-cache entries from earlier positions. This avoids O(seqLen^2) backward
work with negligible effect on LoRA quality.

**Corpus scheduling.** `/train` and `/train-file` keep one document-level `TrainUnit` per
document, then chunk inside `LoraTrainingLoop.flattenChunks`:

- `--lora-chunk-tokens N` sets the BPTT window size (default 32).
- `--lora-max-train-tokens N` / `LORA_MAX_TRAIN_TOKENS` (default 0 = unlimited) applies a
  seeded whole-chunk subsample via `LoraCorpusLimit` when the corpus exceeds the budget. The
  CE objective on included tokens is unchanged. This is epoch sizing, not data filtering.

**Optimizer.** True A-only decoupled AdamW: weight decay applies to A only, using the
pre-update parameter value and the uncorrected learning rate. B is never decayed. LoRA+ scales
B's learning rate by `--lora-plus-ratio` (default `1.0` = ordinary LoRA behavior). Moments see
raw gradients only; weight decay is not included in the gradient norm passed to clipping.

**Schedules.** Constant LR (default) or warmup-cosine (`--lora-lr-schedule cosine`,
`--lora-warmup-steps N`, `--lora-min-lr F`). The schedule tracks optimizer-update count, not
chunk count.

**Dropout.** Train-only inverted dropout (`--lora-dropout F`, default 0 = disabled). Masks
are generated from a stateless index hash of seed, optimizer step, accumulation ordinal, token
position, layer, and projection. Inference and validation are always dropout-free. Identical
seeded runs produce identical adapter weights.

**Validation early stopping.** `--lora-validation-split F` holds out a fraction of training
units as complete variants. Validation runs once per full pass. Best A/B weights are
snapshotted and restored on exit; the optimizer is reset after restoration. Stop triggers:
target loss reached, patience exhausted, overfit guard, or max-iters cap.

**Loss targets.** Training stops automatically when loss drops below the configured target:

| Flag | Default | Applies to |
|------|---------|-----------|
| `--lora-loss-target-qa F` | `1.2` | `/train-qa`, `/train-file-qa` |
| `--lora-loss-target-text F` | `1.8` | `/train`, `/train-file` |
| `--lora-early-stop F` | `0.25` | all (overfit guard; set 0 to disable) |
| `--lora-max-iters N` | `50` | all (hard cap per command) |

**Gradient clipping.** `--lora-max-grad-norm F` (default 1.0). `LoraGradients.prepare` runs
two passes: accumulates squared norm in `double`, rejects non-finite values before optimizer
mutation, then applies one combined normalization and clipping scale. `0` disables clipping
while still normalizing by prediction count.

**GPU training path.** When `--lora-train-device auto|gpu` and VRAM allows, frozen forward
and transpose backward run on device via `LoraResidentWeights` (shared helper for LLaMA-family,
Qwen2, Phi-3, and dense Qwen3). Adapters (A/B matrices) and Adam optimizer remain on host.
This is the production GPU LoRA training path: frozen batched GPU + host adapters.

Default `--lora-microbatch 8` (`LORA_MICROBATCH`) uploads FP32 resident weights and batches
frozen linears across token positions via `GpuBlasOps` (`cublasSgemm_v2` / `rocblas_sgemm`).
Explicit `--lora-microbatch 1` starts directly on FP16 sequential GEMV, which suits VRAM-tight
cards such as Phi-3.5 on ~8 GB.

**VRAM OOM auto-fallback ladder** (managed by `LoraResidentUpload`):

1. FP32 upload with `microbatch > 1` fails due to VRAM OOM.
2. Close partial buffers, set microbatch=1, retry FP16 once.
3. FP16 also fails: under `auto`, log and fall back to CPU quantized; under `gpu`, fail closed.

Never set `JAVA_TOOL_OPTIONS=-Djuno.lora.microbatch=1`. Use the CLI flag instead.

JFR `trainDevice` records the resolved label (`cpu` / `cuda` / `rocm`).

**Measured gates** (NVIDIA GeForce GTX 1080, CUDA 12 / Panama FFI, TinyLlama Q4_K_M, qv,
rank 8, seq 64, microbatch 8):

| Path | e2e ms/step | backward ms | Speedup vs CPU |
|------|-------------|-------------|----------------|
| CPU quantized (oracle) | ~47,907 | ~26,663 | 1.0x |
| GPU FP32 resident, microbatch 8 | ~3,433 | ~2,457 | e2e **14x**, backward **11x** |

---

## Quick start: training

```bash
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Projection targets
# optional: --lora-targets all --lora-targets wq,wk,wv

# Accumulation and clipping
# optional: --lora-gradient-accumulation 4 --lora-max-grad-norm 1.0

# Scheduling
# optional: --lora-lr-schedule cosine --lora-warmup-steps 20 --lora-min-lr 1e-5

# LoRA+ and dropout
# optional: --lora-plus-ratio 4 --lora-dropout 0.05 --lora-seed 42

# Validation early stopping
# optional: --lora-validation-split 0.25 --lora-validation-patience 3

# Corpus scheduling
# optional: --lora-chunk-tokens 128 --lora-max-train-tokens 2048

# GPU training
# optional: --lora-train-device auto --lora-microbatch 8
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (freeform, completion loss) |
| `/train-file <path>` | Fine-tune on a text file. One document-level unit; chunk size from `--lora-chunk-tokens` (default 32; recommend 128 for large files) |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact with auto-generated phrasings |
| `/train-file-qa <path.json>` | Train many Q&A facts from a JSON array in one loop |
| `/save` | Save adapter to `--lora-path` |
| `/reset` | Reinitialize A/B (and DoRA magnitudes), clear chat history, delete the `.lora` checkpoint |
| `/status` | Rank, alpha, optimizer updates, checkpoint path, mode, targets |
| `/merge-hint` | Print the `juno merge` command to bake adapter into a standalone GGUF |
| `/help` | Command reference |
| *(regular input)* | Chat inference with current adapter applied |

**`/train-qa`: Q&A fact training**

Designed for single factual associations. Generates four phrasings automatically to improve
generalisation. Loss is completion-only: gradients update only on answer tokens. This prevents
the collapse failure mode where the model replies with the memorised answer for every prompt.

```
you > /train-qa What is my name? A: Dima

  [1] Q: What is my name?
      A: Dima

  Formatted as 4 Q&A variant(s) from 1 pair(s)  model type: tinyllama  completion-only loss
  Training  rank=8  lr=1.0E-4
  done  loss=1.53 (-0.83)
```

Training completions include the template turn-end token (`</s>`, `<|end|>`, `<|im_end|>`).
`GenerationLoop` strips those markers from streamed text so they do not appear in replies.
If a loaded checkpoint is already at the target, updates are skipped; run `/reset` before
training a new fact on a stuck adapter.

Loss guidance:
- Below ~0.5: reliable recall
- 0.5 to ~1.5: answer may be inconsistent
- Above ~1.5: model likely not learning the fact yet

Tune with `--lora-loss-target-qa`, `--lora-max-iters`, or `--lora-early-stop`.

**`/train-file-qa`: multi-fact Q&A from JSON**

Same chat templates, completion-only masks, and loss targets as `/train-qa`, but all pairs
train in one loop. File must be a `.json` array with `Q` and `A` string fields:

```json
[
  {"Q": "What is my name?", "A": "Dima"},
  {"Q": "Where do I live?", "A": "Kyiv"}
]
```

Each pair expands to four phrasings (4N units for N pairs). Empty arrays, missing keys, or
non-`.json` paths are rejected before training starts.

```
you > /train-file-qa facts.json
```

**HTTP (curl):**

Start the LoRA REPL with an API port (training stays in-process; not available on the cluster
inference API):

```bash
./juno lora --model-path models/mistral-7b-instruct-v0.1-q4_k_m.gguf --heap 12g --api-port 8080
```

```bash
curl -s http://localhost:8080/v1/lora/train-file-qa \
  -H 'Content-Type: application/json' \
  --data-binary @facts.json

curl -s -X POST http://localhost:8080/v1/lora/save
```

`POST /v1/lora/train-file-qa` returns `pairCount`, `unitCount`, `finalTrainLoss`, `passCount`,
`optimizerUpdateCount`, `stopReason`, and `targetReached`. `POST /v1/lora/save` writes the
`.lora` checkpoint.

**Chat template must match.** The `[TRACE] model type (chat template key)` line at REPL startup
shows which template was detected. The same key must appear at inference. If they differ, the
model will not recall trained facts. Rename the model file to include the architecture keyword
(`tinyllama`, `llama-3`, `mistral`, `phi3`, `qwen3`). Qwen2/2.5 paths use ChatML. Qwen3
training uses the empty `<think>` block. Gemma LoRA and Qwen3-MoE / Qwen3.5 training are not
supported.

---

## Quick start: inference with a trained adapter

Trained adapters are applied in any mode without entering the training REPL.

**`local` mode:**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

`--lora-play` uses greedy decoding (`temperature=0`) by default so factual recall is
deterministic. Pass `--temperature F` explicitly for sampled output; at higher temperatures
a nearby base-model continuation may be selected instead of the memorised answer.

**`cluster` mode:**
```bash
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

In cluster mode, `ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path=PATH`
into every forked node JVM. Each node loads the adapter before building its
`ForwardPassHandler`.

**AWS deployed cluster:**
```bash
./launcher.sh juno-deploy.sh setup \
  --lora-play /absolute/path/to/model.lora \
  --model-url https://...
```

See [howto.md](howto.md) for the full AWS deployment flow.

---

## Programmatic API

```java
import cab.ml.juno.lora.*;
import cab.ml.juno.node.*;
import cab.ml.juno.player.LoraTrainer;
import cab.ml.juno.player.LoraTrainingConfig;

// Config-based open (preferred: targets, accumulation, clipping, scheduling)
LoraTrainingConfig cfg = LoraTrainingConfig.builder()
    .rank(8).alpha(8f).learningRate(1e-4)
    .targets("qv")
    .gradientAccumulationSteps(4)
    .maxGradNorm(1.0f)
    .chunkTokens(128)
    .maxTrainTokens(0)      // 0 = unlimited
    .lrSchedule("cosine").warmupSteps(20).minLr(1e-5f)
    .loraMode("lora")       // or "dora", "qa-lora"
    .scaling("standard")    // or "rslora"
    .seed(42)
    .build();
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, cfg)) {
    trainer.trainQaPairUntil("What is my name?", "Dima", "tinyllama", 1.2f, 50);
    trainer.save();
}

// Multi-pair from a list
List<String[]> pairs = List.of(
    new String[]{"What is my name?", "Dima"},
    new String[]{"Where do I live?",  "Kyiv"}
);
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, cfg)) {
    trainer.trainQaPairsUntilResult(pairs, "tinyllama");
    trainer.save();
}

// Low-level: computeGradients + prepare + step
LoraAdapterSet adapters = LoraInitializer.create(llamaCfg, LoraProjection.qv(), 8, 8f, new Random(42));
LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
adapters.zeroAllGrads();
LoraGradientResult r = handler.computeGradients(tokens);
LoraGradients.prepare(adapters, r.predictionCount(), 1.0f);
LoraAdamOptimizer.defaults(1e-4).step(adapters);
```

---

## Architecture (internal)

### Key classes

| Class | Module | Role |
|---|---|---|
| `LoraAdapter` | `lora` | Core math: A/B matrices, forward delta, backward gradient accumulation, dropout masks |
| `LoraAdapterSet` | `lora` | Collection indexed by (layer, projection); binary checkpoint v1/v2 |
| `LoraAdapterConfig` | `lora` | Adapter identity: rank, declared alpha, scaling, initialization, mode |
| `LoraAdamOptimizer` | `lora` | Per-adapter Adam with bias correction; A-only decoupled weight decay; LoRA+ B/A LR ratio |
| `LoraGradients` | `lora` | Two-pass global L2 norm accumulation and gradient clipping |
| `LoraTrainableHandler` | `node` | LLaMA-family training handler: frozen inference + training backward |
| `Qwen2LoraTrainableHandler` | `node` | Qwen2/2.5: same layout + frozen QKV biases |
| `Phi3LoraTrainableHandler` | `node` | Phi-3: fused QKV/gate-up, NeoX RoPE adjoint, fused-slice F32 merge |
| `Qwen3LoraTrainableHandler` | `node` | Dense Qwen3: per-head Q/K RMSNorm, qDim vs hiddenDim |
| `LoraTrainingHandlerFactory` | `node` | Explicit architecture allowlist; explicit rejection for unsupported archs |
| `LoraModelLayout` / `LoraProjectionBinding` | `node` | Logical key to physical GGUF tensor plus row-slice offset mapping |
| `LoraResidentWeights` | `node` | Shared upload / close / VRAM-OOM fallback / matVec routing for all training handlers |
| `LoraResidentUpload` | `node` | FP32 OOM: set microbatch=1, retry FP16; further OOM: CPU fallback or fail-closed |
| `LoraMicrobatch` | `node` | `--lora-microbatch` / `LORA_MICROBATCH`; sets `juno.lora.microbatch` before resident upload |
| `GpuBlasOps` / `DeviceActivationBatch` | `node` | FP32 `cublasSgemm_v2` / `rocblas_sgemm` microbatch for frozen forward and transpose |
| `LoraInitializer` | `node` | Creates adapters in stable layer/projection order (Kaiming or legacy-normal) |
| `LoraProjection` | `node` | Logical key enum with dimensions and GGUF tensor suffix |
| `LoraMerge` | `node` | Writes new GGUF with F32-patched or requantized adapted tensors |
| `GgufKQuantCodec` / `GgufQuantCodec` | `node` | Q4_K / Q5_K / Q6_K decode/encode; versioned encoder `juno-kquant-v1` |
| `QaLoraInitializer` | `node` | Group-width detection from tensor GGML type for QA-LoRA |
| `LoraMetricsIdentity` | `node` | Mode identity carried on every JFR event (algorithm, scaling, arch, trainDevice, ...) |
| `LoraTrainEvent` / `LoraValidationEvent` | `node` | JFR events for optimizer updates and validation passes |
| `LoraNormRefreshEvent`, `LoraMergeEvent`, `LoraPlaybackEvent`, `LoraCheckpointEvent` | `node` | JFR events for DoRA refresh, merge, playback, and checkpoint I/O |
| `ForwardPassHandlerLoader` | `node` | `load(..., LoraAdapterSet)` overload for inference-only adapter application |
| `LoraTrainingLoop` | `juno-player` | Shared orchestration: document units, chunking, validation, best-weight restore |
| `LoraCorpusLimit` | `juno-player` | Seeded whole-chunk subsampling for `--lora-max-train-tokens` corpus caps |
| `LoraTrainer` | `juno-player` | Programmatic facade: config-based `open`, `trainQaPairUntil`, `save` |
| `LoraTrainingConfig` | `juno-player` | Builder-based training config (targets, accumulation, scheduling, GPU, ...) |
| `LoraMergeMain` | `juno-player` | CLI entry point for `juno merge` |

### How `--lora-play` routes through the stack

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

### Rank selection

| rank | Parameters (TinyLlama qv) | When to use |
|---|---|---|
| 4 | ~360K | Quick experiments |
| 8 | ~720K | General fine-tuning (recommended) |
| 16 | ~1.4M | Complex style/domain adaptation |

---

## Training decisions

**Truncated BPTT.** Gradients do not flow backward through KV-cache entries from earlier
positions. This avoids O(seqLen^2) backward work with negligible effect on LoRA quality.
The window is `--lora-chunk-tokens`.

**Frozen weights in backward.** When a GPU backend is active and resident weights fit,
`LoraResidentWeights` routes frozen `W*x` (forward) and `W^T*g` (transpose backward) through
device matrices for all architecture handlers. Adapters (A/B) and Adam optimizer remain on
host. This is the production-supported GPU path: frozen batched GPU + host adapters.

On LLaMA / Qwen2 with default `--lora-microbatch 8`, training uploads FP32 resident weights
and uses `GpuBlasOps` to batch frozen linears across token positions. Phi-3 uploads physical
fused QKV / gate-up tensors. Dense Qwen3 preserves per-head Q/K RMSNorm and the
`qDim != hiddenDim` distinction.

On the CPU path, dequantization runs one 256-element block at a time inside the matVec loop:
O(hiddenDim) peak extra allocation per layer, not O(model).

**VRAM microbatch ladder.** `--lora-microbatch N` / `LORA_MICROBATCH` (default 8, range
1..128) controls batch width for frozen GEMM. `N > 1` prefers FP32 upload for microbatched
GEMM. `N = 1` starts directly on FP16 sequential GEMV. On FP32 VRAM OOM with half support,
`LoraResidentUpload` closes partial buffers, sets microbatch=1, and retries FP16 once. If
FP16 also fails: under `auto`, CPU quantized fallback; under `gpu`, fail closed. Phi-3.5 on
~8 GB cards should use `--lora-microbatch 1` rather than `JAVA_TOOL_OPTIONS`.

**DoRA.** Correctness-complete (train / save / playback / F32 merge). Exact norm refresh is
not production-perf-gated. DoRA cache generation is bumped on `/reset` so inference correctly
drops trained magnitude coefficients. Prefer standard LoRA or rsLoRA for large all-linear jobs
until a measured refresh budget is published.

**QA-LoRA (Tier 5).** Grouped adapter math (sum-pool input groups, grouped A shaped
`[rank x groupCount]`), projected merge metrics, and checkpoint v2 QA entries are implemented.
The held-out quality experiment matrix is deferred. Exact K-quant affine merge into
Q4_K / Q5_K / Q6_K is not supported: those formats are not closed under QA-LoRA learned
additive group shifts. Sidecar adapters and F32 preserve remain the safe production paths.

---

## Producing a standalone merged model (`juno merge`)

```bash
# 1. Fine-tune
./juno lora --model-path /models/tinyllama.gguf
#   you > /train-qa What is your name? A: Juno
#   you > /save

# 2. Merge (produces /models/tinyllama-merged.gguf)
./juno merge --model-path /models/tinyllama.gguf

# 3. Run (no .lora file needed)
./juno local --model-path /models/tinyllama-merged.gguf
#   you > what is your name?
#   bot > Juno
```

The LoRA delta per weight element (~6x10^-4) is smaller than Q4_K quantization noise (~3x10^-3).
Re-quantizing the merged weights back to Q4_K destroys the delta entirely. `LoraMerge` stores
patched projection tensors as F32 and copies all other tensors verbatim in their original
quantized form. All-linear merges expand file size substantially. The output is a valid GGUF v3
file.

Phi-3 fused-slice merge correctly patches `attn_qkv` at Q/K/V row ranges and `ffn_up` at
gate/up row ranges without overwriting earlier slices.

**Merge policies** (`--lora-merge`):

| Policy | Behaviour |
|--------|-----------|
| `f32-preserve` (default) | Adapted tensors written as F32. Safe for all modes and architectures. |
| `source-type-projected` | Decode, add delta, re-encode with versioned `juno-kquant-v1` encoder (approximate requantization). Reports delta retention, RMSE, saturation per tensor. Use only when file size matters more than precision. |
| `sidecar-only` | Forbids bake-in merge; use overlay playback only. |

Exact QA-LoRA zero-point merge into K-quants is not supported. Use `f32-preserve` or sidecar
for production deployment.

**Programmatic API:**

```java
LoraMerge.Result r = LoraMerge.merge(
    Path.of("TinyLlama.Q4_K_M.gguf"),
    Path.of("TinyLlama.Q4_K_M.lora"),
    Path.of("TinyLlama.Q4_K_M-merged.gguf"));

System.out.println("Patched " + r.adaptersApplied() + " tensors");
// Patched 44 tensors
```

---

## JFR and metrics

LoRA training uses the same programmatic JFR recording as `./juno local --jfr`: not the JVM
`-XX:StartFlightRecording` flag. On exit, metrics are auto-extracted to
`target/metrics/metrics.json` and a console summary is printed.

```bash
./juno lora --model-path /path/to/model.gguf --jfr 1m --lora-mode dora
# train, then quit: prints JFR Metrics Summary and writes target/metrics/metrics.json
```

**Event catalog:**

| Event | Emitted when |
|-------|-------------|
| `juno.LoraTrainStep` | Once per optimizer update |
| `juno.LoraValidation` | Once per validation pass (requires `--lora-validation-split > 0`) |
| `juno.LoraNormRefresh` | Once per DoRA norm-cache refresh (DoRA only) |
| `juno.LoraMerge` | Once per `juno merge` completion |
| `juno.LoraPlayback` | Once per `--lora-play` adapter load |
| `juno.LoraCheckpoint` | Once per `/save` or checkpoint load |
| `juno.ForwardPass` | Per transformer layer forward pass |
| `juno.MatVec` | Per matVec call (shared with inference path) |

Every LoRA event carries `LoraMetricsIdentity` fields: `algorithm` (lora / rslora / dora /
qa-lora), `scaling`, `initialization`, `architecture`, `trainDevice` (cpu / cuda / rocm),
`rank`, `alpha`, `targets`, `groupWidth` (QA-LoRA only).

**Key JSON keys** (extracted to `target/metrics/metrics.json`):

```
juno.LoraTrainStep.count
juno.LoraTrainStep.forward_ms.p95
juno.LoraTrainStep.backward_ms.p95
juno.LoraTrainStep.optimizer_ms.p95
juno.LoraTrainStep.total_ms.p95
juno.LoraTrainStep.frozen_forward_ms.p95       # non-zero only on GPU path
juno.LoraTrainStep.frozen_transpose_ms.p95     # non-zero only on GPU path
juno.LoraTrainStep.adapter_backward_ms.p95
juno.LoraTrainStep.loss.last
juno.LoraTrainStep.loss.mean
juno.LoraTrainStep.grad_norm.p95
juno.LoraTrainStep.clipped.fraction
juno.LoraTrainStep.by_algorithm.<algo>.total_ms.p95

juno.LoraValidation.count
juno.LoraValidation.loss.best
juno.LoraValidation.duration_ms.p95

juno.LoraMerge.count
juno.LoraMerge.duration_ms.p95
juno.LoraMerge.rmse.last                       # non-zero for source-type-projected merges
juno.LoraMerge.delta_retention.last            # non-zero for source-type-projected merges

juno.LoraNormRefresh.count
juno.LoraNormRefresh.duration_ms.p95

juno.LoraPlayback.count
juno.LoraPlayback.load_ms.p95

juno.LoraCheckpoint.count
```

Rules:
- Missing series return 0, never NaN.
- Frozen forward/transpose timing fields are zero on CPU-only runs.
- Projected-merge RMSE and delta-retention are approximate requantization quality, not exact
  QA-LoRA closure proofs.
- Older `.jfr` files without new fields still extract cleanly via guarded field reads.

---

## Common pitfalls

**`/train-qa` trains the typo.** If you type `whatos my name` the model learns that exact
string. Clean spelling in the question gives more reliable results.

**Loss still above target after training.** Raise `--lora-max-iters` or lower
`--lora-loss-target-qa` (e.g. `1.0`). For raw text, tune `--lora-loss-target-text`.

**Loss constant at ~log(vocabSize).** B starts at zero so the LoRA delta is zero for the first
forward pass. After the first backward and Adam step B becomes non-zero and loss will begin
moving. If still constant after step 2, check that `loraAdapters.get(li, proj)` is non-null.

**`--lora-play` answered wrong.** Check `[TRACE] model type` at startup. A template mismatch
between training and inference means the model cannot recall trained facts. Rename the file to
include the architecture keyword (`tinyllama`, `llama-3`, `mistral`, `phi3`, `qwen3`).

**Checkpoint loads but inference output is random.** After `LoraAdapterSet.load()`, call
`opt.reset()` before resuming training to clear stale momentum buffers. For inference-only use,
no optimizer is attached.

**FP32 VRAM OOM on Phi-3.5 with microbatch 8.** Use `--lora-microbatch 1` to start on FP16
sequential GEMV, or let the auto-fallback ladder handle it. Do not use
`JAVA_TOOL_OPTIONS=-Djuno.lora.microbatch=1`.

**`--lora-play` answers correctly at first then degrades after `/reset`.** `/reset` bumps the
DoRA cache generation so inference drops trained magnitude coefficients. This is correct
behavior. If training a new fact on a DoRA adapter, always `/reset` before retraining to avoid
stale magnitudes.

---

## Testing checklist

```bash
mvn test -Dtest=LoraAdapterTest                    # numerical gradient check (most important)
mvn test -Dtest=LoraAdapterSetTest                 # round-trip serialisation v1/v2
mvn test -Dtest=LoraAdamOptimizerTest              # update direction, A-only decay, LoRA+
mvn test -Dtest=LoraTrainableHandlerTest           # adjointness: dot(A*x,v) == dot(A^T*v,x)
mvn test -Dtest=LoraMicrobatchTest                 # bounds, apply/current, blank/default
mvn test -Dtest=LoraCorpusLimitTest                # seeded subsampling, budget limits
mvn test -Dtest=LoraTrainableHandlerGpuBackwardTest  # CPU/GPU parity, speed gates
mvn test -pl node -Dgroups=gpu                     # GPU adjoint, parity (NVIDIA)
mvn test -pl node -Dgroups=rocm                    # GPU adjoint, parity (AMD)
```



