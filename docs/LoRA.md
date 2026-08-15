# LoRA Fine-Tuning in Juno

Parameter-efficient fine-tuning for LLaMA-family models on a **quantized GGUF base**,
implemented entirely in Java. This is not QLoRA (no NF4 / double-quant / paged Adam).

No Python, no PEFT library, no separate training process.

See also the feature overview in [features.md](features.md) and [legal.md](legal.md) if you plan to merge or redistribute adapters.

---

## How it works

For each frozen weight matrix **W**, LoRA inserts two small trainable matrices **A** (rank x inDim)
and **B** (outDim x rank):

```
W_effective = W + scale × B × A
```

where `scale = alpha/rank` (standard) or `alpha/√rank` (rsLoRA via `--lora-scaling rslora`).
New adapters default to Kaiming-uniform A init (`--lora-init kaiming-uniform`); use
`legacy-normal` for the historical `N(0, 0.01)` path. **B** starts at zero.

Canonical DoRA (`--lora-mode dora`) further rescales each output row:
`y = (magnitude / ‖W+Δ‖) ⊙ (W·x + Δ)` with the row norm detached from gradients.
Checkpoints are version 2 (v1 still loads as standard + legacy-normal + LoRA).

Default targets are `wq` and `wv`. Use `--lora-targets all` for all seven dense linear
projections (`wq,wk,wv,wo,wgate,wup,wdown`). All-linear training and F32 merge increase
adapter count and merged GGUF size substantially.

For `rank=8` on `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | no | yes |

**Architecture support (Tier 6).** LoRA training and `--lora-play` are routed by
`LoraTrainingHandlerFactory` from GGUF `general.architecture`:

| Architecture | Handler | Notes |
|---|---|---|
| `llama`, `mistral`, `tinyllama` | `LoraTrainableHandler` | Separate Q/K/V/FFN tensors |
| `qwen2`, `qwen2.5` | `Qwen2LoraTrainableHandler` | Same dense layout + frozen QKV biases |
| `phi3` | `Phi3LoraTrainableHandler` | Fused `attn_qkv` / `ffn_up`; NeoX RoPE; fused-slice F32 merge |
| `qwen3` (dense) | `Qwen3LoraTrainableHandler` | Per-head Q/K RMSNorm; `qDim` may differ from `hiddenDim` |
| `qwen3moe`, `qwen35`, `gemma`, unknown | **Rejected** | Explicit allowlist error |

Checkpoint keys stay logical (`wq,wk,wv,wo,wgate,wup,wdown`). Physical GGUF names and
row slices are resolved at load/merge via `LoraModelLayout`. Qwen3 `/train-qa` text must
include the closed empty `<think>` block to match inference (`ChatTrainingFormats` /
`ChatTemplate.qwen3`).

**Training loop.** Gradients are summed over chunks, then divided by total prediction tokens,
optionally clipped by global L2 norm (`--lora-max-grad-norm`), then AdamW steps once per
accumulation group (`--lora-gradient-accumulation`). Reported loss is token-weighted, not the
last chunk's mean. Truncated-BPTT window size is `--lora-chunk-tokens` (default **32** for
reproducibility; recommend **128** for large `/train-file`). `/train` and `/train-file` keep
one document-level unit and chunk inside the loop; `--lora-max-train-tokens` (default `0` =
unlimited) applies a seeded whole-chunk subsample of supervised prediction tokens for epoch
sizing without changing CE on included tokens. Optimizer updates use a scheduled learning rate
(constant or warmup-cosine). `--lora-train-device auto|gpu|cpu` (default `auto`) selects the
LoRA MatVec: `gpu` fails closed without CUDA/ROCm (and on resident-weight OOM); `cpu` forces
host matmul; `auto` tries GPU when available and may fall back. `--lora-microbatch N`
(default **8**, range 1..128; env `LORA_MICROBATCH`) sets frozen GEMM width: `N>1` uploads
FP32 for batched GEMM; `1` starts on FP16 sequential GEMV. On FP32 VRAM OOM with half
support, Juno retries once at microbatch 1; further OOM under `auto` falls back to CPU
(`gpu` fails closed). JFR `trainDevice` records the resolved label (`cpu` / `cuda` / `rocm`).
Weight decay is decoupled AdamW on A only; B is never decayed. LoRA+ scales B's learning rate
by `--lora-plus-ratio` (default `1.0`). Train-only deterministic dropout may mask the LoRA
branch input; inference and validation never apply dropout. With `--lora-validation-split`
and `--lora-validation-patience`, complete units (Q&A variants or text documents / capped
chunk windows) are held out;
best A/B weights are restored on exit and the optimizer is reset. JFR `juno.LoraTrainStep`
fires once per optimizer update (includes A/B LR, LoRA+ ratio, dropout, and mode-identity
fields: algorithm / scaling / init / architecture / trainDevice / rank / alpha / targets /
groupWidth). Held-out evals emit `juno.LoraValidation`. DoRA also emits
`juno.LoraNormRefresh`; merge emits `juno.LoraMerge`; `--lora-play` load emits
`juno.LoraPlayback`; save/load emit `juno.LoraCheckpoint`.

### Profiling with `--jfr`

```bash
./juno lora --model-path /path/to/model.gguf --jfr 1m --lora-mode dora
# train, then quit → prints JFR Metrics Summary and writes target/metrics/metrics.json
```

LoRA `--jfr` uses the same programmatic recording as `./juno local --jfr` (not a JVM
`-XX:StartFlightRecording` flag). Metrics JSON keys include
`juno.LoraTrainStep.*.p95`, `juno.LoraTrainStep.by_algorithm.<algo>.*`,
`juno.LoraValidation.*`, `juno.LoraMerge.rmse.last`, `juno.LoraNormRefresh.count`, and
`juno.LoraPlayback.load_ms.p95`. Missing series are `0` (never NaN). Projected-merge
RMSE / delta-retention are approximate requantization quality, not exact QA-LoRA
closure proofs. Tier-4 timing subsets (`frozen_forward_ms`, `frozen_transpose_ms`, …)
may be zero on CPU-only runs.

---

## Quick start — training

```bash
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf
# optional: --lora-targets all --lora-gradient-accumulation 4 --lora-max-grad-norm 1.0
# optional: --lora-lr-schedule cosine --lora-warmup-steps 20 --lora-plus-ratio 4 --lora-dropout 0.05
# optional: --lora-validation-split 0.25 --lora-validation-patience 3 --lora-seed 42
# optional: --lora-chunk-tokens 128 --lora-max-train-tokens 2048
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (freeform) |
| `/train-file <path>` | Fine-tune on a text file (document-level unit; truncated BPTT chunk default **32**, recommend **128**) |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact with auto-generated phrasings |
| `/train-file-qa <path.json>` | Train many Q&A facts from a JSON array in one loop |
| `/save` | Save adapter to `--lora-path` |
| `/reset` | Reinitialize A/B (and DoRA magnitudes), **clear chat history**, and **delete** the `.lora` checkpoint |
| `/status` | Rank, alpha, optimizer updates, checkpoint path |
| `/merge-hint` | Show the `juno merge` command to bake adapter into a standalone GGUF |
| `/help` | Command reference |
| *(regular input)* | Chat inference with current adapter applied |

**`/train-qa` — Q&A fact training:**

Designed for single factual associations (name, role, domain fact):

```
you > /train-qa What is my name? A: Dima

  [1] Q: What is my name?
      A: Dima

  Formatted as 4 Q&A variant(s) from 1 pair(s)  ·  model type: tinyllama  ·  completion-only loss …
  Training  rank=8 · lr=1.0E-4 · …
  done  loss=1.53 (-0.83)
```

The command auto-generates four phrasings to improve generalisation. Loss is **completion-only**:
gradients update only on the answer tokens (not the user prompt), which prevents the classic
failure mode where LoRA collapses and replies with the memorized answer for every prompt.
Training completions include the template turn-end token (e.g. `</s>`, `<|end|>`, `<|im_end|>`);
`GenerationLoop` strips those markers from streamed and final text so they do not appear in replies.
Training stops automatically when loss drops below the configured target (default `1.2` for
`/train-qa`, `1.8` for `/train`), or when the max-iteration cap is reached. If a loaded
checkpoint is already at the target, updates are skipped — run `/reset` before training a new
fact on a stuck adapter. Loss below ~0.5 gives reliable recall; above ~1.5 the answer may be
inconsistent. Tune with `--lora-loss-target-qa`, `--lora-max-iters`, or `--lora-early-stop`.

**`/train-file-qa` — multi-fact Q&A from JSON:**

Same chat templates, completion-only masks, and loss targets as `/train-qa`, but all pairs
train in **one** loop. File must be a `.json` array of objects with `Q` and `A` string fields:

```json
[
  {"Q": "What is my name?", "A": "Dima"},
  {"Q": "Where do I live?", "A": "Kyiv"}
]
```

```
you > /train-file-qa facts.json
```

Each pair expands to four phrasings (4N units for N pairs). Empty arrays, missing keys, or
non-`.json` paths are rejected before training starts.

**HTTP (curl) — same JSON body:**

Start the LoRA REPL with an API port (training stays in-process; not available on cluster
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
`optimizerUpdateCount`, `stopReason`, and `targetReached`. Uses the same QA loss target /
max-iters as `/train-file-qa`. `POST /v1/lora/save` writes the `.lora` checkpoint.

**Chat template must match.** The `[TRACE] model type (chat template key)` line at REPL startup
shows which template was detected. The same key must appear at inference. If they differ, the
model will not recall trained facts. Rename the model file to include the architecture keyword
(`tinyllama`, `llama-3`, `mistral`, `phi3`, `qwen3`). Qwen2/2.5 paths use ChatML; Qwen3
training uses the empty `<think>` block. Gemma LoRA and Qwen3-MoE / Qwen3.5 training are not
supported.

---

## Quick start — inference with a trained adapter

Trained adapters can be applied in any mode without entering the training REPL.

**`local` mode:**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

`--lora-play` uses greedy decoding (`temperature=0`) by default so factual recall
is deterministic. Pass `--temperature F` explicitly if you want sampled/creative
alternatives; at higher temperatures a nearby base-model continuation may be
selected instead of the memorized answer.

**`cluster` mode:**
```bash
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

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

// Preferred: config-based open (targets, accumulation, clipping)
LoraTrainingConfig cfg = LoraTrainingConfig.builder()
    .rank(8).alpha(8f).learningRate(1e-4)
    .targets("qv")
    .gradientAccumulationSteps(4)
    .maxGradNorm(1.0f)
    .build();
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, cfg)) {
    trainer.trainQaPairUntil("What is my name?", "Dima", "tinyllama", 1.2f, 50);
    trainer.save();
}

// Low-level: computeGradients + prepare + step (legacy trainStep still available)
LoraAdapterSet adapters = LoraInitializer.create(llamaCfg, LoraProjection.qv(), 8, 8f, new Random(42));
LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
adapters.zeroAllGrads();
LoraGradientResult r = handler.computeGradients(tokens);
LoraGradients.prepare(adapters, r.predictionCount(), 1.0f);
LoraAdamOptimizer.defaults(1e-4).step(adapters);
```

---

## Architecture

### Files

| File | Role |
|---|---|
| `LoraAdapter.java` | Core math: A/B matrices, forward delta, backward gradient accumulation |
| `LoraAdapterSet.java` | Collection indexed by (layer, projection), binary checkpoint format |
| `LoraAdamOptimizer.java` | Per-adapter Adam with bias correction; weight decay on A only |
| `LoraTrainableHandler.java` | Full training handler: frozen inference + training backward pass |
| `ForwardPassHandlerLoader.java` | `load(..., LoraAdapterSet)` overload for inference-only adapter application |

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

**Frozen weights in backward.** When a GPU backend is active and resident weights fit,
`LoraTrainableHandler`, `Phi3LoraTrainableHandler`, and `Qwen3LoraTrainableHandler` reuse
device matrices (via shared `LoraResidentWeights` / `LoraResidentUpload`) for forward `W*x`
and transpose `W^T*g` (`GpuMatVec.sgemvTranspose`). On LLaMA/Qwen2 with default
`--lora-microbatch 8`, training uploads **FP32** resident weights and uses `GpuBlasOps`
(`cublasSgemm_v2` / `rocblas_sgemm`) to microbatch frozen linears across token positions;
adapters and Adam stay on the host. VRAM-tight cards (e.g. Phi-3.5 on ~8 GB) can pass
`--lora-microbatch 1` for FP16, or rely on auto-fallback (FP32 → FP16 → CPU under `auto`).
Phi-3 uploads **physical** fused QKV / gate-up tensors; Qwen3 preserves `qDim` vs
`hiddenDim` and keeps per-head Q/K RMSNorm on host. Otherwise the transpose matVec
dequantizes frozen weights one row at a time on CPU: O(hiddenDim) peak extra allocation
per layer, not O(model). Measured TinyLlama GPU training speedups are in
`docs/performance.md` (production “GPU LoRA training” = frozen batched GPU + host adapters).

**DoRA.** Canonical detached-norm DoRA is correctness-complete (train / save / playback / F32
merge). Exact norm refresh is **not** production-perf-gated; prefer standard LoRA or rsLoRA for
large all-linear jobs until a measured refresh budget is published.

**QA-LoRA (Tier 5).** Codec + grouped adapter + projected merge metrics are implemented. The
held-out quality experiment matrix remains deferred; exact K-quant affine merge is unsupported.
Sidecar adapters and F32 merge remain the safe production paths.

---

## Producing a standalone merged model (`juno merge`)

```bash
# 1. Fine-tune
./juno lora --model-path /models/tinyllama.gguf
#   you > /train-qa What is your name? A: Juno
#   you > /save

# 2. Merge (produces /models/tinyllama-merged.gguf, ~1 GB)
./juno merge --model-path /models/tinyllama.gguf

# 3. Run -- no .lora file needed
./juno local --model-path /models/tinyllama-merged.gguf
#   you > what is your name?
#   bot > Juno
```

The LoRA delta per weight element (~6x10^-4) is smaller than Q4_K quantization noise (~3x10^-3).
Re-quantizing the merged weights back to Q4_K destroys the delta entirely. `LoraMerge` stores
patched projection tensors (any of the seven supported keys) as F32 and copies all other tensors
verbatim. All-linear merges expand file size substantially. The output is
a valid GGUF v3 file.

Tier 5 Gate A extracted shared Q4_K / Q5_K / Q6_K codecs (`GgufQuantCodec`, encoder id
`juno-kquant-v1`). Tier 5 also adds **QA-LoRA** (`--lora-mode qa-lora`): sum-pooled input
groups with A shaped `rank × groupCount` (group width auto from tensor type: 32 for Q4_K/Q5_K,
16 for Q6_K). This is not QLoRA.

Merge policies (`--lora-merge`):

| Policy | Behaviour |
|--------|-----------|
| `f32-preserve` (default) | Adapted tensors written as F32 |
| `source-type-projected` | Decode → add delta → encode with `juno-kquant-v1` (approximate requantization; reports delta retention / RMSE) |
| `sidecar-only` | Forbids bake-in merge; use overlay playback |

Exact QA-LoRA zero-point merge into K-quants is **not** supported. The held-out quality
experiment matrix is deferred. Overlay (sidecar) and F32 preserve remain the safe production
paths.

### Programmatic API

```java
LoraMerge.Result r = LoraMerge.merge(
    Path.of("TinyLlama.Q4_K_M.gguf"),
    Path.of("TinyLlama.Q4_K_M.lora"),
    Path.of("TinyLlama.Q4_K_M-merged.gguf"));

System.out.println("Patched " + r.adaptersApplied() + " tensors");
// Patched 44 tensors
```

---

## Common pitfalls

**`/train-qa` trains the typo.** If you type `whatos my name` the model learns that exact
string. Clean spelling in the question gives more reliable results.

**Loss still above target after training.** Raise `--lora-max-iters` or lower
`--lora-loss-target-qa` (e.g. `1.0`). For raw text, tune `--lora-loss-target-text`.

**Loss is constant at ~log(vocabSize).** B starts at zero so the LoRA delta is zero for the
first forward pass. After the first backward + Adam step B becomes non-zero and loss will begin
moving. If it is still constant after step 2, check `loraAdapters.get(li, proj)` is non-null.

**`--lora-play` answered wrong.** Check `[TRACE] model type` at startup. A template mismatch
between training and inference means the model cannot recall trained facts. Rename the file to
include the architecture keyword.

**Checkpoint loads but inference output is random.** After `LoraAdapterSet.load()`, call
`opt.reset()` before resuming training to clear stale momentum buffers. For inference-only use
no optimizer is attached at all.

---

## Testing checklist

```bash
mvn test -Dtest=LoraAdapterTest          # numerical gradient check (most important)
mvn test -Dtest=LoraAdapterSetTest       # round-trip serialisation
mvn test -Dtest=LoraAdamOptimizerTest    # update direction + weight decay
mvn test -Dtest=LoraTrainableHandlerTest # adjointness: dot(A*x,v) == dot(A^T*v,x)
```