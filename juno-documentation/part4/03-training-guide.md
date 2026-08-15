(ch-4-3)=
# 4.3. Training Guide

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

**VRAM OOM auto-fallback ladder** (managed by `LoraResidentUpload`): exits dut to user experience 
to fit training in aavaliable resources.

```{mermaid}
flowchart TD
    Start["Start GPU LoRA --lora-train-device auto; FP32 weights; microbatch 8; cublas|rocblas sgemm"]
    OOM1{{"cudaMalloc / hipMalloc\nOOM?"}}
    FP16["Retry: FP16 resident weights\nmicrobatch = 1\nsequential GEMV\ncublasHSSgemvStridedBatched"]
    OOM2{{"Still OOM?"}}
    Auto{{"--lora-train-device = auto?"}}
    CPU["Fall back to CPU quantized\n(CpuMatVec, IntStream)"]
    Fail["Fail closed\n(exit with error)\n--lora-train-device=gpu"]
    Success["Training proceeds on GPU\nfrozen forward + transpose on device\nAdapters (A/B) + Adam on host"]

    Start --> OOM1
    OOM1 -->|"No"| Success
    OOM1 -->|"Yes"| FP16 --> OOM2
    OOM2 -->|"No"| Success
    OOM2 -->|"Yes"| Auto
    Auto -->|"auto"| CPU
    Auto -->|"gpu"| Fail
```

Never set `JAVA_TOOL_OPTIONS=-Djuno.lora.microbatch=1`. Use the CLI flag instead.

JFR `trainDevice` records the resolved label (`cpu` / `cuda` / `rocm`).

**Measured gates** (NVIDIA GeForce GTX 1080, CUDA 12 / Panama FFI, TinyLlama Q4_K_M, qv,
rank 8, seq 64, microbatch 8):

| Path | e2e ms/step | backward ms | Speedup vs CPU |
|------|-------------|-------------|----------------|
| CPU quantized (oracle) | ~47,907 | ~26,663 | 1.0x |
| GPU FP32 resident, microbatch 8 | ~3,433 | ~2,457 | e2e **14x**, backward **11x** |


**Rank selection**

| rank | Parameters (TinyLlama qv) | When to use |
|---|---|---|
| 4 | ~360K | Quick experiments |
| 8 | ~720K | General fine-tuning (default) |
| 16 | ~1.4M | Complex style/domain adaptation |


---

## Quick start



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


## Profiling training

LoRA training uses the same programmatic JFR recording as `./juno local --jfr`, not the JVM
`-XX:StartFlightRecording` flag. On exit, metrics are auto-extracted to
`target/metrics/metrics.json` and a console summary is printed.

```bash
./juno lora --model-path /path/to/model.gguf --jfr 1m --lora-mode dora
# train, then quit: prints JFR Metrics Summary and writes target/metrics/metrics.json
```

See [JFR and metrics](#ch-7-1) for the full LoRA
event catalog and the JSON key reference.

## See also

- [Chapter 4.1 -- Concepts](#ch-4-1)
- [Chapter 4.7 -- Common Pitfalls](#ch-4-7)
- [Chapter 4.8 -- Testing Checklist](#ch-4-8)
- [Chapter 3.2 -- Flags](#ch-3-2)

---

[<- 4.2 Architecture Support](#ch-4-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.4 Inference with a Trained Adapter ->](#ch-4-4)
