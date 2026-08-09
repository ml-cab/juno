(ch-09)=
# 9. Training and Inference Workflows: the REPL, Q&A Facts, Common Pitfalls

This chapter is the practical companion to [Chapter 8](#ch-08): how to actually run training and
inference sessions, what the REPL commands do, and the mistakes that most commonly derail a
first LoRA session.

## Quick start — training

```bash
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf
# optional: --lora-targets all --lora-gradient-accumulation 4 --lora-max-grad-norm 1.0
# optional: --lora-lr-schedule cosine --lora-warmup-steps 20 --lora-plus-ratio 4 --lora-dropout 0.05
# optional: --lora-validation-split 0.25 --lora-validation-patience 3 --lora-seed 42
```

## REPL commands

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (freeform) |
| `/train-file <path>` | Fine-tune on a text file (auto-chunked into <= 128-token pieces) |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact with auto-generated phrasings |
| `/save` | Save adapter to `--lora-path` |
| `/reset` | Reinitialize A/B (and DoRA magnitudes), **clear chat history**, and **delete** the `.lora` checkpoint |
| `/status` | Rank, alpha, optimizer updates, checkpoint path |
| `/merge-hint` | Show the `juno merge` command to bake adapter into a standalone GGUF |
| `/help` | Command reference |
| *(regular input)* | Chat inference with current adapter applied |

## `/train-qa` — Q&A fact training

Designed for single factual associations (name, role, domain fact):

```
you > /train-qa What is my name? A: Dima

  Question: What is my name?
  Answer  : Dima

  [TRACE] -- formatted training text (repr) ------------------
  <|user|>
  What is my name?</s>
  <|assistant|>
  Dima</s>
  ...
  [TRACE] -- end training text --------------------------------
  [TRACE] token count (excl. BOS): 121

  Formatted as 4 Q&A pairs  .  model type: tinyllama
  Training  rank=8 . lr=1.0E-4 . 40 steps . 4 chunk(s) . 122 tokens
  done  loss=1.53 (-0.83)
```

The command auto-generates four phrasings to improve generalisation. Loss is
**completion-only**: gradients update only on the answer tokens (not the user prompt), which
prevents the classic failure mode where LoRA collapses and replies with the memorized answer
for every prompt. Training stops automatically when loss drops below the configured target
(default `1.2` for `/train-qa`, `1.8` for `/train`), or when the max-iteration cap is reached.
If a loaded checkpoint is already at the target, updates are skipped — run `/reset` before
training a new fact on a stuck adapter. Loss below ~0.5 gives reliable recall; above ~1.5 the
answer may be inconsistent. Tune with `--lora-loss-target-qa`, `--lora-max-iters`, or
`--lora-early-stop` (flags documented in [Chapter 3](#ch-03)).

**Chat template must match.** The `[TRACE] model type (chat template key)` line at REPL startup
shows which template was detected. The same key must appear at inference. If they differ, the
model will not recall trained facts. Rename the model file to include the architecture keyword
(`tinyllama`, `llama-3`, `mistral`, `phi3`, `qwen3`). Qwen2/2.5 paths use ChatML; Qwen3 training
uses the empty `<think>` block. Gemma LoRA and Qwen3-MoE / Qwen3.5 training are not supported
(see [Chapter 8](#ch-08) for the full architecture matrix).

## Quick start — inference with a trained adapter

Trained adapters can be applied in any mode without entering the training REPL.

**`local` mode:**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

`--lora-play` uses greedy decoding (`temperature=0`) by default so factual recall is
deterministic. Pass `--temperature F` explicitly for sampled/creative alternatives; at higher
temperatures a nearby base-model continuation may be selected instead of the memorized answer.

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

See [Chapter 7](#ch-07) for the full AWS deployment flow.

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

See [Chapter 6](#ch-06) for how this fits alongside `JunoPlayer` and the REST client in a larger
JVM application.

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
`opt.reset()` before resuming training to clear stale momentum buffers. For inference-only use,
no optimizer is attached at all.

## Testing checklist

```bash
mvn test -Dtest=LoraAdapterTest          # numerical gradient check (most important)
mvn test -Dtest=LoraAdapterSetTest       # round-trip serialisation
mvn test -Dtest=LoraAdamOptimizerTest    # update direction + weight decay
mvn test -Dtest=LoraTrainableHandlerTest # adjointness: dot(A*x,v) == dot(A^T*v,x)
```

---

[← Chapter 8: LoRA Fundamentals](#ch-08) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 10: Producing Standalone Merged Models →](#ch-10)
