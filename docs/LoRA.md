# LoRA Fine-Tuning in Juno

Parameter-efficient fine-tuning for LLaMA-family models, implemented entirely in Java.
No Python, no PEFT library, no separate training process.

See also the short feature overview [features/lora-and-merge.md](features/lora-and-merge.md) and [legal.md](legal.md) if you plan to merge or redistribute adapters.

---

## How it works

For each frozen weight matrix **W**, LoRA inserts two small trainable matrices **A** (rank x inDim)
and **B** (outDim x rank):

```
W_effective = W + (alpha/rank) x B x A
```

**A** is initialised ~N(0, 0.01). **B** starts at zero. Only **A** and **B** are trained;
**W** is never modified.

For `rank=8` on `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | no | yes |

---

## Quick start — training

```bash
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (freeform) |
| `/train-file <path>` | Fine-tune on a text file (auto-chunked into <= 128-token pieces) |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact with auto-generated phrasings |
| `/save` | Save adapter to `--lora-path` |
| `/reset` | Reinitialise adapters to zero (clears all training) |
| `/status` | Rank, alpha, steps trained, checkpoint path |
| `/merge-hint` | Show the `juno merge` command to bake adapter into a standalone GGUF |
| `/help` | Command reference |
| *(regular input)* | Chat inference with current adapter applied |

**`/train-qa` — Q&A fact training:**

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

The command auto-generates four phrasings to improve generalisation. Loss below ~0.5 gives
reliable recall; above ~1.5 the answer may be inconsistent. Run the same pair 2-3 times or
increase `--lora-steps-qa` to drive loss lower.

**Chat template must match.** The `[TRACE] model type (chat template key)` line at REPL startup
shows which template was detected. The same key must appear at inference. If they differ, the
model will not recall trained facts. Rename the model file to include the architecture keyword
(`tinyllama`, `llama-3`, `mistral`, `phi-3`, `gemma`).

---

## Quick start — inference with a trained adapter

Trained adapters can be applied in any mode without entering the training REPL.

**`local` mode:**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

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

// 1. Load base model
LoraAdapterSet adapters = LoraQvInitializer.qv(cfg, 8, 8f, new Random(42));
LoraTrainableHandler handler = LoraTrainableHandler.load(
    Path.of("TinyLlama.Q4_K_M.gguf"), ctx, adapters);

// 2. Train
LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);
for (int step = 0; step < 1000; step++) {
    float loss = handler.trainStep(tokens, opt);
}

// 3. Save
adapters.save(Path.of("my-finetune.lora"));

// 4. Load for inference only (no optimizer needed)
LoraAdapterSet playAdapters = LoraAdapterSet.load(Path.of("my-finetune.lora"));
ForwardPassHandler h = ForwardPassHandlerLoader.load(modelPath, ctx, backend, playAdapters);
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

**Quantized frozen weights in backward.** The transpose matVec in `backwardLayer` dequantizes
frozen weights one row at a time: O(hiddenDim) peak extra allocation per layer, not O(model).

**Weight decay.** Applied only to **A**, not **B**. B starts at zero; applying decay to it would
counteract learning from scratch.

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
Re-quantizing the merged weights back to Q4_K destroys the delta entirely. `LoraMerge` stores the
44 patched projection tensors (wq/wv) as F32 and copies all other tensors verbatim. The output is
a valid GGUF v3 file.

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

**Loss > 1.5 after training.** Run the same `/train-qa` command 2-3 more times or increase
`--lora-steps-qa 50`.

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