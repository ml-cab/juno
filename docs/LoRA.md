# LoRA Fine-Tuning in Juno

Parameter-Efficient Fine-Tuning for LLaMA-family models, implemented entirely in Java. No Python, no PEFT library, no separate training process.

---

## How it works

For each frozen weight matrix **W**, LoRA inserts two small trainable matrices **A** (rank × inDim) and **B** (outDim × rank):

```
W_effective = W + (alpha/rank) × B × A
```

**A** is initialised ~N(0, 0.01). **B** starts at zero. Only **A** and **B** are trained; **W** is never modified.

For `rank=8` on `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | ❌ | ✅ |

---

## Quick start — training

```bash
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (freeform) |
| `/train-file <path>` | Fine-tune on a text file (auto-chunked into ≤128-token pieces) |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact with auto-generated phrasings |
| `/save` | Save adapter to `--lora-path` |
| `/reset` | Reinitialise adapters to zero |
| `/status` | Rank, α, steps trained, checkpoint path |
| `/merge-hint` | Explain offline merge into GGUF |
| `/help` | Command reference |

**`/train-qa` — Q&A fact training:**

The command is designed for single factual associations like name, role, or domain facts:

```
you > /train-qa What is my name? A: Dima
```

Auto-generates 4 phrasings and formats them with the correct chat template for the model:

```
[TRACE] ── formatted training text (repr) ─────────────────
<|user|>↵
What is my name?</s>↵
<|assistant|>↵
Dima</s>↵
<|user|>↵
What is my name?</s>↵     ← repeated for emphasis
<|assistant|>↵
Dima</s>↵
<|user|>↵
Can you tell me: What is my name?</s>↵
<|assistant|>↵
Dima</s>↵
<|user|>↵
Please answer: What is my name?</s>↵
<|assistant|>↵
Dima</s>↵
[TRACE] ── end training text ───────────────────────────────
[TRACE] token count (excl. BOS): 121
```

**Loss targets:** below ~0.5 for reliable recall; above ~1.5 the answer may be inconsistent. Run the same pair 2–3 times or increase `--lora-steps-qa` to drive loss lower.

**Chat template must match.** The `[TRACE] model type (chat template key)` line at REPL startup shows which template was detected. The same key must appear at inference. If they differ, the model will not recall trained facts. Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`, `phi-3`, `gemma`).

---

## Quick start — inference with a trained adapter

Trained adapters can be applied in three modes without entering the training REPL:

**`local` mode (single JVM):**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

**`cluster` mode (forked JVMs, real gRPC):**
```bash
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

**AWS deployed cluster:**
```bash
./launcher.sh juno-deploy.sh setup \
  --lora-play /absolute/path/to/model.lora \
  --model-url https://...
```

See `docs/howto.md` → AWS section for the full deployment flow.

---

## Programmatic API

```java
// 1. Load base model
LoraAdapterSet adapters = LoraAdapterSet.qv(cfg, 8, 8f, new Random(42));
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
    │
    ├── local mode: LoraAdapterSet.load(path)
    │                    └── ForwardPassHandlerLoader.load(model, ctx, backend, adapters)
    │                              └── LoraTrainableHandler (inference-only, no optimizer)
    │
    └── cluster mode: ClusterHarness.withLoraPlay(path)
                           └── launchNode(): -Djuno.lora.play.path=PATH injected per JVM
                                    └── EmbeddedNodeServer.loadShard()
                                             └── LoraAdapterSet.load(Path.of(property))
                                             └── ForwardPassHandlerLoader.load(..., adapters)
```

### Rank selection

| rank | Parameters (TinyLlama qv) | When to use |
|---|---|---|
| 4 | ~360K | Quick experiments |
| 8 | ~720K | General fine-tuning (recommended) |
| 16 | ~1.4M | Complex style/domain adaptation |

---

## Training decisions

### Truncated BPTT

Gradients do not flow backward through KV-cache entries from earlier positions. This avoids O(seqLen²) backward work with negligible effect on LoRA quality for most tasks.

### Quantised frozen weights in backward

The transpose matVec in `backwardLayer` dequantises frozen weights one row at a time: O(hiddenDim) peak extra allocation per layer, not O(model).

### Weight decay

Applied only to **A**, not **B**. B starts at zero and applying decay to it would counteract learning from scratch.

---

## Common pitfalls

**`/train-qa` trains the typo.**
If you type `whatos my name` the model learns that exact string. The model may still generalize (because 4 phrasings are generated), but clean spelling in the question gives more reliable results.

**Loss > 1.5 after training.**
Run the same `/train-qa` command 2–3 more times (adapters warm up across runs) or increase `--lora-steps-qa 50`.

**Loss is constant at ~log(vocabSize).**
B starts at zero so the LoRA delta is zero for the first forward pass. After the first backward + Adam step B becomes non-zero and loss will begin moving. If it's still constant after step 2, check `loraAdapters.get(li, proj)` is returning non-null.

**`--lora-play` answered wrong.**
Check `[TRACE] model type` at startup. If it shows `chatml` but your model is TinyLlama, the template mismatch means training and inference see different token sequences — the model literally cannot recall facts trained under a different template. Rename the file to include `tinyllama`.

**Checkpoint loads but inference output is random.**
After `LoraAdapterSet.load()`, always call `opt.reset()` before resuming training (clears stale momentum buffers). For inference-only use, no optimizer is attached at all.

---

## Testing checklist

```bash
mvn test -Dtest=LoraAdapterTest          # numerical gradient check (most important)
mvn test -Dtest=LoraAdapterSetTest       # round-trip serialisation
mvn test -Dtest=LoraAdamOptimizerTest    # update direction + weight decay
mvn test -Dtest=LoraTrainableHandlerTest # adjointness: dot(A×x,v) == dot(A^T×v,x)
```