# LoRA Fine-Tuning in Juno

Parameter-Efficient Fine-Tuning for LLaMA-family models, implemented entirely in
Java. No Python, no PEFT library, no separate training process — the same node
that serves inference can be fine-tuned with a few API calls.

---

## How it works

For each frozen weight matrix **W** (e.g. the query projection `wq`), LoRA
inserts two small trainable matrices **A** (rank × inDim) and **B** (outDim × rank):

```
W_effective = W + (alpha/rank) × B × A
```

**A** is initialised ~N(0, 0.01). **B** starts at zero, so the adapter has zero
effect at the beginning of training. Only **A** and **B** are trained; the frozen
**W** weights are never modified.

For `rank=8` applied to `wq` and `wv` across all 22 layers of TinyLlama-1.1B:

| | Frozen | LoRA |
|---|---|---|
| Parameters | 1,100,048,000 | 720,896 |
| Memory (F32) | ~4.3 GB | 2.8 MB |
| Training target | ❌ | ✅ |

---

## Quick start

```java
// 1. Load the base model
GgufReader r = GgufReader.open(Path.of("TinyLlama.Q4_K_M.gguf"));
LlamaConfig cfg = LlamaConfig.from(r);
ShardContext ctx = new ShardContext("node-1", 0, cfg.numLayers(),
                                    true, true, cfg.vocabSize(),
                                    cfg.hiddenDim(), cfg.numHeads());

// 2. Create LoRA adapters (wq + wv on all layers, rank=8)
LoraAdapterSet adapters = LoraAdapterSet.qv(cfg, 8, 8f, new Random(42));

// 3. Load the handler
LoraTrainableHandler handler = LoraTrainableHandler.load(
    Path.of("TinyLlama.Q4_K_M.gguf"), ctx, adapters);

// 4. Training loop
LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);

for (int step = 0; step < 1000; step++) {
    int[] tokens = tokenize(nextTrainingDoc());   // your tokeniser
    float loss = handler.trainStep(tokens, opt);
    System.out.printf("step=%d  loss=%.4f%n", step, loss);
}

// 5. Save checkpoint
adapters.save(Path.of("my-finetune.lora"));
```

### Resuming training

```java
// Load base model exactly as before, then load the saved adapter weights
LoraAdapterSet adapters = LoraAdapterSet.load(Path.of("my-finetune.lora"));
LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);
opt.reset();  // ← always reset after loading; clears stale momentum buffers
```

### Inference with a trained adapter

```java
// The handler implements ForwardPassHandler — just use it like any other handler
ForwardRequest req = ForwardRequest.withTokens("req-1", new int[]{1, 2, 3}, 0);
ForwardResult result = handler.forward(req, ctx);
float[] logits = result.logits();
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

### Where to apply LoRA

The `qv()` factory applies LoRA to **wq** and **wv** — the standard configuration
from the original paper (Hu et al. 2021). You can also add **wk** and **wo**:

```java
LoraAdapterSet adapters = new LoraAdapterSet();
int rank = 8;
float alpha = 8f;
Random rng = new Random(42);
for (int li = 0; li < cfg.numLayers(); li++) {
    adapters.add(li, "wq", new LoraAdapter(rank, cfg.hiddenDim(), cfg.hiddenDim(), alpha, rng));
    adapters.add(li, "wk", new LoraAdapter(rank, cfg.hiddenDim(), cfg.kvDim(),    alpha, rng));
    adapters.add(li, "wv", new LoraAdapter(rank, cfg.hiddenDim(), cfg.kvDim(),    alpha, rng));
    adapters.add(li, "wo", new LoraAdapter(rank, cfg.hiddenDim(), cfg.hiddenDim(), alpha, rng));
}
```

Adding wk and wo roughly doubles training parameters but rarely improves results
significantly. Start with qv-only.

### Rank selection

| rank | Parameters (TinyLlama qv) | When to use |
|---|---|---|
| 4 | ~360K | Quick experiments, minimal regularisation |
| 8 | ~720K | General fine-tuning (recommended default) |
| 16 | ~1.4M | Complex style/domain adaptation |
| 64 | ~5.8M | Approaches full fine-tuning; rarely needed |

### Alpha and scale

`scale = alpha / rank`. Common practice: set `alpha = rank` (gives scale = 1.0).
Some implementations use `alpha = 2 × rank` (scale = 2.0) as a warm-start boost.
The value matters much less than rank.

---

## Training decisions

### Truncated BPTT

Gradients do NOT flow backward through the KV-cache entries from earlier sequence
positions. This is intentional: it avoids O(seqLen²) backward work while having
negligible effect on LoRA training quality in practice. For most fine-tuning tasks
(instruction following, style adaptation, domain adaptation) truncated BPTT
converges to the same loss as full BPTT.

If you need full BPTT (e.g. training on very long documents where long-range
context is critical), the architecture supports it — you would store the attention
weights for ALL positions and sum gradients across them in `backwardLayer`. This
is a known extension point.

### Quantised frozen weights in backward

The transpose matVec in `backwardLayer` dequantises the frozen weight matrices
one row at a time. This means:

- **Memory**: O(hiddenDim) extra temporary allocation per transpose call, not
  O(model). A 22-layer TinyLlama fine-tune peak extra allocation is ~8MB.
- **Speed**: Each backward pass dequantises ~7 matrices per layer, each O(H²) or
  O(H×I). For TinyLlama at sequence length 16: ~300ms per step on a single CPU
  core. With a GPU this drops to ~10ms.
- **Correctness**: The same block-structure logic used in forward inference is used
  in the transpose path. Adjointness tests verify they are consistent.

### Weight decay

Weight decay (L2 regularisation) is applied only to **A**, not **B**. This is
deliberate: B starts at zero and its gradient is proportional to the adapter's
activation magnitude. Applying weight decay to B would counteract learning from
scratch, which is especially harmful early in training.

---

## Testing checklist

Run these in order. Each one validates a prerequisite for the next.

### ✅ 1. `LoraAdapterTest` — numerical gradient check

The most important test in the suite. Verifies the backward pass is
mathematically correct via finite differences.

```
mvn test -Dtest=LoraAdapterTest
```

**If gradA fails but gradB passes**: the outer-product term in gradA has an index
transposition. Check `a[r * inDim + j]` vs `a[j * rank + r]`.

**If all gradients are zero**: B is zero (expected at construction); use
`makeNonZero()` pattern (set B to small random values before running gradient
check).

**If gradients are 2×**: `zeroGrad()` was not called before backward.

### ✅ 2. `LoraAdapterSetTest` — round-trip serialisation

Verifies that saving and loading a checkpoint preserves weights bit-exactly.

```
mvn test -Dtest=LoraAdapterSetTest
```

**Watch for**: endianness bugs in DataOutputStream/DataInputStream. A corrupt
checkpoint will show float values that are garbage but not NaN (Java's endianness
issues produce wrong-magnitude values, not exceptions).

### ✅ 3. `LoraAdamOptimizerTest` — update direction and weight decay

Verifies gradient descent direction and that weight decay doesn't affect B.

```
mvn test -Dtest=LoraAdamOptimizerTest
```

**Watch for**: if loss doesn't decrease during training, first check that
parameters are moving in the correct direction (negative gradient direction). This
test verifies that.

### ✅ 4. `LoraTrainableHandlerTest` — transpose matVec and adjointness

The **adjointness test** is the most useful test in this class:

```
dot(A × x, v) == dot(A^T × v, x)
```

This holds for any matrix regardless of quantisation format. If this fails, the
block offsets in the transpose matVec implementation are wrong.

### ✅ 5. End-to-end loss decrease (manual)

After all unit tests pass, run a real fine-tune on a 16-token sequence for 50
steps and verify loss decreases:

```java
// Expected: loss starts near log(vocabSize) ≈ 10.4 for TinyLlama
// and decreases to < 5.0 within 50 steps on a repeated token sequence
float[] losses = new float[50];
for (int i = 0; i < 50; i++)
    losses[i] = handler.trainStep(new int[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, opt);
assert losses[49] < losses[0] * 0.8f;
```

If loss is flat: check that the LoRA adapter's `b` matrix is non-zero (it starts
at zero; it needs at least one gradient step to leave zero). After step 1, B will
be non-zero and the loss should start decreasing.

If loss diverges: learning rate is too high. Try 1e-5 instead of 1e-4.

---

## Common pitfalls

### "`optimizer.step()` called but loss doesn't change"

`optimizer.step()` is called **inside** `trainStep()`. Do not call it again from
outside — you would apply a second Adam step with zero gradients (because
`zeroAllGrads` runs at the start of the next `trainStep`).

### "Loss is constant at log(vocabSize)"

B starts at zero so the LoRA delta is exactly zero for the first forward pass.
After the first backward + Adam step, B becomes non-zero and the loss will begin
moving. If it's still constant after step 2, check that `loraAdapters.get(li, proj)`
is returning non-null (the adapter is actually registered for that layer/proj).

### "Memory grows during training"

The training KV cache is a local variable inside `trainStep()` — it is allocated
fresh each call and GC'd after. If memory grows, the inference KV cache is the
likely culprit: it grows per request ID and is never evicted. For training-only
usage, use a fixed request ID and the cache stays bounded.

### "Checkpoint loads but inference output is random"

After `LoraAdapterSet.load()`, the adapters have correct weights but
`LoraAdamOptimizer.reset()` must be called before resuming training. For
inference only, the optimizer state doesn't matter at all.