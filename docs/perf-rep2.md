```
read please @CLAUDE.md and the rest what is required
then please give me performance review of overall the project.
think of this questions:
* how to make inference faster on CPU
* how to make inference faster on GPU
* how to make lora faster
List me all the points you can see. Think hard!
```

Now I have a thorough picture of the entire project. Let me compile the full performance review.Now I have everything I need. Here is the full performance review.

---

# Juno — Performance Review

This is a thorough technical audit, built from reading `CLAUDE.md`, all docs, and the actual source code. The project is genuinely impressive — a 19-session, 100% Java distributed inference engine with real transformer math, LoRA training, tensor parallelism, and JFR instrumentation. The architecture decisions are sound. The performance gaps below are real and addressable without redesigning anything.

---

## How to Make CPU Inference Faster

### 1. 🔴 ForkJoinPool oversubscription across JVMs — primary throughput killer

--WASTED--

Every forked node JVM calls `ForkJoinPool.commonPool()` with default parallelism = `availableProcessors() - 1`. On an 8-core machine, three JVMs each believe they own 8 cores — 21 ForkJoin threads compete for 8 real cores. This is the reason throughput is 1–2 tok/s instead of the expected 5–8 tok/s. The fix is in `run.sh` when forking each node:

```bash
-Djava.util.concurrent.ForkJoinPool.common.parallelism=$(($(nproc) / NUM_NODES))
```

For the `--local` in-process mode this is less of an issue since all shards share the same JVM and the pool is naturally shared.

--WASTED--

Instead by prio:

4 - No Vector API / SIMD — matVec is 37% of forward pass time and is the most vectorizable code in the system. 8× potential speedup on the inner dot-product loop.

5 - gRPC hop overhead — responsible for the 200–350ms/token gap between local and cluster. ActivationCodec running sequentially and in big-endian makes this worse.

2 - ByteBuffer allocations in matVec — secondary GC pressure, addressed by the ThreadLocal scratch buffer approach.

- ForkJoinPool parallelism — genuinely matters for tensor-parallel mode. Not relevant for pipeline mode. I overstated this significantly in the review.


### 2. 🔴 ByteBuffer allocated per row inside `matVecF32raw` — GC flooding

Already documented in `perf-rep.md` but deserves emphasis: `ByteBuffer.wrap(raw).order(LE)` inside the parallel lambda creates a new wrapper object per row. For a 2048×2048 matrix that is 2048 objects per matVec call × 154 matVec calls per token = ~315,000 short-lived allocations per token. Fix is zero-cost bit manipulation with no allocation:

```java
private static float readF32LE(byte[] raw, int byteOffset) {
    int bits = (raw[byteOffset] & 0xFF)
             | ((raw[byteOffset+1] & 0xFF) << 8)
             | ((raw[byteOffset+2] & 0xFF) << 16)
             | ((raw[byteOffset+3] & 0xFF) << 24);
    return Float.intBitsToFloat(bits);
}
```

The same pattern applies in `matVecQ8_0raw`, `matVecQ4Kraw`, `matVecQ5Kraw`, `matVecQ6Kraw` — each creates ByteBuffers inside their inner loops.

### 3. 🔴 `new float[rows]` on every matVec — sustained GC pressure

Every matVec call allocates a fresh output array: 154 allocations per token, ranging from 256 to 32,000 floats. Use a `ThreadLocal<float[]>` scratch buffer that only grows, never shrinks:

```java
private static final ThreadLocal<float[]> SCRATCH = ThreadLocal.withInitial(() -> new float[0]);

static float[] matVec(...) {
    float[] y = SCRATCH.get();
    if (y.length < rows) { y = new float[rows]; SCRATCH.set(y); }
    // write into y[0..rows-1], return y
}
```

This requires callers to treat the return value as transient (consumed before the next matVec call on the same thread). Since all call sites are sequential within a `transformerLayer()`, this is safe.

### 4. 🔴 No SIMD / Vector API — the single highest-ROI change

The `--enable-preview` flag is already set but `jdk.incubator.vector` is not used anywhere. The matVec inner loop:

```java
for (int c = 0; c < cols; c++)
    acc += A[base + c] * x[c];
```

…is the hottest code in the system and is a textbook SIMD target. With the Java Vector API and AVX2, you get 8-wide float MAC per cycle vs 1 scalar — up to **8× speedup** on the inner loop. For TinyLlama the FFN projections (5632×2048 and 2048×5632) dominate; those alone would see the full 8× gain.

```java
import jdk.incubator.vector.*;

static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;  // AVX2 = 8 floats

// Inner loop replacement:
FloatVector acc = FloatVector.zero(SPECIES);
int c = 0;
int len = SPECIES.loopBound(cols);
for (; c < len; c += SPECIES.length()) {
    acc = FloatVector.fromArray(SPECIES, A, base + c)
          .fma(FloatVector.fromArray(SPECIES, x, c), acc);
}
float result = acc.reduceLanes(VectorOperators.ADD);
for (; c < cols; c++) result += A[base + c] * x[c];  // scalar tail
```

### 5. 🟠 ActivationCodec: sequential encode/decode + wrong endianness

Every pipeline node hop encodes/decodes the full activation tensor. The FLOAT16 path is entirely sequential (`for` loop, not `IntStream.parallel()`) and uses `BIG_ENDIAN` on x86/ARM machines, forcing a byte-swap on every `putShort`/`getShort`. Fix: parallel loop + little-endian direct byte writes (no ByteBuffer at all). Also: Java 20+ has `Float.floatToFloat16(f)` as a compiler intrinsic — drop the manual `floatToHalf`/`halfToFloat` entirely since you're already on JDK 25.

### 6. 🟠 No batched prefill — O(N) sequential token processing

During prefill, `GenerationLoop` calls `pipeline.forward()` once per prompt token in a `for` loop. This means 500 prompt tokens = 500 sequential gRPC round-trips + 500 sequential attention operations. All major inference engines (vLLM, llama.cpp) process the entire prompt as a batch using matrix-matrix multiply (`SGEMM`) instead of 500 matrix-vector multiplies (`SGEMV`). SGEMM on 500 tokens is ~500× faster than 500 sequential SGEMVs because GPU compute dominates the latency. Even on CPU, batched attention with SIMD would be significantly faster. This is the main reason first-token latency is high.

### 7. 🟠 matVec parallelism threshold is arbitrary — needs per-machine tuning

The threshold `rows >= 256` for enabling parallel IntStream was chosen empirically on a specific machine. On machines with fewer cores, the overhead of forking ForkJoinPool tasks may exceed the gain even at 256 rows. The threshold should be measured at startup via a microbenchmark and stored as a static final, or exposed as a JVM property: `-Djuno.matvec.parallel.threshold=256`.

### 8. 🟡 SimpleTokenizer is not thread-safe

Documented in agent-arch.txt: `SimpleTokenizer uses HashMap + unsynchronized nextId — NOT thread-safe for concurrent encode() calls with new tokens`. Under concurrent requests with the `RequestScheduler` using virtual threads, this is a latent correctness bug. In production, always use `GgufTokenizer` (which is safe) and remove `SimpleTokenizer` from any concurrent path.

### 9. 🟡 KV cache stored as `HashMap<String, float[][]>` — fragmented heap

Each request's KV cache is a separate `float[][]` heap allocation per layer, per position — not contiguous memory. Under the JVM, GQA attention (`gqa()`) iterates over `kvCacheK.get(requestId)` which involves a HashMap lookup, then a float-array stride across scattered memory. A ring-buffer of contiguous `float[]` per request, pre-allocated at the start of the session, would eliminate HashMap overhead and improve CPU cache locality.

### 10. 🟡 Sampler applies softmax to full vocabulary before topK

The sampling pipeline applies temperature → topK → topP → **softmax** → penalty → sample. Softmax over 32,000 (TinyLlama) or 128,256 (Llama 3) floats every token is non-trivial. For topK=50, softmax only needs to run on the top-K elements. Move the topK filter before softmax, normalize only the top-K logits, and save 99.8% of the softmax work.

---

## How to Make GPU Inference Faster

### 11. 🔴 `cudaMalloc` + `cudaFree` on every `sgemv` call — catastrophic GPU overhead

This is the most serious GPU performance issue. Every single matVec call in `CudaMatVec.sgemv(float[], ...)`:
- calls `cudaMalloc` three times (d_A, d_x, d_y)
- copies the full weight matrix from host RAM to GPU VRAM
- computes sgemv
- copies result back
- calls `cudaFree` three times

For TinyLlama, d_A for a 2048×2048 weight matrix is 16MB. 154 matVec calls per token × (3× malloc + 2× full H2D memcpy + 1× D2H memcpy + 3× free) = the GPU spends more time on memory management than actual computation. The `DeviceFloatMatrix` path (`sgemv(DeviceFloatMatrix, float[])`) was designed to fix this by keeping A on the GPU permanently — but session 13 reverted all projection weights back to `QuantizedTensor` (raw bytes) to fix OOM. As a result, `CudaMatVec.sgemv(float[],...)` is the active path and it allocates + frees device memory on every call.

The correct fix: a `GpuMemoryPool` with a ring allocator over a pre-allocated device buffer. Pre-allocate e.g. 512MB on VRAM at startup. All temporary allocations (d_x, d_y) come from this pool with O(1) pointer arithmetic. Since d_x and d_y are consumed synchronously, they can be reused across calls.

### 12. 🔴 Weight matrices are not resident on GPU — full H2D transfer per matVec

Session 13 changed all projection weights from `float[]` to `QuantizedTensor` (raw Q4_K bytes) to fix OOM during tensor-parallel mode. The consequence: every matVec call dequantizes the weight matrix on CPU into a temporary `float[]`, then copies it to the GPU. For a 2048×2048 Q4_K matrix that is ~4.5MB of raw bytes → 16MB of dequantized floats → 16MB H2D transfer, for every single layer projection, every single token.

The correct approach is on-GPU dequantization: keep the raw Q4_K bytes on the GPU permanently, write a CUDA kernel that dequantizes and multiplies in a single pass (this is exactly what llama.cpp's CUDA backend does). This eliminates all per-token H2D weight traffic. As an intermediate step: dequantize weights to FLOAT16 on the CPU once at load time and keep them resident on GPU as `DeviceFloatMatrix` (halving the VRAM cost vs F32). At 4GB VRAM, TinyLlama's 22 layers × 7 projections × 8MB (F16 2048×2048) = ~1.2GB — fits.

### 13. 🔴 `cudaSetDevice` called on every sgemv — unnecessary overhead

Every `CudaMatVec.sgemv()` call starts with `cudart.cudaSetDevice(ctx.deviceIndex())`. This is a CUDA context switch operation. In a single-GPU node, the device never changes. Cache the current device in a `ThreadLocal<Integer>` and only call `cudaSetDevice` when it actually changes.

### 14. 🔴 No CUDA streams — fully synchronous execution

All CUDA calls are synchronous (no `cudaStream_t` usage). This means H2D memcpy, kernel launch, and D2H memcpy are fully sequential with no overlap. With CUDA streams, you can pipeline: copy x for layer i+1 while the kernel for layer i is running. Since each matVec is independent (A and x are read-only), this is safe and could provide meaningful latency reduction on the GPU path.

### 15. 🟠 sgemv instead of sgemm for batched decode

When `BatchConfig` is enabled and multiple requests are decoded in the same step, `generateBatch()` calls `forwardBatch()` which by default calls `forward()` N times serially. The note in the design says "Override in LlamaTransformerHandler: one CUDA batched matmul for the whole batch." This override has not been implemented. Batched decode using `cublasSgemm` (matrix × matrix) rather than N sequential `cublasSgemv` calls (matrix × vector) is 4–8× more GPU-efficient because it allows the GPU's tensor cores to engage.

### 16. 🟠 AllReduce in tensor-parallel mode is on CPU, not GPU

In tensor-parallel mode, the coordinator collects partial logit vectors from each node and sums them in Java (`float[] result = new float[vocabSize]` + element-wise add). For Llama 3 with vocabSize=128256 and 8 nodes, that is ~1M float additions per step running on one CPU core. More critically, partial logits must travel over gRPC from each GPU node back to the CPU coordinator, which adds network latency. The standard fix is in-place GPU reduction: each node reduces directly on GPU, or a NCCL ring AllReduce keeps the result on GPU. For the current Java-only approach, at minimum the reduction should use `IntStream.range().parallel()` to use all coordinator CPU cores.

### 17. 🟡 `cublasSetPointerMode_v2` called on every sgemv

`cublas.cublasSetPointerMode_v2(ctx.handle(), CUBLAS_POINTER_MODE_HOST)` is called inside every sgemv invocation. The pointer mode is a persistent handle property — set it once in `GpuContext.init()` and never touch it again during inference.

---

## How to Make LoRA Faster

### 18. 🔴 `transposedFallback` is O(rows × cols²) — catastrophic for unknown tensor types

If a weight tensor has an unhandled GGML type in `backwardLayer`, `transposedFallback` is called. It computes the transpose via full forward matVec per output column — O(cols) full matVec calls each scanning all rows. For `output.weight` (32000 × 2048), that is 2048 full-row scans per output column call, and it's called 32000 times = 2048 × 32000 × 2048 FLOPs. The warning is emitted but the computation still runs to completion at catastrophic cost. The fix is to ensure all GGML types used by target models have dedicated `transposedQ*` implementations, and to throw immediately on unknown types during development rather than silently running a fallback.

### 19. 🔴 No gradient checkpointing — O(seqLen × numLayers) activation memory

`trainStep()` stores a `LayerState` (activations, normed activations, attention output, FFN output) for every (position, layer) pair. For a 128-token sequence through 22 layers of TinyLlama, that is 128 × 22 = 2816 full `float[hiddenDim]` arrays = ~23MB. For longer sequences or larger models, this grows quadratically in memory. Gradient checkpointing (recompute activations during backward rather than storing them) trades memory for compute — the standard approach in all training frameworks.

### 20. 🔴 Backward pass dequantizes weight matrices per column — O(layers × 7 × cols × rows) CPU work

In `backwardLayer()`, each `transposedQ*` method dequantizes blocks of the weight matrix during the scatter-reduce loop. For 22 layers × 7 matrices per layer × 2048 columns × 2048 rows of dequantization, the backward pass does substantially more dequantization work than the forward pass. Caching the dequantized float representation of each weight matrix for the duration of one `trainStep()` (then releasing it) would avoid redundant dequantization. The dequantized weights are ~16MB for TinyLlama — affordable as a short-lived allocation.

### 21. 🟠 No batch training — only one sequence per `trainStep`

`trainStep(int[] tokens, optimizer)` processes a single token sequence per gradient step. Modern LoRA training batches 8–32 sequences per step, which amortizes the optimizer overhead and produces better gradient estimates. The current architecture could support this by accepting `int[][] tokensBatch` and averaging gradients across the batch before calling `optimizer.step()`. The existing `LoraAdapter.gradA` and `gradB` accumulators already support multiple `backward()` calls before `optimizer.step()`, so this is a relatively clean extension.

### 22. 🟠 No learning rate scheduler

`LoraAdamOptimizer.defaults(lr)` uses a fixed learning rate. Cosine decay from lr to lr/100 over the training steps is standard practice and consistently improves final quality. The `stepCount` field is already tracked in the optimizer — adding cosine decay is a 5-line change: `effectiveLr = lr * 0.5 * (1 + cos(PI * step / totalSteps))`.

### 23. 🟠 No gradient clipping — divergence risk on long sequences

There is no gradient norm clipping before `optimizer.step()`. For long sequences or high learning rates, gradients for the `B` matrix can explode (B starts at zero; early gradients can be large). Standard practice is `clipGradNorm(maxNorm=1.0)`: compute the global L2 norm of all gradients across all adapters, scale down if it exceeds the threshold. This prevents divergence and is cheap (one pass over all adapter gradients).

### 24. 🟡 Adam optimizer stores float32 moments — high memory for large rank

For rank=8 on wq+wv across 22 TinyLlama layers: 44 adapters × 2 matrices (A, B) × 2 moment buffers (m, v) × average ~16K floats = ~5.6M floats = ~22MB. At rank=64 this becomes ~176MB. 8-bit Adam (Dettmers et al.) quantizes the optimizer states to INT8 with per-block scaling, reducing moment memory by 4×. This matters when fine-tuning larger models with higher rank.

### 25. 🟡 Training KV cache allocated fresh on every `trainStep`

Inside `trainStep()`, the local `kvCache` (activation storage for the forward pass during training) is created as a new `List<LayerState[]>` per call. For short training sequences used repeatedly (e.g. the same 10-token sentence for 50 steps), this allocates and GCs the same structure 50 times. Reusing the training KV cache across steps (zeroing it at the start of each step rather than reallocating) would reduce GC pauses during the training loop.

### 26. 🟡 `LoraAdamOptimizer.step()` uses `IdentityHashMap` keyed by adapter object

`IdentityHashMap<LoraAdapter, float[]>` for first/second moment storage means moment lookup is by object identity — correct but means deserialized adapters (from `LoraAdapterSet.load()`) will not find existing moments unless `optimizer.reset()` is called. The design already requires `reset()` after loading a checkpoint, but this is a non-obvious footgun. Keying moments by the adapter's string key (`"layer_N_wq"`) would survive serialization/deserialization transparently and allow warm-resuming optimizer state from a checkpoint.

---

## General Architecture Observations

### 27. No continuous batching — static micro-batching only

`BatchConfig.of(8, 50ms)` uses static batching: wait up to 50ms or 8 requests, then dispatch as a batch. Production inference engines (vLLM, TensorRT-LLM) use continuous batching: new requests join in-flight batches mid-generation, and requests that finish EOS immediately free their KV slot. Static batching means a request that arrives at step 2 of an 8-request batch must wait for all 8 requests to complete before being processed — latency is bounded by the longest sequence in the batch.

### 28. MAX_SEQ_LEN is hardcoded at 2048

```java
private static final int MAX_SEQ_LEN = 2048;
```

Llama 3 supports 128K context; Phi-3.5 supports 128K context. This constant prevents using the models at their actual capabilities. It should be read from GGUF metadata (`llama.context_length`) and stored in `LlamaConfig`, with the KV cache pre-allocated accordingly (or grown on demand up to the model's max).

### 29. No speculative decoding

Speculative decoding (run a small draft model to produce N candidate tokens, then verify in parallel with the large model) can provide 2–5× throughput improvement for autoregressive generation. The `InferencePipeline` interface and `ForwardPassHandler` abstraction are already flexible enough to support this — it would require a `DraftHandler` that runs a small model and a verification step in `GenerationLoop`.

### 30. JFR instrumentation is correct but metrics extraction requires `--local` mode

`ProductivityMain` notes that in cluster mode, JFR attaches only to the coordinator JVM — `juno.MatVec` and `juno.ForwardPass` events (emitted in node JVMs) are invisible. This means the most important performance data (per-layer timing, matVec breakdown) is not captured in the normal cluster workflow. The solution is to start a JFR recording inside each forked node JVM from `ClusterHarness` — pass `-XX:StartFlightRecording` to each `ProcessBuilder` invocation and collect the per-node `.jfr` files after shutdown.
