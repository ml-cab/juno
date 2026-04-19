# Juno — Distributed Java LLM Inference Engine

**Full Architecture Reference** · JDK 25 · Maven · Java-native · Commodity GPU Cluster

---

## Table of Contents

1. Vision
2. Hardware Stack
3. Maven Project Structure
4. API Module
5. System Architecture
6. KV Cache
7. REST / HTTP
8. Actors — Design Decisions
9. Integration Test Infrastructure
10. Activation Compression
11. Full Token Generation Data Flow
12. Full Configuration Reference
13. Technology Summary
14. Build Status
15. Real Model Inference
16. GPU Acceleration Layer
17. Phi-3 Family Support
18. Tensor Parallel
19. LoRA Fine-Tuning
20. KV Cache Wiring + JFR Instrumentation
21. AWS Deployment
22. Metrics Module
23. LoRA Inference Overlay (`--lora-play`)
24. Changelog Summary

---

## 1. Vision

A fully Java-native distributed LLM inference engine — no Python, no GIL, real threads, commodity hardware over premium hardware.

---

## 2. Hardware Stack

**Compute Nodes (×16 old PCs)**

| Component | Spec |
|-----------|------|
| GPU | 4 GB VRAM each — 16×4 GB = 64 GB total |
| CPU | 8+ core |
| RAM | 16–32 GB |
| Storage | NVMe SSD |

**Networking:** 10 GbE / 25 GbE, managed switch, RDMA.

---

## 3. Maven Project Structure


```
juno/
├── api/
├── registry/
├── coordinator/
├── node/           ← ForwardPassHandlerLoader (LoraAdapterSet overload added)
│                      EmbeddedNodeServer (reads juno.lora.play.path)
│                      NodeMain (forwards JUNO_LORA_PLAY_PATH env var)
├── kvcache/
├── tokenizer/
├── sampler/
├── health/
├── player/         ← ConsoleMain (--lora-play flag, /train-qa command)
│                      ClusterHarness (withLoraPlay() method)
├── metrics/
├── juno-node/      ← Fat jar (NodeMain)
└── juno-master/    ← Fat jar (CoordinatorMain, reads JUNO_LORA_PLAY_PATH)
```

---

## 4. API Module

### 4.1 OpenAPI 3.0 REST Spec

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/inference` | Blocking inference |
| POST | `/v1/inference/stream` | SSE streaming |
| POST | `/v1/models` | Load model |
| GET | `/v1/cluster/health` | Cluster overview |
| GET | `/` | Web console |

**Generated models (16):** `ApiError`, `RetryableError`, `ChatMessage`, `SamplingConfig`, `InferenceRequest`, `InferenceResponse`, `TokenEvent`, `LoadModelRequest`, `ModelDescriptor`, `ModelList`, `LayerRange`, `ShardAssignment`, `ShardMap`, `NodeDescriptor`, `NodeList`, `ClusterHealth`

**Generated interfaces (3):** `InferenceApi`, `ModelsApi`, `ClusterApi` — all implemented in coordinator.

### 4.2 gRPC Proto (`inference.proto`)

Internal node-to-node communication. Never exposed to clients. Java package: `cab.ml.juno.api.grpc`.

**Services:**
- `InferenceService` — client → coordinator (`Infer`, `InferStream`)
- `NodeService` — coordinator → each GPU node (`ForwardPass`, `LoadShard`, `UnloadShard`, `GetNodeStatus`)
- `RegistryService` — internal registry queries (`GetShardMap`, `RegisterNode`, `RecomputeShards`)

**Key proto fields:** `ForwardRequest.dtype` (field 9, `ActivationDtype`), `ForwardResponse.dtype` (field 5), `LoadShardRequest.tensor_rank` (field 7), `LoadShardRequest.tensor_world_size` (field 8).

---

## 5. System Architecture

```
[Client]
    │ REST (Javalin) / gRPC streaming
    ▼
[Coordinator]
    ├── GgufTokenizer
    ├── ChatTemplateFormatter
    ├── RequestScheduler (virtual threads)
    ├── Sampler
    ├── KVCacheManager
    └── GenerationLoop
              │
              │ gRPC activations (FLOAT16 / INT8 / FLOAT32)
              │
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3           |
    |  L 0-7        L 8-14       L 15-21          |
    |  + embed                   + output proj    |
    |  LoraAdapterSet (optional, read-only)       |
    +--------------------------------------------+
```

---

## 6. KV Cache

**Decision: Two tiers, RAM only. No disk IO.**

| Tier | Backend | Role |
|------|---------|------|
| GPU VRAM | CUDA device memory (bytedeco) | Hot, active sequences |
| JVM heap | Caffeine (W-TinyLFU) | Warm sequences; evicts under `-Xmx` |

> Disk tier removed: too complex for this deployment scale. OHC/Ehcache/Chronicle all had fatal transitive dep issues; Caffeine is already in the stack.

**Prefix cache:** Trie structure checked before every forward pass. 16 clients sharing a 500-token system prompt → compute once, reuse 16×.

---

## 7. REST / HTTP

Javalin 6.x, no Spring Boot. Metrics on port 9091 via Micrometer + Prometheus.

---

## 8. Actors — Design Decisions

### 8.1 Model Registry + Shard Planner

| Setting | Value |
|---------|-------|
| Registry placement | Hazelcast distributed `IMap` (no SPOF) |
| Seed node election | Score = w1·connectivity + w2·stability + w3·betweenness + w4·vram |
| Min seed nodes | 2 (never 1 — no SPOF) |
| Sharding strategy | Greedy, contiguous layer blocks, VRAM-aware |
| Weight format | GGUF (single file, quantization-aware) |

**Fair layer distribution (ShardPlanner):** The greedy algorithm is capped to prevent a large-VRAM node from starving later nodes:
```
maxLayers = min(layersFit, remainingLayers − (remainingNodes − 1))
```

### 8.2 Coordinator + Scheduler

| Setting | Value |
|---------|-------|
| Batching | Static micro-batching (`BatchConfig`: `maxBatchSize=8`, `batchWindowMs=50`) |
| Coordinator count | 2 (leader + standby) |
| Leader election | Hazelcast CP FencedLock |
| Priority queuing | `PriorityBlockingQueue` — HIGH=3, NORMAL=1, LOW=1 |
| Concurrency | Java 25 Virtual Threads + `CompletableFuture` |
| Queue full response | HTTP 503 + `Retry-After` estimate |

**`BatchConfig` presets:**
- `defaults()` — `maxBatchSize=8`, `batchWindowMs=50` (production)
- `disabled()` — `maxBatchSize=1`, `batchWindowMs=0` (per-request dispatch)

**Autoregressive loop (`GenerationLoop.generate`):**
1. `Tokenizer.encode(chatTemplate.format(messages))` → `int[] promptIds`
2. `kvKey = request.kvCacheKey()` (stable across turns when `sessionId` present)
3. Session: `PrefixCache.findLongestPrefix(promptIds)` → `startPos`
4. Prefill: `pipeline.forward(kvKey, slice, p)` for `p` in `startPos..promptLen-2`
5. Decode loop: forward → sample → stream → repeat until EOS or `maxTokens`
6. Session: `cachePrefix(promptIds, length, kvKey)`; **do NOT evict**
7. Stateless: `kvCache.evict(kvKey)` + no prefix cache

> **EOS suppression (two-layer defence):** (1) token ID check before `decodeToken()`; (2) `isEosMarker(piece)` catches `</s>`, `<|endoftext|>`, `<|eot_id|>`, `<end_of_turn>` emitted by non-EOS IDs.

### 8.3 KV Cache Manager

See [Section 6](#6-kv-cache). `NodeKVCacheAdapter` bridges handler-local `float[][]` arrays and `KVCacheManager`. Write-through on every token position; restore on local miss via GPU→CPU tier lookup.

### 8.4 Health Monitor

| Setting | Value |
|---------|-------|
| Node liveness | Hazelcast `memberRemoved` event |
| GPU health probe | CUDA every 5 s, published to Hazelcast `IMap` |
| VRAM warning | 90% → log only |
| VRAM critical | 98% → circuit open → reshard |
| Metrics | Micrometer + Prometheus via JDK `HttpServer` (port 9091) |

**Fault Tolerance (three classes in coordinator):**

- **`RetryPolicy`:** `none()` / `once()` (default, 50 ms backoff) / `aggressive()` (3 attempts, 100 ms)
- **`FaultTolerantPipeline`:** wraps `List<NodePipeline>` with per-node `CircuitBreaker`. Skips OPEN circuits; on exhaustion throws `PipelineUnavailableException`.
- **`HealthReactor`:** owns `HealthEvaluator` + `FaultTolerantPipeline`. Maps `VRAM_CRITICAL`/`NODE_STALE`/`NODE_RECOVERED` events to circuit state changes.

### 8.5 Tokenizer

`GgufTokenizer` auto-detects SentencePiece BPE vs GPT-2 BPE from GGUF metadata.

**Chat template resolution order:** (1) exact key → (2) substring match longest-first → (3) `chatml` fallback.

Template keys: `llama3`, `mistral`, `gemma`, `chatml`, `tinyllama`/`zephyr`, `phi3`/`phi-3`.

> **Critical for LoRA:** training and inference must use the same template key. The `[TRACE] model type` line at LoRA REPL startup shows the detected key. Mismatch causes complete failure to recall trained facts even when loss converged.

---

## 9. Integration Test Infrastructure

`InProcessClusterIT` (6 tests) · `ThreeNodeClusterIT` (9 tests) · `ModelLiveRunnerIT` (8 checks, `-Pintegration`) · `TensorParallelClusterIT` (5 tests)

---

## 15. Real Model Inference

### 15.1 GgufReader

Pure Java GGUF v2/v3 parser. Reads tensor metadata on open, loads and dequantises on demand (cached after first access). No JNI.

**Supported quantisation types:** F32, F16, BF16, Q8_0, Q4_0, Q4_K (Q4_K_M), Q5_K, Q6_K, Q2_K, Q3_K.

**Public API:** `GgufReader.open(Path)`, `r.tensor(name)` → `float[]`, `r.tensorRaw(name)` → `QuantizedTensor` (raw bytes, lazy dequant), `r.metaInt/Long/Float/String(key, default)`.

> `tensorRaw()` is critical for memory efficiency. Eager `tensor()` for all 32 layers of phi-3.5-mini = ~14.5 GB; lazy `tensorRaw()` = ~2 GB.

### 15.2 LlamaConfig

Extracts hyperparameters from GGUF `llm.*` / `{arch}.*` metadata. Handles `vocabSize` by taking `max(arch_vocab_size, tokenizer_token_count)` — critical for Phi-3 where EOS ID 32000 sits at the tokenizer boundary.

**TinyLlama-1.1B values:** `arch=llama hidden=2048 layers=22 heads=32 kvHeads=4 headDim=64 ffn=5632 vocab=32000 eps=1e-05 ropeTheta=10000`

### 15.3 LlamaTransformerHandler

Full LLaMA-family transformer, pure Java. Supports MatVec backend injection.

**Per-layer computation:** rmsNorm → Q/K/V projections → RoPE → KV cache write → GQA → residual → rmsNorm → SwiGLU FFN → residual.

**GQA:** 32Q / 4KV = 8 query heads sharing each K/V pair (TinyLlama).

**MatVec dispatch table:**

| Type | Backend |
|------|---------|
| `instanceof CudaMatVec` | `DeviceHalfMatrix` — weights dequantised to FP32 once, converted to FP16, uploaded at load time |
| CPU | `matVec(QuantizedTensor, ...)` — F32, Q8_0, Q4_K, Q5_K, Q6_K — all `IntStream.parallel()` |

**GPU path (Llama):** same as Phi-3: `DeviceHalfMatrix` + `cublasHSSgemvStridedBatched`; `cudaMalloc` OOM during upload closes partial buffers and falls back to CPU quantised matmul. **`ForwardPassHandler.releaseGpuResources()`** closes device matrices on shard unload.

**GPU path (Phi-3):** see [Section 17](#17-phi-3-family-support) — `DeviceHalfMatrix` + `cublasHSSgemvStridedBatched`; OOM during upload closes partial buffers and falls back to CPU quantised matmul for those projections.

**`ForwardPassHandlerLoader.selectBackend()`:** reads `JUNO_USE_GPU`; if true and CUDA available → `new CudaMatVec(GpuContext.shared(dev))` with `dev = Integer.getInteger("juno.cuda.device", 0)` (validated against `CudaAvailability.deviceCount()`), else `CpuMatVec.INSTANCE`.

### 15.4 GgufTokenizer

SentencePiece BPE or GPT-2 BPE — auto-detected from `tokenizer.ggml.model` GGUF key.

- **SentencePiece:** ▁ (U+2581) space prefix, greedy BPE merge, BOS prepend
- **GPT-2:** Ġ (U+0120) space prefix, special tokens pre-split on `<|...|>` boundaries
- `decodeToken()` replaces ▁/Ġ with space (both streaming and batch paths)

### 15.5 Prefill / Decode Split

```
Prefill:  for p in 0..promptLen-2: pipeline.forward(kvKey, [promptIds[p]], p)
          startPos = promptLen - 1
Decode:   loop { forward → sample → stream } until EOS or maxTokens
```

### 15.6 Token ID Transport: First-Node Protocol

- **Node 1 (`hasEmbeddings=true`):** `activation` field carries packed `int32` token IDs
- **Nodes 2+ (`hasEmbeddings=false`):** `activation` field carries `ActivationCodec.encode(floats, dtype)` bytes

---

## 16. GPU Acceleration Layer

### MatVec Interface

```java
interface MatVec {
    float[] sgemv(float[] A, float[] x, int rows, int cols);
}
```
Prefill:  for p in 0..promptLen-2: pipeline.forward(kvKey, [promptIds[p]], p)
          startPos = promptLen - 1
Decode:   loop { forward → sample → stream } until EOS or maxTokens
```

### 15.6 Token ID Transport: First-Node Protocol

- **Node 1 (`hasEmbeddings=true`):** `activation` field carries packed `int32` token IDs
- **Nodes 2+ (`hasEmbeddings=false`):** `activation` field carries `ActivationCodec.encode(floats, dtype)` bytes

---

## 16. GPU Acceleration Layer

### MatVec Interface

**`MatVec`** declares **`sgemv(float[] A, float[] x, int rows, int cols)`** for full host matrices. **`default`** **`sgemv(DeviceFloatMatrix A, float[] x)`** / **`sgemv(DeviceHalfMatrix A, float[] x)`** throw unless overridden — **`CudaMatVec`** implements both (FP32 and FP16 resident paths).

**Implementations:**
- `CpuMatVec` — `IntStream.range(0, rows).parallel()` for rows ≥ 256, plain loop below threshold
- `CudaMatVec` — org.bytedeco cuBLAS. Row-major `A[rows×cols]` maps to **`cublasSgemv_v2`** with **`CUBLAS_OP_T`**, **`m=cols`**, **`n=rows`**, **`lda=cols`** (same as a **`GEMV`** on the transpose in column-major layout).
  - **Host weights:** per-call alloc, H2D full `A` + `x`, kernel, D2H `y`, free — used in tests and diagnostics (serialized on `GpuContext.cublasSerializationLock()`).
  - **`sgemv(DeviceFloatMatrix, x)`:** `A` stays on device (FP32); **`cudaMemcpyAsync`** for `x` / `y`, **`cublasSetStream_v2`** + per-thread non-blocking CUDA stream, **`cudaStreamSynchronize`** before reading `y` — JFR **`cuda-resident`**.
  - **`sgemv(DeviceHalfMatrix, x)`:** `A` on device as IEEE FP16; pinned or pageable FP16 staging for `x`; **`cublasHSSgemvStridedBatched`** (`batchCount=1`) with the **same `(trans, m, n, lda)`** as `Sgemv_v2`; async D2H — JFR **`cuda-resident-fp16`**.

### cuBLAS Mapping

```
Row-major A[rows×cols] stored in memory == column-major A^T[cols×rows]
→ call cublasSgemv with CUBLAS_OP_T, m=cols, n=rows, lda=cols
→ computes (A^T)^T × x = A × x
```

### GpuContext

`GpuContext.init(deviceId)` — dedicated handle, **`close()`** destroys cuBLAS. **`GpuContext.shared(deviceId)`** — one lazily created handle **per device index** for the JVM; **`close()`** is a no-op so long-lived nodes do not tear down CUDA under other users. **`cublasSerializationLock()`** — intrinsic mutex; **`CudaMatVec`** must hold it while changing the handle’s CUDA stream or launching work. Multi-GPU: call **`shared(n)`** for each visible device (`CudaAvailability.deviceCount()`).

### CudaAvailability

Wraps `cudaGetDeviceCount()` in try/catch. Result cached at class load.

> **FD inheritance hazard:** `CudaAvailability` opens CUDA device FDs. If triggered in coordinator JVM before `ClusterHarness` forks node JVMs, node processes inherit those FDs and crash. Guard with `Boolean.getBoolean("juno.gpu.test")` in `@BeforeAll`. `GpuForwardPassIT` excluded from default failsafe; activated via `-Pgpu`.

---

## 17. Phi-3 Family Support

### Phi3TransformerHandler

Identical to `LlamaTransformerHandler` in attention math, KV cache, and `MatVec` injection. Differs in tensor layout:

**Difference 1 — Fused QKV:** `blk.{i}.attn_qkv.weight [H + kvDim + kvDim, H]`
- Rows `[0, H)` → Q; `[H, H+kvDim)` → K; `[H+kvDim, end)` → V
- Kept as one `QuantizedTensor`; row-range `matVec` extracts at call time

**Difference 2 — Fused gate+up FFN:** `blk.{i}.ffn_up.weight [2*I, H]`
- Rows `[0, I)` → gate; `[I, 2*I)` → up
- Same treatment

**phi-3.5-mini hyperparameters:** `arch=phi3 hidden=3072 layers=32 heads=32 kvHeads=32 headDim=96 ffn=8192 vocab=32064`

### GPU weights (Phi-3 only, `CudaMatVec` backend)

Fused QKV / gate+up / output-projection slices that are dequantised for matmul are uploaded as **`DeviceHalfMatrix`**. Forward calls **`CudaMatVec.sgemv(DeviceHalfMatrix, x)`** (see [Section 16](#16-gpu-acceleration-layer)). If **`cudaMalloc`** or cuBLAS setup fails during upload, device buffers created so far are **`close()`**d and that slice uses the **CPU quantised** path (`LlamaTransformerHandler.matVec`-style reference) instead. **`Phi3TransformerHandler.releaseGpuResources()`** frees device buffers on unload.

### ForwardPassHandlerLoader

```java
    if (adapters != null) {
        return LoraTrainableHandler.load(modelPath, context, adapters);
    }
    return switch (readArchitecture(modelPath)) {
        case "phi3" -> Phi3TransformerHandler.load(modelPath, context, backend);
        default     -> LlamaTransformerHandler.load(modelPath, context, backend);
    };
}
```

When `adapters != null`, the handler is always `LoraTrainableHandler` (implements `ForwardPassHandler`; inference-only, no optimizer attached). Passing `null` gives the standard dispatch.

---

## 18. Tensor Parallel

### Overview

`pType: tensor` — every node holds **all** transformer layers but owns only a horizontal weight slice.

| Strategy | Pipeline (vertical) | Tensor (horizontal) |
|----------|---------------------|---------------------|
| Each node holds | N/nodes layer weight blocks | All layers, head slice [headStart, headEnd) |
| Per decode step | N−1 serial gRPC hops | N parallel gRPC calls + AllReduce |
| Scales | Model depth (VRAM) | Model width (throughput) |

### AllReduce (star / coordinator-centric)
1. Coordinator broadcasts full token bytes to all N nodes in parallel
2. Each node returns a partial logit vector (`vocabSize` floats)
3. Coordinator element-wise-sums: O(N × vocabSize) — negligible (~1 ms for 70B)
4. Sampler operates on summed logits

### Head distribution
Ceiling-division: 32 heads × 3 nodes = 10/11/11. Hard constraint: `numHeads % 2 == 0` (RoPE sin/cos pairing). `numHeads % worldSize == 0` is NOT required.

### New Classes

| Class | Module | Description |
|-------|--------|-------------|
| `ParallelismType` | registry | enum `PIPELINE \| TENSOR` |
| `TensorShardAssignment` | registry | all-layer assignment + rank + worldSize |
| `TensorShardPlanner` | registry | assigns unique rank; validates `numHeads % 2 == 0` |
| `TensorShardContext` | node | runtime: `headStart()`, `headEnd()`, `sliceDim()` |
| `TensorParallelPipelineClient` | player | `InferencePipeline` with parallel broadcast + AllReduce |

---

## 19. LoRA Fine-Tuning

### Design

For frozen weight `W` (outDim × inDim): `W_eff = W + (alpha/rank) × B × A`
- `A ∈ R^{rank × inDim}` — initialised ~N(0, 0.01)
- `B ∈ R^{outDim × rank}` — initialised 0 (ΔW = 0 at step 0)

TinyLlama-1.1B wq+wv, rank=8: frozen 1.1B params vs LoRA 720K params.

### Key Classes

| Class | Description |
|-------|-------------|
| `LoraAdapter` | Core math: forward delta, backward gradient |
| `LoraAdapterSet` | Keyed by (layerIndex, projectionName); `save()` / `load()` binary checkpoint |
| `LoraAdamOptimizer` | Adam with bias correction; weight decay on A only |
| `LoraTrainableHandler` | `ForwardPassHandler` for inference + `trainStep()` for training |
| `LoraTrainEvent` | JFR event `juno.LoraTrainStep`: step, loss, forward/backward/optimizer ms |

### ConsoleMain `/train-qa`

New REPL command that auto-generates 4 phrasings of a Q&A pair and trains with the correct chat template:

```
/train-qa What is my name? A: Dima
```

Generates: exact phrasing × 2 + `Can you tell me: ...` + `Please answer: ...`. The model type is detected from the model path and used to apply the matching chat template. `--lora-steps-qa N` controls steps per chunk; `--lora-early-stop F` stops early when loss delta falls below F.

### Diagnostic Tracing (`--verbose`)

| Trace line | What it shows |
|-----------|---------------|
| `[TRACE] model type (chat template key)` | Template detected at startup |
| `[TRACE] formatted training text (repr)` | Exact byte sequence sent to the model |
| `[TRACE] token count (excl. BOS)` | Number of tokens in the training sequence |
| `[TRACE] token IDs: [...]` | Raw token IDs (verbose only) |
| `[TRACE] step=N loss=F chunk=M/T ms=D` | Per-step loss during training |
| `[TRACE] inference model type` | Template at inference — must match training |

---
## 20. KV Cache Wiring + JFR Instrumentation

### NodeKVCacheAdapter

Bridges handler-local `float[][]` KV arrays and `KVCacheManager`.

- **`flush(requestId, absLayer, k, v, seqLen, kvDim)`** — write-through after each token position; serialises K+V as float32 LE into `KVBlock`; GPU LRU eviction fires here
- **`tryRestore(requestId, absLayer, kvDim)`** — on local miss; checks GPU tier then CPU; deserialises back to float arrays
- **`evict(requestId)`** — removes from manager (both tiers) and local HashMap

> **Byte-order critical:** always use explicit `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)`. Never call `.asReadOnlyBuffer()` on a `HeapByteBuffer` if byte order must be preserved — it silently reverts to `BIG_ENDIAN`.

### JFR Events

| Event | Class | Key Fields |
|-------|-------|------------|
| `juno.MatVec` | `MatVecEvent` | `backend` ("cpu"\|"cuda"\|"cuda-resident"\|"cuda-resident-fp16"), `rows`, `cols` |
| `juno.ForwardPass` | `ForwardPassEvent` | `handlerType`, `requestId`, `startPosition`, `layerCount` |
| `juno.Tokenizer` | `TokenizerEvent` | `tokenizerType`, `operation`, `inputLength`, `outputLength` |
| `juno.TemplateFormat` | `TemplateFormatEvent` | `modelType`, `messageCount`, `outputLength` |
| `juno.LoraTrainStep` | `LoraTrainEvent` | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` |

All events use `@StackTrace(false)` for low overhead. Backend label for quantised static `matVec` = `"cpu"` (all are pure-Java ForkJoinPool operations).


- **`flush(requestId, absLayer, k, v, seqLen, kvDim)`** — write-through after each token position; serialises K+V as float32 LE into `KVBlock`; GPU LRU eviction fires here
- **`tryRestore(requestId, absLayer, kvDim)`** — on local miss; checks GPU tier then CPU; deserialises back to float arrays
- **`evict(requestId)`** — removes from manager (both tiers) and local HashMap

> **Byte-order critical:** always use explicit `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)`. Never call `.asReadOnlyBuffer()` on a `HeapByteBuffer` if byte order must be preserved — it silently reverts to `BIG_ENDIAN`.


---

## 21. AWS Deployment

### juno-deploy.sh (`scripts/aws/juno-deploy.sh`)

Unified GPU + CPU cluster lifecycle script replacing older per-type scripts.

**Commands:** `setup` · `start` · `stop` · `teardown` · `status` · `scan-regions`

**Setup options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--instance-type` | `g4dn.xlarge` | EC2 instance type |
| `--node-count` | 3 | Inference node count |
| `--coordinator` | `node1` | `node1` (co-located) or `separate` (extra t3.medium) |
| `--model-url` | — | HuggingFace direct URL to GGUF |
| `--ptype` | `pipeline` | `pipeline\|tensor` |
| `--dtype` | `FLOAT16` | `FLOAT32\|FLOAT16` |
| `--jfr` | — | JFR duration (e.g. `5m`) |
| `--lora-play` | — | Local `.lora` file path (resolved to absolute via `realpath`) |

**Setup flow:** resolve AMI → cheapest AZ → create keypair + security group → launch nodes → wait for bootstrap (~5 min) → write `cluster-nodes.env` → `systemctl start juno-coordinator` → poll `/v1/cluster/health` → enter live monitor.

- **User-data passed via `file://`** — prevents double base64 encoding. CLI size trace now shows `first-line: #!/bin/bash`.
- **TRACE logs in `_build_node_userdata` redirect to stderr** — prevents contamination of captured user-data script.
- **`realpath` at `--lora-play` parse time** — relative paths are resolved immediately.
- **Early `[[ -f ]]` validation** in `setup()` — fails before any AWS spend.
- **Synchronous stop/patch/start** in `_scp_lora_to_nodes` — eliminates race condition with coordinator.

**State file:** `~/.juno-deploy-state` — persists instance IDs, SG, key, options.

**Live monitor:** per-node cpu%, mem MB/MB, node/coordinator service status; estimated cost; 20 s refresh. Ctrl+C → auto-stop.

```bash
cd scripts/aws
./launcher.sh juno-deploy.sh setup \
  --instance-type m7i-flex.large \
  --model-url https://huggingface.co/.../TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

**Web console:** `InferenceApiServer` serves `GET /` → self-contained HTML5 chat UI. Polls `/v1/cluster/health` every 10 s, streams tokens via Fetch `ReadableStream`. Verified on AWS eu-north-1 3-node CPU cluster.

---

## 22. Metrics Module

Five JFR event types:

| Event | Key Fields |
|-------|------------|
| `juno.MatVec` | `backend`, `rows`, `cols` |
| `juno.ForwardPass` | `handlerType` (`lora` when adapter applied), `requestId`, `startPosition` |
| `juno.Tokenizer` | `tokenizerType`, `operation`, `inputLength`, `outputLength` |
| `juno.TemplateFormat` | `modelType`, `messageCount`, `outputLength` |
| `juno.LoraTrainStep` | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` |

`handlerType = "lora"` in `juno.ForwardPass` events confirms the adapter is active during inference.

---

## 23. LoRA Inference Overlay (`--lora-play`)

### Overview

Pre-trained `.lora` adapter files can be applied read-only at inference time in any mode without entering the training REPL.

### Propagation chain

```
ConsoleMain
  --lora-play PATH
        │
        ├── local mode
        │     LoraAdapterSet.load(Path)
        │     ForwardPassHandlerLoader.load(model, ctx, backend, adapters)
        │     → LoraTrainableHandler (inference-only, no optimizer)
        │
        └── cluster mode
              ClusterHarness.withLoraPlay(path)
                    │
                    └── launchNode()
                          -Djuno.lora.play.path=PATH (per forked JVM)
                                │
                                └── EmbeddedNodeServer.NodeServiceImpl
                                      loraPlayPath = System.getProperty("juno.lora.play.path")
                                      loadShard() → LoraAdapterSet.load(Path.of(loraPlayPath))
                                      ForwardPassHandlerLoader.load(..., adapters)
```

### AWS deployment chain

```
juno-deploy.sh --lora-play /abs/path/model.lora
    │
    ├── realpath() → absolute path (prevents relative-path mismatches)
    ├── setup() validates [[ -f PATH ]] before launching any instances
    │
    └── _scp_lora_to_nodes() (called after bootstrap, before coordinator start)
          For each node — synchronously:
            1. scp  → /tmp/<basename>
            2. systemctl stop juno-node   (synchronous, waits for JVM exit)
            3. mv + chmod 644 /opt/juno/models/<basename>
            4. sed -i 'JUNO_LORA_PLAY_PATH=/opt/juno/models/<basename>' /etc/juno/node.env
            5. systemctl start juno-node  (synchronous, ~2s to bind gRPC)
            6. [TRACE] node.env patch confirmed in log
          Updates global LORA_PLAY_PATH to remote absolute path
    │
    └── _write_cluster_env_and_start_coordinator()
          JUNO_LORA_PLAY_PATH=/opt/juno/models/<basename> → cluster-nodes.env
          systemctl start juno-coordinator
```

The coordinator starts only after all nodes are confirmed active — loadShard RPCs always find nodes with adapters already loaded.

### Known gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| `lora=none` in node log | `juno.lora.play.path` property not set | `--lora-play` missing, or wrong path |
| Node log shows relative path | `realpath` not applied at parse time | Updated in Session 26 |
| Coordinator starts before nodes have new env | `--no-block` restart; old instance seen as `active` | Synchronous stop+start per node (Session 26) |
| `model load failed: ../models/...` | Relative path baked into `cluster-nodes.env` | `LORA_PLAY_PATH` updated to remote absolute path before coordinator starts (Session 26) |
| cloud-init skips bootstrap script | Double base64 encoding via pre-encoded `--user-data` | Pass `file://` path to AWS CLI (Session 26) |
| cloud-init sees ANSI codes before `#!/bin/bash` | `log()` calls in `_build_node_userdata` wrote to stdout | Redirect to stderr with `>&2` (Session 26) |

