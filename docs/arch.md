# Juno — Distributed Java LLM Inference Engine

**Full Architecture Reference**
JDK 25 · Maven · Java-native · Commodity GPU Cluster

---

## Table of Contents

1. [Vision](#1-vision)
2. [Hardware Stack](#2-hardware-stack)
3. [Maven Project Structure](#3-maven-project-structure)
4. [API Module](#4-api-module)
5. [System Architecture](#5-system-architecture)
6. [KV Cache](#6-kv-cache)
7. [REST / HTTP](#7-rest-http)
8. [Actors — Design Decisions](#8-actors-design-decisions)
9. [Integration Test Infrastructure](#9-integration-test-infrastructure)
10. [Activation Compression](#10-activation-compression)
11. [Full Token Generation Data Flow](#11-full-token-generation-data-flow)
12. [Full Configuration Reference](#12-full-configuration-reference)
13. [Technology Summary](#13-technology-summary)
14. [Build Status](#14-build-status)
15. [Real Model Inference](#15-real-model-inference)
16. [GPU Acceleration Layer](#16-gpu-acceleration-layer)
17. [Phi-3 Family Support](#17-phi-3-family-support)
18. [Tensor Parallel](#18-tensor-parallel)
19. [LoRA Fine-Tuning](#19-lora-fine-tuning)
20. [KV Cache Wiring + JFR Instrumentation](#20-kv-cache-wiring-jfr-instrumentation)
21. [AWS Deployment](#21-aws-deployment)
22. [Metrics Module](#22-metrics-module)
23. [Changelog Summary](#23-changelog-summary)

---

## 1. Vision

A fully Java-native distributed LLM inference engine that runs large language models across a cluster of commodity GPUs — replacing the need for a single expensive high-VRAM card with a network of affordable machines.

**Core philosophy:**
- No Python. No GIL. Real threads.
- No Spring Boot. No framework bloat.
- Commodity hardware over premium hardware
- Java distributed tooling (Hazelcast, gRPC) over NCCL/MPI
- Pipeline parallelism — LAN friendly, no InfiniBand required
- Open source, Java ecosystem first

---

## 2. Hardware Stack

**Compute Nodes (×16 old PCs)**

| Component | Spec |
|-----------|------|
| GPU | 4 GB VRAM each — 16×4 GB = 64 GB total |
| CPU | 8+ core modern (AMD/Intel) |
| RAM | 16–32 GB per node (KV cache JVM heap) |
| Storage | NVMe SSD (fast shard loading) |

**Networking**

| Component | Spec |
|-----------|------|
| NIC | 10 GbE (start) / 25 GbE (ideal) — ~$30–100/each |
| Switch | Managed, jumbo frames enabled — ~$200–500 |
| Protocol | RDMA — GPU to wire, bypasses CPU entirely |

Total extra networking cost: ~$800–1000 for 16 machines. Far cheaper than one 64 GB GPU.

---

## 3. Maven Project Structure

Multi-module Maven project, JDK 25 throughout.
**Group ID:** `cab.ml.juno` · **Artifact ID:** `juno` · **Version:** `0.1.0-SNAPSHOT`

```
juno/                     ← parent POM
├── api/                  ← OpenAPI spec + gRPC proto + generated models/interfaces
├── registry/             ← NodeDescriptor, ShardPlanner, ShardMap, SeedScorer,
│                           ParallelismType, TensorShardPlanner, TensorShardAssignment
├── coordinator/          ← GenerationLoop, RequestScheduler, InferenceRequest,
│                           GenerationResult, TokenConsumer, RequestPriority,
│                           BatchConfig, BatchEntry, FaultTolerantPipeline,
│                           RetryPolicy, PipelineUnavailableException,
│                           HealthReactor, InferenceApiServer, SseTokenConsumer
├── node/                 ← ForwardPassHandler, LlamaTransformerHandler,
│                           Phi3TransformerHandler, ForwardPassHandlerLoader,
│                           GgufReader, LlamaConfig, ActivationCodec, ActivationDtype,
│                           MatVec, CpuMatVec, CudaMatVec, GpuContext, CudaAvailability,
│                           NodeKVCacheAdapter, ForwardRequest, ForwardResult,
│                           ShardContext, TensorShardContext, NodeConfig,
│                           LoraAdapter, LoraAdapterSet, LoraAdamOptimizer,
│                           LoraTrainableHandler, LocalInferencePipeline
├── kvcache/              ← KVCacheManager, GpuKVCache, CpuKVCache,
│                           PrefixCache, KVBlock, KVKey, KVCache, LayerRange
├── tokenizer/            ← Tokenizer, SimpleTokenizer, DJLTokenizer,
│                           GgufTokenizer (SentencePiece BPE + GPT-2 BPE),
│                           ChatMessage, ChatTemplate, ChatTemplateFormatter
├── sampler/              ← Sampler, SamplingParams + full pipeline steps
├── health/               ← CircuitBreaker, CircuitState, HealthEvaluator,
│                           HealthEvent, HealthThresholds, NodeHealth
├── player/               ← ClusterHarness, EmbeddedNodeServer, NodeMain,
│                           ProcessPipelineClient, TensorParallelPipelineClient,
│                           ConsoleMain, ChatHistory, ChatModelType
├── metrics/              ← JfrMetricsExtractor, JfrModelMapper, JfrPercentiles,
│                           MetricsSnapshot, MetricsWriter, ModelsConfig,
│                           ModelsConfigLoader, MetricsMain
├── juno-node/            ← Fat jar (main: NodeMain)
└── juno-master/          ← Fat jar (main: ModelLiveRunner) + all integration tests
```

### Key Dependency Versions

| Dependency | Version |
|------------|---------|
| Java | 25 |
| Hazelcast | 5.4.0 |
| gRPC | 1.63.0 |
| Protobuf | 3.25.3 |
| bytedeco CUDA | 12.6-9.5-1.5.11 |
| DJL | 0.27.0 |
| Caffeine | 3.1.8 |
| Resilience4j | 2.2.0 |
| Micrometer | 1.13.0 |
| Javalin | 6.3.0 |
| openapi-generator | 7.5.0 |
| JUnit | 5.10.2 |

> **Removed from original design** (transitive dep failures): Spring Boot, OHC (dead repo), Ehcache 3 (JAXB mess), Chronicle Map (same).

---

## 4. API Module

### 4.1 OpenAPI 3.0 REST Spec (`openapi.yaml`)

Client-facing REST API served by the coordinator via Javalin. Generator: `openapi-generator-maven-plugin 7.5.0`, `jaxrs-spec` mode.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/inference` | Blocking inference |
| POST | `/v1/inference/stream` | SSE token-by-token streaming |
| POST | `/v1/models` | Load model, triggers sharding |
| GET | `/v1/models` | List all models |
| GET | `/v1/models/{modelId}` | Model status + shard assignment |
| DELETE | `/v1/models/{modelId}` | Unload model, free VRAM |
| GET | `/v1/cluster/health` | Cluster overview |
| GET | `/v1/cluster/nodes` | All node statuses |
| GET | `/v1/cluster/shardmap` | Current layer assignments |
| GET | `/` | Web console (HTML5 chat UI) |

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
    │ REST (Javalin) or gRPC streaming
    ▼
[Load Balancer]  HAProxy / Nginx
    │
    ┌─────────────────────┐
    ▼                     ▼
[Coordinator 1]    [Coordinator 2]
   LEADER             STANDBY
    │
    ├── Javalin REST server (port 8080)
    ├── Tokenizer (GgufTokenizer / DJL)
    ├── RequestScheduler (Virtual threads + CompletableFuture)
    ├── GenerationLoop (autoregressive)
    ├── Sampler (pure Java pipeline)
    ├── PrefixCache (Trie)
    └── InferencePipeline
              │
              │ gRPC        (data plane  — activations)
              │ Hazelcast   (control plane — commands, state, events)
              │
    ════════════════════════════════════
    ║      10/25 GbE RDMA Network     ║
    ════════════════════════════════════
         │         │         │              │
    [Node 1]  [Node 2]  [Node 3]  …  [Node 16]
    Layer 0–1  Layer 2–3  Layer 4–5      Layer N
    + Embed    GPU shard  GPU shard      + Output proj
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

**Configuration:**
```yaml
kv-cache:
  gpu:
    capacity-fraction: 0.85
    eviction: LRU
  cpu:
    max-bytes: 8589934592   # 8 GB
    eviction: W-TinyLFU
```

---

## 7. REST / HTTP

**Decision: No Spring Boot. Javalin 6.x (~1 MB jar, built on Jetty directly).**

```java
Javalin app = Javalin.create().start(8080);
app.post("/v1/inference",        ctx -> inferenceHandler.infer(ctx));
app.post("/v1/inference/stream", ctx -> inferenceHandler.inferStream(ctx));
app.post("/v1/models",           ctx -> modelsHandler.load(ctx));
app.get("/v1/cluster/health",    ctx -> clusterHandler.health(ctx));
```

Metrics scrape: JDK `HttpServer` (built-in since Java 6) + Micrometer on port 9091.

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

| Variant | Use |
|---------|-----|
| `GgufTokenizer` | Primary — SentencePiece BPE **or** GPT-2 BPE, auto-detected from `tokenizer.ggml.model` GGUF key |
| `DJLTokenizer` | Alternative (SentencePiece JNI) |

**GPT-2 BPE detection:** if `tokenizer.ggml.model == "gpt2"` → Llama 3+ path. Space prefix is Ġ (U+0120); special tokens pre-split on `<|...|>` boundaries.

**Chat template registry (`ChatTemplate.BUILT_IN`):**

| Key | Format |
|-----|--------|
| `llama3` | `<\|begin_of_text\|>...<\|eot_id\|>...<\|start_header_id\|>assistant` |
| `mistral` | `[INST] ... [/INST]` |
| `gemma` | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` |
| `chatml` | `<\|im_start\|>role\n...<\|im_end\|>\n` *(default fallback)* |
| `tinyllama` / `zephyr` | `<\|user\|>\n{content}</s>\n<\|assistant\|>\n` |
| `phi3` / `phi-3` | `<\|user\|>\n{user}<\|end\|>\n<\|assistant\|>\n` |

**Resolution order in `forModelType()`:** (1) exact key match → (2) substring match, longest key first (case-insensitive) → (3) `chatml` fallback.

> **Critical:** TinyLlama-1.1B-Chat-v1.0 requires `tinyllama`/`zephyr` template, not ChatML. `decodeToken()` must replace ▁ (U+2581) with a real space.

### 8.6 Sampler

Pure Java, zero external deps. Pipeline: temperature → topK → topP → softmax → repetition penalty → sample. Presets: `defaults()`, `deterministic()`, `creative()`.

> **Note:** `deterministic()` sets `greedy=true`. Using only `withTemperature(0.0f)` is NOT sufficient — `greedy=false` still routes through `weightedSample()`.

---

## 9. Integration Test Infrastructure

### 9.1 InProcessClusterIT (fast, zero network)
Wires 3 `StubForwardPassHandler`s via `LocalInferencePipeline` in the same JVM. No gRPC. Tests full `GenerationLoop` + `RequestScheduler` stack. ~250 ms total. **6 tests.**

### 9.2 ThreeNodeClusterIT (real network, real JVMs)
`ClusterHarness` forks 3 JVMs (ports 19092–19094, `-Xmx4g -XX:+UseZGC`). Each runs `NodeMain` → `EmbeddedNodeServer`. Coordinator uses `ProcessPipelineClient` with real gRPC. **9 tests.**

Memory budget (16 GB host): 3 nodes × 4 GB + 2 GB coordinator + 2 GB OS = 16 GB ✓

Recommended test model: `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` — vocab 32000, hidden 2048, 22 layers, ~670 MB. Layer split: 8/7/7.

### 9.3 ModelLiveRunner (real model; `java -jar juno-master.jar <model>`)
8 automated checks. Exits 0 on all-pass, 1 on any failure.

| # | Check | Description |
|---|-------|-------------|
| 1 | `hello_greeting` | ≥1 greeting word in response |
| 2 | `no_raw_sentencepiece_markers` | No ▁ (U+2581) in output |
| 3 | `question_response` | Non-empty response to "What is 2 plus 2?" |
| 4 | `greedy_determinism` | `SamplingParams.deterministic()` → identical text twice |
| 5 | `multi_turn_conversation` | 3-turn; `promptTokens > 20` |
| 6 | `float16_parity` | FLOAT16 path produces non-empty output |
| 7 | `tensor_parallel_generation` | 3-node tensor-parallel; non-empty output |
| 8 | `tensor_parallel_greedy_determinism` | Tensor-parallel greedy → identical text twice |

### 9.4 TensorParallelClusterIT (5 tests)
3 forked JVMs in TENSOR mode. Verifies AllReduce path, concurrent requests, vocab size consistency.

### 9.5 Concurrent request pattern
```java
List<CompletableFuture<GenerationResult>> futures = new ArrayList<>();
for (int i = 0; i < count; i++)
    futures.add(scheduler.submit(req_i, TokenConsumer.discard()));
CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                 .get(30, TimeUnit.SECONDS);
```

### 9.6 Run Commands
```bash
mvn verify -pl juno-master                                              # full suite
mvn verify -pl juno-master -Dit.test=InProcessClusterIT                # in-process only
java -jar juno-master/target/juno-master.jar /path/to/model.gguf       # live runner
mvn test -pl tokenizer,node                                             # unit tests only
mvn verify -DskipITs                                                    # skip ITs
```

---

## 10. Activation Compression

### Problem
At 70B scale (hidden=8192, seq=4096, FLOAT32): **64 MB per hop** → ~51 ms on 10 GbE.

### Solution
`ActivationDtype` field on `ForwardRequest`/`ForwardResponse` selects wire encoding per-request.

| Dtype | Size | Error | Notes |
|-------|------|-------|-------|
| `FLOAT32` | 4 B/elem | lossless | default |
| `FLOAT16` | 2 B/elem | ~0.1% relative | IEEE 754 half, pure Java bit manipulation |
| `INT8` | 1 B/elem (+4 B scale) | ~1% relative | symmetric quantisation, scale prefix |

**Network impact (70B, hidden=8192, seq=4096):**
- FLOAT32: 64 MB / ~51 ms
- FLOAT16: 32 MB / ~26 ms (saves 25 ms/hop)
- INT8: 16 MB / ~13 ms (saves 38 ms/hop)

### Implementation
- `ActivationDtype.java` — enum `FLOAT32 | FLOAT16 | INT8` (domain, not proto)
- `ActivationCodec.java` — stateless, thread-safe `encode(float[], dtype)` / `decode(byte[], dtype)`
- `ProcessPipelineClient` — encodes before each `ForwardRequest`; decodes after each `ForwardResponse`; final node always returns FLOAT32

> **Two `ActivationDtype` enums are intentional:** `cab.ml.juno.api.grpc.ActivationDtype` (proto/wire) and `cab.ml.juno.node.ActivationDtype` (domain). Bridge via `toProto()` / `fromProto()` in `ProcessPipelineClient`.

---

## 11. Full Token Generation Data Flow

```
 1.  Client  →  POST /v1/inference/stream
 2.  Javalin routes to InferenceHandler
 3.  Coordinator receives InferenceRequest (OpenAPI model)
 4.  RequestScheduler.submit(request, consumer)
         → CompletableFuture registered in ConcurrentHashMap
         → Virtual thread spawned
 5.  GenerationLoop.generate() starts
 6.  ChatTemplateFormatter wraps messages
 7.  Tokenizer.encode() → int[] tokens
 8.  PrefixCache.findLongestPrefix(tokens) → check shared prefix
 9.  pipeline.forwardFromPosition() or pipeline.forward()
10.  Node 1: embedding lookup + layers 0–N → activation (gRPC)
11.  Nodes 2..N: forward their layer ranges → pass activation via gRPC
12.  Last node: final layers + output projection → float[vocab] logits
13.  Logits returned to coordinator via gRPC
14.  Sampler: temperature → topK → topP → softmax → penalty → sample
15.  int nextToken → Tokenizer.decodeToken() → String piece
16.  TokenConsumer.accept(piece, tokenId, step) → SSE / gRPC stream
17.  Token appended to generated sequence
18.  Repeat from 8 until EOS or maxTokens
19.  future.complete(GenerationResult) — caller's join() returns
```

---

## 12. Full Configuration Reference

```yaml
cluster:
  name: juno-cluster
  seed-nodes:
    - 192.168.1.10:5701
    - 192.168.1.11:5701
  seed-node-count: 2
  backup-count: 2

coordinator:
  count: 2
  grpc-port: 9090
  http-port: 8080
  metrics-port: 9091
  max-queue-depth: 1000
  max-batch-size: 8
  preemption-enabled: true
  preemption-strategy: ABORT

scheduler:
  max-wait-ms: 50
  priority-weights:
    HIGH: 3
    NORMAL: 1
    LOW: 1

node:
  grpc-port: 9092
  device-id: 0
  vram-headroom-fraction: 0.10
  pType: pipeline   # pipeline | tensor

kv-cache:
  gpu:
    capacity-fraction: 0.85
    eviction: LRU
  cpu:
    max-bytes: 8589934592    # 8 GB
    eviction: W-TinyLFU

health:
  probe-interval-ms: 5000
  vram-warning-threshold: 0.90
  vram-critical-threshold: 0.98
  circuit-breaker:
    failure-rate-threshold: 50
    sliding-window-size: 10
    wait-duration-seconds: 30

sampling:
  defaults:
    temperature: 0.7
    top-k: 50
    top-p: 0.9
    repetition-penalty: 1.1
    max-tokens: 512
  profiles:
    deterministic:
      temperature: 0.1
      greedy: true
    creative:
      temperature: 1.2
      top-k: 100
      top-p: 0.95
```

---

## 13. Technology Summary

| Concern | Choice |
|---------|--------|
| Language | Java 25 |
| Build | Maven (multi-module) |
| GPU compute | org.bytedeco CUDA (cudart + cublas) |
| Distributed state | Hazelcast 5.x |
| Leader election | Hazelcast CP FencedLock |
| Data plane | gRPC + Protocol Buffers |
| Cluster messaging | Hazelcast Topics + IMap listeners |
| RDMA networking | jVerbs |
| Concurrency | Java 25 Virtual Threads + CompletableFuture |
| REST API server | Javalin 6.x (no Spring) |
| REST API spec | OpenAPI 3.0 — jaxrs-spec generator |
| KV Cache L1 | CUDA device memory |
| KV Cache L2 | Caffeine (JVM heap, W-TinyLFU) |
| Circuit breaker | Resilience4j |
| Metrics | Micrometer + Prometheus (JDK HttpServer) |
| Tokenizer | GgufTokenizer (built-in) / DJL SpTokenizer |
| Weight format | GGUF |
| Sampler | Pure Java, zero external deps |

---

## 14. Build Status

All modules **BUILD SUCCESS**. ~475 unit tests, 0 failures.

| Module | Classes | Notes |
|--------|---------|-------|
| api | OpenAPI + gRPC generated | OpenAPI + proto generated sources |
| registry | 14 | ShardPlanner, TensorShardPlanner, ParallelismType, … |
| tokenizer | 9 | GgufTokenizer (SentencePiece + GPT-2 BPE), ChatTemplate |
| sampler | 9 | Full pipeline, 3 preset profiles |
| kvcache | 8 | KVCacheManager, GpuKVCache, CpuKVCache, PrefixCache |
| health | 6 | CircuitBreaker, HealthEvaluator |
| node | 30+ | LlamaTransformerHandler, Phi3TransformerHandler, LoRA classes, MatVec backends |
| coordinator | 14 | GenerationLoop (session KV reuse), FaultTolerantPipeline, InferenceApiServer |
| player | 7 main | ClusterHarness, ConsoleMain (lora subcommand), TensorParallelPipelineClient |
| metrics | 8 | JFR-based extractor, MetricsMain |
| juno-master | ITs | ModelLiveRunner (8 checks), InProcessClusterIT (6), ThreeNodeClusterIT (9), TensorParallelClusterIT (5) |

**Pending:** Hazelcast leader election wiring in coordinator.

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
| `instanceof CudaMatVec` | `DeviceFloatMatrix` (weights uploaded once at load time) |
| CPU | `matVec(QuantizedTensor, ...)` — F32, Q8_0, Q4_K, Q5_K, Q6_K — all `IntStream.parallel()` |

**GPU path:** weights dequantised to `float[]` once and uploaded via `CudaMatVec.upload()`. Per-token: only `x` and `y` cross the bus.

**`ForwardPassHandlerLoader.selectBackend()`:** reads `JUNO_USE_GPU` system property; if true and CUDA available → `new CudaMatVec(GpuContext.init(0))`, else `CpuMatVec.INSTANCE`.

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

**Implementations:**
- `CpuMatVec` — `IntStream.range(0, rows).parallel()` for rows ≥ 256, plain loop below threshold
- `CudaMatVec` — `cublasSgemv_v2` via org.bytedeco cublas. Row-major → column-major via `CUBLAS_OP_T`. Per-call: alloc, H2D, kernel, D2H, free. `sgemv(DeviceFloatMatrix, x)` overload skips H2D for weight.

### cuBLAS Mapping

```
Row-major A[rows×cols] stored in memory == column-major A^T[cols×rows]
→ call cublasSgemv with CUBLAS_OP_T, m=cols, n=rows, lda=cols
→ computes (A^T)^T × x = A × x
```

### GpuContext

One per node JVM. `AutoCloseable`. `GpuContext.init(deviceId)` → `cublasCreate(handle)`. cuBLAS handles are thread-safe.

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

### ForwardPassHandlerLoader

```java
public static ForwardPassHandler load(Path modelPath, ShardContext context) {
    return switch (r.metaString("general.architecture")) {
        case "phi3" -> Phi3TransformerHandler.load(modelPath, context);
        default    -> LlamaTransformerHandler.load(modelPath, context);
    };
}
```

Adding a new architecture: implement a `ForwardPassHandler` subclass + add a `case` branch.

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
| `LoraAdapter` | Core math unit: `forward(x)`, `backward(gradDelta, x)`, `zeroGrad()` |
| `LoraAdapterSet` | Keyed by (layerIndex, projectionName). `save(Path)` / `load(Path)` binary checkpoint. `asMap()` for iteration. |
| `LoraAdamOptimizer` | Adam with bias correction. Weight decay on A only (not B) |
| `LoraTrainableHandler` | `ForwardPassHandler` for inference + `trainStep()` for training |
| `LoraTrainEvent` | JFR event `juno.LoraTrainStep`: step, loss, forward/backward/optimizer ms |
| `LoraMerge` | GGUF writer: bakes adapter into a new standalone model file |
| `LoraMergeMain` | CLI entry point for `juno merge` |

**Checkpoint format:** magic `0x4C4F5241` ("LORA"), version 1, per-adapter: key, rank, inDim, outDim, alpha, A weights, B weights.

### ConsoleMain LoRA Subcommand

```bash
./juno lora --model-path model.gguf --lora-rank 8 --lora-lr 1e-4
```

REPL commands: `/train <text>`, `/train-file <path>`, `/save`, `/reset`, `/status`, `/merge-hint`, `/help`.

**Verified:** TinyLlama-1.1B Q4_K_M, rank=8, 7 tokens → ~2.2 s/step, loss 6.97 → 3.62 in 50 steps.

### transposedMatVec

The backward pass requires `y = A^T × v`. All five quantisation types have parallel scatter-reduce implementations (`IntStream.range().parallel()` with thread-local accumulators). Bug: initial implementation lacked Q5_K/Q6_K cases; fallback was catastrophically slow (17+ hours for 6 tokens). Fixed by adding dedicated `transposedQ5K` / `transposedQ6K`.

### Native Merge (`juno merge`)

`LoraMerge.merge(modelPath, loraPath, outputPath)` produces a new GGUF file where the LoRA-patched projection tensors are stored as **F32** and all other tensors are copied verbatim.

**Why F32 and not re-quantise?**

| Metric | Value |
|--------|-------|
| Typical LoRA delta per element | ~6×10⁻⁴ |
| Q4_K quantisation noise (half-step) | ~3.3×10⁻³ |
| SNR if re-quantised | **0.18×** — delta erased |
| F32 precision for same weights | ~10⁻⁷ (SNR ~6000×) |

Re-quantising to Q4_K destroys the delta entirely; the merged model behaves identically to the base model. F32 preserves it. The trade-off is a larger file (~1 GB for TinyLlama 1.1B vs 667 MB Q4_K original).

**GGUF writer flow (5 steps):**
1. Copy header + KV section verbatim from source (`ggufFileOffset` → `metadataSectionEnd`).
2. Write new tensor-info section — patched tensors get `type=F32`, others keep original type; all offsets recomputed.
3. Write 32-byte alignment padding relative to output file position 0.
4. Data section: patched tensors written as F32 (dequantise → apply delta → write); all others raw bytes transferred verbatim.

Output is always a plain GGUF v3 even when source is a llamafile ZIP polyglot.

**GgufReader additions** required by the writer:

| Method | Description |
|--------|-------------|
| `ggufFileOffset()` | Byte position of GGUF header in source file (0 for `.gguf`, >0 for llamafile) |
| `metadataSectionEnd()` | First byte after the last KV pair — start of tensor-info section |
| `tensorOrder()` | Tensor names in file order (requires `LinkedHashMap` storage) |
| `tensorNelems(name)` | Total element count for a tensor |

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
| `juno.MatVec` | `MatVecEvent` | `backend` ("cpu"\|"cuda"\|"cuda-resident"), `rows`, `cols` |
| `juno.ForwardPass` | `ForwardPassEvent` | `handlerType`, `requestId`, `startPosition`, `layerCount` |
| `juno.Tokenizer` | `TokenizerEvent` | `tokenizerType`, `operation`, `inputLength`, `outputLength` |
| `juno.TemplateFormat` | `TemplateFormatEvent` | `modelType`, `messageCount`, `outputLength` |
| `juno.LoraTrainStep` | `LoraTrainEvent` | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` |

All events use `@StackTrace(false)` for low overhead. Backend label for quantised static `matVec` = `"cpu"` (all are pure-Java ForkJoinPool operations).

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
| `--jfr` | — | JFR recording duration (e.g. `5m`) |

**Setup flow:** resolve AMI → cheapest AZ → create keypair + security group → launch nodes → wait for bootstrap (~5 min) → write `cluster-nodes.env` → `systemctl start juno-coordinator` → poll `/v1/cluster/health` → enter live monitor.

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

Scans the project root for `*.jfr` files, maps each to a model entry, extracts metrics, writes `target/metrics/metrics.json`.

### JFR Filename Convention
`juno-<modelStem>-YYYYMMDD-HHMMSS.jfr` — model stem embedded so `JfrModelMapper` can correlate without manual configuration.

### Programmatic JFR Lifecycle (ConsoleMain)

- **Local mode:** `jdk.jfr.Recording` API; shutdown hook stops + extracts metrics inline
- **Cluster mode:** coordinator gets programmatic `Recording`; `ClusterHarness.withJfr()` injects `-XX:StartFlightRecording=...` into each forked node JVM → captures `juno.MatVec` / `juno.ForwardPass` from node processes

> In `./juno cluster` mode (without `--jfr` flag), inference events come from node JVMs — only coordinator JVM is recorded. Use `./juno local` for full MatVec/ForwardPass metrics.

### Usage
```bash
./juno local --model-path /models/TinyLlama.Q4_K_M.gguf --jfr 5m
# After session: juno-TinyLlama-1.1B-Chat-v1.0.Q4_K_M-20260403-142311.jfr
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json
```

**`models.json`** (edit to add models):
```json
{ "models": [
  { "name": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M", "path": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" }
]}
```

---

## 23. Changelog Summary

| Session | Key Changes |
|---------|-------------|
| 1–3 | Foundation: sampler, tokenizer, registry, kvcache, coordinator stubs |
| 4 | Real model: `GgufReader`, `LlamaConfig`, `LlamaTransformerHandler`, `GgufTokenizer`, prefill/decode split |
| 5 | Bug fixes: TinyLlama template (ChatML→Zephyr), ▁ leak in streaming, Q6_K dequant loop |
| 6 | Performance: parallel `matVec` (9× speedup), parallel shard loading, EOS piece suppression, FLOAT16 default |
| 7 | Module restructure: `player` module extracted, `ModelLiveRunner` replaces `TinyLlamaLiveIT` |
| 8 | `run.sh` / `run.bat` unified launcher, logback config, `ModelLiveRunner` all 6 tests green |
| 9 | Session KV cache reuse (flat latency per turn), `InferenceRequest.sessionId`, `GenerationLoop.evictSession()` |
| 10 | GPU acceleration: `MatVec` interface, `CudaMatVec` (cublasSgemv), `GpuContext`, `CudaAvailability`, `GpuForwardPassIT` |
| 11 | Phi-3 family: `Phi3TransformerHandler` (fused QKV/FFN tensors), `ForwardPassHandlerLoader`, `QuantizedTensor` lazy dequant (OOM fix), vocab size fix, template routing fix |
| 12 | Pure rename refactor: `GpuForwardPassHandler` merged into `LlamaTransformerHandler`, `CublasMatVec`→`CudaMatVec`, etc. |
| 13 | Tensor parallel: `ParallelismType`, `TensorShardPlanner`, `TensorShardContext`, `TensorParallelPipelineClient`, `TensorParallelClusterIT`; `numHeads % 2 == 0` constraint (not worldSize); `LlamaTransformerHandler` OOM fix (lazy `QuantizedTensor` weights); ModelLiveRunner +2 tensor tests |
| 14 | LoRA fine-tuning: `LoraAdapter`, `LoraAdapterSet`, `LoraAdamOptimizer`, `LoraTrainableHandler`; `transposedMatVec` Q5K/Q6K fix; ConsoleMain `lora` subcommand |
| 15 | KV cache wired: `NodeKVCacheAdapter`, `EmbeddedNodeServer` wiring; JFR: `MatVecEvent`, `ForwardPassEvent`, `TokenizerEvent`, `TemplateFormatEvent`; `LlamaConfig.synthetic()` + `newTestInstance()` |
| 16 | Naming cleanup applied to disk (session 12 renames actually executed, stale files deleted) |
| 17 | AWS scripts: `scripts/aws/` with `launcher.sh`, `juno-infra.sh`, `juno-infra-ft.sh` |
| 18 | Meta-Llama 3 GPT-2 BPE support in `GgufTokenizer`; `MatVecEvent` from quantised path; `TemplateFormatEvent` fix |
| 19 | `metrics` module: `JfrMetricsExtractor`, `JfrModelMapper`, `MetricsMain`; JFR filename embeds model stem |
| 20 | GPU inference actually wired end-to-end: `selectBackend()`, `DeviceFloatMatrix` device weights, `matVecLayer()` dispatch, JFR backend label fix (`"cpu"` for quantised path) |
| 21 | `juno-deploy.sh`: unified AWS lifecycle script, web console in `InferenceApiServer`, verified on eu-north-1 |
| 22 | Q2_K + Q3_K quantisation support |
| 23 | Programmatic JFR lifecycle in `ConsoleMain` (local + cluster); `ClusterHarness.withJfr()`; `MetricsMain.extractToJsonMerged()`; `juno-deploy.sh --jfr`; `models.json` fixed |
| 24 | Configurable activation byte order (`--byteOrder BE\|LE`); `ActivationCodec` static dispatcher; `ActivationBECodec` / `ActivationLECodec`; byte order propagated through `ClusterHarness`, `juno-deploy.sh`, health endpoint |
| 25 | Code quality: `CyclicForwardPassHandler` moved to test scope, `StubForwardPassHandler` inner class in `EmbeddedNodeServer`, docs sweep |
| 26 | Native LoRA merge: `LoraMerge`, `LoraMergeMain`, `juno merge` subcommand. Patched tensors written as F32 (re-quantising to Q4_K erases deltas smaller than quantisation noise). `GgufReader` extended with `ggufFileOffset()`, `metadataSectionEnd()`, `tensorOrder()`, `tensorNelems()`, `LinkedHashMap` tensor storage. Three re-quantiser bugs fixed: Q4_K `d = maxRange/63` → `/(63×15)`, Q5_K `/(63×31)`, Q3_K scRaw packing rewritten. |