# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 x 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Architecture

Two distribution strategies are supported, selected with `--pType`:

### Pipeline parallel (default — `--pType pipeline`)

Layers are split into contiguous blocks across nodes. The activation tensor flows
`node-1 -> node-2 -> node-3` in serial order. Each node computes a different
depth slice of the transformer. Adding nodes increases the total VRAM budget,
enabling larger models. N-1 sequential gRPC hops per decode step.

```
[Client]  REST (Javalin) / gRPC streaming
    |
[Coordinator]
    |-- GgufTokenizer       (BPE from GGUF metadata)
    |-- ChatTemplateFormatter
    |-- RequestScheduler    (virtual threads, CompletableFuture)
    |-- Sampler             (temperature / top-k / top-p / rep. penalty)
    |-- KVCacheManager      (GPU tier + CPU tier + PrefixCache trie)
    +-- GenerationLoop      (prefill + decode + session KV reuse)
              |
              | gRPC activations (FLOAT16 / INT8 / FLOAT32)
              | serial: node-1 -> node-2 -> node-3
              |
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3  ...      |
    |  L 0-7        L 8-14       L 15-21          |
    |  + embed                   + output proj    |
    |  NodeKVCacheAdapter wired into each handler |
    +--------------------------------------------+
```

### Tensor parallel (`--pType tensor`)

Every node holds all transformer layers but only a horizontal slice of the
weight matrices: attention heads `[headStart, headEnd)` and a proportional FFN
width slice. The coordinator broadcasts the input to all nodes simultaneously,
collects partial logit vectors from each node, and reduces them with an
element-wise sum (star AllReduce). Adding nodes increases throughput and reduces
per-node memory pressure. One broadcast + N parallel gRPC calls per decode step.

```
[Coordinator]
    +-- GenerationLoop
              |
              | broadcast same tokens to all nodes (parallel)
              |
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3  ...      |
    |  L 0-21       L 0-21       L 0-21           |
    |  heads 0-10   heads 11-21  heads 22-32      |
    |  rank=0       rank=1       rank=2            |
    +--------------------------------------------+
              |
              | partial logits from each node (parallel)
              |
    [AllReduce: element-wise sum -> full logit vector]
              |
    [Sampler]
```

Constraint: `numHeads % nodeCount == 0` (equal head slice per node).

### Handler routing

```
ForwardPassHandlerLoader  <- reads general.architecture from GGUF
    |
    phi3  -> Phi3TransformerHandler   (fused QKV + gate/up, quantized weights)
    *     -> LlamaTransformerHandler  (separate tensors, quantized weights)

MatVec (injected into handler):
    CpuMatVec    <- parallel IntStream
    CudaMatVec   <- cublasSgemv_v2 via JCublas2
                    weights uploaded once as DeviceFloatMatrix at load time;
                    forward pass copies only x and y across the bus per call

KV cache wiring (per node, after loadShard()):
    NodeKVCacheAdapter  <- serialises float[][] K/V into KVBlock,
                           flushes write-through to KVCacheManager (GPU + CPU tiers),
                           restores on local cache miss,
                           propagates evict() to both stores
```

---

## Quick Start

```bash
# Download a model
# TinyLlama (637 MB, fast on any CPU)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Phi-3.5 Mini (2.4 GB, needs --heap 4g or more)
wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

# Build
mvn clean package -DskipTests

# 3-node pipeline-parallel cluster (default)
./juno --model-path /path/to/model.gguf --heap 4g

# 3-node tensor-parallel cluster
./juno --model-path /path/to/model.gguf --heap 4g --pType tensor

# All layers in one JVM (fastest startup, no network)
./juno local --model-path /path/to/model.gguf --heap 4g

# LoRA fine-tuning REPL (adapter saved to <model>.lora)
./juno lora --model-path /path/to/model.gguf --heap 4g

# Real-model smoke test — 8 checks, exits 0/1
./juno test --model-path /path/to/model.gguf --heap 4g

# Profile with JFR — captures all five juno.* event types
./juno local --model-path /path/to/model.gguf --jfr 5m
# Open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser: juno.MatVec / juno.ForwardPass / juno.Tokenizer / juno.TemplateFormat

# Extract automated metrics summary (counts, durations, p50/p95/p99 per event type)
mvn package -pl productivity -am -DskipTests
java -cp productivity/target/productivity-*.jar cab.ml.juno.productivity.ProductivityMain
# Output: target/productivity/metrics.json
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Path to GGUF file (required) |
| `--pType pipeline\|tensor` | `pipeline` | Parallelism type: `pipeline` = contiguous layer blocks, serial activation flow (vertical scaling); `tensor` = weight-matrix slices, all nodes in parallel with AllReduce (horizontal scaling). Constraint for tensor: `numHeads % nodes == 0` |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | Activation wire format between nodes |
| `--max-tokens N` | `200` | Maximum tokens per response |
| `--temperature F` | `0.6` | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | Nucleus sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | In-process shard count (`local` only) |
| `--jfr DURATION` | — | Java Flight Recording for DURATION (e.g. `30s`, `5m`, `1h`). Writes `juno-<modelStem>-<timestamp>.jfr`. Captures `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, and (for `lora`) `juno.LoraTrainStep` events. Open in JDK Mission Control. Use `./juno local` for full event coverage — cluster mode captures coordinator JVM only. |
| `--verbose` / `-v` | — | Show node startup, gRPC and shard loading logs |

**LoRA flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint file (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α (effective scale = α/rank) |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` command |

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `JAVA_HOME`

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE + GPT-2 BPE; auto-detected), `ChatTemplate`, `SimpleTokenizer`, `TokenizerEvent`, `TemplateFormatEvent` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader`, `EmbeddedNodeServer`, `NodeKVCacheAdapter`, `MatVec`, `CpuMatVec`, `CudaMatVec`, `GgufReader`, `LlamaConfig`, `ActivationCodec`, `ShardContext`, `TensorShardContext`, `LoraAdapter`, `LoraAdapterSet`, `LoraAdamOptimizer`, `LoraTrainableHandler`, `MatVecEvent`, `ForwardPassEvent`, `LoraTrainEvent` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL (`cluster` / `local` / `lora` commands), `ClusterHarness`, `ProcessPipelineClient`, `TensorParallelPipelineClient` |
| `juno-node` | Fat jar (`juno-node.jar`). `NodeMain` — standalone node entry point for remote deployment. Reads `-Dnode.id`, `-Dnode.port`, `-Dmodel.path`, `-DJUNO_USE_GPU` from system properties (command-line args still accepted). Prints `READY:<nodeId>:<port>` on startup. Launched by `juno-node.service` systemd unit on AWS nodes. |
| `juno-master` | Fat jar (`juno-master.jar`; renamed from `integration`). `CoordinatorMain` — standalone coordinator entry point for remote deployment. Reads node addresses from `JUNO_NODE_ADDRESSES`, model from `JUNO_MODEL_PATH`, and tuning from `JUNO_PTYPE` / `JUNO_HTTP_PORT` / `JUNO_DTYPE` / `JUNO_MAX_QUEUE`. No forking, no `ClusterHarness`. Launched by `juno-coordinator.service` on the AWS coordinator host. Integration tests (`InProcessClusterIT`, `ThreeNodeClusterIT`, `TensorParallelClusterIT`, `GpuForwardPassIT`, `ModelLiveRunnerIT`) in package `cab.ml.juno.master`; `ModelLiveRunnerIT` gated behind `-Pintegration` Maven profile. |
| `metrics` | JFR-based productivity metrics extractor (source dir: `productivity/`). `JfrMetricsExtractor`, `JfrModelMapper`, `JfrPercentiles`, `MetricsSnapshot`, `MetricsWriter`, `ModelsConfig`, `ModelsConfigLoader`, `ProductivityMain` |

---

## Supported Models

Any GGUF file with a LLaMA-compatible or Phi-3-compatible architecture. Tested:

| Model | File size | Heap needed |
|-------|-----------|-------------|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | 2g |
| TinyLlama-1.1B-Chat Q2_K | 380 MB | 2g |
| phi-3.5-mini-instruct Q4_K_M | 2.4 GB | 4g |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | 8g |
| Meta-Llama-3.2-1B-Instruct Q8_0 | 1.3 GB | 4g |
| Meta-Llama-3.1-70B-Instruct Q4_K_M | 40 GB | 16 x 4 GB nodes |

**Quantisation types:** F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

**Chat templates:** `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default), `phi3`.
Template resolved from model type string via exact match then substring fallback.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry,player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master             # integration tests — forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT (pipeline) and
                                       # TensorParallelClusterIT (tensor)

mvn verify -pl juno-master -Pintegration -Dmodels=/path/to/models
                                       # ModelLiveRunnerIT — requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test
```

### GPU tests (requires CUDA 12.x and Nvidia GPU)

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl juno-master \
  --enable-native-access=ALL-UNNAMED
```

---

## JFR Profiling

All `juno` commands accept `--jfr DURATION` which activates Java Flight Recording and writes a `juno-<timestamp>.jfr` file on exit. Open it in JDK Mission Control. Five custom event types are available under Event Browser:

| Event | Category | Key fields | Fired by |
|-------|----------|------------|----------|
| `juno.MatVec` | Juno/MatVec | `backend`, `rows`, `cols` | `CpuMatVec`, `CudaMatVec` (both overloads) |
| `juno.ForwardPass` | Juno/Inference | `handlerType`, `requestId`, `startPosition`, `layerCount`, `hasOutputProjection` | All 6 `ForwardPassHandler` implementations |
| `juno.Tokenizer` | Juno/Tokenizer | `tokenizerType`, `operation`, `inputLength`, `outputLength` | `GgufTokenizer`, `DJLTokenizer`, `SimpleTokenizer` |
| `juno.TemplateFormat` | Juno/Tokenizer | `modelType`, `messageCount`, `outputLength` | `ChatTemplateFormatter` |
| `juno.LoraTrainStep` | Juno/LoRA | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` | `LoraTrainableHandler.trainStep()` |

Useful analysis patterns:

- Sort `juno.MatVec` by duration descending to find the most expensive multiply shape (typically the output projection at `32000 × 2048` for TinyLlama).
- Filter `juno.MatVec` by `backend = "cuda-resident"` vs `"cuda"` to isolate the per-call H2D transfer cost from kernel time.
- Filter `juno.ForwardPass` by `startPosition = 0` to measure prefill latency separately from decode-step latency.
- Aggregate `juno.Tokenizer` for `operation = "decodeToken"` to see streaming-decode overhead relative to total generation time.
- Group `juno.TemplateFormat` by `modelType` to compare template cost across model families.

---

## CPU Performance

| Session | Change | ms / 10 tokens |
|---------|--------|----------------|
| 5 | Baseline (FLOAT32, serial matVec) | ~34,891 ms |
| 6 | Parallel matVec + FLOAT16 default | ~3,802 ms (9x) |
| 9 | Session KV cache — turn latency flat | ~7,000-8,000 ms / turn |
| 10 | CudaMatVec (cublasSgemv) | AWS benchmark pending |
| 11 | Phi-3.5-mini on 3-node CPU cluster | ~38,420 ms / turn |
| 13 | Tensor-parallel mode (pType=tensor) | same throughput per node; parallel decode step |

---

## Key Design Decisions

- No Python, no llama.cpp subprocess. JVM reads GGUF binary directly and runs the transformer end to end.
- No Spring Boot. Javalin for REST.
- Two parallelism modes: `pipeline` (LAN-friendly, sequential activation flow, vertical scaling) and `tensor` (parallel per-step AllReduce, horizontal scaling, higher throughput). Selected at startup with `--pType`.
- **Lazy dequantization on CPU; eager upload on GPU.** Projection weights are kept as raw quantized bytes in `GgufReader.QuantizedTensor`. On the CPU path, dequantization runs one 256-element block at a time inside the matmul loop (peak live float footprint ~1 kB instead of ~65 MB). On the GPU path (`CudaMatVec`), weights are dequantized to `float[]` once at load time and uploaded to `DeviceFloatMatrix` on the GPU; forward passes then copy only `x` and `y` across the bus, not the weight matrix.
- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else.
- **KV cache wired to the node-level manager.** `NodeKVCacheAdapter` connects `LlamaTransformerHandler` and `Phi3TransformerHandler` to the `KVCacheManager` (GPU byte-budget LRU + Caffeine W-TinyLFU CPU tier). Every forward pass flushes key/value data write-through into both tiers. If a local entry is evicted under heap pressure, the next forward pass at that position restores it from the manager transparently. `evict(requestId)` propagates to both the local HashMap and both cache tiers, closing the gap that previously made the entire `kvcache` module inert at the node level.
- **Star AllReduce for tensor parallel.** No inter-node communication. The coordinator collects partial logit vectors from all N nodes and sums them in O(N × vocabSize). Simpler than ring-AllReduce and requires no InfiniBand.
- **LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps `LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4–16) on the Q and V projections. The frozen weights stay quantized at all times — backward passes dequantize one block per row via dedicated `transposedQ4K` / `transposedQ5K` / `transposedQ6K` scatter-reduce implementations. Adapters are persisted to a `.lora` binary checkpoint; the GGUF is never modified.
- **GPT-2 BPE and SentencePiece BPE both supported.** `GgufTokenizer` reads `tokenizer.ggml.model` from GGUF metadata. Value `"gpt2"` activates the GPT-2 / tiktoken path (Llama 3+): space-prefixed words use Ġ (U+0120), and special control tokens are pre-split longest-first before BPE. Any other value (null / `"llama"` / `"llama2"`) uses the SentencePiece path (Llama 1/2, TinyLlama, Mistral, Gemma, Phi-3). No configuration required — detection is automatic at load time.
- **AWS infrastructure scripted.** `scripts/aws/launcher.sh` is a credential wrapper. `juno-deploy.sh` is a unified cluster lifecycle script (replaces the earlier `juno-infra.sh` / `juno-infra-ft.sh`): hardware is auto-detected during bootstrap — GPU nodes install CUDA and set `JUNO_USE_GPU=true`, CPU nodes skip it. Commands: `setup | start | stop | teardown | status | scan-regions`. Coordinator can be co-located on node 1 (default, free) or launched as a separate t3.medium. State persisted to `~/.juno-deploy-state`; Ctrl+C auto-stops instances before exit. After `setup` the script bootstraps all nodes (~5 min), writes `cluster-nodes.env`, starts the coordinator, and enters the live cluster monitor showing per-node CPU / mem, health, and estimated cost. Web console served at `http://<coordinator>:8080` once cluster is healthy.
- **JFR metrics extractor.** The `metrics` module (source dir: `productivity/`) scans the project root for `juno-<stem>-*.jfr` files, maps them to `models.json` entries, extracts counts / durations / p50/p95/p99 percentiles for all five `juno.*` event types, and writes `target/productivity/metrics.json`.
- **Full JFR instrumentation across every hot path.** Five custom event types — `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, `juno.LoraTrainStep` — make every layer of the stack observable in JDK Mission Control without any agent or bytecode manipulation. Activated with `--jfr DURATION` on any `juno` command.
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- `MatVec` interface decouples the matmul backend from the transformer logic. `CudaMatVec` implements host `sgemv` (full H2D per call, for tests) and device `sgemv(DeviceFloatMatrix, x)` for resident weights (production path). `CpuMatVec` as CPU fallback and test reference. Backend selection is automatic via `ForwardPassHandlerLoader.selectBackend()` which reads `JUNO_USE_GPU` and calls `CudaAvailability.isAvailable()`. When `CudaMatVec` is selected, `LlamaTransformerHandler` dequantizes all projection weights to `float[]` and uploads them as `DeviceFloatMatrix` once at construction; the new `matVecLayer()` method dispatches through the GPU-resident path on each forward call.
- GPU tests excluded from default CI by failsafe `<excludes>` and a `-Pgpu` profile. `GpuForwardPassIT` additionally guards with `-Djuno.gpu.test=true` to prevent CUDA native libs (bytedeco) loading into the coordinator JVM and poisoning FD inheritance into forked node processes.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub. Integration tests live in `juno-master` (package `cab.ml.juno.master`); `ModelLiveRunnerIT` requires a real model and is gated behind `-Pintegration`.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA / NVIDIA driver (GPU nodes only — not required for CPU mode or any unit/integration tests); Java bindings via org.bytedeco cuda-platform

---

## License

Apache 2.0