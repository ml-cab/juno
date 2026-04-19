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
              | gRPC activations (FLOAT16 / INT8 / FLOAT32, BE or LE wire order)
              | serial: node-1 -> node-2 -> node-3
              |
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3  ...      |
    |  L 0-7        L 8-14       L 15-21          |
    |  + embed                   + output proj    |
    |  NodeKVCacheAdapter wired into each handler |
    |  LoraAdapterSet (optional, read-only)       |
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

LoRA overlay (optional):
    load(..., LoraAdapterSet)  <- wraps base handler in LoraTrainableHandler
                                  adapters applied read-only during inference
                                  base GGUF is never modified

MatVec (injected into handler):
    CpuMatVec    <- parallel IntStream
    CudaMatVec   <- cublasSgemv_v2 (FP32 host path) / resident FP32 or FP16 weights:
                    Llama + Phi-3 GPU use DeviceHalfMatrix + cublasHSSgemvStridedBatched;
                    per-thread CUDA stream + async H2D/D2H around GEMV; GpuContext.shared(dev);
                    weights uploaded once at load time; releaseGpuResources() frees VRAM on unload

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

# Apply a saved .lora adapter at inference — local mode
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Apply a saved .lora adapter at inference — cluster mode (forked JVMs)
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Merge LoRA adapter into a standalone GGUF (no .lora sidecar needed at runtime)
# Patched tensors are stored as F32 to preserve precision; output is ~1.5× larger.
./juno merge --model-path /path/to/model.gguf --heap 4g

# Real-model smoke test — 8 checks, exits 0/1
./juno test --model-path /path/to/model.gguf --heap 4g

# Apply a saved .lora adapter at inference — cluster mode (forked JVMs)
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Profile with JFR
./juno local --model-path /path/to/model.gguf --jfr 5m
# When the 5-minute period expires, metrics JSON is printed inline and written to
# target/metrics/metrics.json — no manual step needed.
# Open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control for detailed inspection.
# Event Browser: juno.MatVec / juno.ForwardPass / juno.Tokenizer / juno.TemplateFormat

# Post-hoc extraction from any .jfr file (e.g. from lora or test runs):
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Path to GGUF file (required) |
| `--pType pipeline\|tensor` | `pipeline` | Parallelism type: `pipeline` = contiguous layer blocks, serial activation flow (vertical scaling); `tensor` = weight-matrix slices, all nodes in parallel with AllReduce (horizontal scaling). Constraint for tensor: `numHeads % nodes == 0` |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | Activation wire format between nodes |
| `--byteOrder BE\|LE` | `BE` | Activation byte order on the gRPC wire. `BE` = big-endian (default, validated on production hardware). `LE` = little-endian (native x86 order, zero unaligned-read overhead). Must match across all JVMs in the cluster — propagated automatically by `ClusterHarness` and `juno-deploy.sh`. |
| `--max-tokens N` | `200` | Maximum tokens per response |
| `--temperature F` | `0.6` | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | Nucleus sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | In-process shard count (`local` only) |
| `--jfr DURATION` | — | Java Flight Recording for DURATION (e.g. `30s`, `5m`, `1h`). Writes `juno-<modelStem>-<timestamp>.jfr`. Captures `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, and (for `lora`) `juno.LoraTrainStep` events. **`local` mode**: recording managed programmatically — metrics JSON is printed automatically when the period expires or the REPL exits. **`cluster` mode**: coordinator gets programmatic JFR; every forked node JVM is also instrumented so `juno.MatVec`/`juno.ForwardPass` events are captured from all nodes; all files are merged and metrics auto-printed on exit. **`lora`/`test` modes**: JVM flag only; open the resulting `.jfr` in JDK Mission Control or run `MetricsMain` manually. |
| `--verbose` / `-v` | — | Show node startup, gRPC and shard loading logs |
| `--cpu` | — | Force CPU inference: sets `JUNO_USE_GPU=false` (cluster / local). Does not enable LoRA mode. |
  `--lora-play PATH` | — | Apply a pre-trained `.lora` adapter at inference in `local` or `cluster` mode. Adapters are read-only — no training. The base GGUF is never modified. In cluster mode the adapter file is forwarded to every forked node JVM via `-Djuno.lora.play.path`. |

**LoRA flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint file (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α (effective scale = α/rank) |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` command |
| `--lora-steps-qa N` | `10` | Gradient steps per `/train-qa` Q&A pair |
| `--lora-early-stop F` | `0.25` | Stop early when loss delta < F |

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `BYTE_ORDER`, 
`MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, 
`LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, 
`LORA_STEPS`, `LORA_PLAY_PATH`, `JAVA_HOME`

**`merge` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output file (always plain GGUF, even if source is llamafile) |
| `--heap SIZE` | `4g` | JVM heap — use at least 2× the model file size |

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `BYTE_ORDER`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `JAVA_HOME`

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE + GPT-2 BPE; auto-detected), `ChatTemplate`, `SimpleTokenizer`, `TokenizerEvent`, `TemplateFormatEvent` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader`, `EmbeddedNodeServer`, `NodeMain`, `NodeKVCacheAdapter`, `MatVec`, `CpuMatVec`, `CudaMatVec`, `GpuContext`, `DeviceFloatMatrix`, `DeviceHalfMatrix`, `GgufReader`, `LlamaConfig`, `ActivationCodec`, `ActivationBECodec`, `ActivationLECodec`, `ShardContext`, `TensorShardContext`, `LoraAdapter`, `LoraAdapterSet`, `LoraAdamOptimizer`, `LoraTrainableHandler`, `MatVecEvent`, `ForwardPassEvent`, `LoraTrainEvent` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL (`cluster` / `local` / `lora` / `merge` commands), `ClusterHarness`, `ProcessPipelineClient`, `TensorParallelPipelineClient`, `LoraMergeMain` |
| `juno-node` | Fat jar (`juno-node.jar`). Entry point: `NodeMain` (`cab.ml.juno.node`). Reads `-Dnode.id`, `-Dnode.port`, `-Dmodel.path`, `-DJUNO_USE_GPU`, `-Djuno.byteOrder` from system properties (command-line args still accepted). Prints `READY:<nodeId>:<port>` on startup. Launched by `juno-node.service` systemd unit on AWS nodes. |
| `juno-master` | Fat jar (`juno-master.jar`). Entry point: `CoordinatorMain` (`cab.ml.juno.master`) — standalone coordinator for remote deployment. Reads node addresses from `JUNO_NODE_ADDRESSES`, model from `JUNO_MODEL_PATH`, and tuning from `JUNO_PTYPE` / `JUNO_HTTP_PORT` / `JUNO_DTYPE` / `JUNO_MAX_QUEUE` / `JUNO_BYTE_ORDER`. No forking, no `ClusterHarness`. Launched by `juno-coordinator.service` on the AWS coordinator host. Integration tests (`InProcessClusterIT`, `ThreeNodeClusterIT`, `TensorParallelClusterIT`, `GpuForwardPassIT`, `ModelLiveRunnerIT`) in package `cab.ml.juno.master`; `ModelLiveRunnerIT` gated behind `-Pintegration` Maven profile. |
| `metrics` | JFR-based metrics extractor. `JfrMetricsExtractor` (single-file and multi-file merge), `JfrModelMapper`, `JfrPercentiles`, `MetricsSnapshot`, `MetricsWriter`, `ModelsConfig`, `ModelsConfigLoader`, `MetricsMain` (standalone entry point + `extractToJson`/`extractToJsonMerged` facades for programmatic use) |

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

## Key Design Decisions

- **LoRA inference overlay (`--lora-play`).** Pre-trained `.lora` adapter files can be applied at inference time in any mode — `local`, `cluster` (forked JVMs), or AWS-deployed clusters — without entering the training REPL. `ForwardPassHandlerLoader.load(path, ctx, backend, adapters)` wraps the base handler in `LoraTrainableHandler` transparently; passing `null` gives the standard base-model path. The base GGUF is never modified. `ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path` into every forked node JVM so each node reads the same adapter file. `NodeMain` forwards `JUNO_LORA_PLAY_PATH` from the environment as a system property for AWS/remote deployments.

- **LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps `LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4–16) on the Q and V projections. The frozen weights stay quantized at all times. Adapters are persisted to a `.lora` binary checkpoint; the GGUF is never modified.

- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else. When a `LoraAdapterSet` is provided, the selected handler is wrapped in `LoraTrainableHandler`. Backend selection via `selectBackend()`.

- **AWS infrastructure scripted.** `juno-deploy.sh` is the unified cluster lifecycle script. `--lora-play PATH` accepts an absolute or relative path; it is resolved to absolute via `realpath` at parse time (relative paths are ambiguous when the script is called from a different directory). After bootstrap, `_scp_lora_to_nodes()` stops each `juno-node.service`, SCPs the file to `/opt/juno/models/`, patches `JUNO_LORA_PLAY_PATH` in `/etc/juno/node.env`, and restarts the service — all synchronously per node. Only after all nodes are confirmed active does the coordinator start, ensuring `loadShard` RPCs always find nodes with adapters loaded.

- User-data scripts are passed to `aws ec2 run-instances` via `file://` (not pre-encoded base64), preventing double-encoding that caused cloud-init to reject the script as `text/x-not-multipart`.


## JFR Profiling

All `juno` commands accept `--jfr DURATION` which activates Java Flight Recording and writes a `juno-<modelStem>-<timestamp>.jfr` file. Five custom event types are available under the JDK Mission Control Event Browser:

| Event | Category | Key fields | Fired by |
|-------|----------|------------|----------|
| `juno.MatVec` | Juno/MatVec | `backend`, `rows`, `cols` | `CpuMatVec`, `CudaMatVec` (host and `DeviceFloatMatrix` / `DeviceHalfMatrix` paths), quantized static path in `LlamaTransformerHandler` |
| `juno.ForwardPass` | Juno/Inference | `handlerType`, `requestId`, `startPosition`, `layerCount`, `hasOutputProjection` | All `ForwardPassHandler` implementations |
| `juno.Tokenizer` | Juno/Tokenizer | `tokenizerType`, `operation`, `inputLength`, `outputLength` | `GgufTokenizer`, `DJLTokenizer`, `SimpleTokenizer` |
| `juno.TemplateFormat` | Juno/Tokenizer | `modelType`, `messageCount`, `outputLength` | `ChatTemplateFormatter` |
| `juno.LoraTrainStep` | Juno/LoRA | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` | `LoraTrainableHandler.trainStep()` |

**Per-mode behaviour:**

- **`local` mode** (`./juno local --jfr DURATION`): JFR is managed programmatically via `jdk.jfr.Recording`. When the recording period expires, `MetricsMain.extractToJson()` is called from a shutdown hook — the metrics JSON is printed inline to the REPL console and written to `target/metrics/metrics.json`. The REPL stays open after extraction. All five event types are captured in the single JVM.
- **`cluster` mode** (`./juno --jfr DURATION`): The coordinator JVM gets a programmatic `jdk.jfr.Recording`. Every forked node JVM is also instrumented via `ClusterHarness.withJfr()` which injects `-XX:StartFlightRecording=...,dumponexit=true` so each node's `.jfr` file is written on process exit. On shutdown, `MetricsMain.extractToJsonMerged()` merges coordinator + all node files into one snapshot before printing. Full event coverage: `juno.MatVec` and `juno.ForwardPass` come from node JVMs; `juno.Tokenizer` and `juno.TemplateFormat` from the coordinator.
- **`lora` / `test` modes**: JVM flag only (`-XX:StartFlightRecording`). Open the resulting `.jfr` in JDK Mission Control or run `MetricsMain` manually.

Useful analysis patterns:

- Sort `juno.MatVec` by duration descending to find the most expensive multiply shape (typically the output projection at `32000 × 2048` for TinyLlama).
- Filter `juno.MatVec` by `backend = "cuda-resident"` vs `"cuda-resident-fp16"` vs `"cuda"` to compare FP32-resident matrices (tests / `DeviceFloatMatrix`), FP16-resident (Llama + Phi GPU), and full host-matrix upload paths.
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
- **Lazy dequantization on CPU; eager upload on GPU.** Projection weights are kept as raw quantized bytes in `GgufReader.QuantizedTensor`. On the CPU path, dequantization runs one 256-element block at a time inside the matmul loop (peak live float footprint ~1 kB instead of ~65 MB). On the GPU path (`CudaMatVec`), **Llama** and **Phi-3** dequantize once and upload to **`DeviceHalfMatrix`** (FP16 on device, ~half the VRAM of FP32 storage); matmul uses **`cublasHSSgemvStridedBatched`**. If `cudaMalloc` fails during upload, **both** handlers close partial GPU buffers and fall back to CPU quantised matmul for those projections. Forward passes copy activations and small FP16 `x` staging plus `y`, not the full weight matrix.
- **Explicit GPU weight lifecycle.** `ForwardPassHandler.releaseGpuResources()` closes all `DeviceHalfMatrix` / `DeviceFloatMatrix` buffers owned by a handler. `EmbeddedNodeServer` calls it on shard unload/reload and before swapping handlers so VRAM is freed without waiting for GC.
- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else. Backend selection is automatic via `selectBackend()` which reads `JUNO_USE_GPU`, `CudaAvailability`, and `-Djuno.cuda.device` (defaults to `0`).
- **Configurable activation byte order.** `ActivationCodec` is a zero-overhead static dispatcher: it reads `juno.byteOrder` once at class-load time and branches to either `ActivationBECodec` (big-endian, default) or `ActivationLECodec` (little-endian, native x86 order) for all encode/decode calls. The byte order is propagated consistently across every JVM in a cluster — `ClusterHarness` injects `-Djuno.byteOrder` into every forked node process; `juno-deploy.sh` writes it into `/etc/juno/node.env` for systemd-managed nodes. The cluster health endpoint reports the active byte order; the web console displays a live badge.
- **KV cache wired to the node-level manager.** `NodeKVCacheAdapter` connects `LlamaTransformerHandler` and `Phi3TransformerHandler` to the `KVCacheManager` (GPU byte-budget LRU + Caffeine W-TinyLFU CPU tier). Every forward pass flushes key/value data write-through into both tiers. If a local entry is evicted under heap pressure, the next forward pass at that position restores it from the manager transparently. `evict(requestId)` propagates to both the local HashMap and both cache tiers, closing the gap that previously made the entire `kvcache` module inert at the node level.
- **Star AllReduce for tensor parallel.** No inter-node communication. The coordinator collects partial logit vectors from all N nodes and sums them in O(N × vocabSize). Simpler than ring-AllReduce and requires no InfiniBand.
- **LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps `LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4–16) on the Q and V projections. The frozen weights stay quantized at all times — backward passes dequantize one block per row via dedicated `transposedQ4K` / `transposedQ5K` / `transposedQ6K` scatter-reduce implementations. Adapters are persisted to a `.lora` binary checkpoint; the GGUF is never modified. To produce a standalone merged model, use `./juno merge` — see **Merge** below.
- **Native LoRA merge (`juno merge`).** `LoraMerge` writes a new GGUF where the 44 LoRA-patched projection tensors (wq/wv on every layer) are stored as **F32** instead of being re-quantised. This is essential: the typical LoRA delta (~6×10⁻⁴ per element) is smaller than Q4_K quantisation noise (~3×10⁻³), so re-quantising would erase all training. The F32 patched tensors are read by `GgufReader` and `LlamaTransformerHandler` identically to any other F32 tensor — no special-casing in inference. All other tensors (norms, FFN, embeddings, output projection) are copied verbatim in their original quantised form. The merged file is larger than the source (~1 GB for TinyLlama 1.1B Q4_K_M vs 667 MB) but requires no `.lora` sidecar at inference time.
- **GPT-2 BPE and SentencePiece BPE both supported.** `GgufTokenizer` reads `tokenizer.ggml.model` from GGUF metadata. Value `"gpt2"` activates the GPT-2 / tiktoken path (Llama 3+): space-prefixed words use Ġ (U+0120), and special control tokens are pre-split longest-first before BPE. Any other value (null / `"llama"` / `"llama2"`) uses the SentencePiece path (Llama 1/2, TinyLlama, Mistral, Gemma, Phi-3). No configuration required — detection is automatic at load time.
- **AWS infrastructure scripted.** `scripts/aws/launcher.sh` is a credential wrapper. `juno-deploy.sh` is a unified cluster lifecycle script (replaces the earlier `juno-infra.sh` / `juno-infra-ft.sh`): hardware is auto-detected during bootstrap — GPU nodes install CUDA and set `JUNO_USE_GPU=true`, CPU nodes skip it. Commands: `setup | start | stop | teardown | status | scan-regions`. Coordinator can be co-located on node 1 (default, free) or launched as a separate t3.medium. State persisted to `~/.juno-deploy-state`; Ctrl+C auto-stops instances before exit. After `setup` the script bootstraps all nodes (~5 min), writes `cluster-nodes.env`, starts the coordinator, and enters the live cluster monitor showing per-node CPU / mem, health, and estimated cost. Web console served at `http://<coordinator>:8080` once cluster is healthy.
- **JFR metrics extractor.** The `metrics` module scans the project root for `juno-<stem>-*.jfr` files (the optional `juno-` prefix is also accepted for AWS node recordings), maps them to `models.json` entries, extracts counts / durations / p50/p95/p99 percentiles for all five `juno.*` event types, and writes `target/metrics/metrics.json`. For `local` and `cluster` modes this runs automatically on JFR period expiry or exit via `MetricsMain.extractToJson()` / `extractToJsonMerged()`; for `lora`/`test` runs it can be invoked manually with `java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain`.
- **Full JFR instrumentation across every hot path.** Five custom event types — `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, `juno.LoraTrainStep` — make every layer of the stack observable in JDK Mission Control without any agent or bytecode manipulation. Activated with `--jfr DURATION` on any `juno` command. In `local` mode all events appear in one file; in `cluster` mode the coordinator and every forked node JVM each write their own `.jfr` file, which are then merged automatically by `MetricsMain.extractToJsonMerged()` on exit for a complete cross-JVM view.
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- `MatVec` interface decouples the matmul backend from the transformer logic. `CudaMatVec` implements host `sgemv` (full H2D per call, for tests), **`sgemv(DeviceFloatMatrix, x)`** (FP32 resident, tests / legacy), and **`sgemv(DeviceHalfMatrix, x)`** for Llama + Phi GPU (FP16 weights, async copies + per-thread stream). `CpuMatVec` as CPU fallback and test reference.
- GPU tests excluded from default CI by failsafe `<excludes>` and a `-Pgpu` profile. `GpuForwardPassIT` additionally guards with `-Djuno.gpu.test=true` to prevent CUDA native libs (bytedeco) loading into the coordinator JVM and poisoning FD inheritance into forked node processes.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub. Before a real shard is loaded, `EmbeddedNodeServer` uses an internal `StubForwardPassHandler` (zero-filled arrays, no test machinery). The test-only `CyclicForwardPassHandler` (deterministic fixed-pattern output, configurable winner token) lives in `node/src/test` and is shared with other modules via the `node:tests` classifier jar. Integration tests live in `juno-master` (package `cab.ml.juno.master`); `ModelLiveRunnerIT` requires a real model and is gated behind `-Pintegration`.

--

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA / NVIDIA driver (GPU nodes only — not required for CPU mode or any unit/integration tests); Java bindings via org.bytedeco cuda-platform

---

## License

Apache 2.0
