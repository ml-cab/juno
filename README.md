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
    CudaMatVec   <- cublasSgemv_v2 via JCublas2
```

---

## Quick Start

```bash
# Download a model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

# Build
mvn clean package -DskipTests

# 3-node pipeline-parallel cluster (default)
./juno --model-path /path/to/model.gguf --heap 4g

# All layers in one JVM (fastest startup, no network)
./juno local --model-path /path/to/model.gguf --heap 4g

# LoRA fine-tuning REPL (adapter saved to <model>.lora)
./juno lora --model-path /path/to/model.gguf --heap 4g

# Apply a saved .lora adapter at inference — local mode
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Apply a saved .lora adapter at inference — cluster mode (forked JVMs)
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Profile with JFR
./juno local --model-path /path/to/model.gguf --jfr 5m
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Path to GGUF file (required) |
| `--pType pipeline\|tensor` | `pipeline` | Parallelism type |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | Activation wire format between nodes |
| `--byteOrder BE\|LE` | `BE` | Activation byte order on the gRPC wire |
| `--max-tokens N` | `200` | Maximum tokens per response |
| `--temperature F` | `0.6` | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | Nucleus sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | In-process shard count (`local` only) |
| `--jfr DURATION` | — | Java Flight Recording for DURATION |
| `--verbose` / `-v` | — | Show node startup, gRPC and shard loading logs |
| `--lora-play PATH` | — | Apply a pre-trained `.lora` adapter at inference in `local` or `cluster` mode. Adapters are read-only — no training. The base GGUF is never modified. In cluster mode the adapter file is forwarded to every forked node JVM via `-Djuno.lora.play.path`. |

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

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `BYTE_ORDER`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `LORA_PLAY_PATH`, `JAVA_HOME`

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer`, `ChatTemplate`, `ChatTemplateFormatter`, `TokenizerEvent`, `TemplateFormatEvent` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader` (now with `LoraAdapterSet` overload), `EmbeddedNodeServer`, `NodeMain`, `NodeKVCacheAdapter`, `MatVec`, `CpuMatVec`, `CudaMatVec`, `GgufReader`, `LlamaConfig`, `ActivationCodec`, `ActivationBECodec`, `ActivationLECodec`, `ShardContext`, `TensorShardContext`, `LoraAdapter`, `LoraAdapterSet`, `LoraAdamOptimizer`, `LoraTrainableHandler`, `MatVecEvent`, `ForwardPassEvent`, `LoraTrainEvent` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL (`cluster` / `local` / `lora` commands), `ClusterHarness` (now with `withLoraPlay()`), `ProcessPipelineClient`, `TensorParallelPipelineClient` |
| `juno-node` | Fat jar (`juno-node.jar`). Entry point: `NodeMain`. Reads `JUNO_LORA_PLAY_PATH` from env and forwards as `-Djuno.lora.play.path` system property so `EmbeddedNodeServer` can load adapters at inference time. |
| `juno-master` | Fat jar (`juno-master.jar`). Entry point: `CoordinatorMain`. Reads `JUNO_LORA_PLAY_PATH` and logs it for diagnostics; the value propagates to nodes via `cluster-nodes.env`. |
| `metrics` | JFR-based metrics extractor. `JfrMetricsExtractor`, `MetricsMain` |

---

## Key Design Decisions

- **LoRA inference overlay (`--lora-play`).** Pre-trained `.lora` adapter files can be applied at inference time in any mode — `local`, `cluster` (forked JVMs), or AWS-deployed clusters — without entering the training REPL. `ForwardPassHandlerLoader.load(path, ctx, backend, adapters)` wraps the base handler in `LoraTrainableHandler` transparently; passing `null` gives the standard base-model path. The base GGUF is never modified. `ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path` into every forked node JVM so each node reads the same adapter file. `NodeMain` forwards `JUNO_LORA_PLAY_PATH` from the environment as a system property for AWS/remote deployments.

- **LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps `LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4–16) on the Q and V projections. The frozen weights stay quantized at all times. Adapters are persisted to a `.lora` binary checkpoint; the GGUF is never modified.

- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else. When a `LoraAdapterSet` is provided, the selected handler is wrapped in `LoraTrainableHandler`. Backend selection via `selectBackend()`.

- **AWS infrastructure scripted.** `juno-deploy.sh` is the unified cluster lifecycle script. `--lora-play PATH` accepts an absolute or relative path; it is resolved to absolute via `realpath` at parse time (relative paths are ambiguous when the script is called from a different directory). After bootstrap, `_scp_lora_to_nodes()` stops each `juno-node.service`, SCPs the file to `/opt/juno/models/`, patches `JUNO_LORA_PLAY_PATH` in `/etc/juno/node.env`, and restarts the service — all synchronously per node. Only after all nodes are confirmed active does the coordinator start, ensuring `loadShard` RPCs always find nodes with adapters loaded.

- User-data scripts are passed to `aws ec2 run-instances` via `file://` (not pre-encoded base64), preventing double-encoding that caused cloud-init to reject the script as `text/x-not-multipart`.

- All other existing design decisions unchanged (see previous sessions).

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA / NVIDIA driver (GPU nodes only)

---

## License

Apache 2.0	