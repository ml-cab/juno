# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 x 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Status

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster

**Session 17** — `--dequant eager|lazy` flag. `WeightDequantMode` enum propagates through the full stack: run scripts → ConsoleMain → ClusterHarness (forked JVMs via `-DJUNO_DEQUANT`) → NodeMain → EmbeddedNodeServer → ForwardPassHandlerLoader → LlamaTransformerHandler. In `eager` mode (default) `GpuWeightShard` dequantizes all projection weights once and pins them on the GPU; every decode step calls `CudaMatVec.sgemv(DeviceFloatMatrix, x)` with no per-token weight copy. In `lazy` mode weights stay as `QuantizedTensor` bytes in JVM heap and are dequantized one block at a time on the CPU, reducing per-node VRAM from ~2.4 GB to ~138 MB (KV cache only) for TinyLlama-1.1B.

**Session 16** — naming cleanup: session-12 rename fully applied to source.

The `KVCacheManager` (GPU + CPU tiers with LRU/W-TinyLFU eviction) was previously disconnected from the transformer handlers: `LlamaTransformerHandler` and `Phi3TransformerHandler` each maintained their own private `HashMap<String, float[][]>` with no eviction, making the entire `kvcache` module inert at the node level. This is now fixed.

`NodeKVCacheAdapter` bridges the in-process KV arrays and the cluster-level `KVCacheManager`. After each token position is written, the adapter serialises K and V data into a `KVBlock` and flushes it write-through into the manager's GPU tier (byte-budget LRU) and CPU tier (Caffeine W-TinyLFU). If a local HashMap entry is absent at position > 0 (evicted under JVM heap pressure), the adapter restores it from whichever tier still holds the block. `EmbeddedNodeServer.loadShard()` creates the adapter and wires it into every real handler via `setKvAdapter()`. `evict(requestId)` now propagates through both the local map and all manager tiers.

Four custom JFR event classes cover every hot path. All are readable in JDK Mission Control under Event Browser. Use `--jfr DURATION` on any `juno` command to capture a recording:

- `juno.MatVec` — emitted by `CpuMatVec.sgemv()` and both `CudaMatVec.sgemv()` overloads. Fields: `backend` (`cpu`/`cuda`/`cuda-resident`), `rows`, `cols`. ~155 events per generated token for TinyLlama.
- `juno.ForwardPass` — emitted by all six `ForwardPassHandler.forward()` implementations. Fields: `handlerType` (`llama`/`phi3`/`cpu`/`gpu`/`cyclic`/`lora`), `requestId`, `startPosition`, `layerCount`, `hasOutputProjection`.
- `juno.Tokenizer` — emitted by `GgufTokenizer`, `DJLTokenizer`, `SimpleTokenizer` for `encode`, `decode`, and `decodeToken`. Fields: `tokenizerType`, `operation`, `inputLength`, `outputLength`.
- `juno.TemplateFormat` — emitted by `ChatTemplateFormatter.format()`. Fields: `modelType`, `messageCount`, `outputLength`.

**Session 14** — LoRA fine-tuning + JFR profiling. `LoraTrainableHandler` implements parameter-efficient fine-tuning (LoRA) on top of frozen quantised weights. Adapters live in a separate `.lora` file — the base GGUF is never modified. `LoraAdapterSet` / `LoraAdamOptimizer` handle checkpoint I/O and gradient updates. `LoraTrainEvent` emits custom JFR events (`juno.LoraTrainStep`) with per-step timing breakdown (forward / backward / optimizer ms). `ConsoleMain` gains a `lora` subcommand with `/train`, `/train-file`, `/save`, `/status`, `/reset`, `/merge-hint` REPL commands. `--jfr DURATION` flag added to all three `run.sh` / `run.bat` commands (`cluster`, `local`, `lora`). Root bug fixed: `transposedMatVec` now covers Q5_K (type=13) and Q6_K (type=14) — without this, the output projection backward for TinyLlama Q4_K_M fell into an O(cols) loop that took hours per step.

```
you > /train My name is Dima. I am a Java engineer.
  Training  rank=8 · lr=1.0E-4 · 50 steps · 1 chunk(s) · 10 tokens
  step  50/50   loss=3.12  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%  2100ms/step  ETA 0s
  ✔ done  loss=▼ 3.12 (−3.50)  105s total

you*> /save
  ✔ Saved → /path/to/model.lora  (44 adapters · 4401 KB · 50 steps trained)
```

```
you> are you alive?
bot> Yes, I'm here and ready to help! What do you need?
     [19 tokens · 38420 ms · FLOAT16]   <- phi-3.5-mini, 3-node CPU cluster

you> hello
bot> Hey! Nice to meet you too.
     [6 tokens · 2922 ms · FLOAT16]     <- TinyLlama, 3-node CPU cluster
```

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

# Low-VRAM GPU mode: weights stay quantized on CPU, only KV cache on GPU
./juno --model-path /path/to/model.gguf --heap 4g --gpu --dequant lazy

# Full GPU performance: weights dequantized once and pinned on device (default)
./juno --model-path /path/to/model.gguf --heap 4g --gpu --dequant eager

# LoRA fine-tuning REPL (adapter saved to <model>.lora)
./juno lora --model-path /path/to/model.gguf --heap 4g

# Real-model smoke test — 8 checks, exits 0/1
./juno test --model-path /path/to/model.gguf --heap 4g

# Profile with JFR — captures all five juno.* event types
./juno local --model-path /path/to/model.gguf --jfr 5m
# Open juno-<timestamp>.jfr in JDK Mission Control
# Event Browser: juno.MatVec / juno.ForwardPass / juno.Tokenizer / juno.TemplateFormat
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
| `--jfr DURATION` | — | Java Flight Recording for DURATION (e.g. `30s`, `5m`, `1h`). Writes `juno-<timestamp>.jfr`. Captures `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, and (for `lora`) `juno.LoraTrainStep` events. Open in JDK Mission Control. |
| `--verbose` / `-v` | — | Show node startup, gRPC and shard loading logs |
| `--dequant eager\|lazy` | `eager` | Weight dequantization strategy. `eager` = dequantize all projection weights once at load time and upload to GPU; low latency, higher VRAM (~2–4x compressed size per shard). `lazy` = keep weights quantized in JVM heap, dequantize one block at a time on CPU each decode step; minimal VRAM, higher CPU load. Set via `JUNO_DEQUANT` env var. |

**LoRA flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint file (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α (effective scale = α/rank) |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` command |

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JUNO_DEQUANT`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `JAVA_HOME`

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE), `ChatTemplate`, `SimpleTokenizer`, `TokenizerEvent`, `TemplateFormatEvent` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader`, `NodeKVCacheAdapter`, `MatVec`, `CpuMatVec`, `CudaMatVec`, `GpuWeightShard`, `DeviceFloatMatrix`, `WeightDequantMode`, `GgufReader`, `LlamaConfig`, `ActivationCodec`, `ShardContext`, `TensorShardContext`, `LoraAdapter`, `LoraAdapterSet`, `LoraAdamOptimizer`, `LoraTrainableHandler`, `MatVecEvent`, `ForwardPassEvent`, `LoraTrainEvent` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL (`cluster` / `local` / `lora` commands), `ClusterHarness`, `ProcessPipelineClient`, `TensorParallelPipelineClient`, `EmbeddedNodeServer` |
| `integration` | `InProcessClusterIT`, `ThreeNodeClusterIT`, `TensorParallelClusterIT`, `ModelLiveRunner`, `GpuForwardPassIT` |

---

## Supported Models

Any GGUF file with a LLaMA-compatible or Phi-3-compatible architecture. Tested:

| Model | File size | Heap needed |
|-------|-----------|-------------|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | 2g |
| phi-3.5-mini-instruct Q4_K_M | 2.4 GB | 4g |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | 8g |
| Llama-3.2-8B-Instruct Q4_K_M | 4.9 GB | 8g |
| Llama-3.1-70B-Instruct Q4_K_M | 40 GB | 16 x 4 GB nodes |

**Quantisation types:** F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K.

**Chat templates:** `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default), `phi3`.
Template resolved from model type string via exact match then substring fallback.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry,player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl integration             # integration tests — forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT (pipeline) and
                                       # TensorParallelClusterIT (tensor)

./juno test --model-path /path/to/model.gguf   # real-model smoke test
```

### GPU tests (requires CUDA 12.x and Nvidia GPU)

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl integration \
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
- **Two-mode weight dequantization (`--dequant eager|lazy`).** The `eager` mode (default) dequantizes all projection weight matrices once at shard load time and uploads them to the GPU as `DeviceFloatMatrix` instances inside a `GpuWeightShard`. Every decode step calls `CudaMatVec.sgemv(DeviceFloatMatrix, x)` — no H2D weight copy per token, GPU utilisation is high. Cost: ~2–4x the compressed file size in VRAM per shard.
  The `lazy` mode keeps weights as raw `GgufReader.QuantizedTensor` bytes in JVM heap. Dequantization runs one 256-element block at a time inside the matmul loop on the CPU — peak live float footprint per matmul is ~1 kB, making it possible to run 3.8B models with `--heap 4g` even on low-VRAM nodes. The flag propagates through the full stack: run scripts → coordinator JVM → forked node JVMs via `JUNO_DEQUANT` system property.
- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else.
- **KV cache wired to the node-level manager.** `NodeKVCacheAdapter` connects `LlamaTransformerHandler` and `Phi3TransformerHandler` to the `KVCacheManager` (GPU byte-budget LRU + Caffeine W-TinyLFU CPU tier). Every forward pass flushes key/value data write-through into both tiers. If a local entry is evicted under heap pressure, the next forward pass at that position restores it from the manager transparently. `evict(requestId)` propagates to both the local HashMap and both cache tiers, closing the gap that previously made the entire `kvcache` module inert at the node level.
- **Star AllReduce for tensor parallel.** No inter-node communication. The coordinator collects partial logit vectors from all N nodes and sums them in O(N × vocabSize). Simpler than ring-AllReduce and requires no InfiniBand.
- **LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps `LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4–16) on the Q and V projections. The frozen weights stay quantized at all times — backward passes dequantize one block per row via dedicated `transposedQ4K` / `transposedQ5K` / `transposedQ6K` scatter-reduce implementations. Adapters are persisted to a `.lora` binary checkpoint; the GGUF is never modified.
- **Full JFR instrumentation across every hot path.** Five custom event types — `juno.MatVec`, `juno.ForwardPass`, `juno.Tokenizer`, `juno.TemplateFormat`, `juno.LoraTrainStep` — make every layer of the stack observable in JDK Mission Control without any agent or bytecode manipulation. Activated with `--jfr DURATION` on any `juno` command.
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- `MatVec` interface decouples the matmul backend from the transformer logic. `CudaMatVec` implements host `sgemv` (full H2D per call) and device `sgemv(DeviceFloatMatrix, x)` for GPU-resident weights. `GpuWeightShard` dequantizes and uploads all projection matrices once at shard load time when `--dequant eager` (the default). `CpuMatVec` as CPU fallback and test reference. Injected into `LlamaTransformerHandler` and `Phi3TransformerHandler` at construction time via `ForwardPassHandlerLoader`; swapping backends or dequant modes changes where arithmetic runs without touching model logic.
- GPU tests excluded from default CI by failsafe `<excludes>` and a `-Pgpu` profile. `GpuForwardPassIT` additionally guards with `-Djuno.gpu.test=true` to prevent CUDA native libs (bytedeco) loading into the coordinator JVM and poisoning FD inheritance into forked node processes.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub.

---

## Requirements

- JDK 21+
- Maven 3.9+
- CUDA / NVIDIA driver (GPU nodes only — not required for CPU mode or any unit/integration tests); Java bindings via org.bytedeco cuda-platform

---

## License

Apache 2.0