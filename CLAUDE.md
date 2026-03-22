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

**Session 13** — Tensor-parallel mode (`--pType tensor`). A new `ParallelismType` enum selects between two distribution strategies at cluster startup. `TensorParallelPipelineClient` broadcasts each decode step to all nodes in parallel and reduces partial logit vectors (star AllReduce). `TensorShardPlanner` assigns sequential tensor ranks; `TensorShardContext` exposes per-node geometry (`headsPerNode`, `headStart`, `headEnd`, `sliceDim`). `ClusterHarness.tensorNodes()` factory wires the 3-node tensor-parallel cluster. `inference.proto LoadShardRequest` gains `tensor_rank` (field 7) and `tensor_world_size` (field 8). `LlamaTransformerHandler` and `Phi3TransformerHandler` use lazy `GgufReader.QuantizedTensor` for all projection weights — O(block) dequantisation at matmul time instead of O(model) at load time, enabling tensor-parallel with `--heap 4g`.

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

Constraint: `numHeads` must be even (divisible by 2). Heads are distributed across nodes by ceiling-division; 32 heads across 3 nodes = 10/11/11.

### Handler routing

```
ForwardPassHandlerLoader  <- reads general.architecture from GGUF
    |
    phi3  -> Phi3TransformerHandler   (fused QKV + gate/up, quantized weights)
    *     -> LlamaTransformerHandler  (separate tensors, quantized weights)

MatVecBackend (injected into handler):
    CpuMatVecBackend    <- parallel IntStream
    CudaMatVecBackend   <- cublasSgemv_v2 via JCublas2
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

# Real-model smoke test — 8 checks, exits 0/1
./juno test --model-path /path/to/model.gguf --heap 4g

# Tensor-parallel checks only
./juno test --model-path /path/to/model.gguf --heap 4g --pType tensor
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Path to GGUF file (required) |
| `--pType pipeline\|tensor` | `pipeline` | Parallelism type: `pipeline` = contiguous layer blocks, serial activation flow (vertical scaling); `tensor` = weight-matrix slices, all nodes in parallel with AllReduce (horizontal scaling). Constraint for tensor: `numHeads` must be even |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | Activation wire format between nodes |
| `--max-tokens N` | `200` | Maximum tokens per response |
| `--temperature F` | `0.6` | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | Nucleus sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | In-process shard count (`local` only) |
| `--verbose` / `-v` | — | Show node startup, gRPC and shard loading logs |

Environment overrides: `MODEL_PATH`, `PTYPE`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE), `ChatTemplate`, `StubTokenizer` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader`, `MatVecBackend`, `CpuMatVecBackend`, `CudaMatVecBackend`, `GgufReader`, `LlamaConfig`, `ActivationCodec`, `ShardContext`, `TensorShardContext` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL, `ClusterHarness`, `ProcessPipelineClient`, `TensorParallelPipelineClient`, `EmbeddedNodeServer` |
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

./juno test --model-path /path/to/model.gguf                  # real-model smoke test — 8 checks (pipeline + tensor), exits 0/1
./juno test --model-path /path/to/model.gguf --pType pipeline  # pipeline checks only (tests 1-6)
./juno test --model-path /path/to/model.gguf --pType tensor    # tensor-parallel checks only (tests 7-8)
```

### GPU tests (requires CUDA 12.x and Nvidia GPU)

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl integration \
  --enable-native-access=ALL-UNNAMED
```

---

## CPU Performance

| Session | Change | ms / 10 tokens |
|---------|--------|----------------|
| 5 | Baseline (FLOAT32, serial matVec) | ~34,891 ms |
| 6 | Parallel matVec + FLOAT16 default | ~3,802 ms (9x) |
| 9 | Session KV cache — turn latency flat | ~7,000-8,000 ms / turn |
| 10 | CudaMatVecBackend (cublasSgemv) | AWS benchmark pending |
| 11 | Phi-3.5-mini on 3-node CPU cluster | ~38,420 ms / turn |
| 13 | Tensor-parallel mode (pType=tensor) | same throughput per node; parallel decode step |

---

## Key Design Decisions

- No Python, no llama.cpp subprocess. JVM reads GGUF binary directly and runs the transformer end to end.
- No Spring Boot. Javalin for REST.
- Two parallelism modes: `pipeline` (LAN-friendly, sequential activation flow, vertical scaling) and `tensor` (parallel per-step AllReduce, horizontal scaling, higher throughput). Selected at startup with `--pType`.
- **Lazy dequantization for large models.** Projection weights are kept as raw quantized bytes in `GgufReader.QuantizedTensor`. Dequantization runs one 256-element block at a time inside the matmul loop. Peak live float footprint per matmul is ~1 kB instead of ~65 MB, making it possible to run 3.8B models with `--heap 4g`.
- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `Phi3TransformerHandler` for `phi3`, `LlamaTransformerHandler` for everything else.
- **Star AllReduce for tensor parallel.** No inter-node communication. The coordinator collects partial logit vectors from all N nodes and sums them in O(N * vocabSize). Simpler than ring-AllReduce and requires no InfiniBand.
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- `MatVecBackend` interface decouples the matmul backend from the transformer logic.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA 12.x (GPU nodes only — not required for CPU mode or any unit/integration tests)

---

## License

Apache 2.0