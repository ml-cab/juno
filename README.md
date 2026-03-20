# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 x 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Status

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile 
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- **phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster** ← new in session 11

**Session 11** — Phi-3 family support. `PhiForwardPassHandler` handles fused-QKV and fused-gate+up tensor layouts. Quantized (Q4_K/Q5_K/Q6_K) weights are kept in raw bytes and dequantized block-by-block during matmul — eliminating the OOM kill that occurred when loading a 3.8B model with `--heap 12g`. All quantized matVec paths run fully parallel across all CPU cores.

```
you> are you alive?
bot> Yes, I'm here and ready to help! What do you need?
     [19 tokens · 38420 ms · FLOAT16]   ← phi-3.5-mini, 3-node CPU cluster

you> hello
bot> Hey! Nice to meet you too.
     [6 tokens · 2922 ms · FLOAT16]     ← TinyLlama, 3-node CPU cluster
```

---

## Architecture

```
[Client]  REST (Javalin) / gRPC streaming
    |
[Coordinator]
    |-- GgufTokenizer       (BPE from GGUF metadata)
    |-- ChatTemplateFormatter  (exact + substring match; phi3/phi-3 aliases)
    |-- RequestScheduler    (virtual threads, CompletableFuture)
    |-- Sampler             (temperature / top-k / top-p / rep. penalty)
    |-- KVCacheManager      (GPU tier + CPU tier + PrefixCache trie)
    +-- GenerationLoop      (prefill + decode + session KV reuse)
              |
              |  gRPC (activations — FLOAT16/INT8/FLOAT32)
              |
    +--------------------------------------+
    |  Node 1    Node 2    Node 3  ...     |  10/25 GbE
    |  L 0-10    L 11-21   L 22-31        |
    |  +embed              +output proj   |
    +--------------------------------------+
              |
    ForwardPassHandlerLoader  ← reads general.architecture from GGUF
              |
      phi3 ──→ PhiForwardPassHandler   (fused QKV + gate/up, quantized weights)
      llama ─→ CpuForwardPassHandler   (separate tensors, quantized weights)
      * ────→ GpuForwardPassHandler    (GpuMatVec backend, any arch)
```

Each node runs either `CpuForwardPassHandler` / `PhiForwardPassHandler` (pure Java, parallel matVec) or `GpuForwardPassHandler` (JCublas). Handler selection is now two-stage: `ForwardPassHandlerLoader` reads `general.architecture` from the GGUF file and routes to the correct implementation.

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

# Interactive REPL — 3-node cluster (forked JVMs)
./juno --model-path /path/to/model.gguf --heap 4g

# Interactive REPL — all nodes in one JVM (fastest startup)
./juno local --model-path /path/to/model.gguf --heap 4g

# Real-model smoke test — 6 checks, exits 0/1
./juno test --model-path /path/to/model.gguf --heap 4g
```

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE), `ChatTemplate`, `StubTokenizer` |
| `node` | `CpuForwardPassHandler`, `PhiForwardPassHandler`, `ForwardPassHandlerLoader`, `GpuForwardPassHandler`, `GpuMatVec`, `CublasMatVec`, `GgufReader`, `LlamaConfig`, `ActivationCodec` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL, `ClusterHarness`, `ProcessPipelineClient`, `ChatHistory` |
| `integration` | `InProcessClusterIT`, `ThreeNodeClusterIT`, `ModelLiveRunner`, `GpuForwardPassIT` |

---

## Supported Models

Any GGUF file with a LLaMA-compatible or Phi-3-compatible architecture. Tested:

| Model | File size | Heap needed |
|-------|-----------|-------------|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | 2g |
| **phi-3.5-mini-instruct Q4_K_M** | **2.4 GB** | **4g** |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | 8g |
| Llama-3.2-8B-Instruct Q4_K_M | 4.9 GB | 8g |
| Llama-3.1-70B-Instruct Q4_K_M | 40 GB | 16 x 4 GB nodes |

**Quantisation types:** F32, F16, BF16, Q8_0, Q4_0, Q4_K, **Q5_K**, Q6_K.
(Q5_K and Q6_K are used internally by Q4_K_M files for sensitive weight tensors.)

**Chat templates:** `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default), **`phi3`**.
Template resolved from model type string via exact match then substring fallback —
`"llama3-8b"` correctly resolves to `llama3`, `"phi-3.5-mini-instruct"` to `phi3`.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry,player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl integration             # integration tests — forks 3 JVM nodes (stub mode)

./juno test --model-path /path/to/model.gguf   # real-model smoke test
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
| 6 | Parallel matVec + FLOAT16 default | ~3,802 ms (9×) |
| 9 | Session KV cache — turn latency now flat | ~7,000–8,000 ms / turn |
| 10 | GpuForwardPassHandler (cublasSgemv) — AWS benchmark pending | — |
| **11** | **Phi-3.5-mini on 3-node CPU cluster** | **~38,420 ms / turn** |

Phi-3.5-mini is 3.4× larger than TinyLlama (3.8B vs 1.1B params) with a wider hidden dimension (3072 vs 2048) — the per-token time scales accordingly. All 16 CPU cores are fully utilised during generation. Tested at Linux Mint (recent) on Lenovo Yoga 7 Slim pro 14/ap7 16 cores/16 GB; Intel Alder Lake-P GT2; 

---

## Key Design Decisions

- No Python, no llama.cpp subprocess. JVM reads GGUF binary directly and runs the transformer end to end.
- No Spring Boot. Javalin for REST.
- Pipeline parallelism over tensor parallelism — LAN-friendly, no InfiniBand required.
- **Lazy dequantization for large models.** Projection weights for Phi-3 (and any other Q4_K/Q5_K/Q6_K model) are kept as raw quantized bytes in `GgufReader.QuantizedTensor`. Dequantization runs one 256-element block at a time inside the matmul loop. Peak live float footprint per matmul is ~1 kB instead of ~65 MB, making it possible to run 3.8B models with `--heap 4g`.
- **Architecture-aware handler routing.** `ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and selects `PhiForwardPassHandler` for `phi3`, `CpuForwardPassHandler` for everything else.
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- `GpuMatVec` interface decouples the matmul backend from the transformer logic.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA 12.x (GPU nodes only — not required for CPU mode or any unit/integration tests)

---

## License

Apache 2.0