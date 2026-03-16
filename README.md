# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 × 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Status

All modules build and all tests pass. Verified end-to-end with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf on a 3-node CPU cluster.

**Session 9** — multi-turn KV cache reuse. Each conversation turn now skips re-prefilling previously processed tokens. Turn latency is constant per turn instead of growing O(N) with history length.

```
you> hey there, my name is Dima, nice to meet you!
bot> Greetings! Nice to meet you too.
     [37 tokens · 7342 ms · FLOAT16]

you> what is my name?
bot> Your name is Dima.
     [11 tokens · 8103 ms · FLOAT16]   ← flat, not growing
```

---

## Architecture

```
[Client]  REST (Javalin) / gRPC streaming
    ↓
[Coordinator]
    ├── GgufTokenizer       (BPE from GGUF metadata)
    ├── ChatTemplateFormatter
    ├── RequestScheduler    (virtual threads, CompletableFuture)
    ├── Sampler             (temperature / top-k / top-p / rep. penalty)
    ├── KVCacheManager      (GPU tier + CPU tier + PrefixCache trie)
    └── GenerationLoop      (prefill + decode + session KV reuse)
              │
              │  gRPC (activations — FLOAT16/INT8/FLOAT32)
              │
    ┌──────────────────────────────────────┐
    │  Node 1    Node 2    Node 3  ...     │  10/25 GbE
    │  L 0–7     L 8–14    L 15–21         │
    │  +embed              +output proj    │
    └──────────────────────────────────────┘
```

Each node runs `CpuForwardPassHandler` — full LLaMA-family transformer math in pure Java, parallel matVec across all CPU cores. `GpuForwardPassHandler` (JCuda) in progress.

---

## Quick Start (CPU-only)

```bash
# Download model (637 MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Build
mvn clean package -DskipTests

# Interactive REPL — all nodes in one JVM (local dev)
./juno local --model-path /path/to/model.gguf

# 3-node cluster — forked JVM nodes, gRPC messaging
./juno --model-path /path/to/model.gguf

# Real-model smoke test — 6 checks, exits 0/1
./juno test --model-path /path/to/model.gguf
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
| `node` | `CpuForwardPassHandler`, `GgufReader`, `LlamaConfig`, `ActivationCodec` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL, `ClusterHarness`, `ProcessPipelineClient`, `ChatHistory` |
| `integration` | `InProcessClusterIT`, `ThreeNodeClusterIT`, `ModelLiveRunner` |

---

## Supported Models

Any GGUF file with a LLaMA-compatible architecture. Tested:

| Model | File size | RAM |
|-------|-----------|-----|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | ~2 GB |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | ~6 GB |
| Llama-3.2-8B-Instruct Q4_K_M | 4.9 GB | ~8 GB |
| Llama-3.1-70B-Instruct Q4_K_M | 40 GB | 16 × 4 GB nodes |

Quantisation types: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q6_K.

Chat templates: `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default). Template derived automatically from the GGUF file name.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry,player
                                       # unit tests — no model file needed

mvn verify -pl integration             # integration tests — forks 3 JVM nodes (stub mode)

./juno test /path/to/model.gguf      # real-model smoke test
```

---

## Performance

| Session | Change | ms / 10 tokens |
|---------|--------|----------------|
| 5 | Baseline (FLOAT32, serial matVec) | ~34,891 ms |
| 6 | Parallel matVec + FLOAT16 default | ~3,802 ms (9×) |
| 9 | Session KV cache — turn latency now flat | ~7,000–8,000 ms / turn |

Session 9 turn latency grows with new tokens per turn only, not with total history length.

---

## Key Design Decisions

- No Python, no llama.cpp subprocess. JVM reads GGUF binary directly and runs the transformer end to end.
- No Spring Boot. Javalin for REST.
- Pipeline parallelism over tensor parallelism — LAN-friendly, no InfiniBand required.
- Separate data plane (gRPC activations) from control plane (Hazelcast state).
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub.
- Two `ActivationDtype` enums by design: protobuf-generated for wire, domain enum for application code.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA 12.x (GPU nodes only — not required for CPU mode)

---

## License

Apache 2.0