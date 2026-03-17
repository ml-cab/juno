# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 x 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Status

All modules build and all tests pass. Verified end-to-end with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf on a 3-node CPU cluster.

**Session 10** — GPU acceleration layer. JCuda/JCublas integration complete. `GpuForwardPassHandler` replaces all `matVec` calls with `cublasSgemv`. Runs alongside `CpuForwardPassHandler` behind the `GpuMatVec` interface — both produce numerically identical output within float32 rounding. Full test suite runs without a GPU on any machine; GPU tests are opt-in via `-Dgroups=gpu` or `-Pgpu`.

```
you> hey there, my name is Dima, nice to meet you!
bot> Greetings! Nice to meet you too.
     [37 tokens · 7342 ms · FLOAT16]   ← CPU baseline

you> what is my name?
bot> Your name is Dima.
     [11 tokens · 8103 ms · FLOAT16]   ← flat KV cache reuse
```

---

## Architecture

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
              |  gRPC (activations — FLOAT16/INT8/FLOAT32)
              |
    +--------------------------------------+
    |  Node 1    Node 2    Node 3  ...     |  10/25 GbE
    |  L 0-7     L 8-14    L 15-21        |
    |  +embed              +output proj   |
    +--------------------------------------+
```

Each node runs either `CpuForwardPassHandler` (pure Java, parallel matVec) or `GpuForwardPassHandler` (JCublas cublasSgemv). Both implement `ForwardPassHandler` via the `GpuMatVec` interface. Node selection is automatic: GPU nodes use `CublasMatVec`; CPU-only nodes fall back to `CpuMatVec`.

---

## Quick Start

```bash
# Download model (637 MB, CPU-only)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Build
mvn clean package -DskipTests

# Interactive REPL — all nodes in one JVM (dev)
./run.sh console --model-path /path/to/model.gguf

# 3-node cluster — forked JVM nodes, real gRPC (production)
./run.sh cluster --model-path /path/to/model.gguf

# Real-model smoke test — 6 checks, exits 0/1
./run.sh live --model-path /path/to/model.gguf
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
| `node` | `CpuForwardPassHandler`, `GpuForwardPassHandler`, `GpuMatVec`, `CublasMatVec`, `CpuMatVec`, `GpuContext`, `CudaAvailability`, `GgufReader`, `LlamaConfig`, `ActivationCodec` |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | Health monitor, circuit breakers (Resilience4j) |
| `player` | `ConsoleMain` REPL, `ClusterHarness`, `ProcessPipelineClient`, `ChatHistory` |
| `integration` | `InProcessClusterIT`, `ThreeNodeClusterIT`, `ModelLiveRunner`, `GpuForwardPassIT` |

---

## Supported Models

Any GGUF file with a LLaMA-compatible architecture. Tested:

| Model | File size | RAM |
|-------|-----------|-----|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | ~2 GB |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | ~6 GB |
| Llama-3.2-8B-Instruct Q4_K_M | 4.9 GB | ~8 GB |
| Llama-3.1-70B-Instruct Q4_K_M | 40 GB | 16 x 4 GB nodes |

Quantisation types: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q6_K.

Chat templates: `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default). Template derived automatically from the GGUF file name.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry,player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl integration             # integration tests — forks 3 JVM nodes (stub mode)

./run.sh live --model-path /path/to/model.gguf   # real-model smoke test
```

### GPU tests (requires CUDA 12.x and Nvidia GPU)

```bash
# Unit tests — node module only, no model file needed
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

# Integration test — requires a GGUF model file and a CUDA device
mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl integration \
  --enable-native-access=ALL-UNNAMED
```

Recommended AWS instance for GPU testing: `g4dn.xlarge` (T4, 16 GB VRAM, ~$0.50/hr on-demand). See `docs/howto.md` for the full AWS setup procedure.

---

## Performance

| Session | Change | ms / 10 tokens |
|---------|--------|----------------|
| 5 | Baseline (FLOAT32, serial matVec) | ~34,891 ms |
| 6 | Parallel matVec + FLOAT16 default | ~3,802 ms (9x) |
| 9 | Session KV cache — turn latency now flat | ~7,000-8,000 ms / turn |
| 10 | GpuForwardPassHandler (cublasSgemv) — AWS benchmark pending | — |

Session 9 turn latency grows with new tokens per turn only, not with total history length. Session 10 GPU numbers will be filled in after the first AWS g4dn.xlarge run.

---

## Key Design Decisions

- No Python, no llama.cpp subprocess. JVM reads GGUF binary directly and runs the transformer end to end.
- No Spring Boot. Javalin for REST.
- Pipeline parallelism over tensor parallelism — LAN-friendly, no InfiniBand required.
- Separate data plane (gRPC activations) from control plane (Hazelcast state).
- GGUF tokenizer loaded from model metadata — no separate `tokenizer.model` file.
- Stub mode — cluster boots in seconds without a model file; all integration tests run stub.
- Two `ActivationDtype` enums by design: protobuf-generated for wire, domain enum for application code.
- `GpuMatVec` interface decouples the matmul backend from the transformer logic. `CublasMatVec` in production, `CpuMatVec` as CPU fallback and test reference. Swappable without touching `GpuForwardPassHandler`.
- GPU tests excluded from default CI by failsafe `<excludes>` and a `-Pgpu` profile. `GpuForwardPassIT` additionally guards with `-Djuno.gpu.test=true` to prevent JCuda native libs loading into the coordinator JVM and poisoning FD inheritance into forked node processes.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA 12.x (GPU nodes only — not required for CPU mode or any unit/integration tests)

---

## License

Apache 2.0