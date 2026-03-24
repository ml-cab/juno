# juno

Distributed Java-native LLM inference engine. Runs large language models across a cluster of commodity GPUs. No Python, no GIL, no Spring.

**16 x 4 GB GPUs = 64 GB total VRAM at a fraction of the cost of a single high-VRAM card.**

---

## Status

All modules build and all tests pass. Verified end-to-end with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf on a 3-node CPU cluster.

**Session 10** — GPU acceleration layer. JCuda/JCublas integration complete. `GpuForwardPassHandler` routes matmul through `cublasSgemv`. Production GPU loads use `GpuForwardPassHandler.loadGpuResident`: each weight matrix is uploaded once (`DeviceFloatMatrix`); per-token matmuls copy only the activation vector and result, not full weights. `GpuForwardPassHandler.load` with host tensors remains for `CpuMatVec` and tests. Full test suite runs without a GPU on any machine; GPU tests are opt-in via `-Dgroups=gpu` or `-Pgpu`.

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

Each node runs either `CpuForwardPassHandler` (pure Java, parallel matVec) or `GpuForwardPassHandler` (cuBLAS cublasSgemv via org.bytedeco cuda). GPU nodes load resident device weights via `loadGpuResident`; CPU-only nodes use `CpuMatVec`. Both implement `ForwardPassHandler` via the `GpuMatVec` interface. Works with various NVIDIA GPUs (e.g. GTX 1080, T4).

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

**GPU cluster:** The harness extracts JavaCPP CUDA natives into a cache and passes `java.library.path` to forked node JVMs so they can load `jnicudart`. You need `cudart64_12.dll` (and on Linux `libcudart.so`) for the loader to succeed. Options: (1) Install the NVIDIA CUDA Toolkit and set `CUDA_PATH` or `CUDA_HOME`, or run `setenv.bat` / `source setenv.sh` so its `bin` is on `PATH`; (2) On Windows only, use the single-DLL option: run `get-cudart.bat` to open https://www.dllme.com/dll/files/cudart64_12 and save the 64-bit DLL as `%USERPROFILE%\.javacpp\cache\cudart64_12.dll` (no full toolkit download).

---

## Runtime Flags (`scripts/run.sh`)

`scripts/run.sh` supports `cluster` (default), `local`, and `test` modes. Aliases match `run.bat`: `console` is the same as `local`; `live` is the same as `test`.

### Common flags (`cluster` and `local`)

| Flag | Description | Default |
|------|-------------|---------|
| `--model-path PATH` | Path to GGUF model file (required unless `MODEL_PATH` is set). | — |
| `--dtype FLOAT32\|FLOAT16\|INT8` | Activation wire format between shards. | `FLOAT16` |
| `--float16`, `--fp16` | Shorthand for `--dtype FLOAT16`. | — |
| `--float32` | Shorthand for `--dtype FLOAT32` (reference/debug runs). | — |
| `--int8` | Shorthand for `--dtype INT8` (max compression). | — |
| `--max-tokens N` | Maximum generated tokens per response. | `200` |
| `--temperature F` | Sampling temperature. | `0.6` |
| `--top-k N` | Top-K sampling cutoff (`0` disables). | `20` |
| `--top-p F` | Nucleus sampling top-p (`0` disables). | `0.95` |
| `--heap SIZE` | JVM heap size (for example `4g`, `8g`, `16g`). | `4g` |
| `--gpu` | Prefer GPU matmul when CUDA is available (passed to `player.jar`). | on |
| `--cpu` | Force CPU matmul. | — |
| `--verbose`, `-v` | Enable verbose startup/runtime logs. | off |

### Mode-specific flags

| Mode | Flag | Description | Default |
|------|------|-------------|---------|
| `local` | `--nodes N` | Number of in-process shards. | `3` |
| `test` | `--model-path PATH` | GGUF model path for `ModelLiveRunner` smoke tests. | — |
| `test` | `--heap SIZE` | JVM heap for smoke tests. | `4g` |

For full mode help:

```bash
scripts/run.sh cluster --help
scripts/run.sh local --help
scripts/run.sh test --help
```

Environment variable overrides (equivalent to flag counterparts): `MODEL_PATH`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `USE_GPU` (`true`/`false`). With `--gpu`, the launcher prepends `$CUDA_PATH/bin` or `$CUDA_HOME/bin` to `PATH` when set, so the CUDA runtime can be found (same idea as `run.bat` on Windows).

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE), `ChatTemplate`, `StubTokenizer` |
| `node` | `CpuForwardPassHandler`, `GpuForwardPassHandler`, `GpuMatVec`, `CublasMatVec`, `DeviceFloatMatrix`, `CpuMatVec`, `GpuContext`, `CudaAvailability`, `GgufReader`, `LlamaConfig`, `ActivationCodec` (GPU: org.bytedeco cuda) |
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

### GPU tests (requires CUDA and Nvidia GPU; uses org.bytedeco cuda-platform)

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
- `GpuMatVec` interface decouples the matmul backend from the transformer logic. `CublasMatVec` implements host `sgemv` (full H2D per call) and device `sgemv(DeviceFloatMatrix, x)` for resident weights. `CpuMatVec` as CPU fallback and test reference. Swappable without touching `GpuForwardPassHandler`.
- GPU tests excluded from default CI by failsafe `<excludes>` and a `-Pgpu` profile. `GpuForwardPassIT` additionally guards with `-Djuno.gpu.test=true` to prevent CUDA native libs (bytedeco) loading into the coordinator JVM and poisoning FD inheritance into forked node processes.

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA / NVIDIA driver (GPU nodes only — not required for CPU mode or any unit/integration tests); Java bindings via org.bytedeco cuda-platform

---

## License

Apache 2.0