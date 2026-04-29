# Juno

**Java Unified Neural Orchestration** — distributed LLM inference and fine-tuning framework.
No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

Juno reads GGUF model files directly, distributes transformer computation across nodes via gRPC,
and exposes a REST/SSE inference API. It supports both **inference** and **LoRA fine-tuning** from
a single launcher — with no external runtime dependencies.

**Core capabilities:**

- **Distributed inference** — pipeline-parallel (layer sharding) or tensor-parallel (weight slicing) across N nodes
- **GPU acceleration** — CUDA/cuBLAS with FP16 resident weights; graceful CPU fallback on OOM
- **LoRA fine-tuning** — train, checkpoint, deploy, and merge adapters without modifying the base GGUF
- **OpenAI-compatible REST API** — `POST /v1/chat/completions` and `GET /v1/models` speak the OpenAI wire format; any client that works with OpenAI needs only a base-URL change
- **JFR instrumentation** — five custom event types covering every hot path, auto-merged across JVMs in cluster mode

> **Full usage reference:** [docs/howto.md](docs/howto.md)
> **LoRA training guide:** [docs/LoRA.md](docs/LoRA.md)
> **Architecture deep-dive:** [docs/arch.md](docs/arch.md)

---

## Architecture

Juno supports two distribution strategies: **pipeline-parallel** (contiguous layer shards flow
serially node-1 → node-2 → node-3, pooling VRAM to enable larger models) and **tensor-parallel**
(all nodes hold all layers but a horizontal weight slice; the coordinator broadcasts tokens and
reduces partial logits via star AllReduce, increasing throughput). For full component diagrams,
module dependency graph, handler routing, and key design decisions see [docs/arch.md](docs/arch.md).

---

## Quick Start

Juno has two deployment modes:

- **Local** — single machine, via `scripts/run.sh` and the built-in `juno-player` REPL
- **AWS Cloud** — distributed cluster via `scripts/aws/juno-deploy.sh`, coordinated by `juno-master` with `juno-node` running on each inference node

**Steps:**

1. **Build**

   ```bash
   mvn clean package -DskipTests
   ```

2. **Run locally** — launch the interactive REPL or serve the OpenAI-compatible REST API:

   ```bash
   ./juno local --model-path /path/to/model.gguf
   ./juno local --model-path /path/to/model.gguf --api-port 8080
   ```

   See [docs/howto.md](docs/howto.md) for the full flag reference, cluster mode, LoRA inference, and more.

3. **Deploy to AWS** — requires the AWS CLI installed and an IAM user with EC2/IAM permissions:

   ```bash
   cd scripts/aws
   ./launcher.sh juno-deploy.sh setup --instance-type g4dn.xlarge --node-count 3
   ./launcher.sh juno-deploy.sh start
   ```

   > Questions about cloud deployment? Reach us at [dev@ml.cab](mailto:dev@ml.cab) or join our Discord (link TBD).

---

## OpenAI-Compatible REST API

Pass `--api-port N` to any `local` or cluster invocation to start the REST API server. The
coordinator exposes three endpoints wire-compatible with OpenAI:

| Endpoint | OpenAI equivalent | Description |
|---|---|---|
| `POST /v1/chat/completions` | `POST /v1/chat/completions` | Blocking or SSE streaming completion |
| `GET /v1/models` | `GET /v1/models` | List all loaded models |
| `GET /v1/models/{model}` | `GET /v1/models/{model}` | Single model metadata |

Any client already targeting the OpenAI API works without modification — change only the
base URL:

```python
# Python openai SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    messages=[{"role": "user", "content": "What is Java?"}],
)
print(response.choices[0].message.content)
```

```bash
# curl — blocking
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Hello"}]}'

# curl — streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**Juno-specific request extensions** (all optional):

| Field | Type | Description |
|---|---|---|
| `x_juno_priority` | string | `HIGH` / `NORMAL` / `LOW` — scheduler queue priority |
| `x_juno_session_id` | string | Stable ID across turns; enables KV-cache reuse |
| `x_juno_top_k` | integer | Top-K cutoff (0 = disabled; default 50) |

**Supported fields:** `model`, `messages`, `temperature`, `top_p`, `max_completion_tokens`,
`max_tokens` (deprecated alias), `frequency_penalty`, `stream`, `n` (only 1 accepted).

**Silently ignored** (accepted for client compatibility): `stop`, `presence_penalty`,
`logit_bias`, `user`, `seed`.

The full OpenAPI 3.0 specification is in `api/src/main/resources/juno-api.yaml`.

---

## GPU Support

GPU acceleration is provided via **CUDA/cuBLAS** (CUDA 12.x). Weights are dequantized once and
uploaded as FP16 on load; VRAM is freed deterministically on shard unload or swap. In cluster
mode, each forked node JVM owns its own GPU context; the coordinator allocates none. Pass `--cpu`
or set `JUNO_USE_GPU=false` to force CPU inference. See [docs/arch.md](docs/arch.md) for the full
GPU weight lifecycle and multi-device details.

---

## JFR Instrumentation

All `juno` commands accept `--jfr DURATION` (e.g. `--jfr 5m`), activating Java Flight Recording
across the full stack. Six custom event types are available in JDK Mission Control:

| Event | Category | Key fields |
|-------|----------|------------|
| `juno.MatVec` | Juno/MatVec | `backend`, `rows`, `cols` |
| `juno.ForwardPass` | Juno/Inference | `handlerType`, `requestId`, `startPosition`, `layerCount` |
| `juno.TokenProduced` | Juno/Inference | `requestId`, `position` |
| `juno.Tokenizer` | Juno/Tokenizer | `tokenizerType`, `operation`, `inputLength`, `outputLength` |
| `juno.TemplateFormat` | Juno/Tokenizer | `modelType`, `messageCount`, `outputLength` |
| `juno.LoraTrainStep` | Juno/LoRA | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` |

In **cluster mode**, coordinator and every node JVM write their own `.jfr` file; on shutdown
`MetricsMain.extractToJsonMerged()` merges all files into a single snapshot written to
`target/metrics/metrics.json`. In **local mode**, extraction happens automatically on period
expiry or REPL exit.

Post-hoc extraction from any `.jfr` file:

```bash
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
```

---

## Modules

| Module | Contents |
|--------|----------|
| `api` | OpenAPI 3.0 spec (`juno-api.yaml` — OpenAI-compatible), legacy `openapi.yaml`, JAX-RS interfaces, `inference.proto` |
| `registry` | `NodeDescriptor`, `ShardPlanner`, `ShardMap`, `ParallelismType`, `TensorShardAssignment`, `TensorShardPlanner` |
| `coordinator` | `GenerationLoop`, `RequestScheduler`, `FaultTolerantPipeline`, `OpenAiChatHandler`, `OpenAiAdapter`, Javalin REST, SSE |
| `kvcache` | `KVCacheManager`, `GpuKVCache`, `CpuKVCache`, `PrefixCache` |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE + GPT-2 BPE; auto-detected), `ChatTemplate`, `SimpleTokenizer` |
| `node` | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `ForwardPassHandlerLoader`, `EmbeddedNodeServer`, `NodeMain`, `NodeKVCacheAdapter`, `MatVec`, `CpuMatVec`, `CudaMatVec`, `GpuContext`, `DeviceFloatMatrix`, `DeviceHalfMatrix`, `GgufReader`, `ActivationCodec`, `ShardContext`, `TensorShardContext`, `LoraQvInitializer`, `LoraMerge`, `LoraTrainableHandler` |
| `lora` | `LoraAdapter`, `LoraAdapterSet` (checkpoint I/O), `LoraAdamOptimizer` — pure Java, no GGUF/CUDA |
| `sampler` | Temperature, top-k, top-p, repetition penalty — pure Java |
| `health` | `NodeHealth`, `HealthReporter`, `HealthEvaluator`, `CircuitBreaker`, Javalin health sidecar (`/health-ui`, `/health/probe`, `/health/circuits`) |
| `metrics` | JFR extractor: `JfrMetricsExtractor`, `JfrModelMapper`, `JfrPercentiles`, `MetricsSnapshot`, `MetricsWriter`, `MetricsMain` |
| `juno-player` | `ConsoleMain` REPL, `ClusterHarness`, `ProcessPipelineClient`, `TensorParallelPipelineClient`, `LoraMergeMain` |
| `juno-node` | Fat jar (`juno-node.jar`). Entry point `NodeMain`. Launched by `juno-node.service` on AWS nodes. |
| `juno-master` | Fat jar (`juno-master.jar`). Entry point `CoordinatorMain`. Standalone coordinator for remote deployment. |

---

## Supported Models

Any GGUF file with a LLaMA-compatible or Phi-3-compatible architecture.

| Model | File size | Heap needed |
|-------|-----------|-------------|
| TinyLlama-1.1B-Chat Q4_K_M | 637 MB | 2g |
| TinyLlama-1.1B-Chat Q2_K | 380 MB | 2g |
| phi-3.5-mini-instruct Q4_K_M | 2.4 GB | 4g |
| Mistral-7B-Instruct Q4_K_M | 4.1 GB | 8g |
| Meta-Llama-3.2-1B-Instruct Q8_0 | 1.3 GB | 4g |
| Meta-Llama-3.1-70B-Instruct Q4_K_M | 40 GB | distributed across nodes |

**Quantization types:** F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

**Chat templates:** `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml` (default), `phi3`.
Template is resolved from the model path via exact match then substring fallback.

---

## Build and Test

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master             # integration tests (stub mode, no model/GPU)
```

For model-file integration tests, GPU tests (requires CUDA 12.x and an NVIDIA GPU), and the
full smoke-test reference see [docs/howto.md](docs/howto.md).

---

## Requirements

- JDK 25+
- Maven 3.9+
- CUDA 12.x + NVIDIA driver (GPU nodes only; not required for CPU mode or tests)

---

## License

Apache 2.0