# Juno

**Java Unified Neural Orchestration** — distributed LLM inference and fine-tuning.
No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## 0. Features

- Distributed inference — pipeline-parallel (layer shards) or tensor-parallel (weight slices) across nodes
- GPU acceleration — CUDA/cuBLAS with FP16 resident weights; CPU fallback
- LoRA — train, checkpoint, inference adapter; optional merge to standalone GGUF
- OpenAI-compatible REST — `POST /v1/chat/completions`, `GET /v1/models`; swap base URL only
- JFR metrics — custom flight-recorder events across hot paths; merged cluster snapshots

## 1. How to use

### 1.1 Maven Central jars and docs (end-to-end)

Depend on **`cab.ml` artifacts at version `0.1.0`** from Maven Central ([search](https://central.sonatype.com/search?q=g:cab.ml+juno)); publisher is verified for **ml.cab**. Library modules plus shaded **`juno-player`**, **`juno-node`**, and **`juno-master`** jars are published from this reactor.

Then follow **[docs/howto.md](docs/howto.md)** for `./juno` commands and **[docs/integration-maven.md](docs/integration-maven.md)** for dependency snippets and classpath notes.

Contributors can build from source: `mvn clean package -DskipTests`.

### 1.2 Local player and LoRA (including Hugging Face–origin weights)

Download a GGUF locally (for example from Hugging Face) and pass **`--model-path`**. Interactive inference: `./juno local --model-path /path/to/model.gguf`. REST alongside REPL: add **`--api-port 8080`**. Training: `./juno lora --model-path ...` (see **[docs/LoRA.md](docs/LoRA.md)**).

Optional **`./juno merge`** bakes a trained `.lora` into a new GGUF so inference needs no sidecar adapter (**[docs/howto.md](docs/howto.md)**).

#### 1.2.1 Merge legality

Redistributing merged weights may trigger base-model and adapter license questions; Juno does not provide a legal determination — see **[docs/legal.md](docs/legal.md)**.

### 1.3 On-prem orchestration

Run **`juno-master`** as the coordinator and **`juno-node`** on each worker with gRPC between them (systemd or your own process manager). Parallelism modes and byte-order flags match local cluster harness behaviour described in **[docs/howto.md](docs/howto.md)**; topology and components are in **[docs/arch.md](docs/arch.md)**. AWS automation under **`scripts/aws/`** is optional cloud packaging of the same roles.

## 2. Feature details

| Topic | Doc |
|-------|-----|
| Distributed inference | [docs/features/distributed-inference.md](docs/features/distributed-inference.md) |
| OpenAI REST API | [docs/features/openai-rest-api.md](docs/features/openai-rest-api.md) |
| LoRA and merge | [docs/features/lora-and-merge.md](docs/features/lora-and-merge.md) |
| GPU acceleration | [docs/features/gpu-acceleration.md](docs/features/gpu-acceleration.md) |
| JFR and metrics | [docs/features/jfr-metrics.md](docs/features/jfr-metrics.md) |

## 3. Integrations

Copy-paste prompts for another agent:

- **Maven:** “How do I add Juno `cab.ml` Maven Central jars at version `0.1.0` to my project so the Juno player or embedded inference runs end-to-end?” (start from **[docs/integration-maven.md](docs/integration-maven.md)** and **[docs/howto.md](docs/howto.md)**.)

- **AWS:** “How do I run a Juno cluster on AWS using `scripts/aws`?” (see **[docs/howto.md](docs/howto.md)** and **[docs/aws-free-tier-billing.md](docs/aws-free-tier-billing.md)**.)

## 4. Stack

Node coordination and inference RPCs use **gRPC** with **protobuf** contracts from the `api` module. GPU matmul uses **CUDA 12.x** and **cuBLAS** via ByteDeko/JavaCPP (`CudaMatVec`), with a CPU quantised path when GPU is off or unavailable. The coordinator HTTP surface (**REST** and **SSE**) is implemented with **Javalin**.

## 5. Useful refs

- Performance matrix: **[docs/juno_test_matrix.html](docs/juno_test_matrix.html)** — methodology companion **[docs/performance.md](docs/performance.md)**
- Legal Q&A: **[docs/legal.md](docs/legal.md)**

---

## Requirements

JDK 25+, Maven 3.9+, CUDA 12.x + NVIDIA driver on GPU nodes (optional for CPU-only).

## Build and test

```bash
mvn clean package -DskipTests

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master
```

GPU or large-model checks: **[docs/howto.md](docs/howto.md)**.

## Supported models

GGUF with LLaMA-compatible or Phi-3-compatible architectures (quantizations include F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). Chat templates include `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`, `phi3`. Examples (heap hints): TinyLlama Q4_K_M (~637 MB, `2g`), phi-3.5-mini Q4_K_M (~2.4 GB, `4g`), Mistral-7B Q4_K_M (~4.1 GB, `8g`), Llama-3.1-70B Q4_K_M distributed.

## Modules (overview)

| Module | Role |
|--------|------|
| `api` | OpenAPI spec, protobuf/gRPC API |
| `registry` | Shard planning, model registry |
| `coordinator` | Scheduler, generation loop, REST |
| `node` | Transformer handlers, GGUF, CUDA matmul |
| `lora` | Adapter tensors, optimizer |
| `tokenizer`, `sampler`, `kvcache`, `health`, `metrics` | Shared infrastructure |
| `juno-player` | CLI REPL and cluster harness |
| `juno-node`, `juno-master` | Shaded deploy jars |

Details: **[docs/arch.md](docs/arch.md)**.

## License

Apache 2.0 — see [LICENSE](LICENSE).
