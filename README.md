# jUno

**Java Unified Neural Orchestration** 

Distributed LLM inference and fine-tuning.

No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## 1. Features

- Playing open-source models, e.g. [huggingface](https://huggingface.co/);
- Providing distributed inference - Layer Sharding (pipeline parallelism) or Weight Slices (tensor parallelism) using pure Java; 0 sidecar processes;
- GPU acceleration supported - CUDA/cuBLAS with FP16 resident weights, CPU fallback in case of OOM;
- LoRA (Low-Rank Adaptation) supported. Train your data arranged by checkpoints; persist LoRA inference adapter for future use;
- OpenAI-compatible REST - `POST /v1/chat/completions`, `GET /v1/models`; swap the base URL only to integrate in your application;
- JFR metrics under the hood - custom flight-recorder events across hot paths; instrumentation driven development;


Please see full feature list **[here](docs/features.md)**

## 2. How to use

### 2.1 JVM Integration

Integrate on **`cab.ml` artifacts at version `0.1.0`** from Maven Central:

```
	<dependencyManagement>
	  <dependencies>
	    <dependency>
	      <groupId>cab.ml</groupId>
	      <artifactId>juno-bom</artifactId>
	      <version>0.1.0</version>
	      <type>pom</type>
	      <scope>import</scope>
	    </dependency>
	  </dependencies>
	</dependencyManagement>
```

then use player all-in-one http endpoints:

```
	<!-- juno-player exposes JunoHttpClient (and JunoPlayer for in-process use). -->
	<dependency>
	    <groupId>cab.ml</groupId>
	    <artifactId>juno-player</artifactId>
	</dependency>
	<!-- tokenizer provides cab.ml.juno.tokenizer.ChatMessage used by JunoHttpClient. -->
	<dependency>
	    <groupId>cab.ml</groupId>
	    <artifactId>tokenizer</artifactId>
	</dependency>
```

Then follow **[docs/howto.md](docs/howto.md)** `JVM integration` section or checkout the [spring-boot example](https://github.com/ml-cab/juno-spring-example/tree/master) for dependency and code snippets.

Maintainer - see **[docs/integration-maven.md](docs/integration-maven.md)**

### 2.2 Local player and LoRA (including Hugging Face–origin weights)

Contributors and enthusiasts can build from source: `mvn clean package -DskipTests`.

Download a GGUF (replace the URL with your chosen model):

```
cd juno/models
wget https://huggingface.co/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

then run local jUno interactive console to try and train inference

```
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**`--model-path`** is relative for juno project dir. REST alongside REPL: add **`--api-port 8080`**. 

Training: `./juno lora --model-path ...` (see **[docs/LoRA.md](docs/LoRA.md)**).

Optional **`./juno merge`** bakes a trained `.lora` into a new GGUF, so that inference needs no sidecar adapter 

More at **[howto.md](docs/howto.md)**.

### 2.3 On-prem orchestration

Run **`juno-master`** as the coordinator and **`juno-node`** on each worker with gRPC between them (systemd or your own process manager). Parallelism modes and byte-order flags match local cluster harness behaviour described in **[docs/howto.md](docs/howto.md)**; topology and components are in **[docs/arch.md](docs/arch.md)**. AWS automation under **`scripts/aws/`** is optional cloud packaging of the same roles.

## 3. Agent integration

Copy-paste prompts for another agent:

- **Maven:** “How do I add Juno `cab.ml` Maven Central jars at version `0.1.0` to my project so the Juno player or embedded inference runs end-to-end?” (start from **[docs/integration-maven.md](docs/integration-maven.md)** and **[docs/howto.md](docs/howto.md)**.)

- **AWS:** “How do I run a Juno cluster on AWS using `scripts/aws`?” (see **[docs/howto.md](docs/howto.md)** and **[docs/aws-free-tier-billing.md](docs/aws-free-tier-billing.md)**.)

## 4. Stack

Node coordination and inference RPCs use **gRPC** with **protobuf** contracts from the `api` module. GPU matmul uses **CUDA 12.x** and **cuBLAS** via **Panama FFI** (`java.lang.foreign` — `CudaBindings` resolves `libcudart.so.12` and `libcublas.so.12` at class-init time; `CudaMatVec` owns all device memory and stream lifecycle), with a CPU quantised path when GPU is off or unavailable. The coordinator HTTP surface (**REST** and **SSE**) is implemented with **Javalin**.

## 5. Useful refs

- Performance matrix: **[docs/juno_test_matrix.html](docs/juno_test_matrix.html)** - methodology companion **[docs/performance.md](docs/performance.md)**
- Legal Q&A: **[docs/legal.md](docs/legal.md)**

---

## Requirements

JDK 25+, Maven 3.9+, CUDA 12.x + NVIDIA driver on GPU nodes (optional for CPU-only).

## Build and test

```bash
mvn clean package -DskipTests

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests - no model file, no GPU needed

mvn verify -pl juno-master
```

GPU or large-model checks: **[docs/howto.md](docs/howto.md)**.

## Supported models

GGUF with LLaMA-compatible or Phi-3-compatible architectures (quantizations include F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). Chat templates include `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`, `phi3`. Examples (heap hints): TinyLlama Q4_K_M (~637 MB, `2g`), phi-3.5-mini Q4_K_M (~2.4 GB, `4g`), Mistral-7B Q4_K_M (~4.1 GB, `8g`), Llama-3.1-70B Q4_K_M distributed.

## Modules (overview)

| Module | Role |
|--------|------|
| `juno-bom` | Maven BOM — aligned versions for all `cab.ml` artifacts |
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

Apache 2.0 - see [LICENSE](LICENSE).