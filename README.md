# Juno

**Java Unified Neural Orchestration** 

Distributed LLM inference and fine-tuning. Pure Java - No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/GPU-ROCm%206%2B-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## 1. What is Juno

- Playing large and tiny language models;
- Providing distributed inference - Layer Sharding (pipeline parallelism) or Weight Slices (tensor parallelism) using pure Java; 0 sidecar processes;
- GPU acceleration supported — *NVIDIA CUDA/cuBLAS* and *AMD ROCm/rocBLAS*, both with FP16 resident weights; CPU fallback on OOM; auto-detected at startup;
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

for more info please refer to [Juno cookbook](https://github.com/ml-cab/juno-cookbook/tree/main)

Then simply use LocalInferencePipelineExample.java if you are going to use only current JVM, reading a model:

```

	private static LocalChat lc;

	@BeforeAll
	static void buildPipline() throws Exception {
		lc = LocalChat.builder(Path.of(MODEL_PATH)).nodeCount(1).useGpu(false)
				.samplingParams(SamplingParams.defaults().withMaxTokens(64).withTemperature(0.7f)).build();
	}

	@AfterAll
	static void closePipeline() {
		if (lc != null) {
			lc.close();
		}
	}

	@Test
	@Order(1)
	@DisplayName("single turn returns a non-empty reply")
	void singleTurnReturnsNonEmptyReply() {
		String reply = lc.chat("Hello, how are you?");
		assertThat(reply).isNotBlank();
	}

```
Then follow **[docs/howto.md](docs/howto.md)** `JVM integration` section.

### 2.2 Local player and LoRA (including Hugging Face–origin weights)

Contributors and enthusiasts can build from source: `mvn clean package -DskipTests`.

Download a GGUF (replace the URL with your chosen model):

```
cd juno/models
wget https://huggingface.co/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

then run local Juno interactive console to try and train inference

```
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**`--model-path`** is relative for juno project dir. REST alongside REPL: add **`--api-port 8080`**. 

Training: `./juno lora --model-path ...` (see **[docs/LoRA.md](docs/LoRA.md)**).

Optional **`./juno merge`** bakes a trained `.lora` into a new GGUF, so that inference needs no sidecar adapter 

More at **[howto.md](docs/howto.md)**.

### 2.3 On-prem orchestration

Run **`juno-master`** as the coordinator and **`juno-node`** on each worker with gRPC between them (systemd or your own process manager). Parallelism modes and byte-order flags match local cluster harness behaviour described in **[docs/howto.md](docs/howto.md)**; topology and components are in **[docs/arch.md](docs/arch.md)**. AWS automation under **`scripts/aws/`** is optional cloud packaging of the same roles.

## 3. Stack

Node coordination and inference RPCs use **gRPC** with **protobuf** contracts from the `api` module. GPU matmul is backed by **Panama FFI** (`java.lang.foreign`) against two vendor libraries:

- **NVIDIA:** CUDA 12.x + cuBLAS — `CudaBindings` resolves `libcudart.so.12` and `libcublas.so.12`; `CudaMatVec` owns all device memory and stream lifecycle.
- **AMD:** ROCm 6+ + rocBLAS — `RocmBindings` resolves `libamdhip64.so` and `librocblas.so`; `RocmMatVec` mirrors the same device-resident FP32/FP16 paths.

Backend is auto-selected at startup: CUDA first, then ROCm, then CPU. Override with `-Djuno.gpu.backend=cuda|rocm|auto`. A CPU quantised path is used when GPU is off or unavailable. The coordinator HTTP surface (**REST** and **SSE**) is implemented with **Javalin**.

## 4. Useful refs

- Performance matrix: **[docs/juno_test_matrix.html](https://ml.cab/juno_test_matrix.html)** - methodology companion **[docs/performance.md](docs/performance.md)**
- Legal Q&A: **[docs/legal.md](docs/legal.md)**

---

## Requirements

JDK 25+, Maven 3.9+. GPU nodes: CUDA 12.x + NVIDIA driver **or** ROCm 6+ + AMD driver (optional — CPU-only inference requires neither).

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
| `node` | Transformer handlers, GGUF, GPU matmul (CUDA + ROCm via Panama FFI) |
| `lora` | Adapter tensors, optimizer |
| `tokenizer`, `sampler`, `kvcache`, `health`, `metrics` | Shared infrastructure |
| `juno-player` | CLI REPL and cluster harness |
| `juno-node`, `juno-master` | Shaded deploy jars |

Details: **[docs/arch.md](docs/arch.md)**.

## License

Apache 2.0 - see [LICENSE](LICENSE).