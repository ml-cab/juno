# Juno

**Java Unified Neural Orchestration** 

Distributed LLM inference and fine-tuning. Pure Java - No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/GPU-ROCm%206%2B-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)


## 1. What is Juno

### Distributed inference

- **Pipeline parallel** — contiguous layer blocks across JVM nodes; activations flow serially over gRPC.
- **Tensor parallel** — full depth on each node with head/FFN slices; coordinator AllReduce on logits.
- Zero sidecar processes: coordinator (**juno-master**) and workers (**juno-node**) are shaded JVM jars.

### GPU acceleration

- **NVIDIA CUDA 12.x / cuBLAS** and **AMD ROCm 6+ / rocBLAS** via Panama FFI (`java.lang.foreign`).
- Auto-selection at startup: CUDA → ROCm → CPU. Override with `-Djuno.gpu.backend=cuda|rocm|auto`.
- Device-resident FP16 weights; automatic CPU quantised fallback on VRAM OOM.

### LoRA fine-tuning

- In-process training REPL: `./juno lora`
- Inference overlay: `--lora-play PATH` (local, cluster, AWS)
- Native merge to standalone GGUF: `./juno merge` (patched tensors stored as F32)

### OpenAI-compatible REST

- `POST /v1/chat/completions` (blocking + SSE)
- `GET /v1/models`, `GET /v1/models/{model}`
- Enable with `--api-port N` on `./juno local` or cluster mode
- Juno extensions: `x_juno_priority`, `x_juno_session_id`, `x_juno_top_k`

### JVM integration

- Maven BOM: `cab.ml:juno-bom:0.1.0`
- Facade API: `JunoPlayer`, `LoraTrainer`, `JunoHttpClient`
- See [docs/howto.md](docs/howto.md) JVM integration section

### Observability

- Custom JFR events across matmul, forward pass, token generation, LoRA training
- Health dashboard with per-node CPU load, coordinator P99 latency, node throughput
- Performance matrix: [docs/juno_test_matrix.html](https://ml.cab/juno_test_matrix.html)

Please see full feature list **[here](docs/features.md)**

## 2. How to use

### 2.1 JVM Integration

Integrate on `**cab.ml` artifacts at version `0.1.0`** from Maven Central:

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

Then simply use LocalChat.java if you are going to use only current JVM, reading a model:

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

Clone the repo:

```
git clone https://github.com/ml-cab/juno.git && cd juno
mvn clean package -DskipTests
```

Download a GGUF (replace the URL with your chosen model):

```
cd juno/models
wget https://huggingface.co/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

then run local Juno interactive console to try and train inference

**Linux / macOS:**
```
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Windows:**
```
juno.bat local --model-path models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

`--model-path` is relative to the juno project dir. REST alongside REPL: add `--api-port 8080`.

Training: `./juno lora --model-path ...` on Linux/macOS, `juno.bat lora --model-path ...` on Windows (see **[docs/LoRA.md](docs/LoRA.md)**).

Optional `./juno merge` (or `juno.bat merge` on Windows) bakes a trained `.lora` into a new GGUF, so that inference needs no sidecar adapter.

More at **[howto.md](docs/howto.md)**.

### 2.3 On-prem orchestration

Run `**juno-master`** as the coordinator and `**juno-node**` on each worker with gRPC between them (systemd or your own process manager). Parallelism modes and byte-order flags match local cluster harness behaviour described in **[docs/howto.md](docs/howto.md)**; topology and components are in **[docs/arch.md](docs/arch.md)**. AWS automation under `**scripts/aws/`** is optional cloud packaging of the same roles.

## 3. Stack

Node coordination and inference RPCs use **gRPC** with **protobuf** contracts from the `api` module. GPU matmul is backed by **Panama FFI** (`java.lang.foreign`) against two vendor libraries:

- **NVIDIA:** CUDA 12.x + cuBLAS — `CudaBindings` resolves `libcudart.so.12` and `libcublas.so.12`; `CudaMatVec` owns all device memory and stream lifecycle.
- **AMD:** ROCm 6+ + rocBLAS — `RocmBindings` resolves `libamdhip64.so` and `librocblas.so`; `RocmMatVec` mirrors the same device-resident FP32/FP16 paths.

Backend is auto-selected at startup: CUDA first, then ROCm, then CPU. Override with `-Djuno.gpu.backend=cuda|rocm|auto`. A CPU quantised path is used when GPU is off or unavailable. The coordinator HTTP surface (**REST** and **SSE**) is implemented with **Javalin**.

## 4. Useful refs

- Contributing and release workflow: **[docs/contributing.md](docs/contributing.md)**
- Release notes: **[RELEASE_NOTES.md](RELEASE_NOTES.md)**
- Security: **[SECURITY.md](SECURITY.md)**
- Performance matrix: **[docs/juno_test_matrix.html](https://ml.cab/juno_test_matrix.html)** - methodology companion **[docs/performance.md](docs/performance.md)**
- Legal Q&A: **[docs/legal.md](docs/legal.md)**

---

## Requirements

JDK 25+, Maven 3.9+. GPU nodes: CUDA 12.x + NVIDIA driver **or** ROCm 6+ + AMD driver (optional — CPU-only inference requires neither).

**Windows:** `juno.bat` at the project root delegates to `scripts\run.bat`. Requires JDK 25+ on `PATH` or `JAVA_HOME` set. CUDA GPU acceleration is supported (NVIDIA only — ROCm is Linux-only). All flags and environment overrides documented in [docs/howto.md](docs/howto.md) apply equally on Windows.

## Supported models

GGUF with LLaMA-compatible architectures (quantizations include F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). Chat templates: `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`, `phi3`. **`phi3`** (Phi-3 / Phi-3.5) is supported via a dedicated handler and template. **Gemma**, **Qwen 2, Qwen3, and Qwen3.5** (`gemma`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`) are **under development** — template and handler groundwork exists for some paths; end-to-end validation is in progress (no LoRA on Gemma/Qwen). Examples (heap hints): TinyLlama Q4_K_M (~637 MB, `2g`), Mistral-7B Q4_K_M (~4.1 GB, `8g`), Phi-3.5-mini Q4_K_M (~2.2 GB, `4g`), Llama-3.1-70B Q4_K_M distributed. 

## Modules (overview)

| Module                                                 | Role                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `juno-bom`                                             | Maven BOM — aligned versions for all `cab.ml` artifacts             |
| `api`                                                  | OpenAPI spec, protobuf/gRPC API                                     |
| `registry`                                             | Shard planning, model registry                                      |
| `coordinator`                                          | Scheduler, generation loop, REST                                    |
| `node`                                                 | Transformer handlers, GGUF, GPU matmul (CUDA + ROCm via Panama FFI) |
| `lora`                                                 | Adapter tensors, optimizer                                          |
| `tokenizer`, `sampler`, `kvcache`, `health`, `metrics` | Shared infrastructure                                               |
| `juno-player`                                          | CLI REPL and cluster harness                                        |
| `juno-node`, `juno-master`                             | Shaded deploy jars                                                  |


Details: **[docs/arch.md](docs/arch.md)**.

## License

Apache 2.0 - see [LICENSE](LICENSE).