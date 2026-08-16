# Juno

**Java Unified Neural Orchestration**

Distributed LLM inference and fine-tuning. Pure Java. No Python, no GIL, no Spring.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Build-Maven%203.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/GPU-ROCm%206%2B-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

Full documentation: **[ml.cab/juno-documentation](https://ml.cab/juno-documentation)**

## 1. What is Juno

Juno runs distributed LLM inference and LoRA fine-tuning on the JVM with CUDA and ROCm GPU
acceleration via Panama FFI. No Python runtime, no sidecar processes.

| Capability | Documentation |
|---|---|
| Distributed inference: pipeline and tensor parallelism over gRPC | [2.2 Distributed Inference](https://ml.cab/juno-documentation/distributed-inference/) |
| GPU acceleration: CUDA 12.x + cuBLAS, ROCm 6+ + rocBLAS | [2.4 GPU Acceleration](https://ml.cab/juno-documentation/gpu-acceleration/) |
| LoRA fine-tuning, GPU training, DoRA, GGUF merge | [Part 4. LoRA Fine-Tuning](https://ml.cab/juno-documentation/concepts/) |
| OpenAI-compatible and Juno-native REST API | [5.2 OpenAI-Compatible API](https://ml.cab/juno-documentation/openai-compatible-api/) |
| JVM facade: `JunoPlayer`, `LoraTrainer`, `JunoHttpClient` | [4.6 Programmatic API](https://ml.cab/juno-documentation/programmatic-api/) |
| Observability: JFR events, per-node health dashboard | [7.1 JFR and Metrics](https://ml.cab/juno-documentation/jfr-and-metrics/) |

Supported models and quantizations: [1.4 Supported Models](https://ml.cab/juno-documentation/supported-models/).
Performance matrix: [ml.cab/juno_test_matrix.html](https://ml.cab/juno_test_matrix.html).

## 2. How to use

### 2.1 JVM Integration

Add the BOM from Maven Central at version `0.1.0`:

```xml
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

Single-JVM quickstart with `LocalChat`:

```java
lc = LocalChat.builder(Path.of(MODEL_PATH)).nodeCount(1).useGpu(false)
        .samplingParams(SamplingParams.defaults().withMaxTokens(64).withTemperature(0.7f)).build();

String reply = lc.chat("Hello, how are you?");
```

Full API and streaming examples: [1.3 Quickstart: JVM Embedding](https://ml.cab/juno-documentation/quickstart-jvm-embedding/)
and [4.6 Programmatic API](https://ml.cab/juno-documentation/programmatic-api/).
Cookbook: [juno-cookbook](https://github.com/ml-cab/juno-cookbook/tree/main).

### 2.2 Local player and LoRA

Build from source and run a local interactive console:

```bash
git clone https://github.com/ml-cab/juno.git && cd juno
mvn clean package -DskipTests

# Download a GGUF, then run the interactive console:
# Linux / macOS:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# Windows:
juno.bat local --model-path models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# REST alongside the REPL:
./juno local --model-path models/... --api-port 8080

# LoRA training:
./juno lora --model-path models/...

# Merge a trained adapter into a standalone GGUF:
./juno merge
```

See [1.2 Quickstart: Local](https://ml.cab/juno-documentation/quickstart-local/),
[Part 3. CLI Reference](https://ml.cab/juno-documentation/commands/),
[4.3 Training Guide](https://ml.cab/juno-documentation/training-guide/),
and [4.5 Merging Adapters](https://ml.cab/juno-documentation/merging-adapters/).

### 2.3 On-prem and cloud orchestration

Run `juno-master` as the coordinator and `juno-node` on each worker over gRPC. AWS automation
scripts are provided under `scripts/aws/`.

See [6.1 On-Prem Cluster](https://ml.cab/juno-documentation/on-prem-cluster/)
and [6.2 AWS Deployment](https://ml.cab/juno-documentation/aws-deployment/).

## 3. Modules

| Module | Role |
|---|---|
| `juno-bom` | Maven BOM. aligned versions for all `cab.ml` artifacts |
| `api` | OpenAPI spec, protobuf/gRPC contracts |
| `registry` | Shard planning, model registry |
| `coordinator` | Scheduler, generation loop, REST |
| `node` | Transformer handlers, GGUF, GPU matmul (CUDA + ROCm via Panama FFI) |
| `lora` | Adapter tensors, optimizer |
| `tokenizer`, `sampler`, `kvcache`, `health`, `metrics` | Shared infrastructure |
| `juno-player` | CLI REPL and cluster harness |
| `juno-node`, `juno-master` | Shaded deploy jars |

Architecture and design decisions: [Part 2. Architecture](https://ml.cab/juno-documentation/overview/).
Full module map: [2.6 Module Map](https://ml.cab/juno-documentation/module-map/).

## 4. Documentation

Full reference: **[ml.cab/juno-documentation](https://ml.cab/juno-documentation)**

| Topic | Page |
|---|---|
| Requirements | [1.1 Requirements](https://ml.cab/juno-documentation/requirements/) |
| Quickstart: Local | [1.2 Quickstart: Local](https://ml.cab/juno-documentation/quickstart-local/) |
| Quickstart: JVM | [1.3 Quickstart: JVM Embedding](https://ml.cab/juno-documentation/quickstart-jvm-embedding/) |
| Supported models | [1.4 Supported Models](https://ml.cab/juno-documentation/supported-models/) |
| CLI flags | [3.2 Flags](https://ml.cab/juno-documentation/flags/) |
| LoRA training | [4.3 Training Guide](https://ml.cab/juno-documentation/training-guide/) |
| REST API | [5.2 OpenAI-Compatible API](https://ml.cab/juno-documentation/openai-compatible-api/) |
| On-prem cluster | [6.1 On-Prem Cluster](https://ml.cab/juno-documentation/on-prem-cluster/) |
| AWS deployment | [6.2 AWS Deployment](https://ml.cab/juno-documentation/aws-deployment/) |
| Windows | [6.3 Windows](https://ml.cab/juno-documentation/windows/) |
| Performance | [7.2 Performance Methodology](https://ml.cab/juno-documentation/performance-methodology/) |
| Contributing | [10.1 Contributing](https://ml.cab/juno-documentation/contributing/) |
| Legal | [9.1 License and Patents](https://ml.cab/juno-documentation/license-and-patents/) |
| Security | [10.3 Security Policy](https://ml.cab/juno-documentation/security-policy/) |
| Release notes | [11.1 Release Notes](https://ml.cab/juno-documentation/release-notes/) |
| Changelog | [11.2 Changelog](https://ml.cab/juno-documentation/changelog/) |

---

## Requirements

JDK 25+, Maven 3.9+. GPU nodes: CUDA 12.x + NVIDIA driver or ROCm 6+ + AMD driver. CPU-only
inference requires neither.

Windows: `juno.bat` at the project root requires JDK 25+ on `PATH` or `JAVA_HOME` set. See
[6.3 Windows](https://ml.cab/juno-documentation/windows/).

Full requirements: [1.1 Requirements](https://ml.cab/juno-documentation/requirements/).

## License

Apache 2.0. See [LICENSE](LICENSE).