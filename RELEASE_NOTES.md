# Juno 0.1.0 — Release Notes

**Java Unified Neural Orchestration** — distributed LLM inference and fine-tuning in pure Java.

License: [Apache 2.0](LICENSE)

---

## Requirements

| Component | Version |
|-----------|---------|
| JDK | 25+ |
| Maven (build from source) | 3.9+ |
| NVIDIA GPU (optional) | CUDA 12.x + driver |
| AMD GPU (optional) | ROCm 6+ + driver |

CPU-only inference requires no GPU stack. The `./juno` launcher enforces JDK 25 at startup.

---

## Highlights

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
- Performance matrix: [docs/juno_test_matrix.html](docs/juno_test_matrix.html)

---

## Supported models

GGUF with LLaMA-compatible

Quantizations: F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

Chat templates: `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`.

---

## Quick start

```bash
mvn clean package -DskipTests

# Download a GGUF, then:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# With OpenAI-compatible API:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --api-port 8080
```

Full reference: [docs/howto.md](docs/howto.md)

---

## Known limitations (0.1.0)

- **Text only** — image or multimodal message content is not supported.
- **OpenAI `n > 1`** — rejected with HTTP 400; only single completions.
- **Partial OpenAI compatibility** — `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` are ignored for client compatibility.
- **No built-in auth or TLS** on the REST server; configure at the reverse proxy or network layer for production.
- **LoRA merge / redistribution** may trigger model-license obligations; see [docs/legal.md](docs/legal.md).
- **EU AI Act** — compliance-oriented features (AI disclosure, audit logging, auth) are not yet built in; see [docs/EU-AI-Act-compliance.md](docs/EU-AI-Act-compliance.md).

---

## Documentation map

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Overview and entry points |
| [docs/howto.md](docs/howto.md) | CLI, REST, JVM API, AWS, tests |
| [docs/arch.md](docs/arch.md) | Internal architecture |
| [docs/features.md](docs/features.md) | Feature summary |
| [docs/LoRA.md](docs/LoRA.md) | LoRA training and merge |
| [docs/performance.md](docs/performance.md) | Benchmark methodology |
| [docs/legal.md](docs/legal.md) | Model weights and merge Q&A |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |
| [api/src/main/resources/juno-api.yaml](api/src/main/resources/juno-api.yaml) | OpenAPI spec |

Developer session history: [CHANGELOG.md](CHANGELOG.md)

---

## Upgrade / migration

This is the first public release. No prior version migration path.

Artifacts publish under `cab.ml` at version `0.1.0` on Maven Central. Import the BOM:

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
