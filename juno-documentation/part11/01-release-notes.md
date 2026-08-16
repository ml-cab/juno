(ch-11-1)=
# 11.1. Release Notes

**Java Unified Neural Orchestration**: distributed LLM inference and fine-tuning in pure Java.

License: [Apache 2.0](https://github.com/ml-cab/juno/blob/main/LICENSE)

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

- **Pipeline parallel**: contiguous layer blocks across JVM nodes; activations flow serially over gRPC.
- **Tensor parallel**: full depth on each node with head/FFN slices; coordinator AllReduce on logits.
- Zero sidecar processes: coordinator (**juno-master**) and workers (**juno-node**) are shaded JVM jars.

### GPU acceleration

- **NVIDIA CUDA 12.x / cuBLAS** and **AMD ROCm 6+ / rocBLAS** via Panama FFI (`java.lang.foreign`).
- Auto-selection at startup: CUDA → ROCm → CPU. Override with `-Djuno.gpu.backend=cuda|rocm|auto`.
- Device-resident FP16 weights; automatic CPU quantised fallback on VRAM OOM.

### LoRA fine-tuning

- In-process training REPL: `./juno lora`
- Inference overlay: `--lora-play PATH` (local, cluster, AWS)
- Native merge to standalone GGUF: `./juno merge` (patched tensors stored as F32)
- Train-file scheduling: `--lora-chunk-tokens` (default 32; recommend 128 for files),
  `--lora-max-train-tokens` seeded corpus caps
- Train device: `--lora-train-device auto|gpu|cpu` (`gpu` fails closed if unavailable)
- Microbatch: `--lora-microbatch N` / `LORA_MICROBATCH` (default 8; `1` = FP16 sequential);
  VRAM OOM auto-retries FP16 then CPU under `auto`
- **GPU LoRA training** (LLaMA/Qwen2): resident FP32 forward/transpose + microbatched GEMM
  (default batch 8); adapters/Adam on host. See [Performance methodology](#ch-7-2).
- Multi-arch resident GPU transpose: LLaMA-family, Qwen2, Phi-3 (fused physical), dense Qwen3
  via shared `LoraResidentWeights` (FP32→FP16→CPU VRAM ladder under `auto`)

### OpenAI-compatible REST

- `POST /v1/chat/completions` (blocking + SSE)
- `GET /v1/models`, `GET /v1/models/{model}`
- Enable with `--api-port N` on `./juno local` or cluster mode
- Juno extensions: `x_juno_priority`, `x_juno_session_id`, `x_juno_top_k`

### JVM integration

- Maven BOM: `cab.ml:juno-bom:0.1.0`
- Facade API: `JunoPlayer`, `LoraTrainer`, `JunoHttpClient`
- See [Quickstart: JVM embedding](#ch-1-3)

### Observability

- Custom JFR events across matmul, forward pass, token generation, LoRA training
- Health dashboard with per-node CPU load, coordinator P99 latency, node throughput
- Performance matrix: [juno_test_matrix.html](https://github.com/ml-cab/juno/blob/main/docs/assets/juno_test_matrix.html)

---

## Supported models

GGUF with LLaMA-compatible architectures.

Quantizations: F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

Chat templates: `llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`, `phi3`. **`phi3`** (Phi-3 / Phi-3.5) is supported via a dedicated handler and template. **Gemma**, **Qwen 2, Qwen3, and Qwen3.5** (`gemma`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`) are **under development**: template and handler groundwork exists for some paths; end-to-end validation is in progress. Limitations for work in flight: no LoRA on Gemma/Qwen, no thinking-mode template, no fused QKV GGUFs on Qwen.

---

## Quick start

```bash
mvn clean package -DskipTests

# Download a GGUF, then:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# With OpenAI-compatible API:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --api-port 8080
```

Full reference: [docs/index.md](https://github.com/ml-cab/juno/blob/main/docs/index.md)

---

## Known limitations (0.1.0)

- **Text only**: image or multimodal message content is not supported.
- **OpenAI `n > 1`**: rejected with HTTP 400; only single completions.
- **Partial OpenAI compatibility**: `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` are ignored for client compatibility.
- **No built-in auth or TLS** on the REST server; configure at the reverse proxy or network layer for production.
- **LoRA merge / redistribution** may trigger model-license obligations; see [LoRA and merge licensing](#ch-9-3).
- **EU AI Act**: compliance-oriented features (AI disclosure, audit logging, auth) are not yet built in; see [EU AI Act compliance](#ch-9-7).

---

## Documentation map

| Document | Purpose |
|----------|---------|
| [README.md](https://github.com/ml-cab/juno/blob/main/README.md) | Overview and entry points |
| [docs/index.md](https://github.com/ml-cab/juno/blob/main/docs/index.md) | Full documentation table of contents |
| [Architecture overview](#ch-2-1) | Internal architecture |
| [README.md](https://github.com/ml-cab/juno/blob/main/README.md) | Feature summary |
| [LoRA concepts](#ch-4-1) | LoRA training and merge |
| [Performance methodology](#ch-7-2) | Benchmark methodology |
| [Legal and compliance](#ch-9-1) | Model weights and merge Q&A |
| [SECURITY.md](https://github.com/ml-cab/juno/blob/main/SECURITY.md) | Vulnerability reporting |
| [api/src/main/resources/juno-api.yaml](https://github.com/ml-cab/juno/blob/main/api/src/main/resources/juno-api.yaml) | OpenAPI spec |

Developer session history: [Changelog](#ch-11-2)

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

---

[<- 10.6 Contributors](#ch-10-6) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [11.2 Changelog ->](#ch-11-2)
