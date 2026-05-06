## Juno — complete how-to reference

```
./juno
```

Unified launcher at the project root. Requires JDK 25+ and pre-built jars (`mvn clean package -DskipTests`).

---

### Commands

| Command | Description |
|---------|-------------|
| `local` | In-process REPL — all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single in-process JVM, adapter persisted to `.lora` file |
| `merge` | Bake a trained `.lora` adapter into a new standalone GGUF — no sidecar needed at inference time |
| *(no command)* | 3-node cluster — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail) |

---

### Flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | — | all | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format |
| `--byteOrder BE\|LE` | `BE` | cluster | Activation byte order. Must match across all JVMs — propagated automatically by `ClusterHarness` and `juno-deploy.sh`. |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response |
| `--temperature F` | `0.6` | all | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | all | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | all | Nucleus sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | all | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | local | Number of in-process shards |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type |
| `--jfr DURATION` | — | cluster, local, lora | Java Flight Recording (e.g. `30s`, `5m`) |
| `--verbose` / `-v` | — | cluster, local | Verbose logging |
| `--cpu` | — | cluster, local | Force CPU inference: sets `JUNO_USE_GPU=false`. Does not enable LoRA mode. |
| `--lora-play PATH` | — | cluster, local | Apply a pre-trained `.lora` adapter at inference (read-only, no training). In cluster mode the file is forwarded as `-Djuno.lora.play.path` to every forked node JVM. |
| `--api-port N` | — | cluster, local | Start the OpenAI-compatible REST API server on port N alongside the REPL. Exposes `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/models/{model}`. Environment override: `API_PORT`. |

**LoRA-specific flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α (effective scale = α/rank) |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` |
| `--lora-steps-qa N` | `10` | Gradient steps per `/train-qa` Q&A pair |
| `--lora-early-stop F` | `0.25` | Stop chunk early when loss delta < F |

**`merge` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output file (always plain GGUF, even if source is llamafile) |
| `--heap SIZE` | `4g` | JVM heap — use at least 2x the model file size |

**Environment overrides:** `MODEL_PATH`, `JUNO_USE_GPU`, `PTYPE`, `DTYPE`, `BYTE_ORDER`,
`MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`,
`LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `LORA_PLAY_PATH`, `API_PORT`

For the `lora` command and `ForwardPassHandlerLoader.selectLoraBackend()`, `JUNO_USE_GPU` unset
means try CUDA when a GPU is present. Set `JUNO_USE_GPU=false` or pass `--cpu` to force CPU.
Cluster and `local` modes use `selectBackend()`, where unset defaults to CPU for safety.

---

### `local` — in-process REPL

```bash
# Minimal
./juno local --model-path /path/to/model.gguf

# With OpenAI-compatible REST API on port 8080
./juno local --model-path /path/to/model.gguf --api-port 8080

# With a pre-trained LoRA adapter applied at inference
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Via env var
LORA_PLAY_PATH=/path/to/model.lora MODEL_PATH=/path/to/model.gguf ./juno local

# With JFR (metrics printed on exit)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Verbose
./juno local --model-path /path/to/model.gguf --verbose
```

When `--lora-play` is given, the startup banner shows:
```
  Loading LoRA adapters for inference: /path/to/model.lora
  Loaded 44 LoRA adapters  (inference-only, no training)
```

When `--api-port` is given, the startup banner shows:
```
  ✔ Local API server on http://localhost:8080 (OpenAI: /v1/chat/completions)
```

---

### `lora` — LoRA fine-tuning REPL

```bash
# Minimal -- auto-loads <model>.lora if it exists
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# With verbose tracing (recommended when debugging training)
./juno lora --model-path /path/to/model.gguf --verbose
```

For a full LoRA training guide, REPL commands, rank selection, and common pitfalls see
[LoRA.md](LoRA.md).

**Using a trained adapter outside `lora` mode:**
```bash
# Chat with adapter, no training REPL overhead
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# 3-node cluster with adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

**Profiling a slow training step:**
```bash
./juno lora --model-path /path/to/model.gguf --jfr 5m
# After exit, open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser -> juno.LoraTrainStep: forwardMs / backwardMs / optimizerMs / loss
```

---

### `merge` — bake a LoRA adapter into a standalone GGUF

Writes a new GGUF where LoRA-patched projection tensors (wq/wv on every layer) are stored as
F32 for full precision. All other tensors are copied verbatim in their original quantized
encoding. The resulting file loads with `./juno local` or `./juno` like any other model.

```bash
# Default: reads <model>.lora, writes <model>-merged.gguf
./juno merge --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Explicit paths
./juno merge --model-path /path/to/model.gguf \
             --lora-path  /adapters/my.lora   \
             --output     /path/to/merged.gguf

# Larger heap for big models (rule of thumb: 2x model file size)
./juno merge --model-path /path/to/Mistral-7B.gguf --heap 12g
```

The LoRA delta per element (~6x10^-4) is smaller than Q4_K quantization noise (~3x10^-3).
Re-quantizing the merged weights back to Q4_K would erase the training entirely. F32 storage
for the 44 patched tensors is the correct trade-off. For TinyLlama 1.1B Q4_K_M (667 MB), the
merged file is approximately 1 GB.

---

### *(no command)* — 3-node cluster (forked JVMs, real gRPC)

Forks 3 separate JVM node processes. Each node loads its own shard of the model.
Two distribution strategies are available via `--pType`:

- **`pipeline`** (default) — contiguous layer blocks, serial activation flow node-1 -> node-2 -> node-3
- **`tensor`** — every node holds all layers but only a horizontal weight slice; coordinator broadcasts
  tokens to all nodes in parallel and reduces partial logit vectors (AllReduce)

```bash
# Pipeline-parallel (default)
./juno --model-path /path/to/model.gguf

# With OpenAI-compatible REST API on port 8080
./juno --model-path /path/to/model.gguf --api-port 8080

# Tensor-parallel
./juno --pType tensor --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf PTYPE=tensor ./juno

# Activation dtype
./juno --model-path /path/to/model.gguf --dtype FLOAT16    # default
./juno --model-path /path/to/model.gguf --dtype FLOAT32    # lossless debug
./juno --model-path /path/to/model.gguf --dtype INT8       # max compression

# With JFR -- coordinator + all node JVMs instrumented; files merged on exit
./juno --model-path /path/to/model.gguf --jfr 5m

# With pre-trained adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Generation params
./juno --model-path /path/to/model.gguf --max-tokens 512 --temperature 0.3

# Verbose
./juno --model-path /path/to/model.gguf --verbose
```

When `--lora-play` is given, `ClusterHarness.withLoraPlay(path)` injects
`-Djuno.lora.play.path=PATH` into every forked node JVM. Each node loads the adapter before
building its `ForwardPassHandler`.

---

### OpenAI-compatible REST API (`--api-port`)

Pass `--api-port N` to any `local` or cluster invocation to start an OpenAI wire-compatible
REST server alongside the REPL. No changes are required to `GenerationLoop`, the scheduler, or
any node code — the API layer is a pure translation shim above `RequestScheduler`.

**Supported endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Blocking or SSE streaming completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/models/{model}` | Retrieve a single model |

**Quick verification:**

```bash
# Start local mode with API
./juno local --model-path /path/to/model.gguf --api-port 8080

# Blocking completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is Java?"}]
  }'

# Streaming completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'

# List models
curl http://localhost:8080/v1/models
```

**OpenAI SDK (Python):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Blocking
response = client.chat.completions.create(
    model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    messages=[{"role": "user", "content": "What is Java?"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Write a haiku."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**LangChain:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",
    model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
)
print(llm.invoke("What is Java?").content)
```

**Request field mapping:**

| OpenAI field | Juno internal | Notes |
|---|---|---|
| `model` | `modelId` | First loaded model if omitted |
| `messages[].role` | `ChatMessage.role` | `system` / `user` / `assistant` |
| `messages[].content` | `ChatMessage.content` | Text only; image content not supported |
| `temperature` | `SamplingParams.temperature` | 0.0–2.0; default 0.7 |
| `top_p` | `SamplingParams.topP` | 0.0–1.0; default 0.9 |
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 512 |
| `max_tokens` | `SamplingParams.maxTokens` | Deprecated alias; `max_completion_tokens` takes precedence |
| `frequency_penalty` | `SamplingParams.repetitionPenalty` | Mapped: `1 + max(0, fp/2)` |
| `stream` | route selection | `false` → blocking JSON; `true` → SSE |
| `n` | — | Only `1` accepted; other values → HTTP 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` | — | Silently ignored for client compatibility |

**Juno request extensions** (namespaced under `x_juno_*` to avoid OpenAI field conflicts):

| Field | Type | Default | Description |
|---|---|---|---|
| `x_juno_priority` | string | `NORMAL` | Scheduler priority: `HIGH` / `NORMAL` / `LOW` |
| `x_juno_session_id` | string | — | Stable session ID; enables KV-cache reuse across turns |
| `x_juno_top_k` | integer | `50` | Top-K sampling cutoff (0 = disabled) |

**Multi-turn conversation with KV-cache reuse:**

```python
SESSION_ID = "sess-my-conversation-001"

def chat(messages):
    return client.chat.completions.create(
        model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        messages=messages,
        extra_body={"x_juno_session_id": SESSION_ID},
    ).choices[0].message.content

history = []
for user_input in ["My name is Alice.", "What is my name?"]:
    history.append({"role": "user", "content": user_input})
    reply = chat(history)
    history.append({"role": "assistant", "content": reply})
    print(reply)
```

**Error responses** follow the OpenAI error envelope (`{"error": {"message": ..., "type": ..., "code": ...}}`):

| HTTP | `code` | Cause |
|------|--------|-------|
| 400 | `invalid_request` | Missing/empty messages, `n` > 1, or invalid body |
| 503 | `service_unavailable` | No model loaded or model not ready |
| 429 | `rate_limit_exceeded` | Scheduler queue full; `Retry-After` header set |
| 500 | `internal_error` | Unexpected inference error |

The full OpenAPI 3.0 specification is at `api/src/main/resources/juno-api.yaml`.

---

### AWS — cluster deployment (`juno-deploy.sh`)

```
./launcher.sh juno-deploy.sh setup      [options]
./launcher.sh juno-deploy.sh start
./launcher.sh juno-deploy.sh stop
./launcher.sh juno-deploy.sh teardown
./launcher.sh juno-deploy.sh status
./launcher.sh juno-deploy.sh scan-regions
```

**Setup options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--instance-type TYPE` | `g4dn.xlarge` | EC2 instance type |
| `--node-count N` | `3` | Number of inference nodes |
| `--coordinator node1\|separate` | `node1` | Co-located or separate coordinator |
| `--model-url URL` | TinyLlama Q4_K_M | Model to download during bootstrap |
| `--ptype pipeline\|tensor` | `pipeline` | Parallelism type |
| `--dtype FLOAT32\|FLOAT16` | `FLOAT16` | Activation wire format |
| `--jfr DURATION` | — | JFR on all JVMs (e.g. `5m`) |
| `--lora-play PATH` | — | Local path to a `.lora` file. Must be absolute or relative to working directory — resolved via `realpath`. The file is SCPed to every node after bootstrap. |

**GPU quota:** the script checks EC2 quota `L-DB2E81BA` before launching. If the quota in vCPUs
is less than `node-count x vCPUs-per-instance`, setup fails immediately with the shortfall and
a link to the Service Quotas console. It never silently reduces node count.

**CUDA on GPU instances:** pre-installed in the golden AMI by `make-ami.sh`. Node bootstrap
only runs `lspci` to detect the GPU and sets `JUNO_USE_GPU=true` — no DKMS compilation at boot.

**LoRA deploy flow:**

```bash
# Train locally
./juno lora --model-path /path/to/model.gguf
you > /train-qa What is my name? A: Dima
you > /save

# Deploy to AWS with adapter
cd scripts/aws
./launcher.sh juno-deploy.sh setup \
  --instance-type m7i-flex.large \
  --model-url https://huggingface.co/.../tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --lora-play /absolute/path/to/model.lora
```

After all nodes finish bootstrap and before starting the coordinator, `_scp_lora_to_nodes()`
stops each `juno-node.service` synchronously, SCPs the file to `/opt/juno/models/`, patches
`JUNO_LORA_PLAY_PATH` in `/etc/juno/node.env`, and restarts the service. The coordinator only
starts after all nodes are confirmed active.

**Expected coordinator log:**
```
INFO: LoRA inference overlay configured -- nodes will load:
      /opt/juno/models/tinyllama-1.1b-chat-v1.0-q4_k_m.lora
```

**Expected node log:**
```
INFO: Detected architecture: llama  backend=CpuMatVec  file=...  lora=44 adapters
```

---

### Diagnostics and tracing

Run any command with `--verbose` to enable `[TRACE]` output:

| Line | What it tells you |
|------|-------------------|
| `[TRACE] model type (chat template key) : tinyllama` | Whether the template matches the model |
| `[TRACE] formatted training text (repr)` | Exact token sequence sent to the model during training |
| `[TRACE] token count (excl. BOS): N` | How many tokens are in the training sequence |
| `[TRACE] step=N loss=F chunk=M/T ms=D` | Per-step loss during training |
| `[TRACE] inference model type: tinyllama` | Template key at inference — must match training |

If the template key at training and inference differ, the model will not recall trained facts.
Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`,
`phi-3`, `gemma`) to ensure `ChatModelType.fromPath()` detects it correctly.

---

### Metrics

```bash
# Automatic in local mode
./juno local --model-path /path/to/model.gguf --jfr 5m

# Manual extraction from any .jfr file
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json
```

The JSON report includes the following `juno.TokenProduced` fields derived from the coordinator
JFR file. These are the primary throughput metrics for performance comparison:

| Field | Description |
|-------|-------------|
| `juno.TokenProduced.count` | Total tokens delivered to clients in the recording window |
| `juno.TokenProduced.elapsed_seconds` | Wall-clock span from first to last delivered token |
| `juno.TokenProduced.tps` | Aggregate tokens per second (`count / elapsed_seconds`) |

AWS cluster JFR:
```bash
./launcher.sh juno-deploy.sh setup --jfr 2m ...
# Ctrl+C -> recordings collected from all nodes -> metrics printed -> instances stopped
```
---

### Build and Test

Requires JDK 25+ and Maven 3.9+.

```bash
mvn clean package -DskipTests          # build — produces shade jars

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master             # integration tests — forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT and TensorParallelClusterIT

mvn verify -pl juno-master -Pintegration -Dmodels=/path/to/models
                                       # ModelLiveRunnerIT — requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test (8 checks, exits 0/1)
```

**GPU tests** (requires CUDA 12.x and an NVIDIA GPU):

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl juno-master \
  --enable-native-access=ALL-UNNAMED
```