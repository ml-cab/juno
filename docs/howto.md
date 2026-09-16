## Juno — complete how-to reference

**Documentation map:** [README.md](../README.md) (overview), [arch.md](arch.md), [LoRA.md](LoRA.md), [performance.md](performance.md), [legal.md](legal.md), [juno_test_matrix.html](https://ml.cab/juno_test_matrix.html), [features.md](features.md).

**Linux / macOS:**
```
./juno
```

**Windows:**
```
juno.bat
```

Unified stand-alone launchers at the project root. `juno.bat` delegates to `scripts\run.bat`. Requires JDK 25+ and pre-built jars (`mvn clean package -DskipTests`).

> **Windows note:** All examples below use `./juno`. Replace with `juno.bat` on Windows and use backslashes for paths (e.g. `--model-path models\model.gguf`). All flags, environment variables, and subcommands are identical across platforms.

---

### Commands

| Command | Description |
|---------|-------------|
| `cluster` | 3-node cluster (default command) — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `local` | In-process REPL — all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single in-process JVM, adapter persisted to `.lora` file |
| `merge` | Bake a trained `.lora` adapter into a new standalone GGUF — no sidecar needed at inference time |
| `gguf-info` | Dump a GGUF's full metadata + tensor layout (name/shape/quant type) as plain text — for architecture review without guessing |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail) |

---

### Flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | — | all | Path to GGUF file (required unless `--hf` is given) |
| `--hf REPO[:QUANT]` | — | cluster, local, lora | Resolve a Hugging Face Hub repo into a local GGUF path and use it as the effective model path — downloads (with resume + ETag caching) if not already cached. `QUANT` defaults to `Q4_K_M` when present in the repo, else the first `.gguf` asset. Combined with `--model-path`, the two must resolve to the same file or startup fails with an error. Env override: `JUNO_HF`. See "Chat templates and Hugging Face downloads" below. |
| `--mmproj-path PATH` | — | local | Path to a separate mmproj GGUF holding the CLIP vision encoder. Required for `/v1/vision/chat` to be registered — real LLaVA/Qwen-VL/SmolVLM GGUF releases keep the vision encoder in a file separate from the base LLM; see `assets/Vision-I2T.md`. Environment override: `MMPROJ_PATH`. |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format |
| `--byteOrder BE\|LE` | `BE` | cluster | Activation byte order. Must match across all JVMs — propagated automatically by `ClusterHarness` and `juno-deploy.sh`. |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response. Same default as REST API and `SamplingParams.defaults()`. |
| `--temperature F` | `0.7` | all | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `50` | all | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.9` | all | Nucleus sampling cutoff (0 = disabled). Same default as REST API and `SamplingParams.defaults()`. |
| `--heap SIZE` | `4g` | all | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | local | Number of in-process shards |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type |
| `--jfr DURATION` | — | cluster, local, lora | Java Flight Recording (e.g. `30s`, `5m`) |
| `--verbose` / `-v` | — | cluster, local | Verbose logging |
| `--cpu` | — | cluster, local | Force CPU inference: sets `JUNO_USE_GPU=false`. Does not enable LoRA mode. |
| `--gpu-layers N\|all\|auto` | `all` | cluster, local | Transformer layers resident on GPU (`JUNO_GPU_LAYERS`). `auto` fits until VRAM OOM. |
| `--mmq on\|off\|auto` | `off` | cluster, local | Packed Q4_K GPU weights (`JUNO_MMQ`) — keeps Q4_K on device instead of FP16-resident dequant. On CUDA this is a **VRAM-fit** path and a **measured decode-throughput win** vs `--mmq off` (see `docs/performance.md`). Applies to Llama-family, Phi-3, and Qwen3 dense text inference, and to `--lora-play` when the CUDA kernel loads (local REPL prints a confirmation line). LoRA **training** ignores `--mmq` (frozen weights stay FP16/FP32; the train REPL warns). `auto` enables when CUDA + kernel load. Default **off**. |
| `--gpu-attention on\|off\|auto` | `off` | cluster, local | GPU-resident attention kernel (`JUNO_GPU_ATTENTION`) — moves QK^T + softmax + weighted-V-sum onto a device-resident FP16 KV mirror instead of scalar CPU Java. CUDA only. A **measured decode/prefill throughput lever** (see `docs/performance.md`), not a peer-latency claim. Applies to Llama-family, Mistral, and Qwen2 dense text inference (`LlamaTransformerHandler`), and to vision automatically (delegates to the same handler). Phi-2, Phi-3, Qwen3, and Qwen3-MoE keep the scalar CPU path (**follow-up**). LoRA **training** and `--lora-play` ignore `--gpu-attention` (attention stays scalar CPU; the train REPL warns). `auto` enables when CUDA is available. Occasional multi-token greedy-decode divergence from FP16 KV rounding is expected and documented, same class of tradeoff as other reduced-precision paths in this codebase. Default **off**. |
| `--cache-type-k f16\|q8_0` | `f16` | cluster, local, lora | K-cache element type (`JUNO_CACHE_TYPE_K`). `f16` is the current float32 path (bit-compatible default). `q8_0` packs keys (~3.8× smaller persistent KV vs float); attention dequants to float. Slight quality tradeoff vs `f16`. On `juno lora` **training**, teacher-forced forward keeps ephemeral float KV (startup WARNING); q8 applies to inference / `--lora-play` maps. |
| `--cache-type-v f16\|q8_0` | `f16` | cluster, local, lora | V-cache element type (`JUNO_CACHE_TYPE_V`). Same semantics as `--cache-type-k`. |
| `--schedule static\|continuous` | `static` | cluster, local, lora | Serving schedule (`JUNO_SCHEDULE`). `static` = dense KV + static micro-batch (SSE isolated). `continuous` = paged KV + running-set batching (SSE shares steps) on **local / in-process only**. Under continuous, long prompts advance in `--prefill-batch` ubatch chunks mixed into the same engine steps as other requests’ decode; when the step slot budget is full, **decode is preferred** so short replies are not starved. Cluster, TP, and PP launchers **auto-fallback to static** with a startup WARNING. On `juno lora` **training**, continuous is an explicit no-op (ephemeral float KV + REPL WARNING; no running-set engine). Per-request `x_juno_loras` is rejected under continuous (use process-wide `--lora-play`). Prefix KV reuse is **session-scoped** (`x_juno_session_id`); shared system prompts across different sessions do not skip prefill. Tools / LoRA play that change the tokenized prefix invalidate cross-turn hits. `--parallel` caps the continuous running set (default cap 8 when parallel is 1). |
| `--kv-page-size N` | `16` | cluster, local, lora | Tokens per KV page when `schedule=continuous` (`JUNO_KV_PAGE_SIZE`). Ignored under `static` (dense; startup note). |
| `--parallel N` | `1` | cluster, local, master | Static micro-batch size (`JUNO_PARALLEL`). `1` disables batching; recommend `8` for API servers. |
| `--batch-window-ms M` | `50` when parallel>1 | cluster, local, master | Batch collect window (`JUNO_BATCH_WINDOW_MS`). |
| `--prefill-batch N` | `32` | cluster, local, master | Max prompt tokens per prefill GPU window (`JUNO_PREFILL_BATCH`). Use `1` for per-token batched prefill. |
| `--prefill single\|batched` | `batched` | cluster, local | Prefill strategy: windowed GEMM vs per-token sequential loop. |
| `--lora-play PATH` | — | cluster, local | Apply a pre-trained `.lora` adapter at inference (read-only, no training). In cluster mode the file is forwarded as `-Djuno.lora.play.path` to every forked node JVM. |
| `--api-port N` | — | cluster, local | Start the OpenAI-compatible REST API server on port N alongside the REPL. Exposes `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/models/{model}`. Environment override: `API_PORT`. |
| `--grammar-file PATH` | — | cluster, local | GBNF file for constrained decoding (REPL and `--api-port`). Env `JUNO_GRAMMAR_FILE`. Mutually exclusive with `--json-schema-file`. Applies to chat completions when the request omits a grammar (`response_format.type=text` stays unconstrained). LoRA **training** ignores the flag (launcher WARNING). |
| `--json-schema-file PATH` | — | cluster, local | JSON Schema file compiled to GBNF (documented subset below). Env `JUNO_JSON_SCHEMA_FILE`. Unsupported keywords fail closed. |

**LoRA specific flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α (effective scale = α/rank) |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` |
| `--lora-steps-qa N` | `10` | Gradient steps per `/train-qa` Q&A pair |
| `--lora-early-stop F` | `0.25` | Stop chunk early when loss delta < F |

**`merge` specific flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output file (always plain GGUF, even if source is llamafile) |
| `--heap SIZE` | `4g` | JVM heap — use at least 2x the model file size |

**Environment overrides:** `MODEL_PATH`, `JUNO_HF`, `JUNO_USE_GPU`, `JUNO_GPU_LAYERS`, `JUNO_MMQ`, `JUNO_CACHE_TYPE_K`, `JUNO_CACHE_TYPE_V`, `JUNO_SCHEDULE`, `JUNO_KV_PAGE_SIZE`, `JUNO_PARALLEL`, `JUNO_BATCH_WINDOW_MS`, `PTYPE`, `DTYPE`, `BYTE_ORDER`,
`MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`,
`LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `LORA_PLAY_PATH`, `API_PORT`

---

### Chat templates and Hugging Face downloads

**Template resolution precedence.** Every text-inference request is formatted into a model-ready
prompt before tokenization. Juno picks the template in this order:

1. **GGUF-embedded template** — when the loaded GGUF carries a `tokenizer.chat_template` metadata
   string, Juno parses and renders it with a restricted Jinja-subset engine (messages loop,
   role/content access, `if`/`elif`/`else`, `and`/`or`/`not`, string concatenation, the `trim`
   filter, `bos_token`/`eos_token` substitution, and Jinja's default whitespace control around
   block tags so templates that rely on that default — not just ones that spell out `{%-`/`-%}`
   explicitly — render without stray blank lines). This covers real-world Llama-3-, Phi-3-, and
   ChatML-style templates.
2. **Named template fallback** — if the metadata is absent, fails to parse, or fails a validation
   render, Juno falls back to the existing named template lookup (model-family detection from the
   file path or model id; ChatML is the default for unrecognized names). This is silent and
   automatic — generation is never blocked or corrupted by an unusable embedded template.

Supported named templates: Llama 3, ChatML (including Qwen2 / Qwen2.5), Qwen3, Phi-3, Mistral,
Gemma, TinyLlama, and a vision (moondream) template.

Template resolution is **wired on the base text-inference surface** (the `local` and `cluster`
REPLs and their `--api-port` REST/OpenAI servers) for both CUDA and ROCm backends. `juno lora`
train and `--lora-play` intentionally keep using the named template only, even when the loaded
GGUF has an embedded one — train-time and inference-time formatting must stay identical for
adapter recall; the LoRA REPL prints a one-line notice when it detects (and ignores) an embedded
template.

**`--hf` downloads.** `--hf org/repo[:quant]` resolves a Hugging Face Hub repository to a local
GGUF and uses it as the effective model path:

```bash
# Default quant preference (Q4_K_M when present, else the first .gguf asset)
./juno local --hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

# Explicit quant
./juno local --hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q8_0

# Works on cluster and lora too
./juno cluster --hf org/repo:Q4_K_M
./juno lora --hf org/repo
```

- Downloads go to `~/.cache/juno/models/<org>_<repo>/` — **not** the repo's local `models/`
  directory, which stays reserved for manually placed test fixtures.
- Downloads resume (HTTP `Range`) if interrupted, and re-resolve without re-downloading when the
  cached file's ETag still matches the remote asset.
- Passing both `--hf` and `--model-path` is allowed only when they resolve to the same file;
  otherwise startup fails with an explicit error rather than silently preferring one.
- No Python dependency anywhere in this path — the fetcher is a plain JDK `HttpClient` consumer
  talking to the public Hugging Face Hub HTTP API.

For the `lora` command and `ForwardPassHandlerLoader.selectLoraBackend()`, `JUNO_USE_GPU` unset
means try GPU (CUDA first, then ROCm) when available. Set `JUNO_USE_GPU=false` or pass `--cpu`
to force CPU. Cluster and `local` modes use `selectBackend()`, where unset defaults to CPU for
safety. Override the vendor with `-Djuno.gpu.backend=cuda|rocm|auto` (default: `auto`).

---

### `local` — in-process REPL, fastest mode of juno-player console, operates within same JVM, GRPC off, uses LocalInferencePipeline.java instead

```bash
# Minimal
./juno local --model-path /path/to/model.gguf

# With OpenAI-compatible REST API on port 8080
./juno local --model-path /path/to/model.gguf --api-port 8080

# API server with static micro-batching (non-stream requests only)
./juno local --model-path /path/to/model.gguf --api-port 8080 --parallel 8 --batch-window-ms 50

# With a pre-trained LoRA adapter applied at inference
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Via env var
LORA_PLAY_PATH=/path/to/model.lora MODEL_PATH=/path/to/model.gguf ./juno local

# With JFR (metrics printed on exit)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Verbose
./juno local --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**
```bat
juno.bat local --model-path models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

juno.bat local --model-path models\model.gguf --api-port 8080

juno.bat local --model-path models\model.gguf --lora-play adapters\model.lora

rem Via environment variable
set MODEL_PATH=C:\models\model.gguf
juno.bat local

juno.bat local --model-path models\model.gguf --jfr 5m

juno.bat local --model-path models\llava-v1.5-7b-Q4_K_M.gguf --mmproj-path models\mmproj-model-f16.gguf --nodes 1 --api-port 8081
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

### `cluster` — 3-node cluster, default command of juno-player console (forked JVMs, real gRPC)

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

# With JFR — coordinator + each node JVM writes its own .jfr file; metrics extracted per file on exit
./juno --model-path /path/to/model.gguf --jfr 5m

# With pre-trained adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Generation params
./juno --model-path /path/to/model.gguf --max-tokens 512 --temperature 0.3

# Verbose
./juno --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**
```bat
juno.bat --model-path models\model.gguf

juno.bat --model-path models\model.gguf --api-port 8080

juno.bat --pType tensor --model-path models\model.gguf

rem Via environment variable
set MODEL_PATH=C:\models\model.gguf
set PTYPE=tensor
juno.bat

juno.bat --model-path models\model.gguf --jfr 5m

juno.bat --model-path models\model.gguf --lora-play adapters\model.lora

juno.bat --model-path models\model.gguf --max-tokens 512 --temperature 0.3
```

When `--lora-play` is given, `ClusterHarness.withLoraPlay(path)` injects
`-Djuno.lora.play.path=PATH` into every forked node JVM. Each node loads the adapter before
building its `ForwardPassHandler`.

---

### `lora` — LoRA fine-tuning REPL

```bash
# Minimal -- auto-loads <model>.lora if it exists
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# With verbose tracing (recommended when debugging training)
./juno lora --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**
```bat
juno.bat lora --model-path models\TinyLlama.Q4_K_M.gguf

juno.bat lora --model-path models\model.gguf --verbose
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

**Windows:**
```bat
juno.bat local --model-path models\model.gguf --lora-play adapters\model.lora
juno.bat --model-path models\model.gguf --lora-play adapters\model.lora
```

**Profiling a slow training step:**

```bash
./juno lora --model-path /path/to/model.gguf --jfr 5m
# After exit, open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser -> juno.LoraTrainStep: forwardMs / backwardMs / optimizerMs / loss
```

**Windows:**
```bat
juno.bat lora --model-path models\model.gguf --jfr 5m
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

**Windows (Command Prompt):**
```bat
juno.bat merge --model-path models\TinyLlama.Q4_K_M.gguf

juno.bat merge --model-path models\model.gguf ^
               --lora-path adapters\my.lora ^
               --output merged\merged.gguf

juno.bat merge --model-path models\Mistral-7B.gguf --heap 12g
```

The LoRA delta per element (~6x10^-4) is smaller than Q4_K quantization noise (~3x10^-3).
Re-quantizing the merged weights back to Q4_K would erase the training entirely. F32 storage
for the 44 patched tensors is the correct trade-off. For TinyLlama 1.1B Q4_K_M (667 MB), the
merged file is approximately 1 GB.

---

### `gguf-info` — dump a GGUF's full metadata and tensor layout

Prints every metadata key/value (alphabetical) and every tensor's name, shape, and
quantization type (declaration order) as plain text. Read-only; does not load tensor data,
so it's fast even on large files.

Use this instead of guessing a model's architecture from a Hugging Face model card or from
partial log lines — for I2T (image-to-text) models in particular, the mmproj file's actual
tensor names (e.g. whether `mm.2.weight` exists at all, and its real shape) are ground truth
that no amount of reading the base architecture's paper or README can substitute for.

```bash
./juno gguf-info --model-path /path/to/llava-v1.5-7b-Q4_K.gguf \
                  --mmproj-path /path/to/llava-v1.5-7b-mmproj-Q4_0.gguf

# Positional args also work
./juno gguf-info /path/to/model.gguf /path/to/mmproj.gguf
```

Linux/macOS only for now — `scripts/run.bat` does not currently wire up `gguf-info` (it only
implements `cluster`/`local`/`lora`/`test`; note `merge`, just above, has the same pre-existing
gap despite the Windows example below it).

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

**Request field mapping:**

| OpenAI field | Juno internal | Notes |
|---|---|---|
| `model` | `modelId` | First loaded model if omitted |
| `messages[].role` | `ChatMessage.role` | `system` / `user` / `assistant` / `tool` |
| `messages[].content` | `ChatMessage.content` | Text only; image content not supported. Assistant `content` may be null when `tool_calls` is set. |
| `messages[].tool_calls` | replayed into the prompt | Prior assistant function calls (`<tool_call>` blocks) |
| `messages[].tool_call_id` | — | Accepted on `role=tool` (result text is wrapped as `<tool_response>`) |
| `temperature` | `SamplingParams.temperature` | 0.0–2.0; default 0.7 |
| `top_p` | `SamplingParams.topP` | 0.0–1.0; default 0.9 |
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 200 |
| `max_tokens` | `SamplingParams.maxTokens` | Deprecated alias; `max_completion_tokens` takes precedence |
| `frequency_penalty` | `SamplingParams.repetitionPenalty` | Mapped: `1 + max(0, fp/2)` |
| `stream` | route selection | `false` → blocking JSON; `true` → SSE |
| `n` | — | Only `1` accepted; other values → HTTP 400 |
| `stop` | `SamplingParams.stopStrings` (+ single-token ids) | String or ≤4 strings; halts decode; `finish_reason=stop` |
| `seed` | `SamplingParams.seed` | Deterministic stochastic sampling when set |
| `presence_penalty` | `SamplingParams.presencePenalty` | OpenAI-style (−2..2); subtracts from seen-token logits |
| `response_format` | `SamplingParams.grammar` | Absent / `type=text` = unconstrained. `json_object` and `json_schema` mask illegal tokens. Unsupported schema keywords → HTTP 400. Cannot combine with `tools`. |
| `tools` | `ToolPrompt` + `ToolCallParser` | OpenAI function tools. Templates: **llama3**, **chatml** (Qwen2 / Qwen2.5), **qwen3**. Other templates → HTTP 400. |
| `tool_choice` | same | `none` (never emit `tool_calls`) / `auto` (default when `tools` is set) / `required` / `{type:function, function:{name}}`. `required` and named choice attach a GBNF envelope. |
| `logit_bias`, `user` | — | Silently ignored for client compatibility |

**Juno request extensions** (namespaced under `x_juno_*` to avoid OpenAI field conflicts):

| Field | Type | Default | Description |
|---|---|---|---|
| `x_juno_priority` | string | `NORMAL` | Scheduler priority: `HIGH` / `NORMAL` / `LOW` |
| `x_juno_session_id` | string | — | Stable session ID; enables KV-cache reuse across turns |
| `x_juno_top_k` | integer | `50` | Top-K sampling cutoff (0 = disabled) |
| `x_juno_grammar` | string | — | Raw GBNF. Cannot be combined with `response_format` `json_object` / `json_schema` or with `tools`. |

**Constrained decoding.** Grammar is applied **before** temperature / top-k / top-p. When no grammar is set, sampling is unchanged. Sample files: [`docs/grammars/yes-no.gbnf`](grammars/yes-no.gbnf), [`docs/grammars/json-object.gbnf`](grammars/json-object.gbnf).

```
./juno local --model-path MODEL.gguf --grammar-file docs/grammars/yes-no.gbnf
./juno local --model-path MODEL.gguf --json-schema-file schema.json --api-port 8080
```

JSON Schema subset (v1): `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, `enum`, `const`, `required`, nested objects/arrays. Object keys are emitted in schema key order (permutations are not generated). Rejected (HTTP 400 / CLI error): `$ref`, `oneOf` / `anyOf` / `allOf`, `pattern`, `format`, numeric/length/item bounds, `additionalProperties` schemas, type unions. `response_format.type=json_object` uses a generic object grammar. Fixture eval: 20 schemas, ≥95% parseable JSON under an adversarial logit prior (`GrammarEvalTest`).

**Function calling.** `POST /v1/chat/completions` accepts OpenAI `tools` and `tool_choice`. Juno injects tool schemas into the chat prompt and parses `<tool_call>{"name","arguments"}</tool_call>` (or a raw JSON object) into `message.tool_calls`. The engine does **not** execute tools — the client runs them and continues with `role: tool` messages. `tool_choice=none` never returns `tool_calls`. `required` / named choice constrain decode with the JSON Schema subset. Combining `tools` with `json_object` / `json_schema` / `x_juno_grammar` / `--grammar-file` returns HTTP 400. `/v1/vision/chat` does not honor `tools`.

Supported chat templates: Llama 3, ChatML (including Qwen2 / Qwen2.5), Qwen3. Phi-3, Mistral, Gemma, TinyLlama, and vision (moondream) templates fail closed.

When `tools` is set and `tool_choice` is not `none`, SSE waits until generation finishes, then emits either `delta.tool_calls` or the full content.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "tool_choice": "required",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } },
          "required": ["city"]
        }
      }
    }],
    "messages": [{"role": "user", "content": "Weather in Boston?"}]
  }'
```

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

**Additional JVM-local endpoints** (same server as above):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/inference` | Blocking JSON completion (`InferenceApiServer` native shape) |
| `POST` | `/v1/inference/stream` | SSE stream; each `data:` line is JSON `{"token":"…","isComplete":false}` until terminal event |

---

### Embeddings API (`--embeddings`, `--pooling`)

`POST /v1/embeddings` is OpenAI wire-compatible but **disabled by default** — pass `--embeddings`
to the same `local` or `cluster` invocation that starts `--api-port` to turn it on. A server
started without `--embeddings` returns HTTP 400 on the route rather than silently ignoring the
request or 404ing.

Juno has no dedicated embedding-model GGUFs yet, so embeddings are extracted from whatever chat
model is loaded: the RMS/LayerNorm-normalized hidden state immediately before the LM head, at
every prompt position, reduced to one vector by `--pooling`:

| Pooling | Behavior | When to use |
|---|---|---|
| `mean` (default) | Average hidden vector across every prompt position | Best default for chat-tuned models — spreads representation across the whole prompt |
| `cls` | Hidden vector at the first prompt position | Rarely useful outside encoder-style (BERT-like) checkpoints, which Juno does not load |
| `last` | Hidden vector at the final prompt position | Cheapest; tends to under-represent earlier tokens on chat-tuned models — the server logs a one-time warning when used |

**Quick verification:**

```bash
./juno local --model-path /path/to/model.gguf --api-port 8080 --embeddings

curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Java?"}'

# Batch input — one embedding object per string, same order as the request
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["first sentence", "second sentence"], "x_juno_pooling": "cls"}'
```

Response shape matches OpenAI's `POST /v1/embeddings`:

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, ...]}],
  "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

`x_juno_pooling` (optional, `mean` / `cls` / `last`) overrides `--pooling` for a single request.

**Concurrency model (v1):** unlike `/v1/chat/completions`, embeddings requests run directly on
the pipeline on the HTTP request's own thread — they do not go through `RequestScheduler`'s
queue-depth limit or 429 semantics. There is no continuous-batching or micro-batching path for
embeddings in v1: `--prefill-batch` chunking (chat completions prefill) does not apply here —
each prompt position runs its own forward pass, so long inputs to `/v1/embeddings` are
proportionally slower than an equivalent-length chat prompt. `--parallel` static batching is
likewise chat-completions-only; embeddings batch input (multiple `input` strings) is processed
one string at a time, not as one micro-batch.

**Scope (v1):** local / single-shard only, matching `--schedule continuous`'s scope. `cluster`
mode accepts `--embeddings` (so scripted launches do not need per-mode branching) but every
request fails closed with HTTP 400 — tensor-parallel and pipeline-parallel node pipelines do not
implement embeddings extraction. `--lora-play` and vision are otherwise unaffected: chat
completions keep working normally with `--embeddings` on, and embeddings themselves reflect the
loaded LoRA overlay (they run through the same handler chain). `juno lora` (train REPL) accepts
`--embeddings` without error but the flag has no effect there — that mode's API server only ever
serves `/v1/lora/train-file-qa` and `/v1/lora/save`; the REPL prints a startup warning rather than
silently ignoring the flag.

---

### JVM integration — BOM, `JunoPlayer` facade, LoRA, embeddings, `Flow`, HTTP client

#### Maven BOM (`juno-bom`)

Import one POM so every `cab.ml` module shares the same version:

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

<dependencies>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>juno-player</artifactId>
    <!-- version comes from juno-bom -->
  </dependency>
</dependencies>
```

#### Runnable jar versus library jar

After `mvn package`, `juno-player/target/` contains:

- `juno-player-0.1.0.jar` — normal thin classpath artifact for dependents (compose with BOM-managed modules).
- `juno-player-0.1.0-shaded.jar` — fat jar with `Main-Class: cab.ml.juno.player.ConsoleMain`. The `./juno` launcher selects this shaded jar when present.

#### In-process facade (`JunoPlayer`)

Loads the GGUF, builds an in-process `LocalInferencePipeline`, `GenerationLoop`, and `RequestScheduler` (same wiring as `./juno local`):

```java
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

try (JunoPlayer player = JunoPlayer.builder(Path.of("/path/to/model.gguf"))
        .nodeCount(3)
        .useGpu(true)
        .samplingParams(SamplingParams.defaults().withMaxTokens(128).withTemperature(0.7f))
        .build()) {

    var messages = List.of(ChatMessage.user("Explain JDK virtual threads in one sentence."));
    var result = player.chat(messages);
    System.out.println(result.text());

    Flow.Publisher<String> pieces = player.streamPublisher(messages);
    pieces.subscribe(new Flow.Subscriber<>() {
        Flow.Subscription s;
        public void onSubscribe(Flow.Subscription s) {
            this.s = s;
            s.request(Long.MAX_VALUE);
        }
        public void onNext(String t) {
            System.out.print(t);
        }
        public void onError(Throwable e) {
            e.printStackTrace();
        }
        public void onComplete() {
            System.out.println();
        }
    });

    float[] vec = player.embed(messages); // length = model hidden dim (last RMS hidden before LM head)

    // Optional OpenAI-compatible REST server on port 8080:
    var api = player.startApiServer(8080);
    Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(api::stop));
}
```

#### Programmatic LoRA (`LoraTrainer`)

Same single-shard layout as `./juno lora`; train from code then `save()`:

```java
import java.nio.file.Path;

import cab.ml.juno.player.ChatModelType;
import cab.ml.juno.player.LoraTrainer;

Path model = Path.of("/path/to/model.gguf");
Path adapter = Path.of("/path/to/model.lora");

try (var trainer = LoraTrainer.open(model, adapter, /*rank*/ 8, /*alpha*/ 8f, /*lr*/ 1e-4)) {
    float loss = trainer.trainRawText("Some prose to adapt style.", /*stepsPerChunk*/ 50, /*chunkTokens*/ 32);
    String modelKey = ChatModelType.fromPath(model.toString());
    trainer.trainQaPair("What is my favorite color?", "Blue.", modelKey, /*stepsPerChunk*/ 10);
    trainer.save();
}
```

For REPL semantics, flags, and pitfalls see [LoRA.md](LoRA.md).

#### `Flow.Publisher` from `TokenConsumer` (`PublisherTokenConsumer`)

For custom scheduling (not using `JunoPlayer.streamPublisher`), wrap any `RequestScheduler` submission:

```java
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PublisherTokenConsumer;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

void stream(RequestScheduler scheduler, String modelId, SamplingParams params) {
    InferenceRequest req = InferenceRequest.of(modelId,
            List.of(ChatMessage.user("Hello")), params, RequestPriority.NORMAL);
    PublisherTokenConsumer bridge = new PublisherTokenConsumer();
    Flow.Publisher<String> pub = bridge.publisher();
    scheduler.submit(req, bridge).whenComplete((r, e) -> bridge.finish());
    // subscribe to pub …
}
```

#### Java HTTP client (`JunoHttpClient`)

Talk to a sidecar started with `./juno local … --api-port 8080` (or `JunoPlayer.startApiServer`):

```java
import java.net.URI;
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.player.JunoHttpClient;
import cab.ml.juno.tokenizer.ChatMessage;

var http = new JunoHttpClient(URI.create("http://localhost:8080"));

// Native blocking inference (/v1/inference)
String text = http.blockingInference("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Ping")), 64);

// Native SSE (/v1/inference/stream) — publisher emits decoded token pieces from JSON events
Flow.Publisher<String> nativeStream = http.streamingInference(null,
        List.of(ChatMessage.user("Stream ping")), 32);

// OpenAI-compatible blocking + SSE (/v1/chat/completions)
String openAiText = http.blockingOpenAiChat("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Ping")), 64, 0.7f);
Flow.Publisher<String> openAiSse = http.streamingOpenAiChat("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Stream")), 32, 0.7f);
```

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

**GPU on AWS instances:** pre-installed in the golden AMI by `make-ami.sh`. Node bootstrap runs `lspci` to detect the GPU vendor and sets `JUNO_USE_GPU=true` — no DKMS compilation at boot.

- **NVIDIA (g4dn, g5, g6, p\*):** CUDA 12.3 + nvidia-open. Backend auto-selects CUDA.
- **AMD Radeon (g4ad):** ROCm 7.2.4 + amdgpu-dkms. The AMI sets `HSA_OVERRIDE_GFX_VERSION=10.1.0` in `/etc/environment` to work around the missing gfx1011 rocBLAS kernels on the Radeon Pro V520 (upstream issue ROCm/rocm-libraries#4347); rocBLAS uses the gfx1010 dispatch path which runs correctly on Navi12 silicon. Backend auto-selects ROCm when CUDA libraries are absent.

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

Run cluster command with `--verbose` to enable `[TRACE]` output:

| Line | What it tells you |
|------|-------------------|
| `[TRACE] model type (chat template key) : tinyllama` | Whether the template matches the model |
| `[TRACE] formatted training text (repr)` | Exact token sequence sent to the model during training |
| `[TRACE] token count (excl. BOS): N` | How many tokens are in the training sequence |
| `[TRACE] step=N loss=F chunk=M/T ms=D` | Per-step loss during training |
| `[TRACE] inference model type: tinyllama` | Template key at inference — must match training |

If the template key at training and inference differ, the model will not recall trained facts.
Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`,
`phi3`) to ensure `ChatModelType.fromPath()` detects it correctly. Gemma, Qwen 2 / Qwen3 /
Qwen3.5 paths are under development — prefer LLaMA-family or Phi-3 models for LoRA
training workflows today.

If the loaded GGUF carries an embedded `tokenizer.chat_template`, `juno lora` prints a one-line
warning that it is ignored — LoRA train and `--lora-play` always use the named template above so
train-time and inference-time formatting stay identical (see "Chat templates and Hugging Face
downloads").

---

### Metrics

```bash
# Automatic in local mode (single JVM — all juno.* events in one .jfr file)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Cluster mode: coordinator + each node write separate .jfr files. On exit the launcher
# calls MetricsMain.extractToJson() once per existing file and prints each summary;
# target/metrics/metrics.json reflects the last processed file. For throughput (TPS),
# use the coordinator recording (juno.TokenProduced lives on the coordinator JVM).

# Manual extraction from .jfr files in the project root
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json (one snapshot per mapped .jfr in project root)
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
mvn clean package -DskipTests          # build — juno-player emits thin jar + *-shaded.jar runnable

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master             # integration tests — forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT and TensorParallelClusterIT

mvn verify -pl juno-master -Pintegration -Dmodels=/path/to/models
                                       # ModelLiveRunnerIT — requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test (8 checks, exits 0/1)
```

**Windows (Command Prompt):**
```bat
mvn clean package -DskipTests

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player

mvn verify -pl juno-master

mvn verify -pl juno-master -Pintegration -Dmodels=C:\models

juno.bat test --model-path models\model.gguf
```

**GPU tests** (NVIDIA — requires CUDA 12.x and an NVIDIA GPU):

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl juno-master \
  --enable-native-access=ALL-UNNAMED
```

**Windows (NVIDIA GPU tests):**
```bat
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=C:\models\model.gguf -pl juno-master ^
  --enable-native-access=ALL-UNNAMED
```

**GPU tests** (AMD — requires ROCm 6+ and an AMD GPU):

```bash
mvn test -Dgroups=rocm -pl node --enable-native-access=ALL-UNNAMED
```

> **Note:** ROCm is Linux-only. AMD GPU tests are not supported on Windows.