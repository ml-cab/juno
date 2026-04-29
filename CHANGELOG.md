## Status

**Session 29** — OpenAI-compatible REST API (`POST /v1/chat/completions`, `GET /v1/models`).

### OpenAI-compatible API

Any client that speaks the OpenAI Chat Completions wire format — LangChain, LlamaIndex,
LiteLLM, the OpenAI Python/Node SDKs, or any internal tool built against `openai.*` — works
against Juno with a single base-URL change. No prompt reformatting, no adapter library, no
glue code.

**New classes (coordinator module):**

- **`OpenAiAdapter`** — pure static mapping helpers between Juno internals and the OpenAI wire
  format: `repetitionPenaltyFromFrequencyPenalty(float)` (OpenAI −2..2 range → Juno ≥1),
  `validateCompletionsN(Integer)` (rejects n ≠ 1), `toOpenAiFinishReason(StopReason)` (`stop`
  / `length` / `error`), and `chatCompletionId(String)` (`chatcmpl-` + compact UUID).
- **`OpenAiChatHandler`** — Javalin handler class owning three endpoints:
  - `POST /v1/chat/completions` — deserialises `OaiChatCompletionRequest` (Jackson,
    `@JsonIgnoreProperties(ignoreUnknown = true)`), validates `n` and `messages`, builds an
    `InferenceRequest` + `SamplingParams`, then dispatches to either
    `scheduler.submitAndWait()` (blocking, returns `ChatCompletion` JSON) or
    `scheduler.submit()` (streaming, writes `text/event-stream` chunks terminated by
    `data: [DONE]`).
  - `GET /v1/models` — filters `ModelRegistry` to `LOADED` status, wraps each
    `ModelDescriptor` in an OpenAI `Model` object with `x_juno_*` extension fields.
  - `GET /v1/models/{modelId}` — single-model lookup; 404 when absent.

**Modified: `InferenceApiServer`** — constructs `OpenAiChatHandler` in the constructor
(passing the latency callback so `HealthReporter` still records P99). Routes
`POST /v1/chat/completions` and `GET /v1/models[/{modelId}]` to the handler.
The existing `POST /v1/inference` and `POST /v1/inference/stream` endpoints are untouched.

**Modified: `ConsoleMain`** (`juno-player` module) — `--api-port N` flag starts a
`RequestScheduler` + `InferenceApiServer` alongside the existing REPL in both `local` and
cluster modes. A virtual-thread shutdown hook calls `apiServer.stop()` on JVM exit.
`buildLocalModelRegistry()` populates a `ModelRegistry` from the in-process `LlamaConfig` so
`GET /v1/models` returns the loaded model immediately.

**Modified: `scripts/run.sh`** — `--api-port N` flag wired into both `cmd_local()` and
`cmd_cluster()`. Environment override: `API_PORT`.

**New file: `api/src/main/resources/juno-api.yaml`** — OpenAPI 3.0.3 spec for the public
client-facing API. Documents all request fields with their Juno internal mappings, the SSE
chunk event sequence, Juno extension fields (`x_juno_priority`, `x_juno_session_id`,
`x_juno_top_k`, `x_juno_latency_ms`, `x_juno_retry_after_ms`, `x_juno_queue_depth`), and
all error codes.

**New test: `OpenAiAdapterTest`** — unit tests for all four mapping helpers.

**Field mapping summary (request):**

| OpenAI field | Juno internal | Notes |
|---|---|---|
| `model` | `modelId` | First loaded model if omitted |
| `messages[].role` / `.content` | `ChatMessage` | Text only; images not supported |
| `temperature` | `SamplingParams.temperature` | 0.0–2.0; default 0.7 |
| `top_p` | `SamplingParams.topP` | 0.0–1.0; default 0.9 |
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 512 |
| `max_tokens` | `SamplingParams.maxTokens` | Deprecated alias |
| `frequency_penalty` | `SamplingParams.repetitionPenalty` | `1 + max(0, fp/2)` |
| `stream` | route selection | false → blocking JSON; true → SSE |
| `n` | — | Only 1 is accepted; other values → 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` | — | Silently ignored |
| `x_juno_priority` | `RequestPriority` | HIGH / NORMAL / LOW |
| `x_juno_session_id` | `InferenceRequest.sessionId` | Enables KV-cache reuse across turns |
| `x_juno_top_k` | `SamplingParams.topK` | 0 = disabled; default 50 |

All modules compile. All existing tests pass. `OpenAiAdapterTest` (4 assertions) passes.

---

## Status

**Session 28** — Health dashboard: CPU load metric, role-conditional secondary metric, node throughput.

### Health dashboard fixes

**Fix 1 — `temperatureCelsius` → `cpuLoad`.**
`/sys/class/thermal` is unavailable on EC2 VMs; the Temperature row always showed `—`. Replaced with process CPU utilisation read from `OperatingSystemMXBean.getCpuLoad()` (0.0–1.0, available on all JVM platforms, no sysfs). Changes:
- `NodeHealth` record: field `temperatureCelsius` removed, `cpuLoad` added (same sentinel -1.0 convention, clamped to 0.0 on first-sample unavailability).
- `HealthReporter.buildProbeJson()`: `readTemperatureCelsius()` + all sysfs helpers (`findThermalZone`, `findHwmonTemp`, thermalPath/thermalProbed state) removed; replaced by 5-line `readCpuLoad()`.
- `HealthMain.NodeHealthDto`: `temperatureCelsius` field → `cpuLoad`.
- Dashboard HTML (both `HealthMain` and `InferenceApiServer` embedded console): "Temperature" row → "CPU load" formatted as `XX.X %`.

**Fix 2 — Role-conditional secondary metric: coordinator shows Latency P99, nodes show Throughput.**
`Latency P99` was populated by `HealthReporter.recordLatency()`, which is only called from `InferenceApiServer` on the coordinator JVM. Worker nodes always showed `—`. Added a `nodeRole` field (`"coordinator"` | `"node"`) to `NodeHealth` and `NodeHealthDto` so the dashboard can branch:
- **Coordinator card** — Latency P99 (ms): end-to-end generation time, already wired via `InferenceApiServer.setLatencyReporter()`.
- **Worker node cards** — Throughput (MB/s): activation bytes forwarded per second via new `HealthReporter.recordBytes(long n)` + `drainThroughput()` (atomic byte counter drained each probe interval).

Wiring:
- `EmbeddedNodeServer`: retained `NodeServiceImpl` reference as `serviceImpl` field; added `setHealthReporter(HealthReporter)` on outer class delegating to a new package-private setter on the inner class. `forwardPass()` calls `hr.recordBytes(encodedOutput.length)` after each `responseObserver.onNext()`.
- `NodeMain`: constructs reporter with `nodeRole="node"`, calls `server.setHealthReporter(reporter)` after `server.start()`.
- `CoordinatorMain`: constructs reporter with `nodeRole="coordinator"`.
- `HealthReporter` constructors: 2-arg and 3-arg remain backward-compatible (default role `"node"`); new canonical 4-arg constructor `(nodeId, nodeRole, healthBaseUrl, intervalMs)`. Added `startForCoordinator(healthBase)` factory alongside existing `startForNode(nodeId, healthBase)`.
- `buildNodeDetail()` switched from `Map.of()` (10-entry limit) to `Map.ofEntries()` to accommodate 12 fields.

**Investigation 3 — Why 1 of 10 concurrent sessions produced no tokens (no code change).**
Root cause: gRPC `ServerBuilder.forPort(port)` with no custom executor defaults to a thread pool bounded by `~2 × CPU count` (4 threads on `m7i-flex.large`). With 9 sessions concurrently running prefill (26 steps × 9 = up to 234 in-flight blocking stubs), all 4 gRPC threads on each node were saturated. The 10th session's first `pipeline.forward()` call queued behind them for ~8.5 minutes until prefill of the other 9 finished. The fix is `ServerBuilder.forPort(port).executor(Executors.newVirtualThreadPerTaskExecutor())` — virtual threads don't block OS threads on gRPC I/O. JFR evidence: `juno.ForwardPass.decode.p95_ms = 3095 ms` on node-1 (coordinator node running layers 0–8 plus the REST server) vs 914 ms on node-2; coordinator log confirms 10 tokenizer encodes but only 9 near-simultaneous prefills.

All modules compile. All existing tests pass (NodeHealth, HealthEvaluator, HealthReactor constructors updated to 9-arg signature).

---

**Session 27** — GPU lifecycle, multi-device shared contexts, CUDA streams, Llama VRAM fallback, docs.

- **`ForwardPassHandler.releaseGpuResources()`** — default no-op; **`LlamaTransformerHandler`** and **`Phi3TransformerHandler`** close all **`DeviceHalfMatrix`** buffers. **`EmbeddedNodeServer`** invokes it on shard reload, load failure, and **`unloadShard`** (then swaps in **`StubForwardPassHandler`**).
- **`GpuContext.shared(int)`** — one process-wide **`GpuContext`** per CUDA device index (map + lock); **`close()`** remains a no-op for shared instances. **`ForwardPassHandlerLoader.selectBackend()`** and **`EmbeddedNodeServer`** honour **`-Djuno.cuda.device=N`**, validated against **`CudaAvailability.deviceCount()`**.
- **`CudaMatVec`** — per-thread **non-blocking CUDA stream**; **`cublasSetStream_v2`** + **`cudaMemcpyAsync`** for resident FP32/FP16 **`x`/`y`** transfers; **`synchronized(gpuContext.cublasSerializationLock())`** around stream binding and kernels. Host **`sgemv(float[],…)`** also runs under the same lock.
- **Llama GPU OOM** — upload wrapped like Phi-3: on **`cudaMalloc`** failure, partial **`DeviceHalfMatrix`** buffers are **`close()`**d and inference falls back to **CPU quantised** matmul for those projections.
- **Docs/tests:** **`README.md`**, **`docs/arch.md`**, **`GpuContextTest`** (multi-GPU assumption), **`NodeMain`** Javadoc for **`juno.cuda.device`**.

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
- TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster
- Phi-3.5 GPU matmul path: `CudaMatVecBackendTest` FP16 resident matvec + `mvn test -Dgroups=gpu -pl node` on CUDA 12.x

**Session 26** — Phi-3 GPU matmul, FP16 resident weights, CLI and local GPU wiring.

`Phi3TransformerHandler` GPU path uploads dequantized fused QKV / FFN slices and output projection as **`DeviceHalfMatrix`** (IEEE FP16 on device, roughly half the VRAM of `DeviceFloatMatrix`). Forward uses **`CudaMatVec.sgemv(DeviceHalfMatrix, x)`**, implemented with **`cublasHSSgemvStridedBatched`** — same `(CUBLAS_OP_T, m=cols, n=rows, lda=cols)` layout contract as the proven **`cublasSgemv_v2`** path for row-major `A`. Host `float[]` activations are converted to FP16 for the per-call device `x` buffer; accumulation stays FP32. Earlier **`cublasSgemmEx` / `cublasGemmEx`** mixed-dtype attempts returned `NOT_SUPPORTED` / `INVALID_VALUE` on common stacks; the HSS strided-batched GEMV avoids that.

**Session 26** — Native LoRA merge (`juno merge`).

`LoraMerge` (new, `node` module) writes a new GGUF file from a base model and a `.lora` checkpoint without re-quantising the patched tensors. The 44 LoRA-adapted projection weights (wq/wv on every layer) are stored as F32; all other tensors are copied verbatim in their original quantised encoding. F32 is required because the LoRA delta (~6×10⁻⁴) is smaller than Q4_K quantisation noise (~3×10⁻³) — re-quantising would silently erase the training. Verified: merged TinyLlama recalls `/train-qa` facts (name "Dima") correctly under `./juno local` with no `.lora` sidecar.

`GgufReader` gains five new public methods needed by the GGUF writer: `ggufFileOffset()`, `metadataSectionEnd()`, `tensorOrder()`, `tensorNelems(name)`, and keeps the existing `tensorAbsoluteOffset` / `tensorType` / `tensorDims`. Internal storage changed from `HashMap` to `LinkedHashMap` so `tensorOrder()` is stable. A `List<String> tensorOrder` field is added to preserve insertion order.

`LoraMergeMain` (`juno-player` module) — CLI entry point for `juno merge`. Reads `--model-path`, `--lora-path`, `--output`, `--heap`. Derives `<model>.lora` and `<model>-merged.gguf` as defaults.

`run.sh` gains `cmd_merge()` and the `merge)` dispatch case.

`ConsoleMain` `/merge-hint` REPL command updated: now prints the actual `./juno merge` invocation instead of the old "contributions welcome" message.

Three bugs fixed during development of `LoraMerge`:
- **Q4_K**: `d = maxRange/63` → `d = maxRange/(63×15)`. Previous formula collapsed all 4-bit quant values to `{0,1}`.
- **Q5_K**: same bug, factor 31. `d = maxRange/63` → `d = maxRange/(63×31)`.
- **Q3_K scRaw packing**: aux0/aux1 high-nibble extraction used a broken two-pass utmp reconstruction; replaced with a clean direct inverse of `GgufReader.loadQ3_K`.

**Session 25** — Code quality: dead code removed, test helpers moved to test scope, docs fully updated.


`CyclicForwardPassHandler` moved from `node/src/main` to `node/src/test`. It is a deterministic stub with no business value without a model; it belongs exclusively in the test compilation unit. `EmbeddedNodeServer` no longer imports it — the three call sites (pre-load placeholder, model-load-failure fallback, no-model stub mode) are now served by a new private `StubForwardPassHandler` inner class that returns zero-filled arrays of the correct shape with no test machinery. `node/pom.xml` gains a `maven-jar-plugin` `test-jar` execution so other modules can still import `CyclicForwardPassHandler`; `coordinator/pom.xml` and `juno-master/pom.xml` declare the `node:tests` classifier dependency.

**VRAM / OOM:** GPU buffer allocation is wrapped; on failure (including `cudaMalloc` OOM), partial device buffers are closed and the handler falls back to **CPU quantised** `LlamaTransformerHandler.matVec`-style matmul for those projections.

**`ConsoleMain`:** missing **`break`** after **`--cpu`** fixed — parsing no longer fell through into **`--lora`**, which incorrectly set `loraMode` when forcing CPU inference.

**`ConsoleMain.runLocalRepl`:** one shared **`GpuContext`** + **`CudaMatVec`** instance for every in-process shard load (avoids redundant cuBLAS contexts and matches production “one GPU per JVM” usage).

**Tests:** `CudaMatVecBackendTest.device_half_matrix_sgemv_matches_host_path` (512×512) anchors FP16 resident correctness vs `LlamaTransformerHandler.matVec`.

**JFR:** `MatVecEvent.backend` **`cuda-resident-fp16`** labels the Phi FP16 device path. (As of session 27, Llama GPU resident weights also use **`cuda-resident-fp16`**; **`cuda-resident`** remains for **`DeviceFloatMatrix`** / tests.)

---

**Session 26** — LoRA inference overlay (`--lora-play`), Q&A training mode (`/train-qa`), diagnostic tracing, and AWS deploy hardening.

### `--lora-play PATH` — apply trained adapters at inference in any mode

Pre-trained `.lora` checkpoint files can now be applied read-only at inference time without entering the `lora` REPL. Three modes are supported:

**`local` mode:**
```bash
./juno local --model-path model.gguf --lora-play /path/to/model.lora
```
`ConsoleMain.runLocalRepl()` calls `LoraAdapterSet.load(Path.of(loraPlayPath))` before building the shard handlers and passes the result into `ForwardPassHandlerLoader.load(..., playAdapters)`.

**`cluster` mode (forked JVMs):**
```bash
./juno --model-path model.gguf --lora-play /path/to/model.lora
```
`ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path=PATH` into every forked node JVM command. `EmbeddedNodeServer.NodeServiceImpl` reads this property at construction and loads adapters inside `loadShard()` before the `ForwardPassHandlerLoader` call.

**AWS deployed cluster:**
```bash
./launcher.sh juno-deploy.sh setup --lora-play /absolute/path/to/model.lora
```
See AWS section below.

### `ForwardPassHandlerLoader` — new LoRA overload

```java
// New canonical overload — all others delegate to this
public static ForwardPassHandler load(
    Path modelPath, ShardContext context, MatVec backend,
    LoraAdapterSet adapters) throws IOException
```

When `adapters != null`, the loader routes to `LoraTrainableHandler` (inference-only, no optimizer attached) instead of the architecture-specific handler. When `adapters == null` the existing `phi3` / `llama` dispatch is unchanged. `selectBackend()` promoted from package-private to `public` so juno-player-module callers can reuse it.

### `ClusterHarness` — `withLoraPlay()` fluent method

```java
harness.withLoraPlay("/path/to/model.lora");
```

Stores the path and injects `-Djuno.lora.play.path=PATH` into the `launchNode()` JVM command, after the JFR flags. Without this, forked node JVMs start with `loraPlayPath=null` and run the base model regardless of what the coordinator is told.

### `/train-qa` — conversational Q&A training

New REPL command in `lora` mode for training single-fact associations:

```
you > /train-qa What is my name? A: Dima
  Question: What is my name?
  Answer  : Dima

  Formatted as 4 Q&A pairs  ·  model type: tinyllama
  Training  rank=8 · lr=1.0E-4 · 40 steps ...
  ✔ done  loss=▼ 1.53 (−0.83)
```

The command auto-generates 4 phrasings of the question (exact, `Can you tell me: ...`, `Please answer: ...`, plus one repeat) to improve generalization. The chat template appropriate for the model type (detected from the model path) is applied to each pair. Flags `--lora-steps-qa N` and `--lora-early-stop F` control training depth.

Separator syntax: `Q: <question> A: <answer>` or `<question> A: <answer>`.

### Diagnostic tracing (`--verbose`)

All tracing is prefixed `[TRACE]` for easy grep. Added to:

| Location | What is shown |
|----------|---------------|
| LoRA REPL startup | Model type (chat template key), model path, all LoRA hyperparameters |
| `/train-qa` | Exact formatted training text with `↵` for newlines, token count, token IDs (verbose only) |
| Per training step (verbose) | `step=N loss=F chunk=M/T ms=D` |
| Cluster inference (verbose) | Chat template key used for each inference request |
| `juno-deploy.sh` bootstrap | Per-node params baked into user-data script |
| `juno-deploy.sh` SCP | Local source, remote target, per-node `node.env` patch |
| `juno-deploy.sh` coordinator env | Full `cluster-nodes.env` contents echoed after write |

### AWS deploy hardening (`juno-deploy.sh`)

Multiple bugs fixed during end-to-end AWS validation:

**Double base64 encoding (cloud-init rejected user-data).** `--user-data` was passed as a pre-base64-encoded string. AWS CLI base64-encodes it again; cloud-init received double-encoded garbage and logged `Unhandled non-multipart (text/x-not-multipart) userdata`. Fix: write user-data to a temp file and pass `file:///tmp/juno-userdata-*.sh` — the CLI reads it raw and does single encoding. The `[TRACE]` size line now also prints `first-line: #!/bin/bash` so shebang presence is visible in the setup log.

**TRACE logs contaminating user-data.** `_build_node_userdata` is called as `USER_DATA=$(_build_node_userdata ...)` which captures all stdout. The four `log` / `[TRACE]` calls inside the function were writing to stdout, prepending ANSI escape codes before `#!/bin/bash`. Cloud-init saw no shebang on line 1 and skipped execution. Fix: all `log` calls inside `_build_node_userdata` now redirect to stderr with `>&2`.

**Relative `--lora-play` path not resolved.** When called from `scripts/aws/`, a path like `../models/model.lora` resolves to `scripts/models/model.lora` (which doesn't exist). `_scp_lora_to_nodes` hit the `[[ ! -f ]]` guard and returned silently, leaving `node.env` with empty `JUNO_LORA_PLAY_PATH`. Fix: `--lora-play` is resolved to absolute path at parse time via `realpath`. `setup()` also validates the file exists before any AWS spend.

**Race condition: coordinator started before node restart completed.** `_scp_lora_to_nodes` previously used `systemctl restart --no-block` and polled `systemctl is-active` to detect readiness. The old instance remained `active` during shutdown so the poll returned immediately, `_write_cluster_env_and_start_coordinator` ran, and the coordinator sent `loadShard` to the old (no-LoRA) instance. The restarted instance came up 19 minutes later, too late. Fix: synchronous stop → patch → start per node: `systemctl stop juno-node` (synchronous, waits for JVM exit), `sed` patch of `node.env`, `systemctl start juno-node` (synchronous, returns once gRPC port is bound, ~2s). Coordinator only starts after all three nodes have confirmed `active` status with correct env.

**Local relative path baked verbatim into `cluster-nodes.env`.** Even when SCP succeeded, the coordinator received `JUNO_LORA_PLAY_PATH=../models/...` (the pre-`realpath` value), causing `model load failed: ../models/...` on the nodes. Fix: `_scp_lora_to_nodes` updates the global `LORA_PLAY_PATH` to the remote absolute path (`/opt/juno/models/<basename>`) before returning, so `_write_cluster_env_and_start_coordinator` writes the correct value.

**`_write_cluster_env_and_start_coordinator` missing closing brace.** The `}` was accidentally elided, causing `scan_regions()` to be parsed as part of the function body.

**End-to-end verification:**
```
you> what is my name?
bot> Dima
```
Confirmed working on 3 × m7i-flex.large AWS cluster (eu-north-1) with TinyLlama-1.1B-Chat-v1.0.Q4_K_M and a `.lora` adapter trained locally, SCPed and deployed via `juno-deploy.sh setup --lora-play`.

---

**Session 25** — Code quality, dead code removed, docs updated. *(unchanged)*

**Session 24** — Configurable activation byte order (`--byteOrder BE|LE`). *(unchanged)*

**Session 22** — Q2_K and Q3_K quantization support. *(unchanged)*

**Session 21** — Two new deployment fat-jar modules and a unified AWS script. *(unchanged)*

**Session 20** — GPU inference actually wired end-to-end. *(unchanged)*

**Session 19** — metrics module, Meta-Llama 3 tokenizer fix, AWS infrastructure scripts. *(unchanged)*

**Session 18** — GPT-2 BPE tokenizer, JFR instrumentation fixes. *(unchanged)*

**Session 17** — AWS infrastructure scripts. *(unchanged)*

**Session 14** — LoRA fine-tuning + JFR profiling. *(unchanged)*