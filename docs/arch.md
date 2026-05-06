# Juno — Architecture Reference

**Java Unified Neural Orchestration** — distributed LLM inference and fine-tuning framework.

This document describes the internal architecture of Juno. For usage instructions see
[howto.md](howto.md). For LoRA see [LoRA.md](LoRA.md).

---

## Distribution Strategies

Two strategies are available, selected with `--pType` at startup.

### Pipeline parallel (`--pType pipeline`, default)

Transformer layers are split into contiguous blocks and assigned to nodes. The activation
tensor flows serially: `node-1 -> node-2 -> node-3`. Each node holds a contiguous depth
slice. Adding nodes increases total VRAM, enabling larger models. Cost: N-1 sequential gRPC
hops per decode step.

```
[Client]  REST (Javalin) / gRPC streaming
    |
[Coordinator]
    |-- GgufTokenizer       (BPE from GGUF metadata)
    |-- ChatTemplateFormatter
    |-- RequestScheduler    (virtual threads, CompletableFuture)
    |-- Sampler             (temperature / top-k / top-p / rep. penalty)
    |-- KVCacheManager      (GPU tier + CPU tier + PrefixCache trie)
    +-- GenerationLoop      (prefill + decode + session KV reuse)
              |
              | gRPC activations (FLOAT16 / INT8 / FLOAT32, BE or LE wire order)
              | serial: node-1 -> node-2 -> node-3
              |
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3  ...      |
    |  L 0-7        L 8-14       L 15-21          |
    |  + embed                   + output proj    |
    |  NodeKVCacheAdapter wired into each handler |
    |  LoraAdapterSet (optional, read-only)       |
    +--------------------------------------------+
```

### Tensor parallel (`--pType tensor`)

Every node holds all transformer layers but only a horizontal slice of the weight matrices:
attention heads `[headStart, headEnd)` and a proportional FFN width slice. The coordinator
broadcasts the input token embedding to all nodes simultaneously, collects partial logit
vectors, and reduces them via element-wise sum (star AllReduce). Adding nodes increases
throughput and reduces per-node memory pressure. Cost: one broadcast + N parallel gRPC calls
per decode step.

```
[Coordinator]
    +-- GenerationLoop
              |
              | broadcast same tokens to all nodes (parallel)
              |
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3  ...      |
    |  L 0-21       L 0-21       L 0-21           |
    |  heads 0-10   heads 11-21  heads 22-32      |
    |  rank=0       rank=1       rank=2            |
    +--------------------------------------------+
              |
              | partial logits from each node (parallel)
              |
    [AllReduce: element-wise sum -> full logit vector]
              |
    [Sampler]
```

Constraint: `numHeads % nodeCount == 0`.

Star AllReduce requires no InfiniBand and no inter-node communication. The coordinator
collects and sums in O(N x vocabSize).

---

## REST API Layer

`InferenceApiServer` (Javalin) is the single HTTP entry point on the coordinator. It exposes
two API surfaces that share the same underlying `RequestScheduler` and `GenerationLoop`.

### Juno native API

| Method | Path | Handler |
|--------|------|---------|
| `POST` | `/v1/inference` | `handleBlockingInference` — blocking, returns `GenerationResult` |
| `POST` | `/v1/inference/stream` | `handleStreamingInference` — SSE, one event per token |
| `GET` | `/v1/models` | `OpenAiChatHandler.handleListModels` |
| `GET` | `/v1/models/{modelId}` | `OpenAiChatHandler.handleGetModel` |
| `DELETE` | `/v1/models/{modelId}` | `handleUnloadModel` |

### OpenAI-compatible API

| Method | Path | Handler |
|--------|------|---------|
| `POST` | `/v1/chat/completions` | `OpenAiChatHandler.handleChatCompletion` |
| `GET` | `/v1/models` | `OpenAiChatHandler.handleListModels` |
| `GET` | `/v1/models/{model}` | `OpenAiChatHandler.handleGetModel` |

Any client that speaks the OpenAI Chat Completions wire format works against Juno with only a
base-URL change — no prompt reformatting, no adapter library, no glue code.

```
[OpenAI SDK / LangChain / LlamaIndex / curl]
    |
    | POST /v1/chat/completions  (JSON body, snake_case fields)
    |
[OpenAiChatHandler]
    |-- deserialise OaiChatCompletionRequest   (Jackson, @JsonIgnoreProperties)
    |-- validate n, messages
    |-- build InferenceRequest + SamplingParams via OpenAiAdapter
    |-- resolveModelId  (first loaded model if omitted)
    |
    +-- stream=false --> scheduler.submitAndWait()
    |                        |
    |                    GenerationResult
    |                        |
    |                    wrap as ChatCompletion JSON (OpenAI envelope)
    |
    +-- stream=true  --> scheduler.submit(request, TokenConsumer)
                             |
                         SSE chunks  (one per token, text/event-stream)
                             |
                         data: [DONE]
```

`OpenAiAdapter` is a pure static utility class with no state:

- `repetitionPenaltyFromFrequencyPenalty(float)` — maps OpenAI's `frequency_penalty` (−2..2)
  to Juno's `repetitionPenalty` (≥1) via `1 + max(0, fp/2)`.
- `validateCompletionsN(Integer)` — returns an error message when `n ≠ 1`; null when valid.
- `toOpenAiFinishReason(StopReason)` — `EOS_TOKEN`/`STOP_TOKEN` → `"stop"`;
  `MAX_TOKENS` → `"length"`; `ERROR` → `"error"`.
- `chatCompletionId(String)` — formats the completion ID as `chatcmpl-` + UUID without hyphens.

No changes to `GenerationLoop`, `RequestScheduler`, the sampler, the tokenizer, or any node
code are required by the OpenAI layer. It is a pure translation shim above the scheduler.

---

## Handler Routing

`ForwardPassHandlerLoader` reads `general.architecture` from GGUF metadata and dispatches:

```
ForwardPassHandlerLoader
    |
    phi3  -> Phi3TransformerHandler   (fused QKV + gate/up, quantized weights)
    *     -> LlamaTransformerHandler  (separate tensors, quantized weights)

LoRA overlay (optional):
    load(..., LoraAdapterSet)  <- wraps base handler in LoraTrainableHandler
                                  adapters applied read-only during inference
                                  base GGUF is never modified

MatVec (injected into handler):
    CpuMatVec    <- parallel IntStream
    CudaMatVec   <- cublasSgemv_v2 (FP32 host path) / resident FP32 or FP16 weights:
                    Llama + Phi-3 GPU use DeviceHalfMatrix + cublasHSSgemvStridedBatched;
                    per-thread CUDA stream + async H2D/D2H around GEMV;
                    synchronized(gpuContext.cublasSerializationLock());
                    GpuContext.shared(dev); weights uploaded once at load time;
                    releaseGpuResources() frees VRAM on unload

KV cache wiring (per node, after loadShard()):
    NodeKVCacheAdapter  <- serialises float[][] K/V into KVBlock,
                           flushes write-through to KVCacheManager (GPU + CPU tiers),
                           restores on local cache miss,
                           propagates evict() to both stores
```

Backend selection is automatic via `selectBackend()`, which reads `JUNO_USE_GPU`,
`CudaAvailability`, and `-Djuno.cuda.device` (defaults to `0`).

---

## Key Design Decisions

**No Python, no subprocess.** The JVM reads GGUF binary directly via `GgufReader` and runs the
full transformer forward pass end to end.

**No Spring Boot.** Javalin for REST. Virtual threads (`Executors.newVirtualThreadPerTaskExecutor()`)
on the gRPC `ServerBuilder` — required to avoid OS-thread saturation under concurrent prefill sessions.

**OpenAI wire compatibility without framework coupling.** `OpenAiChatHandler` and `OpenAiAdapter`
are new classes added to the coordinator module. No existing classes were modified beyond
`InferenceApiServer` wiring and `ConsoleMain` flag parsing. The existing `POST /v1/inference`
and `POST /v1/inference/stream` endpoints are untouched. Adding new classes rather than
extending `InferenceApiServer` keeps each concern isolated and the existing server stable.

**Lazy dequantization on CPU; eager upload on GPU.** On the CPU path, dequantization runs
one 256-element block at a time inside the matmul loop (peak live float footprint ~1 kB instead
of ~65 MB). On the GPU path, Llama and Phi-3 dequantize once on load and upload to
`DeviceHalfMatrix` (FP16 on device). If `cudaMalloc` fails, both handlers close partial GPU
buffers and fall back to CPU quantized matmul for those projections.

**Explicit GPU weight lifecycle.** `ForwardPassHandler.releaseGpuResources()` closes all
`DeviceHalfMatrix` / `DeviceFloatMatrix` buffers. `EmbeddedNodeServer` calls it on shard
unload, reload, and handler swap so VRAM is freed without waiting for GC.

**Configurable activation byte order.** `ActivationCodec` reads `juno.byteOrder` once at
class-load time and branches to `ActivationBECodec` (big-endian, default) or `ActivationLECodec`
(little-endian, native x86 order). `ClusterHarness` injects `-Djuno.byteOrder` into every forked
node process; `juno-deploy.sh` writes it into `/etc/juno/node.env` for systemd-managed nodes.

**KV cache wired at the node level.** `NodeKVCacheAdapter` connects `LlamaTransformerHandler`
and `Phi3TransformerHandler` to `KVCacheManager` (GPU byte-budget LRU + Caffeine W-TinyLFU CPU
tier). Every forward pass flushes K/V data write-through into both tiers. On local cache miss,
the next forward pass at that position restores transparently. `evict(requestId)` propagates to
both the local map and both cache tiers.

**LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps
`LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4-16) on Q
and V projections. Frozen weights stay quantized at all times. Adapters persist to a `.lora`
binary checkpoint; the GGUF is never modified. For a standalone merged model, use `./juno merge`.

**Native LoRA merge.** `LoraMerge` writes a new GGUF where the 44 LoRA-patched projection
tensors (wq/wv per layer) are stored as F32. The LoRA delta (~6x10^-4 per element) is smaller
than Q4_K quantization noise (~3x10^-3); re-quantizing would erase all training. All other
tensors are copied verbatim in their original quantized form.

**GPT-2 BPE and SentencePiece BPE both supported.** `GgufTokenizer` reads
`tokenizer.ggml.model` from GGUF metadata. Value `"gpt2"` activates the GPT-2 / tiktoken path
(Llama 3+). Any other value uses SentencePiece (Llama 1/2, TinyLlama, Mistral, Gemma, Phi-3).
Detection is automatic at load time — no configuration required.

**AWS infrastructure fully scripted.** `juno-deploy.sh` is the unified cluster lifecycle script.
Hardware is auto-detected during bootstrap: GPU nodes set `JUNO_USE_GPU=true` (CUDA is
pre-installed in the golden AMI by `make-ami.sh`). Commands: `setup | start | stop | teardown |
status | scan-regions`. GPU quota is checked before any instances launch; insufficient vCPUs
fail hard. State persisted to `~/.juno-deploy-state`.

**Full JFR instrumentation across every hot path.** Six custom event types —
`juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, `juno.Tokenizer`,
`juno.TemplateFormat`, `juno.LoraTrainStep` — make every layer of the stack observable in
JDK Mission Control without any agent or bytecode manipulation. In cluster mode, coordinator
and every forked node JVM each write their own `.jfr` file, merged automatically by
`MetricsMain.extractToJsonMerged()` on exit.

`juno.TokenProduced` is a coordinator-side instantaneous event fired once per token delivered
to a client after sampling and EOS checks. Because it lives in the coordinator JFR alongside
tokenizer events, `JfrMetricsExtractor` derives aggregate TPS directly from the span between
the first and last event timestamps and the total count — no synthetic timer or counter in
the inference path is needed. The JSON report exposes `juno.TokenProduced.count`,
`juno.TokenProduced.elapsed_seconds`, and `juno.TokenProduced.tps`.

**Stub mode.** `EmbeddedNodeServer` uses an internal `StubForwardPassHandler` (zero-filled arrays)
before a shard is loaded. The test-only `CyclicForwardPassHandler` lives in `node/src/test` and
is shared via the `node:tests` classifier jar. Integration tests in `juno-master` run stub mode —
no model file, no GPU, boots in seconds.

---

## Module Dependencies

```
juno-master (fat jar)
    +-- juno-player
    +-- coordinator
    +-- node
    |     +-- lora
    |     +-- kvcache
    |     +-- tokenizer
    |     +-- sampler
    |     +-- registry
    |     +-- api
    +-- health
    +-- metrics

juno-node (fat jar)
    +-- node
    +-- health
```

All modules share a common parent POM (`cab.ml:juno`) that manages dependency versions,
compiler settings, and plugin configuration.