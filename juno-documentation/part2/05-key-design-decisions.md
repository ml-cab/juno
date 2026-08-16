(ch-2-5)=
# 2.5. Key Design Decisions

GPU-related decisions (Panama FFI, lazy dequantization, explicit weight lifecycle) are covered
separately in [GPU acceleration](#ch-2-4). This page covers the remaining
architecture-level decisions.

**Runs directly inside the JVM.** `GgufReader` parses the GGUF binary format directly, and the
full transformer forward pass executes inside the same process; there is no subprocess and no
separate inference runtime to shell out to.

**Javalin instead of a full application framework.** Javalin serves REST. Virtual threads
(`Executors.newVirtualThreadPerTaskExecutor()`) run on the gRPC `ServerBuilder`, which is
required to avoid OS-thread saturation under concurrent prefill sessions.

**OpenAI wire compatibility without framework coupling.** `OpenAiChatHandler` and
`OpenAiAdapter` are new classes added to the coordinator module. No existing classes were
modified beyond `InferenceApiServer` wiring and `ConsoleMain` flag parsing. The existing
`POST /v1/inference` and `POST /v1/inference/stream` endpoints are untouched. Adding new classes
rather than extending `InferenceApiServer` keeps each concern isolated and the existing server
stable.

**Configurable activation byte order.** `ActivationCodec` reads `juno.byteOrder` once at
class-load time and branches to `ActivationBECodec` (big-endian, default) or `ActivationLECodec`
(little-endian, native x86 order). `ClusterHarness` injects `-Djuno.byteOrder` into every forked
node process; `juno-deploy.sh` writes it into `/etc/juno/node.env` for systemd-managed nodes.

**KV cache wired at the node level.** `NodeKVCacheAdapter` connects `LlamaTransformerHandler` and
`Phi3TransformerHandler` to `KVCacheManager` (GPU byte-budget LRU plus a Caffeine W-TinyLFU CPU
tier). Every forward pass flushes K/V data write-through into both tiers. On a local cache miss,
the next forward pass at that position restores transparently. `evict(requestId)` propagates to
both the local map and both cache tiers.

**LoRA fine-tuning without touching the base model.** `LoraTrainableHandler` wraps
`LlamaTransformerHandler` and adds trainable low-rank adapters (A/B matrices, rank 4-16) on the
Q and V projections. Frozen weights stay quantized at all times. Adapters persist to a `.lora`
binary checkpoint; the GGUF is never modified. For a standalone merged model, use
`./juno merge`. See [LoRA fine-tuning](#ch-4-1) for the full guide.

**Native LoRA merge.** `LoraMerge` writes a new GGUF where the 44 LoRA-patched projection
tensors (`wq`/`wv` per layer) are stored as F32. The LoRA delta (roughly 6e-4 per element) is
smaller than Q4_K quantization noise (roughly 3e-3), so re-quantizing would erase all training.
All other tensors are copied verbatim in their original quantized form.

**GPT-2 BPE and SentencePiece BPE both supported.** `GgufTokenizer` reads
`tokenizer.ggml.model` from GGUF metadata. The value `"gpt2"` activates the GPT-2 / tiktoken path
(Llama 3+). Any other value uses SentencePiece (Llama 1/2, TinyLlama, Mistral). Gemma uses the
same SentencePiece path via `LlamaTransformerHandler` but is under development. Phi-3 uses a
dedicated handler and `phi3` chat template (supported). Gemma, Qwen 2, Qwen3, and Qwen3.5 use
family-specific templates with validation in progress; treat these as under development.
Detection is automatic at load time and requires no configuration.

**AWS infrastructure fully scripted.** `juno-deploy.sh` is the unified cluster lifecycle script.
Hardware is auto-detected during bootstrap: GPU nodes set `JUNO_USE_GPU=true` (CUDA is
pre-installed in the golden AMI by `make-ami.sh`). Commands: `setup | start | stop | teardown |
status | scan-regions`. GPU quota is checked before any instances launch; insufficient vCPUs
fail hard. State persists to `~/.juno-deploy-state`. See
[AWS deployment](#ch-6-2) for the operational walkthrough.

**Full JFR instrumentation across every hot path.** Six custom event types, `juno.MatVec`,
`juno.ForwardPass`, `juno.TokenProduced`, `juno.Tokenizer`, `juno.TemplateFormat`, and
`juno.LoraTrainStep`, make every layer of the stack observable in JDK Mission Control without
any agent or bytecode manipulation. In cluster mode, the coordinator and every forked node JVM
each write their own `.jfr` file. On exit, `ConsoleMain` collects coordinator and node paths and
calls `MetricsMain.extractToJson()` once per existing file, printing a summary for each;
`target/metrics/metrics.json` reflects the last processed file. Use `./juno local --jfr` when you
need all custom events in a single recording. Throughput (TPS) metrics come from the coordinator
file (`juno.TokenProduced`). The programmatic `MetricsMain.extractToJsonMerged()` API merges
event lists across files for percentile math but is not invoked by the cluster shutdown hook
today.

`juno.TokenProduced` is a coordinator-side instantaneous event fired once per token delivered to
a client after sampling and end-of-sequence checks. Because it lives in the coordinator JFR
alongside tokenizer events, `JfrMetricsExtractor` derives aggregate TPS directly from the span
between the first and last event timestamps and the total count, with no synthetic timer or
counter needed in the inference path. The JSON report exposes `juno.TokenProduced.count`,
`juno.TokenProduced.elapsed_seconds`, and `juno.TokenProduced.tps`. See
[JFR and metrics](#ch-7-1) for the extraction
workflow.

**Stub mode.** `EmbeddedNodeServer` uses an internal `StubForwardPassHandler` (zero-filled
arrays) before a shard is loaded. `CyclicForwardPassHandler` lives in `node/src/test` and is
shared with integration tests in `juno-master` and `coordinator` via the `node:tests` classifier
jar. Integration tests run in stub mode: no model file, no GPU, boots in seconds.

## See also

- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)
- [Chapter 2.1 -- Overview](#ch-2-1)
- [Chapter 2.6 -- Module Map](#ch-2-6)

---

[<- 2.4 GPU Acceleration](#ch-2-4) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [2.6 Module Map ->](#ch-2-6)
