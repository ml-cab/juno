## Status

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
- TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster

**Session 22** — Q2_K and Q3_K quantization support.

`GgufReader` gains `loadQ2_K` and `loadQ3_K` — pure Java dequantizers for GGML_TYPE_Q2_K (type 10, 84 bytes / 256 elements) and GGML_TYPE_Q3_K (type 11, 110 bytes / 256 elements). Block layouts and bit-extraction logic mirror llama.cpp `dequantize_row_q2_K` / `dequantize_row_q3_K` exactly.

`LlamaTransformerHandler` gains `matVecQ2Kraw` and `matVecQ3Kraw` for the CPU inference path, and `dequantizeQ2K` / `dequantizeQ3K` for the GPU upload path (`dequantize()` switch). Without these, the `dequantize()` switch threw `UnsupportedOperationException` for every Q2_K weight tensor, causing garbage output whenever `gpu=true`.

`QuantizationType` gains `Q2_K` (0.328 bpp) and `Q3_K` (0.430 bpp) enum entries for VRAM estimation and model registration.

`LlamaConfig.detectQuantization()` corrected: ftype 10 → Q2_K, 11/12/13 → Q3_K, 14/15 → Q4_K_M. The previous `case 12, 15` mapping was wrong — llama_ftype 12 is `MOSTLY_Q3_K_M`, not `Q4_K_S`. `fromFilename()` gains Q2_K and Q3_K substring checks ahead of Q4/Q5/Q6.

Bug fixed in Q2_K dequantization (all three implementations — `loadQ2_K`, `matVecQ2Kraw`, `dequantizeQ2K`): output slots at offsets 16 and 80 within each 128-element half read from the wrong q-byte half. Correct pattern: `(q[l]>>0, q[l+16]>>0, q[l]>>2, q[l+16]>>2, q[l]>>4, q[l+16]>>4, q[l]>>6, q[l+16]>>6)`.

Six new unit tests in `GgufReaderTest`: Q2_K uniform / zero / max-quants and two-block size; Q3_K max / zero-quants and two-block size.

**Session 21** — Two new deployment fat-jar modules and a unified AWS script.

`juno-node` — new module producing `juno-node.jar` (shade, main class `cab.ml.juno.node.NodeMain`). `NodeMain` and `EmbeddedNodeServer` moved from the `player` module into the `node` module (package `cab.ml.juno.node`). Configuration via system properties (`-Dnode.id`, `-Dnode.port`, `-Dmodel.path`, `-DJUNO_USE_GPU`); command-line args still accepted for backward compatibility. Prints `READY:<nodeId>:<port>` to stdout when the gRPC server is up so `ClusterHarness` can poll it.

`integration` module renamed to `juno-master` — fat jar renamed `juno-master.jar` (main class `cab.ml.juno.master.CoordinatorMain`). `CoordinatorMain` is a new standalone coordinator entry point for remote deployment: reads node addresses from `JUNO_NODE_ADDRESSES`, model path from `JUNO_MODEL_PATH`, and other tuning from env vars (`JUNO_PTYPE`, `JUNO_HTTP_PORT`, `JUNO_DTYPE`, `JUNO_MAX_QUEUE`). No forking, no `ClusterHarness` — nodes must already be running via `NodeMain`. Wires `GenerationLoop`, `RequestScheduler`, `KVCacheManager`, and `InferenceApiServer` then blocks. All integration test classes moved to package `cab.ml.juno.master`; `ModelLiveRunner` promoted to `ModelLiveRunnerIT` and gated behind the `-Pintegration` Maven profile.

`juno-deploy.sh` added under `scripts/aws/` — unified cluster lifecycle script replacing the separate `juno-infra.sh` (GPU) and `juno-infra-ft.sh` (CPU). Hardware auto-detected during bootstrap: GPU instances install CUDA and set `JUNO_USE_GPU=true`; CPU instances skip it. `juno-node.service` (systemd) launches `juno-node.jar` on every node; `juno-coordinator.service` launches `juno-master.jar` on the coordinator after `cluster-nodes.env` is written. Commands: `setup | start | stop | teardown | status | scan-regions`. Options: `--instance-type`, `--node-count`, `--coordinator node1|separate`, `--model-url`, `--ptype`, `--dtype`. State persisted to `~/.juno-deploy-state`. Verified end-to-end on a 3 × m7i-flex.large CPU cluster running TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile with the web console at `http://<coordinator>:8080`.

**Session 20** — GPU inference actually wired end-to-end. Three bugs fixed:
(1) `ForwardPassHandlerLoader.load(Path, ShardContext)` always hardcoded `CpuMatVec.INSTANCE` — a new `selectBackend()` method now reads `JUNO_USE_GPU` and calls `CudaAvailability.isAvailable()`, then delegates to the explicit-backend overload.
(2) Even with `CudaMatVec` correctly injected into `LlamaTransformerHandler.backend`, the inference hot path never used it — `transformerLayer` called the static `matVec(QuantizedTensor, ...)` methods directly. Fix: weights are now dequantized and uploaded to `DeviceFloatMatrix` once at load time; a new `matVecLayer()` instance method dispatches through `backend.sgemv(DeviceFloatMatrix, x)` on the GPU path and falls back to the quantized CPU path otherwise. All 8 projection call sites updated.
(3) `juno.MatVec.backend.cpu.count` was always 0 in JFR metrics because `matVecQuantBackendLabel()` returned `"quantized-q4_k"` etc. — all quantized static matVec ops are pure-Java CPU code, so the label is now `"cpu"` throughout, matching what `CpuMatVec.sgemv()` emits.
`CudaMatVec.upload(float[], int, int)` added as a convenience factory.
`ForwardPassHandlerLoaderSelectBackendTest` (5 tests) and `MatVecQuantizedBackendLabelTest` (3 JFR-event tests) added.

**Session 19** — metrics module, Meta-Llama 3 tokenizer fix, AWS infrastructure scripts.

**Session 18** — `GgufTokenizer` now supports GPT-2 / tiktoken BPE (Llama 3+) in addition to SentencePiece BPE. BPE variant is auto-detected from `tokenizer.ggml.model` in GGUF metadata. Special control tokens (`<|begin_of_text|>`, `<|eot_id|>`, etc.) are pre-split before BPE and always map to single vocabulary IDs. `LlamaTransformerHandler.matVec()` (quantised path) now emits `juno.MatVec` events. `ChatTemplateFormatter.format()` now emits `juno.TemplateFormat` events (both were previously missing instrumentation).

**Session 17** — AWS infrastructure scripts added under `scripts/aws/`: `launcher.sh` (credential wrapper), `juno-infra.sh` (3-node GPU cluster lifecycle with live VRAM/cost dashboard), `juno-infra-ft.sh` (CPU fine-tuning cluster). The `--jfr` flag now embeds the model stem in the recording filename (`juno-<modelStem>-YYYYMMDD-HHMMSS.jfr`).

**Session 16** — naming cleanup: session-12 rename fully applied to source.

The `KVCacheManager` (GPU + CPU tiers with LRU/W-TinyLFU eviction) was previously disconnected from the transformer handlers: `LlamaTransformerHandler` and `Phi3TransformerHandler` each maintained their own private `HashMap<String, float[][]>` with no eviction, making the entire `kvcache` module inert at the node level. This is now fixed.

`NodeKVCacheAdapter` bridges the in-process KV arrays and the cluster-level `KVCacheManager`. After each token position is written, the adapter serialises K and V data into a `KVBlock` and flushes it write-through into the manager's GPU tier (byte-budget LRU) and CPU tier (Caffeine W-TinyLFU). If a local HashMap entry is absent at position > 0 (evicted under JVM heap pressure), the adapter restores it from whichever tier still holds the block. `EmbeddedNodeServer.loadShard()` creates the adapter and wires it into every real handler via `setKvAdapter()`. `evict(requestId)` now propagates through both the local map and all manager tiers.

Four custom JFR event classes cover every hot path. All are readable in JDK Mission Control under Event Browser. Use `--jfr DURATION` on any `juno` command to capture a recording:

- `juno.MatVec` — emitted by `CpuMatVec.sgemv()` and both `CudaMatVec.sgemv()` overloads. Fields: `backend` (`cpu`/`cuda`/`cuda-resident`), `rows`, `cols`. ~155 events per generated token for TinyLlama.
- `juno.ForwardPass` — emitted by all six `ForwardPassHandler.forward()` implementations. Fields: `handlerType` (`llama`/`phi3`/`cpu`/`gpu`/`cyclic`/`lora`), `requestId`, `startPosition`, `layerCount`, `hasOutputProjection`.
- `juno.Tokenizer` — emitted by `GgufTokenizer`, `DJLTokenizer`, `SimpleTokenizer` for `encode`, `decode`, and `decodeToken`. Fields: `tokenizerType`, `operation`, `inputLength`, `outputLength`.
- `juno.TemplateFormat` — emitted by `ChatTemplateFormatter.format()`. Fields: `modelType`, `messageCount`, `outputLength`.

**Session 14** — LoRA fine-tuning + JFR profiling. `LoraTrainableHandler` implements parameter-efficient fine-tuning (LoRA) on top of frozen quantised weights. Adapters live in a separate `.lora` file — the base GGUF is never modified. `LoraAdapterSet` / `LoraAdamOptimizer` handle checkpoint I/O and gradient updates. `LoraTrainEvent` emits custom JFR events (`juno.LoraTrainStep`) with per-step timing breakdown (forward / backward / optimizer ms). `ConsoleMain` gains a `lora` subcommand with `/train`, `/train-file`, `/save`, `/status`, `/reset`, `/merge-hint` REPL commands. `--jfr DURATION` flag added to all three `run.sh` / `run.bat` commands (`cluster`, `local`, `lora`). Root bug fixed: `transposedMatVec` now covers Q5_K (type=13) and Q6_K (type=14) — without this, the output projection backward for TinyLlama Q4_K_M fell into an O(cols) loop that took hours per step.

```
you > /train My name is Dima. I am a Java engineer.
  Training  rank=8 · lr=1.0E-4 · 50 steps · 1 chunk(s) · 10 tokens
  step  50/50   loss=3.12  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%  2100ms/step  ETA 0s
  ✔ done  loss=▼ 3.12 (−3.50)  105s total

you*> /save
  ✔ Saved → /path/to/model.lora  (44 adapters · 4401 KB · 50 steps trained)
```

```
you> are you alive?
bot> Yes, I'm here and ready to help! What do you need?
     [19 tokens · 38420 ms · FLOAT16]   <- phi-3.5-mini, 3-node CPU cluster

you> hello
bot> Hey! Nice to meet you too.
     [6 tokens · 2922 ms · FLOAT16]     <- TinyLlama, 3-node CPU cluster
```
