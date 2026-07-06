## Status

**Session 35** — `--local` mode: fixed `--verbose` no-op and vision models never being detected.

### `--verbose` was a no-op in `--local` mode

`ConsoleMain` silenced `java.util.logging` in a `static { }` initializer, which
runs at class-load time — before `main()` calls `parseArgs()`. It always
observed `verbose=false` regardless of the CLI flag, so `--local --verbose`
produced no log output. `--cluster --verbose` happened to work because its
verbosity check lives in `ClusterHarness`, read at fork time, well after
`parseArgs()` has already run.

**Fix:** moved the logging setup out of the static initializer into a new
`configureLogging()` private static method, called explicitly from `main()`
immediately after `parseArgs()` returns.

Regression test: `ConsoleMainLoggingTest` (drives `configureLogging()` via
reflection and asserts on `LogManager` state).

### Vision models (`/v1/vision/chat`) never detected against real downloaded GGUFs

`LlavaHandlerFactory.isVisionArchitecture()` only ever probed
`--model-path` for the CLIP tensor `v.patch_embd.weight`. Every known public
LLaVA/Qwen-VL/SmolVLM/MiniCPM-V GGUF release ships the CLIP vision encoder in
a **separate** `mmproj-*.gguf` file — the base LLM file never contains
`v.patch_embd.weight`. As a result every real downloaded I2T model was
classified as text-only and `/v1/vision/chat` was never registered, even
though `docs/Vision-I2T.md` assumed (incorrectly) that a single merged GGUF
was the standard format.

**Fix:**

- New `--mmproj-path PATH` CLI flag (`ConsoleMain`, `scripts/run.sh` `local`
  command; `MMPROJ_PATH` env var override).
- Windows parity: `scripts/run.bat` `local` command gains `--mmproj-path`
  (`MMPROJ_PATH` env var) and — a prerequisite discovered while adding
  it — `--api-port` (`API_PORT` env var). `run.bat local` had never
  supported `--api-port` at all; `docs/howto.md` documented a
  `juno.bat local --api-port 8080` example that did not actually work.
  `run.bat cluster` and `run.bat lora` still lack `--api-port` /
  `--mmproj-path`; out of scope here since vision routes are `--local`-only
  (see known limitation below).
- New `VisionModelPaths` record (vision module) resolving which file to open
  for vision tensors: the mmproj file when given, else the model file
  (merged-file fallback). Pure logic, no I/O — unit-tested directly.
- `LlavaHandlerFactory.isVisionArchitecture(Path, Path)` and
  `buildFromHandlers(Path, Path, List, LlamaConfig)` now take an optional
  mmproj path; the old single-argument overloads are kept for backward
  compatibility and delegate with `mmprojPath=null`.
- `docs/Vision-I2T.md`, `docs/agent-arch.txt`, `docs/howto.md`, `README.md`
  updated to describe the two-file reality and the new flag, and to note
  that `--cluster` mode does not yet wire vision routes (`--local` only).

Regression test: `VisionModelPathsTest`.

### `--dtype` silently accepted invalid values (discovered while live-testing the fix above)

`parseDtype()`'s `switch` had no explicit `FLOAT32` case — both an explicit
`--dtype FLOAT32` and any unrecognized garbage (a typo, or an unsupported
quantization label like `INT4`) fell into the same `default` branch, so
`--dtype INT4` was silently coerced to `FLOAT32` with zero feedback. The
startup banner's first line echoes the raw CLI string, so `INT4` appeared to
"work" right up until the second banner line (parsed
`ActivationDtype.toString()`) silently showed `FLOAT32` instead.

Note: `--dtype` only controls the wire format for activations shipped
*between* pipeline nodes — it is unrelated to a GGUF's own weight
quantization, so a `Q4_K`/int4-quantized base model paired with an F16
mmproj file (the normal case for every public LLaVA release) is not a
problem to worry about.

**Fix:** added an explicit `FLOAT32`/`F32`/`FP32` case, and the `default`
branch now prints a `WARNING` to stderr naming the rejected value before
falling back to `FLOAT32`.

Regression test: `ConsoleMainDtypeTest`.

### Any model-name mismatch was a hard 503, even with only one model loaded

Reported while live-testing the vision fix above: `curl .../v1/vision/chat`
with `"model":"llava-v1.5-7b"` (copied from a generic example) returned
`{"error":{"code":"service_unavailable","message":"Model 'llava-v1.5-7b' is
not loaded"}}` even though a model *was* loaded — just under a different id
(the loaded GGUF's filename, e.g. `llava-phi-3-mini-int4.gguf`). The same
exact-match-or-503 logic was independently duplicated three times
(`VisionChatHandler.resolveModel`, `OpenAiChatHandler.resolveModelId`,
`InferenceApiServer.resolveModelId`), so every REST entry point had this trap
in --local mode, where exactly one model is ever loaded and the requested
name can therefore never be ambiguous.

**Fix:** new shared `ModelIdResolver` (registry module):

- no model loaded at all → error, unchanged
- blank/absent `"model"` → resolves to the loaded model, unchanged
- exact match → resolves silently, unchanged
- mismatch with **exactly one** model loaded → falls back to it and logs a
  `WARNING` naming both the requested and actual id (new — this is the fix)
- mismatch with **multiple** models loaded → still an error, now listing the
  loaded ids so the client can self-correct (previously named only the
  rejected id)

All three call sites now delegate to `ModelIdResolver`; each keeps its own
response format (OpenAI-style JSON, the native API's JSON, and the vision
handler's JSON respectively) but shares the same resolution/fallback
decision.

**Correction after this broke `InferenceApiServerTest.blocking_inference_unknown_model_returns_503`:**
that test pins a deliberate, different contract for the native
`/v1/inference` API — an explicitly-requested nonexistent model id is always
an error there, even with exactly one model loaded, because that endpoint is
typically driven by generated clients rather than hand-typed `curl`, so a
mismatch is more likely a real bug than a typo. Universally applying the
single-model fallback was wrong. Fixed by making it opt-in per call site via
`ModelIdResolver.FallbackPolicy`:

- `InferenceApiServer.resolveModelId` → `FallbackPolicy.STRICT` (the
  2-argument `resolve(registry, requested)` overload defaults to this, so no
  code change was needed there beyond making the choice explicit)
- `OpenAiChatHandler.resolveModelId`, `VisionChatHandler.resolveModel` →
  `FallbackPolicy.SINGLE_MODEL_FALLBACK`

Regression test: `ModelIdResolverTest` (registry module, exercises the real
in-memory `ModelRegistry` — no mocking needed). Now covers both policies
explicitly, including a test pinning the exact `InferenceApiServerTest`
scenario (one model loaded, nonexistent id requested, `STRICT` → error).

### `/v1/vision/chat` crashed mid-request: `ArrayIndexOutOfBoundsException: Index 1024 out of bounds for length 1024`

Reported against a real llava-phi-3-mini mmproj file, once the two fixes
above got the request that far. The exception's shape (flat JSON matching
`InferenceApiServer`'s *global* Javalin error handler, not `VisionChatHandler`'s
own nested error format) showed it was escaping uncaught from image/vision
encoding, before inference began — narrowing the search to
`ImagePatchEmbedder`, `VisionEncoder`, `CpuMatVec`, and `GgufReader`'s
tensor loaders. None showed an off-by-one on static review.

**Diagnostic gap fixed first:** `InferenceApiServer`'s global exception
handler only ever logged `e.getMessage()`, never the stack trace, so there
was no way to localize the crash to a specific line without one. Now logs the
full trace via `log.log(Level.WARNING, ..., e)`. This produced the trace
pinpointing `VisionEncoder.mlp():376`.

**Root cause:** `VisionEncoder`'s constructor trusted the GGUF tensor names
`ffn_up`/`ffn_down` to mean "H→I expansion" / "I→H contraction" respectively,
and sized the corresponding bias arrays accordingly (`intermediateSize` and
`hiddenSize`). For this particular mmproj file, `v.blk.{i}.ffn_up.bias` is
actually shaped for the *contraction* direction (length `hiddenSize`=1024,
not `intermediateSize`=4096) — the naming convention is not universally
consistent across mmproj GGUF exports in the wild. `mlp()` then indexed that
1024-length array up to `intermediateSize`=4096, crashing at the boundary.

**Fix:** rather than assume any particular naming convention (unverifiable
without the file, and a wrong guess would just move the bug), `VisionEncoder`
now determines each FFN linear layer's actual direction from its own
GGUF-declared output dimension (`GgufReader.tensorDims`), via a new pure,
I/O-free `resolveFfnOrientation()` — used regardless of which literal tensor
name holds which real shape. Throws a clear, descriptive
`IllegalStateException` at model-load time (not a cryptic mid-request crash)
if neither orientation matches the configured `intermediateSize`/`hiddenSize`
at all.

Regression test: `VisionEncoderTest` (new `resolveFfnOrientation` cases,
including the exact llava-phi-3-mini shape pairing that crashed).

### Same model, next crash: `IllegalArgumentException: A.length=3145728 != rows*cols=786432`

Reported immediately after the FFN fix above got the same request further —
the vision encoder now ran for ~35s (all 23 CLIP blocks) before failing in
the final projector step, `VisionEncoder.project()`.

**Root cause:** same class of bug as the FFN fix, one layer up. `VisionConfig`
read `clip.vision.projection_dim` metadata as 768 and used it to size the
`mm.0.weight` projector matmul (768×1024=786,432 expected elements). The
tensor's own GGUF shape is actually `[1024, 3072]` — 3072 being the LLM's
real hidden dimension (`LlamaConfig.hidden=3072` in the same model), the
width the projector must actually produce so patch embeddings can be spliced
into the LLM's embedding space (786,432 vs the real 3,145,728 = 3072×1024).
The `clip.vision.projection_dim` metadata field is evidently not reliable for
determining the actual mm-projector output width across mmproj exports —
consistent with the FFN naming bug above being a symptom of the same
underlying issue: metadata fields describing this file's architecture can't
be trusted at face value, only the tensors' own shapes can.

**Fix:** new `VisionEncoder.outputDim()`, derived from `mm.0.weight`'s own
measured GGUF shape via a new pure `resolveProjectorOutputDim()` (same
shape-over-metadata approach as `resolveFfnOrientation`), replacing
`VisionConfig.projectionDim()` everywhere a caller needs the projector's real
output width: `VisionEncoder.project()` itself, and both
`LlavaHandlerFactory.buildFromHandlers`/`build()` call sites that construct
`VisionAwareForwardPassHandler` (previously sized from the same unreliable
metadata value). Throws a clear `IllegalStateException` at load time if the
tensor's own shape is inconsistent with `hiddenSize` at all, rather than a
cryptic crash mid-request; logs (not errors on) any metadata/tensor-shape
disagreement, since the tensor always wins.

Regression test: `VisionEncoderTest` (new `resolveProjectorOutputDim` cases,
including the exact 768-vs-3072 mismatch that crashed).

---

## Status (continued)

**Session 36** — F16-weighted GGUF models failed every inference request:
`UnsupportedOperationException: Quantized matVec not implemented for GGML
type 1`.

Reported against `llava-phi-3-mini-f16.gguf` (an F16, i.e. entirely
unquantized, weight file) once vision loading itself succeeded — the crash
happened on the very first LLM forward pass (prefill step 0), so this is a
core `node` module bug, unrelated to the vision fixes above; it would affect
any F16 model's text generation too, image or no image.

**Root cause:** `LlamaTransformerHandler` has two dispatches over
`GgufReader.QuantizedTensor.type()` (the GGML type ID) used when a weight is
kept in raw/quantized form rather than eagerly materialized as `float[]`:
`matVecQuantizedNoEvent` (the CPU compute path) and `dequantize` (used once
per weight when uploading to the CUDA backend). Both handled GGML types 0
(F32), 8 (Q8_0), and 10–14 (Q2_K..Q6_K) — but neither had a case for type 1
(F16), despite F16 being one of the most common, standard GGUF weight
formats, not an exotic one. Any F16 tensor reaching either dispatch hit the
`default` branch's `UnsupportedOperationException`.

**Fix:** added the missing `case 1` branch to both switches:

- `matVecF16raw` (new) for the CPU path, mirroring `matVecF32raw`'s exact
  style (manual little-endian byte assembly per row inside a parallel
  `IntStream`, no per-row `ByteBuffer` allocation)
- `dequantizeF16` (new) for the CUDA upload path, mirroring `dequantizeF32`

Both reuse the existing `GgufReader.f16ToF32` half-to-float conversion —
the same one `GgufReader.loadF16` already uses for eager dequantization — so
a weight kept in raw form now produces bit-identical results to one eagerly
converted via `GgufReader.tensor()`. Byte order is fixed little-endian in
both (matching every other raw matVec in this file): GGUF tensor bytes are
always little-endian regardless of the cluster's `--byteOrder` flag, which
only governs inter-node *activation* serialization, never model weight
bytes — an unrelated concern.

A third dispatch, `LoraTrainableHandler.transposedMatVec`, was already safe:
its `default` branch (`transposedFallback`) logs a warning and reuses
`LlamaTransformerHandler.matVec` internally rather than throwing, so it is
automatically fixed as a side effect of the primary fix above — no separate
change was needed there.

Regression test: `LlamaTransformerHandlerF16MatVecTest` (new — constructs a
synthetic F16 `QuantizedTensor` directly, no GGUF file needed; covers both
`matVec` and `dequantize`, including a numerical-correctness check against a
plain-float reference dot product).

---



### Windows launcher (`scripts/run.bat`, `juno.bat`)

All subcommands (`cluster`, `local`, `lora`, `merge`, `test`) and flags are now working on Windows.

**Root cause fixes:**

- **JAR name mismatch.** `run.bat` referenced `juno-player.jar` and `juno-master.jar` — names that Maven never produces. The actual artifacts are `juno-player-<version>-shaded.jar` and `juno-master-<version>.jar`. Fixed by reading the project version from `pom.xml` at startup using `findstr` and constructing the correct paths dynamically.

- **Java version detection hang.** CMD cannot redirect `stderr` in a pipeline (`2>&1`) reliably inside a `for /f` loop when delayed expansion is active. `java -version` writes to stderr and the output was silently lost, leaving `JAVAVER_RAW` undefined. Fixed by capturing `java -version 2> tmpfile` to a temp file and reading the file with `for /f`.

- **`find_java` nested-if failure.** Nested `if ... (if ... (...))` blocks are not reliable in CMD with `setlocal enabledelayedexpansion`. Replaced with a flat goto-based structure (`find_java_where` label).

- **Infinite loop on empty argument.** In argument-parsing loops, `if exist "%~1"` on an empty `%~1` expands to `if exist ""` which matches the current directory (always true), causing an infinite loop. Fixed by guarding with `if not "%~1"==""` before the `if exist` check in the `cluster`, `local`, `lora`, and `test` parsers.

- **JFR block inside `if not ... (for ...)` silently skipped.** CMD does not support a `for` command inside an `if` parenthesized block when delayed expansion is on. Replaced with a goto-based pattern (`lora_jfr_skip` / `test_jfr_skip` labels).

**Documentation updated:**

- `README.md` — Windows launcher note in section 2.2, Windows requirements paragraph, `juno.bat` references for `merge`.
- `docs/howto.md` — Windows note at top; Windows command-prompt examples added to every subcommand section (`local`, `cluster`, `lora`, `merge`) and Build and Test.

---

## Status

**Session 33** — Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development.

### Supported model status (docs)

User-facing docs now state a single, consistent model-support policy:

| Family | `general.architecture` | Status |
|--------|------------------------|--------|
| LLaMA, Mistral, TinyLlama, … | `llama`, `mistral`, … | Supported via `LlamaTransformerHandler` |
| Phi-3 / Phi-3.5 | `phi3` | **Supported** via `Phi3TransformerHandler` |
| Gemma | `gemma` | **Under development** (`LlamaTransformerHandler` + `gemma` template) |
| Qwen 2 / 2.5 | `qwen2` | **Under development** (Llama handler + QKV bias groundwork) |
| Qwen3 dense | `qwen3` | **Under development** (`Qwen3TransformerHandler` in progress) |
| Qwen3-MoE | `qwen3moe` | **Under development** (`Qwen3MoeTransformerHandler` in progress) |
| Qwen3.5 | `qwen35` | **Under development** (hybrid DeltaNet; separate handler) |

**Updated files:**

- **`README.md`**, **`RELEASE_NOTES.md`** — Supported models section
- **`docs/arch.md`** — handler routing and tokenizer notes
- **`docs/features.md`**, **`docs/howto.md`**, **`docs/LoRA.md`** — Phi-3 OK for inference; Gemma and Qwen paths not production-ready; LoRA still LLaMA-family (+ Phi-3 template detection)
- **`docs/phi3-inference-handoff.md`** — status set to supported (retains debug handoff notes)
- **`docs/model_support_summary_972ab30f.plan.md`** — roadmap, dispatch table, chat matrix, gaps, decisions log

**Policy:** Phi-3 is production-ready in docs and validation (local + cluster). Gemma and all Qwen families remain under development until dedicated validation lands.

---

## Status

**Session 32** — ROCm/HIP backend for AMD GPU inference via Panama FFI.

### AMD GPU support (ROCm/HIP + rocBLAS)

Full first-class AMD GPU support alongside the existing NVIDIA CUDA backend. The GPU
abstraction layer auto-selects CUDA > ROCm > CPU at startup with no configuration required.
Tested on AMD Radeon RX 7900 XT (gfx1100, ROCm 7.2.x).

**New production classes (`node` module):**

- **`GpuBindings`** — vendor-neutral interface implemented by `CudaBindings` and `RocmBindings`.
  Exposes all device runtime and BLAS handles as `MethodHandle` accessors, shared constants
  (`H2D`, `D2H`, `STREAM_NON_BLOCKING`), and static helpers (`check`, `callInt`, `loadLibrary`,
  `bind`). Static helpers eliminate per-implementation boilerplate.
- **`GpuMatVec`** — sealed interface (`permits CudaMatVec, RocmMatVec`) extending `MatVec`.
  Exposes `upload(float[], int, int)` and `uploadHalf(float[], int, int)` so transformer
  handlers depend on the GPU abstraction rather than a concrete vendor class.
- **`RocmBindings`** — Panama FFI downcall handles for `libamdhip64.so` and `librocblas.so`.
  Pre-binds `hipHostMalloc flags=0` via `MethodHandles.insertArguments` to match the
  `cudaMallocHost` arity visible to all callers. Key ROCm constants: `opTranspose()=112`
  (`rocblas_operation_transpose`), `hipDeviceProp_t` sizeof=1472, name@0, totalGlobalMem@288
  (measured from ROCm 7.2.x headers, Linux x86_64).
- **`RocmAvailability`** — HIP device detection: `isAvailable()`, `deviceCount()`,
  `deviceName(int)`, `vramBytes(int)`. Mirrors `CudaAvailability` in structure.
- **`RocmMatVec`** — `MatVec` / `GpuMatVec` implementation backed by `rocblas_sgemv` (FP32)
  and `rocblas_hssgemv_strided_batched` (FP16). Three compute paths:
  - Host FP32: temporary device buffers per call; synchronous H2D → kernel → D2H.
  - Device-resident FP32 (`DeviceFloatMatrix`): per-thread scratch for x/y; async stream copies.
  - Device-resident FP16 (`DeviceHalfMatrix`): x converted FP16 in off-heap arena; FP32 accumulation.
  Off-heap `Arena.ofConfined()` staging for all H2D/D2H copies — required by Java 25 Panama
  (heap segments rejected by native downcalls).
- **`MatVecBackend`** — enum replacing ad-hoc string literals for the `juno.MatVec.backend` JFR
  dimension. Values: `CPU`, `CUDA`, `CUDA_RESIDENT`, `CUDA_RESIDENT_FP16`, `ROCM`,
  `ROCM_RESIDENT`, `ROCM_RESIDENT_FP16`. Label strings are part of the JFR contract and unchanged.

**Modified production classes:**

- **`GpuContext`** — refactored from CUDA-only to backend-agnostic. Adds `GpuBindings bindings`
  field, `bindings()` accessor, `selectBindings()` (CUDA → ROCm priority order with
  `-Djuno.gpu.backend=cuda|rocm|auto` override), `createMatVec()` factory, `backendLabel()`
  delegate. `close()` uses `bindings.cublasDestroy()` instead of hardcoded CUDA call.
  Private `deviceName()` and `deviceVram()` helpers use `GpuBindings` struct-offset accessors.
- **`CudaBindings`** — adds `implements GpuBindings`; 20 accessor methods expose the existing
  `MethodHandle` fields to vendor-neutral callers. Zero existing fields or constants removed.
- **`CudaAvailability`** — field-access calls updated to use `CudaBindings.instance()` accessor
  methods (`PROP_NAME_OFFSET` → `instance().PROP_NAME_OFFSET`, etc.).
- **`CudaMatVec`** — implements `GpuMatVec` (was `MatVec`); `upload` / `uploadHalf` made public
  with `@Override`; backend labels replaced by `MatVecBackend` enum calls.
- **`DeviceFloatMatrix` / `DeviceHalfMatrix`** — direct `CudaBindings.instance()` field access
  replaced by `GpuContext#bindings()` method calls (`GpuBindings`). Both classes now work
  identically on CUDA and ROCm. `DeviceHalfMatrix` caches `gpu = ctx.bindings()` at construction.
- **`LlamaTransformerHandler`** — `instanceof CudaMatVec` → `instanceof GpuMatVec` for weight
  upload gate; `cudaMalloc` OOM message check extended to also catch `hipMalloc`;
  `matVecQuantBackendLabel(int)` → `matVecQuantBackend(int)` returns `MatVecBackend.CPU`.
- **`Phi3TransformerHandler`** — same `instanceof` fix; OOM check extended to `hipMalloc`.
- **`LoraTrainableHandler`** — same `instanceof` fix.
- **`ForwardPassHandlerLoader`** — `pickMatVec` checks both `CudaAvailability` and
  `RocmAvailability`; device count query reads from the available backend; `GpuContext.shared(dev).createMatVec()` replaces `new CudaMatVec(...)`.
- **`EmbeddedNodeServer`** — uses `gpuContext.createMatVec()` and `gpuContext.backendLabel()`
  for log messages.
- **`ConsoleMain` / `JunoPlayer`** — `new CudaMatVec(gpuCtx)` → `gpuCtx.createMatVec()`.
- **`MatVecEvent`** — adds `backend(MatVecBackend)` setter to avoid hand-written label strings
  at call sites; public `String backend` field kept for JFR contract.

**New tests (55 total, 0 failures on RX 7900 XT):**

- `RocmMatVecTest` (30) — extends `MatVecBackendContractTest` for full API parity; correctness
  vs CPU reference at 2048×2048, 5632×2048, 32000×2048; trivial known-value cases;
  4-thread concurrent safety; throughput sanity.
- `RocmAvailabilityTest` (8) — device detection present/absent; name format; VRAM bounds;
  out-of-range index fallbacks.
- `GpuContextTest` +5 `@Tag(rocm)` — ROCm context lifecycle, backend priority,
  `createMatVec` factory, shared singleton, system-property override.
- `ForwardPassHandlerLoaderSelectBackendTest` +2 `@Tag(rocm)` — `RocmMatVec` routing,
  process-wide `GpuContext.shared(0)` reuse.
- `ForwardPassHandlerLoaderSelectLoraBackendTest` +1 `@Tag(rocm)` — LoRA routing on ROCm.
- `MatVecQuantizedBackendLabelTest` — updated to use `MatVecBackend` enum constants.

Run ROCm-tagged tests:
```bash
mvn test -pl node -Dgroups=rocm
```

**Performance (RX 7900 XT, ROCm 7.2.x):**

| Shape | Path | Time (5 runs) |
|-------|------|--------------|
| 32000×2048 | `rocblas_sgemv` host FP32 | 408 ms |

All existing 194 unit tests pass unchanged.

---

## Status

**Session 31** — Panama FFI for Juno math: JavaCPP / bytedeco removed, CUDA bindings rewritten with `java.lang.foreign`.

### Panama FFI GPU bindings (`node` module)

The entire CUDA bridge has been rewritten using the Java 25 Panama Foreign Function & Memory API
(`java.lang.foreign.Linker`, `SymbolLookup`, `MemorySegment`, `Arena`). The `org.bytedeco:cuda-platform`
dependency has been removed from `node/pom.xml`.

**New production class:**

- **`CudaBindings`** — Panama FFI downcall handles for `libcudart.so.12` and `libcublas.so.12`.
  Resolves all CUDA Runtime and cuBLAS symbols once at class-init time via `Linker` and
  `SymbolLookup`; resulting `MethodHandle` instances are thread-safe with zero per-call Java overhead.
  Exposes: `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice`, `cudaMalloc`,
  `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMemcpy`, `cudaMemcpyAsync`,
  `cudaStreamCreateWithFlags`, `cudaStreamSynchronize`, `cudaStreamDestroy`,
  `cublasCreate`, `cublasDestroy`, `cublasSetStream`, `cublasSetPointerMode`,
  `cublasSgemv`, `cublasHSSgemvStridedBatched`.
  `cudaDeviceProp` struct-offset constants (`DEVICE_PROP_BYTES=1512`, `PROP_NAME_OFFSET=0`,
  `PROP_TOTAL_MEM_OFFSET=288`) measured from CUDA 12.x headers on Linux x86_64.
  Singleton init: `CudaBindings.instance()` / `CudaBindings.isAvailable()`.

**Modified production classes:**

- **`CudaMatVec`** — all JNI / JavaCPP call sites replaced with `CudaBindings` downcall handles.
  Native memory managed exclusively via `MemorySegment` and `Arena`. Device weight matrices
  (`DeviceFloatMatrix`, `DeviceHalfMatrix`) held resident; `MemorySegment` passed directly to
  cuBLAS as `ADDRESS` — zero H2D copy per token. Per-thread `Fp32Scratch` / `Fp16Scratch`
  scratch on device grown lazily and reused. FP16 x staging packed with `Float.floatToFloat16`
  into a confined off-heap arena in the hot path.
- **`GpuContext`** — cuBLAS handle stored as `MemorySegment` (opaque `cublasHandle_t`); created
  and destroyed via `CudaBindings`. `cublasSerializationLock()` serializes stream-binding and
  kernel submission on the shared handle. `shared(int)` returns a process-wide singleton per
  device index.
- **`DeviceFloatMatrix`** — device memory allocated via `CudaBindings.deviceMalloc`; backing
  `MemorySegment` sized to `rows * cols * 4` bytes; H2D via synchronous `cudaMemcpy`.
- **`DeviceHalfMatrix`** — same pattern; FP16 x staging via confined arena; `MemorySegment.ofArray`
  pins heap array for duration of downcall.
- **`CudaAvailability`** — device detection updated to use `CudaBindings` downcall handles.

**`node/pom.xml`:** `org.bytedeco:cuda-platform` dependency removed.
`maven-surefire-plugin` `argLine` updated: `--enable-native-access=ALL-UNNAMED`,
`--add-opens java.base/java.lang=ALL-UNNAMED`, `--add-opens java.base/java.nio=ALL-UNNAMED`.

**New test: `CudaBindingsTest`** — two scenarios:
- CUDA present (`@Tag("gpu")`): every `MethodHandle` non-null, singleton loads cleanly.
- CUDA absent (CPU-only CI): `isAvailable()` returns false, `instance()` throws `IllegalStateException`.

Run GPU-tagged tests: `mvn test -Dgroups=gpu -pl node`

All existing tests pass unchanged.

---

## Status

**Session 30** — Maven Central publish configuration.

### Maven Central publish (`pom.xml`, all module POMs)

All modules configured for publishing to `central.sonatype.org` via the Central Portal publisher.
Version set to `0.1.0-RC` across root POM and `juno-bom`.

**Changes:**

- **`maven-source-plugin 3.3.1`** — `attach-sources` execution at `verify` phase; produces `-sources.jar`
  required by Maven Central.
- **`maven-javadoc-plugin 3.11.2`** — `attach-javadocs` execution at `verify` phase; `doclint=none`,
  `failOnError=false`; produces `-javadoc.jar` required by Maven Central.
- **`maven-gpg-plugin`** — `sign-release` execution moved from `verify` to `install` phase so
  sources and Javadoc jars are already attached before signing. `--pinentry-mode loopback`
  added to `gpgArguments` to allow `-Dgpg.passphrase=...` without a GUI pinentry agent.
- **`distributionManagement`** — `<repository>` and `<snapshotRepository>` wired to
  `central.sonatype.org` Central Portal publisher endpoint.
- **Developer / SCM metadata** — `<organization>Machine Learning Cabinet</organization>`,
  `<organizationUrl>https://ml.cab/</organizationUrl>`, SCM tag updated to `v0.1.0-RC`.
- **All module POMs** — publish config consolidated into root POM; per-module boilerplate removed.

---

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
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 200 |
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

**Session 34** — Windows launcher fixed: `run.bat`/`juno.bat` fully functional; docs updated with Windows examples. *(this session)*

**Session 33** — Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development. *(unchanged)*

**Session 24** — Configurable activation byte order (`--byteOrder BE|LE`). *(unchanged)*

**Session 22** — Q2_K and Q3_K quantization support. *(unchanged)*

**Session 21** — Two new deployment fat-jar modules and a unified AWS script. *(unchanged)*

**Session 20** — GPU inference actually wired end-to-end. *(unchanged)*

**Session 19** — metrics module, Meta-Llama 3 tokenizer fix, AWS infrastructure scripts. *(unchanged)*

**Session 18** — GPT-2 BPE tokenizer, JFR instrumentation fixes. *(unchanged)*

**Session 17** — AWS infrastructure scripts. *(unchanged)*

**Session 14** — LoRA fine-tuning + JFR profiling. *(unchanged)*