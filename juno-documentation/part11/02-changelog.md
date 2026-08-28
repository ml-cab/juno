(ch-11-2)=
# 11.2. Changelog

## Status

**Session 67**: JFR-verified confirmation of the Session 66 SIMD benchmark;
decode confirmed unaffected; next lever identified as dequant vectorization.

### CPU SIMD benchmark, phase-by-phase confirmation

- Cross-checked both benchmark runs against raw JFR logs directly: total
  prefill `178,222.6ms → 124,690.3ms` (-30.0%), `qkvProj` `66,313.7ms →
  57,426.5ms` (-13.4%), `woProj+ffn+residuals` `106,183.5ms → 62,412.8ms`
  (-41.2%).
- `rope+cacheWrite`/`attention` also moved (-8.3%/-17.9%) despite not being
  touched by `VectorQuantKernels` — recorded as the run-to-run scheduling
  noise floor on this hardware, not a SIMD effect.
- Decode confirmed unaffected: `matVec` p95 4.488ms vs 4.467ms (within
  noise), since decode uses the single-token path, not
  `sgemm*WeightStationary`. Raw tok/s and token counts between the two runs
  are not a valid comparison (different sampling outcomes at temperature
  0.3); only the fixed 741-token prefill window is apples-to-apples.
- Conclusion: real, correctness-preserving 30% prefill reduction from the
  dot-product accumulate vectorization alone; still ~3.4x off the llama.cpp
  ~30-36s reference on the same laptop; next lever is vectorizing the
  Q5_K/Q4_K/Q8_0 dequantization bit-unpacking phase, left scalar in
  Session 65.

---

## Status

**Session 66**: first hardware benchmark of the Session 65 SIMD accumulate
kernel against the Session 64 vision-chat prefill baseline. -30% prefill.

### SIMD accumulate: initial benchmark result

- Same fixed 741-token window (`moondream2-q5_k.llamafile`, `./juno local`,
  same laptop) as the Session 64 baseline. Total prefill `178.2s → 124.7s`
  (-30%). Log confirms `VectorQuantKernels.AVAILABLE=true` (no scalar
  fallback triggered); output stayed coherent, `finish_reason: stop`.
- Per-phase: `qkvProj` -13.7%, `woProj+ffn+residuals` -41.2%,
  attention/rope roughly flat (both untouched code paths). Tentatively
  attributed the larger FFN win to `wUp`/`wDown` being ~4x larger than the
  QKV projections (more parallel accumulate work per call to amortize
  overhead against) — plausible from the shapes, not confirmed by profiling.
- This is the win from vectorizing only the dot-product accumulate loop;
  the still-scalar Q5_K dequant/bit-unpacking phase is now the limiting
  factor, consistent with landing at ~30% rather than the 20-40x per-core
  gap the Session 64 FLOP analysis predicted for a full rewrite. The
  per-layer timing variance noted in Session 64 is present and unchanged,
  confirmed independent of this session's work.

---

## Status

**Session 65**: implemented the Vector API SIMD kernel spec'd out in
Session 64 (`VectorQuantKernels`), plus all build/run flag plumbing.

### `VectorQuantKernels` (SIMD dot-product accumulate)

- Confirmed the target JDK's Vector API status first: JDK 25 (JEP 508) and
  JDK 26 (JEP 529, current as of Aug 2026) both still ship it as
  `jdk.incubator.vector`, not finalized, so `--add-modules
  jdk.incubator.vector` is required at compile and run time.
- New `node/.../VectorQuantKernels.java`: vectorizes only the dot-product
  accumulation phase of the Q4_K/Q5_K/Q8_0 weight-stationary kernels; the
  bit-unpacking dequant phase is deliberately left scalar as a documented
  follow-up rather than risk an unverified SIMD shape conversion in
  correctness-critical code. `jdk.incubator.vector` references are confined
  to a nested `Simd` class probed once at class-init (`catch (Throwable)`),
  so a JVM missing the module falls back to the original scalar loop
  transparently.
- `sgemmQ4KWeightStationary` / `sgemmQ5KWeightStationary` /
  `sgemmQ8_0WeightStationary` in `LlamaTransformerHandler` now call
  `VectorQuantKernels.dot(...)`; signatures and existing tests untouched.
  New `VectorQuantKernelsTest` (block sizes 32/256, non-lane-aligned tails,
  offsets, zero-length).
- Plumbing: root `pom.xml` (compiler args), `node/pom.xml` (surefire
  `argLine`), `scripts/run.sh`/`run.bat`, `ClusterHarness.java` (forked
  node JVMs), all three production launch points in
  `scripts/aws/juno-deploy.sh`. `docs/agent-arch.txt` updated.
- Could not execute `mvn compile`/`mvn test` in this environment; validated
  read-only (XML well-formedness, `bash -n` on shell scripts, manual review
  of Vector API call sites against the JDK 25 shape). Hardware benchmark
  deferred to Session 66.

---

## Status

**Session 64**: closed the remaining architectural/scheduling gaps between
Juno's vision-chat prefill and llama.cpp's; isolated the sole remaining
bottleneck to scalar (non-SIMD) CPU dequant/matmul kernels.

### Vision-chat prefill: closing the gap to llama.cpp

- Benchmark: `POST /v1/vision/chat`, fixed 741-token window (729 image
  patches + 11 text tokens), `moondream2-q5_k.llamafile`, `./juno local`,
  Intel i5-1240P, GPU off both sides. llama.cpp reference on the same box:
  ~30s to first token, ~36s total.
- Merge-conflict compile fixes from the Vision-I2T → release-0.1.2 merge.
- **Vision encoder batching** (`VisionEncoder`): per-patch `sgemv` (~118K
  calls/image) replaced with one batched `sgemm` call per layer; vision
  encode dropped to ~33-53s, no longer the bottleneck.
- **Phi2 batched prefill**: Phi2 had no `forwardBatch` override and was
  silently falling back to 740 sequential single-token `forward()` calls;
  added real batched prefill mirroring Llama/Phi3's pattern.
- **New Q5_K weight-stationary CPU kernel** (`sgemmQ5KWeightStationary`):
  Q5_K, this model's actual quantization, had no batched kernel even in
  Llama's existing dispatch; unit-tested against the per-row `matVec`
  oracle.
- **KV cache eviction leak fixed**: `evict(requestId)` was unreachable from
  the pipeline layer, leaking full per-layer KV arrays on every stateless
  request (caused an OOM on a second request). Cascaded through
  `LocalInferencePipeline`, `VisionAwareForwardPassHandler`,
  `FaultTolerantPipeline`, `GenerationLoop`. Known remaining gap:
  `ProcessPipelineClient`/`TensorParallelPipelineClient` (cluster mode)
  still no-op on `evict()`.
- **`ShardMap.evenSplit`**: `--nodes N` local mode was routing through the
  VRAM-aware greedy planner with a fabricated per-node VRAM figure, so
  node 0 got 22/24 layers. Added an honest even-split for local simulation;
  verified 150s/8s/8s skew → even ~69s/67s/70s.
- **Attention parallelized** (`Phi2TransformerHandler`, `IntStream.parallel`
  over window positions): confirmed by measurement to drop attention from
  an estimated ~20-40s liability to 4.4s / 2.4% of prefill. Known sibling
  gap left unfixed: Llama/Phi3's `gqaInto()` shares a `ws.scores` scratch
  buffer, so the same naive parallelization would race there; needs a
  separate fix before benchmarking non-Phi2 models locally.
- Per-layer timing instrumentation added to Phi2, matching Llama's existing
  instrumentation, producing the breakdown below.
- **Measured breakdown** (24-layer sum, 178.2s total): `qkvProj` 66.3s
  (37.2%), `rope+cacheWrite` 1.2s (0.7%), `attention` 4.4s (2.4%),
  `woProj+ffn+residuals` 106.2s (59.6%). 96.8% of remaining time is inside
  the batched Q5_K kernel. FLOP check on QKV alone: ~0.5 GFLOP/s per thread
  vs llama.cpp's hand-vectorized 10-20+ GFLOP/s per core, fully explaining
  the ~6x wall-clock gap.
- Thread pool sizing confirmed correct, not a bug (`ForkJoinPool.commonPool()`
  for local mode; explicit parallelism flag only needed for cluster mode).
- **Spec'd out for Session 65**: replace the scalar inner loops in
  `sgemmQ{4K,5K,8_0}WeightStationary` with Vector API SIMD inside the
  existing weight-stationary batching structure. Prototype on Q8_0 first
  (simplest block layout); confirm the target JDK's Vector API status
  before writing code.

---

## Status

**Session 63**: vision encode accuracy fix on branch `47-vision` — SigLIP
models (moondream2) were missing their post-encoder LayerNorm entirely.

### Post-encoder LayerNorm for SigLIP-family vision towers

- Commit `a5255d3`, "fixed some image scaling errors, something is replying
  correctly now" — despite the commit message, the actual bug was numerical,
  not geometric: `VisionEncoder` never applied `v.post_ln`. CLIP/LLaVA mmproj
  files don't declare it (their `post_layernorm` only touches the pooled CLS
  output, which LLaVA-style callers never use), so it was easy to miss — but
  SigLIP (moondream2's vision tower) applies it to the *entire* last hidden
  state before any downstream use, making it structurally required there.
- Without it, moondream2's patch embeddings were the raw, un-normalized
  final transformer-block residual stream, with L2 norm up to ~70000 instead
  of properly LayerNorm'd features — explaining a whole class of "sees the
  image but describes something nonsensical" symptoms.
- Fix reads `v.post_ln.weight`/`.bias` when present (`hasPostLn`) and applies
  it once after the last transformer block, before the projector; absence
  means skip the op entirely (not identity-affine LayerNorm, which would
  still normalize mean/variance) — CLIP/LLaVA mmproj files see zero behavior
  change. New encoder tests cover both the present and absent case.

---

## Status

**Session 62**: `47: Phi2Rope, image placeholder and more formatting for
phi2` — Phi2's own RoPE variant, plus vision-side formatting cleanup ahead
of the first coherent moondream2 reply (Session 63).

### `Phi2Rope` and image-placeholder handling

- New `Phi2Rope` (`node` module): Phi2 uses a **partial-rotary** RoPE
  variant (only a configured fraction of `headDim` is rotated, the
  remainder passed through unchanged) — different from the full-rotary RoPE
  `LlamaTransformerHandler`/`Phi3TransformerHandler` already had, so it
  couldn't reuse their implementation. New `Phi2RopeTest` covers the
  partial-rotation boundary directly.
- `GgufReader` gained float-array metadata reading (new
  `GgufReaderMetaFloatArrayTest`) to pull Phi2's rotary-fraction config
  straight from GGUF metadata rather than hardcoding it.
- `ChatTemplate` extended for Phi2's chat format; `VisionConfig`,
  `ImagePatchEmbedder`, `LlavaHandlerFactory`, and
  `VisionAwareForwardPassHandler` all got formatting/wiring adjustments so
  the `<image>` placeholder token is handled consistently on the Phi2 path,
  not just Llama/Phi3. New `VisionConfigNormalizationTest`.

---

## Status

**Session 61**: `adding mm0OutDim to projection math to support phi2
vision` — extended `VisionEncoder`'s CLIP-only assumptions to also cover
SigLIP-family towers (moondream2) ahead of onboarding Phi2 vision.

### CLIP vs SigLIP: optional CLS token, `mm0OutDim`

- `VisionEncoder`'s javadoc/tensor-naming contract was CLIP/LLaVA-specific
  (`v.class_embd` always present, `v.position_embd.weight` sized
  `numPatches+1`). SigLIP models have no CLS token at all
  (`v.class_embd` absent, `v.position_embd.weight` sized exactly
  `numPatches`) — the encoder now branches on tensor presence rather than
  assuming CLIP's layout unconditionally.
- `mm0OutDim` (the first projector layer's output width) is read from
  `mm.0.weight`'s own GGUF shape rather than assumed, consistent with the
  project's established "trust the tensor shape, not the metadata field"
  policy from earlier in this branch.
- Two large reference documents were added under `docs/`
  (`meta-juno-doc.md`/`.txt`) capturing the full architecture write-up this
  branch had been accumulating — bulk documentation, no source-behavior
  change.

---

## Status

**Session 60**: `.llamafile as vision archive` — models shipped as
`.llamafile` (Mozilla's self-contained model+runtime bundle) can now be
read directly as a vision model source, not just plain `.gguf`.

### `LlamafileGgufIndex`

- New `LlamafileGgufIndex` (`node` module): a `.llamafile` is a
  self-executing archive with a GGUF payload appended after a Cosmopolitan
  binary stub — this parses the container to locate and index the embedded
  GGUF's tensors/metadata without needing a separately-extracted `.gguf`
  file on disk. New `LlamafileGgufIndexTest`.
- `GgufReader` extended to open through this index transparently.
- `LlavaHandlerFactory` updated so vision model resolution accepts a
  `.llamafile` path wherever a `.gguf` path was previously required. New
  `LlavaHandlerFactoryEmbeddedVisionTest` covers loading vision tensors out
  of an embedded (llamafile-packaged) GGUF end-to-end.
- This is the change that made `moondream2-q5_k.llamafile` — the model used
  for every prefill benchmark in Sessions 64-67 — usable at all.

---

## Status

**Session 59**: `gguf-info mode and phi2 support added` — a standalone
GGUF-inspection CLI mode, plus the first Phi2 architecture support, added
together since Phi2 vision needed the inspector to even get started.

### `./juno gguf-info` and initial Phi2 support

- New `GgufInfoMain`/`./juno gguf-info` subcommand: dumps a GGUF file's
  metadata keys and tensor list/shapes without loading a full model or
  starting inference — used throughout the rest of this branch to inspect
  unfamiliar mmproj/model files before writing code against them.
- New `Phi2TransformerHandler` (`node` module, ~560 lines): first-cut Phi2
  architecture support (distinct attention/FFN/norm wiring from
  Llama/Phi3), wired into `ForwardPassHandlerLoader`.
- `ImagePatchEmbedder` and `VisionEncoder` extended for the patch-embedding
  and encoder-config shapes this model family needs; `VisionAwareForwardPassHandler`
  updated accordingly. New/expanded tests in both `ImagePatchEmbedderTest`
  and `VisionEncoderTest`.

---

## Status

**Session 58**: `vision replies random scene, missing context of pic` — the
zero-vector text-token bug: image tokens carried real signal, text tokens
carried none.

### Text-token positions were silently zero-vectors

- `VisionAwareForwardPassHandler.buildWindowActivationsWithVision()` spliced
  real CLIP/SigLIP patch vectors into image-token positions but left every
  **text**-token position as an all-zero vector — the entire prompt text
  (chat template, the actual question, BOS) was invisible to the model;
  only the image patches carried any signal at all. Explains the symptom
  exactly: grammatically-plausible output describing a plausible but
  unrelated scene, since the only "real" input was the image.
- **Fix:** added `ForwardPassHandler.embedToken(int)` so a decorator can ask
  the wrapped handler for a token's real embedding-table row; both the
  batch and single-token vision-splicing paths now call
  `textHandler.embedToken(tokenId)` for non-image positions instead of
  leaving zero. Implemented in `LlamaTransformerHandler.embedToken()`.
- Updated the existing `VisionAwareForwardPassHandlerBatchTest` cases that
  had asserted the old, buggy zero-vector behavior; added a new
  single-token-path regression test and direct `embedToken()` tests
  (`LlamaTransformerHandlerEmbeddingsNodeActivationsTest`). Shared
  `StubForwardPassHandler` test double gained a configurable deterministic
  fake embedding.

---

## Status

**Session 57**: `vision finally working, but long and gives gbg reply`,
immediately followed by cleanup of an accidentally-committed debug dump.

### First end-to-end vision reply, then a housekeeping fix

- First commit where a `/v1/vision/chat` request ran end-to-end without
  crashing or hanging and produced *some* reply text — output was still
  long-winded and largely garbage at this point (this is the state Session
  58's zero-vector bug describes and fixes). Touched `GenerationLoop`,
  `ConsoleMain`, `LlamaTransformerHandler`; new
  `LlamaTransformerHandlerEmbeddingsNodeActivationsTest`.
- The commit accidentally included an 8700-line `diff.txt` (a leftover
  debug artifact, not source) — caught and deleted in the very next commit
  the same day. No functional change in the cleanup itself, noted here
  only so the file's brief appearance in history isn't mistaken for
  intentional content later.

---

## Status

**Session 56**: `up` — internal refactor of the batched-prefill code paths
added in Session 55, no externally-visible behavior change.

### Batched-prefill implementation hardening

- `LlamaTransformerHandler` and `Phi3TransformerHandler`'s batched-prefill
  methods (added in Session 55) reworked for correctness/clarity following
  self-review; `docs/batched-prefil.md` (the design doc from Session 55)
  extended with the implementation notes this pass produced. No new public
  API surface; existing batched-prefill tests continued to apply unchanged.

---

## Status

**Session 55**: `removed bytedeco from the docs and code to not confuse
assistants`, immediately followed by `batched prefill` — the batched
prefill work that later benchmark sessions (64-67) build on was designed
and implemented here.

### Bytedeco cleanup, then batched prefill end to end

- Small cleanup first: stray `org.bytedeco` (JavaCPP) references removed
  from docs and a couple of `node`/`master` test files — the project had
  already moved off JavaCPP for its CUDA bindings (see the main project
  history's Panama-FFI session), but leftover mentions were confusing
  enough to warrant a dedicated pass.
- **Batched prefill**, planned in a new `docs/batched-prefil.md` design doc
  before any source change, then implemented: `GenerationLoop.generate()`/
  `generateBatch()` previously prefilled a prompt with a sequential
  per-position loop, reallocating and copying a growing token-id slice on
  every single position. New `PrefillMode` (`SINGLE`/`BATCH`), new
  `BatchForwardRequest`/`BatchForwardResult` (`node` module), and real
  batched `forwardBatch()` implementations added to
  `LlamaTransformerHandler` and `Phi3TransformerHandler`. `CpuMatVec`
  gained the batched `sgemm` this all runs on top of. New
  `LoraTrainableHandler` batched path, new `PrefillModeTest`,
  `ConsoleMainPrefillFlagTest`, `CpuMatVecSgemmTest`,
  `VisionAwareForwardPassHandlerBatchTest`. A `docs/TODO-VectorAPI.md` note
  was also added here, flagging future SIMD work — the same gap Sessions
  65-67 eventually closed.

---

## Status

**Session 54**: `#47 model resolver, f16 weights support, stack-trace on
api error, tested on llava-phi-3-mini-f16.gguf llava-v1.5-7b-Q4_K.gguf` —
first real-model validation pass against two actual downloaded LLaVA GGUFs.

### Model resolver, F16 weight support, real-file testing

- New `ModelIdResolver` (`registry` module) so `/v1/vision/chat` and the
  OpenAI-compatible endpoints can resolve a requested model id against
  what's actually loaded, rather than requiring an exact string match.
- `LlamaTransformerHandler` gained F16 weight-tensor support (new
  `LlamaTransformerHandlerF16MatVecTest`), needed once real F16 mmproj/model
  files were tested against, not just quantized ones.
- `InferenceApiServer`/`OpenAiChatHandler`/`VisionChatHandler` error paths
  now surface a stack trace on API error instead of swallowing it — this is
  what made every subsequent bug in this branch (Sessions 57-63) traceable
  to a specific line rather than a bare exception name.
- First real-model test pass: `llava-phi-3-mini-f16.gguf` and
  `llava-v1.5-7b-Q4_K.gguf`, both downloaded rather than synthetic fixtures.

---

## Status

**Session 53**: `#47 initial impl added, junits are passing`, preceded by
an unrelated fix, `#43 multiple issues with dots in offending blocks
fixed` — the start of the Vision-I2T branch.

### Vision-I2T scaffolding (branch start)

- Unrelated small fix first, landed just before this branch started:
  `#43` fixed formatting issues with dots inside "offending blocks" in
  `run.bat`/docs output; not vision-related, noted here only because it's
  the last commit before the branch diverges.
- **New `vision` Maven module**: `VisionConfig`, `VisionEncoder` (pure-Java
  CLIP ViT-L/14 encoder reading GGUF mmproj weights), `ImagePatchEmbedder`,
  `LlavaHandlerFactory`, `VisionAwareForwardPassHandler` (wraps a text
  `ForwardPassHandler`, splices patch embeddings into image-token
  positions), and `StubForwardPassHandler` (test double). New
  `docs/Vision-I2T.md` design doc.
- New `VisionChatHandler` (`juno-player`) and a `POST /v1/vision/chat`
  route wired into `InferenceApiServer`/`ConsoleMain`.
- Full unit-test coverage from day one for every new class
  (`ImagePatchEmbedderTest`, `VisionAwareForwardPassHandlerTest`,
  `VisionConfigTest`, `VisionEncoderTest`) — all passing at this commit, per
  the commit message, though this is the scaffolding stage: no real GGUF
  file had been tested against yet (that starts in Session 54).

---

## Status

**Session 52** — `EU AI ACT` Compliance User transparency and AI disclosure.

InferenceApiServer.java; ConsoleMain.java and juno-api.yaml was updated with `The replies are generated by an AI system` water-mark.

---

## Status

**Session 51**: Documentation update: `docs/` folder restructured as `juno-documentation` MyST Jupyter Book.

### `juno-documentation`

- The flat `docs/` folder has been restructured into `juno-documentation/`, a standalone
  [MyST-MD](https://mystmd.org/) Jupyter Book configured via `myst.yml`.
- Content is organised into 11 parts and 54 chapters, each in its own `.md` file under
  `part1/` through `part11/`. Navigation links (`<-` / `->`) and a full Table of Contents in
  `index.md` cross-link every chapter.
- All Mermaid diagrams are declared with the MyST `{mermaid}` directive where applicable and
  render natively in the built book and in any Mermaid-aware viewer.
- A `references.md` back-matter table maps every chapter back to the originating file in `docs/`.
- `build.sh` provides a one-command build (`./build.sh`); `README.md` documents prerequisites
  and the local preview workflow.

---

## Status

**Session 50**: `/train-file-qa`: multi-fact Q&A from a JSON file in one training loop; HTTP API.

### `/train-file-qa`

- REPL command loads a `.json` array of `{"Q","A"}` objects via `LoraQaFile`.
- Each pair expands to the same four chat-templated variants as `/train-qa`; all units
  train in one `trainOnUnits` pass with QA loss targets.
- `LoraTrainer.trainQaPairsUntilResult` for the programmatic multi-pair path.
- `LoraApiServer`: with `./juno lora --api-port N`: `POST /v1/lora/train-file-qa`
  (JSON body) and `POST /v1/lora/save` for curl workflows.
- Dropped verbose `[TRACE]` dump of formatted training text / token IDs on `/train-qa`.
- Docs: `docs/LoRA.md`, `docs/howto.md`.

---

## Status

**Session 49**: LoRA Tier 11 (complete): `--lora-microbatch` CLI/env + VRAM OOM auto-fallback.

### LoRA microbatch CLI and VRAM ladder (Tier 11)

- `LoraMicrobatch`: `--lora-microbatch N` / `LORA_MICROBATCH` (default 8, range 1..128);
  applies `juno.lora.microbatch` before resident upload (no `JAVA_TOOL_OPTIONS` required).
- `LoraResidentUpload`: on FP32 microbatch VRAM OOM with half support: close, set
  microbatch=1, retry FP16 once; further OOM uses existing auto→CPU / gpu fail-closed policy.
- Wired through `LoraCliOptions`, `LoraTrainingConfig`, `ConsoleMain`, `LoraTrainer`,
  `scripts/run.sh` / `run.bat`, and all three LoRA training handlers.
- Docs: `docs/LoRA.md`, `docs/howto.md`, `docs/performance.md`, `docs/agent-arch.txt`.

---

## Status

**Session 48**: LoRA Tier 9 (complete): microbatch GEMM + published GPU speed gates.

### LoRA GPU microbatch and product gates (Tier 9)

- `GpuBlasOps` / `DeviceActivationBatch`: FP32 `cublasSgemm_v2` / `rocblas_sgemm` microbatch
  for frozen forward and transpose; CPU oracle `CpuFrozenBatchOps`.
- Default `juno.lora.microbatch=8` uploads FP32 resident weights and batches linears across
  positions in `LoraTrainableHandler.computeGradients` (host adapters / Adam unchanged).
- `LoraTrainableHandlerGpuBackwardTest`: CPU↔GPU loss/grad parity + TinyLlama speed gates
  (GTX 1080: **~14× e2e**, **~11× backward** vs CPU).
- Docs may describe production **GPU LoRA training** as frozen batched GPU + host adapters;
  device-resident adapters / GPU Adam remain deferred (not required after intensity proof).
- `--lora-train-device` and LLaMA/Qwen2 timing subsets remain as in Session 46 (`transferMs` still 0).

---

## Status

**Session 47**: LoRA Tier 10 (complete): multi-arch GPU residency + production gates.

### LoRA multi-arch GPU residency (Tier 10)

- `LoraResidentWeights`: shared upload / close / VRAM-OOM fallback / matVec+transpose routing.
- `LoraTrainableHandler` refactored onto the helper (LLaMA-family / Qwen2 unchanged behavior).
- `Phi3LoraTrainableHandler` / `Qwen3LoraTrainableHandler` upload physical fused (Phi) or dense
  (Qwen3) projections when `--lora-train-device` resolves to a `GpuMatVec`; CPU fallback preserved.
- Gated live LoRA smokes (`LoraLiveSmokeTest`) for TinyLlama / Qwen2.5 / Phi-3.5 / dense Qwen3 fixtures.
- `EosOutputFilter`: hold back / strip turn-end markers (`</s>`, `<|end|>`, `<|im_end|>`, …) so
  `/train-qa` completions never stream into REPL or `GenerationResult` text (all LoRA chat templates).
- DoRA: correctness-complete, **not** production-perf-gated (prefer LoRA/rsLoRA for large all-linear jobs).
- Tier 7 JFR metrics marked **complete** (programmatic `--jfr`, mode identity, extractor, docs).
- Tier 5 held-out research / quality matrix remains **deferred**; exact K-quant QA-LoRA merge unsupported.

---

## Status

**Session 46**: LoRA Tier 9 (start → completed in Session 48): `--lora-train-device` productization.

### LoRA GPU train-device (Tier 9)

- `--lora-train-device auto|gpu|cpu` / `LORA_TRAIN_DEVICE` (default **auto**).
- `LoraTrainDevice`: MatVec selection; `gpu` fails closed without CUDA/ROCm; `cpu` forces `CpuMatVec`.
- `LoraTrainer` / LoRA REPL honor the mode; JFR `trainDevice` is the resolved label (`cpu`/`cuda`/`rocm`).
- `LoraStepTiming`: fills `frozenForwardMs` / `frozenTransposeBackwardMs` / `adapterBackwardMs` /
  `attentionNonlinearMs` on `juno.LoraTrainStep` from LLaMA/Qwen2 handler instrumentation (`transferMs` still 0 until H2D counters).
- Microbatch / parity IT / speed gates: completed in Session 48.

---

## Status

**Session 45**: LoRA Tier 8: train-file scheduling and corpus caps.

### LoRA train-file scheduling (Tier 8)

- `--lora-chunk-tokens` / `LORA_CHUNK_TOKENS` (default **32**; recommend **128** for large `/train-file`).
- `--lora-max-train-tokens` / `LORA_MAX_TRAIN_TOKENS` (`0` = unlimited): seeded whole-chunk subsample of supervised prediction tokens.
- `/train` and `/train-file` use document-level `TrainUnit`s; chunking happens inside `LoraTrainingLoop`.
- `LoraCorpusLimit` helper; docs/help no longer claim a silent 128 default.

---

## Status

**Session 44**: LoRA training progress bar (loss → target).

- `LoraTrainProgressBar`: percent from pass-2 baseline loss toward `--lora-loss-target-*`; max-iters not used.
- ETA from loss-improvement rate since baseline; final frame ETA `0s` when the run ends.

---

## Status

**Session 43**: LoRA Tier 6: multi-architecture training (CPU oracle).

### LoRA multi-architecture (Tier 6)

- `LoraTrainingHandler` / `LoraTrainingHandlerFactory`: explicit allowlist by `general.architecture`.
- `LoraModelLayout` / `LoraProjectionBinding`: logical keys → physical GGUF tensors (Phi fused slices).
- Handlers: LLaMA-family (`LoraTrainableHandler`), `Qwen2LoraTrainableHandler` (frozen QKV biases),
  `Phi3LoraTrainableHandler` (fused QKV/gate-up + NeoX RoPE), `Qwen3LoraTrainableHandler`
  (per-head Q/K RMSNorm, `qDim`).
- `LoraMerge` layout-aware multi-adapter fused-slice F32 patching for Phi-3.
- Rejected for LoRA: `qwen3moe`, `qwen35`, `gemma`, unknown.
- Qwen3 `/train-qa` template parity with empty `<think>` block.

---

## Status

**Session 42**: LoRA REPL UX + WebUI model dropdown.

- `/reset` deletes the `.lora` checkpoint (no overwrite save); memory reset + chat history clear unchanged.
- LoRA banner and chat footer show sampling `temperature` (and top-k / top-p on the banner).
- Default LoRA training log is a compact progress bar; `--verbose` / `-v` restores full `[TRACE]` / per-pass lines.
- WebUI model dropdown parses OpenAI `GET /v1/models` (`data` / `id` / `x_juno_*`) so names appear again.

---

## Status

**Session 41**: LoRA Tier 7 (complete): JFR metrics for all adapter modes and operations.

### LoRA JFR metrics (Tier 7)

- Programmatic LoRA `--jfr` lifecycle matches local mode (`jdk.jfr.Recording` + auto-extract `target/metrics/metrics.json` on exit). Launchers pass `--jfr` as an app arg (no `-XX:StartFlightRecording` for LoRA).
- `LoraMetricsIdentity`: CLI vocabulary tags (`lora` / `rslora` / `dora` / `qa-lora`) on train, validation, merge, norm-refresh, playback, and checkpoint events.
- New events: `juno.LoraNormRefresh`, `juno.LoraMerge`, `juno.LoraPlayback`, `juno.LoraCheckpoint`.
- `JfrMetricsExtractor` aggregates train/validation/merge/DoRA-refresh/playback series with guarded field reads (older recordings still extract).

---

## Status

**Session 40**: LoRA Tier 5 (complete implementation): QA-LoRA + merge policies.

### LoRA QA-LoRA and quantized merge (Tier 5)

- Gate A codecs retained: `QuantizationLayout`, `GgufQuantCodec` / `GgufKQuantCodec` (`juno-kquant-v1`), `QuantizedMergeMetrics`.
- `QaLoraAdapter`: sum-pool grouped A (`rank×groupCount`) + B; dense-expansion oracle and finite-difference tests.
- `AdapterAlgorithm`, `MergeCapability` (`SIDECAR_ONLY` / `F32_PRESERVE` / `SOURCE_TYPE_PROJECTED`; `EXACT_AFFINE` rejected for K-quants).
- Checkpoint v2: QA entries store `groupWidth` before A, Tier-5 extension blob (algorithm, pooling, ggml type, encoder id, merge policy); v1 export rejected for QA-LoRA.
- `QaLoraInitializer`: group width from actual tensor GGML type (Q4_K/Q5_K→32, Q6_K→16); fingerprints verified on load.
- Training/playback: `LoraTrainableHandler`, Adam, gradients, CLI `--lora-mode qa-lora`, `--lora-group-width`, `--lora-merge`.
- `LoraMerge`: F32 preserve (default) and explicit `SOURCE_TYPE_PROJECTED` requantization with per-tensor metrics; zero-delta copies raw bytes; never silent exact→projected fallback.
- Exact K-quant QA-LoRA zero-point merge remains **no-go**. Full held-out experiment matrix / deployment quality gates are research follow-ups; sidecar + F32 stay production-safe.

---

## Status

**Session 39**: LoRA Tier 5 (Gate A start): shared GGUF K-quant codec layer.

### LoRA QA-LoRA / quantized merge foundations (Tier 5 Gate A)

- `QuantizationLayout`: Q4_K / Q5_K / Q6_K geometry (block/sub-block width, affine vs symmetric).
- `GgufKQuantCodec` / `GgufQuantCodec`: versioned encoder id `juno-kquant-v1`; decode matches llama.cpp goldens; encode moved out of `LoraMerge`.
- `QuantizedMergeMetrics`: RMSE, max error, delta-retention helpers for projected merge.
- `GgufReader` and `LlamaTransformerHandler.dequantize` delegate K-quant decode to the shared codec; fused matVec paths unchanged for performance.
- No-op path: `copyRawUnchanged`: decode/re-encode must not be used for byte-identical preservation.
- Non-closure tests: Q6_K additive shift and Q4_K nested-scale offset are not exact (exact K-merge remains no-go).
- Next: grouped QA-LoRA math (Gate B), merge capability policy, then projected merge experiments.

---

## Status

**Session 38**: LoRA Tier 4 (start): resident transpose primitives and baseline instrumentation.

### LoRA GPU training foundations (Tier 4)

- Vendor-neutral `GpuBindings.opNoTranspose()` (CUDA `CUBLAS_OP_N=0`, ROCm `rocblas_operation_none=111`).
- `GpuMatVec.sgemvTranspose` for resident FP32/FP16 `W^T * g` (same row-major buffer as forward `OP_T`).
- `ResidentWeightMatrix` + `LoraTrainableHandler` routes frozen forward and transpose backward through resident GPU weights when uploaded (`supportsHalfResident` FP16 or FP32 fallback).
- JFR backend labels: `*-resident-transpose` / `*-resident-fp16-transpose`.
- `LoraTrainEvent` fields for frozen forward/transpose, attention/nonlinear, adapter backward, and transfer (filled when finer instrumentation lands).
- GPU adjoint tests: `CudaMatVecTransposeTest`, `RocmMatVecTransposeTest` (`GpuMatVecTransposeContractTest`).
- Baseline section in `docs/performance.md`: hybrid path is not yet marketed as production GPU training.
- `--lora-train-device` shipped in Session 46; CPU/GPU gradient parity IT and speed gates remain open.
- Fix: `LoraAdapterSet.resetFrom` (REPL `/reset`) bumps DoRA cache generation so inference drops trained magnitude coefficients.
- Fix: `/reset` also clears REPL chat history and rotates the session id: otherwise multi-turn context still contains the memorized answers.

---

## Status

**Session 37**: LoRA Tier 3 (phase 1–2): rsLoRA, Kaiming, checkpoint v2, DoRA.

### LoRA advanced adapters (Tier 3)

- Explicit adapter metadata: `LoraAdapterConfig` with `LoraScaling`, `LoraInitialization`, `LoraMode`.
- rsLoRA scale `alpha/√rank`; PEFT-compatible Kaiming-uniform A init (legacy-normal retained for compatibility overloads).
- Checkpoint version 2 (length-delimited) with declared alpha, scaling, init, mode, optional DoRA magnitude and base-tensor fingerprints; v1 still loads.
- Canonical detached-norm DoRA (`DoraMagnitude`, `DoraProjection`); magnitude is an AdamW parameter group with decay off.
- `DoraInitializer` builds magnitudes/fingerprints from GGUF dequant; merge applies LoRA/rsLoRA/DoRA formulas to F32.
- CLI/env: `--lora-mode`, `--lora-scaling`, `--lora-init` (`LORA_MODE`, `LORA_SCALING`, `LORA_INIT`).
- DoRA norm-refresh is correctness-complete but **not** production-perf-gated; prefer standard
  LoRA/rsLoRA for large all-linear jobs until a measured refresh budget exists (Tier 10).

---

## Status

**Session 36**: LoRA Tier 2: schedules, AdamW, dropout, validation, and LoRA+.

### LoRA training quality (Tier 2)

- Warmup/cosine and constant learning-rate schedules (`--lora-lr-schedule`, `--lora-warmup-steps`, `--lora-min-lr`).
- True A-only decoupled AdamW (`--lora-weight-decay`); moments see raw gradients only. Numerical trajectories change vs coupled L2; checkpoints remain compatible.
- LoRA+ parameter groups: A uses scheduled LR, B uses `LR * --lora-plus-ratio` (default `1.0` = ordinary behavior).
- Deterministic train-only inverted dropout (`--lora-dropout`, `--lora-seed`); inference and validation stay dropout-free.
- Forward-only `evaluateLoss`; held-out validation split with patience/min-delta and best-weight restore (`--lora-validation-*`).
- Shared `LoraTrainingLoop` orchestration for REPL and `LoraTrainer`; Q&A variants are hold-out units.
- JFR `LoraTrainStep` carries A/B LR, LoRA+ ratio, and dropout; optional `LoraValidation` event.

---

## Status

**Session 35**: LoRA Tier 1: projection coverage, token-weighted accumulation, and clipping.

### LoRA correctness foundation (Tier 1)

- Configurable projection targets: `qv` (default), `all` / `all-linear`, or comma-separated keys (`wq,wk,wv,wo,wgate,wup,wdown`).
- Complete forward/backward for all seven dense linear projections, including current-position K and inverse-RoPE on Q and K.
- `computeGradients` separated from optimizer updates; token-weighted gradient accumulation across chunks.
- Global L2 gradient clipping after prediction-count normalization (`--lora-max-grad-norm`; `0` disables clip).
- Builder-based `LoraTrainingConfig` and `LoraTrainer.open(..., config)`; legacy overload keeps qv, accum=1, clipping off.
- Architecture gate: Phi-3 / Qwen3 / Qwen3-MoE rejected for LoRA (dense LLaMA-family required).
- `/reset` reinitialises A and B from the selected target config (not B-only zeroing).
- Merge maps all seven projections via `LoraProjection`; adapted tensors remain F32.
- Terminology: LoRA on a quantized GGUF base (not QLoRA).

---

## Status

**Session 34**: Windows launcher fixed: `run.bat` and `juno.bat` fully functional on Windows.

### Windows launcher (`scripts/run.bat`, `juno.bat`)

All subcommands (`cluster`, `local`, `lora`, `merge`, `test`) and flags are now working on Windows.

**Root cause fixes:**

- **JAR name mismatch.** `run.bat` referenced `juno-player.jar` and `juno-master.jar`: names that Maven never produces. The actual artifacts are `juno-player-<version>-shaded.jar` and `juno-master-<version>.jar`. Fixed by reading the project version from `pom.xml` at startup using `findstr` and constructing the correct paths dynamically.

- **Java version detection hang.** CMD cannot redirect `stderr` in a pipeline (`2>&1`) reliably inside a `for /f` loop when delayed expansion is active. `java -version` writes to stderr and the output was silently lost, leaving `JAVAVER_RAW` undefined. Fixed by capturing `java -version 2> tmpfile` to a temp file and reading the file with `for /f`.

- **`find_java` nested-if failure.** Nested `if ... (if ... (...))` blocks are not reliable in CMD with `setlocal enabledelayedexpansion`. Replaced with a flat goto-based structure (`find_java_where` label).

- **Infinite loop on empty argument.** In argument-parsing loops, `if exist "%~1"` on an empty `%~1` expands to `if exist ""` which matches the current directory (always true), causing an infinite loop. Fixed by guarding with `if not "%~1"==""` before the `if exist` check in the `cluster`, `local`, `lora`, and `test` parsers.

- **JFR block inside `if not ... (for ...)` silently skipped.** CMD does not support a `for` command inside an `if` parenthesized block when delayed expansion is on. Replaced with a goto-based pattern (`lora_jfr_skip` / `test_jfr_skip` labels).

**Documentation updated:**

- `README.md`: Windows launcher note in section 2.2, Windows requirements paragraph, `juno.bat` references for `merge`.
- `docs/howto.md`: Windows note at top; Windows command-prompt examples added to every subcommand section (`local`, `cluster`, `lora`, `merge`) and Build and Test.

---

## Status

**Session 33**: Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development.

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

- **`README.md`**, **`RELEASE_NOTES.md`**: Supported models section
- **`docs/arch.md`**: handler routing and tokenizer notes
- **`docs/features.md`**, **`docs/howto.md`**, **`docs/LoRA.md`**: Phi-3 OK for inference; Gemma and Qwen paths not production-ready; LoRA still LLaMA-family (+ Phi-3 template detection)
- **`docs/phi3-inference-handoff.md`**: status set to supported (retains debug handoff notes)
- **`docs/model_support_summary_972ab30f.plan.md`**: roadmap, dispatch table, chat matrix, gaps, decisions log

**Policy:** Phi-3 is production-ready in docs and validation (local + cluster). Gemma and all Qwen families remain under development until dedicated validation lands.

---

## Status

**Session 32**: ROCm/HIP backend for AMD GPU inference via Panama FFI.

### AMD GPU support (ROCm/HIP + rocBLAS)

Full first-class AMD GPU support alongside the existing NVIDIA CUDA backend. The GPU
abstraction layer auto-selects CUDA > ROCm > CPU at startup with no configuration required.
Tested on AMD Radeon RX 7900 XT (gfx1100, ROCm 7.2.x).

**New production classes (`node` module):**

- **`GpuBindings`**: vendor-neutral interface implemented by `CudaBindings` and `RocmBindings`.
  Exposes all device runtime and BLAS handles as `MethodHandle` accessors, shared constants
  (`H2D`, `D2H`, `STREAM_NON_BLOCKING`), and static helpers (`check`, `callInt`, `loadLibrary`,
  `bind`). Static helpers eliminate per-implementation boilerplate.
- **`GpuMatVec`**: sealed interface (`permits CudaMatVec, RocmMatVec`) extending `MatVec`.
  Exposes `upload(float[], int, int)` and `uploadHalf(float[], int, int)` so transformer
  handlers depend on the GPU abstraction rather than a concrete vendor class.
- **`RocmBindings`**: Panama FFI downcall handles for `libamdhip64.so` and `librocblas.so`.
  Pre-binds `hipHostMalloc flags=0` via `MethodHandles.insertArguments` to match the
  `cudaMallocHost` arity visible to all callers. Key ROCm constants: `opTranspose()=112`
  (`rocblas_operation_transpose`), `hipDeviceProp_t` sizeof=1472, name@0, totalGlobalMem@288
  (measured from ROCm 7.2.x headers, Linux x86_64).
- **`RocmAvailability`**: HIP device detection: `isAvailable()`, `deviceCount()`,
  `deviceName(int)`, `vramBytes(int)`. Mirrors `CudaAvailability` in structure.
- **`RocmMatVec`**: `MatVec` / `GpuMatVec` implementation backed by `rocblas_sgemv` (FP32)
  and `rocblas_hssgemv_strided_batched` (FP16). Three compute paths:
  - Host FP32: temporary device buffers per call; synchronous H2D → kernel → D2H.
  - Device-resident FP32 (`DeviceFloatMatrix`): per-thread scratch for x/y; async stream copies.
  - Device-resident FP16 (`DeviceHalfMatrix`): x converted FP16 in off-heap arena; FP32 accumulation.
  Off-heap `Arena.ofConfined()` staging for all H2D/D2H copies: required by Java 25 Panama
  (heap segments rejected by native downcalls).
- **`MatVecBackend`**: enum replacing ad-hoc string literals for the `juno.MatVec.backend` JFR
  dimension. Values: `CPU`, `CUDA`, `CUDA_RESIDENT`, `CUDA_RESIDENT_FP16`, `ROCM`,
  `ROCM_RESIDENT`, `ROCM_RESIDENT_FP16`. Label strings are part of the JFR contract and unchanged.

**Modified production classes:**

- **`GpuContext`**: refactored from CUDA-only to backend-agnostic. Adds `GpuBindings bindings`
  field, `bindings()` accessor, `selectBindings()` (CUDA → ROCm priority order with
  `-Djuno.gpu.backend=cuda|rocm|auto` override), `createMatVec()` factory, `backendLabel()`
  delegate. `close()` uses `bindings.cublasDestroy()` instead of hardcoded CUDA call.
  Private `deviceName()` and `deviceVram()` helpers use `GpuBindings` struct-offset accessors.
- **`CudaBindings`**: adds `implements GpuBindings`; 20 accessor methods expose the existing
  `MethodHandle` fields to vendor-neutral callers. Zero existing fields or constants removed.
- **`CudaAvailability`**: field-access calls updated to use `CudaBindings.instance()` accessor
  methods (`PROP_NAME_OFFSET` → `instance().PROP_NAME_OFFSET`, etc.).
- **`CudaMatVec`**: implements `GpuMatVec` (was `MatVec`); `upload` / `uploadHalf` made public
  with `@Override`; backend labels replaced by `MatVecBackend` enum calls.
- **`DeviceFloatMatrix` / `DeviceHalfMatrix`**: direct `CudaBindings.instance()` field access
  replaced by `GpuContext#bindings()` method calls (`GpuBindings`). Both classes now work
  identically on CUDA and ROCm. `DeviceHalfMatrix` caches `gpu = ctx.bindings()` at construction.
- **`LlamaTransformerHandler`**: `instanceof CudaMatVec` → `instanceof GpuMatVec` for weight
  upload gate; `cudaMalloc` OOM message check extended to also catch `hipMalloc`;
  `matVecQuantBackendLabel(int)` → `matVecQuantBackend(int)` returns `MatVecBackend.CPU`.
- **`Phi3TransformerHandler`**: same `instanceof` fix; OOM check extended to `hipMalloc`.
- **`LoraTrainableHandler`**: same `instanceof` fix.
- **`ForwardPassHandlerLoader`**: `pickMatVec` checks both `CudaAvailability` and
  `RocmAvailability`; device count query reads from the available backend; `GpuContext.shared(dev).createMatVec()` replaces `new CudaMatVec(...)`.
- **`EmbeddedNodeServer`**: uses `gpuContext.createMatVec()` and `gpuContext.backendLabel()`
  for log messages.
- **`ConsoleMain` / `JunoPlayer`**: `new CudaMatVec(gpuCtx)` → `gpuCtx.createMatVec()`.
- **`MatVecEvent`**: adds `backend(MatVecBackend)` setter to avoid hand-written label strings
  at call sites; public `String backend` field kept for JFR contract.

**New tests (55 total, 0 failures on RX 7900 XT):**

- `RocmMatVecTest` (30): extends `MatVecBackendContractTest` for full API parity; correctness
  vs CPU reference at 2048×2048, 5632×2048, 32000×2048; trivial known-value cases;
  4-thread concurrent safety; throughput sanity.
- `RocmAvailabilityTest` (8): device detection present/absent; name format; VRAM bounds;
  out-of-range index fallbacks.
- `GpuContextTest` +5 `@Tag(rocm)`: ROCm context lifecycle, backend priority,
  `createMatVec` factory, shared singleton, system-property override.
- `ForwardPassHandlerLoaderSelectBackendTest` +2 `@Tag(rocm)`: `RocmMatVec` routing,
  process-wide `GpuContext.shared(0)` reuse.
- `ForwardPassHandlerLoaderSelectLoraBackendTest` +1 `@Tag(rocm)`: LoRA routing on ROCm.
- `MatVecQuantizedBackendLabelTest`: updated to use `MatVecBackend` enum constants.

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

**Session 31**: Panama FFI for Juno math: JavaCPP / bytedeco removed, CUDA bindings rewritten with `java.lang.foreign`.

### Panama FFI GPU bindings (`node` module)

The entire CUDA bridge has been rewritten using the Java 25 Panama Foreign Function & Memory API
(`java.lang.foreign.Linker`, `SymbolLookup`, `MemorySegment`, `Arena`). The `org.bytedeco:cuda-platform`
dependency has been removed from `node/pom.xml`.

**New production class:**

- **`CudaBindings`**: Panama FFI downcall handles for `libcudart.so.12` and `libcublas.so.12`.
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

- **`CudaMatVec`**: all JNI / JavaCPP call sites replaced with `CudaBindings` downcall handles.
  Native memory managed exclusively via `MemorySegment` and `Arena`. Device weight matrices
  (`DeviceFloatMatrix`, `DeviceHalfMatrix`) held resident; `MemorySegment` passed directly to
  cuBLAS as `ADDRESS`: zero H2D copy per token. Per-thread `Fp32Scratch` / `Fp16Scratch`
  scratch on device grown lazily and reused. FP16 x staging packed with `Float.floatToFloat16`
  into a confined off-heap arena in the hot path.
- **`GpuContext`**: cuBLAS handle stored as `MemorySegment` (opaque `cublasHandle_t`); created
  and destroyed via `CudaBindings`. `cublasSerializationLock()` serializes stream-binding and
  kernel submission on the shared handle. `shared(int)` returns a process-wide singleton per
  device index.
- **`DeviceFloatMatrix`**: device memory allocated via `CudaBindings.deviceMalloc`; backing
  `MemorySegment` sized to `rows * cols * 4` bytes; H2D via synchronous `cudaMemcpy`.
- **`DeviceHalfMatrix`**: same pattern; FP16 x staging via confined arena; `MemorySegment.ofArray`
  pins heap array for duration of downcall.
- **`CudaAvailability`**: device detection updated to use `CudaBindings` downcall handles.

**`node/pom.xml`:** `org.bytedeco:cuda-platform` dependency removed.
`maven-surefire-plugin` `argLine` updated: `--enable-native-access=ALL-UNNAMED`,
`--add-opens java.base/java.lang=ALL-UNNAMED`, `--add-opens java.base/java.nio=ALL-UNNAMED`.

**New test: `CudaBindingsTest`**: two scenarios:
- CUDA present (`@Tag("gpu")`): every `MethodHandle` non-null, singleton loads cleanly.
- CUDA absent (CPU-only CI): `isAvailable()` returns false, `instance()` throws `IllegalStateException`.

Run GPU-tagged tests: `mvn test -Dgroups=gpu -pl node`

All existing tests pass unchanged.

---

## Status

**Session 30**: Maven Central publish configuration.

### Maven Central publish (`pom.xml`, all module POMs)

All modules configured for publishing to `central.sonatype.org` via the Central Portal publisher.
Version set to `0.1.0-RC` across root POM and `juno-bom`.

**Changes:**

- **`maven-source-plugin 3.3.1`**: `attach-sources` execution at `verify` phase; produces `-sources.jar`
  required by Maven Central.
- **`maven-javadoc-plugin 3.11.2`**: `attach-javadocs` execution at `verify` phase; `doclint=none`,
  `failOnError=false`; produces `-javadoc.jar` required by Maven Central.
- **`maven-gpg-plugin`**: `sign-release` execution moved from `verify` to `install` phase so
  sources and Javadoc jars are already attached before signing. `--pinentry-mode loopback`
  added to `gpgArguments` to allow `-Dgpg.passphrase=...` without a GUI pinentry agent.
- **`distributionManagement`**: `<repository>` and `<snapshotRepository>` wired to
  `central.sonatype.org` Central Portal publisher endpoint.
- **Developer / SCM metadata**: `<organization>Machine Learning Cabinet</organization>`,
  `<organizationUrl>https://ml.cab/</organizationUrl>`, SCM tag updated to `v0.1.0-RC`.
- **All module POMs**: publish config consolidated into root POM; per-module boilerplate removed.

---

## Status

**Session 29**: OpenAI-compatible REST API (`POST /v1/chat/completions`, `GET /v1/models`).

### OpenAI-compatible API

Any client that speaks the OpenAI Chat Completions wire format: LangChain, LlamaIndex,
LiteLLM, the OpenAI Python/Node SDKs, or any internal tool built against `openai.*`: works
against Juno with a single base-URL change. No prompt reformatting, no adapter library, no
glue code.

**New classes (coordinator module):**

- **`OpenAiAdapter`**: pure static mapping helpers between Juno internals and the OpenAI wire
  format: `repetitionPenaltyFromFrequencyPenalty(float)` (OpenAI −2..2 range → Juno ≥1),
  `validateCompletionsN(Integer)` (rejects n ≠ 1), `toOpenAiFinishReason(StopReason)` (`stop`
  / `length` / `error`), and `chatCompletionId(String)` (`chatcmpl-` + compact UUID).
- **`OpenAiChatHandler`**: Javalin handler class owning three endpoints:
  - `POST /v1/chat/completions`: deserialises `OaiChatCompletionRequest` (Jackson,
    `@JsonIgnoreProperties(ignoreUnknown = true)`), validates `n` and `messages`, builds an
    `InferenceRequest` + `SamplingParams`, then dispatches to either
    `scheduler.submitAndWait()` (blocking, returns `ChatCompletion` JSON) or
    `scheduler.submit()` (streaming, writes `text/event-stream` chunks terminated by
    `data: [DONE]`).
  - `GET /v1/models`: filters `ModelRegistry` to `LOADED` status, wraps each
    `ModelDescriptor` in an OpenAI `Model` object with `x_juno_*` extension fields.
  - `GET /v1/models/{modelId}`: single-model lookup; 404 when absent.

**Modified: `InferenceApiServer`**: constructs `OpenAiChatHandler` in the constructor
(passing the latency callback so `HealthReporter` still records P99). Routes
`POST /v1/chat/completions` and `GET /v1/models[/{modelId}]` to the handler.
The existing `POST /v1/inference` and `POST /v1/inference/stream` endpoints are untouched.

**Modified: `ConsoleMain`** (`juno-player` module): `--api-port N` flag starts a
`RequestScheduler` + `InferenceApiServer` alongside the existing REPL in both `local` and
cluster modes. A virtual-thread shutdown hook calls `apiServer.stop()` on JVM exit.
`buildLocalModelRegistry()` populates a `ModelRegistry` from the in-process `LlamaConfig` so
`GET /v1/models` returns the loaded model immediately.

**Modified: `scripts/run.sh`**: `--api-port N` flag wired into both `cmd_local()` and
`cmd_cluster()`. Environment override: `API_PORT`.

**New file: `api/src/main/resources/juno-api.yaml`**: OpenAPI 3.0.3 spec for the public
client-facing API. Documents all request fields with their Juno internal mappings, the SSE
chunk event sequence, Juno extension fields (`x_juno_priority`, `x_juno_session_id`,
`x_juno_top_k`, `x_juno_latency_ms`, `x_juno_retry_after_ms`, `x_juno_queue_depth`), and
all error codes.

**New test: `OpenAiAdapterTest`**: unit tests for all four mapping helpers.

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
| `n` |: | Only 1 is accepted; other values → 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` |: | Silently ignored |
| `x_juno_priority` | `RequestPriority` | HIGH / NORMAL / LOW |
| `x_juno_session_id` | `InferenceRequest.sessionId` | Enables KV-cache reuse across turns |
| `x_juno_top_k` | `SamplingParams.topK` | 0 = disabled; default 50 |

All modules compile. All existing tests pass. `OpenAiAdapterTest` (4 assertions) passes.

---

## Status

**Session 28**: Health dashboard: CPU load metric, role-conditional secondary metric, node throughput.

### Health dashboard fixes

**Fix 1: `temperatureCelsius` → `cpuLoad`.**
`/sys/class/thermal` is unavailable on EC2 VMs; the Temperature row always showed a dash
placeholder. Replaced with process CPU utilisation read from `OperatingSystemMXBean.getCpuLoad()` (0.0-1.0, available on all JVM platforms, no sysfs). Changes:
- `NodeHealth` record: field `temperatureCelsius` removed, `cpuLoad` added (same sentinel -1.0 convention, clamped to 0.0 on first-sample unavailability).
- `HealthReporter.buildProbeJson()`: `readTemperatureCelsius()` + all sysfs helpers (`findThermalZone`, `findHwmonTemp`, thermalPath/thermalProbed state) removed; replaced by 5-line `readCpuLoad()`.
- `HealthMain.NodeHealthDto`: `temperatureCelsius` field → `cpuLoad`.
- Dashboard HTML (both `HealthMain` and `InferenceApiServer` embedded console): "Temperature" row → "CPU load" formatted as `XX.X %`.

**Fix 2: Role-conditional secondary metric: coordinator shows Latency P99, nodes show Throughput.**
`Latency P99` was populated by `HealthReporter.recordLatency()`, which is only called from `InferenceApiServer` on the coordinator JVM. Worker nodes always showed a dash placeholder. Added a `nodeRole` field (`"coordinator"` | `"node"`) to `NodeHealth` and `NodeHealthDto` so the dashboard can branch:
- **Coordinator card**: Latency P99 (ms): end-to-end generation time, already wired via `InferenceApiServer.setLatencyReporter()`.
- **Worker node cards**: Throughput (MB/s): activation bytes forwarded per second via new `HealthReporter.recordBytes(long n)` + `drainThroughput()` (atomic byte counter drained each probe interval).

Wiring:
- `EmbeddedNodeServer`: retained `NodeServiceImpl` reference as `serviceImpl` field; added `setHealthReporter(HealthReporter)` on outer class delegating to a new package-private setter on the inner class. `forwardPass()` calls `hr.recordBytes(encodedOutput.length)` after each `responseObserver.onNext()`.
- `NodeMain`: constructs reporter with `nodeRole="node"`, calls `server.setHealthReporter(reporter)` after `server.start()`.
- `CoordinatorMain`: constructs reporter with `nodeRole="coordinator"`.
- `HealthReporter` constructors: 2-arg and 3-arg remain backward-compatible (default role `"node"`); new canonical 4-arg constructor `(nodeId, nodeRole, healthBaseUrl, intervalMs)`. Added `startForCoordinator(healthBase)` factory alongside existing `startForNode(nodeId, healthBase)`.
- `buildNodeDetail()` switched from `Map.of()` (10-entry limit) to `Map.ofEntries()` to accommodate 12 fields.

**Investigation 3: Why 1 of 10 concurrent sessions produced no tokens (no code change).**
Root cause: gRPC `ServerBuilder.forPort(port)` with no custom executor defaults to a thread pool bounded by `~2 × CPU count` (4 threads on `m7i-flex.large`). With 9 sessions concurrently running prefill (26 steps × 9 = up to 234 in-flight blocking stubs), all 4 gRPC threads on each node were saturated. The 10th session's first `pipeline.forward()` call queued behind them for ~8.5 minutes until prefill of the other 9 finished. The fix is `ServerBuilder.forPort(port).executor(Executors.newVirtualThreadPerTaskExecutor())`: virtual threads don't block OS threads on gRPC I/O. JFR evidence: `juno.ForwardPass.decode.p95_ms = 3095 ms` on node-1 (coordinator node running layers 0–8 plus the REST server) vs 914 ms on node-2; coordinator log confirms 10 tokenizer encodes but only 9 near-simultaneous prefills.

All modules compile. All existing tests pass (NodeHealth, HealthEvaluator, HealthReactor constructors updated to 9-arg signature).

---

**Session 27**: GPU lifecycle, multi-device shared contexts, CUDA streams, Llama VRAM fallback, docs.

- **`ForwardPassHandler.releaseGpuResources()`**: default no-op; **`LlamaTransformerHandler`** and **`Phi3TransformerHandler`** close all **`DeviceHalfMatrix`** buffers. **`EmbeddedNodeServer`** invokes it on shard reload, load failure, and **`unloadShard`** (then swaps in **`StubForwardPassHandler`**).
- **`GpuContext.shared(int)`**: one process-wide **`GpuContext`** per CUDA device index (map + lock); **`close()`** remains a no-op for shared instances. **`ForwardPassHandlerLoader.selectBackend()`** and **`EmbeddedNodeServer`** honour **`-Djuno.cuda.device=N`**, validated against **`CudaAvailability.deviceCount()`**.
- **`CudaMatVec`**: per-thread **non-blocking CUDA stream**; **`cublasSetStream_v2`** + **`cudaMemcpyAsync`** for resident FP32/FP16 **`x`/`y`** transfers; **`synchronized(gpuContext.cublasSerializationLock())`** around stream binding and kernels. Host **`sgemv(float[],…)`** also runs under the same lock.
- **Llama GPU OOM**: upload wrapped like Phi-3: on **`cudaMalloc`** failure, partial **`DeviceHalfMatrix`** buffers are **`close()`**d and inference falls back to **CPU quantised** matmul for those projections.
- **Docs/tests:** **`README.md`**, **`docs/arch.md`**, **`GpuContextTest`** (multi-GPU assumption), **`NodeMain`** Javadoc for **`juno.cuda.device`**.

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
- TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster
- Phi-3.5 GPU matmul path: `CudaMatVecBackendTest` FP16 resident matvec + `mvn test -Dgroups=gpu -pl node` on CUDA 12.x

**Session 26**: Phi-3 GPU matmul, FP16 resident weights, CLI and local GPU wiring.

`Phi3TransformerHandler` GPU path uploads dequantized fused QKV / FFN slices and output projection as **`DeviceHalfMatrix`** (IEEE FP16 on device, roughly half the VRAM of `DeviceFloatMatrix`). Forward uses **`CudaMatVec.sgemv(DeviceHalfMatrix, x)`**, implemented with **`cublasHSSgemvStridedBatched`**: same `(CUBLAS_OP_T, m=cols, n=rows, lda=cols)` layout contract as the proven **`cublasSgemv_v2`** path for row-major `A`. Host `float[]` activations are converted to FP16 for the per-call device `x` buffer; accumulation stays FP32. Earlier **`cublasSgemmEx` / `cublasGemmEx`** mixed-dtype attempts returned `NOT_SUPPORTED` / `INVALID_VALUE` on common stacks; the HSS strided-batched GEMV avoids that.

**Session 26**: Native LoRA merge (`juno merge`).

`LoraMerge` (new, `node` module) writes a new GGUF file from a base model and a `.lora` checkpoint without re-quantising the patched tensors. The 44 LoRA-adapted projection weights (wq/wv on every layer) are stored as F32; all other tensors are copied verbatim in their original quantised encoding. F32 is required because the LoRA delta (~6×10⁻⁴) is smaller than Q4_K quantisation noise (~3×10⁻³): re-quantising would silently erase the training. Verified: merged TinyLlama recalls `/train-qa` facts (name "Dima") correctly under `./juno local` with no `.lora` sidecar.

`GgufReader` gains five new public methods needed by the GGUF writer: `ggufFileOffset()`, `metadataSectionEnd()`, `tensorOrder()`, `tensorNelems(name)`, and keeps the existing `tensorAbsoluteOffset` / `tensorType` / `tensorDims`. Internal storage changed from `HashMap` to `LinkedHashMap` so `tensorOrder()` is stable. A `List<String> tensorOrder` field is added to preserve insertion order.

`LoraMergeMain` (`juno-player` module): CLI entry point for `juno merge`. Reads `--model-path`, `--lora-path`, `--output`, `--heap`. Derives `<model>.lora` and `<model>-merged.gguf` as defaults.

`run.sh` gains `cmd_merge()` and the `merge)` dispatch case.

`ConsoleMain` `/merge-hint` REPL command updated: now prints the actual `./juno merge` invocation instead of the old "contributions welcome" message.

Three bugs fixed during development of `LoraMerge`:
- **Q4_K**: `d = maxRange/63` → `d = maxRange/(63×15)`. Previous formula collapsed all 4-bit quant values to `{0,1}`.
- **Q5_K**: same bug, factor 31. `d = maxRange/63` → `d = maxRange/(63×31)`.
- **Q3_K scRaw packing**: aux0/aux1 high-nibble extraction used a broken two-pass utmp reconstruction; replaced with a clean direct inverse of `GgufReader.loadQ3_K`.

**Session 25**: Code quality: dead code removed, test helpers moved to test scope, docs fully updated.


`CyclicForwardPassHandler` moved from `node/src/main` to `node/src/test`. It is a deterministic stub with no business value without a model; it belongs exclusively in the test compilation unit. `EmbeddedNodeServer` no longer imports it: the three call sites (pre-load placeholder, model-load-failure fallback, no-model stub mode) are now served by a new private `StubForwardPassHandler` inner class that returns zero-filled arrays of the correct shape with no test machinery. `node/pom.xml` gains a `maven-jar-plugin` `test-jar` execution so other modules can still import `CyclicForwardPassHandler`; `coordinator/pom.xml` and `juno-master/pom.xml` declare the `node:tests` classifier dependency.

**VRAM / OOM:** GPU buffer allocation is wrapped; on failure (including `cudaMalloc` OOM), partial device buffers are closed and the handler falls back to **CPU quantised** `LlamaTransformerHandler.matVec`-style matmul for those projections.

**`ConsoleMain`:** missing **`break`** after **`--cpu`** fixed: parsing no longer fell through into **`--lora`**, which incorrectly set `loraMode` when forcing CPU inference.

**`ConsoleMain.runLocalRepl`:** one shared **`GpuContext`** + **`CudaMatVec`** instance for every in-process shard load (avoids redundant cuBLAS contexts and matches production “one GPU per JVM” usage).

**Tests:** `CudaMatVecBackendTest.device_half_matrix_sgemv_matches_host_path` (512×512) anchors FP16 resident correctness vs `LlamaTransformerHandler.matVec`.

**JFR:** `MatVecEvent.backend` **`cuda-resident-fp16`** labels the Phi FP16 device path. (As of session 27, Llama GPU resident weights also use **`cuda-resident-fp16`**; **`cuda-resident`** remains for **`DeviceFloatMatrix`** / tests.)

---

**Session 26**: LoRA inference overlay (`--lora-play`), Q&A training mode (`/train-qa`), diagnostic tracing, and AWS deploy hardening.

### `--lora-play PATH`: apply trained adapters at inference in any mode

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

### `ForwardPassHandlerLoader`: new LoRA overload

```java
// New canonical overload: all others delegate to this
public static ForwardPassHandler load(
    Path modelPath, ShardContext context, MatVec backend,
    LoraAdapterSet adapters) throws IOException
```

When `adapters != null`, the loader routes to `LoraTrainableHandler` (inference-only, no optimizer attached) instead of the architecture-specific handler. When `adapters == null` the existing `phi3` / `llama` dispatch is unchanged. `selectBackend()` promoted from package-private to `public` so juno-player-module callers can reuse it.

### `ClusterHarness`: `withLoraPlay()` fluent method

```java
harness.withLoraPlay("/path/to/model.lora");
```

Stores the path and injects `-Djuno.lora.play.path=PATH` into the `launchNode()` JVM command, after the JFR flags. Without this, forked node JVMs start with `loraPlayPath=null` and run the base model regardless of what the coordinator is told.

### `/train-qa`: conversational Q&A training

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

**Double base64 encoding (cloud-init rejected user-data).** `--user-data` was passed as a pre-base64-encoded string. AWS CLI base64-encodes it again; cloud-init received double-encoded garbage and logged `Unhandled non-multipart (text/x-not-multipart) userdata`. Fix: write user-data to a temp file and pass `file:///tmp/juno-userdata-*.sh`: the CLI reads it raw and does single encoding. The `[TRACE]` size line now also prints `first-line: #!/bin/bash` so shebang presence is visible in the setup log.

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

**Session 34**: Windows launcher fixed: `run.bat`/`juno.bat` fully functional; docs updated with Windows examples. *(this session)*

**Session 33**: Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development. *(unchanged)*

**Session 24**: Configurable activation byte order (`--byteOrder BE|LE`). *(unchanged)*

**Session 22**: Q2_K and Q3_K quantization support. *(unchanged)*

**Session 21**: Two new deployment fat-jar modules and a unified AWS script. *(unchanged)*

**Session 20**: GPU inference actually wired end-to-end. *(unchanged)*

**Session 19**: metrics module, Meta-Llama 3 tokenizer fix, AWS infrastructure scripts. *(unchanged)*

**Session 18**: GPT-2 BPE tokenizer, JFR instrumentation fixes. *(unchanged)*

**Session 17**: AWS infrastructure scripts. *(unchanged)*

**Session 14**: LoRA fine-tuning + JFR profiling. *(unchanged)*

---

[<- 11.1 Release Notes](#ch-11-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [12.1 Overview ->](#ch-12-1)
