## Status

**Session 61** — two things, both aimed at not wasting another ~25-minute
cycle on ambiguous results: (1) a real, previously-unchecked bug found and
fixed in image loading, (2) logging made unconditional/comprehensive enough
that the next run's log is self-diagnosing.

**Found and fixed: EXIF orientation was never applied.** `ImagePatchEmbedder`
called `ImageIO.read()` directly and fed the result straight to the encoder.
`ImageIO.read()` does NOT apply EXIF orientation correction — it returns raw
sensor-orientation pixels as-is. Phone/camera photos routinely carry an EXIF
Orientation tag (portrait shots are commonly stored as landscape pixel data
+ orientation=6); a synthetic image (the pixel-art JUNO logo used in earlier
sessions) essentially never does. This means every real-photo test this
session could have been feeding the model a sideways or upside-down image
regardless of any other fix — a confound that was never isolated because
only the synthetic image was tested until Session 40.

**Fix:** added a self-contained JPEG APP1/Exif segment parser
(`readExifOrientation`) and the standard 8-case EXIF orientation transform
(`applyExifOrientation`) to `ImagePatchEmbedder`. Runs before resize/
normalize, fails safe (any parse issue → orientation=1 → no-op, never
applies a wrong rotation). Logs raw vs EXIF-corrected dimensions per image.
Added tests for the byte-level parser (no-EXIF JPEG, non-JPEG, garbage
bytes, synthetic little/big-endian TIFF segments, all 8 orientation values)
and for the transform (dimension swap/preserve per orientation family, a
180°-twice-is-identity invariant, and a corner-pixel placement check —
deliberately avoiding hand-derived CW/CCW pixel assertions, since getting
those backwards in the test would give false confidence either way).
**Not yet run** — same standing caveat as every fix this session.

**Logging made unconditional and comprehensive**, specifically to remove the
ambiguity that cost two prior sessions:
- `ConsoleMain.prepareVisionHandler()` now logs an explicit build/version
  marker as its very first line — fires in under a second, so a stale build
  (we hit this twice: `StubForwardPassHandler`'s missing constructor, and
  the now-removed single-sample text-embedding log never appearing) is
  caught before the ~25-minute prefill, not after.
- `VisionAwareForwardPassHandler.forwardBatch()` and `.forward()` now log an
  **unconditional** entry line — no branch, no flag, nothing that can
  silently fail to fire — proving this exact code executed for this exact
  request. This directly replaces the previous single-sample
  `compareAndSet`-gated text-embedding-stats log, which — despite being
  logically correct (verified twice by re-reading the code) — never once
  appeared across two full runs, almost certainly because of a stale build
  rather than a logic bug, but there was no way to tell the difference from
  the log alone. There now is.
- The text-embedding-stats comparison itself is now an **aggregate over
  every text token in the window** (previously just the first one
  encountered), logged unconditionally once per `forwardBatch` call, plus an
  explicit image-token-vs-text-token count. More statistically meaningful
  than a single sample, and — critically — not gated behind any condition
  that could fail to trigger.

---

## Status

**Session 60** — used the new `./juno gguf-info` tool (Session 40) against the
real llava-v1.5-7b files for the first time. Found one concrete, provable bug
and confirmed several previously-uncertain code paths are actually correct.

**Found and fixed: wrong GELU variant in every ViT transformer block.**
The mmproj file's metadata explicitly declares `clip.use_gelu = false` — this
is llama.cpp's own flag distinguishing standard (tanh-approx) GELU from
`quick_gelu` (`x * sigmoid(1.702x)`, OpenAI CLIP's original activation).
`VisionEncoder`'s FFN (`mlp()`) called `gelu()` unconditionally at every one
of the 23 transformer blocks, regardless of what the file declared — this
metadata key was never read anywhere in the codebase before this session.
Both activation functions are smooth and bounded, so this was never going to
crash or produce NaNs — it would silently distort every patch's features in
a way fully consistent with the "coherent but semantically wrong" output
we've been chasing since Session 38.

**Fix:** added `VisionConfig.useGelu` (read from `clip.use_gelu`, default
`true` for files that don't declare it — preserves prior behavior for any
mmproj without this key) and `GgufReader.metaBool()` (new, mirrors
`metaInt`/`metaFloat`). `VisionEncoder.mlp()` now calls a new `activation()`
dispatcher instead of `gelu()` directly. Added direct unit tests for
`quickGelu()` (zero, asymptotic behavior, shape, and an explicit
"differs from standard gelu" test so the fix can't silently regress into a
no-op). **Not yet confirmed by an actual rerun** — same caveat as always:
this is evidence-backed, not guessed, but it hasn't been run yet.

**Confirmed correct (ruling these out for good, not just "looks fine"):**
using the *real* declared tensor shapes from `gguf-info` output rather than
assumption:
- `v.patch_embd.weight`'s real shape `[14, 14, 3, 1024]` and `patchEmbed()`'s
  flat indexing order (`dx` fastest, then `dy`, then `c`) match exactly, once
  accounting for GGUF's standard PyTorch-axis-reversal convention.
- `v.position_embd.weight`'s real shape `[1024, 577]` and the `posEmbd[i * H
  + d]` indexing in `encode()` match exactly (position slow-varying, hidden
  dim fast-varying, as declared).
- The `ffn_up`/`ffn_down` name-swap warning logged at every startup is now
  directly confirmed, not inferred: `v.blk.N.ffn_down.weight` really does
  have shape `[1024, 4096]` (an expand/up shape) and `v.blk.N.ffn_up.weight`
  really does have shape `[4096, 1024]` (a contract/down shape) — the
  existing shape-based correction in `loadFfn()` is necessary and correct,
  not a guess.
- `mm.2.weight` really exists with shape `[4096, 4096]` (the expected
  outputDim→outputDim shape for a 2-layer `mlp2x_gelu` projector) — the
  Session 38 hypothesis about projector shape was right; Session 39's
  revert was about applying it, not about whether it exists. Still not
  re-enabled — the earlier regression (degenerate repeating-token output)
  needs its own root cause before touching that code path again, and
  nothing in this session's findings explains *that specific* failure mode.

---

## Status

**Session 59** — investigating why image content is still wrong even after
Session 39's revert (output is coherent, varies between requests, but never
describes the actual image — a pixel-art "JUNO" logo). No code fix in this
session; code review + diagnostic instrumentation only.

**Code paths reviewed and confirmed correct** (ruling these out as the
cause): `ImagePatchEmbedder` (CLIP mean/std, RGB bit-shifts, CHW layout —
all correct), `VisionEncoder.patchEmbed()` (raster-order patch extraction,
correct CHW flattening matching `patch_embd.weight`'s documented shape),
`VisionEncoder.selfAttention()` (genuinely full bidirectional attention, no
causal mask — confirmed by reading the code, not just the comment),
`resolveFfnOrientation()` (shape-verified against measured tensor
dimensions, not a blind guess — a real transpose bug here would produce
gibberish, not coherent generic captions, which is inconsistent with what
we're seeing).

**Two live hypotheses, not yet distinguished:**
1. **Scale/magnitude mismatch.** If the projector's output magnitude is very
   different from a real LLM token embedding's magnitude, the transformer's
   residual stream could effectively drown out the (much smaller, or much
   larger) image signal, causing the model to fall back on language-model
   priors — which look exactly like "plausible, generic, wrong" captions.
2. **Model/architecture limitation, not a bug.** LLaVA-1.5-7B is
   well-documented to perform poorly on non-photographic imagery — pixel
   art, UI screenshots, diagrams, text-heavy graphics — because its training
   distribution (COCO-style captioning/VQA data) is dominated by natural
   photographs. A blocky, abstract, colorful logo is far outside that
   distribution. Both observed captions ("room with a doorway and a cat",
   "man standing in front of a camera") independently describe an indoor
   scene with a person — consistent with the model falling back to its most
   common training-data pattern when given a weak or out-of-distribution
   visual signal, which is also consistent with hypothesis 1.

**Added (diagnostic only, zero behavior change):** `VisionEncoder` now logs
min/max/mean/std and per-patch L2 norm across all projected patch embeddings
once per request. `VisionAwareForwardPassHandler` now logs the same stats
for one real text-token embedding, once per process, explicitly labeled for
comparison against the patch-embedding log line above it. Next run's log
will show whether the two are in the same ballpark (pointing at hypothesis
2 / model limitation) or wildly different (pointing at hypothesis 1 / a
real scale bug still to be found).

**Suggested next step to disambiguate independent of these logs:** run the
same request against an actual photograph (not pixel art) instead of the
JUNO logo. If the model correctly describes a real photo, that's strong
evidence for hypothesis 2 (model capability limitation, not a pipeline bug)
and further code changes are probably not the right lever. If it still
produces a generically wrong caption for a real photo too, that confirms a
remaining pipeline bug and rules out hypothesis 2.

---

## Status

**Session 58** — REVERT of Session 38's mm.2 projector change. Confirmed by
the user to be a regression, not a fix: output went from "coherent English
describing the wrong thing" (Session 38's starting point) to a fully
degenerate repeating `<image>` token loop (`finish_reason: length`, zero real
content) — a materially worse failure mode.

Applying `GELU(mm.0(x))` then `mm.2` produced numerically broken patch
embeddings for this specific mmproj file. The exact root cause is not
understood — the shape checks passed (mm.2.weight really is `[outputDim,
outputDim]`), so this is not a simple dimension-mismatch bug; it's either a
data-layout mismatch specific to this tensor, a wrong assumption about what
`mm.2` actually represents in this particular export, or something else not
yet identified.

**Reverted:** `VisionEncoder.project()` now always calls `applyProjector`
with `w2=null, b2=null` — i.e. mm.0 only, unconditionally — regardless of
whether `mm.2.weight` is present in the file. `mm.2` is still detected,
loaded, and shape-validated at load time purely for diagnostic logging (zero
effect on the actual data path); the log line makes clear it is present but
NOT applied. The `applyProjector` unit tests are unchanged (they test the
utility function's math in isolation) but are now explicitly annotated as
not being evidence that re-enabling mm.2 is safe.

**Do not re-enable mm.2 without new evidence.** If picked up again, the
right next step is dumping actual intermediate values (min/max/mean/NaN
count of the patch vectors immediately after `mm.0`+GELU, and again after
`mm.2`) for a real request, rather than another shape-level guess — this
failure mode (shape-valid, numerically degenerate) is exactly the kind of
bug that unit tests with small synthetic matrices cannot catch and that
needs runtime inspection of real quantized weights.

---

## Status

**Session 57** — vision output is now grammatically coherent (Session 37's
zero-vector-text-token fix confirmed working by the user's own log/curl
output) but describes the wrong image content ("a room with a doorway and a
cat" for a pixel-art logo). Root cause found by code inspection + comparison
against the standard llava-v1.5 architecture — **NOT yet confirmed by the
user re-running it.**

`VisionEncoder`'s projector (`mm.0.weight`/`mm.0.bias`) was a single linear
layer only. LLaVA-1.5's actual `mm_projector_type` is `mlp2x_gelu`: mm.0 →
GELU → mm.2 (a second `outputDim × outputDim` linear layer; llama.cpp mmproj
GGUF files name it `mm.2` because `mm.1` is the implicit, weight-less GELU).
The code never checked for `mm.2.weight` at all — it applied only half the
learned projector, which produces a shape-valid (so no crash, no shape
warnings) but semantically arbitrary vector for each image patch. That
explains the failure mode precisely: fluent English (text tokens are correct
after Session 37's fix) describing a plausible but unrelated scene (the image
tokens carry no real signal from the picture).

**Proposed fix (tried, then reverted — see Session 39 above):** `VisionEncoder`
checked `hasTensor("mm.2.weight")` at load time and, when present, loaded
`mm.2.weight`/`mm.2.bias` and applied the full `mm.0 → GELU → mm.2` chain.
This was confirmed by the user to be a regression, not a fix — see Session 39.

**This was a hypothesis about this specific failure mode that turned out to
be wrong (or at least this implementation of it was) — see Session 39 for
the confirmed outcome.**

---

## Status

**Session 56** — the vision pipeline wiring/hang fixes from Session 36 were
confirmed working end-to-end by the user (full log from `Prefill:` through
`generate() RETURNING`, 32/32 layers completed, curl returned a response).
Output was grammatically coherent but semantically incoherent ("
работаonc анаási tienenпозиступа...") — traced to a second, independent bug:

`VisionAwareForwardPassHandler.buildWindowActivationsWithVision()` (and the
single-token `buildActivationWithVision()`) spliced real CLIP patch vectors
into image-token positions but left every **text**-token position as an
all-zero vector — meaning the entire prompt text (chat template, the actual
question, BOS) was invisible to the model; only the image patches carried any
signal.

**Fix:** added `ForwardPassHandler.embedToken(int)` (default: throws
`UnsupportedOperationException`) so a decorator can ask the wrapped handler
for a single token's real embedding-table row. Implemented in
`LlamaTransformerHandler.embedToken()`, reusing the existing clamping logic.
Both vision-splicing methods now call `textHandler.embedToken(tokenId)` for
non-image positions instead of leaving zero. **Confirmed fixed** by the user:
rerunning the same curl produced grammatically fluent, on-topic English
output for the first time (see Session 38 above for the next bug this
uncovered — content was fluent but still hallucinated, unrelated to the
projector fix below it in this file).

Updated the two existing `VisionAwareForwardPassHandlerBatchTest` cases that
had asserted the *old, buggy* zero-vector behavior (they'd now be actively
wrong), added a new single-token-path regression test, and added direct
`embedToken()` tests to `LlamaTransformerHandlerEmbeddingsNodeActivationsTest`.
The shared `StubForwardPassHandler` test double (lives in
`vision/src/main/java`, not `test` — pre-existing structure, not something
introduced here) gained a configurable deterministic fake embedding.

---

## Status

**Session 55** — `--local` mode with `--mmproj-path`: fixed the vision request
hang/never-replies bug (two stacked bugs, both traced via JFR thread dumps,
not guessed). LoRA (`--lora-play`) hang reported by the same user is still
**unresolved** — no reproducing JFR capture was available to trace it; see
"Still open" below.

### Vision requests via `--mmproj-path` never replied — root cause traced via JFR

Symptom: `POST /v1/vision/chat` against a LLaVA model loaded with
`--mmproj-path` never returned, even for a tiny image, and looked like an
infinite loop from the outside (`curl` just hangs).

A previous debugging session (see `bug-trace.txt`) worked through several
hypotheses by reading code alone and could not confirm a root cause. This
session captured a JFR recording of an actual hung request
(`juno-llava-v1.5-7b-Q4_K-*.jfr`) and used `jdk.jfr.consumer.RecordingFile`
(no `jfr` CLI available in this environment) to read the embedded
`jdk.ThreadDump` events. The live stack trace showed the request thread
executing straight through `LocalInferencePipeline.prefillBatch() →
LlamaTransformerHandler.forwardBatch() → runLayersBatch() →
transformerLayerBatch() → gqaInto()` — **with no
`VisionAwareForwardPassHandler` frame anywhere in the stack.** That pointed
directly at two independent, stacked bugs:

**Bug 1 — vision handler wrap discarded (`ConsoleMain.runLocalRepl()`).**
`wireVisionRoutes()` (which wraps `handlers.get(0)` with
`VisionAwareForwardPassHandler` via `LlavaHandlerFactory.buildFromHandlers()`)
was called *after* `LocalInferencePipeline.from(shardMap, new
ArrayList<>(handlers), ...)`. `LocalInferencePipeline.from()` reads
`handlers.get(i)` once, at construction, and stores that exact reference in
each `NodeStage` — it never re-reads the list. Wrapping `handlers.get(0)`
afterwards therefore had no effect on the already-built pipeline: every
request silently ran through the plain-text `LlamaTransformerHandler`, which
has no knowledge of the CLIP patch vectors `VisionChatHandler` registered.
576 `<image>` placeholder tokens were looked up as ordinary (meaningless)
vocabulary rows and batched through a full `W=1187`-wide prefill on a 7B
model, purely on CPU (weight-stationary Q4_K GEMM, no SIMD/GPU) — tens of
trillions of MACs, genuinely finite but so slow it looked exactly like a
hang. Even had it returned, the reply would never have described the image.

**Fix:** split `wireVisionRoutes()` into `prepareVisionHandler()` (wraps
`handlers.get(0)` — now called *before* `LocalInferencePipeline.from()`) and
`registerVisionRoutes()` (registers the two HTTP routes once `apiServer`
exists, unchanged timing). `LlavaHandlerFactory`'s class-level javadoc, which
documented the old (buggy) call order as the correct usage, is corrected.

**Bug 2 — `LlamaTransformerHandler` branched on the wrong flag (would have
crashed once Bug 1 was fixed).** Both `getInitialActivation()` (the
single-token `forward()` path) and `forwardBatch()` decided whether to read
`request.tokenIds()` or `request.activations()` using the handler's own fixed
`hasEmbeddings` field (always `true` for node 0) instead of asking the
request what it actually carries. `VisionAwareForwardPassHandler` correctly
builds an activations-based request for node 0 (`withActivations(...)`,
which sets `tokenIds = null`) — but `LlamaTransformerHandler` would still try
`request.tokenIds()[b]` and throw `NullPointerException`. This bug was
latent/unreachable while Bug 1 stood (node 0 never received an activations
request in the first place), which is why the symptom was "hang", not
"crash" — fixing Bug 1 alone would have turned the hang into an NPE.

**Fix:** both call sites now check `hasEmbeddings && request.isFirstNode()`
(`ForwardRequest`/`BatchForwardRequest` already exposed `isFirstNode()` —
`tokenIds() != null` — for exactly this purpose; it just wasn't being
consulted). Behaviourally identical to before for ordinary text requests,
since a real first-node request always has non-null `tokenIds`.

Regression tests:
- `LocalInferencePipelineTest` — two new tests
  (`mutating_handlers_list_after_pipeline_construction_has_no_effect`,
  `...before_pipeline_construction_is_picked_up`) pin down the exact
  handler-list snapshot-timing contract that Bug 1 violated, without needing
  real GGUF fixtures.
- `LlamaTransformerHandlerEmbeddingsNodeActivationsTest` (new file) — covers
  Bug 2 for both `forward()` and `forwardBatch()`: activations-based requests
  on the embeddings node no longer throw, produce correctly shaped output,
  and diverge from a token-embedding lookup; ordinary token-based requests on
  the same node are asserted unchanged.

**Tracing method for future reference:** this environment ships a JRE with
the `jdk.jfr` module but no `jfr` CLI tool and no `javac`. `java
SomeTool.java recording.jfr` (JDK 11+ single-file source launch) using
`jdk.jfr.consumer.RecordingFile` to iterate `RecordedEvent`s — filtering on
`jdk.ThreadDump` for stack traces, or on the project's custom `juno.*` events
(`juno.ForwardPass`, `juno.MatVec`, `juno.LoraTrainStep`) for progress
counters — is enough to distinguish "genuinely stuck/deadlocked" (identical
stack across dumps, thread parked/blocked) from "just very slow" (stack
advances, CPU time and event counters keep climbing) without needing to
attach a debugger or reproduce interactively.

**Still open — `--lora-play` hang on ordinary (non-vision) handlers.** The
user also reported that LoRA adapter playback hangs. None of the JFR
recordings included in this session's evidence contain a
`juno.LoraTrainStep` event or a `handlerType=lora` `juno.ForwardPass` event,
so there is no captured stack trace of an actual LoRA hang to trace — every
available recording is either a plain `llama` run or vision-related. Per
CLAUDE.md ("don't guess"), no root cause is claimed here. A quick source
check shows `LoraTrainableHandler` does *not* share the `VisionAwareForwardPassHandler`
wiring path (vision only ever wraps `handlers.get(0)`; LoRA is unrelated to
that code path), so Bug 1/Bug 2 above are unlikely to be the same root
cause — but this needs its own JFR capture of a real hang to confirm.
**Next step:** reproduce with `--local ... --lora-play PATH --jfr 5m` (the
existing `--jfr DURATION` flag ConsoleMain already supports, e.g.
`startLocalJfr()`) while a LoRA request is hung, then read the resulting
`juno-<model>-<timestamp>.jfr` the same way described above.

---

**Session 54** — `--local` mode: fixed `--verbose` no-op and vision models never being detected.

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

**Session 53** — F16-weighted GGUF models failed every inference request:
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

**Session 52** — EU AI ACT Complience User transparency and AI disclosure.

InferenceApiServer.java; ConsoleMain.java and juno-api.yaml was updated with `The replies are generated by an AI system` water-mark.

##Status

**Session 51** — Documentation update: docs/ folder restructured as juno-documentation MyST Jupyter Book.
juno-documentation

The flat docs/ folder has been restructured into juno-documentation/, a standalone MyST-MD

Jupyter Book configured via myst.yml.

Content is organised into 11 parts and 54 chapters, each in its own .md file under part1/ through part11/. Navigation links (<- / ->) and a full Table of Contents in index.md cross-link every chapter.

All Mermaid diagrams are declared with the MyST {mermaid} directive where applicable and render natively in the built book and in any Mermaid-aware viewer.

A references.md back-matter table maps every chapter back to the originating file in docs/.

build.sh provides a one-command build (./build.sh); README.md documents prerequisites and the local preview workflow.

## Status

**Session 50** — `/train-file-qa`: multi-fact Q&A from a JSON file in one training loop; HTTP API.

### `/train-file-qa`

- REPL command loads a `.json` array of `{"Q","A"}` objects via `LoraQaFile`.
- Each pair expands to the same four chat-templated variants as `/train-qa`; all units
  train in one `trainOnUnits` pass with QA loss targets.
- `LoraTrainer.trainQaPairsUntilResult` for the programmatic multi-pair path.
- `LoraApiServer` — with `./juno lora --api-port N`: `POST /v1/lora/train-file-qa`
  (JSON body) and `POST /v1/lora/save` for curl workflows.
- Dropped verbose `[TRACE]` dump of formatted training text / token IDs on `/train-qa`.
- Docs: `docs/LoRA.md`, `docs/howto.md`.

---

## Status

**Session 49** — LoRA Tier 11 (complete): `--lora-microbatch` CLI/env + VRAM OOM auto-fallback.

### LoRA microbatch CLI and VRAM ladder (Tier 11)

- `LoraMicrobatch` — `--lora-microbatch N` / `LORA_MICROBATCH` (default 8, range 1..128);
  applies `juno.lora.microbatch` before resident upload (no `JAVA_TOOL_OPTIONS` required).
- `LoraResidentUpload` — on FP32 microbatch VRAM OOM with half support: close, set
  microbatch=1, retry FP16 once; further OOM uses existing auto→CPU / gpu fail-closed policy.
- Wired through `LoraCliOptions`, `LoraTrainingConfig`, `ConsoleMain`, `LoraTrainer`,
  `scripts/run.sh` / `run.bat`, and all three LoRA training handlers.
- Docs: `docs/LoRA.md`, `docs/howto.md`, `docs/performance.md`, `docs/agent-arch.txt`.

---

## Status

**Session 48** — LoRA Tier 9 (complete): microbatch GEMM + published GPU speed gates.

### LoRA GPU microbatch and product gates (Tier 9)

- `GpuBlasOps` / `DeviceActivationBatch` — FP32 `cublasSgemm_v2` / `rocblas_sgemm` microbatch
  for frozen forward and transpose; CPU oracle `CpuFrozenBatchOps`.
- Default `juno.lora.microbatch=8` uploads FP32 resident weights and batches linears across
  positions in `LoraTrainableHandler.computeGradients` (host adapters / Adam unchanged).
- `LoraTrainableHandlerGpuBackwardTest` — CPU↔GPU loss/grad parity + TinyLlama speed gates
  (GTX 1080: **~14× e2e**, **~11× backward** vs CPU).
- Docs may describe production **GPU LoRA training** as frozen batched GPU + host adapters;
  device-resident adapters / GPU Adam remain deferred (not required after intensity proof).
- `--lora-train-device` and LLaMA/Qwen2 timing subsets remain as in Session 46 (`transferMs` still 0).

---

## Status

**Session 47** — LoRA Tier 10 (complete): multi-arch GPU residency + production gates.

### LoRA multi-arch GPU residency (Tier 10)

- `LoraResidentWeights` — shared upload / close / VRAM-OOM fallback / matVec+transpose routing.
- `LoraTrainableHandler` refactored onto the helper (LLaMA-family / Qwen2 unchanged behavior).
- `Phi3LoraTrainableHandler` / `Qwen3LoraTrainableHandler` upload physical fused (Phi) or dense
  (Qwen3) projections when `--lora-train-device` resolves to a `GpuMatVec`; CPU fallback preserved.
- Gated live LoRA smokes (`LoraLiveSmokeTest`) for TinyLlama / Qwen2.5 / Phi-3.5 / dense Qwen3 fixtures.
- `EosOutputFilter` — hold back / strip turn-end markers (`</s>`, `<|end|>`, `<|im_end|>`, …) so
  `/train-qa` completions never stream into REPL or `GenerationResult` text (all LoRA chat templates).
- DoRA: correctness-complete, **not** production-perf-gated (prefer LoRA/rsLoRA for large all-linear jobs).
- Tier 7 JFR metrics marked **complete** (programmatic `--jfr`, mode identity, extractor, docs).
- Tier 5 held-out research / quality matrix remains **deferred**; exact K-quant QA-LoRA merge unsupported.

---

## Status

**Session 46** — LoRA Tier 9 (start → completed in Session 48): `--lora-train-device` productization.

### LoRA GPU train-device (Tier 9)

- `--lora-train-device auto|gpu|cpu` / `LORA_TRAIN_DEVICE` (default **auto**).
- `LoraTrainDevice` — MatVec selection; `gpu` fails closed without CUDA/ROCm; `cpu` forces `CpuMatVec`.
- `LoraTrainer` / LoRA REPL honor the mode; JFR `trainDevice` is the resolved label (`cpu`/`cuda`/`rocm`).
- `LoraStepTiming` — fills `frozenForwardMs` / `frozenTransposeBackwardMs` / `adapterBackwardMs` /
  `attentionNonlinearMs` on `juno.LoraTrainStep` from LLaMA/Qwen2 handler instrumentation (`transferMs` still 0 until H2D counters).
- Microbatch / parity IT / speed gates: completed in Session 48.

---

## Status

**Session 45** — LoRA Tier 8: train-file scheduling and corpus caps.

### LoRA train-file scheduling (Tier 8)

- `--lora-chunk-tokens` / `LORA_CHUNK_TOKENS` (default **32**; recommend **128** for large `/train-file`).
- `--lora-max-train-tokens` / `LORA_MAX_TRAIN_TOKENS` (`0` = unlimited): seeded whole-chunk subsample of supervised prediction tokens.
- `/train` and `/train-file` use document-level `TrainUnit`s; chunking happens inside `LoraTrainingLoop`.
- `LoraCorpusLimit` helper; docs/help no longer claim a silent 128 default.

---

## Status

**Session 44** — LoRA training progress bar (loss → target).

- `LoraTrainProgressBar` — percent from pass-2 baseline loss toward `--lora-loss-target-*`; max-iters not used.
- ETA from loss-improvement rate since baseline; final frame ETA `0s` when the run ends.

---

## Status

**Session 43** — LoRA Tier 6: multi-architecture training (CPU oracle).

### LoRA multi-architecture (Tier 6)

- `LoraTrainingHandler` / `LoraTrainingHandlerFactory` — explicit allowlist by `general.architecture`.
- `LoraModelLayout` / `LoraProjectionBinding` — logical keys → physical GGUF tensors (Phi fused slices).
- Handlers: LLaMA-family (`LoraTrainableHandler`), `Qwen2LoraTrainableHandler` (frozen QKV biases),
  `Phi3LoraTrainableHandler` (fused QKV/gate-up + NeoX RoPE), `Qwen3LoraTrainableHandler`
  (per-head Q/K RMSNorm, `qDim`).
- `LoraMerge` layout-aware multi-adapter fused-slice F32 patching for Phi-3.
- Rejected for LoRA: `qwen3moe`, `qwen35`, `gemma`, unknown.
- Qwen3 `/train-qa` template parity with empty `<think>` block.

---

## Status

**Session 42** — LoRA REPL UX + WebUI model dropdown.

- `/reset` deletes the `.lora` checkpoint (no overwrite save); memory reset + chat history clear unchanged.
- LoRA banner and chat footer show sampling `temperature` (and top-k / top-p on the banner).
- Default LoRA training log is a compact progress bar; `--verbose` / `-v` restores full `[TRACE]` / per-pass lines.
- WebUI model dropdown parses OpenAI `GET /v1/models` (`data` / `id` / `x_juno_*`) so names appear again.

---

## Status

**Session 41** — LoRA Tier 7 (complete): JFR metrics for all adapter modes and operations.

### LoRA JFR metrics (Tier 7)

- Programmatic LoRA `--jfr` lifecycle matches local mode (`jdk.jfr.Recording` + auto-extract `target/metrics/metrics.json` on exit). Launchers pass `--jfr` as an app arg (no `-XX:StartFlightRecording` for LoRA).
- `LoraMetricsIdentity` — CLI vocabulary tags (`lora` / `rslora` / `dora` / `qa-lora`) on train, validation, merge, norm-refresh, playback, and checkpoint events.
- New events: `juno.LoraNormRefresh`, `juno.LoraMerge`, `juno.LoraPlayback`, `juno.LoraCheckpoint`.
- `JfrMetricsExtractor` aggregates train/validation/merge/DoRA-refresh/playback series with guarded field reads (older recordings still extract).

---

## Status

**Session 40** — LoRA Tier 5 (complete implementation): QA-LoRA + merge policies.

### LoRA QA-LoRA and quantized merge (Tier 5)

- Gate A codecs retained: `QuantizationLayout`, `GgufQuantCodec` / `GgufKQuantCodec` (`juno-kquant-v1`), `QuantizedMergeMetrics`.
- `QaLoraAdapter` — sum-pool grouped A (`rank×groupCount`) + B; dense-expansion oracle and finite-difference tests.
- `AdapterAlgorithm`, `MergeCapability` (`SIDECAR_ONLY` / `F32_PRESERVE` / `SOURCE_TYPE_PROJECTED`; `EXACT_AFFINE` rejected for K-quants).
- Checkpoint v2: QA entries store `groupWidth` before A, Tier-5 extension blob (algorithm, pooling, ggml type, encoder id, merge policy); v1 export rejected for QA-LoRA.
- `QaLoraInitializer` — group width from actual tensor GGML type (Q4_K/Q5_K→32, Q6_K→16); fingerprints verified on load.
- Training/playback: `LoraTrainableHandler`, Adam, gradients, CLI `--lora-mode qa-lora`, `--lora-group-width`, `--lora-merge`.
- `LoraMerge` — F32 preserve (default) and explicit `SOURCE_TYPE_PROJECTED` requantization with per-tensor metrics; zero-delta copies raw bytes; never silent exact→projected fallback.
- Exact K-quant QA-LoRA zero-point merge remains **no-go**. Full held-out experiment matrix / deployment quality gates are research follow-ups; sidecar + F32 stay production-safe.

---

## Status

**Session 39** — LoRA Tier 5 (Gate A start): shared GGUF K-quant codec layer.

### LoRA QA-LoRA / quantized merge foundations (Tier 5 Gate A)

- `QuantizationLayout` — Q4_K / Q5_K / Q6_K geometry (block/sub-block width, affine vs symmetric).
- `GgufKQuantCodec` / `GgufQuantCodec` — versioned encoder id `juno-kquant-v1`; decode matches llama.cpp goldens; encode moved out of `LoraMerge`.
- `QuantizedMergeMetrics` — RMSE, max error, delta-retention helpers for projected merge.
- `GgufReader` and `LlamaTransformerHandler.dequantize` delegate K-quant decode to the shared codec; fused matVec paths unchanged for performance.
- No-op path: `copyRawUnchanged` — decode/re-encode must not be used for byte-identical preservation.
- Non-closure tests: Q6_K additive shift and Q4_K nested-scale offset are not exact (exact K-merge remains no-go).
- Next: grouped QA-LoRA math (Gate B), merge capability policy, then projected merge experiments.

---

## Status

**Session 38** — LoRA Tier 4 (start): resident transpose primitives and baseline instrumentation.

### LoRA GPU training foundations (Tier 4)

- Vendor-neutral `GpuBindings.opNoTranspose()` (CUDA `CUBLAS_OP_N=0`, ROCm `rocblas_operation_none=111`).
- `GpuMatVec.sgemvTranspose` for resident FP32/FP16 `W^T * g` (same row-major buffer as forward `OP_T`).
- `ResidentWeightMatrix` + `LoraTrainableHandler` routes frozen forward and transpose backward through resident GPU weights when uploaded (`supportsHalfResident` FP16 or FP32 fallback).
- JFR backend labels: `*-resident-transpose` / `*-resident-fp16-transpose`.
- `LoraTrainEvent` fields for frozen forward/transpose, attention/nonlinear, adapter backward, and transfer (filled when finer instrumentation lands).
- GPU adjoint tests: `CudaMatVecTransposeTest`, `RocmMatVecTransposeTest` (`GpuMatVecTransposeContractTest`).
- Baseline section in `docs/performance.md` — hybrid path is not yet marketed as production GPU training.
- `--lora-train-device` shipped in Session 46; CPU/GPU gradient parity IT and speed gates remain open.
- Fix: `LoraAdapterSet.resetFrom` (REPL `/reset`) bumps DoRA cache generation so inference drops trained magnitude coefficients.
- Fix: `/reset` also clears REPL chat history and rotates the session id — otherwise multi-turn context still contains the memorized answers.

---

## Status

**Session 37** — LoRA Tier 3 (phase 1–2): rsLoRA, Kaiming, checkpoint v2, DoRA.

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

**Session 36** — LoRA Tier 2: schedules, AdamW, dropout, validation, and LoRA+.

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

**Session 35** — LoRA Tier 1: projection coverage, token-weighted accumulation, and clipping.

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

**Session 34** — Windows launcher fixed: `run.bat` and `juno.bat` fully functional on Windows.

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

---

## Status

**Session 23** — JFR auto-extraction for local and cluster modes; AWS deploy gains `--jfr` with remote JFR collection. *(unchanged)*

**Session 16** — Naming cleanup: the Session-12 GPU/hardware rename applied consistently across remaining files, tests, and Javadoc. *(unchanged)*

**Session 13** — Tensor parallelism added as a second parallelism mode (`--ptype tensor`, alongside pipeline). Star/coordinator-centric AllReduce. Follow-up fixes: `ClusterHarness.startTensorParallel()` no longer hardcodes TinyLlama constants; fixed an eager `float[]` OOM in tensor-parallel mode; added tensor-parallel checks 7–8 to `ModelLiveRunner`.

**Session 12** — Renamed the GPU/hardware-facing classes for clarity ahead of the ROCm work (`GpuMatVec` → `MatVec`, etc.); removed the now-redundant `GpuForwardPassHandler`; kept `ForwardPassHandler` and `TransformerHandler` as separate interfaces by design (routing vs. math).

**Session 11** — Phi-3 family support: `Phi3TransformerHandler` (Phi-3's attention/FFN shape differs from LLaMA's), `ForwardPassHandlerLoader` routes by GGUF architecture metadata, lazy dequantization via `GgufReader.QuantizedTensor`/`tensorRaw()`, a vocab-size fix in `LlamaConfig`, and a chat-template routing fix so `ChatTemplate.forModelType()` resolves by exact match before falling back to substring match.

**Session 10** — GPU acceleration layer: `MatVec` interface abstracts the matmul backend, with `CpuMatVec` (existing parallel path) and `CudaMatVec` (cuBLAS via `org.bytedeco`) implementations; `DeviceFloatMatrix` uploads weights to VRAM once; `GpuContext` owns the cuBLAS handle lifecycle; `CudaAvailability` does safe, cached CUDA runtime detection so CPU-only machines don't crash on class load. GPU tests gated behind a `-Pgpu` Maven profile and `@Tag("gpu")` to keep default CI GPU-free.

**Session 9** — Multi-turn session KV cache reuse. Previously every REPL turn re-prefilled the entire conversation from scratch (latency grew turn over turn: 23s → 75s on TinyLlama). Added a stable `sessionId`-keyed cache path (`InferenceRequest.ofSession()`, `GenerationLoop.kvCacheKey()`) so only new tokens are prefilled. Turn latency is now flat (~7-8s) regardless of conversation length.

**Session 8** — Cross-platform launcher: unified `juno` / `juno.bat` entry points delegating to `scripts/run.sh` / `scripts/run.bat`, with `console`, `cluster`, and `live` subcommands and shared JVM flags (`--enable-preview`, `--enable-native-access`, ZGC/G1 tuning). Added `logback.xml` to `juno-player` and `integration` to silence default DEBUG-level gRPC/Netty logging. Fixed all 6 `ModelLiveRunner` checks (greeting-vocabulary coverage, raw SentencePiece marker leakage, determinism assertions).

**Session 7** — Fixed broken multi-turn REPL output. Root cause: `GenerationLoop` evicted KV cache after every turn while still hitting the prefix cache, serving stale/freed KV. Interim fix: disable prefix cache for the single-request path and always re-prefill (correct but O(N), later superseded by Session 9's proper fix). Also fixed chat-template selection for TinyLlama via `ChatModelType.fromPath()`.

**Session 6** — Performance and correctness: parallelized `matVec()` in `LlamaTransformerHandler`, parallelized shard loading across nodes, suppressed the EOS token's decoded piece from streamed output, and switched the default activation dtype to FLOAT16 with JVM tuning flags in `scripts/run.sh`.

**Session 5** — Three correctness bugs found during real-model verification with TinyLlama, all fixed: (1) missing `"tinyllama"` chat-template registration caused ChatML tokens to be sent to a Zephyr-trained model, producing garbage output; (2) `decodeToken()` leaked the raw SentencePiece `▁` space-marker character into streamed output (batch `decode()` was already correct); (3) `Q6_K` dequantization used a flat loop that indexed the wrong `qh` byte for block positions ≥32, corrupting the majority of every Q6_K-quantized weight tensor. Added a golden-value regression test (`GgufReaderTest`) built against synthetic in-memory GGUF files.

**Session 4** — Real model inference wired end-to-end for the first time: `GgufReader` (pure-Java GGUF v2/v3 parser, no JNI, dequantizes Q4_0/Q4_K/Q6_K/Q8_0/etc.), `LlamaConfig` (hyperparameters from GGUF metadata), `LlamaTransformerHandler` (LLaMA-family forward pass: RMSNorm, RoPE, GQA attention, SwiGLU FFN, residual connections), and `GgufTokenizer` (SentencePiece BPE reading its vocabulary straight from GGUF metadata, no external `tokenizer.model` file). Also split prefill and decode into two explicit phases in `GenerationLoop`, matching standard LLM inference practice.

---

## 13. Build Status (snapshot, session 15)

All modules built SUCCESS on JDK 25: `api`, `registry` (11 classes), `tokenizer` (9), `sampler` (9), `kvcache` (8), `health` (6), `node` (26, largest module — transformer handlers, GGUF codec, GPU matmul backends, LoRA), `coordinator` (14), `juno-player` (6 main classes), and the `integration` test module (`ModelLiveRunner`, `InProcessClusterIT`, `ThreeNodeClusterIT`, `TensorParallelClusterIT`, `GpuForwardPassIT`). Roughly 475 `@Test` methods total, 0 failures, 0 errors.

## 12. Technology Summary

Java 25, multi-module Maven build. GPU compute via `org.bytedeco` CUDA bindings (cudart + cuBLAS); distributed state and leader election via Hazelcast (`CP FencedLock`); node-to-node data plane over gRPC/Protobuf; RDMA networking via jVerbs; concurrency via Java 25 virtual threads and `CompletableFuture`; REST API served by Javalin (deliberately not Spring Boot), spec generated from OpenAPI 3.0; KV cache is two-tier (GPU VRAM, then Caffeine on JVM heap, no disk tier); circuit breaking via Resilience4j; metrics via Micrometer/Prometheus behind the JDK's built-in `HttpServer`; tokenizer via DJL SentencePiece JNI (with a pure-Java `GgufTokenizer` fallback); weights in GGUF format; sampler is pure Java with zero external dependencies.

## 11. Full Configuration Reference

Original YAML configuration skeleton covering `cluster` (seed nodes, backup count), `coordinator` (ports, queue depth, batch size, preemption strategy), `scheduler` (max wait, priority weights), `node` (gRPC port, VRAM headroom), `kv-cache` (GPU/CPU tier capacity and eviction policy — no disk tier), `health` (probe interval, VRAM warning/critical thresholds, circuit breaker parameters), and `sampling` (default and named profiles: `deterministic`, `creative`). Current authoritative flag/config reference lives in the CLI Reference part of `juno-documentation`.

## 10. Full Token Generation Data Flow

End-to-end trace of a single request: tokenizer encodes the chat-templated prompt; the prefix cache is checked for a reusable KV prefix; the pipeline prefills all prompt positions except the last; the decode loop then calls `pipeline.forward()` once per step, samples the next token, streams it to the client (SSE/gRPC), and repeats until EOS or `maxTokens`; on completion the future resolves and (for sessions) the prefix is cached rather than evicted.

## 9. Activation Compression and Integration Test Infrastructure

**Activation compression:** pipeline-parallel node hops ship full activation tensors over the network; at 70B scale (hidden_dim 8192, seq_len 4096) that's 64MB per hop in FLOAT32. Added an `ActivationDtype` field (FLOAT32/FLOAT16/INT8) negotiated per request so hops can trade precision for bandwidth (FLOAT16 halves the payload, INT8 quarters it), and so heterogeneous nodes (different VRAM budgets) can each request the precision they can afford — this is the quantization-aware sharding mechanism, complementing the GGUF file's own per-layer weight quantization. Implemented in `node/ActivationDtype.java` and `node/ActivationCodec.java` (manual IEEE-754 half-float bit manipulation, no JNI).

**Integration test infrastructure:** `InProcessClusterIT` (zero network, in-JVM stub pipeline, ~250ms) and `ThreeNodeClusterIT` (forks 3 real `NodeMain` JVMs, real gRPC, ~16GB memory budget) exercise the cluster end to end. `ModelLiveRunnerIT` runs 6 real-model checks (greeting response, no raw SentencePiece markers, question answering, greedy determinism, multi-turn conversation, FLOAT16 parity) against an actual GGUF file, disabled by default and activated with `-Pintegration`. `LoadShardsParallelTest` is the timing regression anchor proving shard loading happens in parallel, not serially.

## 8. Actors — Design Decisions

Model registry and shard planning live in a Hazelcast distributed `IMap` (no single point of failure); seed-node election uses an IMQ-inspired weighted score (connectivity, stability, betweenness centrality, VRAM); sharding is greedy and VRAM-aware but capped per node so a single large-VRAM node can't starve later nodes of layers (`ShardPlanner`'s fairness cap). The coordinator uses static micro-batching (configurable window/size, default 8 requests / 50ms), a `PriorityBlockingQueue` (HIGH/NORMAL/LOW), and Java virtual threads throughout; `FaultTolerantPipeline` wraps each node in its own circuit breaker with a configurable retry policy (`none`/`once`/`aggressive`) and reports `CIRCUIT_OPEN` or `RETRIES_EXHAUSTED` as HTTP 503 with a `Retry-After` hint. The tokenizer supports LLaMA/TinyLlama/Mistral/Gemma chat templates by model-id lookup, defaulting to ChatML. The sampler is a pure-Java pipeline: temperature → top-k → top-p → softmax → repetition penalty → sample.

## 7. REST / HTTP — Revised Design

Deliberately not Spring Boot — too heavy for the target footprint. REST is served by Javalin (built on Jetty, ~1MB, explicit routing, no annotation magic, a good fit for virtual threads); the metrics endpoint uses the JDK's built-in `HttpServer` plus Micrometer, avoiding any extra framework dependency for a single `/metrics` scrape endpoint.

## 6. KV Cache — Revised Design

The original three-tier design (GPU + off-heap + disk) was simplified to two tiers, RAM only, no disk IO ever: GPU VRAM (hot, active sequences) and JVM heap via Caffeine (warm sequences, W-TinyLFU eviction). The off-heap and disk-backed candidates (OHC, Ehcache 3, Chronicle Map) were all dropped for the same reason: dead or JAXB-transitive-dependency-poisoned Maven artifacts that don't build cleanly on JDK 25. The prefix cache (a trie, checked before every forward pass) is unchanged — its purpose is to let concurrent clients sharing a system prompt pay for that prefix's compute once.

## 5. System Architecture

Clients reach the cluster via REST (Javalin) or streaming gRPC, behind a load balancer, to one of two coordinators (leader/standby, Hazelcast `CP FencedLock` for leader election). The leader owns tokenization, request scheduling, the autoregressive generation loop, sampling, and the prefix cache, and drives an `InferencePipeline` that fans out over the node cluster: gRPC carries the data plane (activations), Hazelcast carries the control plane (commands, state, health events). Each GPU node owns a contiguous slice of transformer layers; the first node also owns the embedding table, the last node owns the output projection.

## 4. API Module — What Was Built

Client-facing REST surface (`POST /v1/inference`, `/v1/inference/stream` for SSE, model load/list/status/unload, cluster health/nodes/shardmap) generated from an OpenAPI 3.0 spec via `openapi-generator-maven-plugin` (jaxrs-spec mode). Internal node-to-node communication is a separate gRPC surface (`inference.proto`, never exposed to clients) with three services: `InferenceService` (client-facing), `NodeService` (coordinator-to-node forward pass/shard load/unload), and `RegistryService` (internal shard-map queries).

## 3. Maven Project Structure

Multi-module Maven build, JDK 25 throughout, group `cab.ml.juno`, artifact `juno`. Shared libraries: `api` (OpenAPI + gRPC + proto codegen), `registry`, `tokenizer`, `sampler`, `kvcache`, `health`, `lora`. Core engine: `node` (GGUF parsing, quantization codecs, CPU/CUDA/ROCm matmul backends, transformer handlers). Orchestration: `coordinator` (scheduling, generation loop, REST) and `juno-player` (the REPL and cluster harness). Executables: `juno-master` (`CoordinatorMain`) and `juno-node` (`NodeMain`). Key dependency versions locked early and unchanged since: Hazelcast 5.4.0, gRPC 1.63.0, Protobuf 3.25.3, `org.bytedeco` CUDA 12.6-9.5-1.5.11, DJL 0.27.0, Caffeine 3.1.8 (the only cache library — see §6), Resilience4j 2.2.0, Micrometer 1.13.0, Javalin 6.3.0 (explicitly not Spring Boot), JUnit 5.10.2. Spring Boot, OHC, Ehcache, and Chronicle Map were all evaluated and removed early for transitive dependency failures (dead repos, JAXB dependency chains).

## 2. Hardware Stack

Reference cluster: 16 commodity PCs, each with a 4GB-VRAM GPU (64GB total VRAM across the cluster), an 8+ core CPU, 16-32GB RAM for the KV cache JVM heap, and NVMe storage for fast shard loading. Networking: 10GbE to start (25GbE ideal), a managed switch with jumbo frames, RDMA (GPU-to-wire, bypassing the CPU). Total extra networking cost for 16 machines: roughly $800-1000 — far cheaper than a single 64GB GPU.

## 1. Vision

A fully Java-native distributed LLM inference engine that runs large language models across a cluster of commodity GPUs, replacing the need for a single expensive high-VRAM card with a network of affordable machines. Core philosophy: no Python, no GIL, real threads; no Spring Boot, no framework bloat; commodity hardware over premium hardware; Java distributed tooling (Hazelcast, gRPC) over NCCL/MPI; pipeline parallelism (LAN-friendly, no InfiniBand required); open source, Java ecosystem first.

```
  juno — Distributed Java LLM Inference Engine
  Full Architecture Design Document
  JDK 25 · Maven · Java-native · Commodity GPU Cluster
```
