(ch-12-5)=
# 12.5. Known Issues and Fixes

Vision support was brought up against two real model families, moondream2 (Phi-2 + SigLIP,
embedded-GGUF) and LLaVA-1.5 / llava-phi-3-mini (LLaMA-2 or Phi-3 + CLIP, two-file). Each
uncovered a distinct set of bugs. This chapter records what was found, fixed, confirmed, or is
still open, so the same failure is not re-diagnosed from scratch. Status labels below (VERIFIED,
FIXED, OPEN) reflect the project's own `CHANGELOG.md` session notes.

## moondream2 (Phi-2 + SigLIP) -- VERIFIED fixed, tested against a real photo

Three stacked bugs were found and fixed together, then confirmed against a real photograph
(three people in front of a lit building) on the actual `moondream2-q5_k.llamafile`:

| | before any fix | RoPE fix only | + normalisation | + normalisation + post_ln |
|---|---|---|---|---|
| output | word salad, real words no grammar | grammatical, wrong content ("wooden stick and brush") | grammatical, vague/generic ("close-up... small details") | grammatical, substantially correct ("three people standing together in front of some buildings... urban setting") |
| per-patch L2 norm (mean) | 7117 | 9272 | 9272 | 71 |

The final per-patch L2 norm (mean 71.0, std 1.67 across 2048 dims) sits in the expected
LayerNorm'd range for this output width, confirming the third fix is genuinely exercising on the
real model file rather than only in a synthetic unit test. Remaining inaccuracies in this
specific response (a fabricated "collage" framing, extra objects not in the photo) are
consistent with normal hallucination for a roughly 1.6B-parameter, q5_k-quantized VLM, not a
further pipeline bug: the model correctly identifies the actual subject and hedges
appropriately ("appears to be", "possibly") rather than stating fabricated detail with false
confidence.

### FIXED: wrong RoPE pairing produced grammatically-plausible but incoherent output

**Symptom:** the vision pipeline ran end to end (patch injection, prefill, decode all completed
without exception), but `/v1/vision/chat` returned word-salad, real English words in implausible
order, no coherent description. llama.cpp against the same GGUF and photo produced a correct,
coherent caption.

**Root cause:** `Phi2TransformerHandler` rotated Q/K with adjacent-pair RoPE (`(x[2i],
x[2i+1])`, the LLaMA/original-rope convention), but Phi-2's GGUF tensor layout requires GPT-NeoX
split-half pairing (`(x[i], x[i+ropeDim/2])`, HF `rotate_half` convention), the same distinction
already identified and fixed for `Phi3TransformerHandler` separately. Because RoPE runs on every
position, both the 729 image-patch tokens and the text tokens, the wrong pairing corrupted
positional information for the whole sequence: per-token processing stayed intact enough to emit
real vocabulary, but attention no longer attended coherently across positions.

**Fix:** new `Phi2Rope` class implementing NeoX split-half partial rotary.
`Phi2TransformerHandler` calls `Phi2Rope.ropePartial` instead of its own adjacent-pair variant.
Test: `Phi2RopeTest` covers position-0 identity, pass-through of dimensions beyond `ropeDim`,
norm preservation, and an explicit split-half-vs-adjacent-pair worked example.

### FIXED: coherent output, wrong visual content, due to CLIP normalization constants on a SigLIP encoder

**Symptom:** after the RoPE fix, output was grammatical and on-topic-sounding but described
content not in the image, "a wooden stick and brush" for a photo of a family in front of a lit
Christmas tree.

**Root cause:** `ImagePatchEmbedder` hardcoded OpenAI CLIP pixel normalisation constants for
every model. moondream2's encoder is SigLIP, trained with `image_mean=image_std=[0.5,0.5,0.5]`.
Every pixel handed to the encoder was on the wrong scale, so patch embeddings were computed from
systematically mis-normalised input.

**Fix:** `VisionConfig` gained `imageMean`/`imageStd`, resolved with the same priority order
llama.cpp's own `clip.cpp` uses: prefer the GGUF's own `clip.vision.image_mean` /
`clip.vision.image_std` metadata when declared, else default by encoder family via CLS-token
detection. `ImagePatchEmbedder` normalises with the resolved per-model values. Tests:
`GgufReaderMetaFloatArrayTest`, `VisionConfigNormalizationTest`.

### FIXED: patch embeddings never went through the SigLIP post-encoder LayerNorm

**Symptom:** after the normalisation fix, `VisionEncoder`'s own diagnostic log got worse, not
better (per-patch L2 norm mean rose from 7117 to 9272). Output stayed coherent but vague and
generic, the model falling back on its own priors because the vision signal carried little
usable information.

**Root cause:** this encoder never applied a post-encoder LayerNorm. That is correct for
CLIP/LLaVA, CLIP's `post_layernorm` applies only to the pooled CLS output, which LLaVA-style
callers never touch. It is not correct for SigLIP: HF's `SiglipVisionTransformer.forward`
applies `post_layernorm` to the full `last_hidden_state`, every patch, before any downstream use
(confirmed against llama.cpp's own moondream-support PR, ggml-org/llama.cpp#6899). Without it,
patch vectors handed to the projector were the raw final-transformer-block residual stream,
unbounded in magnitude, starving the model of usable visual signal.

**Fix:** `VisionEncoder` now loads `v.post_ln.weight`/`v.post_ln.bias` as a genuinely optional
pair, unlike `v.pre_ln`'s identity-affine default: absence means the step is skipped entirely,
not run with trivial weight=1/bias=0, so CLIP files see zero behavior change. When present,
applies LayerNorm to the full sequence after the last transformer block and before the
projector. Test: `VisionEncoderTest`'s `layer_norm_collapses_unnormalised_transformer_output`
regression case, using the actual magnitudes observed in production.

## LLaVA-1.5 / llava-phi-3-mini (LLaMA-2 or Phi-3 + CLIP)

This family's investigation is longer and, as of the latest session, still partly open. Fixes
below are grouped by what they addressed; each is FIXED and shipped unless marked OPEN.

### FIXED: vision detection missed every real two-file release

`isVisionArchitecture()` originally only probed `--model-path` for `v.patch_embd.weight`. Every
known public LLaVA/Qwen-VL/SmolVLM/MiniCPM-V GGUF release ships the vision encoder in a separate
`mmproj-*.gguf` file, so every real downloaded model was classified text-only and
`/v1/vision/chat` was never registered. Fixed by adding the `--mmproj-path` flag and the
two-argument `isVisionArchitecture(Path, Path)` / `buildFromHandlers(...)` overloads described in
[Chapter 12.2](#ch-12-2). Test: `VisionModelPathsTest`.

### FIXED: two stacked wiring bugs made vision requests hang, traced via JFR thread dumps

Symptom: `POST /v1/vision/chat` never returned, looking like an infinite loop from the outside.
A JFR recording of an actual hung request, read via `jdk.jfr.consumer.RecordingFile` for its
embedded `jdk.ThreadDump` events, showed the request thread executing straight through
`LlamaTransformerHandler` with no `VisionAwareForwardPassHandler` frame anywhere in the stack.

**Bug 1 -- vision handler wrap discarded.** The handler-wrapping call originally ran *after*
`LocalInferencePipeline.from()`, which snapshots the handler list once, at construction, and
never re-reads it. Wrapping afterward had no effect: every request silently ran through the
plain-text handler, treating 576 `<image>` placeholder tokens as ordinary vocabulary and paying
for a full-width prefill with no chance of ever describing the image. Fixed by splitting the
wiring into `prepareVisionHandler()` (now called before `LocalInferencePipeline.from()`) and
`registerVisionRoutes()` (unchanged timing); see the sequence diagram in
[Chapter 12.2](#ch-12-2).

**Bug 2 -- wrong-flag branch, latent until Bug 1 was fixed.** Both the single-token and batched
forward paths in `LlamaTransformerHandler` decided whether to read token ids or activations
using the handler's own fixed `hasEmbeddings` field instead of asking the request what it
actually carried. Fixed by checking `hasEmbeddings && request.isFirstNode()` at both call sites.
Regression tests: `LocalInferencePipelineTest` (handler-list snapshot timing),
`LlamaTransformerHandlerEmbeddingsNodeActivationsTest` (activations-based requests on the
embeddings node).

### FIXED: text-token positions carried a zero vector instead of a real embedding

Once the wiring above was confirmed working end to end, output was grammatically coherent but
semantically nonsense. `VisionAwareForwardPassHandler`'s window-building methods spliced real
patch vectors into image-token positions but left every text-token position, the chat template,
the actual question, BOS, as an all-zero vector: only the image patches carried any signal.

**Fix:** new `ForwardPassHandler.embedToken(int)` default method, implemented in
`LlamaTransformerHandler` by reusing its existing embedding-table lookup and clamping logic. Both
vision-splicing methods now call `textHandler.embedToken(tokenId)` for non-image positions.
Confirmed fixed: rerunning the same request produced fluent, on-topic English for the first time.

### OPEN, then reverted: the LLaVA `mlp2x_gelu` projector's second layer

With text tokens fixed, output was fluent English describing the wrong scene. LLaVA-1.5's real
`mm_projector_type` is `mlp2x_gelu`: `mm.0` then GELU then `mm.2`, a second linear layer;
llama.cpp mmproj files name it `mm.2` because `mm.1` is the implicit, weight-less GELU. The code
originally applied only `mm.0`, a shape-valid but semantically arbitrary half of the learned
projector.

Applying the full chain (`mm.0` -> GELU -> `mm.2`) was tried and made output measurably worse: a
fully degenerate repeating `<image>`-token loop, not an improvement. The change was reverted.
The current, shipped behavior distinguishes by shape rather than by architecture name:

- **Non-square `mm.2`** (`mm0OutDim != mm2OutDim`), moondream2's case, `mm.0 [1152->8192] + GELU
  + mm.2 [8192->2048]`, is structurally required, without it the patch vectors are the wrong
  width for the LLM, and is applied.
- **Square `mm.2`** (`mm0OutDim == mm2OutDim`), LLaVA-1.5's case, is loaded and shape-validated
  at load time for diagnostics, but deliberately not applied in `VisionEncoder.project()`. The
  exact numerical root cause of the degeneration is not understood; re-enabling it without new
  evidence, specifically dumping actual intermediate min/max/mean/NaN counts immediately after
  `mm.0`+GELU and again after `mm.2` on a real request, is not recommended.

### FIXED: wrong GELU variant applied at every transformer block

`clip.use_gelu = false` in the mmproj metadata means "use `quick_gelu`
(`x * sigmoid(1.702x)`)", but `VisionEncoder.mlp()` called standard `gelu()` unconditionally,
this metadata key was never read anywhere before this fix. Neither activation crashes or
produces NaNs, so this silently distorted every patch's features in a way consistent with the
"coherent but semantically wrong" failure mode being chased at the time. Fixed via
`VisionConfig.useGelu` (default `true` for files that omit the key, preserving prior behavior)
and a new `activation()` dispatcher in `VisionEncoder.mlp()`. Test: direct `quickGelu()` unit
tests including an explicit "differs from standard gelu" case.

### FIXED: two crashes traced to unreliable GGUF metadata, resolved from tensor shapes instead

Two separate mid-request crashes against a real llava-phi-3-mini mmproj file:

- `ArrayIndexOutOfBoundsException` in `VisionEncoder.mlp()`: the code trusted tensor names
  `ffn_up`/`ffn_down` to imply direction, but this particular file's `ffn_up.bias` was shaped for
  the contraction direction, the naming convention is not universally consistent across mmproj
  exports in the wild. Fixed by `resolveFfnOrientation()`, a pure function that determines each
  FFN layer's real direction from its own GGUF-declared output dimension rather than its name,
  throwing a clear `IllegalStateException` at load time if neither orientation matches.
- `IllegalArgumentException: A.length=3145728 != rows*cols=786432` in the projector step: the
  code read `clip.vision.projection_dim` metadata (768) to size the `mm.0.weight` matmul, but the
  tensor's own shape is `[1024, 3072]`, 3072 being the LLM's real hidden dimension, the width the
  projector must actually produce. `clip.vision.projection_dim` is not reliable across mmproj
  exports. Fixed by `resolveProjectorOutputDim()`, the same shape-over-metadata approach, used
  everywhere a caller needs the projector's real output width.

Both fixes established the working principle for this codebase: GGUF metadata fields describing
architecture cannot be trusted at face value for mmproj files in the wild, only the tensors' own
declared shapes can.

### FIXED: EXIF orientation was never applied to real photographs

**Symptom:** every real-photo test could have been silently feeding the model a sideways or
upside-down image; a synthetic test image (a pixel-art logo) essentially never carries EXIF
orientation data, so this confound went unnoticed for several sessions of investigating an
unrelated "wrong content" symptom.

**Root cause:** `ImagePatchEmbedder` called `ImageIO.read()` directly. `ImageIO.read()` does not
apply EXIF orientation correction, it returns raw sensor-orientation pixels as-is. Phone and
camera photos routinely carry an EXIF `Orientation` tag (portrait shots are commonly stored as
landscape pixel data plus `orientation=6`).

**Fix:** a self-contained JPEG APP1/Exif segment parser and the standard 8-case EXIF orientation
transform, applied before resize and normalise. Fails safe: any parse issue defaults to
orientation=1 (no-op), never applies a wrong rotation. Tests cover the byte-level parser
(no-EXIF JPEG, non-JPEG, garbage bytes, synthetic little/big-endian TIFF segments, all 8
orientation values) and the transform (dimension swap/preserve per orientation family, a
180-degrees-twice-is-identity invariant, corner-pixel placement).

### OPEN: image content still wrong on out-of-distribution imagery

As of the latest recorded session, the EXIF fix above has not yet been confirmed by a rerun. The
underlying "correct grammar, wrong visual content" symptom on a pixel-art test logo has two live,
undistinguished hypotheses:

1. **Scale/magnitude mismatch** between the projector's output and a real LLM token embedding's
   magnitude, which could let the residual stream drown out the (much smaller or much larger)
   image signal.
2. **Model/architecture limitation, not a bug.** LLaVA-1.5-7B is documented to perform poorly on
   non-photographic imagery, pixel art, UI screenshots, diagrams, since its training distribution
   is dominated by natural photographs.

Diagnostic-only logging (min/max/mean/std and per-patch L2 norm for both projected patch
embeddings and a sample real text-token embedding, for direct comparison) was added with zero
behavior change, to disambiguate on the next run. The suggested next step to separate the two
hypotheses independent of that log is to run the same request against an actual photograph
instead of pixel art: a correct description would point at hypothesis 2 (capability limit, not
worth chasing further pipeline changes), while a still-wrong description would confirm a
remaining pipeline bug.

### FIXED (adjacent, discovered while live-testing vision fixes above)

Three unrelated bugs surfaced while manually exercising the fixes in this chapter and are fixed:

- **`--verbose` was a no-op in `--local` mode**, since logging setup ran in a static initializer
  before `parseArgs()` had a chance to read the flag. Fixed by moving setup into an explicit
  `configureLogging()` call after argument parsing.
- **`--dtype` silently accepted invalid values** (for example `INT4`) and coerced them to
  `FLOAT32` with no feedback. Fixed with an explicit `FLOAT32`/`F32`/`FP32` case and a `WARNING`
  on any other unrecognized value.
- **Any model-name mismatch was a hard 503**, even with exactly one model loaded and the mismatch
  being an obvious typo. Fixed by the shared `ModelIdResolver` and its `FallbackPolicy`
  described in [Chapter 12.3](#ch-12-3).

## Tracing method for future vision hangs

This environment's JRE ships the `jdk.jfr` module but no `jfr` CLI tool. A single-file-source
launch (`java SomeTool.java recording.jfr`) using `jdk.jfr.consumer.RecordingFile` to iterate
`RecordedEvent`s, filtering on `jdk.ThreadDump` for stack traces, or on the project's own
`juno.*` events (`juno.ForwardPass`, `juno.MatVec`, `juno.LoraTrainStep`) for progress counters,
is enough to distinguish "genuinely stuck" (identical stack across dumps, thread parked or
blocked) from "just very slow" (stack advances, counters keep climbing) without attaching a
debugger or reproducing interactively. See [Chapter 7.1](#ch-7-1) for the general JFR event
catalog this technique reads from.

## See also

- [Chapter 12.2 -- Model Requirements and Loading](#ch-12-2)
- [Chapter 12.4 -- Architecture](#ch-12-4)
- [Chapter 7.1 -- JFR and Metrics](#ch-7-1)
- [Chapter 3.8 -- Diagnostics and Tracing](#ch-3-8)

---

[<- 12.4 Architecture](#ch-12-4) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [12.6 Performance ->](#ch-12-6)