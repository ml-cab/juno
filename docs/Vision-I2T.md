# Image-to-Text (I2T) — Vision Language Models

Juno supports multimodal inference (image + text → text) through the `vision`
module.  Tested architectures: LLaVA-1.5 (LLaMA-2 backbone), LLaVA-1.6
(Mistral backbone).

---

## Model requirements

Detection is based on the presence of the CLIP/SigLIP patch-embedding tensor
`v.patch_embd.weight`, not on `general.architecture` — LLaVA-family models
report `general.architecture=llama` (or `qwen2`, `phi3`, ...) because that is
the text backbone; checking the architecture string alone will never find a
vision model.

### Two-file models (LLaVA, Qwen-VL, SmolVLM, ...)

Most public llama.cpp-format multimodal releases ship the vision encoder in a
**separate** GGUF file, conventionally named `mmproj-*.gguf`. Pass it
explicitly:

```bash
./juno local --model-path ../models/llava-v1.5-7b-Q4_K_M.gguf \
             --mmproj-path ../models/mmproj-model-f16.gguf \
             --api-port 8081
```

| Model | Base file | mmproj file |
|---|---|---|
| llava-v1.5-7b | `llava-v1.5-7b-Q4_K_M.gguf` | `mmproj-model-f16.gguf` |
| llava-v1.6-mistral-7b | `llava-v1.6-mistral-7b.Q4_K_M.gguf` | `mmproj-model-f16.gguf` |

### Embedded-GGUF models (moondream2 llamafile)

Some llamafiles bundle **both** the LLM and the vision encoder as two separate
GGUF entries inside the same ZIP. moondream2 is the primary example: its
`.llamafile` contains a phi-2 text model (first entry) and a SigLIP vision
encoder (second entry).

Juno detects this automatically via `LlamafileGgufIndex`. No `--mmproj-path`
is needed — just point at the llamafile:

```bash
./juno local --model-path ../models/moondream2-q5_k.llamafile --api-port 8081
```

At startup you will see:

```
[vision] Found embedded vision GGUF inside llamafile: "mmproj.gguf"  dataOffset=...
[vision] isVisionArchitecture=true
```

The image token ID for moondream2 defaults to `32000` (same system property as
LLaVA). Override if needed:

```bash
-Djuno.vision.image_token_id=<ID>
```

Check the startup log line `[vision] isVisionArchitecture check` to confirm
which file was probed.

---

## Loading

`ConsoleMain.wireVisionRoutes()` (juno-player, `--local` mode) checks
`LlavaHandlerFactory.isVisionArchitecture(modelPath, mmprojPath)` once the
text pipeline has finished loading. If it returns true, it calls
`LlavaHandlerFactory.buildFromHandlers()`, which reads only the CLIP
encoder tensors from the resolved vision-weights file (the mmproj file when
`--mmproj-path` is given) and wraps the first loaded text handler in a
`VisionAwareForwardPassHandler`. The `/v1/vision/chat` and
`/v1/vision/chat/stream` routes are then registered on the same
`InferenceApiServer` used for text chat.

```bash
./juno local --model-path ../models/llava-v1.5-7b-Q4_K_M.gguf \
             --mmproj-path ../models/mmproj-model-f16.gguf \
             --nodes 1 --api-port 8081 --verbose
```

No code change is needed to use a new LLaVA-family checkpoint — only a
correct `--mmproj-path` pointing at that checkpoint's own mmproj file.
Mixing mmproj files across unrelated models fails with an embedding-dimension
mismatch when `VisionAwareForwardPassHandler` is constructed.

Override the image token ID via system property when using a non-LLaVA model:

```bash
-Djuno.vision.image_token_id=32044    # Phi-3 Vision
```

**Known limitation**: only `--local` mode wires vision routes today.
`runClusterRepl()` (`--cluster` mode) does not call `wireVisionRoutes()`, so
`/v1/vision/chat` is not registered when forking separate node JVMs — use
`--local --nodes 1` for vision models.

---

## API

### POST /v1/vision/chat — blocking

Request: `multipart/form-data` with two parts.

| Part | Type | Description |
|---|---|---|
| `image` | file | JPEG, PNG, GIF, or BMP |
| `request` | text/JSON | `VisionChatRequest` body (see below) |

`request` JSON schema:

```json
{
  "model": "llava-v1.5-7b-Q4_K_M.gguf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Describe this image in detail."}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

`"model"` must be the loaded GGUF's exact filename — in `--local` mode that
is `Path.of(modelPath).getFileName()`, printed at startup as
`Model 'X' registered as LOADED`. It is **not** a friendly display name and
is unrelated to the mmproj filename. Simplest: omit `"model"` entirely —
`--local` mode only ever loads one model, so it resolves unambiguously
without it. See "Model id resolution" below.

Response (HTTP 200):

```json
{
  "id": "vizcmpl-...",
  "object": "vision.completion",
  "model": "llava-v1.5-7b-Q4_K_M.gguf",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The image shows..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 598, "completion_tokens": 84, "total_tokens": 682},
  "x_juno_latency_ms": 4120
}
```

### POST /v1/vision/chat/stream — SSE streaming

Same multipart request format as blocking.  Response is a stream of
`text/event-stream` chunks, one token piece per chunk, terminated by
`data: [DONE]`.

### Model id resolution

`"model"` is resolved by the shared `cab.ml.juno.registry.ModelIdResolver`
using `FallbackPolicy.SINGLE_MODEL_FALLBACK` (also used by
`/v1/chat/completions`; the lower-level native `/v1/inference` API opts into
the stricter `FallbackPolicy.STRICT` instead, since it is typically driven by
generated clients rather than hand-typed `curl`):

- absent/blank → the loaded model (unambiguous with a single `--local` model)
- exact match → that model
- mismatch, exactly one model loaded → falls back to it, with a `WARNING`
  logged naming both the requested and actual id — check the server log if
  a response looks like it came from the wrong model
- mismatch, multiple models loaded → `503 service_unavailable`, listing the
  loaded ids

### Error responses

| HTTP | code | Cause |
|---|---|---|
| 400 | `invalid_request` | Missing `image` part or unparseable JSON |
| 400 | `invalid_image` | ImageIO cannot decode the supplied bytes |
| 429 | `rate_limit_exceeded` | Scheduler queue full |
| 501 | `not_implemented` | No vision model loaded |
| 503 | `service_unavailable` | No model loaded, or requested model name is ambiguous among several loaded models |

---

## curl example

```bash
curl -X POST http://localhost:8080/v1/vision/chat \
  -F "image=@/path/to/photo.jpg" \
  -F 'request={"messages":[{"role":"user","content":"What is in this image?"}],"max_tokens":256}'
```

`"model"` is omitted above — safe and unambiguous in `--local` mode. To be
explicit, use the loaded GGUF's exact filename:

```bash
curl -X POST http://localhost:8080/v1/vision/chat \
  -F "image=@/path/to/photo.jpg" \
  -F 'request={"model":"llava-v1.5-7b-Q4_K_M.gguf","messages":[{"role":"user","content":"What is in this image?"}],"max_tokens":256}'
```

---

## Architecture

```
POST /v1/vision/chat
        │
VisionChatHandler
  ├─ ImagePatchEmbedder.toPixelTensor()    decode + resize + CLIP normalise
  ├─ VisionEncoder.encode()               CLIP ViT forward pass → float[numPatches][projDim]
  ├─ registerVisionEmbeddings(requestId)  store patches keyed by requestId
  ├─ InferenceRequest.of()                text with <image>×numPatches placeholder tokens
  └─ RequestScheduler.submitAndWait()
          │
    GenerationLoop
          │  (for each image-token position during prefill)
    VisionAwareForwardPassHandler.forward()
          ├─ detects IMAGE_TOKEN_ID at last position
          ├─ replaces embedding lookup with patch[patchIdx]
          └─ delegates rest of layers to LlamaTransformerHandler
```

### Module layout

```
vision/
  src/main/java/cab/ml/juno/vision/
    VisionConfig.java                 GGUF metadata → encoder shape
    VisionModelPaths.java             resolves base-model vs mmproj file for vision tensors
    ImagePatchEmbedder.java           raw bytes → float[3*H*W] CHW tensor
    VisionEncoder.java                CLIP ViT forward pass (pure Java)
    VisionAwareForwardPassHandler.java  ForwardPassHandler decorator
    VisionInferenceRequest.java       request record with imageBytes field
    VisionChatHandler.java            Javalin route handler
  src/test/java/cab/ml/juno/vision/
    VisionConfigTest.java
    VisionModelPathsTest.java
    ImagePatchEmbedderTest.java
    VisionEncoderTest.java
    VisionAwareForwardPassHandlerTest.java
```

### Key constraints

- **No new dependencies**: image decoding uses `javax.imageio` (JDK built-in).
- **No GGUF write**: the vision module is read-only with respect to the model file.
- **Thread-safe**: `VisionEncoder` weights are immutable after load;
  `VisionAwareForwardPassHandler` uses `ConcurrentHashMap` keyed by requestId.
- **Memory**: patch embeddings are released immediately after
  `scheduler.submitAndWait()` returns via the `finally` block in `VisionChatHandler`.

---

## Running tests

```bash
mvn test -pl vision
```

No model file, no GPU, no network required.

---

## Known issues / fixes

### FIXED — moondream2 (phi2) produces grammatically-plausible but incoherent output

**Symptom:** vision pipeline loads and runs end to end (patch injection, prefill,
decode all complete without exception), but `/v1/vision/chat` returns word-salad,
e.g. real English words in implausible order, no coherent description of the image.
Confirmed on CPU, single node, `--local` mode. llama.cpp / llamafile against the
same `.gguf` and the same photo produces a correct, coherent caption.

**Root cause:** `Phi2TransformerHandler` rotated Q/K with adjacent-pair RoPE
(`(x[2i], x[2i+1])`, the LLaMA/original-rope convention), but Phi-2's GGUF
tensor layout requires GPT-NeoX split-half pairing (`(x[i], x[i+ropeDim/2])`,
HF `rotate_half` convention) — the same distinction already identified and
fixed for `Phi3TransformerHandler` on 2026-06-11 (see
`docs/phi3-inference-handoff.md`, section C). Because RoPE runs on every
position (both the 729 image-patch tokens and the text tokens), the wrong
pairing corrupts positional information for the whole sequence: per-token
processing (embedding lookup, LayerNorm, FFN, detokenization) stays intact
enough to emit real vocabulary, but attention no longer attends coherently
across positions, producing exactly this word-salad-with-real-words pattern.
Vision-specific stages (patch embedding, SigLIP encoder, projector, image
token injection) were verified correct and are not the cause.

**Fix:** new `Phi2Rope` class implementing NeoX split-half partial rotary
(frequency formula unchanged; only the dimension pairing differs).
`Phi2TransformerHandler` calls `Phi2Rope.ropePartial` instead of its own
adjacent-pair `ropePartial`.

**Files:** `Phi2Rope.java` (new), `Phi2TransformerHandler.java`.
**Test:** `Phi2RopeTest.java` (new) — pos=0 identity, pass-through of dims
beyond `ropeDim`, norm preservation, and an explicit split-half-vs-adjacent-pair
worked example.

### FIXED — moondream2 output coherent but visually wrong (after the RoPE fix)

**Symptom:** after fixing the RoPE pairing above, `/v1/vision/chat` returns
grammatical, on-topic-sounding English (a real sentence, correct EOS
stopping) but describing content not actually in the image, e.g. "a wooden
stick and brush" for a photo of a family in front of a lit Christmas tree.
llama.cpp / llamafile against the same GGUF and photo correctly describes the
scene (people, Christmas lights, a building).

**Root cause:** `ImagePatchEmbedder` hardcoded the OpenAI CLIP pixel
normalisation constants (`mean=[0.4815,0.4578,0.4082]`,
`std=[0.2686,0.2613,0.2758]`) for every model. moondream2's vision encoder is
SigLIP, and SigLIP is trained with `image_mean=image_std=[0.5,0.5,0.5]`
(HF `SiglipImageProcessor` defaults). Every pixel handed to the SigLIP
encoder was on the wrong scale, so `VisionEncoder` was computing patch
embeddings from systematically mis-normalised input: attention (already
fixed above) now works correctly, but it is now correctly attending to
wrong/distorted visual features.

**Fix:** `VisionConfig` gained `imageMean`/`imageStd`, resolved in
`VisionConfig.from(GgufReader)` with the same priority order llama.cpp's own
`clip.cpp` uses: prefer the GGUF's own `clip.vision.image_mean` /
`clip.vision.image_std` metadata arrays when the file declares them (new
`GgufReader.metaFloatArray`), else default by encoder family using the
existing CLS-token detection (`v.class_embd` present -> CLIP defaults;
absent -> SigLIP defaults). `ImagePatchEmbedder` now normalises with the
resolved per-model values instead of the hardcoded CLIP constants.

**Files:** `GgufReader.java` (new `metaFloatArray`), `VisionConfig.java`,
`ImagePatchEmbedder.java`.
**Tests:** `GgufReaderMetaFloatArrayTest.java` (new),
`VisionConfigNormalizationTest.java` (new) — covers the CLS-token default
selection both ways and explicit-metadata override taking priority.