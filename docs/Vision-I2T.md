# Image-to-Text (I2T) — Vision Language Models

Juno supports multimodal inference (image + text → text) through the `vision`
module.  Tested architectures: LLaVA-1.5 (LLaMA-2 backbone), LLaVA-1.6
(Mistral backbone).

---

## Model requirements

The model must be a GGUF file whose `general.architecture` is `llava`,
`llava-1.5`, or `llava-qwen2`, and must contain both the LLM weights and
the CLIP vision encoder weights in the same file (the standard llama.cpp
mmproj-merged format).

Compatible models from Hugging Face (run through `llama.cpp convert` to GGUF):

| Model | Architecture key |
|---|---|
| llava-v1.5-7b-Q4_K_M.gguf | `llava` |
| llava-v1.6-mistral-7b.Q4_K_M.gguf | `llava-1.5` |

---

## Loading

`ForwardPassHandlerLoader.load()` detects the `llava` architecture and
automatically wraps the LLaMA text handler with `VisionAwareForwardPassHandler`.
No code change is needed in `CoordinatorMain`.

```bash
# same launch command as text-only — the handler is selected from GGUF metadata
./juno serve --model /path/to/llava-v1.5-7b-Q4_K_M.gguf
```

Wire the `VisionChatHandler` into `InferenceApiServer` once the model is loaded:

```java
VisionConfig vCfg   = VisionConfig.from(ggufReader);
VisionEncoder enc   = VisionEncoder.load(modelPath, backend);
int imgTokenId      = Integer.getInteger("juno.vision.image_token_id", 32000);
var visionHandler   = new VisionAwareForwardPassHandler(textHandler, imgTokenId, vCfg.projectionDim());
var visionChatHndlr = new VisionChatHandler(scheduler, registry, enc, visionHandler);
server.withVisionHandler(visionChatHndlr);
```

Override the image token ID via system property when using a non-LLaVA model:

```bash
-Djuno.vision.image_token_id=32044    # Phi-3 Vision
```

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
  "model": "llava-v1.5-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Describe this image in detail."}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

Response (HTTP 200):

```json
{
  "id": "vizcmpl-...",
  "object": "vision.completion",
  "model": "llava-v1.5-7b",
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

### Error responses

| HTTP | code | Cause |
|---|---|---|
| 400 | `invalid_request` | Missing `image` part or unparseable JSON |
| 400 | `invalid_image` | ImageIO cannot decode the supplied bytes |
| 429 | `rate_limit_exceeded` | Scheduler queue full |
| 501 | `not_implemented` | No vision model loaded |
| 503 | `service_unavailable` | Requested model not loaded |

---

## curl example

```bash
curl -X POST http://localhost:8080/v1/vision/chat \
  -F "image=@/path/to/photo.jpg" \
  -F 'request={"model":"llava-v1.5-7b","messages":[{"role":"user","content":"What is in this image?"}],"max_tokens":256}'
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
    ImagePatchEmbedder.java           raw bytes → float[3*H*W] CHW tensor
    VisionEncoder.java                CLIP ViT forward pass (pure Java)
    VisionAwareForwardPassHandler.java  ForwardPassHandler decorator
    VisionInferenceRequest.java       request record with imageBytes field
    VisionChatHandler.java            Javalin route handler
  src/test/java/cab/ml/juno/vision/
    VisionConfigTest.java
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