(ch-12-3)=
# 12.3. REST API

## POST /v1/vision/chat -- blocking

Request: `multipart/form-data` with two parts.

| Part | Type | Description |
|---|---|---|
| `image` | file | JPEG, PNG, GIF, or BMP |
| `request` | text/JSON | `VisionChatRequest` body, see below |

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

`"model"` must be the loaded GGUF's exact filename, in `--local` mode that is
`Path.of(modelPath).getFileName()`, printed at startup as `Model 'X' registered as LOADED`. It
is not a friendly display name and is unrelated to the mmproj filename. Simplest: omit `"model"`
entirely, `--local` mode only ever loads one model, so it resolves unambiguously without it. See
"Model id resolution" below.

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

## POST /v1/vision/chat/stream -- SSE streaming

Same multipart request format as blocking. Response is a stream of `text/event-stream` chunks,
one token piece per chunk, terminated by `data: [DONE]`.

## Model id resolution

`"model"` is resolved by the shared `cab.ml.juno.registry.ModelIdResolver` using
`FallbackPolicy.SINGLE_MODEL_FALLBACK`, the same policy `POST /v1/chat/completions` uses; the
lower-level native `/v1/inference` API opts into the stricter `FallbackPolicy.STRICT` instead,
since it is typically driven by generated clients rather than hand-typed `curl`. See
[Chapter 5.1](#ch-5-1) for the full resolver contract.

- absent/blank -> the loaded model, unambiguous with a single `--local` model
- exact match -> that model
- mismatch, exactly one model loaded -> falls back to it, with a `WARNING` logged naming both
  the requested and actual id, check the server log if a response looks like it came from the
  wrong model
- mismatch, multiple models loaded -> `503 service_unavailable`, listing the loaded ids

## Error responses

| HTTP | code | Cause |
|---|---|---|
| 400 | `invalid_request` | Missing `request` or `image` form part, or unparseable `request` JSON |
| 400 | `invalid_image` | `ImageIO` cannot decode the supplied bytes |
| 429 | `rate_limit_exceeded` | Scheduler queue full |
| 503 | `service_unavailable` | No model loaded, or the requested model name is ambiguous among several loaded models |

If no vision model was ever loaded for this process, `/v1/vision/chat` was never registered on
`InferenceApiServer` in the first place; see [Chapter 12.2](#ch-12-2). A request against that
path returns Javalin's default 404, not one of the JSON error envelopes above, since the route
simply does not exist for that run.

## curl example

```bash
curl -X POST http://localhost:8080/v1/vision/chat \
  -F "image=@/path/to/photo.jpg" \
  -F 'request={"messages":[{"role":"user","content":"What is in this image?"}],"max_tokens":256}'
```

`"model"` is omitted above, safe and unambiguous in `--local` mode. To be explicit, use the
loaded GGUF's exact filename:

```bash
curl -X POST http://localhost:8080/v1/vision/chat \
  -F "image=@/path/to/photo.jpg" \
  -F 'request={"model":"llava-v1.5-7b-Q4_K_M.gguf","messages":[{"role":"user","content":"What is in this image?"}],"max_tokens":256}'
```

## See also

- [Chapter 5.1 -- Juno Native API](#ch-5-1)
- [Chapter 5.2 -- OpenAI-Compatible API](#ch-5-2)
- [Chapter 5.3 -- Error Handling](#ch-5-3)
- [Chapter 12.2 -- Model Requirements and Loading](#ch-12-2)

---

[<- 12.2 Model Requirements and Loading](#ch-12-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [12.4 Architecture ->](#ch-12-4)