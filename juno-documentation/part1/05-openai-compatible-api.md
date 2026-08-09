(ch-05)=
# 5. The OpenAI-Compatible REST API

Pass `--api-port N` to any `local` or cluster invocation (see [Chapter 4](#ch-04)) to start an
OpenAI wire-compatible REST server alongside the REPL. No changes are required to
`GenerationLoop`, the scheduler, or any node code — the API layer is a pure translation shim
above `RequestScheduler` (architecture in [Chapter 2](#ch-02)).

## Supported endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Blocking or SSE streaming completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/models/{model}` | Retrieve a single model |

## Quick verification

```bash
# Start local mode with API
./juno local --model-path /path/to/model.gguf --api-port 8080

# Blocking completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is Java?"}]
  }'

# Streaming completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'

# List models
curl http://localhost:8080/v1/models
```

## Request field mapping

| OpenAI field | Juno internal | Notes |
|---|---|---|
| `model` | `modelId` | First loaded model if omitted |
| `messages[].role` | `ChatMessage.role` | `system` / `user` / `assistant` |
| `messages[].content` | `ChatMessage.content` | Text only; image content not supported |
| `temperature` | `SamplingParams.temperature` | 0.0–2.0; default 0.7 |
| `top_p` | `SamplingParams.topP` | 0.0–1.0; default 0.9 |
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 200 |
| `max_tokens` | `SamplingParams.maxTokens` | Deprecated alias; `max_completion_tokens` takes precedence |
| `frequency_penalty` | `SamplingParams.repetitionPenalty` | Mapped: `1 + max(0, fp/2)` |
| `stream` | route selection | `false` → blocking JSON; `true` → SSE |
| `n` | — | Only `1` accepted; other values → HTTP 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` | — | Silently ignored for client compatibility |

**Juno request extensions** (namespaced under `x_juno_*` to avoid OpenAI field conflicts):

| Field | Type | Default | Description |
|---|---|---|---|
| `x_juno_priority` | string | `NORMAL` | Scheduler priority: `HIGH` / `NORMAL` / `LOW` |
| `x_juno_session_id` | string | — | Stable session ID; enables KV-cache reuse across turns |
| `x_juno_top_k` | integer | `50` | Top-K sampling cutoff (0 = disabled) |

## Multi-turn conversation with KV-cache reuse

```python
SESSION_ID = "sess-my-conversation-001"

def chat(messages):
    return client.chat.completions.create(
        model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        messages=messages,
        extra_body={"x_juno_session_id": SESSION_ID},
    ).choices[0].message.content

history = []
for user_input in ["My name is Alice.", "What is my name?"]:
    history.append({"role": "user", "content": user_input})
    reply = chat(history)
    history.append({"role": "assistant", "content": reply})
    print(reply)
```

## Error responses

Errors follow the OpenAI error envelope
(`{"error": {"message": ..., "type": ..., "code": ...}}`):

| HTTP | `code` | Cause |
|------|--------|-------|
| 400 | `invalid_request` | Missing/empty messages, `n` > 1, or invalid body |
| 503 | `service_unavailable` | No model loaded or model not ready |
| 429 | `rate_limit_exceeded` | Scheduler queue full; `Retry-After` header set |
| 500 | `internal_error` | Unexpected inference error |

The full OpenAPI 3.0 specification is at `api/src/main/resources/juno-api.yaml`.

## Additional JVM-local endpoints

Same server as above, Juno-native (non-OpenAI) shape:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/inference` | Blocking JSON completion (`InferenceApiServer` native shape) |
| `POST` | `/v1/inference/stream` | SSE stream; each `data:` line is JSON `{"token":"…","isComplete":false}` until terminal event |

For programmatic access to this same server from JVM code — rather than curl or an OpenAI SDK
— see `JunoHttpClient` in [Chapter 6](#ch-06).

---

[← Chapter 4: Running Modes](#ch-04) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 6: JVM Integration →](#ch-06)
