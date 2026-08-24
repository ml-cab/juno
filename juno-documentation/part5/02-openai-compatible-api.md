(ch-5-2)=
# 5.2. OpenAI-Compatible API

Pass `--api-port N` to any `local` or cluster invocation to start an OpenAI wire-compatible REST
server alongside the REPL. No changes are required to `GenerationLoop`, the scheduler, or any
node code; the API layer is a pure translation shim above `RequestScheduler`. Any client that
speaks the OpenAI Chat Completions wire format works against Juno with only a base-URL change,
no prompt reformatting, no adapter library, no glue code.

**Supported endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Blocking or SSE streaming completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/models/{model}` | Retrieve a single model |

**Request flow**

```{mermaid}
sequenceDiagram
    participant Client as Client<br/>(OpenAI SDK / LangChain / curl)
    participant OAH as OpenAiChatHandler
    participant OAA as OpenAiAdapter<br/>(static mapping)
    participant Sched as RequestScheduler
    participant GL as GenerationLoop

    Client->>OAH: POST /v1/chat/completions<br/>{model, messages, temperature, stream, ...}
    OAH->>OAH: deserialise OaiChatCompletionRequest (Jackson)
    OAH->>OAH: validate n, messages
    OAH->>OAA: build InferenceRequest + SamplingParams
    OAA-->>OAH: InferenceRequest

    alt stream=false (blocking)
        OAH->>Sched: submitAndWait(request)
        Sched->>GL: run generation
        GL-->>Sched: GenerationResult
        Sched-->>OAH: GenerationResult
        OAH-->>Client: ChatCompletion JSON
    else stream=true (SSE)
        OAH->>Sched: submit(request, TokenConsumer)
        loop per token
            Sched->>GL: next token
            GL-->>OAH: token
            OAH-->>Client: data: {"choices":[{"delta":{"content":"..."}}]}
        end
        OAH-->>Client: data: [DONE]
    end
```

**Quick verification:**

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

**Request field mapping:**

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
| `stream` | route selection | `false` -> blocking JSON; `true` -> SSE |
| `n` | N/A | Only `1` accepted; other values -> HTTP 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` | N/A | Silently ignored for client compatibility |

**Juno request extensions** (namespaced under `x_juno_*` to avoid OpenAI field conflicts):

| Field | Type | Default | Description |
|---|---|---|---|
| `x_juno_priority` | string | `NORMAL` | Scheduler priority: `HIGH` / `NORMAL` / `LOW` |
| `x_juno_session_id` | string | none | Stable session ID; enables KV-cache reuse across turns |
| `x_juno_top_k` | integer | `50` | Top-K sampling cutoff (0 = disabled) |
| `x_juno_disclosure` | boolean | `true` | EU AI Act Article 50 opt-out. `true` includes `x_juno_ai_disclosure` in the response; `false` omits it. Set `false` only for API-to-API integrations with no human end-user present. See [Chapter 9.7](#ch-9-7) |

**Multi-turn conversation with KV-cache reuse:**

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

**Response extension:** every response (blocking, and the first SSE chunk of a streaming
response) includes `x_juno_ai_disclosure`, a short text notice, unless the request set
`x_juno_disclosure` to `false`. This satisfies the EU AI Act Article 50 transparency
obligation: natural persons must be notified they are interacting with an AI system. See
[Chapter 9.7 -- EU AI Act Compliance](#ch-9-7) for the full analysis.

## See also

- [Chapter 5.1 -- Juno Native API](#ch-5-1)
- [Chapter 5.3 -- Error Handling](#ch-5-3)
- [Chapter 5.4 -- OpenAPI Spec](#ch-5-4)
- [Chapter 9.7 -- EU AI Act Compliance](#ch-9-7)

---

[<- 5.1 Juno Native API](#ch-5-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [5.3 Error Handling ->](#ch-5-3)