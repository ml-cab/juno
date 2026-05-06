# OpenAI-compatible REST API

Pass `--api-port N` to `local` or cluster modes to start Javalin on the coordinator with **`POST /v1/chat/completions`** (blocking or SSE), **`GET /v1/models`**, and **`GET /v1/models/{model}`** using the same JSON shapes as OpenAI; clients only change `base_url`. Optional Juno extensions include `x_juno_priority`, `x_juno_session_id`, and `x_juno_top_k`.

| Endpoint | OpenAI equivalent | Description |
|----------|-------------------|-------------|
| `POST /v1/chat/completions` | `POST /v1/chat/completions` | Blocking or SSE streaming completion |
| `GET /v1/models` | `GET /v1/models` | List loaded models |
| `GET /v1/models/{model}` | `GET /v1/models/{model}` | Single model metadata |

Optional extensions:

| Field | Type | Description |
|-------|------|-------------|
| `x_juno_priority` | string | `HIGH` / `NORMAL` / `LOW` |
| `x_juno_session_id` | string | Stable ID for KV-cache reuse |
| `x_juno_top_k` | integer | Top-K cutoff (0 = disabled; default 50) |

**Supported fields:** `model`, `messages`, `temperature`, `top_p`, `max_completion_tokens`, `max_tokens` (deprecated alias), `frequency_penalty`, `stream`, `n` (only 1 accepted). **Ignored for compatibility:** `stop`, `presence_penalty`, `logit_bias`, `user`, `seed`.

The coordinator still exposes Juno-native inference endpoints alongside this surface; behaviour is documented in [arch.md](../arch.md). The authoritative OpenAPI 3 spec is [`api/src/main/resources/juno-api.yaml`](../../api/src/main/resources/juno-api.yaml). Examples and flags are in [howto.md](../howto.md).
