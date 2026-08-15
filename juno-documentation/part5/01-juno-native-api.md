(ch-5-1)=
# 5.1. Juno Native API

`InferenceApiServer` (Javalin) is the single HTTP entry point on the coordinator. It exposes the
Juno-native surface below alongside the [OpenAI-compatible API](#ch-5-2). Both
surfaces share the same underlying `RequestScheduler` and `GenerationLoop`, so behavior and
performance are identical regardless of which surface a client uses.

## Native inference endpoints

| Method | Path | Handler |
|--------|------|---------|
| `POST` | `/v1/inference` | `handleBlockingInference`: blocking, returns `GenerationResult` |
| `POST` | `/v1/inference/stream` | `handleStreamingInference`: SSE, one event per token |
| `GET` | `/v1/models` | `OpenAiChatHandler.handleListModels` |
| `GET` | `/v1/models/{modelId}` | `OpenAiChatHandler.handleGetModel` |
| `DELETE` | `/v1/models/{modelId}` | `handleUnloadModel` |
| `GET` | `/v1/cluster/health` | `handleClusterHealth`: per-node health rollup |

## Health and console

| Method | Path | Handler |
|--------|------|---------|
| `GET` | `/` | `handleConsole`: embedded coordinator web console |
| `GET` | `/health-ui` | `handleHealthDashboard`: node health dashboard HTML |
| `POST` | `/health/probe` | `handleHealthProbeProxy`: proxies probe to `HealthReporter` |
| `GET` | `/health-data` | `handleHealthDataProxy`: proxies health JSON from nodes |

## See also

- [Chapter 5.2 -- OpenAI-Compatible API](#ch-5-2)
- [Chapter 5.3 -- Error Handling](#ch-5-3)
- [Chapter 5.4 -- OpenAPI Spec](#ch-5-4)

---

[<- 4.8 Testing Checklist](#ch-4-8) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [5.2 OpenAI-Compatible API ->](#ch-5-2)