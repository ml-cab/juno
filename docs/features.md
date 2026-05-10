# LoRA training, inference, and merge

Train low-rank adapters in-process with `./juno lora`, persist checkpoints as `.lora`, apply them read-only at inference with `--lora-play`, or bake weights into a new GGUF using `./juno merge`. The base GGUF file is never modified during training; merge produces a standalone artifact for deployment without a sidecar adapter.

Operational detail, REPL commands, and hyperparameters are in [LoRA.md](../LoRA.md). Redistributing merged models may interact with base-model and adapter licenses; see [legal.md](../legal.md).

# JFR and metrics

Every launcher mode accepts `--jfr DURATION` to record Java Flight Recorder with custom events (`juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, tokenizer events, `juno.LoraTrainStep`). Coordinator and forked nodes each emit `.jfr` files in cluster runs; `MetricsMain.extractToJsonMerged()` merges them into `target/metrics/metrics.json`.

Aggregate throughput can be read from `juno.TokenProduced` spans without extra counters; see [arch.md](../arch.md). Publishable scenario tables and CPU/GPU comparisons are in [juno_test_matrix.html](../juno_test_matrix.html); extraction CLI remains in [howto.md](../howto.md) and [performance.md](../performance.md).

# GPU acceleration

CUDA 12.x drives matrix work through cuBLAS-backed paths (`CudaMatVec`) using ByteDeko/JavaCPP presets; weights upload as FP16 on GPU load with deterministic release on shard unload. Pass `--cpu` or `JUNO_USE_GPU=false` to force CPU quantised matmul; cluster coordinators stay CPU-only while each node JVM owns its GPU context.

Lifecycle and handler routing are described under GPU sections of [arch.md](../arch.md). CPU vs GPU throughput snapshots appear in [juno_test_matrix.html](../juno_test_matrix.html).

# Distributed inference

Juno splits transformer work across JVM processes connected by gRPC. **Pipeline parallel** assigns contiguous layer ranges per node so activations flow serially and pooled VRAM fits larger models; **tensor parallel** keeps full depth on each node with head or FFN slices and combines partial logits at the coordinator via star AllReduce (constraint: head count divisible by node count).

Use `./juno` with cluster defaults or explicit `--pType pipeline|tensor`; remote deployments pair **juno-master** (coordinator) with **juno-node** workers. Full diagrams, REST vs native routes, and KV wiring live in [arch.md](../arch.md). Command-line flags and smoke tests are in [howto.md](../howto.md).

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

# Performance reporting

The primary Juno performance artifact is the interactive HTML matrix **[juno_test_matrix.html](juno_test_matrix.html)** (model, CPU vs GPU scenarios, throughput and latency insights). Open it from a checkout in a browser; refresh or regenerate the file when harness inputs or hardware baselines change.

Measurements tie back to JFR custom events (especially `juno.TokenProduced`, `juno.MatVec`, `juno.ForwardPass`): extract `.jfr` snapshots with the metrics module as described in [howto.md](howto.md). Cluster runs merge per-JVM recordings via `cab.ml.juno.metrics.MetricsMain`.

# EU AI Act friendly 

Redistributing merged weights may trigger base-model and adapter license questions; Juno does not (yet) provide a legal determination of points highlighted in EU AI Act **[research](docs/EU-AI-Act-compliance.md)** as `Compliance Gaps`. Please wait for those gaps to be addressed, contact us [via email](mailto:dev@ml.cab?subject=Help%20Request) or implement the ones you are interested in yourself.