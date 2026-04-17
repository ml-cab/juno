# Juno — Distributed Java LLM Inference Engine

**Full Architecture Reference** · JDK 25 · Maven · Java-native · Commodity GPU Cluster

---

## Table of Contents

1. Vision
2. Hardware Stack
3. Maven Project Structure
4. API Module
5. System Architecture
6. KV Cache
7. REST / HTTP
8. Actors — Design Decisions
9. Integration Test Infrastructure
10. Activation Compression
11. Full Token Generation Data Flow
12. Full Configuration Reference
13. Technology Summary
14. Build Status
15. Real Model Inference
16. GPU Acceleration Layer
17. Phi-3 Family Support
18. Tensor Parallel
19. LoRA Fine-Tuning
20. KV Cache Wiring + JFR Instrumentation
21. AWS Deployment
22. Metrics Module
23. LoRA Inference Overlay (`--lora-play`)
24. Changelog Summary

---

## 1. Vision

A fully Java-native distributed LLM inference engine — no Python, no GIL, real threads, commodity hardware over premium hardware.

---

## 2. Hardware Stack

**Compute Nodes (×16 old PCs)**

| Component | Spec |
|-----------|------|
| GPU | 4 GB VRAM each — 16×4 GB = 64 GB total |
| CPU | 8+ core |
| RAM | 16–32 GB |
| Storage | NVMe SSD |

**Networking:** 10 GbE / 25 GbE, managed switch, RDMA.

---

## 3. Maven Project Structure

```
juno/
├── api/
├── registry/
├── coordinator/
├── node/           ← ForwardPassHandlerLoader (LoraAdapterSet overload added)
│                      EmbeddedNodeServer (reads juno.lora.play.path)
│                      NodeMain (forwards JUNO_LORA_PLAY_PATH env var)
├── kvcache/
├── tokenizer/
├── sampler/
├── health/
├── player/         ← ConsoleMain (--lora-play flag, /train-qa command)
│                      ClusterHarness (withLoraPlay() method)
├── metrics/
├── juno-node/      ← Fat jar (NodeMain)
└── juno-master/    ← Fat jar (CoordinatorMain, reads JUNO_LORA_PLAY_PATH)
```

---

## 4. API Module

### 4.1 OpenAPI 3.0 REST Spec

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/inference` | Blocking inference |
| POST | `/v1/inference/stream` | SSE streaming |
| POST | `/v1/models` | Load model |
| GET | `/v1/cluster/health` | Cluster overview |
| GET | `/` | Web console |

### 4.2 gRPC Proto

`NodeService` — coordinator → each node (`ForwardPass`, `LoadShard`, `UnloadShard`, `GetNodeStatus`)

---

## 5. System Architecture

```
[Client]
    │ REST (Javalin) / gRPC streaming
    ▼
[Coordinator]
    ├── GgufTokenizer
    ├── ChatTemplateFormatter
    ├── RequestScheduler (virtual threads)
    ├── Sampler
    ├── KVCacheManager
    └── GenerationLoop
              │
              │ gRPC activations (FLOAT16 / INT8 / FLOAT32)
              │
    +--------------------------------------------+
    |  Node 1       Node 2       Node 3           |
    |  L 0-7        L 8-14       L 15-21          |
    |  + embed                   + output proj    |
    |  LoraAdapterSet (optional, read-only)       |
    +--------------------------------------------+
```

---

## 6. KV Cache

Two tiers: GPU VRAM (CUDA LRU) + JVM heap (Caffeine W-TinyLFU). Prefix cache Trie for shared system prompts.

---

## 7. REST / HTTP

Javalin 6.x, no Spring Boot. Metrics on port 9091 via Micrometer + Prometheus.

---

## 8. Actors — Design Decisions

### 8.5 Tokenizer

`GgufTokenizer` auto-detects SentencePiece BPE vs GPT-2 BPE from GGUF metadata.

**Chat template resolution order:** (1) exact key → (2) substring match longest-first → (3) `chatml` fallback.

Template keys: `llama3`, `mistral`, `gemma`, `chatml`, `tinyllama`/`zephyr`, `phi3`/`phi-3`.

> **Critical for LoRA:** training and inference must use the same template key. The `[TRACE] model type` line at LoRA REPL startup shows the detected key. Mismatch causes complete failure to recall trained facts even when loss converged.

---

## 9. Integration Test Infrastructure

`InProcessClusterIT` (6 tests) · `ThreeNodeClusterIT` (9 tests) · `ModelLiveRunnerIT` (8 checks, `-Pintegration`) · `TensorParallelClusterIT` (5 tests)

---

## 15. Real Model Inference

### ForwardPassHandlerLoader

```java
// Architecture routing + optional LoRA overlay
public static ForwardPassHandler load(
    Path modelPath, ShardContext context, MatVec backend,
    LoraAdapterSet adapters) throws IOException {

    if (adapters != null) {
        return LoraTrainableHandler.load(modelPath, context, adapters);
    }
    return switch (readArchitecture(modelPath)) {
        case "phi3" -> Phi3TransformerHandler.load(modelPath, context, backend);
        default     -> LlamaTransformerHandler.load(modelPath, context, backend);
    };
}
```

When `adapters != null`, the handler is always `LoraTrainableHandler` (implements `ForwardPassHandler`; inference-only, no optimizer attached). Passing `null` gives the standard dispatch.

`selectBackend()` promoted from package-private to `public` so player-module callers can invoke it directly.

---

## 19. LoRA Fine-Tuning

### Key Classes

| Class | Description |
|-------|-------------|
| `LoraAdapter` | Core math: forward delta, backward gradient |
| `LoraAdapterSet` | Keyed by (layerIndex, projectionName); `save()` / `load()` binary checkpoint |
| `LoraAdamOptimizer` | Adam with bias correction; weight decay on A only |
| `LoraTrainableHandler` | `ForwardPassHandler` for inference + `trainStep()` for training |
| `LoraTrainEvent` | JFR event `juno.LoraTrainStep`: step, loss, forward/backward/optimizer ms |

### ConsoleMain `/train-qa`

New REPL command that auto-generates 4 phrasings of a Q&A pair and trains with the correct chat template:

```
/train-qa What is my name? A: Dima
```

Generates: exact phrasing × 2 + `Can you tell me: ...` + `Please answer: ...`. The model type is detected from the model path and used to apply the matching chat template. `--lora-steps-qa N` controls steps per chunk; `--lora-early-stop F` stops early when loss delta falls below F.

### Diagnostic Tracing (`--verbose`)

| Trace line | What it shows |
|-----------|---------------|
| `[TRACE] model type (chat template key)` | Template detected at startup |
| `[TRACE] formatted training text (repr)` | Exact byte sequence sent to the model |
| `[TRACE] token count (excl. BOS)` | Number of tokens in the training sequence |
| `[TRACE] token IDs: [...]` | Raw token IDs (verbose only) |
| `[TRACE] step=N loss=F chunk=M/T ms=D` | Per-step loss during training |
| `[TRACE] inference model type` | Template at inference — must match training |

---

## 23. LoRA Inference Overlay (`--lora-play`)

### Overview

Pre-trained `.lora` adapter files can be applied read-only at inference time in any mode without entering the training REPL.

### Propagation chain

```
ConsoleMain
  --lora-play PATH
        │
        ├── local mode
        │     LoraAdapterSet.load(Path)
        │     ForwardPassHandlerLoader.load(model, ctx, backend, adapters)
        │     → LoraTrainableHandler (inference-only, no optimizer)
        │
        └── cluster mode
              ClusterHarness.withLoraPlay(path)
                    │
                    └── launchNode()
                          -Djuno.lora.play.path=PATH (per forked JVM)
                                │
                                └── EmbeddedNodeServer.NodeServiceImpl
                                      loraPlayPath = System.getProperty("juno.lora.play.path")
                                      loadShard() → LoraAdapterSet.load(Path.of(loraPlayPath))
                                      ForwardPassHandlerLoader.load(..., adapters)
```

### AWS deployment chain

```
juno-deploy.sh --lora-play /abs/path/model.lora
    │
    ├── realpath() → absolute path (prevents relative-path mismatches)
    ├── setup() validates [[ -f PATH ]] before launching any instances
    │
    └── _scp_lora_to_nodes() (called after bootstrap, before coordinator start)
          For each node — synchronously:
            1. scp  → /tmp/<basename>
            2. systemctl stop juno-node   (synchronous, waits for JVM exit)
            3. mv + chmod 644 /opt/juno/models/<basename>
            4. sed -i 'JUNO_LORA_PLAY_PATH=/opt/juno/models/<basename>' /etc/juno/node.env
            5. systemctl start juno-node  (synchronous, ~2s to bind gRPC)
            6. [TRACE] node.env patch confirmed in log
          Updates global LORA_PLAY_PATH to remote absolute path
    │
    └── _write_cluster_env_and_start_coordinator()
          JUNO_LORA_PLAY_PATH=/opt/juno/models/<basename> → cluster-nodes.env
          systemctl start juno-coordinator
```

The coordinator starts only after all nodes are confirmed active — loadShard RPCs always find nodes with adapters already loaded.

### Known gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| `lora=none` in node log | `juno.lora.play.path` property not set | `--lora-play` missing, or wrong path |
| Node log shows relative path | `realpath` not applied at parse time | Updated in Session 26 |
| Coordinator starts before nodes have new env | `--no-block` restart; old instance seen as `active` | Synchronous stop+start per node (Session 26) |
| `model load failed: ../models/...` | Relative path baked into `cluster-nodes.env` | `LORA_PLAY_PATH` updated to remote absolute path before coordinator starts (Session 26) |
| cloud-init skips bootstrap script | Double base64 encoding via pre-encoded `--user-data` | Pass `file://` path to AWS CLI (Session 26) |
| cloud-init sees ANSI codes before `#!/bin/bash` | `log()` calls in `_build_node_userdata` wrote to stdout | Redirect to stderr with `>&2` (Session 26) |

---

## 21. AWS Deployment

### `juno-deploy.sh` Setup Options (updated)

| Flag | Default | Description |
|------|---------|-------------|
| `--instance-type` | `g4dn.xlarge` | EC2 instance type |
| `--node-count` | 3 | Inference node count |
| `--coordinator` | `node1` | `node1` or `separate` |
| `--model-url` | — | HuggingFace GGUF URL |
| `--ptype` | `pipeline` | `pipeline\|tensor` |
| `--dtype` | `FLOAT16` | `FLOAT32\|FLOAT16` |
| `--jfr` | — | JFR duration (e.g. `5m`) |
| `--lora-play` | — | Local `.lora` file path (resolved to absolute via `realpath`) |

### Hardening changes (Session 26)

- **User-data passed via `file://`** — prevents double base64 encoding. CLI size trace now shows `first-line: #!/bin/bash`.
- **TRACE logs in `_build_node_userdata` redirect to stderr** — prevents contamination of captured user-data script.
- **`realpath` at `--lora-play` parse time** — relative paths are resolved immediately.
- **Early `[[ -f ]]` validation** in `setup()` — fails before any AWS spend.
- **Synchronous stop/patch/start** in `_scp_lora_to_nodes` — eliminates race condition with coordinator.

---

## 22. Metrics Module

Five JFR event types:

| Event | Key Fields |
|-------|------------|
| `juno.MatVec` | `backend`, `rows`, `cols` |
| `juno.ForwardPass` | `handlerType` (`lora` when adapter applied), `requestId`, `startPosition` |
| `juno.Tokenizer` | `tokenizerType`, `operation`, `inputLength`, `outputLength` |
| `juno.TemplateFormat` | `modelType`, `messageCount`, `outputLength` |
| `juno.LoraTrainStep` | `step`, `loss`, `forwardMs`, `backwardMs`, `optimizerMs` |

`handlerType = "lora"` in `juno.ForwardPass` events confirms the adapter is active during inference.

---

## 24. Changelog Summary

| Session | Key Changes |
|---------|-------------|
| 1–3 | Foundation: sampler, tokenizer, registry, kvcache, coordinator stubs |
| 4 | Real model: GgufReader, LlamaConfig, LlamaTransformerHandler, GgufTokenizer |
| 5–6 | Bug fixes, parallel matVec (9× speedup), FLOAT16 |
| 7–9 | Player module, run.sh, session KV cache |
| 10 | GPU acceleration: CudaMatVec, GpuContext |
| 11 | Phi-3 family: Phi3TransformerHandler, ForwardPassHandlerLoader |
| 12 | Pure rename refactor |
| 13 | Tensor parallel: TensorShardPlanner, TensorParallelPipelineClient |
| 14 | LoRA fine-tuning: LoraAdapter, LoraAdamOptimizer, LoraTrainableHandler, /train, /save |
| 15 | KV cache wired: NodeKVCacheAdapter; JFR instrumentation |
| 16 | Naming cleanup |
| 17–19 | AWS scripts, GPT-2 BPE, metrics module |
| 20 | GPU inference end-to-end wired |
| 21 | juno-deploy.sh unified lifecycle script, web console |
| 22 | Q2_K + Q3_K quantisation |
| 23 | Programmatic JFR lifecycle, cluster JFR merge |
| 24 | Configurable byte order (BE/LE) |
| 25 | Code quality, dead code removed, docs |
| 26 | **`--lora-play` inference overlay** across local/cluster/AWS; **`/train-qa`** Q&A training with chat template formatting and 4-phrasing auto-generation; **diagnostic tracing** (`[TRACE]` in lora REPL, model type, formatted text, per-step loss, inference template key); **AWS deploy hardening** (double-base64 fix via `file://`, stderr redirect for log calls, `realpath` for `--lora-play`, synchronous stop/patch/start in `_scp_lora_to_nodes`, early file validation, missing `}` brace fix) |