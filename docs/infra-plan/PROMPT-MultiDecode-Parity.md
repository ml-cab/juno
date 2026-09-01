# Agent prompt: multi-request decode batching parity (all handlers)

Copy everything below the line into a new agent session.

---

## Task

Implement **`forwardMultiDecode` parity** across all supported GGUF handler families so `--parallel N` static batching gets real batched decode (not serial `ForwardPassHandler` default) on **Phi-2, Phi-3, Qwen3, and Qwen3 MoE**, matching what already ships for **Llama-family** models.

This closes the gap left by Tier 1 multi-request decode work (Llama-only) and satisfies **ROADMAP Execution rules §5** ([`PLAN-Infra-SUPPORTED-MODELS.md`](PLAN-Infra-SUPPORTED-MODELS.md)).

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — tests first, KISS, prefer new classes, list changed files (no zip)
2. [`docs/infra-plan/PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1–§5
3. [`docs/infra-plan/PLAN-Infra-SUPPORTED-MODELS.md`](PLAN-Infra-SUPPORTED-MODELS.md)
4. [`docs/infra-plan/PLAN-Infra-Tier1.md`](PLAN-Infra-Tier1.md) — static batch / `--parallel` scope
5. [`docs/perf-compare/README.md`](../perf-compare/README.md) — multi-session section; baseline run [`20260901T173121Z-parallel`](../perf-compare/20260901T173121Z-parallel/) (Llama GPU 1.11×)

### Reference implementation (Llama — done)

| Piece | Location |
|-------|----------|
| Request/result types | `node/.../MultiDecodeForwardRequest.java`, `MultiDecodeForwardResult.java` |
| Handler default (serial) | `ForwardPassHandler.forwardMultiDecode()` |
| Pipeline routing | `LocalInferencePipeline.forwardBatch()` → `forwardMultiDecode()` chain |
| Llama batched decode | `LlamaTransformerHandler.forwardMultiDecode()`, `runLayersMultiDecode()`, `transformerLayerMultiDecode()`, `outputProjectionBatch()` |
| GPU batched GEMV | `CudaMatVec.sgemm(DeviceHalfMatrix/DeviceFloatMatrix)` — batch ≤ 8, `strideA=0` for shared weights |
| Parity test pattern | `node/.../LlamaTransformerHandlerMultiDecodeTest.java` |
| Pipeline call-count test | `node/.../LocalInferencePipelineForwardBatchTest.java` |

### Current handler state

| Handler | Prefill `forwardBatch` | Multi-request `forwardMultiDecode` |
|---------|------------------------|-------------------------------------|
| `LlamaTransformerHandler` | Yes (`runLayersBatch`) | **Yes** |
| `Phi2TransformerHandler` | Yes (`runLayersBatch`) | **No** (serial default) |
| `Phi3TransformerHandler` | Yes (`runLayersBatch`, fused QKV) | **No** |
| `Qwen3TransformerHandler` | **No** (serial default) | **No** |
| `Qwen3MoeTransformerHandler` | **No** | **No** |
| `LoraTrainableHandler` | Yes (Llama-style) | **No** (optional follow-up) |

`GenerationLoop.generateBatch()` already calls `pipeline.forwardBatch()` once per decode step — no coordinator changes expected unless tests reveal gaps.

## Design constraints

- **Static batching only** — N requests, each at seq len 1, independent KV positions. Not continuous batching (Tiers 14–16).
- **Distinct from `BatchForwardRequest`** — that batches W contiguous prefill tokens for **one** `requestId`; multi-decode batches **N requestIds** at one position each.
- **Streaming/SSE** — still bypassed via `TokenConsumer.batchEligible()`; do not change.
- **GPU** — reuse `CudaMatVec` / `GpuBlasOps` batched paths; `GpuContext.cublasSerializationLock()` still serializes — batch to reduce kernel count, not assume lock-free multi-stream.
- **No infra tier numbers** in code comments, JFR, CLI help, or user-facing docs (`.cursor/rules/juno-no-infra-tier-labels.mdc`).
- **KISS** — prefer new small helper classes over growing handlers further if a pattern repeats across Phi/Qwen.

## Suggested approach

1. **Phi-3 first** (has `runLayersBatch` / `transformerLayerBatch` — closest to Llama; bake-off model in default compare set).
2. **Phi-2** — parallel-attn+FFN, LayerNorm, GELU; mirror Phi-3 pattern from its existing `runLayersBatch`.
3. **Qwen3 dense** — add `forwardBatch` if missing, then `forwardMultiDecode` (attention norm / QKV layout differs from Llama).
4. **Qwen3 MoE** — MoE layer in batch path; may need per-row expert routing (attention batched, MoE per token — document if MoE stays serial inside batched layer).

Per handler:

- Implement `forwardMultiDecode(MultiDecodeForwardRequest, ShardContext)` analogous to Llama.
- Add `runLayersMultiDecode` + `transformerLayerMultiDecode` (or arch-specific names) — batched linear ops, per-row RoPE/attention/KV at `positions[b]` and `requestIds.get(b)`.
- Batched LM head where applicable (`outputProjectionBatch` pattern).

Optional: extract shared multi-decode scaffolding if duplication exceeds ~100 lines across handlers (e.g. KV setup loop) — only if it stays simpler than copy-paste.

## Tests (required per §5)

For **each** handler family you touch:

1. **Parity:** `forwardMultiDecode` logits match N serial `forward()` calls within tolerance (mirror `LlamaTransformerHandlerMultiDecodeTest` — use `newTestInstance` or existing synthetic fixtures).
2. **KV isolation:** independent `requestId` caches updated correctly across batched steps.
3. Do **not** add trivial tests.

Existing tests to extend or use as patterns:

- `Phi3TransformerHandlerTest`, `Phi3GreedyDecodeIntegrationTest`
- `Qwen3GreedyDecodeIntegrationTest`, `Qwen3MoeGreedyDecodeIntegrationTest`
- `coordinator/.../GenerationLoopBatchTest` (pipeline-level; should stay green)

## Verification

```bash
mvn test -pl coordinator,node,juno-player,vision -am
```

**Perf** (document per architecture in `docs/perf-compare/README.md`):

```bash
# Llama baseline already green
./scripts/performance-tests/compare-parallel.sh --gpu --sessions 8 --max-tokens 64

# After Phi-3 parity — run with Phi-3 model or add --model flag if script supports it
# Minimum: confirm Phi-3.5-mini loads and completes 8 concurrent non-stream requests with --parallel 8
```

Exit target for this task:

- All five handler families either implement batched `forwardMultiDecode` **or** tier doc explicitly lists documented serial fallback with issue link (prefer full parity).
- Tests per family as above.
- `docs/perf-compare/README.md` updated: which models were validated for multi-session batch speedup.
- `docs/agent-arch.txt` updated if architecture changes.

## Out of scope

- Continuous batching / paged KV (Tiers 14–16)
- Cluster `ProcessPipelineClient` / `TensorParallelPipelineClient` multi-request batch (local path first)
- Parallelizing prefill across requests in `generateBatch()` (lower priority)
- Changing single-stream `compare-llama-cpp.sh` baselines

## Git

Read-only unless user asks to commit.

## Deliverables

1. Working `forwardMultiDecode` on Phi-2, Phi-3, Qwen3, Qwen3 MoE (and LoRA wrapper if low-cost)
2. Per-family parity tests
3. Perf notes / compare run rows for non-Llama models where feasible
4. **List of changed files for preview** (no zip)
