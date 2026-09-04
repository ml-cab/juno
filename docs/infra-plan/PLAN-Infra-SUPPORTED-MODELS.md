# Supported models — Infra coverage rule

Authoritative execution rule: [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules §5.

## Purpose

Infra work (batching, GPU paths, KV, scheduler, prefill) must not ship tested on TinyLlama or Llama-only fixtures alone. Every change and test matrix below applies to **all** architectures Juno loads in production.

## Architecture → handler

Source: `node/.../ForwardPassHandlerLoader.java`.

| Architecture key | Example models | Handler |
|------------------|----------------|---------|
| LLaMA-family (default) | TinyLlama, Mistral-7B, Qwen2.5-3B, Llama-3 | `LlamaTransformerHandler` |
| `phi2` | Phi-2 | `Phi2TransformerHandler` |
| `phi3` | Phi-3.5-mini | `Phi3TransformerHandler` |
| `qwen3` | Qwen3 dense | `Qwen3TransformerHandler` |
| `qwen3moe` | Qwen3 MoE | `Qwen3MoeTransformerHandler` |

Vision and LoRA paths delegate to the text handler above; parity requirements follow the wrapped handler. Vision loading (`LlamafileGgufIndex`, `VisionModelPaths`) and the Vision I2T parallel track are documented in [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Parallel tracks — not as a silent exception to this rule.

## What “apply to all” means

### Implementation

| Change type | Requirement |
|-------------|-------------|
| Shared pipeline API (`InferencePipeline`, `GenerationLoop`, scheduler) | Must work for every handler via interface defaults or explicit overrides |
| Handler matmul / batch / KV | Parity in **each** handler family, or one shared helper called from all handlers |
| GPU backend (`CudaMatVec`, `GpuBlasOps`) | Exercised on at least one LLaMA-family and one non-LLaMA model in bake-off when GPU-specific |

Incremental Llama-first landings are allowed only if the active tier doc lists remaining architectures before exit.

### Tests

| Layer | Minimum per tier |
|-------|------------------|
| Node / handler | Parity or shape tests per architecture family affected (synthetic GGUF or `newTestInstance` where available) |
| Coordinator | Batch/scheduler tests remain handler-agnostic; add arch-specific live tests when behavior differs |
| Integration | At least one non-Llama arch in CI or documented manual gate when CI cannot load full GGUF |

Do not mark a tier complete with only `LlamaTransformerHandler*Test` updates when Phi-3 or Qwen3 share the same code path.

### Perf compare

Default bake-off set (`compare-llama-cpp.sh`):

- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` — LLaMA-family
- `qwen2.5-3b-instruct-q4_k_m.gguf` — LLaMA-family
- `Phi-3.5-mini-instruct-Q4_K_M.gguf` — `phi3`
- `mistral-7b-instruct-v0.1-q4_k_m.gguf` — LLaMA-family

Add `qwen3` / `qwen3moe` GGUFs to the matrix when those handlers gain the feature under test.

Specialized scripts (e.g. multi-session `compare-parallel.sh`) must document which architectures they cover in `docs/perf-compare/README.md`.

## Agent handoff checklist

Before marking an Infra tier complete:

1. List which handler families were changed.
2. List tests added/updated per family.
3. List perf runs per model (or document “serial fallback / not yet batched” per arch).
4. Confirm ROADMAP execution rules §1–§5 (one tier, perf publish including vision/LoRA when applicable, doc naming, no tier labels in shipped surfaces, all supported models).

**Agent prompts (linked from ROADMAP Parallel tracks):**

| Prompt | Purpose |
|--------|---------|
| [`PROMPT-MultiDecode-Parity.md`](PROMPT-MultiDecode-Parity.md) | Multi-request `forwardMultiDecode` on Phi-2/3, Qwen3, Qwen3 MoE (Tier 1 follow-up — **landed** on `67-inference`; keep as reference) |
| [`PROMPT-Vision-Perf.md`](PROMPT-Vision-Perf.md) | `compare-vision.sh` harness |
| [`PROMPT-Vision-Regression-Fix.md`](PROMPT-Vision-Regression-Fix.md) | Moondream hang / Q5_K×SIMD regression |
