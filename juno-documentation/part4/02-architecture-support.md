(ch-4-2)=
# 4.2. Architecture Support

LoRA training, `--lora-play` inference overlay, and `./juno merge` are all routed by
`LoraTrainingHandlerFactory` using the GGUF `general.architecture` field:

| `general.architecture` | Handler | Support | Notes |
|---|---|---|---|
| `llama`, `mistral`, `tinyllama` | `LoraTrainableHandler` | Full | Separate Q/K/V/FFN tensors, adjacent-pair RoPE |
| `qwen2`, `qwen2.5` | `Qwen2LoraTrainableHandler` | Full | Same dense layout with frozen QKV biases |
| `phi3` | `Phi3LoraTrainableHandler` | Full | Fused `attn_qkv` / `ffn_up`; NeoX RoPE adjoint; fused-slice F32 merge |
| `qwen3` (dense) | `Qwen3LoraTrainableHandler` | Full | Per-head Q/K RMSNorm; `qDim` may differ from `hiddenDim` |
| `qwen3moe`, `qwen35`, `gemma`, unknown | **Rejected** | None | Explicit allowlist error; no silent routing |

Checkpoint keys remain logical (`wq,wk,wv,wo,wgate,wup,wdown`). Physical GGUF tensor names
and row-slice offsets are resolved at load and merge time via `LoraModelLayout` and
`LoraProjectionBinding`. Phi-3 fused-slice merge patches `attn_qkv` at Q/K/V row ranges and
`ffn_up` at gate/up row ranges; multiple logical adapters map to one physical tensor without
overwriting each other.

Qwen3 `/train-qa` text must include the closed empty `<think>` block to match inference
formatting (`ChatTrainingFormats` / `ChatTemplate.qwen3`).

---

## See also

- [Chapter 2.3 -- Handler Routing](#ch-2-3)
- [Chapter 4.1 -- Concepts](#ch-4-1)
- [Chapter 4.7 -- Common Pitfalls](#ch-4-7)

---

[<- 4.1 Concepts](#ch-4-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.3 Training Guide ->](#ch-4-3)
