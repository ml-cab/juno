(ch-1-4)=
# 1.4. Supported Models

Juno loads GGUF files with LLaMA-compatible architectures. Supported quantizations: F32, F16,
BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

## Architecture support matrix

| Model family | Chat template key | Handler | Status |
|---|---|---|---|
| LLaMA 3 | `llama3` | `LlamaTransformerHandler` | Supported |
| Mistral | `mistral` | `LlamaTransformerHandler` | Supported |
| TinyLlama / Zephyr | `tinyllama` | `LlamaTransformerHandler` | Supported |
| Phi-3 / Phi-3.5 | `phi3` | `Phi3TransformerHandler` | Supported |
| Gemma | `gemma` | `LlamaTransformerHandler` | Under development |
| Qwen 2 / 2.5 | `chatml` | `LlamaTransformerHandler` (+ QKV bias) | Under development |
| Qwen3 (dense) | `chatml` | `Qwen3TransformerHandler` | Under development |
| Qwen3-MoE | `chatml` | `Qwen3MoeTransformerHandler` | Under development |
| Qwen3.5 | `chatml` (partial) | Not yet implemented (hybrid DeltaNet architecture) | Under development |

"Under development" means template and handler groundwork exists for some paths, but
end-to-end validation is still in progress. LoRA training and `--lora-play` inference are not
available for architectures marked under development. See
[Handler routing](#ch-2-3) for how dispatch works internally.

## Example heap sizing

| Model | Approximate size | Suggested `--heap` |
|---|---|---|
| TinyLlama Q4_K_M | ~637 MB | `2g` |
| Phi-3.5-mini Q4_K_M | ~2.2 GB | `4g` |
| Mistral-7B Q4_K_M | ~4.1 GB | `8g` |
| Llama-3.1-70B Q4_K_M | distributed across nodes | see [Distributed inference](#ch-2-2) |

## See also

- [Chapter 2.3 -- Handler Routing](#ch-2-3)
- [Chapter 1.1 -- Requirements](#ch-1-1)
- [Chapter 4.2 -- Architecture Support](#ch-4-2)

---

[<- 1.3 Quickstart: JVM Embedding](#ch-1-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [2.1 Overview ->](#ch-2-1)
