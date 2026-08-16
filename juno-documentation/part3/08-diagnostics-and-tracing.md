(ch-3-8)=
# 3.8. Diagnostics and Tracing

Without `--verbose`, LoRA training prints a single-line progress bar
(`pass N . loss . bar . % . ETA`). Percent is loss progress from the pass-2 baseline toward the
loss target, not `pass/max-iters`. Pass `--verbose` / `-v` for full `[TRACE]` output:

| Line | What it tells you |
|------|-------------------|
| `[TRACE] model type (chat template key) : tinyllama` | Whether the template matches the model |
| `[train-qa] iter=N loss=...` | Per-pass loss during training |
| `[TRACE] inference model type: tinyllama` | Template key at inference; must match training |

If the template key at training and inference differ, the model will not recall trained facts.
Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`,
`phi3`, `qwen3`) so `ChatModelType.fromPath()` picks the matching chat template. Qwen2/2.5 use
ChatML; Qwen3 training uses the empty `<think>` block. LoRA training supports those dense
architectures via `LoraTrainingHandlerFactory`; Gemma, Qwen3-MoE, and Qwen3.5 LoRA remain
unsupported.

## See also

- [Chapter 3.5 -- LoRA Mode](#ch-3-5)
- [Chapter 4.7 -- Common Pitfalls](#ch-4-7)
- [Chapter 7.1 -- JFR and Metrics](#ch-7-1)

---

[<- 3.7 Test Mode](#ch-3-7) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.1 Concepts ->](#ch-4-1)
