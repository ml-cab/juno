(ch-4-7)=
# 4.7. Common Pitfalls

**`/train-qa` trains the typo.** If you type `whatos my name` the model learns that exact
string. Clean spelling in the question gives more reliable results.

**Loss still above target after training.** Raise `--lora-max-iters` or lower
`--lora-loss-target-qa` (e.g. `1.0`). For raw text, tune `--lora-loss-target-text`.

**Loss constant at ~log(vocabSize).** B starts at zero so the LoRA delta is zero for the first
forward pass. After the first backward and Adam step B becomes non-zero and loss will begin
moving. If still constant after step 2, check that `loraAdapters.get(li, proj)` is non-null.

**`--lora-play` answered wrong.** Check `[TRACE] model type` at startup. A template mismatch
between training and inference means the model cannot recall trained facts. Rename the file to
include the architecture keyword (`tinyllama`, `llama-3`, `mistral`, `phi3`, `qwen3`).

**Checkpoint loads but inference output is random.** After `LoraAdapterSet.load()`, call
`opt.reset()` before resuming training to clear stale momentum buffers. For inference-only use,
no optimizer is attached.

**FP32 VRAM OOM on Phi-3.5 with microbatch 8.** Use `--lora-microbatch 1` to start on FP16
sequential GEMV, or let the auto-fallback ladder handle it. Do not use
`JAVA_TOOL_OPTIONS=-Djuno.lora.microbatch=1`.

**`--lora-play` answers correctly at first then degrades after `/reset`.** `/reset` bumps the
DoRA cache generation so inference drops trained magnitude coefficients. This is correct
behavior. If training a new fact on a DoRA adapter, always `/reset` before retraining to avoid
stale magnitudes.

## See also

- [Chapter 3.8 -- Diagnostics and Tracing](#ch-3-8)
- [Chapter 4.3 -- Training Guide](#ch-4-3)
- [Chapter 4.8 -- Testing Checklist](#ch-4-8)

---

[<- 4.6 Programmatic API](#ch-4-6) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.8 Testing Checklist ->](#ch-4-8)
