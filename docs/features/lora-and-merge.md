# LoRA training, inference, and merge

Train low-rank adapters in-process with `./juno lora`, persist checkpoints as `.lora`, apply them read-only at inference with `--lora-play`, or bake weights into a new GGUF using `./juno merge`. The base GGUF file is never modified during training; merge produces a standalone artifact for deployment without a sidecar adapter.

Operational detail, REPL commands, and hyperparameters are in [LoRA.md](../LoRA.md). Redistributing merged models may interact with base-model and adapter licenses; see [legal.md](../legal.md).
