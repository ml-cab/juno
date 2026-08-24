(ch-3-5)=
# 3.5. LoRA Mode

`./juno lora` starts the LoRA fine-tuning REPL: a single in-process JVM that trains low-rank
adapters and persists them to a `.lora` checkpoint file.

```bash
# Minimal -- auto-loads <model>.lora if it exists
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# With verbose tracing (recommended when debugging training)
./juno lora --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**

```bat
juno.bat lora --model-path models\TinyLlama.Q4_K_M.gguf

juno.bat lora --model-path models\model.gguf --verbose
```

For the full LoRA training guide, REPL commands, rank selection, and common pitfalls, see
[LoRA fine-tuning](#ch-4-1). Multi-fact Q&A training uses
`/train-file-qa facts.json` with a JSON array of `{"Q":"...","A":"..."}` objects (one training
loop). With `--api-port N` the same JSON can be posted via curl to
`POST /v1/lora/train-file-qa`, followed by `POST /v1/lora/save`.

## Using a trained adapter outside `lora` mode

```bash
# Chat with adapter, no training REPL overhead
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# 3-node cluster with adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

**Windows:**

```bat
juno.bat local --model-path models\model.gguf --lora-play adapters\model.lora
juno.bat --model-path models\model.gguf --lora-play adapters\model.lora
```

## Profiling a slow training step

```bash
./juno lora --model-path /path/to/model.gguf --jfr 5m
# After exit, open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser -> juno.LoraTrainStep: forwardMs / backwardMs / optimizerMs / loss
```

**Windows:**

```bat
juno.bat lora --model-path models\model.gguf --jfr 5m
```

## See also

- [Chapter 3.2 -- Flags](#ch-3-2)
- [Chapter 4.1 -- Concepts](#ch-4-1)
- [Chapter 3.6 -- Merge Mode](#ch-3-6)

---

[<- 3.4 Cluster Mode](#ch-3-4) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.6 Merge Mode ->](#ch-3-6)
