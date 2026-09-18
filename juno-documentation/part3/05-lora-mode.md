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

## Multiple adapters and per-adapter scales

`--lora-play` accepts more than one adapter, each with its own scale:
`path[:scale][,path[:scale]...]`. A bare path (no colon) defaults to scale `1.0`, so the
single-file form above is unchanged.

```bash
./juno local --model-path /path/to/model.gguf \
             --lora-play /adapters/style.lora:0.5,/adapters/facts.lora:1.0
```

The effective delta is the sum of each adapter's own contribution scaled by its playback scale —
mathematically identical to running each adapter separately and adding the results, computed
without extra per-token overhead by folding the scales into a single merged adapter at load time.
QA-LoRA and DoRA checkpoints only support the single-file, scale-`1.0` form for now; combining one
with another adapter, or giving it a non-default scale, fails closed with a clear error rather
than silently dropping its quantization-aware or magnitude term.

Per-request adapter selection via the OpenAI-compatible API's `x_juno_loras` field is not wired
yet on any serving schedule — a request that sets it is rejected (HTTP 400); use process-wide
`--lora-play` with the multi-adapter syntax above instead.

## Importing a GGUF LoRA adapter

`./juno lora-import` converts a GGUF LoRA adapter (the tensor naming convention is documented in
the CLI's own `--help`) into a Juno `.lora` v2 checkpoint usable with `--lora-play`:

```bash
./juno lora-import --gguf /path/to/hub-adapter.gguf --out /adapters/hub-adapter.lora
./juno local --model-path /path/to/model.gguf --lora-play /adapters/hub-adapter.lora
```

This is a format converter, not a training path — Juno's own adapters are still trained with
`juno lora` as described above. Unrecognized tensor names or a rank/shape mismatch between an
adapter's two halves fail the import closed with a clear message.

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
