(ch-04)=
# 4. Running Modes: local, cluster, lora, merge, test

This chapter walks through each command introduced in [Chapter 3](#ch-03) with runnable
examples. LoRA-specific workflows are covered in depth in [Chapter 9](#ch-09) and
[Chapter 10](#ch-10); this chapter only shows how `lora` and `merge` fit alongside the other
modes.

## `local` — in-process REPL

Fastest mode of the `juno-player` console. Operates within a single JVM; gRPC is off, and
`LocalInferencePipeline` is used instead of the process-boundary client.

```bash
# Minimal
./juno local --model-path /path/to/model.gguf

# With OpenAI-compatible REST API on port 8080
./juno local --model-path /path/to/model.gguf --api-port 8080

# With a pre-trained LoRA adapter applied at inference
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Via env var
LORA_PLAY_PATH=/path/to/model.lora MODEL_PATH=/path/to/model.gguf ./juno local

# With JFR (metrics printed on exit)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Verbose
./juno local --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**
```bat
juno.bat local --model-path models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

juno.bat local --model-path models\model.gguf --api-port 8080

juno.bat local --model-path models\model.gguf --lora-play adapters\model.lora

rem Via environment variable
set MODEL_PATH=C:\models\model.gguf
juno.bat local

juno.bat local --model-path models\model.gguf --jfr 5m
```

When `--lora-play` is given, the startup banner shows:

```
  Loading LoRA adapters for inference: /path/to/model.lora
  Loaded 44 LoRA adapters  (inference-only, no training)
```

When `--api-port` is given:

```
  ✔ Local API server on http://localhost:8080 (OpenAI: /v1/chat/completions)
```

## `cluster` — 3-node cluster (default command)

Forks 3 separate JVM node processes; each node loads its own shard of the model. Both
parallelism strategies from [Chapter 2](#ch-02) are available via `--pType`:

```bash
# Pipeline-parallel (default)
./juno --model-path /path/to/model.gguf

# With OpenAI-compatible REST API on port 8080
./juno --model-path /path/to/model.gguf --api-port 8080

# Tensor-parallel
./juno --pType tensor --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf PTYPE=tensor ./juno

# Activation dtype
./juno --model-path /path/to/model.gguf --dtype FLOAT16    # default
./juno --model-path /path/to/model.gguf --dtype FLOAT32    # lossless debug
./juno --model-path /path/to/model.gguf --dtype INT8       # max compression

# With JFR — coordinator + each node JVM writes its own .jfr file; metrics extracted per file on exit
./juno --model-path /path/to/model.gguf --jfr 5m

# With pre-trained adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Generation params
./juno --model-path /path/to/model.gguf --max-tokens 512 --temperature 0.3

# Verbose
./juno --model-path /path/to/model.gguf --verbose
```

**Windows (Command Prompt):**
```bat
juno.bat --model-path models\model.gguf

juno.bat --model-path models\model.gguf --api-port 8080

juno.bat --pType tensor --model-path models\model.gguf

rem Via environment variable
set MODEL_PATH=C:\models\model.gguf
set PTYPE=tensor
juno.bat

juno.bat --model-path models\model.gguf --jfr 5m

juno.bat --model-path models\model.gguf --lora-play adapters\model.lora

juno.bat --model-path models\model.gguf --max-tokens 512 --temperature 0.3
```

When `--lora-play` is given, `ClusterHarness.withLoraPlay(path)` injects
`-Djuno.lora.play.path=PATH` into every forked node JVM. Each node loads the adapter before
building its `ForwardPassHandler`.

## `lora` — LoRA fine-tuning REPL

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

For the full LoRA training guide, REPL commands, rank selection, and common pitfalls see
[Chapter 8](#ch-08) and [Chapter 9](#ch-09).

**Using a trained adapter outside `lora` mode:**

```bash
# Chat with adapter, no training REPL overhead
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# 3-node cluster with adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

**Profiling a slow training step:**

```bash
./juno lora --model-path /path/to/model.gguf --jfr 5m
# After exit, open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser -> juno.LoraTrainStep: forwardMs / backwardMs / optimizerMs / loss
```

## `merge` — bake a LoRA adapter into a standalone GGUF

Writes a new GGUF where LoRA-patched projection tensors (wq/wv on every layer, by default) are
stored as F32 for full precision. All other tensors are copied verbatim in their original
quantized encoding. The resulting file loads with `./juno local` or `./juno` like any other
model.

```bash
# Default: reads <model>.lora, writes <model>-merged.gguf
./juno merge --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Explicit paths
./juno merge --model-path /path/to/model.gguf \
             --lora-path  /adapters/my.lora   \
             --output     /path/to/merged.gguf

# Larger heap for big models (rule of thumb: 2x model file size)
./juno merge --model-path /path/to/Mistral-7B.gguf --heap 12g
```

**Windows (Command Prompt):**
```bat
juno.bat merge --model-path models\TinyLlama.Q4_K_M.gguf

juno.bat merge --model-path models\model.gguf ^
               --lora-path adapters\my.lora ^
               --output merged\merged.gguf

juno.bat merge --model-path models\Mistral-7B.gguf --heap 12g
```

For TinyLlama 1.1B Q4_K_M (667 MB), the merged file is approximately 1 GB. Full merge policy
details are in [Chapter 10](#ch-10).

## Diagnostics and tracing

Without `--verbose`, LoRA training prints a single-line progress bar
(`pass N · loss · bar · % · ETA`). Percent is loss progress from the pass-2 baseline toward the
loss target (not `pass/max-iters`). Pass `--verbose` / `-v` for full `[TRACE]` output:

| Line | What it tells you |
|------|-------------------|
| `[TRACE] model type (chat template key) : tinyllama` | Whether the template matches the model |
| `[TRACE] formatted training text (repr)` | Exact token sequence sent to the model during training |
| `[TRACE] token count (excl. BOS): N` | How many tokens are in the training sequence |
| `[train-qa] iter=N loss=…` | Per-pass loss during training |
| `[TRACE] inference model type: tinyllama` | Template key at inference — must match training |

If the template key at training and inference differ, the model will not recall trained facts.
Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`,
`phi3`, `qwen3`) so `ChatModelType.fromPath()` picks the matching chat template.

## Metrics

```bash
# Automatic in local mode (single JVM — all juno.* events in one .jfr file)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Cluster mode: coordinator + each node write separate .jfr files. On exit the launcher
# calls MetricsMain.extractToJson() once per existing file and prints each summary;
# target/metrics/metrics.json reflects the last processed file. For throughput (TPS),
# use the coordinator recording (juno.TokenProduced lives on the coordinator JVM).

# Manual extraction from .jfr files in the project root
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json (one snapshot per mapped .jfr in project root)
```

The JSON report includes these `juno.TokenProduced` fields, the primary throughput metrics for
performance comparison:

| Field | Description |
|-------|-------------|
| `juno.TokenProduced.count` | Total tokens delivered to clients in the recording window |
| `juno.TokenProduced.elapsed_seconds` | Wall-clock span from first to last delivered token |
| `juno.TokenProduced.tps` | Aggregate tokens per second (`count / elapsed_seconds`) |

Full reproduction methodology and matrix column definitions are in [Chapter 18](#ch-18).

---

[← Chapter 3: Commands and Flags](#ch-03) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 5: The OpenAI-Compatible REST API →](#ch-05)
