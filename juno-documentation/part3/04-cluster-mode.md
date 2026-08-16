(ch-3-4)=
# 3.4. Cluster Mode

`./juno` (no subcommand, or `cluster` explicitly) is the default command: a 3-node cluster with
forked JVM node processes and real gRPC. Each node loads its own shard of the model. Two
distribution strategies are available via `--pType`; see
[Distributed inference](#ch-2-2) for how they work
internally.

- **`pipeline`** (default): contiguous layer blocks, serial activation flow node-1 to node-2 to
  node-3.
- **`tensor`**: every node holds all layers but only a horizontal weight slice; the coordinator
  broadcasts tokens to all nodes in parallel and reduces partial logit vectors (AllReduce).

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

# With JFR: coordinator and each node JVM writes its own .jfr file; metrics are extracted per file on exit
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

## See also

- [Chapter 3.2 -- Flags](#ch-3-2)
- [Chapter 2.2 -- Distributed Inference](#ch-2-2)
- [Chapter 6.1 -- On-Prem Cluster](#ch-6-1)

---

[<- 3.3 Local Mode](#ch-3-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.5 LoRA Mode ->](#ch-3-5)
