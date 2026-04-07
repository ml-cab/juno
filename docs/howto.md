## Juno — complete how-to reference

```
./juno
```

Unified launcher at the project root. Detects the OS and delegates to
`scripts/run.sh` (Linux / macOS / Git Bash / WSL) or `scripts/run.bat` (Windows).
Requires a JDK 25+ and pre-built jars (`mvn clean package -DskipTests`).

---

### Commands

| Command | Description |
|---------|-------------|
| `local` | In-process REPL — all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single in-process JVM, adapter persisted to `.lora` file |
| *(no command)* | 3-node cluster — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail). Use `--pType pipeline\|tensor\|all` to filter |

### Flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | — | all | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format between nodes |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response |
| `--temperature F` | `0.6` | cluster, local, lora | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | cluster, local, lora | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | cluster, local, lora | Nucleus (top-p) sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | all | JVM heap per node, e.g. `4g` for Phi-3, `8g` for 7B models |
| `--nodes N` | `3` | local | Number of pipeline nodes (in-process only) |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type: `pipeline` = contiguous layer blocks; `tensor` = weight slices + AllReduce |
| `--jfr DURATION` | — | cluster, local, lora | Java Flight Recording for DURATION (e.g. `30s`, `5m`, `1h`). Writes `juno-<timestamp>.jfr` on exit. For `lora`, event browser shows `juno.LoraTrainStep` events. |
| `--verbose` / `-v` | — | cluster, local | Show node startup, gRPC, and shard loading logs |

**LoRA-specific flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint file. Loaded automatically if it exists next to the model. |
| `--gpu` | (default) | Use GPU when available |
| `--cpu` | — | Use CPU only |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension. `4`=minimal, `8`=standard, `16`=expressive. |
| `--lora-alpha F` | `= rank` | Scaling factor α. Effective scale = α/rank. |
| `--lora-lr F` | `1e-4` | Adam learning rate for LoRA adapter parameters. |
| `--lora-steps N` | `50` | Gradient steps applied per `/train` command. |

Without `--gpu` or `--cpu`, GPU is used by default. Use `--cpu` to force CPU.

 For GPU cluster runs, either set `CUDA_PATH`/`CUDA_HOME` (or run `setenv.bat` / `source setenv.sh`), or on Windows use the single-DLL option: run `get-cudart.bat` and save the downloaded `cudart64_12.dll` to `%USERPROFILE%\.javacpp\cache\` (see https://www.dllme.com/dll/files/cudart64_12).
 
**Environment overrides:** `MODEL_PATH`, `JUNO_USE_GPU`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`


---

### Download a model

```bash
# TinyLlama 1.1B — smallest, fastest, 637 MB, needs ~2g heap
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

# Phi-3.5 Mini Instruct — 3.8B, good quality, 2.4 GB, needs 4g heap
wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

# Mistral 7B — 4.1 GB, needs 8g heap
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# TinyLlama as llamafile (self-contained executable, ZIP polyglot)
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

---

### `local` — in-process REPL (fastest startup, everyday use)

All transformer shards run inside one JVM. No network, no forked processes.
`ForwardPassHandlerLoader` auto-detects model architecture from the GGUF file —
`phi3` routes to `Phi3TransformerHandler`, everything else to `LlamaTransformerHandler`.

```bash
# Minimal
./juno local --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf ./juno local

# Phi-3.5 Mini (needs at least 4g)
./juno local --model-path /path/to/phi-3.5-mini-instruct-Q4_K_M.gguf --heap 4g

# TinyLlama (comfortable at 2g)
./juno local --model-path /path/to/TinyLlama.Q4_K_M.gguf --heap 2g

# Generation params
./juno local --model-path /path/to/model.gguf --max-tokens 512 --temperature 0.3
./juno local --model-path /path/to/model.gguf --top-k 40              # wider sampling
./juno local --model-path /path/to/model.gguf --top-k 1               # near-greedy
./juno local --model-path /path/to/model.gguf --top-p 0.8             # tighter nucleus
./juno local --model-path /path/to/model.gguf --top-k 0 --top-p 0     # disable both filters

# More / fewer in-process shards (default 3)
./juno local --model-path /path/to/model.gguf --nodes 1   # single shard
./juno local --model-path /path/to/model.gguf --nodes 6   # more shards

# Activation dtype
./juno local --model-path /path/to/model.gguf --dtype FLOAT32   # lossless debug
./juno local --model-path /path/to/model.gguf --dtype INT8      # max compression

# Java Flight Recording — profiles the JVM for DURATION then dumps on exit
./juno local --model-path /path/to/model.gguf --jfr 5m
./juno local --model-path /path/to/model.gguf --jfr 30s
# Open the resulting juno-<timestamp>.jfr in JDK Mission Control

# Verbose — shows shard loading, architecture detection, token timing
./juno local --model-path /path/to/model.gguf --verbose

# Everything combined
./juno local --model-path /path/to/model.gguf --dtype FLOAT16 --max-tokens 512 \
  --temperature 0.5 --top-k 40 --top-p 0.9 --nodes 3 --heap 4g -v

# All via env vars
MODEL_PATH=/path/to/model.gguf DTYPE=FLOAT16 MAX_TOKENS=200 NODES=3 HEAP=4g \
  TOP_K=40 TOP_P=0.9 ./juno local
```

---

### `lora` — LoRA fine-tuning REPL

Runs a single in-process `LoraTrainableHandler` that serves both inference and
training. The base GGUF is **never modified**. Adapters are persisted to a
separate `.lora` checkpoint file alongside the model. See `docs/LoRA.md` for
the full design, gradient math, and checkpoint format.

```bash
# Minimal — auto-loads <model>.lora if it exists, creates new adapters otherwise
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Explicit adapter path
./juno lora --model-path /path/to/model.gguf --lora-path /adapters/my.lora

# Custom rank and learning rate
./juno lora --model-path /path/to/model.gguf --lora-rank 16 --lora-lr 5e-5

# More gradient steps per /train command (default 50)
./juno lora --model-path /path/to/model.gguf --lora-steps 100

# Larger heap (lora loads the full model in one JVM — use 2× the model file size)
./juno lora --model-path /path/to/model.gguf --heap 4g

# With JFR — records juno.LoraTrainStep events for each gradient step
./juno lora --model-path /path/to/model.gguf --jfr 5m
# After training: open juno-<timestamp>.jfr → Event Browser → juno.LoraTrainStep
# Each event shows: step, numTokens, loss, forwardMs, backwardMs, optimizerMs, totalMs

# Via env vars
MODEL_PATH=/path/to/model.gguf LORA_RANK=8 LORA_LR=0.0001 LORA_STEPS=50 ./juno lora

# --pType is accepted but silently ignored (lora always uses a single node)
./juno lora --pType tensor --model-path /path/to/model.gguf
```

**REPL commands inside lora mode:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text (`--lora-steps` gradient steps) |
| `/train-file <path>` | Fine-tune on a text file (auto-chunked into ≤128-token pieces) |
| `/save` | Save adapter checkpoint to `--lora-path` |
| `/reset` | Reinitialise adapters to B=0 (clears all training) |
| `/status` | Show adapter info: rank, α, parameter count, steps trained, checkpoint path |
| `/merge-hint` | Explain how to bake adapters into a new GGUF (offline merge) |
| `/help` | Show REPL command reference |
| *(regular input)* | Chat inference with current adapter applied |

The prompt shows `you*>` (asterisk) when there are unsaved adapter changes.
Typing `exit` with unsaved changes prompts before quitting.

**Session example:**
```
you > /train My name is Dima. I am a Java engineer.
  Training  rank=8 · lr=1.0E-4 · 50 steps · 1 chunk(s) · 10 tokens
  ──────────────────────────────────────────────────────────────
  step  50/50   loss=3.12  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%  2100ms/step  ETA 0s
  ✔ done  loss=▼ 3.12 (−3.50)  105s total  · /save to persist

you*> /save
  ✔ Saved → /path/to/model.lora  (44 adapters · 4401 KB · 50 steps trained)

you > What is your name?
bot> My name is Dima...
```

**Using a saved adapter:**
```bash
# Auto-load: if <model>.lora exists next to the GGUF, it loads automatically
./juno lora --model-path /path/to/model.gguf
# ✔ Loaded checkpoint: 44 adapters from /path/to/model.lora

# Explicit path
./juno lora --model-path /path/to/model.gguf --lora-path /adapters/my.lora
```

**Comparing base vs adapted model (two terminals):**
```bash
# Terminal 1 — base model, no adapter
./juno local --model-path /path/to/model.gguf --nodes 1

# Terminal 2 — with your adapter
./juno lora --model-path /path/to/model.gguf
```

**Profiling a slow training step:**
```bash
./juno lora --model-path /path/to/model.gguf --jfr 5m
# Inside the REPL:
you > /train some training text
# ... training runs ...
# After exit, open juno-<timestamp>.jfr in JDK Mission Control
# Event Browser → juno.LoraTrainStep shows per-step timing breakdown:
#   forwardMs / backwardMs / optimizerMs / totalMs / loss
```

**CPU performance note:** LoRA training on CPU runs the full forward+backward pass
through all transformer layers. For TinyLlama Q4_K_M on a typical 8-core machine,
expect ~2–5 seconds per gradient step for short sequences (7–10 tokens). Longer
sequences scale linearly with token count. Use `--lora-steps 5` for quick iteration
and `--lora-steps 100` when convergence matters. GPU training (via `CudaMatVecBackend`)
would be 20–50× faster.

---

### *(no command)* — 3-node cluster (forked JVMs, real gRPC)

Forks 3 separate JVM node processes. Each node loads its own shard of the model.
`ForwardPassHandlerLoader` runs inside each node JVM. Supports two distribution
strategies selected with `--pType`:

- **`pipeline`** (default) — contiguous layer blocks, serial activation flow node-1→node-2→node-3
- **`tensor`** — every node holds all layers but only a horizontal weight slice; coordinator broadcasts tokens to all nodes in parallel and reduces partial logit vectors (AllReduce)

```bash
# Minimal — pipeline-parallel (default)
./juno --model-path /path/to/model.gguf

# Tensor-parallel cluster
./juno --pType tensor --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf ./juno
MODEL_PATH=/path/to/model.gguf PTYPE=tensor ./juno

# Phi-3.5 Mini (2.4 GB model)
./juno --model-path /path/to/phi-3.5-mini-instruct-Q4_K_M.gguf --heap 4g

# Activation dtype between nodes
./juno --model-path /path/to/model.gguf --float16      # default
./juno --model-path /path/to/model.gguf --float32      # lossless debug
./juno --model-path /path/to/model.gguf --int8         # max compression

# Java Flight Recording
./juno --model-path /path/to/model.gguf --jfr 5m
./juno --pType tensor --model-path /path/to/model.gguf --jfr 30s

# Generation params
./juno --model-path /path/to/model.gguf --max-tokens 512
./juno --model-path /path/to/model.gguf --temperature 0.3
./juno --model-path /path/to/model.gguf --top-k 40
./juno --model-path /path/to/model.gguf --top-p 0.8
./juno --model-path /path/to/model.gguf --top-k 0 --top-p 0    # disable both filters

# Verbose — shows gRPC, shard assignments, node startup
./juno --model-path /path/to/model.gguf --verbose
./juno --model-path /path/to/model.gguf -v

# Everything combined — tensor-parallel
./juno --pType tensor --model-path /path/to/model.gguf --dtype FLOAT16 \
  --max-tokens 512 --temperature 0.5 --top-k 40 --top-p 0.9 --heap 4g -v

# All via env vars
MODEL_PATH=/path/to/model.gguf PTYPE=tensor DTYPE=FLOAT16 MAX_TOKENS=512 \
  TEMPERATURE=0.5 TOP_K=40 TOP_P=0.9 HEAP=4g ./juno

# Custom JDK
JAVA_HOME=/opt/jdk-25 ./juno --model-path /path/to/model.gguf
```

---

### `test` — real-model smoke test (8 checks, exits 0/1)

Runs `ModelLiveRunner`: 8 automated checks with coloured pass/fail output.
Use after any change to `node`, `tokenizer`, or `coordinator` before committing.

```bash
# Full suite (pipeline tests 1-6 + tensor tests 7-8)
./juno test --model-path /path/to/model.gguf

# Model as env var
MODEL_PATH=/path/to/model.gguf ./juno test

# Bigger heap for larger models
./juno test --model-path /path/to/phi-3.5-mini.gguf --heap 4g

# Pipeline-parallel checks only (tests 1-6, faster)
./juno test --model-path /path/to/model.gguf --pType pipeline

# Tensor-parallel checks only (tests 7-8)
./juno test --model-path /path/to/model.gguf --pType tensor

# With JFR — profile what ModelLiveRunner spends time on
./juno test --model-path /path/to/model.gguf --jfr 5m

# Verbose — shows prefill timing and token IDs per step
./juno test --model-path /path/to/model.gguf --verbose
```

Checks run (in order):

Pipeline-parallel cluster (tests 1-6):
1. `hello_greeting` — response contains a greeting word
2. `no_raw_sentencepiece_markers` — no `▁` (U+2581) in output
3. `question_response` — non-empty response to "What is 2 plus 2?"
4. `greedy_determinism` — identical output on two runs with `SamplingParams.deterministic()`
5. `multi_turn_conversation` — 3-turn conversation, prompt grows correctly
6. `float16_parity` — FLOAT16 activation pipeline runs end-to-end without error

Tensor-parallel cluster (tests 7-8, fresh 3-node cluster started after pipeline cluster stops):
7. `tensor_parallel_generation` — non-empty output from AllReduce forward pass
8. `tensor_parallel_greedy_determinism` — identical output on two greedy runs through AllReduce

---

### Supported models and heap requirements

| Model | Size GB | `--heap` |
|-------|---------|----------|
| TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf | 0.67 | 2g |
| TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile | 0.97 | 2g |
| Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile | 1.6 | 2g |
| phi-3.5-mini-instruct.Q4_K_M.gguf | 2.4 | 4g |
| Mistral-7B-Instruct-v0.2.Q4_K_M.gguf | 4.1 | 8g |
| Llama-3.2-8B-Instruct-Q4_K_M.gguf | 4.9 | 8g |


**Chat template** is derived automatically from the model path:
`phi-3*` → phi3, `tinyllama*` / `zephyr*` → tinyllama, `llama-3*` / `llama3*` → llama3,
`mistral*` → mistral, `gemma*` → gemma, everything else → chatml.

**Architecture routing** (`ForwardPassHandlerLoader`):
`phi3` → `Phi3TransformerHandler`, all others → `LlamaTransformerHandler`.

---

Or using pre-built jars (faster startup, no Maven):
```bash
# Build once
mvn clean package -DskipTests

# Then use juno directly for all runs
./juno local --model-path /path/to/TinyLlama.gguf
./juno lora  --model-path /path/to/TinyLlama.gguf
./juno test  --model-path /path/to/TinyLlama.gguf
```

---

### Remote deployment fat jars

Two fat jars are produced for deploying Juno to remote machines (AWS, bare-metal):

**`juno-node/target/juno-node.jar`** — main class `cab.ml.juno.node.NodeMain`

Standalone node process. Reads configuration from system properties:

```
-Dnode.id=<nodeId>          required — e.g. node-1
-Dnode.port=<port>          required — gRPC port, e.g. 19092
-Dmodel.path=<modelPath>    optional — if absent, runs with dummy handler
-DJUNO_USE_GPU=<true|false> optional — defaults to true
```

Command-line args (`nodeId port [modelPath]`) are also accepted for backward compatibility; system properties take precedence.

```bash
java -Dnode.id=node-1 -Dnode.port=19092 -Dmodel.path=/models/TinyLlama.gguf \
     -jar juno-node/target/juno-node.jar
```

Prints `READY:<nodeId>:<port>` to stdout when the gRPC server is up. On AWS, launched by `juno-node.service` (systemd).

**`juno-master/target/juno-master.jar`** — main class `cab.ml.juno.master.CoordinatorMain`

Standalone coordinator process. No forking, no `ClusterHarness` — nodes must already be running via `NodeMain`. Reads all configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `JUNO_NODE_ADDRESSES` | — | Comma-separated `host:port` list, one per node (required) |
| `JUNO_MODEL_PATH` | — | Local path to GGUF file for tokenizer + config (required) |
| `JUNO_PTYPE` | `pipeline` | `pipeline` or `tensor` |
| `JUNO_HTTP_PORT` | `8080` | REST / web console port |
| `JUNO_DTYPE` | `FLOAT16` | Activation wire format |
| `JUNO_MAX_QUEUE` | `1000` | Scheduler queue depth |

```bash
JUNO_NODE_ADDRESSES=10.0.0.1:19092,10.0.0.2:19093,10.0.0.3:19094 \
JUNO_MODEL_PATH=/models/TinyLlama.gguf \
java -jar juno-master/target/juno-master.jar
```

Prints `COORDINATOR_READY:<port>` to stdout when the REST server is accepting requests. On AWS, launched by `juno-coordinator.service` (systemd) after `juno-deploy.sh` writes `cluster-nodes.env`.

---

### GPU testing

GPU stack: org.bytedeco cuda-platform (cudart + cublas). Works with various
NVIDIA GPUs (e.g. GTX 1080, T4). Requires NVIDIA driver; CUDA runtime is
bundled in the cuda-platform dependency. On Windows ensure nvidia-smi shows
your GPU; no extra PATH needed for tests (JavaCPP loads natives from the jar).

Production GPU loads use `GpuForwardPassHandler.loadGpuResident`: each matmul
weight matrix is uploaded once (`DeviceFloatMatrix`). Call
`GpuForwardPassHandler.releaseGpuResources()` before closing `GpuContext`.
`GpuForwardPassHandler.load` with host tensors remains for `CpuMatVec` and tests.


GPU tests are tagged `@Tag("gpu")` and excluded from the default test run.
A CUDA 12.x device and the JCuda native libs are required.

```bash
# Run all tests including GPU-tagged (requires CUDA + Nvidia GPU, org.bytedeco cuda)
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

# Skip GPU tests (default — works on any machine)
mvn test -Dgroups='!gpu' -pl node
```
GPU integration test — requires CUDA, Nvidia GPU, and a GGUF model file


`CudaMatVecBackendTest` inherits all 17 contract tests from `MatVecBackendContractTest`
and adds CUDA-specific numerical comparison and timing assertions.
`GpuContextTest` verifies cuBLAS handle lifecycle (6 tests, all `@Tag("gpu")`).
`CudaAvailabilityTest` has 4 always-run tests + 4 `@Tag("gpu")` tests.

```bash
# GPU integration test — requires CUDA 12.x, GPU, and a GGUF model file
mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl integration \
  --enable-native-access=ALL-UNNAMED

# Via env var
MODEL_PATH=/path/to/model.gguf mvn verify -Pgpu -pl integration \
  --enable-native-access=ALL-UNNAMED
```

`GpuForwardPassIT` is excluded from the default failsafe scan to prevent CUDA
native libraries (bytedeco) from loading into the coordinator JVM and poisoning FD
The `-Pgpu` Maven
profile re-includes it and sets `-Djuno.gpu.test=true` — a guard checked in
`@BeforeAll` before any JCuda class is loaded. This prevents the JCuda native
library from loading into the coordinator JVM and poisoning FD inheritance into
the node processes forked by `ClusterHarness`.

Wiring a GPU node (code reference):
```java
GpuContext ctx = GpuContext.init(0);
MatVec matVec  = new CudaMatVec(ctx);
ForwardPassHandler h = LlamaTransformerHandler.load(modelPath, shardCtx, matVec);
// or for Phi-3:
ForwardPassHandler h = Phi3TransformerHandler.load(modelPath, shardCtx, matVec);
```

---

### AWS setup for GPU testing (g4dn.xlarge — T4 16 GB VRAM, ~$0.50/hr)

```bash
# 1. Install CUDA 12.x
sudo apt update && sudo apt install -y nvidia-cuda-toolkit

# 2. Verify GPU
nvidia-smi

# 3. Install JDK 25 and Maven
sudo apt install -y openjdk-25-jdk maven

# 4. Clone and build
git clone https://github.com/ml-cab/juno
cd juno
mvn clean package -DskipTests

# 5. Download a model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

# 6. GPU unit tests first (validates CUDA wiring, no model needed)
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

# 7. GPU integration test
mvn verify -Pgpu -Dit.model.path=$(pwd)/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
  -pl integration --enable-native-access=ALL-UNNAMED

# 8. Interactive session on GPU
./juno local --model-path $(pwd)/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
```

---


### GPU testing

GPU tests are tagged `@Tag("gpu")` and excluded from the default test run.
A CUDA 12.x device and the JCuda native libs are required.

```bash
# GPU unit tests — node module only, no model file needed
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

# Skip GPU tests (default — works on any machine)
mvn test -Dgroups='!gpu' -pl node
```

`CudaMatVecBackendTest` inherits all 17 contract tests from `MatVecBackendContractTest`
and adds CUDA-specific numerical comparison and timing assertions.
`GpuContextTest` verifies cuBLAS handle lifecycle (6 tests, all `@Tag("gpu")`).
`CudaAvailabilityTest` has 4 always-run tests + 4 `@Tag("gpu")` tests.

```bash
# GPU integration test — requires CUDA 12.x, GPU, and a GGUF model file
mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl integration \
  --enable-native-access=ALL-UNNAMED

# Via env var
MODEL_PATH=/path/to/model.gguf mvn verify -Pgpu -pl integration \
  --enable-native-access=ALL-UNNAMED
```

`GpuForwardPassIT` is excluded from the default failsafe scan. The `-Pgpu` Maven
profile re-includes it and sets `-Djuno.gpu.test=true` — a guard checked in
`@BeforeAll` before any JCuda class is loaded. This prevents the JCuda native
library from loading into the coordinator JVM and poisoning FD inheritance into
the node processes forked by `ClusterHarness`.

Wiring a GPU node (code reference):
```java
GpuContext ctx = GpuContext.init(0);              // open cuBLAS handle
MatVec matVec  = new CudaMatVec(ctx);             // CUDA backend
// Weights are dequantized + uploaded to DeviceFloatMatrix at load time.
// Forward passes copy only x and y across the bus, not the weight matrix.
ForwardPassHandler h = LlamaTransformerHandler.load(modelPath, shardCtx, matVec);
// or for Phi-3:
ForwardPassHandler h = Phi3TransformerHandler.load(modelPath, shardCtx, matVec);
```

For the local in-process pipeline, backend selection is automatic:
`ForwardPassHandlerLoader.selectBackend()` reads the `JUNO_USE_GPU` system
property and calls `CudaAvailability.isAvailable()`. Pass `--gpu` on the
command line (or set `JUNO_USE_GPU=true`) to activate the GPU path.

---

### AWS — cluster deployment (`juno-deploy.sh`)

`juno-deploy.sh` is the unified cluster lifecycle script. It replaces the earlier
`juno-infra.sh` (GPU) and `juno-infra-ft.sh` (CPU). Hardware is auto-detected
during bootstrap: GPU instances install CUDA and start with `JUNO_USE_GPU=true`;
CPU instances skip CUDA entirely. Both cluster types use the same script and the
same command surface.

**One-time setup — fill in credentials:**

```bash
cd scripts/aws
nano launcher.sh    # set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

**Commands:**

```
./launcher.sh juno-deploy.sh setup      [options]   # provision + bootstrap + open web console
./launcher.sh juno-deploy.sh start                  # start stopped instances
./launcher.sh juno-deploy.sh stop                   # stop instances (EBS + key retained)
./launcher.sh juno-deploy.sh teardown               # terminate everything — no lingering costs
./launcher.sh juno-deploy.sh status                 # show instance states
./launcher.sh juno-deploy.sh scan-regions           # find cheapest AZ for instance type
```

**Setup options (all optional):**

| Option | Default | Description |
|--------|---------|-------------|
| `--instance-type TYPE` | `g4dn.xlarge` | GPU: `g4dn.xlarge/2xlarge/4xlarge`. CPU: `m7i-flex.large`, `c7i-flex.large`, `t3.medium` |
| `--node-count N` | `3` | Number of inference nodes |
| `--coordinator node1` | (default) | Co-locate coordinator JVM on node 1 (free) |
| `--coordinator separate` | — | Launch extra t3.medium coordinator instance |
| `--model-url URL` | TinyLlama Q4_K_M | Model to download during bootstrap |
| `--ptype pipeline\|tensor` | `pipeline` | Parallelism type passed to nodes |
| `--dtype FLOAT32\|FLOAT16` | `FLOAT16` | Activation wire format |

**GPU cluster — 3 × g4dn.xlarge (T4 16 GB VRAM, ~$0.53/hr each):**

```bash
cd scripts/aws
./launcher.sh juno-deploy.sh setup
```

NB: requires switching your AWS account off Free Tier.
See `docs/1buks-aws-vcpu-infra.md` for quota-request and billing guidance.

**CPU cluster — 3 × m7i-flex.large (8 GB RAM, 2 vCPUs, ~$0.048/hr each):**

```bash
cd scripts/aws
./launcher.sh juno-deploy.sh setup \
  --instance-type m7i-flex.large \
  --model-url https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

Free Tier compatible.

**Teardown:**

```bash
./launcher.sh juno-deploy.sh teardown
# [juno]   Terminating 3 instances…
# [juno]   SG deleted / Key pair deleted / Local key removed
# [juno]   Cluster fully torn down. No lingering AWS costs.
```

**What happens during `setup`:**

1. Resolves Ubuntu 22.04 LTS AMI and cheapest AZ for the instance type.
2. Creates key pair (saved to `~/.ssh/juno-deploy-key.pem`) and security group
   (SSH from your IP, gRPC internal, REST public on port 8080).
3. Launches N node instances. If `--coordinator separate`, also launches a t3.medium.
4. Waits for instances to reach `running` state.
5. Bootstraps all nodes in parallel (~5 min): installs JDK 25 + Maven, detects
   GPU via `lspci` (installs CUDA if found), clones and builds juno, downloads model,
   writes `/etc/juno/env`, starts `juno-node.service` via systemd.
   Bootstrap log: `/var/log/juno-bootstrap.log` on each instance.
6. SSHes into the coordinator host, writes `/opt/juno/cluster-nodes.env` with
   private IPs of all nodes, then starts `juno-coordinator.service`.
7. Waits for coordinator REST to respond on `http://<coordinator>:8080/v1/cluster/health`.
8. Prints cluster summary and enters the live monitor (refreshed every 20 s).

**Live monitor output (example — 3 × m7i-flex.large, pipeline):**

```
  Uptime      :  00:04:33
  Est. cost   :  $0.0108  ($0.0479/hr × 3 instances)
  Console     :  http://51.20.255.51:8080

  Nodes:
    node-1   51.20.255.51     sys:ok  ready:yes cpu:22% mem:3268/7780MB  coord:active
    node-2   51.21.220.189    sys:ok  ready:yes cpu:18% mem:384/7780MB   coord:inactive
    node-3   51.21.218.9      sys:ok  ready:yes cpu:17% mem:389/7780MB   coord:inactive
```

Ctrl+C auto-stops all instances before exit (EBS volumes and key pair retained
for `start`). Use `teardown` to terminate everything.

**State file:** `~/.juno-deploy-state` — persists instance IDs, SG ID, and setup
parameters so `start / stop / teardown` work without repeating options.

**Web console:** served at `http://<coordinator>:8080` once the cluster is healthy.
Supports streaming chat, model selection, and displays per-request token counts,
latency, and throughput. Verified working with TinyLlama-1.1B-Chat-v1.0.Q5_K_M
on a 3-node CPU cluster.

---

### Metrics — extracting metrics data from JFR recordings

The `metrics` module reads JFR files produced by
`--jfr` and writes a JSON summary to `target/metrics/metrics.json`.

**Step 1 — capture a recording with the model stem in the filename.**
The `--jfr` flag now automatically names the file
`juno-<modelStem>-YYYYMMDD-HHMMSS.jfr`, so the extractor can correlate it with
the right `models.json` entry:

```bash
./juno local --model-path /models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf --jfr 5m
# produces: juno-TinyLlama-1.1B-Chat-v1.0.Q4_K_M-20260403-142311.jfr
```

Use `./juno local`, not `./juno` (cluster). In cluster mode JFR attaches to
the coordinator JVM only; inference runs in forked node JVMs and their
`juno.MatVec` / `juno.ForwardPass` events are not captured.

**Step 2 — add the model to `metrics/src/main/resources/models.json`**
if it is not already there:

```json
{
  "models": [
    { "name": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",
      "path": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" }
  ]
}
```

**Step 3 — build and run the extractor:**

```bash
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar \
     cab.ml.juno.metrics.metricsMain
# Output: target/metrics/metrics.json
```

The JSON contains counts, total durations, and p50/p95/p99 percentiles for
every `juno.*` event type: `juno.MatVec`, `juno.ForwardPass`,
`juno.Tokenizer`, `juno.TemplateFormat`, `juno.LoraTrainStep`.

---

### Llama 3 / Meta-Llama models — GPT-2 BPE tokenizer

Llama 3.x models (e.g. `Meta-Llama-3.2-1B-Instruct-Q8_0`) use GPT-2 /
tiktoken BPE, which is detected automatically from the GGUF metadata key
`tokenizer.ggml.model == "gpt2"`. This is distinct from the SentencePiece BPE
used by Llama 1/2, TinyLlama, Mistral, Gemma, and Phi-3.

The tokenizer logs the detected variant on load:

```
Tokenizer loaded: vocabSize=128256 bos=128000 eos=128001 model=gpt2 [GPT-2 BPE]
```

No configuration is required. Control tokens such as `<|begin_of_text|>`,
`<|eot_id|>`, `<|start_header_id|>`, and `<|end_header_id|>` are pre-split
before BPE and mapped to their single vocabulary IDs.