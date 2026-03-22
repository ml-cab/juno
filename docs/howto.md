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
| *(no command)* | 3-node cluster — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail). Use `--pType pipeline\|tensor\|all` to filter |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | Activation wire format between nodes |
| `--max-tokens N` | `200` | Maximum tokens per response |
| `--temperature F` | `0.6` | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `20` | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.95` | Nucleus (top-p) sampling cutoff (0 = disabled) |
| `--heap SIZE` | `4g` | JVM heap per node, e.g. `4g` for Phi-3, `8g` for 7B models |
| `--nodes N` | `3` | Number of pipeline nodes (`local` only) |
| `--pType pipeline|tensor` | `pipeline` | Parallelism type: `pipeline` = contiguous layer blocks (vertical scaling); `tensor` = weight-matrix slices, all nodes in parallel (horizontal scaling). Constraint for tensor: `numHeads` must be even |
| `--verbose` / `-v` | — | Show node startup, gRPC, and shard loading logs |

**Environment overrides:** `MODEL_PATH`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`

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

# Activation dtype (local mode uses this for intermediate shard communication)
./juno local --model-path /path/to/model.gguf --dtype FLOAT32   # lossless debug
./juno local --model-path /path/to/model.gguf --dtype INT8      # max compression

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

# Phi-3.5 Mini (2.4 GB model, pipeline: 3 nodes each load ~800 MB of weights)
./juno --model-path /path/to/phi-3.5-mini-instruct-Q4_K_M.gguf --heap 4g

# Phi-3.5 Mini — tensor-parallel (each node loads full weights, lazy dequant keeps it in 4g)
./juno --pType tensor --model-path /path/to/phi-3.5-mini-instruct-Q4_K_M.gguf --heap 4g

# 7B model
./juno --model-path /path/to/mistral-7b.gguf --heap 8g

# Activation dtype between nodes
./juno --model-path /path/to/model.gguf --float16      # default
./juno --model-path /path/to/model.gguf --float32      # lossless debug
./juno --model-path /path/to/model.gguf --int8         # max compression
./juno --model-path /path/to/model.gguf --dtype FLOAT16

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

### `hyper.sh` — developer build+test tool

`hyper.sh` is the Maven-based dev runner. Unlike `juno` (which uses pre-built jars),
`hyper.sh` compiles source before running.

```bash
# Unit tests — all modules (~10–15s)
./hyper.sh test

# One module at a time
./hyper.sh test-module node
./hyper.sh test-module tokenizer
./hyper.sh test-module coordinator
./hyper.sh test-module player
./hyper.sh test-module kvcache
./hyper.sh test-module sampler
./hyper.sh test-module health
./hyper.sh test-module registry

# Fault-tolerance tests only
./hyper.sh test-fault

# Integration tests — forks 3 JVM node processes in stub mode (~30s)
./hyper.sh integration

# Fast integration — in-process only, zero network (~250ms)
./hyper.sh integration-fast

# Real-model smoke test (same 6 checks as ./juno test, but recompiles first)
MODEL_PATH=/path/to/model.gguf ./hyper.sh live

# Interactive cluster with recompile
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster

# Skip recompile (~10s saved when source hasn't changed)
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster -B
MODEL_PATH=/path/to/model.gguf ./hyper.sh live -B

# Build only (no tests)
./hyper.sh build

# Clean all target/ directories
./hyper.sh clean

# Full: compile + unit tests + integration tests
./hyper.sh verify

# Reference / demo
./hyper.sh health-demo      # fault-tolerance wiring walkthrough
./hyper.sh curl-demo        # example REST API curl commands
./hyper.sh watch node       # auto-rerun tests on file change (requires fswatch)
```

**Global env overrides for hyper.sh:**
```bash
MVN=/opt/maven/bin/mvn ./hyper.sh test    # custom Maven
PORT=9090 ./hyper.sh curl-demo            # custom coordinator port
```

---

### Typical dev session flow

```bash
# 1. First run — full build, verify everything passes
./hyper.sh clean && ./hyper.sh verify

# 2. After changing node/ or tokenizer/ — fast unit loop
./hyper.sh test-module node
./hyper.sh test-module tokenizer

# 3. Smoke the cluster without a model (stub mode)
./hyper.sh cluster -B                     # no MODEL_PATH → CyclicForwardPassHandler stubs

# 4. Interactive session with real model (recompile + run)
MODEL_PATH=/path/to/TinyLlama.gguf ./hyper.sh cluster -B

# 5. Interactive session with Phi-3 (larger heap)
MODEL_PATH=/path/to/phi-3.5-mini.gguf HEAP=4g ./hyper.sh cluster -B

# 6. Regression check
MODEL_PATH=/path/to/TinyLlama.gguf ./hyper.sh live -B

# 7. Before committing — full suite
./hyper.sh verify
```

Or using pre-built jars (faster startup, no Maven):
```bash
# Build once
mvn clean package -DskipTests

# Then use juno directly for all runs
./juno local --model-path /path/to/TinyLlama.gguf
./juno test --model-path /path/to/TinyLlama.gguf
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
GpuContext ctx      = GpuContext.init(0);            // open cuBLAS handle
MatVecBackend matVec = new CudaMatVecBackend(ctx);   // CUDA backend
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

### Port conflicts

`ThreeNodeClusterIT` and the cluster mode use ports 19092–19094. If a previous
run crashed without cleanup:

```bash
# Check
lsof -i :19092,19093,19094

# Kill
kill <pid>
```