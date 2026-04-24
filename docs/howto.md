## Juno — complete how-to reference

```
./juno
```

Unified launcher at the project root. Requires JDK 25+ and pre-built jars (`mvn clean package -DskipTests`).

---

### Commands

| Command | Description |
|---------|-------------|
| `local` | In-process REPL — all shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single JVM, adapter persisted to `.lora` file |
| *(no command)* | 3-node cluster — forked JVMs, real gRPC |
| `test` | 8 automated real-model smoke checks, exits 0/1 |

---

| `local` | In-process REPL — all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single in-process JVM, adapter persisted to `.lora` file |
| `merge` | Bake a trained `.lora` adapter into a new standalone GGUF — no sidecar needed at inference time |
| *(no command)* | 3-node cluster — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail). Use `--pType pipeline\|tensor\|all` to filter |

### Flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | — | all | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response |
| `--temperature F` | `0.6` | all | Sampling temperature |
| `--top-k N` | `20` | all | Top-K sampling cutoff |
| `--top-p F` | `0.95` | all | Nucleus sampling cutoff |
| `--heap SIZE` | `4g` | all | JVM heap per node |
| `--nodes N` | `3` | local | Number of in-process shards |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type |
| `--jfr DURATION` | — | cluster, local, lora | Java Flight Recording |
| `--verbose` / `-v` | — | cluster, local | Verbose logging |
| `--lora-play PATH` | — | cluster, local | Apply a pre-trained `.lora` adapter at inference (read-only, no training). Path must be absolute or relative to the working directory where `./juno` is invoked. In cluster mode the file is forwarded as `-Djuno.lora.play.path` to every forked node JVM. |

**LoRA-specific flags** (`lora` command only):

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Scaling factor α |
| `--lora-lr F` | `1e-4` | Adam learning rate |
| `--lora-steps N` | `50` | Gradient steps per `/train` |
| `--lora-steps-qa N` | `10` | Gradient steps per `/train-qa` Q&A pair |
| `--lora-early-stop F` | `0.25` | Stop chunk early when loss delta < F |

**Environment overrides:** `MODEL_PATH`, `JUNO_USE_GPU`, `DTYPE`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_STEPS`, `LORA_PLAY_PATH`

---

### `local` — in-process REPL

```bash
# Minimal
./juno local --model-path /path/to/model.gguf

# With a pre-trained LoRA adapter applied at inference
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# Via env var
LORA_PLAY_PATH=/path/to/model.lora MODEL_PATH=/path/to/model.gguf ./juno local

# With JFR
./juno local --model-path /path/to/model.gguf --jfr 5m

# Verbose (shows [TRACE] model type, LoRA adapter count, etc.)
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora --verbose
```

When `--lora-play` is given, the startup banner shows:
```
  ⚙ Loading LoRA adapters for inference: /path/to/model.lora
  ✔ Loaded 44 LoRA adapters  (inference-only, no training)
```

---

### `lora` — LoRA fine-tuning REPL

```bash
# Minimal — auto-loads <model>.lora if it exists
./juno lora --model-path /path/to/TinyLlama.Q4_K_M.gguf

# With verbose tracing (recommended when debugging training)
./juno lora --model-path /path/to/model.gguf --verbose
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `/train <text>` | Fine-tune on inline text |
| `/train-file <path>` | Fine-tune on a text file |
| `/train-qa <question> A: <answer>` | Train a single Q&A fact (auto-generates 4 phrasings) |
| `/save` | Save adapter checkpoint |
| `/reset` | Reinitialise adapters to zero |
| `/status` | Show rank, α, steps trained, checkpoint path |
| `/merge-hint` | Explain how to bake adapters into a new GGUF |
| `/help` | Show command reference |

| `/train <text>` | Fine-tune on inline text (`--lora-steps` gradient steps) |
| `/train-file <path>` | Fine-tune on a text file (auto-chunked into ≤128-token pieces) |
| `/save` | Save adapter checkpoint to `--lora-path` |
| `/reset` | Reinitialise adapters to B=0 (clears all training) |
| `/status` | Show adapter info: rank, α, parameter count, steps trained, checkpoint path |
| `/merge-hint` | Show the `juno merge` command to bake adapter into a standalone GGUF |
| `/help` | Show REPL command reference |
| *(regular input)* | Chat inference with current adapter applied |

**`/train-qa` — conversational fact training:**

```
you > /train-qa What is my name? A: Dima

  Question: What is my name?
  Answer  : Dima

  [TRACE] ── formatted training text (repr) ──────────────────
  <|user|>↵
  What is my name?</s>↵
  <|assistant|>↵
  Dima</s>↵
  ...
  [TRACE] ── end training text ────────────────────────────────
  [TRACE] token count (excl. BOS): 121

  Formatted as 4 Q&A pairs  ·  model type: tinyllama
  Training  rank=8 · lr=1.0E-4 · 40 steps · 4 chunk(s) · 122 tokens
  ✔ done  loss=▼ 1.53 (−0.83)
```

The command auto-generates four phrasings of the question to aid generalization. Loss below ~0.5 gives reliable recall; above ~1.5 the answer may be inconsistent. Use `--lora-steps-qa 50` and repeat the command 2–3 times to drive loss lower.

**Using a trained adapter outside `lora` mode:**
```bash
# Chat with adapter, no training REPL overhead
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# 3-node cluster with adapter on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

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
# After exit, open juno-<modelStem>-<timestamp>.jfr in JDK Mission Control
# Event Browser → juno.LoraTrainStep shows per-step timing breakdown:
#   forwardMs / backwardMs / optimizerMs / loss
```

**CPU performance note:** LoRA training on CPU runs the full forward+backward pass
through all transformer layers. For TinyLlama Q4_K_M on a typical 8-core machine,
expect ~2–5 seconds per gradient step for short sequences (7–10 tokens). Longer
sequences scale linearly with token count. Use `--lora-steps 5` for quick iteration
and `--lora-steps 100` when convergence matters. GPU training (via `CudaMatVecBackend`)
would be 20–50× faster.

---

### `merge` — bake a LoRA adapter into a standalone GGUF

Writes a new GGUF where the LoRA-patched projection tensors (wq/wv on every
layer) are stored as **F32** for full precision. All other tensors are copied
verbatim in their original quantised encoding. The resulting file needs no
`.lora` sidecar at inference time and loads with `./juno local` or `./juno`
(cluster) like any other model.

```bash
# Default: reads <model>.lora, writes <model>-merged.gguf
./juno merge --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Explicit paths
./juno merge --model-path /path/to/model.gguf \
             --lora-path  /adapters/my.lora   \
             --output     /path/to/merged.gguf

# Larger heap for big models (rule of thumb: 2× model file size)
./juno merge --model-path /path/to/Mistral-7B.gguf --heap 12g
```

**Why the merged file is larger than the source:**

The LoRA delta per element (~6×10⁻⁴) is smaller than Q4_K quantisation noise
(~3×10⁻³). Re-quantising the merged weights back to Q4_K would erase the training
entirely — the model would answer as if it had never been fine-tuned. F32 storage
for the 44 patched tensors is the correct trade-off. For TinyLlama 1.1B Q4_K_M
(667 MB), the merged file is approximately 1 GB.

**Full workflow:**

```bash
# 1. Fine-tune
./juno lora --model-path /models/tinyllama.gguf
#   you > /train-qa "What is your name?" A: "Juno"
#   you > /save
#   ✔ Saved → /models/tinyllama.lora

# 2. Merge (produces /models/tinyllama-merged.gguf, ~1 GB)
./juno merge --model-path /models/tinyllama.gguf

# 3. Run — no .lora file needed
./juno local --model-path /models/tinyllama-merged.gguf
#   you > what is your name?
#   bot > Juno
```

**`merge` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output path (always plain GGUF, even if source is llamafile) |
| `--heap SIZE` | `4g` | JVM heap — use at least 2× model file size |

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

# Java Flight Recording — coordinator + all node JVMs instrumented; files merged on exit
./juno --model-path /path/to/model.gguf --jfr 5m
./juno --pType tensor --model-path /path/to/model.gguf --jfr 30s
# On exit, metrics are merged across all JVMs and printed to stdout.
# Individual .jfr files (coordinator + one per node) also available for JDK Mission Control.

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

### *(no command)* — 3-node cluster

```bash
# With a pre-trained adapter applied on every node
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora

# With JFR on coordinator + all node JVMs
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora --jfr 5m
```

When `--lora-play` is given, `ConsoleMain` calls `ClusterHarness.withLoraPlay(path)` which injects `-Djuno.lora.play.path=PATH` into every forked node JVM. Each node's `EmbeddedNodeServer` reads this property when `loadShard` arrives and loads adapters before building the `ForwardPassHandler`.

---

### AWS — cluster deployment (`juno-deploy.sh`)

**Commands:**

```
./launcher.sh juno-deploy.sh setup      [options]
./launcher.sh juno-deploy.sh start
./launcher.sh juno-deploy.sh stop
./launcher.sh juno-deploy.sh teardown
./launcher.sh juno-deploy.sh status
./launcher.sh juno-deploy.sh scan-regions
```

**Setup options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--instance-type TYPE` | `g4dn.xlarge` | EC2 instance type |
| `--node-count N` | `3` | Number of inference nodes |
| `--coordinator node1\|separate` | `node1` | Co-located or separate coordinator |
| `--model-url URL` | TinyLlama Q4_K_M | Model to download during bootstrap |
| `--ptype pipeline\|tensor` | `pipeline` | Parallelism type |
| `--dtype FLOAT32\|FLOAT16` | `FLOAT16` | Activation wire format |
| `--jfr DURATION` | — | JFR on all JVMs (e.g. `5m`) |
| `--lora-play PATH` | — | Local path to a `.lora` file. **Must be an absolute path or relative to your current working directory.** The path is resolved to absolute at parse time via `realpath`. After bootstrap, the file is SCPed to `/opt/juno/models/` on every node. |

**GPU quota:** the script checks EC2 quota `L-DB2E81BA` (Running On-Demand G and VT instances) before launching. If the quota in vCPUs is less than `node-count × vCPUs-per-instance`, setup fails immediately with the shortfall and a link to the AWS Service Quotas console. It never silently reduces the node count.

**Coordinator mode — `separate`:** the coordinator is launched *before* nodes so its private IP is known at node creation time and baked into each node's `JUNO_HEALTH_URL` in `/etc/juno/node.env`. This is required for health probes to reach the coordinator. The separate coordinator runs on a `t3.medium` and starts with `-DJUNO_HEALTH=true -DJUNO_HEALTH_PORT=8081` so the health sidecar is active and `POST /health/probe` from nodes is accepted.

**CUDA on GPU instances:** CUDA drivers and toolkit are pre-installed in the golden AMI by `make-ami.sh`. Node bootstrap does not install CUDA — it only runs `lspci` to detect the GPU and sets `JUNO_USE_GPU=true`. This eliminates the 17-20 minute DKMS kernel module compilation that would otherwise block every launch.

**LoRA deploy flow:**

```bash
# Train locally
./juno lora --model-path /path/to/model.gguf
you > /train-qa What is my name? A: Dima
you > /save

# Deploy to AWS with adapter
cd scripts/aws
./launcher.sh juno-deploy.sh setup \
  --instance-type m7i-flex.large \
  --model-url https://huggingface.co/.../tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --lora-play /absolute/path/to/model.lora
```

**What `_scp_lora_to_nodes` does:**

After all nodes finish bootstrap, and **before** starting the coordinator:

1. Validates the local file exists (fails fast at `setup()` start if not found).
2. For each node — synchronously, in sequence:
   - `scp` the `.lora` file to `/tmp/<basename>` on the node
   - `/bin/sudo /bin/systemctl stop juno-node` — synchronous; waits for JVM to exit
   - `/bin/sudo /bin/mv /tmp/<file> /opt/juno/models/<file>` + `chmod 644`
   - `/bin/sudo /bin/sed -i` patches `JUNO_LORA_PLAY_PATH=` in `/etc/juno/node.env`
   - `/bin/sudo /bin/systemctl start juno-node` — synchronous; waits until gRPC port is bound (~2s)
3. Logs `[TRACE]` of the patched `node.env` line for verification.
4. Updates the global `LORA_PLAY_PATH` to the remote absolute path so `cluster-nodes.env` gets the correct value.
5. Coordinator starts only after all nodes are confirmed active.

This guarantees the coordinator's `loadShard` RPCs always find nodes with adapters loaded.

**Note:** All SSH remote commands use absolute binary paths (`/bin/sudo`, `/bin/systemctl`, `/usr/bin/tee`, etc.) or prefix `export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`. EC2 non-login SSH shells do not inherit a usable `PATH`.

**Expected coordinator log after correct deployment:**
```
INFO: LoRA inference overlay configured — nodes will load:
      /opt/juno/models/tinyllama-1.1b-chat-v1.0-q4_k_m.lora
```

**Expected node log:**
```
INFO: Detected architecture: llama  backend=CpuMatVec  file=...  lora=44 adapters
```

---

### Diagnostics and tracing

Run any `lora` command with `--verbose` to see `[TRACE]` output:

```bash
./juno lora --model-path /path/to/model.gguf --verbose
```

Key trace lines:

| Line | What it tells you |
|------|-------------------|
| `[TRACE] model type (chat template key) : tinyllama` | Whether the template matches the model |
| `[TRACE] formatted training text (repr)` | Exact token sequence sent to the model during training |
| `[TRACE] token count (excl. BOS): N` | How many tokens are in the training sequence |
| `[TRACE] step=N loss=F chunk=M/T ms=D` | Per-step loss during training (only with `--verbose`) |
| `[TRACE] inference model type: tinyllama` | Template key at inference — must match training |

If the template key at training and inference differ, the model will not recall trained facts. Rename the model file to include the architecture keyword (`tinyllama`, `llama-3`, `mistral`, `phi-3`, `gemma`) to ensure `ChatModelType.fromPath()` detects it correctly.

---

### Metrics

```bash
# Automatic in local mode
./juno local --model-path /path/to/model.gguf --jfr 5m

# Manual extraction from any .jfr
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
```

AWS cluster JFR:
```bash
./launcher.sh juno-deploy.sh setup --jfr 2m ...
# ^C → recordings collected from all nodes → metrics printed → instances stopped
```