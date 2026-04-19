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
   - `systemctl stop juno-node` — synchronous; waits for JVM to exit
   - `sudo mv /tmp/<file> /opt/juno/models/<file>` + `chmod 644`
   - `sed -i` patches `JUNO_LORA_PLAY_PATH=` in `/etc/juno/node.env`
   - `systemctl start juno-node` — synchronous; waits until gRPC port is bound (~2s)
3. Logs `[TRACE]` of the patched `node.env` line for verification.
4. Updates the global `LORA_PLAY_PATH` to the remote absolute path so `cluster-nodes.env` gets the correct value.
5. Coordinator starts only after all nodes are confirmed active.

This guarantees the coordinator's `loadShard` RPCs always find nodes with adapters loaded.

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