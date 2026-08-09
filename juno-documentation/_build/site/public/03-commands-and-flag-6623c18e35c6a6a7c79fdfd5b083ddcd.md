(ch-03)=
# 3. Commands and Flags: The Complete CLI Reference

**Linux / macOS:**
```
./juno
```

**Windows:**
```
juno.bat
```

Unified stand-alone launchers live at the project root. `juno.bat` delegates to
`scripts\run.bat`. Both require JDK 25+ and pre-built jars (`mvn clean package -DskipTests`).

> **Windows note:** every example in this book uses `./juno`. Replace with `juno.bat` on
> Windows and use backslashes for paths (e.g. `--model-path models\model.gguf`). All flags,
> environment variables, and subcommands are identical across platforms.

## Commands

| Command | Description |
|---------|-------------|
| `cluster` | 3-node cluster (default command) — forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `local` | In-process REPL — all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL — single in-process JVM, adapter persisted to `.lora` file |
| `merge` | Bake a trained `.lora` adapter into a new standalone GGUF — no sidecar needed at inference time |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor), exits 0 (all pass) or 1 (any fail) |

Usage examples for each command are in [Chapter 4](#ch-04).

## General flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | — | all | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format |
| `--byteOrder BE\|LE` | `BE` | cluster | Activation byte order. Must match across all JVMs — propagated automatically by `ClusterHarness` and `juno-deploy.sh`. |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response. Same default as REST API and `SamplingParams.defaults()`. |
| `--temperature F` | `0.7` | all | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `50` | all | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.9` | all | Nucleus sampling cutoff (0 = disabled). Same default as REST API and `SamplingParams.defaults()`. |
| `--heap SIZE` | `4g` | all | JVM heap per node, e.g. `4g`, `8g` |
| `--nodes N` | `3` | local | Number of in-process shards |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type (see [Chapter 2](#ch-02)) |
| `--jfr DURATION` | — | cluster, local, lora | Java Flight Recording (e.g. `30s`, `5m`) |
| `--verbose` / `-v` | — | cluster, local, lora | Full logging; LoRA default is a progress bar |
| `--cpu` | — | cluster, local | Force CPU inference: sets `JUNO_USE_GPU=false`. Does not enable LoRA mode. |
| `--lora-play PATH` | — | cluster, local | Apply a pre-trained `.lora` adapter at inference (read-only, no training). In cluster mode the file is forwarded as `-Djuno.lora.play.path` to every forked node JVM. |
| `--api-port N` | — | cluster, local | Start the OpenAI-compatible REST API server on port N alongside the REPL. See [Chapter 5](#ch-05). Environment override: `API_PORT`. |

## LoRA-specific flags (`lora` command only)

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint (auto-loaded if exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | `= rank` | Declared α (standard scale = α/rank; rsLoRA = α/√rank) |
| `--lora-mode` | `lora` | `lora` or `dora` |
| `--lora-scaling` | `standard` | `standard` or `rslora` |
| `--lora-init` | `kaiming-uniform` | `kaiming-uniform` or `legacy-normal` |
| `--lora-lr F` | `1e-4` | Peak / base AdamW learning rate |
| `--lora-max-iters N` | `50` | Max training passes per `/train` or `/train-qa` (safety cap) |
| `--lora-loss-target-text F` | `1.8` | Stop `/train` when loss ≤ F |
| `--lora-loss-target-qa F` | `1.2` | Stop `/train-qa` when loss ≤ F |
| `--lora-steps N` | — | Alias for `--lora-max-iters` (`/train` cap) |
| `--lora-steps-qa N` | `50` | Max passes for `/train-qa` |
| `--lora-early-stop F` | `0.25` | Overfit guard: stop when loss < F (set 0 to disable) |
| `--lora-targets SPEC` | `qv` | `qv`, `all` / `all-linear`, or comma keys (`wq,wk,wv,wo,wgate,wup,wdown`) |
| `--lora-gradient-accumulation N` | `1` | Chunks accumulated per optimizer update (token-weighted) |
| `--lora-max-grad-norm F` | `1.0` | Global L2 clip after token normalization; `0` disables clipping |
| `--lora-lr-schedule M` | `constant` | `constant` or `cosine` (warmup then cosine decay) |
| `--lora-warmup-steps N` | `0` | Warmup optimizer updates for cosine schedule |
| `--lora-min-lr F` | `0` | Cosine floor learning rate |
| `--lora-weight-decay F` | `0.01` | Decoupled AdamW decay on A only |
| `--lora-plus-ratio F` | `1.0` | B/A learning-rate ratio (`1.0` = ordinary LoRA) |
| `--lora-dropout F` | `0` | Train-only inverted dropout on LoRA branch input |
| `--lora-seed N` | `42` | Seed for init, validation split, and dropout masks |
| `--lora-validation-split F` | `0` | Fraction of units held out (`0` disables) |
| `--lora-validation-patience N` | `0` | Validation checks without improvement before stop |
| `--lora-validation-min-delta F` | `0` | Minimum validation improvement to reset patience |

Full explanation of each hyperparameter's role in training is in [Chapter 8](#ch-08) and
[Chapter 9](#ch-09).

## `merge` specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | — | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output file (always plain GGUF, even if source is llamafile) |
| `--heap SIZE` | `4g` | JVM heap — use at least 2x the model file size |

## Environment overrides

`MODEL_PATH`, `JUNO_USE_GPU`, `PTYPE`, `DTYPE`, `BYTE_ORDER`,
`MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`,
`LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`, `LORA_LR`, `LORA_MAX_ITERS`, `LORA_LOSS_TARGET_TEXT`,
`LORA_LOSS_TARGET_QA`, `LORA_STEPS` (alias), `LORA_PLAY_PATH`, `LORA_TARGETS`,
`LORA_GRADIENT_ACCUMULATION`, `LORA_MAX_GRAD_NORM`, `API_PORT`

For the `lora` command and `ForwardPassHandlerLoader.selectLoraBackend()`, `JUNO_USE_GPU` unset
means try GPU (CUDA first, then ROCm) when available; set `JUNO_USE_GPU=false` or pass `--cpu`
to force CPU. Cluster and `local` modes use `selectBackend()`, where unset defaults to CPU for
safety. Override the vendor with `-Djuno.gpu.backend=cuda|rocm|auto` (default: `auto`).

## Build and test

Requires JDK 25+ and Maven 3.9+.

```bash
mvn clean package -DskipTests          # build — juno-player emits thin jar + *-shaded.jar runnable

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests — no model file, no GPU needed

mvn verify -pl juno-master             # integration tests — forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT and TensorParallelClusterIT

mvn verify -pl juno-master -Pintegration -Dmodels=/path/to/models
                                       # ModelLiveRunnerIT — requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test (8 checks, exits 0/1)
```

**Windows (Command Prompt):**
```bat
mvn clean package -DskipTests

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player

mvn verify -pl juno-master

mvn verify -pl juno-master -Pintegration -Dmodels=C:\models

juno.bat test --model-path models\model.gguf
```

**GPU tests** (NVIDIA — requires CUDA 12.x and an NVIDIA GPU):

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl juno-master \
  --enable-native-access=ALL-UNNAMED
```

**GPU tests** (AMD — requires ROCm 6+ and an AMD GPU):

```bash
mvn test -Dgroups=rocm -pl node --enable-native-access=ALL-UNNAMED
```

> ROCm is Linux-only; AMD GPU tests are not supported on Windows.

---

[← Chapter 2: Architecture Reference](#ch-02) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 4: Running Modes →](#ch-04)
