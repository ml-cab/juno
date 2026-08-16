(ch-3-2)=
# 3.2. Flags

## Global flags

| Flag | Default | Commands | Description |
|------|---------|----------|-------------|
| `--model-path PATH` | (none) | all | Path to GGUF file (required) |
| `--dtype FLOAT32\|FLOAT16\|INT8` | `FLOAT16` | cluster, local | Activation wire format |
| `--byteOrder BE\|LE` | `BE` | cluster | Activation byte order. Must match across all JVMs; propagated automatically by `ClusterHarness` and `juno-deploy.sh`. |
| `--max-tokens N` | `200` | cluster, local, lora | Maximum tokens per response. Same default as the REST API and `SamplingParams.defaults()`. |
| `--temperature F` | `0.7` | all | Sampling temperature (0.0 = deterministic) |
| `--top-k N` | `50` | all | Top-K sampling cutoff (0 = disabled) |
| `--top-p F` | `0.9` | all | Nucleus sampling cutoff (0 = disabled). Same default as the REST API and `SamplingParams.defaults()`. |
| `--heap SIZE` | `4g` | all | JVM heap per node, for example `4g`, `8g` |
| `--nodes N` | `3` | local | Number of in-process shards |
| `--pType pipeline\|tensor` | `pipeline` | cluster, test | Parallelism type |
| `--jfr DURATION` | (none) | cluster, local, lora | Java Flight Recording, for example `30s`, `5m` |
| `--verbose` / `-v` | (none) | cluster, local, lora | Full logging; LoRA default is a progress bar |
| `--cpu` | (none) | cluster, local | Force CPU inference: sets `JUNO_USE_GPU=false`. Does not enable LoRA mode. |
| `--lora-play PATH` | (none) | cluster, local | Apply a pre-trained `.lora` adapter at inference (read-only, no training). In cluster mode the file is forwarded as `-Djuno.lora.play.path` to every forked node JVM. |
| `--api-port N` | (none) | cluster, local | Start the OpenAI-compatible REST API server on port N alongside the REPL. Exposes `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/models/{model}`. Environment override: `API_PORT`. |

## LoRA-specific flags (`lora` command only)

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-path PATH` | `<model>.lora` | Adapter checkpoint (auto-loaded if it exists) |
| `--lora-rank N` | `8` | Low-rank bottleneck dimension |
| `--lora-alpha F` | equal to rank | Declared alpha (standard scale = alpha/rank; rsLoRA = alpha/sqrt(rank)) |
| `--lora-mode` | `lora` | `lora` or `dora` |
| `--lora-scaling` | `standard` | `standard` or `rslora` |
| `--lora-init` | `kaiming-uniform` | `kaiming-uniform` or `legacy-normal` |
| `--lora-lr F` | `1e-4` | Peak / base AdamW learning rate |
| `--lora-max-iters N` | `50` | Max training passes per `/train`, `/train-qa`, or `/train-file-qa` (safety cap) |
| `--lora-loss-target-text F` | `1.8` | Stop `/train` when loss is at or below F |
| `--lora-loss-target-qa F` | `1.2` | Stop `/train-qa` / `/train-file-qa` when loss is at or below F |
| `--lora-steps N` | (none) | Alias for `--lora-max-iters` (`/train` cap) |
| `--lora-steps-qa N` | `50` | Max passes for `/train-qa` / `/train-file-qa` |
| `--lora-early-stop F` | `0.25` | Overfit guard: stop when loss is below F (set 0 to disable) |
| `--lora-targets SPEC` | `qv` | `qv`, `all` / `all-linear`, or comma-separated keys (`wq,wk,wv,wo,wgate,wup,wdown`) |
| `--lora-gradient-accumulation N` | `1` | Chunks accumulated per optimizer update (token-weighted) |
| `--lora-max-grad-norm F` | `1.0` | Global L2 clip after token normalization; `0` disables clipping |
| `--lora-lr-schedule M` | `constant` | `constant` or `cosine` (warmup then cosine decay) |
| `--lora-warmup-steps N` | `0` | Warmup optimizer updates for the cosine schedule |
| `--lora-min-lr F` | `0` | Cosine floor learning rate |
| `--lora-weight-decay F` | `0.01` | Decoupled AdamW decay on A only |
| `--lora-plus-ratio F` | `1.0` | B/A learning-rate ratio (`1.0` is ordinary LoRA) |
| `--lora-dropout F` | `0` | Train-only inverted dropout on the LoRA branch input |
| `--lora-seed N` | `42` | Seed for init, validation split, dropout masks, and corpus caps |
| `--lora-validation-split F` | `0` | Fraction of units held out (`0` disables) |
| `--lora-validation-patience N` | `0` | Validation checks without improvement before stop |
| `--lora-validation-min-delta F` | `0` | Minimum validation improvement to reset patience |
| `--lora-chunk-tokens N` | `32` | Truncated-BPTT window size; use `128` for large `/train-file` runs |
| `--lora-max-train-tokens N` | `0` | Cap on supervised prediction tokens per train (`0` is unlimited); seeded whole-chunk subsample |

## `merge`-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | (none) | Source GGUF or llamafile (required) |
| `--lora-path PATH` | `<model>.lora` | Trained adapter checkpoint |
| `--output PATH` | `<model>-merged.gguf` | Output file (always plain GGUF, even if source is a llamafile) |
| `--heap SIZE` | `4g` | JVM heap; use at least 2x the model file size |

## Environment variable overrides

`MODEL_PATH`, `JUNO_USE_GPU`, `PTYPE`, `DTYPE`, `BYTE_ORDER`, `MAX_TOKENS`, `TEMPERATURE`,
`TOP_K`, `TOP_P`, `HEAP`, `NODES`, `JAVA_HOME`, `LORA_PATH`, `LORA_RANK`, `LORA_ALPHA`,
`LORA_LR`, `LORA_MAX_ITERS`, `LORA_LOSS_TARGET_TEXT`, `LORA_LOSS_TARGET_QA`, `LORA_STEPS`
(alias), `LORA_PLAY_PATH`, `LORA_TARGETS`, `LORA_GRADIENT_ACCUMULATION`, `LORA_MAX_GRAD_NORM`,
`LORA_CHUNK_TOKENS`, `LORA_MAX_TRAIN_TOKENS`, `LORA_TRAIN_DEVICE`, `LORA_MICROBATCH`,
`API_PORT`.

## GPU backend selection for LoRA

For the `lora` command and `ForwardPassHandlerLoader.selectLoraBackend()`, an unset
`JUNO_USE_GPU` means Juno tries GPU (CUDA first, then ROCm) when available. Set
`JUNO_USE_GPU=false` or pass `--cpu` to force CPU under `--lora-train-device=auto` (the
default). Use `--lora-train-device=gpu` to fail closed when CUDA/ROCm is unavailable, or
`--lora-train-device=cpu` to force CPU MatVec for LoRA regardless of `--gpu`.

With GPU LoRA, the default `--lora-microbatch 8` (`LORA_MICROBATCH`) uses FP32 resident GEMM
for frozen linears; set `1` for sequential GEMV / FP16 residency, or let VRAM out-of-memory
auto-retry drop to 1. Phi-3.5 on roughly 8 GB cards should use `--lora-microbatch 1` rather than
`JAVA_TOOL_OPTIONS=-Djuno.lora.microbatch=1`.

Cluster and `local` modes use `selectBackend()`, where unset defaults to CPU for safety.
Override the vendor with `-Djuno.gpu.backend=cuda|rocm|auto` (default: `auto`).

## See also

- [Chapter 3.1 -- Commands](#ch-3-1)
- [Chapter 3.5 -- LoRA Mode](#ch-3-5)
- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)

---

[<- 3.1 Commands](#ch-3-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.3 Local Mode ->](#ch-3-3)
