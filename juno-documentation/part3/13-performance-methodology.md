(ch-13)=
# 13. Performance Methodology: Reproducing and Reading the Test Matrix

Juno publishes an interactive performance matrix (`docs/juno_test_matrix.html`) covering CPU
and GPU hardware, both parallelism strategies, and single- versus multi-session load. This
chapter covers how to reproduce a measurement, extract numbers from JFR, and read the matrix
columns; it is the companion reference to that HTML file.

## Baseline hardware

| Role | Instance | Notes |
|------|----------|-------|
| CPU | `m7i-flex.large` (AWS) | 2 vCPU, 8 GB RAM, no GPU |
| GPU | `g4dn.2xlarge` (AWS) | 8 vCPU, 32 GB RAM, NVIDIA T4 16 GB VRAM |

All runs use `tinyllama-1.1b-chat-v1.0-q4_k_m.gguf` unless stated otherwise. TPS is the
coordinator-side `juno.TokenProduced.tps` value extracted from the merged JFR recording (see
[Chapter 2](#ch-02) for what emits this event and [Chapter 4](#ch-04) for the general `--jfr`
workflow).

## Reproducing a run

```bash
mvn clean package -DskipTests

# CPU single-node, pipeline, FP16, 50 tokens
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --dtype FLOAT16 \
  --max-tokens 50 \
  --jfr 5m

# 3-node CPU cluster
./juno \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --pType pipeline --nodes 3 \
  --max-tokens 50 \
  --jfr 5m

# GPU single-node, pipeline, FP16, 200 tokens
JUNO_USE_GPU=true \
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --dtype FLOAT16 \
  --max-tokens 200 \
  --jfr 5m
```

JFR files are written as `juno-<modelStem>-<timestamp>.jfr` (local mode or the coordinator in
cluster mode) or `juno-<nodeId>-<modelStem>-<timestamp>.jfr` (cluster nodes) in the project
root; cluster runs produce one file per JVM. LoRA training uses the same programmatic `--jfr`
path (see [Chapter 8](#ch-08) for the LoRA-specific event catalog):

```bash
./juno lora --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf --jfr 1m
```

## Extracting metrics

```bash
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
cat target/metrics/metrics.json
```

The CLI maps each `juno-<modelStem>-*.jfr` file in the project root to an entry in
`metrics/src/main/resources/models.json` and writes one snapshot per matched file. After a
cluster run with `--jfr`, the launcher already prints per-file summaries on exit;
`metrics.json` reflects whichever file was processed last. For throughput, read the coordinator
recording — `juno.TokenProduced` is a coordinator-only event. For a merged percentile across
every JVM in a cluster run from Java code, call
`MetricsMain.extractToJsonMerged(List<Path>, modelStem, modelFilename)`.

| JFR event | Field | Matrix column |
|-----------|-------|---------------|
| `juno.TokenProduced` | `tps` | TPS value |
| `juno.ForwardPass` | `durationMs` p95 | Node decode p95 |
| `juno.ForwardPass` | `prefillMs` p95 | Node prefill p95 |
| `juno.MatVec` | `durationMs` p99 | MatVec hot-path overhead |

## Concurrent session tests (`s9`)

The `s9` columns measure aggregate TPS across 9 simultaneous sessions.

```bash
./juno test \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --jfr 5m
```

`./juno test` runs 6 pipeline and 2 tensor smoke checks and exits 0 when all pass (see
[Chapter 3](#ch-03)). For a raw 9-session load against the REST API:

```bash
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --api-port 8080 \
  --jfr 5m &

for i in $(seq 1 9); do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"tinyllama","messages":[{"role":"user","content":"count to 50"}],"max_tokens":50}' &
done
wait
```

## Matrix column definitions

| Column | Meaning |
|--------|---------|
| `hw` | `cpu` or `gpu` |
| `pt` | Parallelism type: `pipeline` or `tensor` |
| `n` | Number of transformer nodes |
| `co` | Coordinator placement: `embedded` (same JVM as node-1) or `separate` |
| `dt` | Activation wire dtype: `FP16`, `FP32`, or `INT8` |
| `bo` | Byte order: `BE` or `LE` |
| `lo` | LoRA adapter overlay: `off` or adapter rank |
| `l1` | Long-form / single session TPS |
| `l9` | Long-form / 9 concurrent sessions aggregate TPS |
| `c1` | Conversational (growing KV context) / single session TPS |
| `c9` | Conversational / 9 concurrent sessions aggregate TPS |

Cell status codes in `scripts/performance-tests/matrix.tsv` (prefix before `:`):

| Code | Meaning |
|------|---------|
| `D` | Done — TPS measured (value follows `:`) |
| `P` | Pending — planned, not yet run |
| `A` | Added — suggested extra cell |
| `NA` | Not applicable for this row |

HTTP prompts, session counts, and token limits are defined in
`scripts/performance-tests/scenarios.yaml`.

## AWS performance runner

`scripts/performance-tests/matrix.tsv` is the single source of truth for which configurations
exist and what has been measured. `scripts/performance-tests/performance-test.sh` selects cells
directly from that file — there is no separate queue file. After each successful cell, it writes
the coordinator's `juno.TokenProduced.tps` into the matrix and regenerates
`docs/juno_test_matrix.html`.

**Per-cell lifecycle:** each selected cell (`l1`, `l9`, `c1`, `c9`) runs one full AWS cycle —
`juno-deploy.sh setup --detach --no-browser` (exits once the coordinator is healthy), an HTTP
workload against `POST /v1/chat/completions` driven by `scenarios.yaml`, `juno-deploy.sh finish`
(JFR gather plus cluster teardown, see [Chapter 7](#ch-07)), metrics JSON written to
`target/perf/runs/metrics-<row>-<col>.json`, and an update to both `matrix.tsv` and
`juno_test_matrix.html`.

| Command | Description |
|---------|-------------|
| `./scripts/performance-tests/performance-test.sh` | Screen worker: run selection in background (`juno-perf` session) |
| `./scripts/performance-tests/performance-test.sh --foreground` | Same worker, logs to terminal |
| `./scripts/performance-tests/performance-test.sh --attach` | Attach to the running screen session |
| `./scripts/performance-tests/performance-test.sh --status` | Screen session status plus tail of `target/perf/nohup.log` |
| `./scripts/performance-tests/performance-test.sh --list` | Print selected `row_id` and column, then exit |
| `./scripts/performance-tests/performance-test.sh --parse` | Parse `test-scenario.txt` into matrix + HTML |

**Selection flags** (source: `scripts/performance-tests/matrix.tsv`, override with `--matrix
FILE`):

| Flag | Description |
|------|-------------|
| `--all` | Every applicable cell (not `NA`), including already-measured (`D:`) cells |
| `--pending` | Only `P:` or `A:` cells — the default when no selection flag is given |
| `--row ID` | Limit to one matrix row id |
| `--col COL` | Limit to one column: `l1`, `l9`, `c1`, `c9` |
| `--from ID` / `--to ID` | Inclusive row id range |

Setting `--row`, `--col`, or `--from`/`--to` alone defaults the run mode to `all` for matching
non-`NA` cells, so a specific cell can be re-measured without also passing `--all`. Combine with
`--pending` to restrict a range to unfinished cells only.

**Other flags:**

| Flag | Description |
|------|-------------|
| `--git REF` | Branch, tag, or commit for `juno-deploy.sh` on EC2 (default `main`) |
| `--scenario FILE` | Input for `--parse` (default `test-scenario.txt`) |
| `--html FILE` | HTML output path (default `docs/juno_test_matrix.html`) |
| `-n`, `--dry-run` | With `--parse`: preview HTML rows without writing |

```bash
# Preview every non-NA cell
./scripts/performance-tests/performance-test.sh --list --all

# Run every applicable cell
./scripts/performance-tests/performance-test.sh --foreground --all --git perftest

# Run only unfinished cells (default mode)
./scripts/performance-tests/performance-test.sh --foreground --git perftest

# One cell -- GPU pipeline, long-form, single session (row 16)
./scripts/performance-tests/performance-test.sh --foreground --row 16 --col l1 --git perftest

# Inclusive row range, all columns per row
./scripts/performance-tests/performance-test.sh --foreground --from 15 --to 16 --git perftest

# Background worker for long runs, then attach
./scripts/performance-tests/performance-test.sh --all --git perftest
./scripts/performance-tests/performance-test.sh --attach
```

**Artifacts:**

| Path | Content |
|------|---------|
| `target/perf/nohup.log` | Worker log (screen mode) |
| `target/perf/runs/deploy-<row>-<col>.log` | Deploy and JFR console output |
| `target/perf/runs/http-<row>-<col>/` | Chat completion JSON responses |
| `target/perf/runs/metrics-<row>-<col>.json` | Merged JFR metrics |
| `scripts/performance-tests/matrix.tsv` | Updated TPS per cell after each run |

## Submitting results

Send a metrics summary to [dev@ml.cab](mailto:dev@ml.cab) including GPU card details, the exact
`./juno` startup command, the conversation log, and the JFR metrics summary section — in
particular `juno.TokenProduced.tps` and `juno.ForwardPass` p95 decode latency. To regenerate the
matrix from a captured scenario log manually:

```bash
./scripts/performance-tests/performance-test.sh --parse
# reads test-scenario.txt, writes docs/juno_test_matrix.html and scripts/performance-tests/matrix.tsv
```

Automated AWS runs update the matrix and HTML after each cell; `--parse` is only needed when
ingesting pasted JFR output manually.

## LoRA training GPU baseline

Current status: GPU instrumentation and resident-transpose primitives are in place, but the
present hybrid path — GPU-resident frozen forward pass paired with CPU quantized-transpose
backward pass — is not a fully GPU-resident training loop, and its throughput should not be
read as representative of production GPU LoRA training (see [Chapter 8](#ch-08) for the
training-loop internals this affects).

Reference configuration for measuring this path:

| Item | Value |
|------|-------|
| Model | TinyLlama Q4_K_M |
| Sequence length | 64 / 128 |
| Rank | 8 |
| Targets | `qv` and `all-linear` |
| Warm-up / measured updates | 10 / ≥ 20 |
| Hardware | NVIDIA (`g4dn`) and AMD reference instances |

Per-path metrics worth recording: tokens/s; `forwardMs`, `frozenForwardMs`,
`attentionNonlinearMs`; `backwardMs`, `frozenTransposeBackwardMs`, `adapterBackwardMs`,
`transferMs`; `optimizerMs`; H2D/D2H byte counts where transfer counters are wired; peak heap;
peak VRAM.

JFR labels covering the resident-transpose primitives: `cuda-resident-transpose` /
`cuda-resident-fp16-transpose`, `rocm-resident-transpose` / `rocm-resident-fp16-transpose`. The
adjoint identity `dot(W·x, g) == dot(x, Wᵀ·g)` is checked by `CudaMatVecTransposeTest` and
`RocmMatVecTransposeTest` (`-Dgroups=gpu` / `-Dgroups=rocm`).

---

[← Chapter 12: Phi-3 Inference Internals](#ch-12) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 14: Governance →](#ch-14)
