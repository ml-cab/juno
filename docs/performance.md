# Juno performance — methodology and reproduction

Companion to [juno_test_matrix.html](juno_test_matrix.html). That file contains the
interactive results table and scenario narratives; this document covers how to reproduce
a run, extract numbers from JFR, and interpret the matrix columns.

---

## Baseline hardware

| Role | Instance | Notes |
|------|----------|-------|
| CPU | `m7i-flex.large` (AWS) | 2 vCPU, 8 GB RAM; no GPU |
| GPU | `g4dn.2xlarge` (AWS) | 8 vCPU, 32 GB RAM; NVIDIA T4 16 GB VRAM |

All runs use `tinyllama-1.1b-chat-v1.0-q4_k_m.gguf` unless stated otherwise. TPS is
coordinator-side `juno.TokenProduced.tps` extracted from the merged JFR file.

---

## Reproducing a run

### 1. Build

```bash
mvn clean package -DskipTests
```

### 2. Run with JFR enabled

```bash
# CPU single-node, pipeline, FP16, 50 tokens — matches matrix row id:1
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --dtype FLOAT16 \
  --max-tokens 50 \
  --jfr 5m

# 3-node CPU cluster — matches matrix row id:3
./juno \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --pType pipeline --nodes 3 \
  --max-tokens 50 \
  --jfr 5m

# GPU single-node, pipeline, FP16, 200 tokens — matches matrix row id:16
JUNO_USE_GPU=true \
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --dtype FLOAT16 \
  --max-tokens 200 \
  --jfr 5m
```

JFR files are written as `juno-<modelStem>-<timestamp>.jfr` in the project root.
Cluster runs produce one file per JVM; pass all of them to the extractor.

### 3. Extract metrics

```bash
# Merge and extract to target/metrics/metrics.json
java -cp juno-player/target/juno-player-*-shaded.jar \
  cab.ml.juno.metrics.MetricsMain \
  --merge \
  juno-*.jfr

cat target/metrics/metrics.json
```

Key fields to record:

| JFR event | Field | Matrix column |
|-----------|-------|---------------|
| `juno.TokenProduced` | `tps` | TPS value |
| `juno.ForwardPass` | `durationMs` p95 | Node decode p95 |
| `juno.ForwardPass` | `prefillMs` p95 | Node prefill p95 |
| `juno.MatVec` | `durationMs` p99 | MatVec hot-path overhead |

---

## Concurrent session tests (s9)

The `s9` columns measure aggregate TPS across 9 simultaneous sessions. Reproduce with
`ClusterHarness` or the `test` command:

```bash
./juno test \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --jfr 5m
```

The `test` command runs 6 pipeline and 2 tensor smoke checks and exits 0 on all-pass.
For a raw s9 load, open 9 concurrent REST connections to `POST /v1/chat/completions`
with `--api-port 8080` active:

```bash
./juno local \
  --model-path models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --api-port 8080 \
  --jfr 5m &

# in a separate shell, send 9 concurrent requests
for i in $(seq 1 9); do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"tinyllama","messages":[{"role":"user","content":"count to 50"}],"max_tokens":50}' &
done
wait
```

---

## Matrix column definitions

| Column | Meaning |
|--------|---------|
| `hw` | `cpu` or `gpu` |
| `pt` | Parallelism type: `pipeline` or `tensor` |
| `n` | Number of transformer nodes |
| `co` | Coordinator placement: `embedded` (same JVM as node-1) or `separate` |
| `dt` | Activation wire dtype: `FP16`, `FP32`, or `INT8` |
| `bo` | Byte order: `BE` (big-endian) or `LE` (little-endian) |
| `lo` | LoRA adapter overlay: `off` or adapter rank |
| `l1` | Long-form / single session TPS |
| `l9` | Long-form / 9 concurrent sessions aggregate TPS |
| `c1` | Conversational (growing KV context) / single session TPS |
| `c9` | Conversational / 9 concurrent sessions aggregate TPS |

Cell status in `scripts/performance-tests/matrix.tsv` (prefix before `:`):

| Code | Meaning |
|------|---------|
| `D` | Done — TPS measured (value after `:`) |
| `P` | Pending — planned, not yet run |
| `A` | Added — suggested extra cell |
| `NA` | Not applicable for this row |

HTTP prompts, session counts, and token limits come from [scenarios.yaml](../scripts/performance-tests/scenarios.yaml).

---

## AWS performance runner (`scripts/performance-tests/performance-test.sh`)

`scripts/performance-tests/matrix.tsv` is the **single source of truth** for which configurations exist and what has been measured. The runner selects cells directly from that file (no separate queue file). After each successful cell it writes coordinator `juno.TokenProduced.tps` into the matrix and regenerates [juno_test_matrix.html](juno_test_matrix.html).

### Per-cell lifecycle

Each selected cell (`l1`, `l9`, `c1`, `c9`) runs one full AWS cycle:

1. `juno-deploy.sh setup --detach --no-browser` (exits after coordinator healthy)
2. HTTP workload via `POST /v1/chat/completions` (from `scripts/performance-tests/scenarios.yaml`)
3. `juno-deploy.sh finish` — JFR gather + cluster teardown
4. Metrics JSON → `target/perf/runs/metrics-<row>-<col>.json`
5. Update `scripts/performance-tests/matrix.tsv` and `docs/juno_test_matrix.html`; open matrix in browser

### Commands

| Command | Description |
|---------|-------------|
| `./scripts/performance-tests/performance-test.sh` | Screen worker: run selection in background (`juno-perf` session) |
| `./scripts/performance-tests/performance-test.sh --foreground` | Same worker, log to terminal |
| `./scripts/performance-tests/performance-test.sh --attach` | Attach to screen session |
| `./scripts/performance-tests/performance-test.sh --status` | Screen session + tail `target/perf/nohup.log` |
| `./scripts/performance-tests/performance-test.sh --list` | Print selected `row_id` + column, then exit |
| `./scripts/performance-tests/performance-test.sh --parse` | Parse `test-scenario.txt` → matrix + HTML |

### Selection flags

All selection is from `scripts/performance-tests/matrix.tsv` (override path with `--matrix FILE`).

| Flag | Description |
|------|-------------|
| `--all` | Every applicable cell (not `NA`), including already measured (`D:`) |
| `--pending` | Only `P:` or `A:` cells |
| `--row ID` | Limit to one matrix row id |
| `--col COL` | Limit to one column: `l1`, `l9`, `c1`, `c9` |
| `--from ID` | Inclusive row id range start (use with `--to`) |
| `--to ID` | Inclusive row id range end |

**Default mode** when no selection flags are given: `--pending` (unfinished cells only).

If you set **`--row`**, **`--col`**, or **`--from` / `--to`**, mode defaults to **`all`** for matching non-`NA` cells so you can re-run measured cells without also passing `--all`. Use `--pending` with a range to restrict to unfinished cells only.

### Other options

| Flag | Description |
|------|-------------|
| `--git REF` | Branch, tag, or commit for `juno-deploy.sh` on EC2 (default: `main`) |
| `--scenario FILE` | Input for `--parse` (default: `test-scenario.txt`) |
| `--html FILE` | HTML output (default: `docs/juno_test_matrix.html`) |
| `-n`, `--dry-run` | `--parse` only: preview HTML rows, do not write |
| `-h`, `--help` | Full usage |

`--queue` was removed; use matrix selection flags instead.

### Examples

```bash
# Preview: all non-NA cells in the matrix
./scripts/performance-tests/performance-test.sh --list --all

# Run every applicable cell (23 rows × up to 4 columns)
./scripts/performance-tests/performance-test.sh --foreground --all --git perftest

# Run only unfinished cells (default mode)
./scripts/performance-tests/performance-test.sh --foreground --git perftest

# One cell — GPU pipeline long/s1 (row 16)
./scripts/performance-tests/performance-test.sh --foreground --row 16 --col l1 --git perftest

# Inclusive row range, all columns per row
./scripts/performance-tests/performance-test.sh --foreground --from 15 --to 16 --git perftest

# Same range but only pending/suggested cells
./scripts/performance-tests/performance-test.sh --foreground --from 15 --to 23 --pending --git perftest

# One row, one scenario column
./scripts/performance-tests/performance-test.sh --foreground --row 16 --col l9 --git perftest

# Background worker (long runs)
./scripts/performance-tests/performance-test.sh --all --git perftest
./scripts/performance-tests/performance-test.sh --attach
```

### Artifacts

| Path | Content |
|------|---------|
| `target/perf/nohup.log` | Worker log (screen mode) |
| `target/perf/runs/deploy-<row>-<col>.log` | Deploy + JFR console |
| `target/perf/runs/http-<row>-<col>/` | Chat completion JSON responses |
| `target/perf/runs/metrics-<row>-<col>.json` | Merged JFR metrics |
| `scripts/performance-tests/matrix.tsv` | Updated TPS per cell after each run |

---

## Submitting results

Send a Metrics summary to [dev@ml.cab](mailto:dev@ml.cab) with: GPU card details,
juno startup command, conversation log, and the JFR Metrics Summary section. Include
the `juno.TokenProduced.tps` value and `juno.ForwardPass` p95 decode latency.

Regenerate the matrix from a captured scenario log (manual / legacy path):

```bash
./scripts/performance-tests/performance-test.sh --parse
# reads test-scenario.txt, writes docs/juno_test_matrix.html and scripts/performance-tests/matrix.tsv
```

Automated AWS runs update the matrix and HTML after each cell; `--parse` is only needed when ingesting pasted JFR output into `test-scenario.txt`.