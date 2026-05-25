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

Status codes: `D` = done (measured), `A` = awaiting measurement, `NA` = not applicable
for this configuration.

---

## Submitting results

Send a Metrics summary to [dev@ml.cab](mailto:dev@ml.cab) with: GPU card details,
juno startup command, conversation log, and the JFR Metrics Summary section. Include
the `juno.TokenProduced.tps` value and `juno.ForwardPass` p95 decode latency.
Results are incorporated into [juno_test_matrix.html](juno_test_matrix.html).