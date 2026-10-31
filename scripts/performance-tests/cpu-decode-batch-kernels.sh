#!/usr/bin/env bash
# CPU decode-batch matmuls: what a decode batch of 1, 2, 4 and 8 rows costs per
# weight matrix in the cache-blocked prefill matmul, in the decode-batch matmul
# (each row bit for bit the single-row kernel's), and as one single-row call per
# row, at TinyLlama 1.1B and Qwen2.5-3B shapes, on the default thread count.
#
# Usage: cpu-decode-batch-kernels.sh [--out DIR] [--threads N] [--reps N] [--skip-build] [-- MICROBENCH ARGS]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="target/cpu-decode-batch-kernels"
THREADS="$(nproc)"
REPS=5
SKIP_BUILD=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[cpu-decode-batch] building node..."
  mvn -q -pl node -am package -DskipTests
fi
CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"

echo "[cpu-decode-batch] threads=$THREADS"
java --add-modules jdk.incubator.vector -Xms2g -Xmx2g -XX:+UseG1GC -XX:+AlwaysPreTouch \
  "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((THREADS - 1))" "-Djuno.threads=$THREADS" \
  -cp "$CP" cab.ml.juno.node.CpuDecodeBatchMicrobench \
  --reps "$REPS" --out "$OUT_DIR/report-threads$THREADS.md" "${EXTRA[@]}" \
  | tee "$OUT_DIR/run-threads$THREADS.log"
echo "[cpu-decode-batch] done: $OUT_DIR"
