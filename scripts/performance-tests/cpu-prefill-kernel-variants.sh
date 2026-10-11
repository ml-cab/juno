#!/usr/bin/env bash
# CPU prefill kernel variants: the shipped float weight-stationary matmuls
# against split-accumulator float, scalar integer and cache-blocked Vector API
# integer matmuls, at prefill batch widths (default 128 and 741, the vision
# prefill width), once at the default thread count (common pool parallelism
# THREADS-1 plus the caller).
#
# Usage: cpu-prefill-kernel-variants.sh [--out DIR] [--threads N] [--reps N]
#                                       [--batch LIST] [--types LIST] [--skip-build]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="target/cpu-prefill-kernel-variants"
THREADS="$(nproc)"
REPS=3
BATCH="128,741"
TYPES="12,13,14"
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --types) TYPES="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[cpu-prefill-variants] building node..."
  mvn -q -pl node -am package -DskipTests
fi
CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"

echo "[cpu-prefill-variants] threads=$THREADS"
java --add-modules jdk.incubator.vector -Xms6g -Xmx6g -XX:+UseG1GC -XX:+AlwaysPreTouch \
  "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((THREADS - 1))" "-Djuno.threads=$THREADS" \
  -cp "$CP" cab.ml.juno.node.CpuPrefillKernelVariantMicrobench \
  --reps "$REPS" --batch "$BATCH" --types "$TYPES" --out "$OUT_DIR/report-threads$THREADS.md" \
  | tee "$OUT_DIR/run-threads$THREADS.log"
echo "[cpu-prefill-variants] done: $OUT_DIR"
