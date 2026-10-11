#!/usr/bin/env bash
# CPU decode kernel variants: how much of a quantized matVec is the dependency
# chain of accumulating a whole row into one float.
#
# Times the shipped single-accumulator Q4_K and Q6_K decode kernels against the
# same arithmetic with 4 and 8 independent accumulators (scalar, no Vector API),
# at Mistral 7B's FFN shapes, once at the default thread count (common pool
# parallelism THREADS-1 plus the caller) and once single-threaded. Single-threaded
# isolates the compute side; multi-threaded is what decode runs.
#
# Usage: cpu-decode-kernel-variants.sh [--out DIR] [--threads N] [--reps N] [--skip-build]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="target/cpu-decode-kernel-variants"
THREADS="$(nproc)"
REPS=5
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[cpu-decode-variants] building node..."
  mvn -q -pl node -am package -DskipTests
fi
CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"

for t in "$THREADS" 1; do
  echo "[cpu-decode-variants] threads=$t"
  java --add-modules jdk.incubator.vector -Xms2g -Xmx2g -XX:+UseG1GC -XX:+AlwaysPreTouch \
    "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((t - 1))" "-Djuno.threads=$t" \
    -cp "$CP" cab.ml.juno.node.CpuDecodeKernelVariantMicrobench \
    --reps "$REPS" --out "$OUT_DIR/report-threads$t.md" \
    | tee "$OUT_DIR/run-threads$t.log"
done
echo "[cpu-decode-variants] done: $OUT_DIR"
