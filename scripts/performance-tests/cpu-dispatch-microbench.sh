#!/usr/bin/env bash
# CPU dispatch: the integer CPU kernels' row loops on common-pool parallel
# streams against a persistent worker pool, alternating on identical work, at
# decode width (single matmuls and one decode layer) and prefill width.
# Thread counts are matched: common pool parallelism THREADS-1 plus the caller,
# against a worker pool of THREADS threads (caller included).
#
# Usage: cpu-dispatch-microbench.sh [--out DIR] [--threads N] [--reps N] [--skip-build]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="target/cpu-dispatch-microbench"
THREADS="$(nproc)"
REPS=5
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT_DIR"
if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[cpu-dispatch] building node..."
  mvn -q -pl node -am package -DskipTests
fi
CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"
java --add-modules jdk.incubator.vector -Xms3g -Xmx3g -XX:+UseG1GC -XX:+AlwaysPreTouch \
  "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((THREADS - 1))" "-Djuno.threads=$THREADS" \
  -cp "$CP" cab.ml.juno.node.CpuDispatchMicrobench --reps "$REPS" --out "$OUT_DIR/report-threads$THREADS.md" \
  | tee "$OUT_DIR/run-threads$THREADS.log"
echo "[cpu-dispatch] done: $OUT_DIR"
