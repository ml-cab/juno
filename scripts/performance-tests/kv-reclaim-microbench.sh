#!/usr/bin/env bash
# KV page reclaim pause microbench (paged KV, continuous schedule): how long one pass releasing
# every idle page takes on each sweep model's pools at ctx 8k / batch 8 / page 16, run on the
# decode thread (the worst placement), and the decode step's KV writes alone and with a
# reclaimer working the same pools on another thread.
# Gate: the longest pass <= 5 ms (exit code 2 when over).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-target/kv-reclaim}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/kv-reclaim-$STAMP.md"
LOG="$OUT_DIR/run-$STAMP.log"

echo "[kv-reclaim] building kvcache..."
mvn -q -pl kvcache -am package -DskipTests

CP=$(mvn -q -pl kvcache -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="kvcache/target/classes:$CP"

echo "[kv-reclaim] running -> $REPORT"
java -Xms6g -Xmx6g -XX:+UseG1GC -cp "$CP" cab.ml.juno.kvcache.KvReclaimPauseMicrobench \
  --ctx 8192 --batch 8 --page-size 16 --kv-dim 16 --reps 5 --steps 256 \
  --out "$REPORT" 2>&1 | tee "$LOG"

echo "[kv-reclaim] done: $REPORT"
