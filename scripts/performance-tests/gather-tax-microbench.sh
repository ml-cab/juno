#!/usr/bin/env bash
# Paged-KV attention microbench (block KV / continuous path): attention over dense KV against
# attention reading the pages in place; gather_ms is zero by construction.
# Gate: paged_vs_dense <= 2% at batch 8 / ctx 8k (exit code 2 when over).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-target/gather-tax}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/gather-tax-$STAMP.md"
LOG="$OUT_DIR/run-$STAMP.log"

echo "[gather-tax] building kvcache…"
mvn -q -pl kvcache -am package -DskipTests

CP=$(mvn -q -pl kvcache -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="kvcache/target/classes:$CP"

echo "[gather-tax] running matrix → $REPORT"
java -cp "$CP" cab.ml.juno.kvcache.GatherTaxMicrobench \
  --ctx 2048,8192,32768 \
  --batch 1,8,32 \
  --page-size 16,64,128 \
  --warmup 3 \
  --iters 5 \
  --out "$REPORT" | tee "$LOG"

echo "[gather-tax] done: $REPORT"
