#!/usr/bin/env bash
# Gather-tax microbench (block KV / continuous path).
# See docs/infra-plan/PLAN-Infra-Tier14.md — budget ≤ ~15% gather at batch 8 / ctx 8k.
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
