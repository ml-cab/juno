#!/usr/bin/env bash
# Resident RMS-norm + RoPE chain microbench (GPU activation residency).
#
# Times a two-operation chain - RMS norm, then RoPE on the normalized rows -
# four ways: the scalar CPU path the transformer handler runs today, the two
# operations each staging their own activation to the device and back
# (op-at-a-time), the two operations on an activation uploaded once and
# downloaded once (resident chain), and the two operations on an activation
# already on the device (device-only). The resident-chain lane against the
# op-at-a-time lane is what removing the host round trip between two
# operations is worth; against the CPU lane it is whether the chain pays at all.
#
# Reports decode width (batch 1) and prefill width (batch 512) separately: a
# result at one width says nothing about the other.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-target/resident-chain}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DIM="${DIM:-2048}"
HEAD_DIM="${HEAD_DIM:-64}"
THETA="${THETA:-10000}"
DECODE_POS="${DECODE_POS:-512}"
BATCHES="${BATCHES:-1,512}"
REPS="${REPS:-3}"
WARMUP_MS="${WARMUP_MS:-3000}"
TARGET_MS="${TARGET_MS:-800}"

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/resident-chain-$STAMP.md"
LOG="$OUT_DIR/run-$STAMP.log"
JFR="$OUT_DIR/resident-chain-$STAMP.jfr"
JFR_JSON="$OUT_DIR/resident-chain-$STAMP-jfr.json"
JFC="$ROOT/scripts/performance-tests/juno-perf.jfc"

echo "[resident-chain] building node and metrics..."
mvn -q -pl node,metrics -am package -DskipTests

CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"

echo "[resident-chain] running -> $REPORT"
java \
  --enable-preview \
  --enable-native-access=ALL-UNNAMED \
  --add-modules jdk.incubator.vector \
  -XX:+UseG1GC \
  -XX:+AlwaysPreTouch \
  -XX:StartFlightRecording=settings="$JFC",filename="$JFR",dumponexit=true \
  -cp "$CP" \
  cab.ml.juno.node.ResidentChainMicrobench \
  --dim "$DIM" \
  --head-dim "$HEAD_DIM" \
  --theta "$THETA" \
  --decode-pos "$DECODE_POS" \
  --batch "$BATCHES" \
  --reps "$REPS" \
  --warmup-ms "$WARMUP_MS" \
  --target-ms "$TARGET_MS" \
  --out "$REPORT" 2>&1 | tee "$LOG"

# GC pauses and allocation rate alongside the primary metric, so a contaminated
# run is visible rather than inferred. Row-level repetition dispersion in the
# report is what decides whether a row is scorable.
METRICS_CP=$(mvn -q -pl metrics -DincludeScope=runtime -DforceStdout dependency:build-classpath)
java -cp "metrics/target/classes:$METRICS_CP" \
  cab.ml.juno.metrics.JfrMetricsCli "$JFR" "$JFR_JSON" resident-chain resident-chain \
  || echo "[resident-chain] JFR extraction failed - report stands, GC figures unavailable"

echo "[resident-chain] done: $REPORT"
