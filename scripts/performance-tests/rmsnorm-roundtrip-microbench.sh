#!/usr/bin/env bash
# RMS-norm host-round-trip microbench (GPU activation-residency baseline).
#
# Re-establishes, as a repeatable measurement, what one GPU RMS norm costs today
# when every call stages its own activation to the device and reads the result
# back. That cost is why LlamaTransformerHandler leaves CudaRmsNorm
# unconstructed, and it is the "before" side the activation-residency work is
# scored against.
#
# Reports decode width (batch 1) and prefill width (batch 512) separately: a
# result at one width says nothing about the other.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-target/rmsnorm-roundtrip}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DIM="${DIM:-2048}"
BATCHES="${BATCHES:-1,512}"
REPS="${REPS:-3}"
WARMUP_MS="${WARMUP_MS:-3000}"
TARGET_MS="${TARGET_MS:-800}"

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/rmsnorm-roundtrip-$STAMP.md"
LOG="$OUT_DIR/run-$STAMP.log"
JFR="$OUT_DIR/rmsnorm-roundtrip-$STAMP.jfr"
JFR_JSON="$OUT_DIR/rmsnorm-roundtrip-$STAMP-jfr.json"
JFC="$ROOT/scripts/performance-tests/juno-perf.jfc"

echo "[rmsnorm-roundtrip] building node and metrics..."
mvn -q -pl node,metrics -am package -DskipTests

CP=$(mvn -q -pl node -DincludeScope=runtime -DforceStdout dependency:build-classpath)
CP="node/target/classes:$CP"

echo "[rmsnorm-roundtrip] running -> $REPORT"
java \
  --enable-preview \
  --enable-native-access=ALL-UNNAMED \
  --add-modules jdk.incubator.vector \
  -XX:+UseG1GC \
  -XX:+AlwaysPreTouch \
  -XX:StartFlightRecording=settings="$JFC",filename="$JFR",dumponexit=true \
  -cp "$CP" \
  cab.ml.juno.node.RmsNormRoundTripMicrobench \
  --dim "$DIM" \
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
  cab.ml.juno.metrics.JfrMetricsCli "$JFR" "$JFR_JSON" rmsnorm-roundtrip rmsnorm-roundtrip \
  || echo "[rmsnorm-roundtrip] JFR extraction failed - report stands, GC figures unavailable"

echo "[rmsnorm-roundtrip] done: $REPORT"
