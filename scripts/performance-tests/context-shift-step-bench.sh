#!/usr/bin/env bash
# Context-shift step latency: the decode step that performs a context shift against an ordinary decode step at
# the same depth, at the context limit, on the CPU and the GPU (ContextShiftStepBench).
#
# A shift moves the kept keys and rotates them to their new positions, and on a GPU rebuilds the device KV copy
# attention reads; the step that does it must stay within a few ordinary steps, or a stream stalls for seconds.
# The bench fills the host KV directly to the limit (a cache's contents do not change what a step costs) instead of
# prefilling 32,768 tokens, then per repetition times five decode steps ending at the limit and the shift step at
# the full context. One warm-up repetition at a shorter depth is not reported.
#
# Gate: median over --reps repetitions of (shift step / median decode step) <= --max-ratio (default 3.0), per model
# and backend.
#
# Usage:
#   scripts/performance-tests/context-shift-step-bench.sh [--jar <shaded jar>] [--models a,b] [--backends cpu,gpu]
#       [--depth N] [--keep N] [--reps N] [--heap 14g] [--max-ratio 3.0] [--pin-clocks] [--out DIR]
# Models are names under models/ without .gguf (default: tinyllama-1.1b-chat-v1.0.Q4_K_M,
# mistral-7b-instruct-v0.1-q4_k_m). --pin-clocks needs prompt-free sudo (run 'sudo -v' first).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

JAR=""
MODELS="tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m"
BACKENDS="cpu,gpu"
DEPTH=32768
KEEP=32
REPS=3
HEAP=14g
MAX_RATIO=3.0
PIN_CLOCKS=0
OUT="target/context-shift-step/$(date -u +%Y%m%dT%H%M%SZ)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jar) JAR="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --heap) HEAP="$2"; shift 2 ;;
    --max-ratio) MAX_RATIO="$2"; shift 2 ;;
    --pin-clocks) PIN_CLOCKS=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$JAR" ]] || JAR=$(ls juno-player/target/juno-player-*-shaded.jar 2>/dev/null | head -1)
[[ -f "$JAR" ]] || { echo "no shaded jar (build with mvn package -DskipTests, or pass --jar)" >&2; exit 2; }
mkdir -p "$OUT"

PREV_GOVERNOR=""
PREV_NO_TURBO=""
PIN_STATE="not requested"
restore_clocks() {
  [[ -n "$PREV_GOVERNOR" ]] || return 0
  printf '%s\n' "$PREV_GOVERNOR" | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 \
    || echo "warning: could not restore the CPU governor" >&2
  if [[ -n "$PREV_NO_TURBO" ]]; then
    printf '%s\n' "$PREV_NO_TURBO" | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1 \
      || echo "warning: could not restore the turbo setting" >&2
  fi
  PREV_GOVERNOR=""
}
trap restore_clocks EXIT
if [[ $PIN_CLOCKS -eq 1 ]]; then
  sudo -n true 2>/dev/null || { echo "--pin-clocks needs prompt-free sudo: run 'sudo -v' first" >&2; exit 2; }
  PREV_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  printf 'performance\n' | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null \
    || { echo "could not set the performance governor" >&2; exit 2; }
  PIN_STATE="cpu governor performance (was $PREV_GOVERNOR)"
  if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    PREV_NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    printf '1\n' | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null \
      || { echo "could not turn turbo off" >&2; exit 2; }
    PIN_STATE+=", turbo off"
  else
    echo "no intel_pstate turbo control on this host; cannot pin the CPU clock" >&2; exit 2
  fi
  PIN_STATE+=", gpu clock recorded, not fixed"
fi

JAR_SHA=$(sha256sum "$JAR" | cut -c1-16)
SUMMARY="$OUT/summary.md"
{
  echo "# Context-shift step latency"
  echo
  echo "- Jar: \`$(basename "$JAR")\` sha256 \`$JAR_SHA\`; commit $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (dirty tree)')"
  echo "- Clocks: $PIN_STATE"
  echo "- Depth $DEPTH, keep $KEEP, $REPS repetitions, heap $HEAP; gate: median ratio <= $MAX_RATIO"
  echo
  echo "| Model | Backend | Median ratio | Min | Max | Result | Log |"
  echo "|---|---|---|---|---|---|---|"
} > "$SUMMARY"

status=0
IFS=',' read -r -a models <<<"$MODELS"
IFS=',' read -r -a backends <<<"$BACKENDS"
for m in "${models[@]}"; do
  for b in "${backends[@]}"; do
    log="$OUT/$m-$b.log"
    echo "=== $m $b ($(date +%H:%M:%S))"
    gpu_flag=(); [[ $b == gpu ]] && gpu_flag=(--gpu)
    java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -XX:+UseG1GC \
      -Xms"$HEAP" -Xmx"$HEAP" -cp "$JAR" cab.ml.juno.node.ContextShiftStepBench \
      --model "models/$m.gguf" "${gpu_flag[@]}" --depth "$DEPTH" --keep "$KEEP" --reps "$REPS" > "$log" 2>&1
    rc=$?
    grep -E "^(model=|warmup|rep |RESULT)" "$log" | sed 's/^/    /'
    result=$(grep '^RESULT' "$log" | tail -1)
    if [[ $rc -ne 0 || -z "$result" ]]; then
      echo "    run failed (exit $rc); see $log"
      echo "| $m | $b | - | - | - | run failed | \`$(basename "$log")\` |" >> "$SUMMARY"
      status=1
      continue
    fi
    med=$(sed -E 's/.*median_ratio=([0-9.]+).*/\1/' <<<"$result")
    mn=$(sed -E 's/.* min=([0-9.]+).*/\1/' <<<"$result")
    mx=$(sed -E 's/.* max=([0-9.]+).*/\1/' <<<"$result")
    if awk -v r="$med" -v b="$MAX_RATIO" 'BEGIN{exit !(r <= b)}'; then verdict=met; else verdict=MISSED; status=1; fi
    echo "| $m | $b | $med | $mn | $mx | $verdict | \`$(basename "$log")\` |" >> "$SUMMARY"
  done
done
echo
cat "$SUMMARY"
[[ $status -eq 0 ]] && echo "GATE MET" || echo "GATE MISSED"
exit $status
