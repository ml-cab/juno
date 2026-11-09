#!/usr/bin/env bash
# One batch of N concurrent decode requests: Juno launched as the earlier decode decomposition launched it (GPU lane,
# default flags) plus --parallel N and a batch window long enough that the N requests form one batch. Two warm-up
# batches, then one measured batch: under JFR (juno-perf.jfc) when TRACE=0, under a CUDA activity trace (nsys, no
# JFR) when TRACE=1. Every request has the same prompt of PROMPT_WORDS raw words and decodes exactly N_GEN tokens
# (min_tokens), so the N requests stay in one batch to the end. N=1 runs --parallel 1: the single-request step.
# Usage: nsys-batch-decode.sh MODEL_FILE HEAP PROMPT_WORDS N_GEN N OUT_PREFIX
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MODEL="$1"; HEAP="$2"; WORDS="$3"; NGEN="$4"; N="$5"; OUT="$6"
JAR="${JAR:-$ROOT/juno-player/target/juno-player-0.1.2-shaded.jar}"
PORT="${PORT:-18181}"
SESSION="juno$$"
TRACE="${TRACE:-0}"

prompt="x"; for ((i = 1; i < WORDS; i++)); do prompt+=" x"; done

fifo="$(mktemp -u)"; mkfifo "$fifo"; exec {FD}<>"$fifo"
java_args=(--enable-preview --enable-native-access=ALL-UNNAMED
  --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED
  --add-modules jdk.incubator.vector -XX:+UseG1GC -XX:+AlwaysPreTouch
  -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps
  -Xms"$HEAP" -Xmx"$HEAP" -Djuno.byteOrder=BE -jar "$JAR"
  --model-path "$ROOT/models/$MODEL" --dtype FLOAT16 --byteOrder BE --max-tokens "$NGEN"
  --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local --gpu --threads 12 --api-port "$PORT"
  --parallel "$N" --batch-window-ms 500)
if (( TRACE )); then
  nsys launch --session-new="$SESSION" -t cuda --cuda-memory-usage=false java "${java_args[@]}" \
    <"$fifo" {FD}>&- >"$OUT.log" 2>&1 &
else
  java "${java_args[@]}" <"$fifo" {FD}>&- >"$OUT.log" 2>&1 &
fi
LPID=$!
cleanup() {
  if (( TRACE )); then nsys shutdown --session="$SESSION" >/dev/null 2>&1 || true; fi
  pkill -f -- "--api-port $PORT" 2>/dev/null || true
  wait "$LPID" 2>/dev/null || true
  exec {FD}>&-; rm -f "$fifo"
}
trap cleanup EXIT

for ((t = 0; t < 600; t++)); do
  curl -sf "http://127.0.0.1:$PORT/v1/cluster/health" >/dev/null 2>&1 && break
  kill -0 "$LPID" 2>/dev/null || { echo "engine died" >&2; tail -20 "$OUT.log" >&2; exit 1; }
  sleep 1
done
model_id="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id')"
req() {
  curl -sS --max-time 3600 -o "$1" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$model_id" --arg p "$prompt" --argjson n "$NGEN" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,min_tokens:$n,temperature:0,stream:false}')"
}
batch() { # TAG: N concurrent requests, replies at OUT-TAG-<i>.json
  local i pids=()
  for ((i = 1; i <= N; i++)); do req "$OUT-$1-$i.json" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
}
JPID="$(pgrep -f -- "--api-port $PORT" | while read -r p; do
  [[ "$(cat /proc/$p/comm)" == java ]] && echo "$p"; done | head -1)"
batch warmup1
batch warmup2
if (( TRACE )); then
  nsys start --session="$SESSION" -o "$OUT" -f true >/dev/null
  batch measured
  nsys stop --session="$SESSION" >/dev/null
else
  jcmd "$JPID" JFR.start name=meas settings="$ROOT/scripts/performance-tests/juno-perf.jfc" >/dev/null
  batch measured
  jcmd "$JPID" JFR.stop name=meas filename="$OUT.jfr" >/dev/null
fi
for ((i = 1; i <= N; i++)); do
  jq -c '{prompt_tokens:.usage.prompt_tokens, completion_tokens:.usage.completion_tokens, latency_ms:.x_juno_latency_ms}' \
    "$OUT-measured-$i.json"
done
