#!/usr/bin/env bash
# One traced decode request: Juno launched as compare-llama-cpp.sh launches it (GPU lane,
# default flags), warmed up, then CUDA-traced (nsys) and JFR-recorded over one measured
# request of N_GEN tokens at a prompt of PROMPT_WORDS raw words.
# Usage: nsys-decode.sh MODEL_FILE HEAP PROMPT_WORDS N_GEN OUT_PREFIX [extra juno args...]
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MODEL="$1"; HEAP="$2"; WORDS="$3"; NGEN="$4"; OUT="$5"; shift 5
JAR="${JAR:-$ROOT/juno-player/target/juno-player-0.1.2-shaded.jar}"
PORT="${PORT:-18181}"
SESSION="juno$$"
TRACE="${TRACE:-1}"

prompt="x"; for ((i = 1; i < WORDS; i++)); do prompt+=" x"; done

fifo="$(mktemp -u)"; mkfifo "$fifo"; exec {FD}<>"$fifo"
java_args=(--enable-preview --enable-native-access=ALL-UNNAMED
  --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED
  --add-modules jdk.incubator.vector -XX:+UseG1GC -XX:+AlwaysPreTouch
  -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djuno.threads=12
  -Xms"$HEAP" -Xmx"$HEAP" -Djuno.byteOrder=BE -jar "$JAR"
  --model-path "$ROOT/models/$MODEL" --dtype FLOAT16 --byteOrder BE --max-tokens "$NGEN"
  --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local --gpu --api-port "$PORT" "$@")
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
JPID="$(pgrep -f -- "--api-port $PORT" | while read -r p; do
  [[ "$(cat /proc/$p/comm)" == java ]] && echo "$p"; done | head -1)"
req "$OUT-warmup1.json"
jcmd "$JPID" JFR.start name=warm settings="$ROOT/scripts/performance-tests/juno-perf.jfc" >/dev/null
req "$OUT-warmup2.json"
jcmd "$JPID" JFR.stop name=warm >/dev/null
req "$OUT-untraced.json"
jcmd "$JPID" JFR.start name=meas settings="$ROOT/scripts/performance-tests/juno-perf.jfc" \
  settings="$ROOT/scripts/performance-tests/juno-perf-spans.jfc" >/dev/null
if (( TRACE )); then nsys start --session="$SESSION" -o "$OUT" -f true >/dev/null; fi
req "$OUT-measured.json"
if (( TRACE )); then nsys stop --session="$SESSION" >/dev/null; fi
jcmd "$JPID" JFR.stop name=meas filename="$OUT.jfr" >/dev/null
jq -c '{prompt_tokens:.usage.prompt_tokens, completion_tokens:.usage.completion_tokens, latency_ms:.x_juno_latency_ms}' \
  "$OUT-untraced.json" "$OUT-measured.json"
