#!/usr/bin/env bash
# Two sessions open with the same prompt on a real model; the second's reply must equal a stateless reply.
# usage: xsession.sh <jar> <static|continuous> <logdir>
set -uo pipefail
JAR="$1"; SCHED="$2"; OUT="$3"; mkdir -p "$OUT"
PORT=18093
MODEL=<repo>/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
cd <repo>
FIFO="$OUT/stdin-$SCHED"; rm -f "$FIFO"; mkfifo "$FIFO"
sleep 3600 > "$FIFO" & HOLD=$!
java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -Xmx4g \
  -jar "$JAR" --model-path "$MODEL" --cpu --max-tokens 12 --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local \
  --api-port $PORT --schedule "$SCHED" > "$OUT/server-$SCHED.log" 2>&1 < "$FIFO" &
PID=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:$PORT/v1/cluster/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited early"; kill $HOLD; exit 1; }
  sleep 2
done
ask() { # ask <session-id or empty> <user>
  jq -n --arg u "$2" --arg sid "$1" \
    '{messages:[{role:"system",content:"You are a helpful assistant. Answer in one short sentence."},{role:"user",content:$u}],temperature:0,max_tokens:12} + (if $sid == "" then {} else {x_juno_session_id:$sid} end)' |
    curl -sS -H 'Content-Type: application/json' http://127.0.0.1:$PORT/v1/chat/completions -d @- |
    jq -r '.choices[0].message.content // ("ERROR: " + (. | tostring))'
}
S0="$(ask "" "What is the capital of France?")"
SA="$(ask "user-a" "What is the capital of France?")"
SB="$(ask "user-b" "What is the capital of France?")"
kill $PID 2>/dev/null; wait $PID 2>/dev/null; kill $HOLD 2>/dev/null; rm -f "$FIFO"
printf 'schedule=%s\nstateless: %s\nsession a: %s\nsession b: %s\n' "$SCHED" "$S0" "$SA" "$SB"
[[ "$SB" == "$S0" && "$SA" == "$S0" ]] && echo "RESULT equal" || echo "RESULT differ"
