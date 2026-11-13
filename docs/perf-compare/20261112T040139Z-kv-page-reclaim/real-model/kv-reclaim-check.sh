#!/usr/bin/env bash
# Idle KV pages reclaimed on a real model (continuous schedule): a burst of requests, then idle; the
# health endpoint's page counts before, at the peak and after two reclaim periods; a greedy reply
# and a session's next turn before and after the reclaim.
# usage: kv-reclaim-check.sh <jar> <out-dir> [cpu|gpu] [lora]
set -uo pipefail
JAR="$1"; OUT="$2"; DEV="${3:-cpu}"; LORA="${4:-}"
PORT=18095
ROOT="${JUNO_ROOT:-$(git rev-parse --show-toplevel)}"
MODEL="$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
PERIOD_MS=1000
mkdir -p "$OUT"
TAG="$DEV${LORA:+-lora}"
FIFO="$OUT/stdin-$TAG"; rm -f "$FIFO"; mkfifo "$FIFO"
sleep 3600 > "$FIFO" & HOLD=$!
EXTRA=(); [[ -n "$LORA" ]] && EXTRA=(--lora-play "$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora")
DEVFLAG=--cpu; [[ "$DEV" == gpu ]] && DEVFLAG=--gpu
java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -Xmx6g \
  -Djuno.kv.reclaimPeriodMs=$PERIOD_MS \
  -jar "$JAR" --model-path "$MODEL" "$DEVFLAG" --max-tokens 16 --temperature 0 --top-k 0 --top-p 0 --nodes 1 \
  --local --api-port $PORT --schedule continuous --parallel 8 --prefix-cache-tokens 0 "${EXTRA[@]}" \
  > "$OUT/server-$TAG.log" 2>&1 < "$FIFO" &
PID=$!
for _ in $(seq 1 150); do
  curl -sf http://127.0.0.1:$PORT/v1/cluster/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited early"; kill $HOLD; exit 1; }
  sleep 2
done
SYS="You are a support assistant for a library. Follow these house rules."
for i in $(seq 1 12); do
  SYS+=" Rule $i: when a reader asks about item $i, answer politely, name the shelf, and keep the reply short."
done
# ask <session-id or empty> <user>
ask() {
  local body
  body="$(jq -n --arg s "$SYS" --arg u "$2" --arg sid "$1" \
    '{messages:[{role:"system",content:$s},{role:"user",content:$u}],temperature:0,max_tokens:16}
     + (if $sid == "" then {} else {x_juno_session_id:$sid} end)')"
  curl -sS -H 'Content-Type: application/json' http://127.0.0.1:$PORT/v1/chat/completions -d "$body" \
    | jq -r '.choices[0].message.content // ("ERROR: " + (. | tostring))' | tr '\n\t' '  '
}
health() {
  curl -sf http://127.0.0.1:$PORT/v1/cluster/health \
    | jq -c --arg l "$1" '{at:$l, kvPagesLive, kvPagesIdle, kvPagesReclaimed, kvPageBytes, prefixHits}'
}
{
  health "1 started"
  R1="$(ask "" "Which shelf holds item 3?")"
  S1="$(ask "user-s" "Which shelf holds item 5?")"
  health "2 after one request and a session turn"
  BURST=()
  for i in $(seq 1 8); do ask "" "Which shelf holds item $i? Answer for reader $i." > /dev/null & BURST+=($!); done
  wait "${BURST[@]}"
  health "3 after a burst of 8 concurrent requests"
  sleep $(( 3 * PERIOD_MS / 1000 + 1 ))
  health "4 idle for over two reclaim periods"
  R2="$(ask "" "Which shelf holds item 3?")"
  S2="$(ask "user-s" "Which shelf holds item 5?")"
  health "5 after the same request and the session's next turn"
  printf 'reply before reclaim\t%s\nreply after reclaim\t%s\nequal\t%s\n' "$R1" "$R2" "$([[ "$R1" == "$R2" ]] && echo yes || echo NO)"
  printf 'session turn 1\t%s\nsession turn 2\t%s\n' "$S1" "$S2"
} > "$OUT/result-$TAG.txt"
kill $PID 2>/dev/null; wait $PID 2>/dev/null; kill $HOLD 2>/dev/null; rm -f "$FIFO"
echo "== $TAG"; cat "$OUT/result-$TAG.txt"
