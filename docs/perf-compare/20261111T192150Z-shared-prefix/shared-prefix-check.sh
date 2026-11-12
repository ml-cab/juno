#!/usr/bin/env bash
# Replies with shared prompt prefixes on, against the same server with them off, on a real model.
# usage: shared-prefix-check.sh <jar> <static|continuous> <prefix-cache-tokens|default> <out-dir> [cpu|gpu] [parallel]
set -uo pipefail
JAR="$1"; SCHED="$2"; BUDGET="$3"; OUT="$4"; DEV="${5:-cpu}"; PAR="${6:-1}"
PORT=18094
ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel 2>/dev/null || pwd)"
ROOT="${JUNO_ROOT:-$ROOT}"
MODEL="$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
mkdir -p "$OUT"
TAG="$SCHED-$BUDGET-$DEV-p$PAR"
FIFO="$OUT/stdin-$TAG"; rm -f "$FIFO"; mkfifo "$FIFO"
sleep 3600 > "$FIFO" & HOLD=$!
EXTRA=(); [[ "$BUDGET" != default ]] && EXTRA=(--prefix-cache-tokens "$BUDGET")
DEVFLAG=--cpu; [[ "$DEV" == gpu ]] && DEVFLAG=--gpu
java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -Xmx6g \
  -jar "$JAR" --model-path "$MODEL" "$DEVFLAG" --max-tokens 16 --temperature 0 --top-k 0 --top-p 0 --nodes 1 \
  --local --api-port $PORT --schedule "$SCHED" --parallel "$PAR" "${EXTRA[@]}" > "$OUT/server-$TAG.log" 2>&1 < "$FIFO" &
PID=$!
for _ in $(seq 1 150); do
  curl -sf http://127.0.0.1:$PORT/v1/cluster/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited early"; kill $HOLD; exit 1; }
  sleep 2
done
SHORT="You are a helpful assistant. Answer in one short sentence."
LONG="You are a support assistant for a library. Follow these house rules."
for i in $(seq 1 40); do
  LONG+=" Rule $i: when a reader asks about item $i, answer politely, name the shelf, and keep the reply short."
done
# ask <label> <session-id or empty> <system> <user>
ask() {
  local body t
  body="$(jq -n --arg s "$3" --arg u "$4" --arg sid "$2" \
    '{messages:[{role:"system",content:$s},{role:"user",content:$u}],temperature:0,max_tokens:16}
     + (if $sid == "" then {} else {x_juno_session_id:$sid} end)')"
  t="$(curl -sS -o "$OUT/reply.json" -w '%{time_total}' -H 'Content-Type: application/json' \
    http://127.0.0.1:$PORT/v1/chat/completions -d "$body")"
  printf '%s\t%s\t%s\n' "$1" "$t" "$(jq -r '.choices[0].message.content // ("ERROR: " + (. | tostring))' "$OUT/reply.json" | tr '\n\t' '  ')"
}
{
  ask "1 short stateless capital" "" "$SHORT" "What is the capital of France?"
  ask "2 short stateless color" "" "$SHORT" "Name one primary color."
  ask "3 short stateless capital again" "" "$SHORT" "What is the capital of France?"
  ask "4 short session user-a capital" "user-a" "$SHORT" "What is the capital of France?"
  ask "5 short session user-b capital" "user-b" "$SHORT" "What is the capital of France?"
  ask "6 long stateless rule 3" "" "$LONG" "Which shelf holds item 3?"
  ask "7 long stateless rule 7" "" "$LONG" "Which shelf holds item 7?"
  ask "8 long session user-c rule 3" "user-c" "$LONG" "Which shelf holds item 3?"
  ask "9 long stateless rule 12" "" "$LONG" "Which shelf holds item 12?"
} > "$OUT/replies-$TAG.tsv"
curl -sf http://127.0.0.1:$PORT/v1/cluster/health | jq '{schedule, prefixHits, sharedPrefixHits, sharedPrefixTokens, sharedPrefixHeldTokens, sharedPrefixBudget}' > "$OUT/health-$TAG.json"
kill $PID 2>/dev/null; wait $PID 2>/dev/null; kill $HOLD 2>/dev/null; rm -f "$FIFO" "$OUT/reply.json"
echo "$TAG"; cat "$OUT/replies-$TAG.tsv"; cat "$OUT/health-$TAG.json"
