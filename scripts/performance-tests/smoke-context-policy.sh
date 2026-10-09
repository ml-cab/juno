#!/usr/bin/env bash
# smoke-context-policy.sh — context shifting over /v1/chat/completions on a real model
#
# Uses Phi-3.5-mini by default: its context limit is 4096 positions (the long RoPE factors are
# held back), so the limit is reached in minutes on a GPU. For each schedule (static and
# continuous) a server starts with the default (--context-shift off) and runs, in the order 2, 3, 1
# (a request that fails keeps its KV until the server stops, so the failing run goes last):
#   1. a conversation grows turn by turn (canned assistant replies, so every run sends the same
#      messages) until its history is well past the limit. Sent without the opt-in, a turn fails
#      once the request reaches the limit: HTTP 500 naming the model's original context length,
#      and every earlier turn answered;
#   2. the same conversation with x_juno_context_shift true (and a session id) answers every turn,
#      including the ones past the point where (1) failed;
#   3. one request whose prompt sits just under the limit and whose min_tokens holds it open
#      shifts during decode and returns every requested token.
# Then, on the static schedule, a server started with --context-shift on:
#   4. answers the crossing turn without any field (the server default applies), and fails it with
#      x_juno_context_shift false (the request's own choice wins).
#
# Environment overrides: JUNO_JAR (shaded jar under test), MODELS_DIR, SMOKE_HEAP (JVM -Xmx, default
# 16g: Phi-3.5-mini's prefill near 4096 tokens does not fit 8g).
# Writes logs, responses and results.json under target/context-policy-smoke/<timestamp>/.
# Exits 0 when every check passes, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-context-policy.sh
#   ./scripts/performance-tests/smoke-context-policy.sh --schedules static --cpu
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/context-policy-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
MODEL_FILE="Phi-3.5-mini-instruct-Q4_K_M.gguf"
LIMIT=4096
API_PORT=18093
BACKEND="--gpu"
SCHEDULES="static continuous"
NOTES_PER_TURN=25
TURNS=10
REPLY_TOKENS=16
DECODE_TOKENS=200
JUNO_PID=""
CROSSING_STATIC=0
failures=0
RESULTS="[]"

log()  { printf '[context-smoke] %s\n' "$*"; }
warn() { printf '[context-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[context-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[context-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); record "$1" false; }
pass() { printf '[context-smoke] PASS: %s\n' "$*"; record "$1" true; }
record() { RESULTS="$(jq -c --arg c "$1" --argjson ok "$2" '. + [{check:$c, passed:$ok}]' <<<"$RESULTS")"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) BACKEND="--cpu"; shift ;;
    --gpu) BACKEND="--gpu"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
MODEL="${MODELS_DIR}/${MODEL_FILE}"
[[ -f "$MODEL" ]] || die "model not found: ${MODEL}"
mkdir -p "$OUT"
trap 'stop_juno' EXIT

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}
JAR="$(find_juno_jar)"

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || { perf_engine_stdin_release; return 0; }
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
  perf_engine_stdin_release
}

# start_server <stem> <schedule> <context-shift on|off>
start_server() {
  local stem="$1" schedule="$2" shift_default="$3"
  local logf="${OUT}/${stem}.server.log" start now heap="${SMOKE_HEAP:-16g}"
  : >"$logf"
  curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1 \
    && die "port ${API_PORT} already has a healthy Juno API"
  perf_engine_stdin_open || die "cannot create the engine stdin pipe"
  (
    cd "$ROOT"
    perf_engine_exec java --enable-preview --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
      --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms"$heap" -Xmx"$heap" -Djuno.byteOrder=BE \
      -jar "$JAR" --model-path "$MODEL" --dtype FLOAT16 --byteOrder BE \
      --max-tokens "$REPLY_TOKENS" --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local "$BACKEND" \
      --api-port "$API_PORT" --schedule "$schedule" --context-shift "$shift_default"
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  start="$(date +%s)"
  until curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; do
    kill -0 "$JUNO_PID" 2>/dev/null || { warn "server exited early; see ${logf}"; JUNO_PID=""; return 1; }
    now="$(date +%s)"
    (( now - start < 600 )) || { warn "server did not become healthy; see ${logf}"; stop_juno; return 1; }
    sleep 2
  done
  log "up: ${stem} (pid ${JUNO_PID})"
}

PLACES=(harbour orchard library bridge market lighthouse mill chapel)
COLOURS=(red green blue white yellow grey black orange)

# notes <first> <count>: distinct numbered notes, about 24 tokens each
notes() {
  local i
  for (( i = $1; i < $1 + $2; i++ )); do
    printf 'Note %d: the %s in district %d was painted %s in the year %d.\n' \
      "$i" "${PLACES[$((i % 8))]}" "$((i * 7 % 31))" "${COLOURS[$((i * 3 % 8))]}" "$((1800 + i * 13))"
  done
}

# history <turns>: the conversation's messages up to and including user turn <turns>,
# with a canned assistant reply after each earlier turn
history() {
  local t msgs='[{"role":"system","content":"You keep a ledger of notes. Answer briefly."}]'
  for (( t = 0; t < $1; t++ )); do
    msgs="$(jq -c --arg u "Add these notes to the ledger:
$(notes $((t * NOTES_PER_TURN + 1)) "$NOTES_PER_TURN")" '. + [{role:"user",content:$u}]' <<<"$msgs")"
    (( t + 1 < $1 )) && msgs="$(jq -c '. + [{role:"assistant",content:"Noted."}]' <<<"$msgs")"
  done
  printf '%s' "$msgs"
}

# send <messages-json> <extra-json> <max> <min> <dest>: prints the HTTP status
send() {
  jq -n --argjson m "$1" --argjson x "$2" --argjson n "$3" --argjson k "$4" \
    '{messages:$m, temperature:0, max_tokens:$n, min_tokens:$k} + $x' |
    curl -sS --max-time 1800 -o "$5" -w '%{http_code}' -H 'Content-Type: application/json' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" || printf '000'
}

# conversation <stem> <extra-json>: runs turns 1..TURNS; prints the first failing turn (0 if none)
conversation() {
  local stem="$1" extra="$2" t code first=0
  for (( t = 1; t <= TURNS; t++ )); do
    code="$(send "$(history "$t")" "$extra" "$REPLY_TOKENS" 0 "${OUT}/${stem}.turn${t}.json")"
    printf '%s turn %d: HTTP %s prompt_tokens=%s\n' "$stem" "$t" "$code" \
      "$(jq -r '.usage.prompt_tokens // "-"' "${OUT}/${stem}.turn${t}.json" 2>/dev/null)" >>"${OUT}/turns.log"
    if [[ "$code" != 200 ]]; then
      first="$t"
      break
    fi
  done
  printf '%s' "$first"
}

for schedule in $SCHEDULES; do
  start_server "off-${schedule}" "$schedule" off || { fail "server start (${schedule})"; continue; }

  shifted_failed="$(conversation "shift-${schedule}" "{\"x_juno_context_shift\":true,\"x_juno_session_id\":\"smoke-${schedule}\"}")"
  if [[ "$shifted_failed" == 0 ]]; then
    pass "${schedule}: with x_juno_context_shift every one of ${TURNS} turns answers"
  else
    fail "${schedule}: with x_juno_context_shift turn ${shifted_failed} failed"
  fi

  # A prompt just under the limit, held open past it: the shift happens during decode. The note
  # count comes from the engine's own token counts for 1 and 101 notes.
  one="$(send "$(jq -c -n --arg u "$(notes 1 1)" '[{role:"system",content:"Answer briefly."},{role:"user",content:$u}]')" '{}' 1 0 "${OUT}/probe1.json")"
  many="$(send "$(jq -c -n --arg u "$(notes 1 101)" '[{role:"system",content:"Answer briefly."},{role:"user",content:$u}]')" '{}' 1 0 "${OUT}/probe101.json")"
  p1="$(jq -r '.usage.prompt_tokens // 0' "${OUT}/probe1.json" 2>/dev/null)"
  p101="$(jq -r '.usage.prompt_tokens // 0' "${OUT}/probe101.json" 2>/dev/null)"
  near=0
  if [[ "$one" == 200 && "$many" == 200 && "$p101" -gt "$p1" ]]; then
    near=$(( 1 + (LIMIT - 100 - p1) * 100 / (p101 - p1) ))
  fi
  msgs="$(jq -c -n --arg u "$(notes 1 "$near")" '[{role:"system",content:"Answer briefly."},{role:"user",content:$u}]')"
  code="$(send "$msgs" '{"x_juno_context_shift":true}' "$DECODE_TOKENS" "$DECODE_TOKENS" "${OUT}/decode-${schedule}.json")"
  got="$(jq -r '(.usage.prompt_tokens // 0) + (.usage.completion_tokens // 0)' "${OUT}/decode-${schedule}.json" 2>/dev/null)"
  done_tokens="$(jq -r '.usage.completion_tokens // 0' "${OUT}/decode-${schedule}.json" 2>/dev/null)"
  if [[ "$code" == 200 && "$done_tokens" == "$DECODE_TOKENS" && "$got" -gt "$LIMIT" ]]; then
    pass "${schedule}: a request held open past the limit shifts during decode (${got} positions in all)"
  else
    fail "${schedule}: decode-time shift: HTTP ${code}, ${done_tokens} of ${DECODE_TOKENS} tokens, ${got} positions"
  fi
  # Last on this server: a stateless request that fails is not evicted, so its KV stays held
  # until the server stops, and it must not starve the requests above.
  failed_at="$(conversation "noshift-${schedule}" '{}')"
  [[ "$schedule" == static ]] && CROSSING_STATIC="$failed_at"
  if [[ "$failed_at" -gt 1 ]] && grep -q "original context length" "${OUT}/noshift-${schedule}.turn${failed_at}.json"; then
    pass "${schedule}: without the opt-in, turn ${failed_at} fails at the ${LIMIT}-position limit with the documented error; turns ${failed_at}-${TURNS} answered above with the opt-in"
  else
    fail "${schedule}: without the opt-in, expected a later turn to fail at the limit (failed at turn ${failed_at})"
  fi

  stop_juno
done

if [[ " $SCHEDULES " == *" static "* ]]; then
  if start_server "on-static" static on; then
    crossing="${CROSSING_STATIC:-0}"
    if [[ "$crossing" -gt 0 ]]; then
      code_default="$(send "$(history "$crossing")" '{}' "$REPLY_TOKENS" 0 "${OUT}/default-on.json")"
      code_false="$(send "$(history "$crossing")" '{"x_juno_context_shift":false}' "$REPLY_TOKENS" 0 "${OUT}/default-on-false.json")"
      if [[ "$code_default" == 200 && "$code_false" != 200 ]]; then
        pass "static, --context-shift on: the crossing turn answers by default and fails with an explicit false"
      else
        fail "static, --context-shift on: default HTTP ${code_default}, explicit false HTTP ${code_false}"
      fi
    else
      fail "static, --context-shift on: no crossing turn known from the earlier run"
    fi
    stop_juno
  else
    fail "server start (static, --context-shift on)"
  fi
fi

jq -n --arg run "$RUN_ID" --arg model "$MODEL_FILE" --arg backend "$BACKEND" --argjson checks "$RESULTS" \
  --argjson failures "$failures" '{run:$run, model:$model, backend:$backend, failures:$failures, checks:$checks}' \
  >"${OUT}/results.json"
log "results: ${OUT}/results.json"
if (( failures > 0 )); then
  log "${failures} check(s) failed"
  exit 1
fi
log "all checks passed"
exit 0
