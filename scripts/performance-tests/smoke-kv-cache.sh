#!/usr/bin/env bash
# smoke-kv-cache.sh — KV cache features on a real local server (TinyLlama, REST)
#
#   1. shared prompts: two concurrent clients whose prompts start with a finished request's system prompt
#      skip that prefix (sharedPrefixHits counts both) and reply as with --prefix-cache-tokens 0; with --gpu
#      both servers run --prefill-batch 1, because GPU prefill rounds a position by the width of the window
#      it is computed in and a shared prefix moves the windows, so only one-position windows compare sharing;
#   2. page reclaim: a churn of requests on the continuous schedule leaves idle pages that the background
#      pass releases (kvPagesReclaimed > 0), and a fixed request replies the same before and after;
#   3. session save and restore across a restart: a session saved over --session-dir, the server killed,
#      a new server (continuous schedule) restores it and its next turn equals the uninterrupted one,
#      served from the restored KV; a missing session answers 404, a bad id 400, no directory 400.
#
# Writes HTTP bodies and logs under target/kv-cache-smoke/<timestamp>/. Exits 1 on any failure.
#
# Usage:
#   ./scripts/performance-tests/smoke-kv-cache.sh [--api-port 18084] [--gpu]
set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
unset JUNO_PREFIX_CACHE_TOKENS # perf-lib.sh turns the shared prompts off for benchmarks; here the default applies
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/kv-cache-smoke/${RUN_ID}"
MODEL="${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
API_PORT=18084
BACKEND="--cpu"
SHARED_PREFILL=() # --gpu: one-position prefill windows for part 1
JUNO_PID=""
failures=0

log()  { printf '[kv-cache-smoke] %s\n' "$*"; }
warn() { printf '[kv-cache-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[kv-cache-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[kv-cache-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[kv-cache-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port) API_PORT="$2"; shift 2 ;;
    --gpu)      BACKEND=""; SHARED_PREFILL=(--prefill-batch 1); shift ;;
    --out)      OUT_ROOT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

[[ -f "$MODEL" ]] || die "model missing: $MODEL"
command -v jq >/dev/null 2>&1 || die "jq not found"
mkdir -p "$OUT_ROOT"

find_juno_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/juno-player/target/"juno-player-*-shaded.jar )
  shopt -u nullglob
  [[ ${#jars[@]} -gt 0 ]] || die "juno-player shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

find_java() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    printf '%s' "${JAVA_HOME}/bin/java"
    return
  fi
  command -v java >/dev/null 2>&1 || die "java not found"
  printf '%s' java
}

health() { curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health"; }

wait_for_api() {
  local start; start="$(date +%s)"
  while ! health >/dev/null 2>&1; do
    kill -0 "$JUNO_PID" 2>/dev/null || { warn "server exited before its API came up"; return 1; }
    (( $(date +%s) - start < 300 )) || return 1
    sleep 2
  done
}

# stop_juno [KILL]: TERM and wait, or KILL at once (a crash, no shutdown hook)
stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || { perf_engine_stdin_release; return 0; }
  if kill -0 "$pid" 2>/dev/null; then
    if [[ "${1:-}" == "KILL" ]]; then
      kill -KILL "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
      local i
      for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
  perf_engine_stdin_release
}
trap 'stop_juno' EXIT

# start_juno STEM [JVM -D options...] -- [Juno flags...]
start_juno() {
  local stem="$1"; shift
  local jvm=() app=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do jvm+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  app=("$@")
  local jar java_bin logf
  jar="$(find_juno_jar)"; java_bin="$(find_java)"
  logf="${OUT_ROOT}/${stem}.log"
  health >/dev/null 2>&1 && die "port ${API_PORT} already has a Juno API"
  perf_engine_stdin_open || die "cannot create the engine stdin pipe"
  (
    cd "$ROOT"
    perf_engine_exec "$java_bin" --enable-preview --enable-native-access=ALL-UNNAMED \
      --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms512m -Xmx4g "${jvm[@]}" \
      -jar "$jar" --model-path "$MODEL" --max-tokens 24 --temperature 0 --top-k 0 --top-p 0 \
      --nodes 1 --local ${BACKEND} --api-port "$API_PORT" "${app[@]}"
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  wait_for_api || { warn "API did not come up; see ${logf}"; stop_juno; return 1; }
  log "up: ${stem} pid=${JUNO_PID}"
}

# chat NAME JSON → HTTP code; body in NAME.json
chat() {
  curl -sS -o "${OUT_ROOT}/$1.json" -w '%{http_code}' -H 'Content-Type: application/json' \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d "$2" || true
}

# session_call NAME METHOD PATH → HTTP code; body in NAME.json
session_call() {
  curl -sS -o "${OUT_ROOT}/$1.json" -w '%{http_code}' -X "$2" "http://127.0.0.1:${API_PORT}/v1$3" || true
}

reply() { jq -r '.choices[0].message.content // empty' "${OUT_ROOT}/$1.json"; }
field() { jq -r ".$2" "${OUT_ROOT}/$1.json"; }
hfield() { health | jq -r ".$1"; }

SYSTEM="You are a patient tutor who explains ideas with one short everyday example, avoids jargon, and keeps every answer under three sentences."
body() {  # body USER [SESSION] [EXTRA_MESSAGES_JSON]
  local msgs
  msgs="$(jq -cn --arg s "$SYSTEM" --arg u "$1" '[{role:"system",content:$s},{role:"user",content:$u}]')"
  [[ -n "${3:-}" ]] && msgs="$(jq -cn --argjson a "$msgs" --argjson b "$3" '$a + $b')"
  if [[ -n "${2:-}" ]]; then
    jq -cn --argjson m "$msgs" --arg id "$2" '{model:"tinyllama",max_tokens:24,temperature:0,messages:$m,x_juno_session_id:$id}'
  else
    jq -cn --argjson m "$msgs" '{model:"tinyllama",max_tokens:24,temperature:0,messages:$m}'
  fi
}

# ── 1. shared prompts ─────────────────────────────────────────────────────────
shared_round() {  # shared_round TAG: one finished request, then two concurrent ones
  [[ "$(chat "$1-warm" "$(body 'What is gravity?')")" == 200 ]] || fail "$1: warm request"
  chat "$1-a" "$(body 'What is friction?')" >"${OUT_ROOT}/$1-a.code" &
  local a=$!
  chat "$1-b" "$(body 'What is inertia?')" >"${OUT_ROOT}/$1-b.code" &
  local b=$!
  wait "$a" "$b"
  [[ "$(cat "${OUT_ROOT}/$1-a.code")" == 200 && "$(cat "${OUT_ROOT}/$1-b.code")" == 200 ]] || fail "$1: concurrent requests"
}
log "1. shared prompts"
start_juno shared-off -- --prefix-cache-tokens 0 "${SHARED_PREFILL[@]}" || die "server did not start"
shared_round off
stop_juno
start_juno shared-on -- "${SHARED_PREFILL[@]}" || die "server did not start"
shared_round on
hits="$(hfield sharedPrefixHits)"
stop_juno
if [[ "$hits" -ge 2 ]]; then pass "two concurrent requests started from the shared system prompt (sharedPrefixHits=${hits})"
else fail "sharedPrefixHits=${hits}, expected >= 2"; fi
for x in a b; do
  if [[ "$(reply "on-$x")" == "$(reply "off-$x")" ]]; then pass "reply ${x} equals sharing off"
  else fail "reply ${x} differs with sharing on: \"$(reply "on-$x")\" vs \"$(reply "off-$x")\""; fi
done

# ── 2. page reclaim ───────────────────────────────────────────────────────────
log "2. page reclaim under churn (continuous schedule)"
start_juno reclaim -Djuno.kv.reclaimPeriodMs=1000 -- --schedule continuous || die "server did not start"
[[ "$(chat probe-before "$(body 'Name a fruit.')")" == 200 ]] || fail "probe before churn"
for i in $(seq 1 12); do
  q="Count to $((i * 3)) and name $i animals that live in $( ((i % 2)) && echo forests || echo oceans )."
  [[ "$(chat "churn-$i" "$(body "$q")")" == 200 ]] || fail "churn request $i"
done
sleep 4
[[ "$(chat probe-after "$(body 'Name a fruit.')")" == 200 ]] || fail "probe after churn"
reclaimed="$(hfield kvPagesReclaimed)"
stop_juno
if [[ "$reclaimed" -gt 0 ]]; then pass "idle pages released (kvPagesReclaimed=${reclaimed})"; else fail "kvPagesReclaimed=${reclaimed}"; fi
if [[ "$(reply probe-after)" == "$(reply probe-before)" ]]; then pass "reply unchanged across the churn and reclaim"
else fail "reply changed: \"$(reply probe-before)\" vs \"$(reply probe-after)\""; fi

# ── 3. session save and restore across a restart ─────────────────────────────
log "3. session save, kill, restore"
SESSIONS="${OUT_ROOT}/sessions"
start_juno no-dir || die "server did not start"
code="$(session_call nodir-save POST /sessions/s1/save)"
stop_juno
if [[ "$code" == 400 ]] && grep -q -- '--session-dir' "${OUT_ROOT}/nodir-save.json"; then pass "no directory: 400 naming --session-dir"
else fail "no directory: HTTP ${code}"; fi

start_juno save -- --session-dir "$SESSIONS" || die "server did not start"
[[ "$(chat turn1 "$(body 'Name three colors of the rainbow.' s1)")" == 200 ]] || fail "turn 1"
first="$(reply turn1)"
code="$(session_call save POST /sessions/s1/save)"
if [[ "$code" == 200 && "$(field save tokens)" -gt 0 ]]; then pass "saved: $(field save tokens) tokens, $(field save bytes) bytes"
else fail "save: HTTP ${code}"; fi
more="$(jq -cn --arg r "$first" '[{role:"assistant",content:$r},{role:"user",content:"Now name one more."}]')"
[[ "$(chat turn2-unbroken "$(body 'Name three colors of the rainbow.' s1 "$more")")" == 200 ]] || fail "turn 2, uninterrupted"
[[ "$(session_call bad-id POST '/sessions/bad,id/save')" == 400 ]] && pass "bad id: 400" || fail "bad id"
[[ "$(session_call missing POST /sessions/nobody/restore)" == 404 ]] && pass "nothing saved: 404" || fail "missing session"
stop_juno KILL
log "server killed"

start_juno restore -- --session-dir "$SESSIONS" --schedule continuous || die "server did not start"
code="$(session_call restore POST /sessions/s1/restore)"
if [[ "$code" == 200 && "$(field restore tokens)" == "$(field save tokens)" ]]; then pass "restored after the restart: $(field restore tokens) tokens"
else fail "restore: HTTP ${code} $(cat "${OUT_ROOT}/restore.json")"; fi
before="$(hfield prefixHits)"
[[ "$(chat turn2-restored "$(body 'Name three colors of the rainbow.' s1 "$more")")" == 200 ]] || fail "turn 2 after the restart"
after="$(hfield prefixHits)"
[[ "$(session_call delete DELETE /sessions/s1)" == 200 ]] && pass "deleted: 200" || fail "delete"
stop_juno
if [[ "$after" -eq $((before + 1)) ]]; then pass "the second turn started from the restored KV"; else fail "prefixHits ${before} -> ${after}"; fi
if [[ "$(reply turn2-restored)" == "$(reply turn2-unbroken)" ]]; then pass "second turn after the restart equals the uninterrupted one"
else fail "second turn differs: \"$(reply turn2-restored)\" vs \"$(reply turn2-unbroken)\""; fi

log "results under ${OUT_ROOT}"
if (( failures > 0 )); then log "${failures} failure(s)"; exit 1; fi
log "all checks passed"
