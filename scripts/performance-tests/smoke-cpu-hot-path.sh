#!/usr/bin/env bash
# smoke-cpu-hot-path.sh — CPU inference on the integer kernels, at more than one --threads value
#
# For each model (TinyLlama, Qwen2.5-3B, Phi-3.5-mini and Mistral 7B by default) and each thread count
# (4 and the host's processor count by default), on the CPU backend:
#   1. the server starts with --threads N and logs "CPU matmul threads: N";
#   2. a short greedy (temperature 0) chat request decodes: HTTP 200, non-empty text;
#   3. a greedy request with a prompt of about 512 tokens prefills and answers: HTTP 200, non-empty
#      text, usage.prompt_tokens within 10% of 512;
#   4. both replies are identical at every thread count (the CPU kernels are deterministic, and a
#      row is computed by one thread whatever the count).
# Then the vision case: moondream2 on the CPU answers POST /v1/vision/chat for the standard test image
# at each thread count with HTTP 200, a non-empty, identical reply, within --vision-max-s seconds (the
# vision prefill is the B of about 741 width at which a CPU kernel once ran for hours).
#
# Environment overrides: JUNO_JAR (shaded jar under test), MODELS_DIR, SMOKE_HEAP (JVM -Xmx),
# VISION_TEST_IMAGE. Writes logs, responses and results.json under target/cpu-hot-path-smoke/<timestamp>/.
# Exits 0 when every check passes, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-cpu-hot-path.sh
#   ./scripts/performance-tests/smoke-cpu-hot-path.sh --models tinyllama --threads "2 12" --no-vision
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/cpu-hot-path-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
API_PORT=18093
MODELS="tinyllama,qwen25,phi35,mistral"
THREAD_COUNTS="4 $(nproc 2>/dev/null || echo 12)"
VISION=1
VISION_MAX_S=600
IMAGE_PATH="${VISION_TEST_IMAGE:-${PERF_SCRIPTS}/fixtures/vision-bench.jpg}"
MAX_TOKENS=16
JUNO_PID=""
failures=0

log()  { printf '[cpu-smoke] %s\n' "$*"; }
warn() { printf '[cpu-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[cpu-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[cpu-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[cpu-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --threads) THREAD_COUNTS="$2"; shift 2 ;;
    --no-vision) VISION=0; shift ;;
    --vision-max-s) VISION_MAX_S="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
mkdir -p "$OUT"
trap 'stop_juno' EXIT

model_file() {
  case "$1" in
    tinyllama) printf '%s' "${MODELS_DIR}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ;;
    qwen25) printf '%s' "${MODELS_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf" ;;
    phi35) printf '%s' "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf" ;;
    mistral) printf '%s' "${MODELS_DIR}/mistral-7b-instruct-v0.1-q4_k_m.gguf" ;;
    moondream) printf '%s' "${MODELS_DIR}/moondream2-q5_k.llamafile" ;;
    *) die "unknown model key: $1 (tinyllama, qwen25, phi35, mistral)" ;;
  esac
}

model_heap() {
  if [[ -n "${SMOKE_HEAP:-}" ]]; then printf '%s' "$SMOKE_HEAP"; return; fi
  case "$1" in
    mistral) printf '9g' ;;
    qwen25|phi35) printf '6g' ;;
    *) printf '4g' ;;
  esac
}

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

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

# start_server <stem> <model-path> <heap> <threads>
start_server() {
  local stem="$1" model="$2" heap="$3" threads="$4"
  local logf="${OUT}/${stem}.server.log" start now
  : >"$logf"
  curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1 \
    && die "port ${API_PORT} already has a healthy Juno API"
  perf_engine_stdin_open || die "cannot create the engine stdin pipe"
  (
    cd "$ROOT"
    perf_engine_exec java --enable-preview --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
      --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms"$heap" -Xmx"$heap" -Djuno.byteOrder=BE \
      -jar "$JAR" --model-path "$model" --dtype FLOAT16 --byteOrder BE \
      --max-tokens "$MAX_TOKENS" --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local --cpu \
      --threads "$threads" --api-port "$API_PORT"
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  start="$(date +%s)"
  until curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; do
    kill -0 "$JUNO_PID" 2>/dev/null || { warn "server exited early; see ${logf}"; JUNO_PID=""; return 1; }
    now="$(date +%s)"
    (( now - start < 900 )) || { warn "server did not become healthy; see ${logf}"; stop_juno; return 1; }
    sleep 2
  done
  log "up: ${stem} (pid ${JUNO_PID})"
}

# chat <prompt> <max-tokens> <dest-json>: non-streaming greedy request; prints the HTTP status.
chat() {
  jq -n --arg u "$1" --argjson n "$2" '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n}' |
    curl -sS --max-time 3600 -o "$3" -w '%{http_code}' -H 'Content-Type: application/json' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" || printf '000'
}

# long_prompt <words>: numbered notes and a question; about 1.3 tokens per word.
long_prompt() {
  local w="$1" i
  printf 'Read the notes below, then answer the question at the end.\n'
  for (( i = 1; i * 12 <= w; i++ )); do
    printf 'Note %d: the bridge in district %d was painted in the year %d by crew %d.\n' \
      "$i" "$((i * 7 % 31))" "$((1800 + i * 13))" "$((i * 3 % 17))"
  done
  printf 'Question: in which year was the bridge in note 1 painted? Answer in one sentence.'
}

JAR="$(find_juno_jar)"
log "jar: ${JAR}"
declare -A REPLY_SHORT REPLY_LONG

IFS=',' read -r -a MODEL_KEYS <<<"$MODELS"
for key in "${MODEL_KEYS[@]}"; do
  model="$(model_file "$key")"
  [[ -f "$model" ]] || { warn "model not present, skipped: ${model}"; continue; }
  heap="$(model_heap "$key")"
  # calibrate the long prompt to about 512 tokens with the engine's own count, at the first thread count
  words=390
  for t in $THREAD_COUNTS; do
    stem="${key}-threads${t}"
    if ! start_server "$stem" "$model" "$heap" "$t"; then fail "${stem}: server did not start"; continue; fi
    if grep -q "CPU matmul threads: ${t}\b" "${OUT}/${stem}.server.log"; then
      pass "${stem}: server runs CPU matmuls on ${t} threads"
    else
      fail "${stem}: no 'CPU matmul threads: ${t}' line in the server log"
    fi
    code="$(chat "Name three colours." "$MAX_TOKENS" "${OUT}/${stem}.short.json")"
    text="$(jq -r '.choices[0].message.content // ""' "${OUT}/${stem}.short.json" 2>/dev/null)"
    if [[ "$code" == 200 && -n "$text" ]]; then pass "${stem}: decode HTTP 200, ${#text} chars"
    else fail "${stem}: decode HTTP ${code}, text '${text}'"; fi
    REPLY_SHORT["${key},${t}"]="$text"

    for attempt in 1 2 3; do
      code="$(chat "$(long_prompt "$words")" 4 "${OUT}/${stem}.long.json")"
      pt="$(jq -r '.usage.prompt_tokens // 0' "${OUT}/${stem}.long.json" 2>/dev/null)"
      [[ "$code" == 200 && "$pt" -gt 0 ]] || break
      (( pt >= 461 && pt <= 563 )) && break
      words=$(( words * 512 / pt ))
    done
    text="$(jq -r '.choices[0].message.content // ""' "${OUT}/${stem}.long.json" 2>/dev/null)"
    if [[ "$code" == 200 && -n "$text" && "$pt" -ge 461 && "$pt" -le 563 ]]; then
      pass "${stem}: 512-token prefill HTTP 200, prompt_tokens ${pt}"
    else
      fail "${stem}: 512-token prefill HTTP ${code}, prompt_tokens ${pt}, text '${text}'"
    fi
    REPLY_LONG["${key},${t}"]="$text"
    stop_juno
  done
  first=""
  for t in $THREAD_COUNTS; do
    if [[ -z "$first" ]]; then first="$t"; continue; fi
    if [[ "${REPLY_SHORT["${key},${t}"]}" == "${REPLY_SHORT["${key},${first}"]}" \
          && "${REPLY_LONG["${key},${t}"]}" == "${REPLY_LONG["${key},${first}"]}" ]]; then
      pass "${key}: greedy replies identical at ${first} and ${t} threads"
    else
      fail "${key}: greedy replies differ between ${first} and ${t} threads"
    fi
  done
done

if (( VISION )); then
  model="$(model_file moondream)"
  if [[ ! -f "$model" || ! -f "$IMAGE_PATH" ]]; then
    warn "vision case skipped: ${model} or ${IMAGE_PATH} not present"
  else
    declare -A VREPLY
    for t in $THREAD_COUNTS; do
      stem="vision-threads${t}"
      if ! start_server "$stem" "$model" 4g "$t"; then fail "${stem}: server did not start"; continue; fi
      s0="$(date +%s)"
      code="$(curl -sS --max-time "$VISION_MAX_S" -o "${OUT}/${stem}.json" -w '%{http_code}' \
        -X POST "http://127.0.0.1:${API_PORT}/v1/vision/chat" -F "image=@${IMAGE_PATH}" \
        -F "request=$(jq -nc '{messages:[{role:"user",content:"Describe the image in one sentence."}],max_tokens:24,temperature:0}')" \
        2>>"${OUT}/curl.err" || printf '000')"
      s1="$(date +%s)"
      text="$(jq -r '.choices[0].message.content // .content // ""' "${OUT}/${stem}.json" 2>/dev/null)"
      if [[ "$code" == 200 && -n "$text" && $(( s1 - s0 )) -le "$VISION_MAX_S" ]]; then
        pass "${stem}: vision HTTP 200 in $(( s1 - s0 )) s, ${#text} chars"
      else
        fail "${stem}: vision HTTP ${code} in $(( s1 - s0 )) s, text '${text}'"
      fi
      VREPLY["$t"]="$text"
      stop_juno
    done
    first=""
    for t in $THREAD_COUNTS; do
      if [[ -z "$first" ]]; then first="$t"; continue; fi
      if [[ "${VREPLY[$t]}" == "${VREPLY[$first]}" ]]; then pass "vision: replies identical at ${first} and ${t} threads"
      else fail "vision: replies differ between ${first} and ${t} threads"; fi
    done
  fi
fi

jq -n --arg run "$RUN_ID" --arg jar "$JAR" --arg threads "$THREAD_COUNTS" --argjson failures "$failures" \
  '{run_id:$run, jar:$jar, thread_counts:$threads, failures:$failures}' >"${OUT}/results.json"
if (( failures == 0 )); then log "all checks passed (${OUT})"; exit 0; fi
log "${failures} check(s) failed (${OUT})"
exit 1
