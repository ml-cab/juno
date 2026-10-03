#!/usr/bin/env bash
# smoke-long-prompt-prefill.sh — long-prompt prefill over /v1/chat/completions on real model files
#
# For each model (TinyLlama and Mistral 7B by default), each schedule (static and continuous)
# and each prompt length (128, 512 and 2048 prompt tokens by default; a length the model's
# context cannot hold together with the generated tokens is capped at that context and the
# cap is reported):
#   1. a greedy (temperature 0) non-streaming request prefills the prompt and answers: HTTP 200,
#      non-empty text, and usage.prompt_tokens within 10% of the requested length;
#   2. the same request streamed returns the same text, and its time to first token (TTFT) is
#      recorded;
#   3. with --baseline-jar, the same requests against that build return the same text, so greedy
#      decoding is shown unchanged against an earlier build on every model, schedule and length.
#
# The prompt is a list of distinct numbered notes ending in a question about the first one;
# whether the answer names the right colour is recorded for reading, not asserted.
#
# Environment overrides: JUNO_JAR (shaded jar under test), MODELS_DIR, SMOKE_HEAP (JVM -Xmx).
# Writes logs, responses and results.json under target/prefill-smoke/<timestamp>/.
# Exits 0 when every check passes, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-long-prompt-prefill.sh
#   ./scripts/performance-tests/smoke-long-prompt-prefill.sh --baseline-jar /path/to/earlier-shaded.jar
#   ./scripts/performance-tests/smoke-long-prompt-prefill.sh --models tinyllama --lengths "128 512" --cpu
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/prefill-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
API_PORT=18091
BACKEND="--gpu"
MODELS="tinyllama,mistral"
LENGTHS="128 512 2048"
SCHEDULES="static continuous"
MAX_TOKENS=16
BASELINE_JAR=""
JUNO_PID=""
failures=0

log()  { printf '[prefill-smoke] %s\n' "$*"; }
warn() { printf '[prefill-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[prefill-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[prefill-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[prefill-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --baseline-jar) BASELINE_JAR="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) BACKEND="--cpu"; shift ;;
    --gpu) BACKEND="--gpu"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
[[ -z "$BASELINE_JAR" || -f "$BASELINE_JAR" ]] || die "baseline jar not found: ${BASELINE_JAR}"
mkdir -p "$OUT"
trap 'stop_juno' EXIT

model_file() {
  case "$1" in
    tinyllama) printf '%s' "${MODELS_DIR}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ;;
    mistral) printf '%s' "${MODELS_DIR}/mistral-7b-instruct-v0.1-q4_k_m.gguf" ;;
    *) die "unknown model key: $1 (tinyllama or mistral)" ;;
  esac
}

model_heap() {
  if [[ -n "${SMOKE_HEAP:-}" ]]; then printf '%s' "$SMOKE_HEAP"; return; fi
  case "$1" in
    mistral) printf '8g' ;;
    *) printf '4g' ;;
  esac
}

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

context_length() {
  "${ROOT}/juno" gguf-info --model-path "$1" 2>/dev/null \
    | sed -n 's/^ *[a-z0-9_]*\.context_length = \([0-9][0-9]*\)$/\1/p' | head -1
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

# start_server <stem> <jar> <model-path> <heap> <schedule>
start_server() {
  local stem="$1" jar="$2" model="$3" heap="$4" schedule="$5"
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
      -jar "$jar" --model-path "$model" --dtype FLOAT16 --byteOrder BE \
      --max-tokens "$MAX_TOKENS" --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local "$BACKEND" \
      --api-port "$API_PORT" --schedule "$schedule"
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

FILLER=(please read every note above with care before you answer)

# notes_prompt <note-count> [<filler-words>]: distinct numbered notes, an optional remark of
# filler words (a note is about 24 tokens, so the remark makes up the remainder), then a
# question about note 1.
notes_prompt() {
  local n="$1" w="${2:-0}" i
  printf 'Read the notes below, then answer the question at the end.\n'
  for (( i = 1; i <= n; i++ )); do
    printf 'Note %d: the %s in district %d was painted %s in the year %d.\n' \
      "$i" "${PLACES[$((i % 8))]}" "$((i * 7 % 31))" "${COLOURS[$((i * 3 % 8))]}" "$((1800 + i * 13))"
  done
  if (( w > 0 )); then
    printf 'Remark:'
    for (( i = 0; i < w; i++ )); do printf ' %s' "${FILLER[$((i % ${#FILLER[@]}))]}"; done
    printf '.\n'
  fi
  printf 'Question: according to note 1, what colour was the %s painted? Answer in one sentence.' "${PLACES[1]}"
}

# prompt_tokens_of <note-count> <filler-words> <dest-json>: the engine's own count, or 0
prompt_tokens_of() {
  [[ "$(chat "$(notes_prompt "$1" "$2")" 1 "$3")" == 200 ]] || { printf '0'; return; }
  jq -r '.usage.prompt_tokens // 0' "$3"
}

# chat <prompt> <max-tokens> <dest-json>: non-streaming greedy request; prints the HTTP status.
chat() {
  jq -n --arg u "$1" --argjson n "$2" '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n}' |
    curl -sS --max-time 1800 -o "$3" -w '%{http_code}' -H 'Content-Type: application/json' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" || printf '000'
}

# chat_stream <prompt> <dest-json>: streamed greedy request; writes {ttft_ms, text, chunks}.
chat_stream() {
  jq -n --arg u "$1" --argjson n "$MAX_TOKENS" \
    '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n,stream:true}' |
    curl -sS --max-time 1800 -N -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" |
    python3 -c '
import json, sys, time
t0 = time.time() * 1000.0
ttft, text, chunks = None, [], 0
for line in sys.stdin:
    if not line.startswith("data:"):
        continue
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        continue
    try:
        c = ((json.loads(raw).get("choices") or [{}])[0].get("delta") or {}).get("content")
    except Exception:
        continue
    if c:
        if ttft is None:
            ttft = time.time() * 1000.0 - t0
        text.append(c)
        chunks += 1
json.dump({"ttft_ms": ttft, "text": "".join(text), "chunks": chunks}, open(sys.argv[1], "w"))
' "$2"
}

# Note and filler-word counts per requested length, per model: calibrated once against the
# engine's own usage.prompt_tokens (a linear fit over notes, then filler words for the remainder
# with one correction), and reused for every schedule and build, since tokenization depends on
# neither.
declare -A NOTES=()
declare -A WORDS=()
declare -A TARGET=()

calibrate() {
  local key="$1" ctx="$2" len a b ta tb per target n w t
  a=2; b=40
  ta="$(prompt_tokens_of "$a" 0 "${OUT}/${key}.calib-a.json")"
  tb="$(prompt_tokens_of "$b" 0 "${OUT}/${key}.calib-b.json")"
  (( ta > 0 && tb > ta )) || return 1
  # per-note tokens in hundredths, so the fit stays integer
  per=$(( (tb - ta) * 100 / (b - a) ))
  for len in $LENGTHS; do
    target="$len"
    if [[ -n "$ctx" ]] && (( target + MAX_TOKENS + 8 > ctx )); then
      target=$(( ctx - MAX_TOKENS - 8 ))
      log "${key}: ${len}-token length capped at ${target} (context ${ctx}, ${MAX_TOKENS} generated tokens)"
    fi
    n=$(( a + (target - ta) * 100 / per ))
    (( n < 1 )) && n=1
    t="$(prompt_tokens_of "$n" 0 "${OUT}/${key}.calib-${len}-0.json")"
    (( t > 0 )) || return 1
    w=$(( target - t > 0 ? target - t : 0 ))
    if (( w > 0 )); then
      t="$(prompt_tokens_of "$n" "$w" "${OUT}/${key}.calib-${len}-1.json")"
      (( t > 0 )) || return 1
      w=$(( w + target - t )); (( w < 0 )) && w=0
    fi
    NOTES["${key}:${len}"]="$n"
    WORDS["${key}:${len}"]="$w"
    TARGET["${key}:${len}"]="$target"
    log "${key}: ${len} -> ${n} notes + ${w} filler words (target ${target} prompt tokens)"
  done
}

# run_requests <label> : one non-streamed and one streamed request per length; records results
run_requests() {
  local label="$1" key="$2" schedule="$3" build="$4" len n target code text pt ttft stext row recall
  for len in $LENGTHS; do
    n="${NOTES["${key}:${len}"]}"; target="${TARGET["${key}:${len}"]}"
    local prompt; prompt="$(notes_prompt "$n" "${WORDS["${key}:${len}"]}")"
    local resp="${OUT}/${label}.${len}.json" sresp="${OUT}/${label}.${len}.stream.json"
    code="$(chat "$prompt" "$MAX_TOKENS" "$resp")"
    text="$(jq -r '.choices[0].message.content // empty' "$resp" 2>/dev/null)"
    pt="$(jq -r '.usage.prompt_tokens // 0' "$resp" 2>/dev/null)"
    chat_stream "$prompt" "$sresp"
    ttft="$(jq -r '.ttft_ms // "null"' "$sresp" 2>/dev/null)"
    stext="$(jq -r '.text // empty' "$sresp" 2>/dev/null)"
    recall=false
    [[ "${text,,}" == *"${COLOURS[3]}"* ]] && recall=true

    if [[ "$build" == candidate ]]; then
      if [[ "$code" == 200 && -n "${text//[[:space:]]/}" ]]; then
        pass "${label} ${len}: HTTP 200, non-empty answer"
      else
        fail "${label} ${len}: HTTP ${code}, answer '${text}'; see ${resp}"
      fi
      if [[ "$pt" =~ ^[0-9]+$ ]] && (( pt * 10 >= target * 9 && pt * 10 <= target * 11 )); then
        pass "${label} ${len}: prompt_tokens ${pt} within 10% of ${target}"
      else
        fail "${label} ${len}: prompt_tokens ${pt}, requested ${target}"
      fi
      if [[ "$stext" == "$text" ]]; then
        pass "${label} ${len}: streamed text equals non-streamed (TTFT ${ttft} ms)"
      else
        fail "${label} ${len}: streamed '${stext}' differs from non-streamed '${text}'"
      fi
    fi
    row="$(jq -nc --arg model "$key" --arg schedule "$schedule" --arg build "$build" --argjson len "$len" \
      --argjson target "$target" --argjson pt "${pt:-0}" --arg code "$code" --arg text "$text" \
      --argjson ttft "${ttft:-null}" --argjson recall "$recall" \
      '{model:$model,schedule:$schedule,build:$build,requested_tokens:$len,target_tokens:$target,
        prompt_tokens:$pt,http:$code,ttft_ms:$ttft,answer_names_colour:$recall,text:$text}')"
    printf '%s\n' "$row" >>"${OUT}/results.jsonl"
  done
}

compare_to_baseline() {
  local key="$1" schedule="$2" len cand base
  for len in $LENGTHS; do
    cand="$(jq -r '.choices[0].message.content // empty' "${OUT}/${key}-${schedule}-candidate.${len}.json" 2>/dev/null)"
    base="$(jq -r '.choices[0].message.content // empty' "${OUT}/${key}-${schedule}-baseline.${len}.json" 2>/dev/null)"
    if [[ -n "$base" && "$cand" == "$base" ]]; then
      pass "${key} ${schedule} ${len}: greedy text identical to the baseline build"
    else
      fail "${key} ${schedule} ${len}: greedy text differs. baseline='${base}' candidate='${cand}'"
    fi
  done
}

CAND_JAR="$(find_juno_jar)"
log "candidate jar: ${CAND_JAR} ($(sha256sum "$CAND_JAR" | cut -c1-16))"
[[ -n "$BASELINE_JAR" ]] && log "baseline jar: ${BASELINE_JAR} ($(sha256sum "$BASELINE_JAR" | cut -c1-16))"
[[ -n "$BASELINE_JAR" ]] || log "no --baseline-jar: greedy identity against an earlier build not checked"
: >"${OUT}/results.jsonl"

IFS=',' read -r -a MODEL_KEYS <<<"$MODELS"
for key in "${MODEL_KEYS[@]}"; do
  model="$(model_file "$key")"
  [[ -f "$model" ]] || { fail "${key}: model file not present (${model})"; continue; }
  heap="$(model_heap "$key")"
  ctx="$(context_length "$model")"
  calibrated=0
  for schedule in $SCHEDULES; do
    builds=(candidate)
    [[ -n "$BASELINE_JAR" ]] && builds+=(baseline)
    for build in "${builds[@]}"; do
      jar="$CAND_JAR"; [[ "$build" == baseline ]] && jar="$BASELINE_JAR"
      label="${key}-${schedule}-${build}"
      start_server "$label" "$jar" "$model" "$heap" "$schedule" || { fail "${label}: server did not start"; continue; }
      if (( calibrated == 0 )); then
        calibrate "$key" "$ctx" || { fail "${key}: prompt calibration failed"; stop_juno; continue 3; }
        calibrated=1
      fi
      run_requests "$label" "$key" "$schedule" "$build"
      stop_juno
    done
    [[ -n "$BASELINE_JAR" ]] && compare_to_baseline "$key" "$schedule"
  done
done

jq -s '.' "${OUT}/results.jsonl" >"${OUT}/results.json"
printf '\n%-10s %-11s %-9s %8s %8s %10s %7s\n' model schedule build tokens actual ttft_ms recall
jq -r '.[] | [.model, .schedule, .build, .target_tokens, .prompt_tokens, (.ttft_ms // "-"), .answer_names_colour]
  | @tsv' "${OUT}/results.json" |
  while IFS=$'\t' read -r m s b t p f r; do
    printf '%-10s %-11s %-9s %8s %8s %10.0f %7s\n' "$m" "$s" "$b" "$t" "$p" "${f/-/0}" "$r"
  done

if (( failures > 0 )); then
  printf '[prefill-smoke] %d check(s) FAILED. Output: %s\n' "$failures" "$OUT" >&2
  exit 1
fi
printf '[prefill-smoke] all checks passed. Output: %s\n' "$OUT"
