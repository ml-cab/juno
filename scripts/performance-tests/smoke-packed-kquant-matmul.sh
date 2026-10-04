#!/usr/bin/env bash
# smoke-packed-kquant-matmul.sh — packed K-quant prefill matmul on real model files over /v1/chat/completions
#
# A prefill window wider than 8 rows multiplies the still-packed Q4_K/Q5_K/Q6_K weights with the
# tiled integer kernel instead of expanding them to FP16. For each model (the four sweep models by
# default: TinyLlama, Qwen2.5-3B, Phi-3.5-mini and Mistral 7B, all Q4_K_M) and each schedule
# (static and continuous), on the GPU:
#   1. a greedy (temperature 0) streamed request prefills a prompt of about 512 tokens and generates
#      exactly 512 tokens (min_tokens = max_tokens): HTTP 200, usage.prompt_tokens within 10% of the
#      requested length, usage.completion_tokens equal to the requested count, one stream chunk per
#      generated token; its time to first token (TTFT) is recorded;
#   2. the same request unstreamed returns the same text: two identical greedy requests in one
#      process give the same 512 tokens (determinism);
#   3. the server log shows the tiled kernel module loaded and the fused packed weights enabled, and
#      no fallback to the dequantizing route;
#   4. the engine process's peak device memory during the requests is sampled and recorded;
#   5. with --baseline-jar (a build whose prefill still dequantizes to FP16), the same requests are
#      sent to that build and the generated tokens are compared position by position. The first
#      difference (the agreeing prefix) and the positions agreeing anywhere are recorded and printed,
#      not asserted: over hundreds of greedy tokens any change in rounding eventually flips a
#      near-tie, so where that happens characterises the two builds rather than ranking them.
#
# Environment overrides: JUNO_JAR (shaded jar under test), MODELS_DIR, SMOKE_HEAP (JVM -Xmx).
# Writes logs, responses and results.json under target/packed-kquant-smoke/<timestamp>/.
# Exits 0 when every check passes, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-packed-kquant-matmul.sh
#   ./scripts/performance-tests/smoke-packed-kquant-matmul.sh --baseline-jar /path/to/earlier-shaded.jar
#   ./scripts/performance-tests/smoke-packed-kquant-matmul.sh --models tinyllama --schedules static
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/packed-kquant-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
API_PORT=18092
MODELS="tinyllama,qwen,phi35,mistral"
SCHEDULES="static continuous"
PROMPT_TOKENS=512
GEN_TOKENS=512
BASELINE_JAR=""
JUNO_PID=""
SAMPLER_PID=""
failures=0

log()  { printf '[packed-kquant-smoke] %s\n' "$*"; }
warn() { printf '[packed-kquant-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[packed-kquant-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[packed-kquant-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[packed-kquant-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --baseline-jar) BASELINE_JAR="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

for tool in jq curl python3 nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || die "${tool} is required"
done
[[ -z "$BASELINE_JAR" || -f "$BASELINE_JAR" ]] || die "baseline jar not found: ${BASELINE_JAR}"
mkdir -p "$OUT"
trap 'stop_juno' EXIT

model_file() {
  case "$1" in
    tinyllama) printf '%s' "${MODELS_DIR}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ;;
    qwen) printf '%s' "${MODELS_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf" ;;
    phi35) printf '%s' "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf" ;;
    mistral) printf '%s' "${MODELS_DIR}/mistral-7b-instruct-v0.1-q4_k_m.gguf" ;;
    *) die "unknown model key: $1 (tinyllama, qwen, phi35 or mistral)" ;;
  esac
}

model_heap() {
  if [[ -n "${SMOKE_HEAP:-}" ]]; then printf '%s' "$SMOKE_HEAP"; return; fi
  case "$1" in
    mistral) printf '8g' ;;
    qwen|phi35) printf '6g' ;;
    *) printf '4g' ;;
  esac
}

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

# Samples the engine's device memory (MiB) every 200 ms into <file> until stopped.
start_sampler() {
  local pid="$1" file="$2"
  : >"$file"
  (
    while kill -0 "$pid" 2>/dev/null; do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' -v p="$pid" '$1 == p { print $2 }' >>"$file"
      sleep 0.2
    done
  ) &
  SAMPLER_PID=$!
}

stop_sampler() {
  [[ -n "${SAMPLER_PID:-}" ]] || return 0
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
  SAMPLER_PID=""
}

peak_of() { sort -n "$1" 2>/dev/null | tail -1; }

stop_juno() {
  stop_sampler
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
      --max-tokens "$GEN_TOKENS" --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local --gpu \
      --verbose --api-port "$API_PORT" --schedule "$schedule"
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
# filler words making up the remainder, then a request for a long answer, so that the 512
# generated tokens are a continuation the model is actually writing.
notes_prompt() {
  local n="$1" w="${2:-0}" i
  printf 'Read the notes below, then answer the request at the end.\n'
  for (( i = 1; i <= n; i++ )); do
    printf 'Note %d: the %s in district %d was painted %s in the year %d.\n' \
      "$i" "${PLACES[$((i % 8))]}" "$((i * 7 % 31))" "${COLOURS[$((i * 3 % 8))]}" "$((1800 + i * 13))"
  done
  if (( w > 0 )); then
    printf 'Remark:'
    for (( i = 0; i < w; i++ )); do printf ' %s' "${FILLER[$((i % ${#FILLER[@]}))]}"; done
    printf '.\n'
  fi
  printf 'Request: retell every note in order as a short story, one paragraph per note.'
}

# chat <prompt> <max-tokens> <min-tokens> <dest-json>: unstreamed greedy request; prints the HTTP status.
chat() {
  jq -n --arg u "$1" --argjson n "$2" --argjson m "$3" \
    '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n,min_tokens:$m}' |
    curl -sS --max-time 3600 -o "$4" -w '%{http_code}' -H 'Content-Type: application/json' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" || printf '000'
}

# chat_stream <prompt> <dest-json>: streamed greedy request; writes {ttft_ms, chunks:[...]}, one
# entry per stream chunk that carries content.
chat_stream() {
  jq -n --arg u "$1" --argjson n "$GEN_TOKENS" \
    '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n,min_tokens:$n,stream:true}' |
    curl -sS --max-time 3600 -N -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- 2>>"${OUT}/curl.err" |
    python3 -c '
import json, sys, time
t0 = time.time() * 1000.0
ttft, chunks = None, []
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
    if c is not None and c != "":
        if ttft is None:
            ttft = time.time() * 1000.0 - t0
        chunks.append(c)
json.dump({"ttft_ms": ttft, "chunks": chunks}, open(sys.argv[1], "w"))
' "$2"
}

# prompt_tokens_of <note-count> <filler-words> <dest-json>: the engine's own count, or 0
prompt_tokens_of() {
  [[ "$(chat "$(notes_prompt "$1" "$2")" 1 0 "$3")" == 200 ]] || { printf '0'; return; }
  jq -r '.usage.prompt_tokens // 0' "$3"
}

# Calibrates note and filler-word counts against the engine's own usage.prompt_tokens (a linear
# fit over notes, then filler words for the remainder with one correction). Tokenization depends on
# neither the schedule nor the build, so it runs once per model.
NOTES=0
WORDS=0
calibrate() {
  local key="$1" a=2 b=40 ta tb per n w t
  ta="$(prompt_tokens_of "$a" 0 "${OUT}/${key}.calib-a.json")"
  tb="$(prompt_tokens_of "$b" 0 "${OUT}/${key}.calib-b.json")"
  (( ta > 0 && tb > ta )) || return 1
  per=$(( (tb - ta) * 100 / (b - a) ))
  n=$(( a + (PROMPT_TOKENS - ta) * 100 / per ))
  (( n < 1 )) && n=1
  t="$(prompt_tokens_of "$n" 0 "${OUT}/${key}.calib-0.json")"
  (( t > 0 )) || return 1
  w=$(( PROMPT_TOKENS - t > 0 ? PROMPT_TOKENS - t : 0 ))
  if (( w > 0 )); then
    t="$(prompt_tokens_of "$n" "$w" "${OUT}/${key}.calib-1.json")"
    (( t > 0 )) || return 1
    w=$(( w + PROMPT_TOKENS - t )); (( w < 0 )) && w=0
  fi
  NOTES="$n"; WORDS="$w"
  log "${key}: ${PROMPT_TOKENS} -> ${n} notes + ${w} filler words"
}

# run_cell <label> <key> <schedule> <build>: one streamed and one unstreamed request; records results
run_cell() {
  local label="$1" key="$2" schedule="$3" build="$4"
  local prompt resp="${OUT}/${label}.json" sresp="${OUT}/${label}.stream.json" vram="${OUT}/${label}.vram"
  local code text pt ct ttft stext nchunks peak logf="${OUT}/${label}.server.log"
  prompt="$(notes_prompt "$NOTES" "$WORDS")"
  start_sampler "$JUNO_PID" "$vram"
  chat_stream "$prompt" "$sresp"
  code="$(chat "$prompt" "$GEN_TOKENS" "$GEN_TOKENS" "$resp")"
  stop_sampler
  text="$(jq -r '.choices[0].message.content // empty' "$resp" 2>/dev/null)"
  pt="$(jq -r '.usage.prompt_tokens // 0' "$resp" 2>/dev/null)"
  ct="$(jq -r '.usage.completion_tokens // 0' "$resp" 2>/dev/null)"
  ttft="$(jq -r '.ttft_ms // "null"' "$sresp" 2>/dev/null)"
  stext="$(jq -r '.chunks | join("")' "$sresp" 2>/dev/null)"
  nchunks="$(jq -r '.chunks | length' "$sresp" 2>/dev/null)"
  peak="$(peak_of "$vram")"

  if [[ "$build" == candidate ]]; then
    if [[ "$code" == 200 && -n "${text//[[:space:]]/}" ]]; then
      pass "${label}: HTTP 200, non-empty answer"
    else
      fail "${label}: HTTP ${code}, answer '${text:0:80}'; see ${resp}"
    fi
    if [[ "$pt" =~ ^[0-9]+$ ]] && (( pt * 10 >= PROMPT_TOKENS * 9 && pt * 10 <= PROMPT_TOKENS * 11 )); then
      pass "${label}: prompt_tokens ${pt} within 10% of ${PROMPT_TOKENS}"
    else
      fail "${label}: prompt_tokens ${pt}, requested ${PROMPT_TOKENS}"
    fi
    if [[ "$ct" == "$GEN_TOKENS" && "$nchunks" == "$GEN_TOKENS" ]]; then
      pass "${label}: ${GEN_TOKENS} tokens generated, one stream chunk each"
    else
      fail "${label}: completion_tokens ${ct}, stream chunks ${nchunks}, requested ${GEN_TOKENS}"
    fi
    if [[ "$stext" == "$text" ]]; then
      pass "${label}: streamed text equals unstreamed (TTFT ${ttft} ms)"
    else
      fail "${label}: streamed text differs from unstreamed; see ${sresp} and ${resp}"
    fi
    if grep -q 'Tiled K-quant GEMM kernels loaded' "$logf" && grep -q 'Fused Q4_K MMQ enabled' "$logf" \
        && ! grep -q 'Tiled K-quant GEMM kernel unavailable' "$logf"; then
      pass "${label}: packed weights resident and the tiled kernel loaded, no dequantizing fallback"
    else
      fail "${label}: server log does not show the packed path active; see ${logf}"
    fi
    if [[ "$peak" =~ ^[0-9]+$ ]]; then
      pass "${label}: peak device memory ${peak} MiB"
    else
      fail "${label}: no device memory reading for the engine process"
    fi
  fi
  jq -nc --arg model "$key" --arg schedule "$schedule" --arg build "$build" --argjson pt "${pt:-0}" \
    --argjson ct "${ct:-0}" --arg code "$code" --argjson ttft "${ttft:-null}" \
    --argjson peak "${peak:-null}" \
    '{model:$model,schedule:$schedule,build:$build,prompt_tokens:$pt,completion_tokens:$ct,http:$code,
      ttft_ms:$ttft,peak_vram_mib:$peak}' >>"${OUT}/results.jsonl"
}

# compare_to_baseline <key> <schedule>: records the agreeing prefix and positionwise agreement of the
# streamed tokens (reported, not asserted)
compare_to_baseline() {
  local key="$1" schedule="$2" row prefix agree
  row="$(python3 - "${OUT}/${key}-${schedule}-candidate.stream.json" \
      "${OUT}/${key}-${schedule}-baseline.stream.json" "$GEN_TOKENS" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["chunks"]
b = json.load(open(sys.argv[2]))["chunks"]
n = int(sys.argv[3])
prefix = 0
while prefix < min(len(c), len(b), n) and c[prefix] == b[prefix]:
    prefix += 1
same = sum(1 for i in range(min(len(c), len(b), n)) if c[i] == b[i])
print(json.dumps({"prefix": prefix, "positionwise": same, "of": n,
                  "first_difference": None if prefix >= n else prefix}))
PY
)"
  prefix="$(jq -r '.prefix' <<<"$row")"
  agree="$(awk -v p="$prefix" -v n="$GEN_TOKENS" 'BEGIN{printf "%.4f", p / n}')"
  jq -c --arg model "$key" --arg schedule "$schedule" --argjson agreement "$agree" \
    '. + {model:$model,schedule:$schedule,agreement:$agreement}' <<<"$row" >>"${OUT}/agreement.jsonl"
  log "${key} ${schedule}: first difference from the baseline after ${prefix} of ${GEN_TOKENS} tokens; positionwise $(jq -r '.positionwise' <<<"$row") agree (recorded)"
}

CAND_JAR="$(find_juno_jar)"
log "candidate jar: ${CAND_JAR} ($(sha256sum "$CAND_JAR" | cut -c1-16))"
if [[ -n "$BASELINE_JAR" ]]; then
  log "baseline jar: ${BASELINE_JAR} ($(sha256sum "$BASELINE_JAR" | cut -c1-16))"
else
  log "no --baseline-jar: greedy agreement against the dequantizing route not checked"
fi
: >"${OUT}/results.jsonl"
: >"${OUT}/agreement.jsonl"

IFS=',' read -r -a MODEL_KEYS <<<"$MODELS"
for key in "${MODEL_KEYS[@]}"; do
  model="$(model_file "$key")"
  [[ -f "$model" ]] || { fail "${key}: model file not present (${model})"; continue; }
  heap="$(model_heap "$key")"
  calibrated=0
  for schedule in $SCHEDULES; do
    builds=(candidate)
    [[ -n "$BASELINE_JAR" ]] && builds+=(baseline)
    for build in "${builds[@]}"; do
      jar="$CAND_JAR"; [[ "$build" == baseline ]] && jar="$BASELINE_JAR"
      label="${key}-${schedule}-${build}"
      start_server "$label" "$jar" "$model" "$heap" "$schedule" || { fail "${label}: server did not start"; continue; }
      if (( calibrated == 0 )); then
        calibrate "$key" || { fail "${key}: prompt calibration failed"; stop_juno; continue 3; }
        calibrated=1
      fi
      run_cell "$label" "$key" "$schedule" "$build"
      stop_juno
    done
    [[ -n "$BASELINE_JAR" ]] && compare_to_baseline "$key" "$schedule"
  done
done

jq -s '.' "${OUT}/results.jsonl" >"${OUT}/results.json"
jq -s '.' "${OUT}/agreement.jsonl" >"${OUT}/agreement.json"
printf '\n%-10s %-11s %-9s %7s %7s %10s %9s\n' model schedule build prompt gen ttft_ms vram_mib
jq -r '.[] | [.model, .schedule, .build, .prompt_tokens, .completion_tokens, (.ttft_ms // 0), (.peak_vram_mib // 0)]
  | @tsv' "${OUT}/results.json" |
  while IFS=$'\t' read -r m s b p g f v; do
    printf '%-10s %-11s %-9s %7s %7s %10.0f %9s\n' "$m" "$s" "$b" "$p" "$g" "$f" "$v"
  done
if [[ -n "$BASELINE_JAR" ]]; then
  printf '\n%-10s %-11s %8s %12s %9s\n' model schedule prefix positionwise agreement
  jq -r '.[] | [.model, .schedule, .prefix, .positionwise, .agreement] | @tsv' "${OUT}/agreement.json" |
    while IFS=$'\t' read -r m s p q a; do printf '%-10s %-11s %8s %12s %9s\n' "$m" "$s" "$p" "$q" "$a"; done
fi

if (( failures > 0 )); then
  printf '[packed-kquant-smoke] %d check(s) FAILED. Output: %s\n' "$failures" "$OUT" >&2
  exit 1
fi
printf '[packed-kquant-smoke] all checks passed. Output: %s\n' "$OUT"
