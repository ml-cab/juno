#!/usr/bin/env bash
# smoke-gpu-residency.sh — --gpu-residency on real model files, on the GPU
#
# For each model (default: tinyllama Q4_K_M, mistral-7b, llama-1-30b):
#   1. Local mode (three in-process nodes, the default) with --gpu-residency off, then on:
#      greedy output over /v1/chat/completions, on vs off, reported token for token.
#   2. The on run logs that the device-resident decode region is active (and on how many
#      layers); a model it cannot run says so instead.
#   3. Neither mode keeps device memory per request (nvidia-smi, per process; run on an otherwise
#      idle device). Read over the second half of the requests (from request max(2, ceil(N/2))):
#      each mode's GPU memory grows by at most LEAK_MIB_PER_REQUEST per request on average, and
#      no more with the region on than off. The early requests are excluded because they size
#      buffers the process keeps, and a model at the device's capacity takes several requests to
#      settle its placement, at a different pace with the region on and off. The limit sits well
#      below the per-request scratch leak this check was tightened after (23 MiB per request on
#      TinyLlama, 114 on Mistral-7B) and above the settling steps measured on LLaMA-30B at the
#      card's capacity (at most 5.5 MiB per request over such a window).
# Then, on the first model only, cluster mode with --gpu-residency on, pipeline and tensor:
#   4. The forked nodes answer, their output equals the local on-run's, and no node JVM is left.
#
# Fails on: a crash or empty response, a missing activation notice where the region should
# run, device memory that grows across requests, or a cluster run that does not answer.
# A difference between on and off output is reported with the first token where it appears,
# and fails only for the first model (tinyllama), where the region is measured to keep every
# greedy token of a 24-position decode.
#
# Environment overrides: JUNO_JAR (shaded jar to run), MODELS_DIR.
# Writes logs, responses and summary.md under target/gpu-residency-smoke/<timestamp>/.
#
# Usage:
#   ./scripts/performance-tests/smoke-gpu-residency.sh
#   ./scripts/performance-tests/smoke-gpu-residency.sh --models tinyllama --requests 3
#   ./scripts/performance-tests/smoke-gpu-residency.sh --no-cluster --n-gen 16
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/gpu-residency-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
MODELS="tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m,llama-1-30b.Q4_K_M"
API_PORT=18095
N_GEN=32
REQUESTS=4
LEAK_MIB_PER_REQUEST=8
RUN_CLUSTER=1
JUNO_PID=""
failures=0

log()  { printf '[gpu-residency-smoke] %s\n' "$*"; }
warn() { printf '[gpu-residency-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[gpu-residency-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[gpu-residency-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); echo "- FAIL: $*" >>"${OUT}/summary.md"; }
pass() { printf '[gpu-residency-smoke] PASS: %s\n' "$*"; echo "- PASS: $*" >>"${OUT}/summary.md"; }
note() { printf '[gpu-residency-smoke] %s\n' "$*"; echo "- $*" >>"${OUT}/summary.md"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --n-gen) N_GEN="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --no-cluster) RUN_CLUSTER=0; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done
(( REQUESTS >= 1 )) || die "--requests must be at least 1"
WINDOW_START=$(( (REQUESTS + 1) / 2 )); (( WINDOW_START < 2 )) && WINDOW_START=2
(( WINDOW_START > REQUESTS )) && WINDOW_START=$REQUESTS

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required (CUDA host)"
mkdir -p "$OUT"
printf '# --gpu-residency smoke — %s\n\n' "$RUN_ID" >"${OUT}/summary.md"
trap 'stop_juno' EXIT

[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] \
  || warn "the GPU is not idle; per-process memory readings still hold, device-wide ones would not"

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

resolve_model() {
  local stem="$1" f
  for f in "${MODELS_DIR}/${stem}.gguf" "${MODELS_DIR}/${stem}"; do
    [[ -f "$f" ]] && { printf '%s' "$f"; return; }
  done
  f="$(ls "${MODELS_DIR}"/*"${stem}"*.gguf 2>/dev/null | head -1)"
  [[ -n "$f" ]] && printf '%s' "$f"
}

heap_for() {
  local bytes gib
  bytes="$(stat -c %s "$1")"
  gib=$(( bytes / 1073741824 + 4 ))
  (( gib < 4 )) && gib=4
  printf '%sg' "$gib"
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
  sleep 2
}

# start_server <stem> <model> <mode-args...>
start_server() {
  local stem="$1" model="$2"; shift 2
  local logf="${OUT}/${stem}.server.log" start now
  : >"$logf"
  curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1 \
    && die "port ${API_PORT} already has a healthy Juno API"
  (
    cd "$ROOT"
    exec java --enable-preview --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
      --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms512m -Xmx"$(heap_for "$model")" -Djuno.byteOrder=BE \
      -jar "$(find_juno_jar)" --model-path "$model" --dtype FLOAT32 --byteOrder BE \
      --max-tokens "$N_GEN" --temperature 0 --top-k 0 --top-p 0 --gpu --verbose \
      --api-port "$API_PORT" "$@" \
      < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  start="$(date +%s)"
  until curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; do
    kill -0 "$JUNO_PID" 2>/dev/null || { warn "server exited early; see ${logf}"; JUNO_PID=""; return 1; }
    now="$(date +%s)"
    (( now - start < 1800 )) || { warn "server did not become healthy; see ${logf}"; stop_juno; return 1; }
    sleep 3
  done
  log "up: ${stem} (pid ${JUNO_PID})"
}

PROMPT="Write three sentences about the history of the printing press."

# ask <dest-json>  -> prints message content
ask() {
  jq -n --arg u "$PROMPT" --argjson n "$N_GEN" \
    '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n,min_tokens:$n}' |
    curl -sS --max-time 3600 -H 'Content-Type: application/json' \
      "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- -o "$1"
  jq -r '.choices[0].message.content // ""' "$1"
}

# GPU memory (MiB) held by every process in this server's tree
gpu_mib() {
  local pids total=0 line pid mem
  pids="$(pgrep -P "$JUNO_PID" 2>/dev/null) $JUNO_PID $(pgrep -f '[c]ab.ml.juno.node.NodeMain' 2>/dev/null)"
  while IFS=, read -r pid mem; do
    pid="${pid// /}"; mem="${mem// /}"; mem="${mem%MiB}"
    [[ -z "$pid" ]] && continue
    if [[ " $pids " == *" $pid "* ]]; then total=$(( total + mem )); fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  printf '%s' "$total"
}

# first_diff <a> <b>  -> index of the first differing whitespace-separated word, or -1
first_diff() {
  local -a a b; read -ra a <<<"$1"; read -ra b <<<"$2"
  local i n=$(( ${#a[@]} > ${#b[@]} ? ${#a[@]} : ${#b[@]} ))
  for (( i = 0; i < n; i++ )); do [[ "${a[i]:-}" == "${b[i]:-}" ]] || { echo "$i"; return; }; done
  echo -1
}

# run_local <stem> <model> <mode>  -> sets REPLY_TEXT, MEM_FIRST, MEM_LAST, MEM_SERIES
run_local() {
  local stem="$1" model="$2" mode="$3" r text first=""
  REPLY_TEXT=""; MEM_FIRST=""; MEM_LAST=""; MEM_SERIES=""
  start_server "${stem}-local-${mode}" "$model" --local --gpu-residency "$mode" || { fail "${stem} local ${mode}: server did not start"; return 1; }
  for (( r = 1; r <= REQUESTS; r++ )); do
    text="$(ask "${OUT}/${stem}-local-${mode}-req${r}.json")"
    [[ -n "$text" ]] || { fail "${stem} local ${mode}: request ${r} returned no text; see ${OUT}/${stem}-local-${mode}-req${r}.json"; stop_juno; return 1; }
    [[ -z "$first" ]] && first="$text"
    [[ "$text" == "$first" ]] || fail "${stem} local ${mode}: request ${r} differs from request 1 (greedy)"
    MEM_LAST="$(gpu_mib)"
    MEM_SERIES="${MEM_SERIES:+${MEM_SERIES} }${MEM_LAST}"
    if (( r == WINDOW_START || REQUESTS == 1 )); then MEM_FIRST="$MEM_LAST"; fi
  done
  REPLY_TEXT="$first"
  printf '%s\n' "$first" >"${OUT}/${stem}-local-${mode}.txt"
  stop_juno
}

check_model() {
  local stem="$1" model off_text on_text d log_on
  model="$(resolve_model "$stem")"
  [[ -n "$model" ]] || { note "${stem}: model file not present, skipped"; return; }
  printf '\n## %s\n\n' "$stem" >>"${OUT}/summary.md"

  run_local "$stem" "$model" off || return
  off_text="$REPLY_TEXT"
  local off_first="$MEM_FIRST" off_last="$MEM_LAST" off_series="$MEM_SERIES"
  note "${stem}: GPU MiB after each request, off: ${off_series}"
  run_local "$stem" "$model" on || return
  on_text="$REPLY_TEXT"
  note "${stem}: GPU MiB after each request, on:  ${MEM_SERIES}"
  log_on="${OUT}/${stem}-local-on.server.log"

  if grep -q "GPU-resident decode region active" "$log_on"; then
    pass "${stem}: $(grep -o 'GPU-resident decode region active[^:]*' "$log_on" | head -1)"
  elif grep -q "gpu-residency=.*requested, but" "$log_on"; then
    note "${stem}: declined - $(grep -o 'requested, but.*' "$log_on" | head -1)"
  else
    fail "${stem}: no activation or decline notice in ${log_on}"
  fi

  # Growth with the region on, net of whatever the same requests grow with it off.
  local on_growth=$(( MEM_LAST - MEM_FIRST )) off_growth=$(( off_last - off_first ))
  local span=$(( REQUESTS - WINDOW_START )) limit
  limit=$(( LEAK_MIB_PER_REQUEST * (span > 0 ? span : 1) ))
  if (( off_growth > limit )); then
    fail "${stem}: GPU memory grows ${off_growth} MiB with the region off over requests ${WINDOW_START}..${REQUESTS} (limit ${limit} MiB, ${LEAK_MIB_PER_REQUEST} per request)"
  elif (( on_growth > limit )); then
    fail "${stem}: GPU memory grows ${on_growth} MiB with the region on over requests ${WINDOW_START}..${REQUESTS} (limit ${limit} MiB, ${LEAK_MIB_PER_REQUEST} per request)"
  elif (( on_growth > off_growth )); then
    fail "${stem}: GPU memory grows more with the region on (${on_growth} MiB) than off (${off_growth} MiB) over requests ${WINDOW_START}..${REQUESTS}"
  else
    pass "${stem}: GPU memory growth over requests ${WINDOW_START}..${REQUESTS}: on ${on_growth} MiB, off ${off_growth} MiB (limit ${limit} MiB)"
  fi

  d="$(first_diff "$off_text" "$on_text")"
  if [[ "$d" == "-1" ]]; then
    pass "${stem}: greedy output identical on vs off (${N_GEN} tokens)"
  elif [[ "$stem" == tinyllama* ]]; then
    fail "${stem}: greedy output differs on vs off from word ${d}"
  else
    note "${stem}: greedy output differs on vs off from word ${d} (reported, not failed; see ${stem}-local-on.txt / -off.txt)"
  fi
  LOCAL_ON_TEXT="$on_text"
}

check_cluster() {
  local stem="$1" model ptype text leftover
  model="$(resolve_model "$stem")"
  [[ -n "$model" ]] || return
  printf '\n## cluster (%s)\n\n' "$stem" >>"${OUT}/summary.md"
  for ptype in pipeline tensor; do
    start_server "${stem}-cluster-${ptype}" "$model" --pType "$ptype" --gpu-residency on \
      || { fail "cluster ${ptype}: server did not start"; continue; }
    text="$(ask "${OUT}/${stem}-cluster-${ptype}.json")"
    stop_juno
    sleep 3
    leftover="$(pgrep -f '[c]ab.ml.juno.node.NodeMain' || true)"
    if [[ -z "$text" ]]; then
      fail "cluster ${ptype} on: no text; see ${OUT}/${stem}-cluster-${ptype}.json"
    elif [[ "$text" == "${LOCAL_ON_TEXT:-}" ]]; then
      pass "cluster ${ptype} on: answers, output equals local mode's"
    else
      note "cluster ${ptype} on: answers; output differs from local mode's from word $(first_diff "$LOCAL_ON_TEXT" "$text")"
    fi
    [[ -z "$leftover" ]] && pass "cluster ${ptype}: no node JVM left" || fail "cluster ${ptype}: node JVMs left: ${leftover}"
  done
}

IFS=',' read -ra MODEL_LIST <<<"$MODELS"
LOCAL_ON_TEXT=""
first_stem=""
for stem in "${MODEL_LIST[@]}"; do
  check_model "$stem"
  if [[ -z "$first_stem" ]]; then
    first_stem="$stem"
    (( RUN_CLUSTER )) && check_cluster "$stem"
  fi
done

printf '\nFailures: %s\n' "$failures" >>"${OUT}/summary.md"
log "summary: ${OUT}/summary.md (failures=${failures})"
(( failures == 0 ))
