#!/usr/bin/env bash
# smoke-tier00-consistency.sh — correctness / fail-closed checks against real model files
#
# 1. gguf-info reports the expected general.architecture for each audited file.
# 2. Local mode and cluster mode (pipeline and tensor) reject the four architectures that have
#    no verified handler (qwen35, gemma4, mistral3, minimax-m2) with an error that names the
#    architecture, print no generated text, and leave no node JVMs running.
# 3. Local mode still loads and answers on a known-good file (tinyllama, llama).
# 4. Static batching (--parallel 2) never resumes from KV it did not write: with a
#    repeated prompt in a later batch, every response over /v1/chat/completions and
#    /v1/inference equals the single-request response for that prompt, and the server
#    reports zero prefix-cache hits for stateless traffic. Runs on CPU, and on the GPU
#    when one is present (skip with --no-gpu).
#
# 5. A session request (x_juno_session_id) that repeats a prompt already served to stateless
#    traffic must prefill in full, on the static schedule after batched traffic and on the
#    continuous schedule: its response equals the single-request response.
#
# Environment overrides: JUNO_JAR (shaded jar to run), MODELS_DIR (directory holding the models).
# Writes logs and response bodies under target/consistency-smoke/<timestamp>/.
# Exits 0 when every check passes, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-tier00-consistency.sh
#   ./scripts/performance-tests/smoke-tier00-consistency.sh --no-gpu --api-port 18090
#   ./scripts/performance-tests/smoke-tier00-consistency.sh --skip-audit   # only the batching checks
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/consistency-smoke/${RUN_ID}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
SERVER_PARALLEL=2
GOOD_MODEL="${MODELS_DIR}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
API_PORT=18090
RUN_GPU=1
RUN_AUDIT=1
JUNO_PID=""
failures=0

log()  { printf '[consistency-smoke] %s\n' "$*"; }
warn() { printf '[consistency-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[consistency-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[consistency-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[consistency-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port) API_PORT="$2"; shift 2 ;;
    --no-gpu) RUN_GPU=0; shift ;;
    --skip-audit) RUN_AUDIT=0; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
mkdir -p "$OUT"
trap 'stop_juno' EXIT

# file : general.architecture
UNSUPPORTED=(
  "Qwen3.5-0.8B.Q4_K_M.gguf:qwen35"
  "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf:gemma4"
  "Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S.gguf:mistral3"
  "minimax-m2.5-tiny-24e-iq4_nl-imat.gguf:minimax-m2"
)

# ── 1 + 2: architecture audit ────────────────────────────────────────────────

audit_unsupported() {
  local entry file arch path info out rc
  for entry in "${UNSUPPORTED[@]}"; do
    file="${entry%%:*}"; arch="${entry##*:}"; path="${MODELS_DIR}/${file}"
    if [[ ! -f "$path" ]]; then fail "${file}: model file not present"; continue; fi

    info="${OUT}/${file}.gguf-info.txt"
    timeout 300 "${ROOT}/juno" gguf-info --model-path "$path" >"$info" 2>&1
    if grep -q "general.architecture = ${arch}\$" "$info"; then
      pass "gguf-info ${file}: general.architecture = ${arch}"
    else
      fail "gguf-info ${file}: expected general.architecture = ${arch}"
    fi

    out="${OUT}/${file}.local.txt"
    printf 'hello\nexit\n' | timeout 600 "${ROOT}/juno" local --model-path "$path" --cpu --nodes 1 \
      --max-tokens 8 --temperature 0 >"$out" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]] && grep -q "Unsupported model architecture '${arch}'" "$out"; then
      pass "local ${file}: rejected, error names '${arch}' (exit ${rc})"
    else
      fail "local ${file}: expected non-zero exit and \"Unsupported model architecture '${arch}'\" (exit ${rc}); see ${out}"
    fi

    local ptype
    for ptype in pipeline tensor; do
      out="${OUT}/${file}.cluster-${ptype}.txt"
      printf 'hello\nexit\n' | timeout 600 "${ROOT}/juno" cluster --model-path "$path" --pType "$ptype" --cpu \
        --max-tokens 8 --temperature 0 >"$out" 2>&1
      rc=$?
      sleep 2
      if [[ $rc -ne 0 ]] && grep -q "Unsupported model architecture '${arch}'" "$out" \
          && ! pgrep -f '[c]ab.ml.juno.node.NodeMain' >/dev/null; then
        pass "cluster ${ptype} ${file}: start fails naming '${arch}', no node JVMs left (exit ${rc})"
      else
        fail "cluster ${ptype} ${file}: expected non-zero exit, \"Unsupported model architecture '${arch}'\" and no leftover nodes (exit ${rc}); see ${out}"
      fi
    done
  done
}

audit_known_good() {
  local out="${OUT}/tinyllama.local.txt" rc
  [[ -f "$GOOD_MODEL" ]] || { fail "tinyllama: model file not present"; return; }
  printf 'hello\nexit\n' | timeout 900 "${ROOT}/juno" local --model-path "$GOOD_MODEL" --cpu --nodes 1 \
    --max-tokens 8 --temperature 0 >"$out" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]] && ! grep -q "Unsupported model architecture\|Exception in thread" "$out"; then
    pass "local tinyllama: loads and exits cleanly (exit ${rc})"
  else
    fail "local tinyllama: expected clean exit without an architecture rejection (exit ${rc}); see ${out}"
  fi
}

# ── 4: static batching invariant over REST ───────────────────────────────────

find_juno_jar() {
  if [[ -n "${JUNO_JAR:-}" ]]; then printf '%s' "$JUNO_JAR"; return; fi
  local jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "shaded jar missing; run: mvn package -DskipTests"
  printf '%s' "${jars[0]}"
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
}

start_server() {
  local stem="$1"; shift
  local logf="${OUT}/${stem}.server.log" start now
  : >"$logf"
  curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1 \
    && die "port ${API_PORT} already has a healthy Juno API"
  (
    cd "$ROOT"
    exec java --enable-preview --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
      --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms512m -Xmx4g -Djuno.byteOrder=BE \
      -jar "$(find_juno_jar)" --model-path "$GOOD_MODEL" --dtype FLOAT16 --byteOrder BE \
      --max-tokens 8 --temperature 0 --top-k 0 --top-p 0 --nodes 1 --local \
      --api-port "$API_PORT" --parallel "$SERVER_PARALLEL" --batch-window-ms 300 "$@" \
      < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  start="$(date +%s)"
  until curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; do
    kill -0 "$JUNO_PID" 2>/dev/null || { warn "server exited early; see ${logf}"; JUNO_PID=""; return 1; }
    now="$(date +%s)"
    (( now - start < 300 )) || { warn "server did not become healthy; see ${logf}"; stop_juno; return 1; }
    sleep 2
  done
  log "up: ${stem} (pid ${JUNO_PID})"
}

SYSTEM_PROMPT="You are a helpful assistant. Answer in one short sentence."
PROMPTS=("What is the capital of France?" "Name one primary color." "How many legs does a spider have?")

# Requests use a plain temperature of 0, which selects greedy decoding, so the responses can be
# compared across requests and across batch composition.
# openai <user-prompt>  -> prints message content
openai() {
  jq -n --arg s "$SYSTEM_PROMPT" --arg u "$1" \
    '{messages:[{role:"system",content:$s},{role:"user",content:$u}],temperature:0,max_tokens:8}' |
    curl -sS -H 'Content-Type: application/json' "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- |
    jq -r '.choices[0].message.content // ("ERROR: " + (. | tostring))'
}

# openai_session <session-id> <user-prompt>  -> prints message content
openai_session() {
  jq -n --arg s "$SYSTEM_PROMPT" --arg u "$2" --arg sid "$1" \
    '{messages:[{role:"system",content:$s},{role:"user",content:$u}],temperature:0,max_tokens:8,x_juno_session_id:$sid}' |
    curl -sS -H 'Content-Type: application/json' "http://127.0.0.1:${API_PORT}/v1/chat/completions" -d @- |
    jq -r '.choices[0].message.content // ("ERROR: " + (. | tostring))'
}

# native <model-id> <user-prompt>  -> prints text
native() {
  jq -n --arg m "$1" --arg s "$SYSTEM_PROMPT" --arg u "$2" \
    '{modelId:$m,messages:[{role:"system",content:$s},{role:"user",content:$u}],sampling:{temperature:0,maxTokens:8}}' |
    curl -sS -H 'Content-Type: application/json' "http://127.0.0.1:${API_PORT}/v1/inference" -d @- |
    jq -r '.text // ("ERROR: " + (. | tostring))'
}

# round <surface-fn> <extra-arg> <label> <prompt-index>...  : run all prompts concurrently, compare to baseline
round() {
  local fn="$1" extra="$2" label="$3"; shift 3
  local idx i pids=() files=()
  for idx in "$@"; do
    f="${OUT}/${label}.${idx}.$$.${#files[@]}.txt"; files+=("$f")
    if [[ -n "$extra" ]]; then ( "$fn" "$extra" "${PROMPTS[$idx]}" >"$f" ) & else ( "$fn" "${PROMPTS[$idx]}" >"$f" ) & fi
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i"; done
  local n=0
  for idx in "$@"; do
    local got; got="$(cat "${files[$n]}")"
    if [[ "$got" == "${BASELINE[$idx]}" ]]; then
      pass "${label}: prompt ${idx} matches single-request output"
    else
      fail "${label}: prompt ${idx} differs. single='${BASELINE[$idx]}' batched='${got}'"
    fi
    n=$((n + 1))
  done
}

static_batch_invariant() {
  local mode="$1"; shift
  local model_id hits lookups idx
  start_server "static-batch-${mode}" "$@" || { fail "static-batch ${mode}: server did not start"; return; }
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" | jq -r '.data[0].id // empty')"
  [[ -n "$model_id" ]] || { fail "static-batch ${mode}: no model id from /v1/models"; stop_juno; return; }

  declare -gA BASELINE=()
  for idx in 0 1 2; do BASELINE[$idx]="$(openai "${PROMPTS[$idx]}")"; done
  log "baseline (${mode}): 0='${BASELINE[0]}' 1='${BASELINE[1]}' 2='${BASELINE[2]}'"

  round openai "" "openai-${mode}-round1" 0 1 2 0
  round openai "" "openai-${mode}-round2" 0 0 1 2
  round native "$model_id" "native-${mode}-round1" 0 1 2 0
  round native "$model_id" "native-${mode}-round2" 0 0 1 2

  local sess
  sess="$(openai_session "smoke-${mode}-session" "${PROMPTS[0]}")"
  if [[ "$sess" == "${BASELINE[0]}" ]]; then
    pass "static-batch ${mode}: session request after batched stateless traffic matches single-request output"
  else
    fail "static-batch ${mode}: session request differs. single='${BASELINE[0]}' session='${sess}'"
  fi

  hits="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" | jq -r '.prefixHits // 0')"
  lookups="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" | jq -r '.prefixLookups // 0')"
  # the session request above is the only one allowed to consult the trie, and it starts empty
  if [[ "$hits" == "0" ]]; then
    pass "static-batch ${mode}: zero prefix-cache hits (lookups=${lookups})"
  else
    fail "static-batch ${mode}: ${hits} prefix-cache hits"
  fi
  stop_juno
}

continuous_session_after_stateless() {
  local base sess
  SERVER_PARALLEL=4
  start_server "continuous-cpu" --cpu --schedule continuous || { fail "continuous: server did not start"; SERVER_PARALLEL=2; return; }
  SERVER_PARALLEL=2
  base="$(openai "${PROMPTS[0]}")"
  sess="$(openai_session "smoke-continuous-session" "${PROMPTS[0]}")"
  if [[ "$base" == "$sess" && "$base" != ERROR* ]]; then
    pass "continuous: session request after a stateless request matches it ('${base}')"
  else
    fail "continuous: session request differs. stateless='${base}' session='${sess}'"
  fi
  stop_juno
}

if [[ $RUN_AUDIT -eq 1 ]]; then
  audit_unsupported
  audit_known_good
fi
if [[ -f "$GOOD_MODEL" ]]; then
  static_batch_invariant cpu --cpu
  continuous_session_after_stateless
  if [[ $RUN_GPU -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    static_batch_invariant gpu --gpu
  else
    log "skipping GPU static-batch check (no GPU or --no-gpu)"
  fi
fi

if (( failures > 0 )); then
  printf '[consistency-smoke] %d check(s) FAILED. Output: %s\n' "$failures" "$OUT" >&2
  exit 1
fi
printf '[consistency-smoke] all checks passed. Output: %s\n' "$OUT"
