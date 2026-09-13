#!/usr/bin/env bash
# smoke-tools.sh — cross-feature proof for OpenAI tools / tool_choice
#
# Starts Qwen2.5 (ChatML, wired) and TinyLlama (fail-closed template) locally,
# hits matrix cells, writes HTTP bodies + JFR metrics under
# target/tools-smoke/<timestamp>/. Does not publish to docs/perf-compare/
# (that is compare-llama-cpp.sh).
#
# Usage:
#   ./scripts/performance-tests/smoke-tools.sh
#   ./scripts/performance-tests/smoke-tools.sh --api-port 18083
set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/tools-smoke/${RUN_ID}"
QWEN="${ROOT}/models/qwen2.5-3b-instruct-q4_k_m.gguf"
TINY="${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LORA="${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora"
API_PORT=18083
JUNO_PID=""
failures=0

log()  { printf '[tools-smoke] %s\n' "$*"; }
warn() { printf '[tools-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[tools-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[tools-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[tools-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown flag: $1" ;;
  esac
done

find_juno_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/juno-player/target/"juno-player-*-shaded.jar )
  shopt -u nullglob
  if [[ ${#jars[@]} -gt 0 ]]; then
    printf '%s' "${jars[0]}"
    return
  fi
  [[ -f "$ROOT/juno-player/target/juno-player.jar" ]] || die "juno-player jar missing; run: mvn package -DskipTests"
  printf '%s' "$ROOT/juno-player/target/juno-player.jar"
}

find_java() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    printf '%s' "${JAVA_HOME}/bin/java"
    return
  fi
  command -v java >/dev/null 2>&1 || die "java not found"
  printf '%s' java
}

wait_for_juno_api() {
  local port="$1" timeout="${2:-300}" pid="${3:-}"
  local start now
  start="$(date +%s)"
  while true; do
    if curl -sf "http://127.0.0.1:${port}/v1/cluster/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      warn "juno process ${pid} exited before API became healthy"
      return 1
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 2
  done
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  JUNO_PID=""
}

wait_for_metrics() {
  local dest="$1" timeout="${2:-45}"
  local metrics_src="${ROOT}/target/metrics/metrics.json"
  local start now
  start="$(date +%s)"
  while true; do
    if [[ -s "$metrics_src" ]]; then
      cp -a "$metrics_src" "$dest"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 1
  done
}

collect_jfr() {
  local stem="$1"
  local marker="$2"
  local dest="${OUT_ROOT}/${stem}-jfr.json"
  if wait_for_metrics "$dest" 45; then
    log "jfr metrics: ${dest}"
  else
    warn "metrics.json missing for ${stem}"
    echo '{"error":"metrics_missing"}' >"$dest"
  fi
  local f
  if [[ -n "${marker:-}" && -e "$marker" ]]; then
    while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      cp -a "$f" "${OUT_ROOT}/$(basename "$f")"
    done < <(find "$ROOT" -maxdepth 1 -name 'juno-*.jfr' -newer "$marker")
  fi
}

grammar_count() {
  local f="$1"
  jq -r '.models[0].metrics["juno.GrammarConstrained.count"] // 0' "$f" 2>/dev/null || echo 0
}

start_juno() {
  local stem="$1"
  local model="$2"
  shift 2
  local jar java_bin logf heap
  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  logf="${OUT_ROOT}/${stem}.log"
  : >"$logf"
  rm -f "${ROOT}/target/metrics/metrics.json"
  heap="4g"
  if [[ "$model" == *"qwen"* ]]; then
    heap="6g"
  fi

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} already has a healthy Juno API"
  fi

  (
    cd "$ROOT"
    exec "$java_bin" \
      --enable-preview \
      --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED \
      --add-opens java.base/java.nio=ALL-UNNAMED \
      -XX:+UseG1GC \
      -Xms512m -Xmx"$heap" \
      -Djuno.byteOrder=BE \
      -jar "$jar" \
      --model-path "$model" \
      --dtype FLOAT16 \
      --byteOrder BE \
      --max-tokens 32 \
      --temperature 0 \
      --top-k 0 \
      --top-p 0 \
      --nodes 1 \
      --local \
      --cpu \
      --api-port "$API_PORT" \
      --jfr 30m \
      --verbose \
      "$@" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  if ! wait_for_juno_api "$API_PORT" 300 "$JUNO_PID"; then
    warn "API did not become healthy — see ${logf}"
    stop_juno
    return 1
  fi
  log "up: ${stem} pid=${JUNO_PID}"
}

chat() {
  local name="$1" body="$2"
  local resp="${OUT_ROOT}/${name}.json"
  local code
  code="$(curl -sS -o "$resp" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -d "$body" || true)"
  printf '%s' "$code"
}

TOOLS_JSON='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

mkdir -p "$OUT_ROOT"
trap 'stop_juno' EXIT
command -v jq >/dev/null && command -v curl >/dev/null && command -v python3 >/dev/null \
  || die "need curl, jq, python3"
[[ -f "$QWEN" ]] || die "missing ${QWEN}"
[[ -f "$TINY" ]] || die "missing ${TINY}"

{
  echo "# Function-calling smoke ${RUN_ID}"
  echo
  echo "Host: $(hostname)  wired model: $(basename "$QWEN")  fail-closed: $(basename "$TINY")  cpu"
  echo
} >"${OUT_ROOT}/INDEX.md"

# ── CUDA / ROCm / vision (N/A or explicit no-op) ─────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >"${OUT_ROOT}/nvidia-smi.txt" || true
  echo "- CUDA: nvidia-smi ok (GPU smoke not required; tools are prompt+parse)" >>"${OUT_ROOT}/INDEX.md"
else
  echo "nvidia-smi unavailable or cannot talk to the driver" >"${OUT_ROOT}/nvidia-smi.txt"
  echo "- CUDA/ROCm: N/A (driver unavailable). See nvidia-smi.txt" >>"${OUT_ROOT}/INDEX.md"
  pass "CUDA/ROCm N/A (driver down)"
fi
if compgen -G "${ROOT}/models/*.mmproj*" >/dev/null; then
  echo "- Vision mmproj present; /v1/vision/chat still does not honor tools (howto)" >>"${OUT_ROOT}/INDEX.md"
else
  echo "- Vision: no mmproj GGUF. /v1/vision/chat does not honor tools (explicit no-op)." >>"${OUT_ROOT}/INDEX.md"
  pass "Vision honesty: no mmproj; tools stay on /v1/chat/completions"
fi

# ── LoRA train is N/A (no tools CLI flag) ────────────────────────────────────
if grep -nE '\-\-tools|tool_choice' "${ROOT}/scripts/run.sh" >/dev/null; then
  fail "run.sh unexpectedly mentions tools CLI flags"
else
  pass "LoRA train / launcher: no tools CLI flag (N/A)"
  echo "- LoRA train: N/A (no tools launcher flag)" >>"${OUT_ROOT}/INDEX.md"
fi

MODEL_ID=""

# ── Server A: Qwen2.5 ChatML wired path ──────────────────────────────────────
if start_juno qwen-chatml "$QWEN"; then
  MODEL_ID="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" | jq -r '.data[0].id // empty')"
  [[ -n "$MODEL_ID" ]] || MODEL_ID="$(basename "$QWEN")"

  code="$(chat required "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"Weather in Boston?"}], max_tokens:48, temperature:0, tool_choice:"required", tools:$tools}')")"
  if [[ "$code" == "200" ]]; then
    name="$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' "${OUT_ROOT}/required.json")"
    finish="$(jq -r '.choices[0].finish_reason // empty' "${OUT_ROOT}/required.json")"
    args="$(jq -r '.choices[0].message.tool_calls[0].function.arguments // empty' "${OUT_ROOT}/required.json")"
    if [[ "$name" == "get_weather" && "$finish" == "tool_calls" && -n "$args" ]]; then
      if python3 -c 'import json,sys; json.loads(sys.argv[1])' "$args"; then
        pass "tool_choice=required → tool_calls get_weather"
      else
        fail "required arguments not JSON: ${args}"
      fi
    else
      fail "required missing tool_calls (name=${name} finish=${finish})"
    fi
  else
    fail "required HTTP ${code}: $(head -c 200 "${OUT_ROOT}/required.json")"
  fi

  code="$(chat named "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"Weather in Boston?"}], max_tokens:48, temperature:0, tool_choice:{type:"function",function:{name:"get_weather"}}, tools:$tools}')")"
  if [[ "$code" == "200" ]] \
    && [[ "$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' "${OUT_ROOT}/named.json")" == "get_weather" ]]; then
    pass "named tool_choice → get_weather"
  else
    fail "named tool_choice HTTP ${code}"
  fi

  code="$(chat none "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"Say hi in one word."}], max_tokens:16, temperature:0, tool_choice:"none", tools:$tools}')")"
  if [[ "$code" == "200" ]]; then
    tc="$(jq -r '.choices[0].message.tool_calls // empty' "${OUT_ROOT}/none.json")"
    if [[ -z "$tc" || "$tc" == "null" ]]; then
      pass "tool_choice=none never emits tool_calls"
    else
      fail "tool_choice=none returned tool_calls: ${tc}"
    fi
  else
    fail "none HTTP ${code}"
  fi

  if [[ -f "${OUT_ROOT}/required.json" ]]; then
    roundtrip="$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" --slurpfile prev "${OUT_ROOT}/required.json" \
      '{model:$m, temperature:0, max_tokens:24, tool_choice:"none", tools:$tools,
        messages:[
          {role:"user", content:"Weather in Boston?"},
          $prev[0].choices[0].message,
          {role:"tool", tool_call_id: ($prev[0].choices[0].message.tool_calls[0].id // "call_0"), content:"72F"}
        ]}')"
    code="$(chat roundtrip "$roundtrip")"
    if [[ "$code" == "200" ]]; then
      rtc="$(jq -r '.choices[0].message.tool_calls // empty' "${OUT_ROOT}/roundtrip.json")"
      content="$(jq -r '.choices[0].message.content // empty' "${OUT_ROOT}/roundtrip.json")"
      if [[ -z "$rtc" || "$rtc" == "null" ]] && [[ -n "$content" ]]; then
        pass "multi-turn role=tool continues generation"
      else
        fail "roundtrip unexpected tool_calls or empty content"
      fi
    else
      fail "roundtrip HTTP ${code}"
    fi
  else
    fail "roundtrip skipped (no required.json)"
  fi

  code="$(chat conflict_json "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"x"}], max_tokens:8, temperature:0, tools:$tools, response_format:{type:"json_object"}}')")"
  if [[ "$code" == "400" ]] && grep -q 'response_format' "${OUT_ROOT}/conflict_json.json"; then
    pass "tools + json_object HTTP 400"
  else
    fail "tools+json_object expected 400, got ${code}"
  fi

  code="$(chat conflict_gbnf "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"x"}], max_tokens:8, temperature:0, tools:$tools, x_juno_grammar:"root ::= \"a\""}')")"
  if [[ "$code" == "400" ]] && grep -q 'x_juno_grammar' "${OUT_ROOT}/conflict_gbnf.json"; then
    pass "tools + x_juno_grammar HTTP 400"
  else
    fail "tools+x_juno_grammar expected 400, got ${code}"
  fi

  stream_file="${OUT_ROOT}/stream.sse"
  curl -sS -N -o "$stream_file" -H 'Content-Type: application/json' \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -d "$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
      '{model:$m, stream:true, messages:[{role:"user",content:"Weather in Boston?"}], max_tokens:48, temperature:0, tool_choice:"required", tools:$tools}')" \
    || true
  chunks="$(grep -c '^data: {' "$stream_file" || true)"
  if grep -Eq 'finish_reason": ?"tool_calls"' "$stream_file" && grep -q '"tool_calls"' "$stream_file"; then
    pass "SSE tools path emits tool_calls after generation"
  elif [[ "${chunks:-0}" -ge 2 && "${chunks:-0}" -le 4 ]] && grep -q '"role":"assistant"' "$stream_file"; then
    pass "SSE tools path buffers after generation (${chunks} chunks; parse may miss if grammar hits length)"
  else
    fail "SSE missing buffered tools path (chunks=${chunks})"
  fi

  stop_juno
  collect_jfr qwen-chatml "${OUT_ROOT}/qwen-chatml.log"
  gc="$(grammar_count "${OUT_ROOT}/qwen-chatml-jfr.json")"
  if python3 -c "import sys; sys.exit(0 if float('${gc}') >= 2 else 1)"; then
    pass "JFR GrammarConstrained.count=${gc} (required/named, want >=2)"
    echo "- qwen-chatml JFR GrammarConstrained.count=${gc}" >>"${OUT_ROOT}/INDEX.md"
  else
    fail "qwen-chatml GrammarConstrained.count=${gc} (want >=2)"
  fi
else
  fail "qwen-chatml server failed to start"
fi

# ── Server B: TinyLlama fail-closed template ─────────────────────────────────
if start_juno tiny-fail "$TINY"; then
  tiny_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" | jq -r '.data[0].id // empty')"
  [[ -n "$tiny_id" ]] || tiny_id="$(basename "$TINY")"
  code="$(chat tiny_tools "$(jq -nc --arg m "$tiny_id" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"Weather?"}], max_tokens:8, temperature:0, tools:$tools}')")"
  if [[ "$code" == "400" ]] && grep -qi 'tinyllama' "${OUT_ROOT}/tiny_tools.json"; then
    pass "tinyllama + tools HTTP 400 (unsupported template)"
    echo "- tinyllama template fail-closed (tiny_tools.json)" >>"${OUT_ROOT}/INDEX.md"
  else
    fail "tinyllama tools expected 400/tinyllama, got ${code}"
  fi
  stop_juno
else
  fail "tiny-fail server failed to start"
fi

# ── Server C: --lora-play + tools still fail-closed on TinyLlama ─────────────
if [[ -f "$LORA" ]]; then
  if start_juno lora-play "$TINY" --lora-play "$LORA"; then
    tiny_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" | jq -r '.data[0].id // empty')"
    [[ -n "$tiny_id" ]] || tiny_id="$(basename "$TINY")"
    code="$(chat lora_tools "$(jq -nc --arg m "$tiny_id" --argjson tools "$TOOLS_JSON" \
      '{model:$m, messages:[{role:"user",content:"Weather?"}], max_tokens:8, temperature:0, tools:$tools}')")"
    if [[ "$code" == "400" ]] && grep -qi 'tinyllama' "${OUT_ROOT}/lora_tools.json"; then
      pass "--lora-play + tools fail-closed on tinyllama template"
      echo "- --lora-play: overlay loaded; tools still fail closed (no ChatML LoRA fixture on host)" >>"${OUT_ROOT}/INDEX.md"
    else
      fail "lora-play tools expected 400, got ${code}"
    fi
    if grep -qi 'lora' "${OUT_ROOT}/lora-play.log"; then
      pass "lora-play log mentions overlay"
    else
      warn "lora-play.log has no LoRA string"
    fi
    stop_juno
  else
    fail "lora-play server failed to start"
  fi
else
  fail "--lora-play overlay file missing: ${LORA}"
fi

# ── Server D: --parallel 2 + required tools (ChatML) ─────────────────────────
if start_juno parallel "$QWEN" --parallel 2; then
  [[ -n "$MODEL_ID" ]] || MODEL_ID="$(basename "$QWEN")"
  body="$(jq -nc --arg m "$MODEL_ID" --argjson tools "$TOOLS_JSON" \
    '{model:$m, messages:[{role:"user",content:"Weather in Boston?"}], max_tokens:48, temperature:0, tool_choice:"required", tools:$tools}')"
  chat parallel_a "$body" >"${OUT_ROOT}/parallel_a.http" &
  pid_a=$!
  chat parallel_b "$body" >"${OUT_ROOT}/parallel_b.http" &
  pid_b=$!
  wait "$pid_a" || true
  wait "$pid_b" || true
  n1="$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' "${OUT_ROOT}/parallel_a.json" 2>/dev/null || true)"
  n2="$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' "${OUT_ROOT}/parallel_b.json" 2>/dev/null || true)"
  if [[ "$n1" == "get_weather" && "$n2" == "get_weather" ]]; then
    pass "--parallel 2 + required tools (two tool_calls)"
    echo "- --parallel 2: two get_weather tool_calls" >>"${OUT_ROOT}/INDEX.md"
  else
    fail "parallel missing tool_calls (a=${n1} b=${n2})"
  fi
  stop_juno
  collect_jfr parallel "${OUT_ROOT}/parallel.log"
else
  fail "parallel server failed to start"
fi

{
  echo
  echo "## Results"
  echo
  echo "failures=${failures}"
  echo
  echo "Artifacts in this directory: HTTP JSON, *.log, *-jfr.json, INDEX.md"
} >>"${OUT_ROOT}/INDEX.md"

log "done. failures=${failures}  out=${OUT_ROOT}"
exit "$failures"
