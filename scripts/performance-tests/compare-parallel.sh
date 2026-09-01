#!/usr/bin/env bash
# compare-parallel.sh — multi-session static batch bake-off (--parallel 1 vs N)
#
# Launches juno local API, fires SESSIONS concurrent blocking chat completions,
# records aggregate throughput (sum completion_tokens / wall seconds).
#
# Usage:
#   ./scripts/performance-tests/compare-parallel.sh --cpu
#   ./scripts/performance-tests/compare-parallel.sh --gpu --sessions 8 --max-tokens 64
#   ./scripts/performance-tests/compare-parallel.sh --gpu --parallel-values 1,8

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-parallel/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
SESSIONS=8
MAX_TOKENS=64
BATCH_WINDOW_MS=50
PARALLEL_VALUES="1,8"
API_PORT=18082
USE_GPU=0
PROMPT_TEXT="could you please write me a short poem about love and war"
PUBLISH=1

log()  { printf '[parallel] %s\n' "$*"; }
warn() { printf '[parallel] warn: %s\n' "$*" >&2; }
die()  { printf '[parallel] error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs aggregate multi-session throughput for each --parallel value (default: 1,8).

Options:
  --model FILE           GGUF basename under models/ (default: ${MODEL})
  --sessions N           Concurrent clients (default: ${SESSIONS})
  --max-tokens N         Tokens per session (default: ${MAX_TOKENS})
  --parallel-values LIST Comma-separated parallel sizes (default: ${PARALLEL_VALUES})
  --batch-window-ms M    Batch window when parallel>1 (default: ${BATCH_WINDOW_MS})
  --api-port N           REST port (default: ${API_PORT})
  --cpu / --gpu          Backend (default: CPU)
  --prompt TEXT          User prompt
  --out DIR              Output directory
  --no-publish           Skip docs/perf-compare copy
  -h, --help             This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --parallel-values) PARALLEL_VALUES="$2"; shift 2 ;;
    --batch-window-ms) BATCH_WINDOW_MS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --prompt) PROMPT_TEXT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --no-publish) PUBLISH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
require_cmd curl
require_cmd jq
require_cmd awk

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH}"

find_juno_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/juno-player/target/"juno-player-*-shaded.jar )
  shopt -u nullglob
  [[ ${#jars[@]} -gt 0 ]] && { printf '%s' "${jars[0]}"; return; }
  [[ -f "$ROOT/juno-player/target/juno-player.jar" ]] || die "build jar: mvn package -DskipTests -pl juno-player -am"
  printf '%s' "$ROOT/juno-player/target/juno-player.jar"
}

find_java() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    printf '%s' "${JAVA_HOME}/bin/java"; return
  fi
  command -v java || die "java not found"
}

setup_cuda_env() {
  [[ "$USE_GPU" -eq 1 ]] || return 0
  if [[ -e /usr/lib/x86_64-linux-gnu/libcudart.so.12 ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export JUNO_USE_GPU=true
}

wait_for_api() {
  local port="$1" timeout="${2:-300}" pid="${3:-}"
  local start now; start="$(date +%s)"
  while true; do
    curl -sf "http://127.0.0.1:${port}/v1/cluster/health" >/dev/null 2>&1 && return 0
    [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && return 1
    now="$(date +%s)"
    (( now - start >= timeout )) && return 1
    sleep 2
  done
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  JUNO_PID=""
}

run_parallel_value() {
  local parallel="$1"
  local stem="parallel-${parallel}"
  local logf="${OUT_ROOT}/${stem}.log"
  local out_json="${OUT_ROOT}/${stem}.json"
  local resp_dir="${OUT_ROOT}/${stem}-responses"
  mkdir -p "$resp_dir"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} in use — stop other juno or pass --api-port"
  fi

  local jar java_bin heap backend_flag
  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  heap="${COMPARE_HEAP:-4g}"
  backend_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && backend_flag="--gpu"

  local -a java_args=(
    --enable-preview --enable-native-access=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED
    --add-opens java.base/java.nio=ALL-UNNAMED
    -XX:+UseG1GC -XX:+AlwaysPreTouch -Xms512m -Xmx"${heap}"
    -Djuno.byteOrder=BE
    -jar "$jar"
    --model-path "$MODEL_PATH"
    --dtype FLOAT16 --byteOrder BE
    --max-tokens "$MAX_TOKENS"
    --temperature 0 --top-k 0 --top-p 0
    --nodes 1 --local "$backend_flag"
    --api-port "$API_PORT"
    --parallel "$parallel"
  )
  if (( parallel > 1 )); then
    java_args+=(--batch-window-ms "$BATCH_WINDOW_MS")
  fi

  log "start juno parallel=${parallel} sessions=${SESSIONS} max_tokens=${MAX_TOKENS} ${backend_flag#--}"
  : >"$logf"
  (
    cd "$ROOT"
    exec "$java_bin" "${java_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  if ! wait_for_api "$API_PORT" 600 "$JUNO_PID"; then
    stop_juno
    die "API failed for parallel=${parallel} — see ${logf}"
  fi

  local model_id
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  [[ -n "$model_id" ]] || model_id="${MODEL%.gguf}"

  local start_ns end_ns wall_ms
  start_ns="$(date +%s%N)"
  local i pids=() rc=0
  for i in $(seq 1 "$SESSIONS"); do
    (
      set +e
      http="$(curl -sS --max-time 7200 -o "${resp_dir}/resp-${i}.json" -w '%{http_code}' \
        -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$model_id" --arg p "$PROMPT_TEXT" --argjson n "$MAX_TOKENS" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')")"
      echo "$http" >"${resp_dir}/resp-${i}.http"
      exit 0
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local total_completion=0 total_prompt=0 total_latency=0 ok=0 fail=0
  for i in $(seq 1 "$SESSIONS"); do
    local resp="${resp_dir}/resp-${i}.json"
    local http; http="$(cat "${resp_dir}/resp-${i}.http" 2>/dev/null || echo 000)"
    if [[ "$http" == "200" && -s "$resp" ]]; then
      ok=$((ok + 1))
      total_completion=$((total_completion + $(jq -r '.usage.completion_tokens // 0' "$resp")))
      total_prompt=$((total_prompt + $(jq -r '.usage.prompt_tokens // 0' "$resp")))
      total_latency=$((total_latency + $(jq -r '.x_juno_latency_ms // 0' "$resp")))
    else
      fail=$((fail + 1))
    fi
  done

  stop_juno
  sleep 2

  local agg_tps="null" mean_latency="null"
  if (( wall_ms > 0 && total_completion > 0 )); then
    agg_tps="$(awk -v t="$total_completion" -v ms="$wall_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
  fi
  if (( ok > 0 )); then
    mean_latency="$(awk -v l="$total_latency" -v n="$ok" 'BEGIN { printf "%.2f", l / n }')"
  fi

  jq -n \
    --arg run_id "$RUN_ID" \
    --arg model "$MODEL" \
    --arg model_path "$MODEL_PATH" \
    --arg backend "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" \
    --argjson parallel "$parallel" \
    --argjson batch_window_ms "$BATCH_WINDOW_MS" \
    --argjson sessions "$SESSIONS" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson wall_ms "$wall_ms" \
    --argjson ok "$ok" \
    --argjson fail "$fail" \
    --argjson total_completion_tokens "$total_completion" \
    --argjson total_prompt_tokens "$total_prompt" \
    --argjson aggregate_tps "${agg_tps:-null}" \
    --argjson mean_latency_ms "${mean_latency:-null}" \
    --arg prompt "$PROMPT_TEXT" \
    --arg log "$logf" \
  '{
    run_id: $run_id,
    tool: "compare-parallel",
    status: (if $fail == 0 then "success" else "partial_failure" end),
    model: $model,
    model_path: $model_path,
    backend: $backend,
    parallel: $parallel,
    batch_window_ms: $batch_window_ms,
    sessions: $sessions,
    max_tokens: $max_tokens,
    wall_ms: $wall_ms,
    sessions_ok: $ok,
    sessions_failed: $fail,
    total_completion_tokens: $total_completion_tokens,
    total_prompt_tokens: $total_prompt_tokens,
    aggregate_token_gen_tps: $aggregate_tps,
    mean_per_session_latency_ms: $mean_latency_ms,
    prompt: $prompt,
    log: $log
  }' >"$out_json"

  log "parallel=${parallel}: agg_tps=${agg_tps:-?} ok=${ok}/${SESSIONS} wall_ms=${wall_ms}"
  [[ "$fail" -eq 0 ]] || warn "parallel=${parallel}: ${fail} session(s) failed"
}

mkdir -p "$OUT_ROOT"

setup_cuda_env

IFS=',' read -r -a PAR_LIST <<<"$PARALLEL_VALUES"
for pv in "${PAR_LIST[@]}"; do
  pv="$(echo "$pv" | tr -d ' ')"
  [[ -n "$pv" ]] || continue
  run_parallel_value "$pv"
done

# Compare summary
compare_json="${OUT_ROOT}/compare.json"
jq -s '
  {
    run_id: (.[0].run_id // ""),
    model: (.[0].model // ""),
    backend: (.[0].backend // ""),
    sessions: (.[0].sessions // 0),
    max_tokens: (.[0].max_tokens // 0),
    results: [.[] | {parallel, aggregate_token_gen_tps, wall_ms, sessions_ok, sessions_failed}],
    speedup_parallel_over_1:
      (if ([.[] | select(.parallel == 1) | .aggregate_token_gen_tps][0] // 0) > 0
       then ([.[] | select(.parallel == 8) | .aggregate_token_gen_tps][0] // null) /
            ([.[] | select(.parallel == 1) | .aggregate_token_gen_tps][0])
       else null end)
  }
' "${OUT_ROOT}"/parallel-*.json >"$compare_json" 2>/dev/null || true

index_md="${OUT_ROOT}/INDEX.md"
{
  echo "# Multi-session static batch — ${RUN_ID}"
  echo
  echo "Model: \`${MODEL}\` · sessions=${SESSIONS} · max_tokens=${MAX_TOKENS} · backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)"
  echo
  echo "| parallel | aggregate tg t/s | wall ms | ok/fail |"
  echo "|---------:|-----------------:|--------:|--------:|"
  for f in "${OUT_ROOT}"/parallel-*.json; do
    [[ -f "$f" ]] || continue
    jq -r '"| \(.parallel) | \(.aggregate_token_gen_tps // "-") | \(.wall_ms) | \(.sessions_ok)/\(.sessions) |"' "$f"
  done
  echo
  if [[ -f "$compare_json" ]]; then
    echo "Speedup parallel=8 over parallel=1: $(jq -r '.speedup_parallel_over_1 // "-"' "$compare_json")"
  fi
} >"$index_md"

if [[ "$PUBLISH" -eq 1 ]]; then
  dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-parallel"
  mkdir -p "$dest"
  cp -a "$OUT_ROOT"/* "$dest/"
  log "published → docs/perf-compare/${RUN_ID}-parallel/"
fi

log "done: ${OUT_ROOT}"
cat "$index_md"
