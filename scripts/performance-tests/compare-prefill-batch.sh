#!/usr/bin/env bash
# compare-prefill-batch.sh — prefill microbatch bake-off (--prefill-batch 1 vs N)
#
# Launches juno local API with a long raw prompt, records JFR prefill metrics
# for each chunk size.
#
# Usage:
#   ./scripts/performance-tests/compare-prefill-batch.sh --cpu
#   ./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512
#   ./scripts/performance-tests/compare-prefill-batch.sh --gpu --prefill-values 1,32,128

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-prefill/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
N_PROMPT=256
MAX_TOKENS=8
PREFILL_VALUES="1,32"
API_PORT=18083
USE_GPU=0
GPU_ATTENTION=""
PUBLISH=1
JFR_DURATION="${JFR_DURATION:-10m}"

log()  { printf '[prefill-batch] %s\n' "$*"; }
warn() { printf '[prefill-batch] warn: %s\n' "$*" >&2; }
die()  { printf '[prefill-batch] error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs one blocking chat completion per --prefill-batch value with a long raw prompt.

Options:
  --model FILE            GGUF basename under models/ (default: ${MODEL})
  --n-prompt N            Raw prompt token target (default: ${N_PROMPT})
  --max-tokens N          Decode tokens (default: ${MAX_TOKENS})
  --prefill-values LIST   Comma-separated chunk sizes (default: ${PREFILL_VALUES})
  --api-port N            REST port (default: ${API_PORT})
  --cpu / --gpu           Backend (default: CPU)
  --gpu-attention on|off|auto  Juno --gpu-attention (GPU-resident attention kernel; default off)
  --out DIR               Output directory
  --no-publish            Skip docs/perf-compare copy
  -h, --help              This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --n-prompt) N_PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --prefill-values) PREFILL_VALUES="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --gpu-attention) GPU_ATTENTION="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --no-publish) PUBLISH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
require_cmd curl
require_cmd jq
require_cmd python3

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH}"

PROMPT_TEXT="$(python3 -c "print(' '.join(['x'] * ${N_PROMPT}))")"

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

wait_for_metrics() {
  local dest="$1" timeout="${2:-45}"
  local metrics_src="${ROOT}/target/metrics/metrics.json"
  local start now; start="$(date +%s)"
  while true; do
    if [[ -s "$metrics_src" ]]; then
      cp -a "$metrics_src" "$dest"
      return 0
    fi
    now="$(date +%s)"
    (( now - start >= timeout )) && return 1
    sleep 1
  done
}

run_prefill_value() {
  local prefill_batch="$1"
  local stem="prefill-${prefill_batch}"
  local logf="${OUT_ROOT}/${stem}.log"
  local out_json="${OUT_ROOT}/${stem}.json"
  local resp="${OUT_ROOT}/${stem}-response.json"
  local metrics="${OUT_ROOT}/${stem}-jfr.json"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} in use"
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
    --prefill-batch "$prefill_batch"
    --jfr "$JFR_DURATION"
  )
  [[ -n "$GPU_ATTENTION" ]] && java_args+=(--gpu-attention "$GPU_ATTENTION")

  log "start prefill-batch=${prefill_batch} n_prompt≈${N_PROMPT} ${backend_flag#--}"
  : >"$logf"
  (
    cd "$ROOT"
    exec "$java_bin" "${java_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  if ! wait_for_api "$API_PORT" 600 "$JUNO_PID"; then
    stop_juno
    die "API failed prefill-batch=${prefill_batch} — see ${logf}"
  fi

  local model_id
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  [[ -n "$model_id" ]] || model_id="$(basename "$MODEL" .gguf)"

  local start_ns end_ns wall_ms http_code rc=0
  start_ns="$(date +%s%N)"
  set +e
  http_code="$(curl -sS --max-time 7200 -o "$resp" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$model_id" --arg p "$PROMPT_TEXT" --argjson n "$MAX_TOKENS" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')")"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local prompt_tokens completion_tokens prefill_ms prefill_count pp_tps
  prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$resp" 2>/dev/null || echo 0)"
  completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$resp" 2>/dev/null || echo 0)"

  stop_juno
  sleep 2

  prefill_ms="null"
  prefill_count="null"
  pp_tps="null"
  attention_prefill_ms="null"
  attention_share_pct="null"
  if wait_for_metrics "$metrics" 45; then
    prefill_ms="$(jq -r '.models[0].metrics."juno.ForwardPass.prefill.total_ms" // null' "$metrics")"
    prefill_count="$(jq -r '.models[0].metrics."juno.ForwardPass.prefill.count" // null' "$metrics")"
    attention_prefill_ms="$(jq -r '.models[0].metrics."juno.Attention.prefill.total_ms" // null' "$metrics")"
    if [[ "$prefill_ms" != "null" && "$prompt_tokens" =~ ^[0-9]+$ && "$prompt_tokens" -gt 0 ]]; then
      pp_tps="$(awk -v pt="$prompt_tokens" -v ms="$prefill_ms" 'BEGIN { if (ms>0) printf "%.4f", pt/(ms/1000); }')"
    fi
    if [[ "$attention_prefill_ms" != "null" && "$prefill_ms" != "null" ]]; then
      attention_share_pct="$(awk -v a="$attention_prefill_ms" -v f="$prefill_ms" 'BEGIN { if (f>0) printf "%.1f", (a/f)*100; }')"
    fi
  else
    warn "metrics.json missing for prefill-batch=${prefill_batch}"
  fi

  jq -n \
    --arg tool "compare-prefill-batch" \
    --arg model "$(basename "$MODEL")" \
    --argjson prefill_batch "$prefill_batch" \
    --argjson n_prompt "$N_PROMPT" \
    --argjson prompt_tokens "${prompt_tokens:-0}" \
    --argjson completion_tokens "${completion_tokens:-0}" \
    --argjson wall_ms "$wall_ms" \
    --argjson http_code "${http_code:-0}" \
    --argjson rc "$rc" \
    --argjson prefill_total_ms "${prefill_ms:-null}" \
    --argjson prefill_count "${prefill_count:-null}" \
    --argjson prompt_eval_tps "${pp_tps:-null}" \
    --argjson use_gpu "$USE_GPU" \
    --arg gpu_attention "$GPU_ATTENTION" \
    --argjson attention_prefill_total_ms "${attention_prefill_ms:-null}" \
    --argjson attention_share_pct "${attention_share_pct:-null}" \
    '{
      tool: $tool,
      model: $model,
      prefill_batch: $prefill_batch,
      n_prompt_target: $n_prompt,
      prompt_tokens: $prompt_tokens,
      completion_tokens: $completion_tokens,
      wall_ms: $wall_ms,
      http_code: $http_code,
      exit_code: $rc,
      forward_pass_prefill_total_ms: $prefill_total_ms,
      forward_pass_prefill_count: $prefill_count,
      prompt_eval_tps: $prompt_eval_tps,
      use_gpu: $use_gpu,
      gpu_attention: $gpu_attention,
      attention_prefill_total_ms: $attention_prefill_total_ms,
      attention_share_pct: $attention_share_pct
    }' >"$out_json"

  log "prefill-batch=${prefill_batch}: pp_tps=${pp_tps:-?} prefill_ms=${prefill_ms:-?} count=${prefill_count:-?} wall_ms=${wall_ms} attention_share_pct=${attention_share_pct:-?}"
  [[ "$http_code" == "200" ]] || warn "http=${http_code} for prefill-batch=${prefill_batch}"
}

mkdir -p "$OUT_ROOT"
setup_cuda_env

IFS=',' read -ra PV <<< "$PREFILL_VALUES"
declare -a RESULT_JSONS=()
for pv in "${PV[@]}"; do
  run_prefill_value "$pv"
  RESULT_JSONS+=("${OUT_ROOT}/prefill-${pv}.json")
done

compare_json="${OUT_ROOT}/compare.json"
jq -s '{
  run_id: "'"${RUN_ID}"'",
  model: "'$(basename "$MODEL")'",
  n_prompt_target: '"${N_PROMPT}"',
  results: [.[] | {prefill_batch, prompt_eval_tps, forward_pass_prefill_total_ms, forward_pass_prefill_count, wall_ms}],
  speedup_32_over_1:
    (if ([.[] | select(.prefill_batch == 1) | .prompt_eval_tps][0] // 0) > 0
     then ([.[] | select(.prefill_batch == 32) | .prompt_eval_tps][0] // null) /
          ([.[] | select(.prefill_batch == 1) | .prompt_eval_tps][0])
     else null end)
}' "${RESULT_JSONS[@]}" >"$compare_json" 2>/dev/null || true

INDEX="${OUT_ROOT}/INDEX.md"
{
  echo "# Prefill microbatch — ${RUN_ID}"
  echo
  echo "Model: \`${MODEL}\` · raw prompt target: ${N_PROMPT} tokens · backend: $([ "$USE_GPU" -eq 1 ] && echo GPU || echo CPU) · gpu-attention: ${GPU_ATTENTION:-off (default)}"
  echo
  echo "| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |"
  echo "|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|"
  for f in "${RESULT_JSONS[@]}"; do
    [[ -f "$f" ]] || continue
    jq -r '"| \(.prefill_batch) | \(.prompt_eval_tps // "-") | \(.forward_pass_prefill_total_ms // "-") | \(.forward_pass_prefill_count // "-") | \(.wall_ms) | \(.attention_share_pct // "-")% |"' "$f"
  done
  if [[ -f "$compare_json" ]]; then
    echo
    echo "Speedup prefill-batch=32 over 1: $(jq -r '.speedup_32_over_1 // "-"' "$compare_json")"
  fi
} >"$INDEX"

if [[ "$PUBLISH" -eq 1 ]]; then
  dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-prefill"
  mkdir -p "$dest"
  cp -a "${OUT_ROOT}/." "$dest/"
  log "published → docs/perf-compare/${RUN_ID}-prefill/"
fi

log "done → ${OUT_ROOT}"
