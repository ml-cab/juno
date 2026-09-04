#!/usr/bin/env bash
# compare-vision.sh — local vision inference benchmark (moondream llamafile)
#
# Launches ./juno local --jfr, drives POST /v1/vision/chat (multipart image+JSON),
# writes JSON under target/perf-compare-vision/, optionally compares baseline git
# ref vs HEAD, and publishes to docs/perf-compare/<run-id>-vision/.
#
# Usage:
#   ./scripts/performance-tests/compare-vision.sh --gpu
#   ./scripts/performance-tests/compare-vision.sh --gpu --baseline release-0.1.2
#   ./scripts/performance-tests/compare-vision.sh --cpu --no-publish
#
# Regression gate (default): latency_ms ratio > 1.25 or decode tps ratio < 0.80 → exit 1.
#
# Test image: scripts/performance-tests/fixtures/vision-bench.jpg
# Override: VISION_TEST_IMAGE or --image PATH
#
# Prefill: defaults to --prefill single. Batched Phi2/Q5_K prefill on current
# inference branches completes after the hang fix but yields wrong captions vs
# 47-vision; single-token prefill matches the known-good sequential path.
# Pass --prefill batched only when measuring / validating that path.

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-vision/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="moondream2-q5_k.llamafile"
MMPROJ_PATH=""
IMAGE_PATH="${VISION_TEST_IMAGE:-${PERF_SCRIPTS}/fixtures/vision-bench.jpg}"
PROMPT="${VISION_PERF_PROMPT:-What is in this image?}"
MAX_TOKENS="${VISION_PERF_MAX_TOKENS:-32}"
USE_GPU=1
HEAP="${COMPARE_HEAP:-}"
API_PORT="${VISION_PERF_API_PORT:-18081}"
JFR_DURATION="${JFR_DURATION:-30m}"
PREFILL_MODE="${VISION_PERF_PREFILL:-single}"
PUBLISH=1
BASELINE_REF=""
CURRENT_REF=""
REGRESSION_RATIO="${VISION_PERF_REGRESSION_RATIO:-1.25}"
DRY_RUN=0
SKIP_BUILD=0
EXPECTED_PROMPT_TOKENS_MIN=700
EXPECTED_PROMPT_TOKENS_MAX=800

JUNO_PID=""

log()  { printf '[vision-perf] %s\n' "$*"; }
warn() { printf '[vision-perf] warn: %s\n' "$*" >&2; }
die()  { printf '[vision-perf] error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \?//'
  cat <<EOF

Options:
  --model FILE          Llamafile/GGUF basename under models/ (default: ${MODEL})
  --mmproj-path PATH    Separate mmproj GGUF (two-file LLaVA/Qwen-VL; v2)
  --image PATH          Test image (default: fixtures/vision-bench.jpg)
  --prompt TEXT         User prompt (default: ${PROMPT})
  --max-tokens N        Decode cap (default: ${MAX_TOKENS})
  --gpu / --cpu         Inference backend (default: gpu)
  --heap SIZE           JVM -Xmx (default: auto from model size)
  --api-port N          REST port (default: ${API_PORT})
  --jfr DURATION        JFR recording duration (default: ${JFR_DURATION})
  --prefill MODE        Prefill strategy: single (default, quality) or batched
  --out DIR             Output directory (default: target/perf-compare-vision/<timestamp>)
  --baseline REF        Git ref to compare against (e.g. release-0.1.2)
  --current REF         Git ref for current run (default: HEAD)
  --regression-ratio F  Fail when latency ratio > F or tps ratio < 1/F (default: ${REGRESSION_RATIO})
  --skip-build          Skip mvn package (use existing jar)
  --no-publish          Skip docs/perf-compare copy
  -n, --dry-run         Print planned steps only
  -h, --help            This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --mmproj-path) MMPROJ_PATH="$2"; shift 2 ;;
    --image) IMAGE_PATH="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --gpu) USE_GPU=1; shift ;;
    --cpu) USE_GPU=0; shift ;;
    --heap) HEAP="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --jfr) JFR_DURATION="$2"; shift 2 ;;
    --prefill)
      PREFILL_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
      case "$PREFILL_MODE" in
        single|batched) ;;
        *) die "--prefill must be single or batched (got: $2)" ;;
      esac
      shift 2
      ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --baseline) BASELINE_REF="$2"; shift 2 ;;
    --current) CURRENT_REF="$2"; shift 2 ;;
    --regression-ratio) REGRESSION_RATIO="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }
require_cmd git
require_cmd jq
require_cmd curl
require_cmd awk

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH}"
[[ -f "$IMAGE_PATH" ]] || die "test image not found: ${IMAGE_PATH} (set VISION_TEST_IMAGE or --image)"
if [[ -n "$MMPROJ_PATH" && ! -f "$MMPROJ_PATH" ]]; then
  die "mmproj not found: ${MMPROJ_PATH}"
fi

setup_cuda_env() {
  [[ "$USE_GPU" -eq 1 ]] || return 0
  if [[ -f "${ROOT}/scripts/set_cuda_env.sh" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${ROOT}/scripts/set_cuda_env.sh" || true
    set -u
  fi
  local cuda_root=""
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
    cuda_root="${CUDA_HOME}"
  elif [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]]; then
    cuda_root="${CUDA_PATH}"
  elif [[ -d /usr/local/cuda/lib64 ]]; then
    cuda_root="/usr/local/cuda"
  fi
  if [[ -n "$cuda_root" ]]; then
    export PATH="${cuda_root}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  # Distro CUDA packages often land in /usr/lib/x86_64-linux-gnu.
  if [[ -e /usr/lib/x86_64-linux-gnu/libcudart.so.12 || -e /usr/lib/x86_64-linux-gnu/libcudart.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export JUNO_USE_GPU=true
}

strip_ansi() {
  sed 's/\x1b\[[0-9;]*m//g'
}

heap_for_model() {
  local path="$1" bytes heap_g
  if [[ -n "$HEAP" ]]; then
    printf '%s' "$HEAP"
    return
  fi
  bytes="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")"
  heap_g=$(( (bytes * 3 / 2 + 2 * 1024 * 1024 * 1024 + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024) ))
  (( heap_g < 4 )) && heap_g=4
  (( heap_g > 48 )) && heap_g=48
  printf '%sg' "$heap_g"
}

build_juno() {
  local dir="$1"
  [[ "$SKIP_BUILD" -eq 1 || "$DRY_RUN" -eq 1 ]] && return 0
  log "building juno-player in ${dir}…"
  (cd "$dir" && mvn -q package -DskipTests -pl juno-player -am)
}

wait_for_juno_api() {
  local port="$1" timeout="${2:-600}" pid="${3:-}"
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

wait_for_vision_routes() {
  local logf="$1" timeout="${2:-120}"
  local start now
  start="$(date +%s)"
  while true; do
    if grep -q 'Vision routes registered' < <(strip_ansi <"$logf" 2>/dev/null || true); then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 1
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
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
}

wait_for_jfr_metrics() {
  local src="$1" dest="$2" timeout="${3:-60}"
  local start now
  start="$(date +%s)"
  while true; do
    if [[ -s "$src" ]]; then
      cp -a "$src" "$dest"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 1
  done
}

# Derive pp/tg from JFR metrics + API token counts; writes compact JSON object to stdout.
jfr_summary_json() {
  local jfr_file="$1" prompt_tokens="$2" completion_tokens="$3" latency_ms="${4:-0}"
  [[ -f "$jfr_file" ]] || { echo null; return 0; }
  jq -nc --arg f "$jfr_file" \
    --argjson pt "${prompt_tokens:-0}" --argjson ct "${completion_tokens:-0}" \
    --argjson latency_ms "${latency_ms:-0}" \
    --slurpfile raw "$jfr_file" '
    ($raw[0].models[0].metrics // {}) as $m |
    ($m."juno.ForwardPass.prefill.total_ms" // 0) as $prefill_ms |
    ($m."juno.ForwardPass.decode.total_ms" // 0) as $decode_ms |
    ($m."juno.TokenProduced.tps" // null) as $token_tps |
    ($m."juno.MatVec.duration.total_ms" // null) as $matvec_ms |
    (if $decode_ms > 0 and $ct > 0 then ($ct / ($decode_ms / 1000.0)) else null end) as $decode_derived_tps |
    (if $prefill_ms > 0 and $pt > 0 then ($pt / ($prefill_ms / 1000.0))
     elif ($latency_ms > $decode_ms and $pt > 0)
     then ($pt / (($latency_ms - $decode_ms) / 1000.0))
     else null end) as $pp_tps |
    (if $prefill_ms > 0 then "jfr_prefill_total_ms"
     elif ($latency_ms > $decode_ms and $pt > 0) then "wall_minus_decode"
     else null end) as $pp_source |
    {
      metrics_file: $f,
      jfr_file: ($raw[0].models[0].jfrFile // null),
      prompt_eval_tps: $pp_tps,
      prompt_eval_tps_source: $pp_source,
      token_gen_tps: (if $token_tps != null and $token_tps > 0 then $token_tps else $decode_derived_tps end),
      token_gen_tps_source: (if $token_tps != null and $token_tps > 0 then "TokenProduced.tps" else "ForwardPass.decode.total_ms" end),
      token_gen_tps_decode_derived: $decode_derived_tps,
      forward_pass_prefill_total_ms: $prefill_ms,
      forward_pass_decode_total_ms: $decode_ms,
      matvec_duration_total_ms: $matvec_ms,
      forward_pass_prefill_p95_ms: ($m."juno.ForwardPass.prefill.p95_ms" // null),
      forward_pass_decode_p95_ms: ($m."juno.ForwardPass.decode.p95_ms" // null),
      token_produced_tps: $token_tps
    }
  '
}

write_bench_json() {
  local out_json="$1"
  local ref="$2"
  local commit="$3"
  local host="$4"
  local resp="$5"
  local http_code="$6"
  local jfr_metrics="$7"
  local logf="$8"
  local load_ms="$9"
  local wall_ms="${10}"

  local latency_ms=0 prompt_tokens=0 completion_tokens=0 finish_reason="" reply=""
  local api_tps="null" compare_tps="null" jfr_block="null" jfr_model="null"
  local status="failure"

  if [[ "$http_code" == "200" && -s "$resp" ]]; then
    latency_ms="$(jq -r '.x_juno_latency_ms // 0' "$resp")"
    prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$resp")"
    completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$resp")"
    finish_reason="$(jq -r '.choices[0].finish_reason // empty' "$resp")"
    reply="$(jq -r '.choices[0].message.content // empty' "$resp")"
    if [[ "$latency_ms" =~ ^[0-9]+$ ]] && (( latency_ms > 0 && completion_tokens > 0 )); then
      api_tps="$(awk -v t="$completion_tokens" -v ms="$latency_ms" 'BEGIN { printf "%.6f", t / (ms/1000.0) }')"
    fi
    if [[ -n "$reply" && "$finish_reason" != "error" ]]; then
      status="success"
    elif [[ -n "$reply" ]]; then
      status="empty_or_error_finish"
    else
      status="empty_reply"
    fi
  fi

  compare_tps="$api_tps"
  if [[ -f "$jfr_metrics" ]]; then
    jfr_model="$(jq -c '.models[0] // null' "$jfr_metrics" 2>/dev/null || echo null)"
    jfr_block="$(jfr_summary_json "$jfr_metrics" "$prompt_tokens" "$completion_tokens" "$latency_ms")"
    local jfr_tps
    jfr_tps="$(jq -r '.token_gen_tps // empty' <<<"$jfr_block" 2>/dev/null || true)"
    if [[ -n "$jfr_tps" && "$jfr_tps" != "null" ]]; then
      compare_tps="$jfr_tps"
    fi
  fi

  if (( prompt_tokens > 0 )) && \
     { (( prompt_tokens < EXPECTED_PROMPT_TOKENS_MIN )) || (( prompt_tokens > EXPECTED_PROMPT_TOKENS_MAX )); }; then
    warn "prompt_tokens=${prompt_tokens} outside expected band ${EXPECTED_PROMPT_TOKENS_MIN}–${EXPECTED_PROMPT_TOKENS_MAX} for moondream"
  fi

  local resolved_heap
  resolved_heap="$(heap_for_model "$MODEL_PATH")"

  jq -n \
    --arg tool "compare-vision+jfr" \
    --arg run_id "$RUN_ID" \
    --arg git_ref "$ref" \
    --arg git_commit "$commit" \
    --arg host "$host" \
    --arg model "$MODEL" \
    --arg model_path "$MODEL_PATH" \
    --arg mmproj_path "${MMPROJ_PATH:-}" \
    --arg backend "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" \
    --arg heap "$resolved_heap" \
    --arg jfr_duration "$JFR_DURATION" \
    --arg prefill_mode "$PREFILL_MODE" \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    --arg image_path "$IMAGE_PATH" \
    --argjson api_port "$API_PORT" \
    --arg http_code "$http_code" \
    --argjson load_ms "$load_ms" \
    --argjson wall_ms "$wall_ms" \
    --argjson latency_ms "$latency_ms" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson completion_tokens "$completion_tokens" \
    --arg finish_reason "$finish_reason" \
    --arg reply "$reply" \
    --arg api_token_gen_tps "$api_tps" \
    --arg token_gen_tps "$compare_tps" \
    --arg response_json "$resp" \
    --arg log "$logf" \
    --argjson jfr_summary "$jfr_block" \
    --argjson jfr_model "$jfr_model" \
    --arg status "$status" \
    '{
      tool: $tool,
      run_id: $run_id,
      git_ref: $git_ref,
      git_commit: $git_commit,
      host: $host,
      model: $model,
      model_path: $model_path,
      mmproj_path: (if $mmproj_path == "" then null else $mmproj_path end),
      backend: $backend,
      heap: $heap,
      use_jfr: true,
      jfr_duration: $jfr_duration,
      prefill_mode: $prefill_mode,
      scenario: {
        prompt: $prompt,
        max_tokens: $max_tokens,
        temperature: 0,
        prefill: $prefill_mode,
        image_path: $image_path,
        expected_prompt_tokens: [700, 800]
      },
      api_port: $api_port,
      http_code: $http_code,
      load_ms: $load_ms,
      wall_ms: $wall_ms,
      latency_ms: $latency_ms,
      usage: {
        prompt_tokens: $prompt_tokens,
        completion_tokens: $completion_tokens
      },
      finish_reason: $finish_reason,
      reply: $reply,
      api_token_gen_tps: (if $api_token_gen_tps == "null" then null else ($api_token_gen_tps | tonumber) end),
      tps: (if $token_gen_tps == "null" then null else ($token_gen_tps | tonumber) end),
      tps_source: (if ($jfr_summary.token_gen_tps_source // null) != null then $jfr_summary.token_gen_tps_source else "api_latency" end),
      jfr: $jfr_summary,
      jfr_model: $jfr_model,
      response_json: $response_json,
      log: $log,
      status: $status
    }' >"$out_json"
}

run_bench_at() {
  local workdir="$1"
  local ref="$2"
  local out_json="$3"
  local logf="${out_json%.json}.log"
  local resp="${out_json%.json}-response.json"
  local jfr_metrics="${out_json%.json}-jfr.json"
  local juno="${workdir}/juno"
  local metrics_src="${workdir}/target/metrics/metrics.json"
  local backend_flag="--gpu"
  [[ "$USE_GPU" -eq 0 ]] && backend_flag="--cpu"
  local resolved_heap
  resolved_heap="$(heap_for_model "$MODEL_PATH")"

  local commit host
  commit="$(git -C "$workdir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  host="$(hostname 2>/dev/null || echo unknown)"

  log "benchmark ref=${ref} commit=${commit} backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu) prefill=${PREFILL_MODE} jfr=${JFR_DURATION} port=${API_PORT}"

  local -a juno_args=(
    local
    --model-path "$MODEL_PATH"
    --api-port "$API_PORT"
    --jfr "$JFR_DURATION"
    --max-tokens "$MAX_TOKENS"
    --temperature 0
    --heap "$resolved_heap"
    --prefill "$PREFILL_MODE"
    "$backend_flag"
  )
  if [[ -n "$MMPROJ_PATH" ]]; then
    juno_args+=(--mmproj-path "$MMPROJ_PATH")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: (cd ${workdir} && ${juno} ${juno_args[*]}) + POST /v1/vision/chat"
    return 0
  fi

  [[ -x "$juno" ]] || die "juno launcher missing in ${workdir} — build failed?"
  rm -f "$metrics_src"
  mkdir -p "${workdir}/target/metrics"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} already has a healthy Juno API — stop it or pass --api-port"
  fi

  setup_cuda_env
  : >"$logf"

  (
    cd "$workdir"
    # shellcheck disable=SC2094
    exec "$juno" "${juno_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  local load_start load_end load_ms wall_ms http_code rc=0
  load_start="$(date +%s%N)"

  if ! wait_for_juno_api "$API_PORT" 600 "$JUNO_PID"; then
    stop_juno
    die "API did not become healthy — see ${logf}"
  fi

  if ! wait_for_vision_routes "$logf" 120; then
    stop_juno
    die "Vision routes not registered — see ${logf} (cluster mode lacks vision routes; use ./juno local)"
  fi

  load_end="$(date +%s%N)"
  load_ms=$(( (load_end - load_start) / 1000000 ))

  local model_id
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  [[ -n "$model_id" ]] || model_id="${MODEL%.llamafile}"

  local start_ns end_ns
  start_ns="$(date +%s%N)"
  set +e
  http_code="$(curl -sS --max-time 7200 -o "$resp" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${API_PORT}/v1/vision/chat" \
    -F "image=@${IMAGE_PATH}" \
    -F "request=$(jq -nc --arg m "$model_id" --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}')" \
    2>>"$logf")"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  stop_juno
  sleep 2

  if ! wait_for_jfr_metrics "$metrics_src" "$jfr_metrics" 60; then
    warn "JFR metrics.json not found — continuing with API metrics only"
    echo '{"models":[]}' >"$jfr_metrics"
  fi

  write_bench_json "$out_json" "$ref" "$commit" "$host" "$resp" "$http_code" "$jfr_metrics" "$logf" "$load_ms" "$wall_ms"

  if [[ "$rc" -ne 0 || "$http_code" != "200" ]]; then
    die "vision request failed http=${http_code} rc=${rc} — see ${logf} ${resp}"
  fi

  jq -e '.status == "success"' "$out_json" >/dev/null \
    || die "quality gate failed for ref=${ref} — see ${out_json}"

  local pt lat tps
  pt="$(jq -r '.usage.prompt_tokens' "$out_json")"
  lat="$(jq -r '.latency_ms' "$out_json")"
  tps="$(jq -r '.tps // "?"' "$out_json")"
  log "ref=${ref} prompt_tokens=${pt} latency_ms=${lat} tps=${tps}"
}

stage_worktree() {
  local ref="$1"
  local dest="$2"
  git -C "$ROOT" worktree add --detach "$dest" "$ref" >/dev/null
  mkdir -p "$dest/scripts/performance-tests/fixtures"
  cp "${PERF_SCRIPTS}/compare-vision.sh" "$dest/scripts/performance-tests/"
  cp "$IMAGE_PATH" "$dest/scripts/performance-tests/fixtures/vision-bench.jpg"
}

cleanup_worktree() {
  local dest="$1"
  git -C "$ROOT" worktree remove --force "$dest" 2>/dev/null || rm -rf "$dest"
}

run_at_ref() {
  local ref="$1"
  local label="$2"
  local out_json="${OUT_ROOT}/${label}.json"

  if [[ "$ref" == "HEAD" || "$ref" == "$(git -C "$ROOT" rev-parse HEAD)" ]]; then
    build_juno "$ROOT"
    run_bench_at "$ROOT" "$ref" "$out_json"
    return 0
  fi

  local wt
  wt="$(mktemp -d "${ROOT}/target/vision-perf-wt-XXXXXX")"
  trap 'cleanup_worktree "$wt"; stop_juno' RETURN
  stage_worktree "$ref" "$wt"
  build_juno "$wt"
  run_bench_at "$wt" "$ref" "$out_json"
  cleanup_worktree "$wt"
  trap - RETURN
}

write_compare_json() {
  local baseline_json="$1"
  local current_json="$2"
  local compare_json="$3"

  jq -s --argjson ratio "$REGRESSION_RATIO" '
    .[0] as $base | .[1] as $cur |
    {
      tool: "compare-vision",
      run_id: ($cur.run_id // $base.run_id),
      model: ($cur.model // $base.model),
      backend: ($cur.backend // $base.backend),
      use_jfr: true,
      baseline: {
        git_ref: $base.git_ref,
        git_commit: $base.git_commit,
        latency_ms: $base.latency_ms,
        prefill_ms: ($base.jfr.forward_pass_prefill_total_ms // null),
        tps: $base.tps,
        prompt_tokens: $base.usage.prompt_tokens,
        status: $base.status
      },
      current: {
        git_ref: $cur.git_ref,
        git_commit: $cur.git_commit,
        latency_ms: $cur.latency_ms,
        prefill_ms: ($cur.jfr.forward_pass_prefill_total_ms // null),
        tps: $cur.tps,
        prompt_tokens: $cur.usage.prompt_tokens,
        status: $cur.status
      },
      ratios: {
        latency_ms: (if ($base.latency_ms // 0) > 0 then ($cur.latency_ms / $base.latency_ms) else null end),
        prefill_ms: (if ($base.jfr.forward_pass_prefill_total_ms // 0) > 0
                     then ($cur.jfr.forward_pass_prefill_total_ms / $base.jfr.forward_pass_prefill_total_ms)
                     else null end),
        tps: (if ($base.tps // 0) > 0 then ($cur.tps / $base.tps) else null end)
      },
      regression_ratio_limit: $ratio,
      regressions: [
        (if $base.status != "success" or $cur.status != "success" then "quality" else empty end),
        (if ($base.latency_ms // 0) > 0 and ($cur.latency_ms / $base.latency_ms) > $ratio
         then "latency_ms" else empty end),
        (if ($base.tps // 0) > 0 and ($cur.tps / $base.tps) < (1 / $ratio)
         then "tps" else empty end)
      ],
      status: (
        if ($base.status != "success" or $cur.status != "success") then "quality_failed"
        elif (
          (($base.latency_ms // 0) > 0 and ($cur.latency_ms / $base.latency_ms) > $ratio) or
          (($base.tps // 0) > 0 and ($cur.tps / $base.tps) < (1 / $ratio))
        ) then "regression"
        else "ok"
        end
      )
    }
  ' "$baseline_json" "$current_json" >"$compare_json"
}

write_index_md() {
  local index_md="$1"
  local baseline_json="$2"
  local current_json="$3"
  local compare_json="$4"

  {
    echo "# Vision perf compare — ${RUN_ID}"
    echo
    echo "Model: \`${MODEL}\` · backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu) · prefill=\`${PREFILL_MODE}\` · prompt: *${PROMPT}* · max_tokens=${MAX_TOKENS} · JFR \`${JFR_DURATION}\`"
    echo
    echo "| ref | commit | prompt tok | latency ms | prefill ms | decode tps | status |"
    echo "|-----|--------|----------:|-----------:|-----------:|-----------:|:------:|"
    if [[ -n "$compare_json" && -f "$compare_json" ]]; then
      for f in "$baseline_json" "$current_json"; do
        [[ -f "$f" ]] || continue
        jq -r '"| \(.git_ref // "?") | \(.git_commit // "?") | \(.usage.prompt_tokens) | \(.latency_ms) | \(.jfr.forward_pass_prefill_total_ms // "-") | \(.tps // "-") | \(.status) |"' "$f"
      done
    elif [[ -f "$current_json" ]]; then
      jq -r '"| \(.git_ref // "?") | \(.git_commit // "?") | \(.usage.prompt_tokens) | \(.latency_ms) | \(.jfr.forward_pass_prefill_total_ms // "-") | \(.tps // "-") | \(.status) |"' "$current_json"
    fi
    echo
    if [[ -n "$compare_json" && -f "$compare_json" ]]; then
      echo "## Comparison"
      echo
      jq -r '
        "- baseline: \(.baseline.git_ref) (\(.baseline.git_commit))",
        "- current: \(.current.git_ref) (\(.current.git_commit))",
        "- latency_ms ratio: \(.ratios.latency_ms // "-")",
        "- prefill_ms ratio: \(.ratios.prefill_ms // "-")",
        "- decode tps ratio: \(.ratios.tps // "-")",
        "- status: **\(.status)**",
        (if (.regressions | length) > 0 then "- regressions: " + (.regressions | join(", ")) else empty end)
      ' "$compare_json"
    fi
    echo
    echo "Decode tps prefers \`juno.TokenProduced.tps\` from JFR when present; latency from \`x_juno_latency_ms\`."
    echo "Prefill: \`juno.ForwardPass.prefill.total_ms\` (vision+text). Default launch uses \`--prefill single\` for caption quality."
  } >"$index_md"
}

# ── main ─────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_ROOT"
CURRENT_REF="${CURRENT_REF:-HEAD}"
trap 'stop_juno' EXIT

if [[ -n "$BASELINE_REF" ]]; then
  log "comparing baseline=${BASELINE_REF} vs current=${CURRENT_REF}"
  run_at_ref "$BASELINE_REF" "baseline"
  run_at_ref "$CURRENT_REF" "current"
  write_compare_json "${OUT_ROOT}/baseline.json" "${OUT_ROOT}/current.json" "${OUT_ROOT}/compare.json"
  write_index_md "${OUT_ROOT}/INDEX.md" "${OUT_ROOT}/baseline.json" "${OUT_ROOT}/current.json" "${OUT_ROOT}/compare.json"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    status="$(jq -r '.status' "${OUT_ROOT}/compare.json")"
    log "comparison status: ${status}"
    jq -r '"latency_ms ratio=\(.ratios.latency_ms // "-") prefill_ms ratio=\(.ratios.prefill_ms // "-") tps ratio=\(.ratios.tps // "-")"' \
      "${OUT_ROOT}/compare.json"
    [[ "$status" == "ok" ]] || exit 1
  fi
else
  run_at_ref "$CURRENT_REF" "current"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    cp "${OUT_ROOT}/current.json" "${OUT_ROOT}/vision-perf.json"
  fi
  write_index_md "${OUT_ROOT}/INDEX.md" "${OUT_ROOT}/current.json" "${OUT_ROOT}/current.json" ""
fi

if [[ "$PUBLISH" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-vision"
  mkdir -p "$dest"
  cp -a "$OUT_ROOT"/* "$dest/"
  log "published → docs/perf-compare/${RUN_ID}-vision/"
fi

log "done: ${OUT_ROOT}"
