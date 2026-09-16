#!/usr/bin/env bash
# compare-llama-cpp.sh — local perf compare: llama.cpp vs Juno (CPU or GPU)
#
# For each selected GGUF under models/, runs:
#   1) llama-bench  (prompt eval + token gen tokens/s)
#   2) juno local API with --jfr (pp/tg from JFR ForwardPass + TokenProduced; API latency kept too)
# and writes one result JSON per engine per model under target/perf-compare/<run-id>/.
# On success, also publishes metrics into docs/perf-compare/<run-id>/.
#
# Baselines: docs/perf-compare/README.md
#
# Usage:
#   ./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 0
#   ./scripts/performance-tests/compare-llama-cpp.sh --gpu
#   ./scripts/performance-tests/compare-llama-cpp.sh --gpu --models tinyllama,qwen2.5-3b
#   ./scripts/performance-tests/compare-llama-cpp.sh --list
#   ./scripts/performance-tests/compare-llama-cpp.sh --no-publish
#
# Env:
#   LLAMA_CPP_BIN   dir containing llama-bench (default: CUDA build if --gpu and present,
#                   else ../llama.cpp-bin/llama-b9551)
#   JUNO_USE_VECTOR 1 = keep jdk.incubator.vector (default); 0 = scalar kernels
#   COMPARE_HEAP    JVM -Xmx for juno (default: auto from model size)
#   CUDA_HOME / CUDA_PATH  optional; prepended to LD_LIBRARY_PATH for Juno GPU

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

N_PROMPT=128
N_GEN=64
REPS=3
N_THREADS="$(nproc 2>/dev/null || echo 6)"
NGL=""
API_PORT=18080
JUNO_USE_VECTOR="${JUNO_USE_VECTOR:-1}"
PROMPT_TEXT="could you please write me a short poem about love and war"
RAW_PROMPT=0
JUNO_GPU_LAYERS=""
JUNO_MMQ=""
JUNO_GPU_ATTENTION=""
JUNO_SCHEDULE=""
JUNO_CACHE_TYPE_K=""
JUNO_CACHE_TYPE_V=""
JUNO_KV_PAGE_SIZE=""
MODEL_FILTER=""
MISTRAL_TUNED_LANE=1
DRY_RUN=0
LIST_ONLY=0
PUBLISH=1
USE_GPU=0
LLAMA_CPP_BIN_EXPLICIT=""
USE_JFR=1
JFR_DURATION="${JFR_DURATION:-30m}"

# Default model set: text GGUFs both engines load locally (skip huge / vision / llamafile /
# architectures Juno cannot load yet, e.g. Qwen3.5).
DEFAULT_MODELS=(
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  qwen2.5-3b-instruct-q4_k_m.gguf
  Phi-3.5-mini-instruct-Q4_K_M.gguf
  mistral-7b-instruct-v0.1-q4_k_m.gguf
)

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
  cat <<EOF

Options:
  --models LIST     Comma-separated basenames or substrings (default: curated set)
  --all-gguf        Every *.gguf under models/ (except skipped patterns)
  --n-prompt N      llama-bench prompt tokens (default: ${N_PROMPT})
  --n-gen N         gen tokens for both engines (default: ${N_GEN})
  --reps N          llama-bench repetitions (default: ${REPS})
  --threads N       CPU threads (default: ${N_THREADS})
  --cpu             Force CPU (llama -ngl 0, juno --cpu) [default]
  --gpu             GPU mode (llama -ngl 99 unless --ngl set, juno --gpu)
  --ngl N           llama.cpp GPU layers (overrides --cpu/--gpu default)
  --gpu-layers N|all|auto  Juno --gpu-layers (default: all in GPU mode)
  --mmq on|off|auto Juno --mmq (packed Q4_K device GEMV; default off)
  --gpu-attention on|off|auto  Juno --gpu-attention (GPU-resident attention kernel; default off)
  --schedule static|continuous  Juno --schedule (default static)
  --cache-type-k f16|q8_0  Juno --cache-type-k (default f16)
  --cache-type-v f16|q8_0  Juno --cache-type-v (default f16)
  --kv-page-size N  Juno --kv-page-size (default 16, schedule=continuous only)
  --raw-prompt      Repeat a minimal token pattern (~1 tok/word) for prompt-length parity
  --api-port N      Juno REST port (default: ${API_PORT})
  --out DIR         Output directory (default: target/perf-compare/<timestamp>)
  --llama-bin DIR   Directory with llama-bench
  --vector 0|1      Pass jdk.incubator.vector to Juno (default: ${JUNO_USE_VECTOR})
  --jfr DURATION    Enable Juno JFR for DURATION (default: ${JFR_DURATION}; on by default)
  --no-jfr          Skip Juno --jfr (API latency only for Juno tg)
  --publish         Copy metrics JSON+INDEX into docs/perf-compare/ (default)
  --no-publish      Skip docs/perf-compare publish
  --no-mistral-tuned-lane  Skip the extra mistral-7b tuned lane (--mmq on --gpu-layers auto)
                    that GPU runs add automatically when mistral-7b is selected (default: on).
                    The vanilla default-flags lane always runs regardless; this only controls
                    the second, additional run. See docs/infra-plan/PLAN-Infra-Review-Fixes.md
                    item 8 — the default-flags Mistral-7B number is not representative of a
                    production config, so the regression sweep also measures the tuned one.
  --list            List selected models and exit
  -n, --dry-run     Print commands only
  -h, --help        This help
EOF
}

log()  { printf '[compare] %s\n' "$*"; }
warn() { printf '[compare] warn: %s\n' "$*" >&2; }
die()  { printf '[compare] error: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

json_escape() {
  # minimal JSON string escape
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --all-gguf) MODEL_FILTER="__ALL__"; shift ;;
    --n-prompt) N_PROMPT="$2"; shift 2 ;;
    --n-gen) N_GEN="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --threads) N_THREADS="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --ngl) NGL="$2"; shift 2 ;;
    --gpu-layers) JUNO_GPU_LAYERS="$2"; shift 2 ;;
    --mmq) JUNO_MMQ="$2"; shift 2 ;;
    --gpu-attention) JUNO_GPU_ATTENTION="$2"; shift 2 ;;
    --schedule) JUNO_SCHEDULE="$2"; shift 2 ;;
    --cache-type-k) JUNO_CACHE_TYPE_K="$2"; shift 2 ;;
    --cache-type-v) JUNO_CACHE_TYPE_V="$2"; shift 2 ;;
    --kv-page-size) JUNO_KV_PAGE_SIZE="$2"; shift 2 ;;
    --raw-prompt) RAW_PROMPT=1; shift ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --llama-bin) LLAMA_CPP_BIN_EXPLICIT="$2"; shift 2 ;;
    --vector) JUNO_USE_VECTOR="$2"; shift 2 ;;
    --jfr) USE_JFR=1; JFR_DURATION="$2"; shift 2 ;;
    --no-jfr) USE_JFR=0; shift ;;
    --publish) PUBLISH=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    --no-mistral-tuned-lane) MISTRAL_TUNED_LANE=0; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# Resolve backend defaults after flag parse.
if [[ -z "$NGL" ]]; then
  if [[ "$USE_GPU" -eq 1 ]]; then
    NGL=99
  else
    NGL=0
  fi
else
  # Explicit --ngl implies GPU intent when > 0.
  if [[ "$NGL" != "0" ]]; then
    USE_GPU=1
  fi
fi

resolve_llama_bin() {
  if [[ -n "$LLAMA_CPP_BIN_EXPLICIT" ]]; then
    LLAMA_CPP_BIN="$LLAMA_CPP_BIN_EXPLICIT"
    return
  fi
  if [[ -n "${LLAMA_CPP_BIN:-}" ]]; then
    return
  fi
  local cuda_build="${ROOT}/../llama.cpp/build-cuda/bin"
  local stock="${ROOT}/../llama.cpp-bin/llama-b9551"
  if [[ "$USE_GPU" -eq 1 && -x "${cuda_build}/llama-bench" ]]; then
    LLAMA_CPP_BIN="$(cd "$cuda_build" && pwd)"
  elif [[ -x "${stock}/llama-bench" ]]; then
    LLAMA_CPP_BIN="$(cd "$stock" && pwd)"
  else
    LLAMA_CPP_BIN=""
  fi
}

setup_cuda_env() {
  [[ "$USE_GPU" -eq 1 ]] || return 0
  local cuda_root=""
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
    cuda_root="${CUDA_HOME}"
  elif [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]]; then
    cuda_root="${CUDA_PATH}"
  fi
  if [[ -n "$cuda_root" ]]; then
    export PATH="${cuda_root}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  # Distro CUDA packages often land in /usr/lib/x86_64-linux-gnu.
  if [[ -e /usr/lib/x86_64-linux-gnu/libcudart.so.12 ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export JUNO_USE_GPU=true
}

backend_label() {
  if [[ "$USE_GPU" -eq 1 ]]; then
    printf 'gpu'
  else
    printf 'cpu'
  fi
}

select_models() {
  local -a out=()
  local f base
  if [[ "$MODEL_FILTER" == "__ALL__" ]]; then
    for f in "${MODELS_DIR}"/*.gguf; do
      [[ -f "$f" ]] || continue
      base="$(basename "$f")"
      case "$base" in
        *mmproj*|moondream*|llama-1-30b*) continue ;;
      esac
      out+=("$base")
    done
  elif [[ -n "$MODEL_FILTER" ]]; then
    local IFS=',' part
    for part in $MODEL_FILTER; do
      part="$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      [[ -n "$part" ]] || continue
      if [[ -f "${MODELS_DIR}/${part}" ]]; then
        out+=("$part")
        continue
      fi
      local matched=0
      for f in "${MODELS_DIR}"/*.gguf; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        if [[ "$base" == *"$part"* ]]; then
          out+=("$base")
          matched=1
        fi
      done
      [[ "$matched" -eq 1 ]] || die "no model matching: ${part}"
    done
  else
    for base in "${DEFAULT_MODELS[@]}"; do
      [[ -f "${MODELS_DIR}/${base}" ]] || { warn "skip missing default model: ${base}"; continue; }
      out+=("$base")
    done
  fi
  # unique preserve order
  local -A seen=()
  SELECTED_MODELS=()
  for base in "${out[@]}"; do
    [[ -n "${seen[$base]:-}" ]] && continue
    seen[$base]=1
    SELECTED_MODELS+=("$base")
  done
  [[ ${#SELECTED_MODELS[@]} -gt 0 ]] || die "no models selected"
}

heap_for_model() {
  local path="$1" bytes heap_g
  if [[ -n "${COMPARE_HEAP:-}" ]]; then
    printf '%s' "$COMPARE_HEAP"
    return
  fi
  bytes="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")"
  # Rough: file_size * 1.5 + 2 GiB headroom, clamp 4g..48g
  heap_g=$(( (bytes * 3 / 2 + 2 * 1024 * 1024 * 1024 + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024) ))
  (( heap_g < 4 )) && heap_g=4
  (( heap_g > 48 )) && heap_g=48
  printf '%sg' "$heap_g"
}

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
  command -v java >/dev/null 2>&1 || die "java not found (need JDK 25+)"
  printf '%s' java
}

host_meta_json() {
  local cpu mem
  cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || uname -m)"
  mem="$(awk '/MemTotal/ {printf "%.1f GiB", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo unknown)"
  cat <<EOF
{
  "run_id": "$(json_escape "$RUN_ID")",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(json_escape "$(hostname 2>/dev/null || echo unknown)")",
  "cpu": "$(json_escape "$cpu")",
  "mem_total": "$(json_escape "$mem")",
  "n_threads": ${N_THREADS},
  "n_prompt": ${N_PROMPT},
  "n_gen": ${N_GEN},
  "reps": ${REPS},
  "ngl": ${NGL},
  "backend": "$(backend_label)",
  "use_gpu": ${USE_GPU},
  "prompt": "$(json_escape "$PROMPT_TEXT")",
  "raw_prompt": ${RAW_PROMPT},
  "juno_gpu_layers": "$(json_escape "${JUNO_GPU_LAYERS:-}")",
  "juno_mmq": "$(json_escape "${JUNO_MMQ:-}")",
  "juno_gpu_attention": "$(json_escape "${JUNO_GPU_ATTENTION:-}")",
  "juno_schedule": "$(json_escape "${JUNO_SCHEDULE:-}")",
  "juno_cache_type_k": "$(json_escape "${JUNO_CACHE_TYPE_K:-}")",
  "juno_cache_type_v": "$(json_escape "${JUNO_CACHE_TYPE_V:-}")",
  "juno_kv_page_size": "$(json_escape "${JUNO_KV_PAGE_SIZE:-}")",
  "juno_use_vector": ${JUNO_USE_VECTOR},
  "juno_jfr": ${USE_JFR},
  "jfr_duration": "$(json_escape "${JFR_DURATION}")",
  "llama_cpp_bin": "$(json_escape "${LLAMA_CPP_BIN:-}")"
}
EOF
}

run_llama_bench() {
  local model_path="$1" stem="$2"
  local out_json="${OUT_ROOT}/${stem}-llama-cpp.json"
  local raw="${OUT_ROOT}/${stem}-llama-cpp.raw.json"
  local bench="${LLAMA_CPP_BIN}/llama-bench"
  local logf="${OUT_ROOT}/${stem}-llama-cpp.log"

  [[ -x "$bench" ]] || die "llama-bench not found/executable: ${bench}"

  log "llama.cpp: ${stem} (pp=${N_PROMPT} tg=${N_GEN} reps=${REPS} ngl=${NGL})"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: LD_LIBRARY_PATH=${LLAMA_CPP_BIN} ${bench} -m ${model_path} -p ${N_PROMPT} -n ${N_GEN} -r ${REPS} -t ${N_THREADS} -ngl ${NGL} -o json"
    return 0
  fi

  local start_ns end_ns elapsed_ms rc=0
  start_ns="$(date +%s%N)"
  set +e
  (
    cd "$LLAMA_CPP_BIN"
    export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ./llama-bench -m "$model_path" -p "$N_PROMPT" -n "$N_GEN" -r "$REPS" \
      -t "$N_THREADS" -ngl "$NGL" -o json
  ) >"$raw" 2>"$logf"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

  local pp_tps tg_tps build_number model_type
  pp_tps="$(jq -r '[.[] | select(.n_prompt > 0 and .n_gen == 0) | .avg_ts] | if length>0 then .[0] else empty end' "$raw" 2>/dev/null || true)"
  tg_tps="$(jq -r '[.[] | select(.n_gen > 0 and .n_prompt == 0) | .avg_ts] | if length>0 then .[0] else empty end' "$raw" 2>/dev/null || true)"
  build_number="$(jq -r '.[0].build_number // empty' "$raw" 2>/dev/null || true)"
  model_type="$(jq -r '.[0].model_type // empty' "$raw" 2>/dev/null || true)"

  cat >"$out_json" <<EOF
{
  "engine": "llama.cpp",
  "tool": "llama-bench",
  "status": "$([ "$rc" -eq 0 ] && echo success || echo failure)",
  "exit_code": ${rc},
  "model": "$(json_escape "$(basename "$model_path")")",
  "model_path": "$(json_escape "$model_path")",
  "model_type": "$(json_escape "${model_type:-}")",
  "build_number": ${build_number:-null},
  "n_prompt": ${N_PROMPT},
  "n_gen": ${N_GEN},
  "repetitions": ${REPS},
  "threads": ${N_THREADS},
  "n_gpu_layers": ${NGL},
  "prompt_eval_tps": ${pp_tps:-null},
  "token_gen_tps": ${tg_tps:-null},
  "wall_ms": ${elapsed_ms},
  "raw_json": "$(json_escape "$raw")",
  "log": "$(json_escape "$logf")",
  "host": $(host_meta_json)
}
EOF
  # Attach llama-bench samples when raw output is valid JSON.
  if [[ -s "$raw" ]] && jq -e 'type == "array"' "$raw" >/dev/null 2>&1; then
    jq --slurpfile raw "$raw" '. + {llama_bench: $raw[0]}' "$out_json" >"${out_json}.tmp" \
      && mv "${out_json}.tmp" "$out_json"
  fi

  if [[ "$rc" -ne 0 ]]; then
    warn "llama-bench failed for ${stem} (rc=${rc}) — see ${logf}"
    return 1
  fi
  log "llama.cpp result: ${out_json}  (pp=${pp_tps:-?} tg=${tg_tps:-?} t/s)"
  return 0
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
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
}

wait_for_juno_metrics() {
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
      forward_pass_prefill_p95_ms: ($m."juno.ForwardPass.prefill.p95_ms" // null),
      forward_pass_decode_p95_ms: ($m."juno.ForwardPass.decode.p95_ms" // null),
      forward_pass_prefill_count: ($m."juno.ForwardPass.prefill.count" // null),
      forward_pass_decode_count: ($m."juno.ForwardPass.decode.count" // null),
      token_produced_count: ($m."juno.TokenProduced.count" // null),
      token_produced_tps: $token_tps,
      token_produced_elapsed_s: ($m."juno.TokenProduced.elapsed_seconds" // null)
    }
  '
}

run_juno() {
  local model_path="$1" stem="$2"
  local out_json="${OUT_ROOT}/${stem}-juno.json"
  local resp="${OUT_ROOT}/${stem}-juno-response.json"
  local logf="${OUT_ROOT}/${stem}-juno.log"
  local jar java_bin heap
  local -a java_args=()

  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  heap="$(heap_for_model "$model_path")"

  local juno_backend_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && juno_backend_flag="--gpu"

  java_args=(
    --enable-preview
    --enable-native-access=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED
    --add-opens java.base/java.nio=ALL-UNNAMED
  )
  if [[ "$JUNO_USE_VECTOR" == "1" ]]; then
    java_args+=(--add-modules jdk.incubator.vector)
  fi
  java_args+=(
    -XX:+UseG1GC
    -XX:+AlwaysPreTouch
    -Xms512m
    -Xmx"${heap}"
    -Djuno.byteOrder=BE
    -jar "$jar"
    --model-path "$model_path"
    --dtype FLOAT16
    --byteOrder BE
    --max-tokens "$N_GEN"
    --temperature 0
    --top-k 0
    --top-p 0
    --nodes 1
    --local
    "$juno_backend_flag"
    --api-port "$API_PORT"
  )
  if [[ "$USE_JFR" -eq 1 ]]; then
    java_args+=(--jfr "$JFR_DURATION")
  fi
  if [[ -n "$JUNO_GPU_LAYERS" ]]; then
    java_args+=(--gpu-layers "$JUNO_GPU_LAYERS")
  fi
  if [[ -n "$JUNO_MMQ" ]]; then
    java_args+=(--mmq "$JUNO_MMQ")
  fi
  if [[ -n "$JUNO_GPU_ATTENTION" ]]; then
    java_args+=(--gpu-attention "$JUNO_GPU_ATTENTION")
  fi
  if [[ -n "${JUNO_PREFILL_BATCH:-}" ]]; then
    java_args+=(--prefill-batch "$JUNO_PREFILL_BATCH")
  fi
  if [[ -n "$JUNO_SCHEDULE" ]]; then
    java_args+=(--schedule "$JUNO_SCHEDULE")
  fi
  if [[ -n "$JUNO_CACHE_TYPE_K" ]]; then
    java_args+=(--cache-type-k "$JUNO_CACHE_TYPE_K")
  fi
  if [[ -n "$JUNO_CACHE_TYPE_V" ]]; then
    java_args+=(--cache-type-v "$JUNO_CACHE_TYPE_V")
  fi
  if [[ -n "$JUNO_KV_PAGE_SIZE" ]]; then
    java_args+=(--kv-page-size "$JUNO_KV_PAGE_SIZE")
  fi

  log "juno: ${stem} (backend=$(backend_label) max_tokens=${N_GEN} heap=${heap} vector=${JUNO_USE_VECTOR} jfr=${USE_JFR} port=${API_PORT} gpu_layers=${JUNO_GPU_LAYERS:-default})"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: ${java_bin} ${java_args[*]}"
    return 0
  fi

  # Ensure port free
  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} already has a healthy Juno API — stop it or pass --api-port"
  fi

  : >"$logf"
  # Keep stdin open: ConsoleMain REPL exits on EOF (readLine == null), which would
  # tear down the API mid-benchmark when launched non-interactively.
  (
    cd "$ROOT"
    # shellcheck disable=SC2094
    exec "$java_bin" "${java_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  local start_ns end_ns wall_ms load_ms gen_ms rc=0
  local completion_tokens=0 prompt_tokens=0 latency_ms=0 tps=0
  local model_id="" finish_reason=""
  local jfr_metrics="${OUT_ROOT}/${stem}-juno-jfr.json"
  local jfr_block="null"
  local jfr_pp_tps="null" jfr_tg_tps="null"

  load_start="$(date +%s%N)"
  if ! wait_for_juno_api "$API_PORT" 600 "$JUNO_PID"; then
    warn "juno API did not become healthy — see ${logf}"
    stop_juno
    local err_snip
    err_snip="$(grep -E 'Exception|Error|Tensor not found|Unsupported' "$logf" 2>/dev/null | tail -3 | tr '\n' ' ' | head -c 400 || true)"
    cat >"$out_json" <<EOF
{
  "engine": "juno",
  "tool": "local-api",
  "status": "failure",
  "error": "api_health_timeout_or_crash",
  "error_detail": "$(json_escape "${err_snip:-}")",
  "model": "$(json_escape "$(basename "$model_path")")",
  "model_path": "$(json_escape "$model_path")",
  "log": "$(json_escape "$logf")",
  "juno_use_vector": ${JUNO_USE_VECTOR},
  "backend": "$(backend_label)",
  "use_gpu": ${USE_GPU},
  "heap": "$(json_escape "$heap")",
  "host": $(host_meta_json)
}
EOF
    return 1
  fi
  load_end="$(date +%s%N)"
  load_ms=$(( (load_end - load_start) / 1000000 ))

  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  if [[ -z "$model_id" ]]; then
    # OpenAI list shape uses .data[].id; native may differ — fall back to filename stem
    model_id="$(basename "$model_path")"
    model_id="${model_id%.gguf}"
  fi

  start_ns="$(date +%s%N)"
  set +e
  curl -sS --max-time 7200 -o "$resp" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$model_id" --arg p "$PROMPT_TEXT" --argjson n "$N_GEN" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')" \
    >"${OUT_ROOT}/${stem}-juno.http" 2>>"$logf"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local http_code
  http_code="$(cat "${OUT_ROOT}/${stem}-juno.http" 2>/dev/null || echo 000)"

  if [[ "$rc" -eq 0 && "$http_code" == "200" && -s "$resp" ]]; then
    completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$resp")"
    prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$resp")"
    latency_ms="$(jq -r '.x_juno_latency_ms // 0' "$resp")"
    finish_reason="$(jq -r '.choices[0].finish_reason // empty' "$resp")"
    if [[ "$latency_ms" =~ ^[0-9]+$ ]] && (( latency_ms > 0 && completion_tokens > 0 )); then
      tps="$(awk -v t="$completion_tokens" -v ms="$latency_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
    elif (( wall_ms > 0 && completion_tokens > 0 )); then
      tps="$(awk -v t="$completion_tokens" -v ms="$wall_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
      latency_ms="$wall_ms"
    fi
    gen_ms="$latency_ms"
  else
    warn "juno chat failed http=${http_code} rc=${rc} — see ${logf} ${resp}"
    rc=1
  fi

  stop_juno
  sleep 2

  if [[ "$USE_JFR" -eq 1 ]]; then
    if wait_for_juno_metrics "$jfr_metrics" 45; then
      jfr_block="$(jfr_summary_json "$jfr_metrics" "$prompt_tokens" "$completion_tokens" "${gen_ms:-0}")"
      jfr_pp_tps="$(jq -r '.prompt_eval_tps // empty' <<<"$jfr_block" 2>/dev/null || true)"
      jfr_tg_tps="$(jq -r '.token_gen_tps // empty' <<<"$jfr_block" 2>/dev/null || true)"
    else
      warn "JFR metrics.json not found after stop — see ${logf}"
    fi
  fi

  # Prefer JFR tg for compare when available; keep API tg separately.
  local compare_pp="$jfr_pp_tps" compare_tg="$tps"
  if [[ -n "$jfr_tg_tps" && "$jfr_tg_tps" != "null" ]]; then
    compare_tg="$jfr_tg_tps"
  fi
  [[ -z "$compare_pp" ]] && compare_pp=null
  [[ -z "$compare_tg" ]] && compare_tg=null
  [[ -z "$tps" ]] && tps=null
  [[ -z "${gen_ms:-}" ]] && gen_ms=0

  jq -n \
    --arg engine "juno" \
    --arg tool "local-api+jfr" \
    --arg status "$([ "$rc" -eq 0 ] && echo success || echo failure)" \
    --argjson exit_code "$rc" \
    --arg http_code "$http_code" \
    --arg model "$(basename "$model_path")" \
    --arg model_path "$model_path" \
    --arg model_id "$model_id" \
    --argjson n_gen "$N_GEN" \
    --argjson prompt_tokens "${prompt_tokens:-0}" \
    --argjson completion_tokens "${completion_tokens:-0}" \
    --arg finish_reason "${finish_reason:-}" \
    --argjson load_ms "$load_ms" \
    --argjson latency_ms "${gen_ms:-0}" \
    --argjson wall_ms "$wall_ms" \
    --argjson api_token_gen_tps "${tps:-null}" \
    --argjson prompt_eval_tps "${compare_pp:-null}" \
    --argjson token_gen_tps "${compare_tg:-null}" \
    --argjson juno_use_vector "$JUNO_USE_VECTOR" \
    --arg backend "$(backend_label)" \
    --argjson use_gpu "$USE_GPU" \
    --arg heap "$heap" \
    --argjson threads_hint "$N_THREADS" \
    --argjson use_jfr "$USE_JFR" \
    --arg jfr_duration "$JFR_DURATION" \
    --arg response_json "$resp" \
    --arg log "$logf" \
    --argjson jfr "$jfr_block" \
    --argjson host "$(host_meta_json)" \
    '{
      engine: $engine,
      tool: (if $use_jfr == 1 then $tool else "local-api" end),
      status: $status,
      exit_code: $exit_code,
      http_code: $http_code,
      model: $model,
      model_path: $model_path,
      model_id: $model_id,
      n_gen: $n_gen,
      prompt_tokens: $prompt_tokens,
      completion_tokens: $completion_tokens,
      finish_reason: $finish_reason,
      load_ms: $load_ms,
      latency_ms: $latency_ms,
      wall_ms: $wall_ms,
      api_token_gen_tps: $api_token_gen_tps,
      prompt_eval_tps: $prompt_eval_tps,
      token_gen_tps: $token_gen_tps,
      juno_use_vector: $juno_use_vector,
      backend: $backend,
      use_gpu: $use_gpu,
      heap: $heap,
      threads_hint: $threads_hint,
      use_jfr: $use_jfr,
      jfr_duration: $jfr_duration,
      jfr: (if $use_jfr == 1 then $jfr else null end),
      response_json: $response_json,
      log: $log,
      host: $host
    }' >"$out_json"

  if [[ "$rc" -ne 0 ]]; then
    return 1
  fi
  log "juno result: ${out_json}  (jfr pp=${compare_pp:-?} tg=${compare_tg:-?} t/s; api tg=${tps:-?} t/s)"
  return 0
}

run_mistral_tuned_lane() {
  # Mistral-7B's default-flags (--mmq off, --gpu-layers unset) lane sits at a
  # ~30-40x deficit vs the --mmq on --gpu-layers auto config on 8 GiB cards
  # (see docs/infra-plan/PLAN-Infra-Review-Fixes.md item 8) — a config nobody
  # would actually run in production. Add a second lane so the standing GPU
  # regression sweep measures both, not just the unrepresentative default.
  local model_path="$1" stem="$2"
  local tuned_stem="${stem}-tuned"
  local saved_mmq="$JUNO_MMQ" saved_gpu_layers="$JUNO_GPU_LAYERS"

  JUNO_MMQ="on"
  JUNO_GPU_LAYERS="auto"
  log "=== mistral tuned lane: ${stem} (--mmq on --gpu-layers auto) ==="
  run_juno "$model_path" "$tuned_stem"
  local rc=$?
  JUNO_MMQ="$saved_mmq"
  JUNO_GPU_LAYERS="$saved_gpu_layers"

  # Reuse the vanilla lane's llama.cpp reference numbers — the reference engine
  # doesn't read Juno's flags, so re-running llama-bench would just add noise.
  if [[ -f "${OUT_ROOT}/${stem}-llama-cpp.json" ]]; then
    cp -a "${OUT_ROOT}/${stem}-llama-cpp.json" "${OUT_ROOT}/${tuned_stem}-llama-cpp.json"
  fi
  write_pair_summary "$tuned_stem"
  STEMS+=("$tuned_stem")
  return "$rc"
}

write_pair_summary() {
  local stem="$1"
  local llama_f="${OUT_ROOT}/${stem}-llama-cpp.json"
  local juno_f="${OUT_ROOT}/${stem}-juno.json"
  local out="${OUT_ROOT}/${stem}-compare.json"
  [[ -f "$llama_f" && -f "$juno_f" ]] || return 0
  if ! command -v jq >/dev/null 2>&1; then
    return 0
  fi
  jq -n --slurpfile l "$llama_f" --slurpfile j "$juno_f" \
    --arg llama_result "$llama_f" --arg juno_result "$juno_f" '
    def num(x): if x == null then null else x end;
    {
      model: (( $l[0].model ) // ( $j[0].model )),
      llama_cpp: {
        status: $l[0].status,
        prompt_eval_tps: num($l[0].prompt_eval_tps),
        token_gen_tps: num($l[0].token_gen_tps)
      },
      juno: {
        status: $j[0].status,
        prompt_eval_tps: num($j[0].prompt_eval_tps),
        token_gen_tps: num($j[0].token_gen_tps),
        api_token_gen_tps: num($j[0].api_token_gen_tps),
        latency_ms: $j[0].latency_ms,
        prompt_tokens: $j[0].prompt_tokens,
        completion_tokens: $j[0].completion_tokens,
        juno_use_vector: $j[0].juno_use_vector,
        use_jfr: $j[0].use_jfr
      },
      ratio_juno_over_llamacpp_pp:
        (if ($l[0].prompt_eval_tps != null and $j[0].prompt_eval_tps != null and $l[0].prompt_eval_tps > 0)
         then ($j[0].prompt_eval_tps / $l[0].prompt_eval_tps)
         else null end),
      ratio_juno_over_llamacpp_tg:
        (if ($l[0].token_gen_tps != null and $j[0].token_gen_tps != null and $l[0].token_gen_tps > 0)
         then ($j[0].token_gen_tps / $l[0].token_gen_tps)
         else null end),
      result_files: {
        llama_cpp_raw: ($l[0].raw_json | values),
        juno_response: ($j[0].response_json | values),
        llama_result: $llama_result,
        juno_result: $juno_result,
        juno_jfr: (if ($j[0].use_jfr // false) then ($juno_result | sub("-juno\\.json$"; "-juno-jfr.json")) else null end)
      }
    }
  ' >"$out" || { warn "compare json failed for ${stem}"; return 0; }
  log "compare: ${out}"
}

write_run_index() {
  local index="${OUT_ROOT}/INDEX.md"
  local juno_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && juno_flag="--gpu"
  {
    echo "# llama.cpp vs Juno - ${RUN_ID} ($(backend_label))"
    echo
    echo "| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |"
    echo "|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|"
    local stem llama_f juno_f cmp pp tg jpp jt ratio_pp ratio_tg
    for stem in "${STEMS[@]}"; do
      llama_f="${OUT_ROOT}/${stem}-llama-cpp.json"
      juno_f="${OUT_ROOT}/${stem}-juno.json"
      cmp="${OUT_ROOT}/${stem}-compare.json"
      pp="$(jq -r 'if .prompt_eval_tps == null then "-" else .prompt_eval_tps end' "$llama_f" 2>/dev/null || echo -)"
      tg="$(jq -r 'if .token_gen_tps == null then "-" else .token_gen_tps end' "$llama_f" 2>/dev/null || echo -)"
      jpp="$(jq -r 'if .prompt_eval_tps == null then "-" else .prompt_eval_tps end' "$juno_f" 2>/dev/null || echo -)"
      jt="$(jq -r 'if .token_gen_tps == null then "-" else .token_gen_tps end' "$juno_f" 2>/dev/null || echo -)"
      if [[ -f "$cmp" ]]; then
        ratio_pp="$(jq -r 'if .ratio_juno_over_llamacpp_pp == null then "-" else .ratio_juno_over_llamacpp_pp end' "$cmp" 2>/dev/null || echo -)"
        ratio_tg="$(jq -r 'if .ratio_juno_over_llamacpp_tg == null then "-" else .ratio_juno_over_llamacpp_tg end' "$cmp" 2>/dev/null || echo -)"
      else
        ratio_pp="-"
        ratio_tg="-"
      fi
      printf '| %s | %s | %s | %s | %s | %s | %s | %s-*.json |\n' \
        "$stem" "$pp" "$tg" "$jpp" "$jt" "$ratio_pp" "$ratio_tg" "$stem"
    done
    echo
    echo "Host meta: see any *-llama-cpp.json .host field."
    echo
    echo "Notes:"
    echo "- llama.cpp metrics from llama-bench (avg_ts)."
    if [[ "$USE_JFR" -eq 1 ]]; then
      echo "- Juno pp/tg from JFR (--jfr ${JFR_DURATION}): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms)."
    else
      echo "- Juno tg from POST /v1/chat/completions (x_juno_latency_ms / completion_tokens); JFR disabled (--no-jfr)."
    fi
    echo "- Backend: $(backend_label) (llama -ngl ${NGL} / juno ${juno_flag}), temperature 0, max_tokens=${N_GEN}."
    if [[ "$JUNO_USE_VECTOR" == "0" ]]; then
      echo "- Juno ran without jdk.incubator.vector (scalar kernels)."
    else
      echo "- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0."
    fi
  } >"$index"
  log "index: ${index}"
}

publish_results() {
  [[ "$PUBLISH" -eq 1 ]] || { log "publish skipped (--no-publish)"; return 0; }
  [[ "$DRY_RUN" -eq 1 ]] && { log "dry-run: would publish ${OUT_ROOT} → ${DOCS_PUBLISH_ROOT}"; return 0; }
  [[ -d "$OUT_ROOT" ]] || return 0

  local docs_dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}"
  mkdir -p "$docs_dest"

  # Metrics only (skip large logs / response dumps).
  local f
  for f in "$OUT_ROOT"/INDEX.md "$OUT_ROOT"/host.json \
           "$OUT_ROOT"/*-llama-cpp.json "$OUT_ROOT"/*-juno.json "$OUT_ROOT"/*-juno-jfr.json "$OUT_ROOT"/*-compare.json; do
    [[ -e "$f" ]] || continue
    cp -a "$f" "$docs_dest/"
  done

  log "published: ${docs_dest}"
  if [[ -f "${DOCS_PUBLISH_ROOT}/README.md" ]]; then
    log "summary doc: ${DOCS_PUBLISH_ROOT}/README.md (update manually for narrative baselines)"
  fi
}

# ── main ─────────────────────────────────────────────────────────────────────

require_cmd jq
require_cmd curl
require_cmd awk
require_cmd date

resolve_llama_bin
select_models

if [[ "$RAW_PROMPT" -eq 1 ]]; then
  PROMPT_TEXT="$(python3 -c "print(' '.join(['x'] * ${N_PROMPT}))")"
  log "raw-prompt: using ${N_PROMPT} minimal space-separated tokens for Juno chat prefill parity"
fi

if [[ "$LIST_ONLY" -eq 1 ]]; then
  log "selected ${#SELECTED_MODELS[@]} model(s) (backend=$(backend_label)):"
  printf '  %s\n' "${SELECTED_MODELS[@]}"
  exit 0
fi

[[ -n "${LLAMA_CPP_BIN:-}" && -d "$LLAMA_CPP_BIN" ]] \
  || die "llama.cpp bin dir not found (set LLAMA_CPP_BIN or --llama-bin). Tried: ${LLAMA_CPP_BIN:-<empty>}"
[[ -x "${LLAMA_CPP_BIN}/llama-bench" ]] || die "llama-bench missing in ${LLAMA_CPP_BIN}"

setup_cuda_env
if [[ "$USE_GPU" -eq 1 ]]; then
  log "GPU mode: ngl=${NGL} juno=--gpu llama_bin=${LLAMA_CPP_BIN}"
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi failed — GPU run may fall back to CPU"
  else
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  fi
fi

mkdir -p "$OUT_ROOT"
host_meta_json >"${OUT_ROOT}/host.json"
log "output: ${OUT_ROOT}"
log "models: ${#SELECTED_MODELS[@]}"

STEMS=()
failures=0
trap 'stop_juno' EXIT

for base in "${SELECTED_MODELS[@]}"; do
  model_path="${MODELS_DIR}/${base}"
  stem="${base%.gguf}"
  stem="${stem//\//_}"
  STEMS+=("$stem")
  log "=== model: ${base} ==="
  run_llama_bench "$model_path" "$stem" || failures=$((failures + 1))
  run_juno "$model_path" "$stem" || failures=$((failures + 1))
  write_pair_summary "$stem"
  if [[ "$USE_GPU" -eq 1 && "$MISTRAL_TUNED_LANE" -eq 1 && "$base" == mistral-7b* ]]; then
    run_mistral_tuned_lane "$model_path" "$stem" || failures=$((failures + 1))
  fi
done

write_run_index
publish_results
log "done. failures=${failures}  results in ${OUT_ROOT}"
exit "$failures"
