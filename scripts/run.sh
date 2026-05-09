#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# juno — runtime launcher  (no Maven required)
# Uses pre-built shade jars from target/.  Build first with:
#   mvn clean package -DskipTests
#
# Requires: JDK 21+
# Runs on:  Linux · macOS · Windows (Git Bash / WSL)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer shaded runnable (juno-player-*-shaded.jar); fall back to legacy juno-player.jar.
shopt -s nullglob
_JUNO_SHADED=( "$DIR/juno-player/target/"juno-player-*-shaded.jar )
shopt -u nullglob
if [[ ${#_JUNO_SHADED[@]} -gt 0 ]]; then
  JUNO_PLAYER_JAR="${_JUNO_SHADED[0]}"
else
  JUNO_PLAYER_JAR="$DIR/juno-player/target/juno-player.jar"
fi
unset _JUNO_SHADED
LIVE_JAR="$DIR/juno-master/target/juno-master.jar"
HEALTH_JAR="$DIR/health/target/juno-health.jar"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'
info() { echo -e "${CYAN}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✔ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✖ $*${NC}" >&2; exit 1; }

# ── OS detection ──────────────────────────────────────────────────────────────
detect_os() {
  case "${OSTYPE:-}" in
    linux*)                   echo "linux"   ;;
    darwin*)                  echo "macos"   ;;
    msys* | cygwin* | win32*) echo "windows" ;;
    *)
      # OSTYPE not set (some minimal shells) — fall back to uname
      local u
      u="$(uname -s 2>/dev/null || echo unknown)"
      case "$u" in
        Linux*)               echo "linux"   ;;
        Darwin*)              echo "macos"   ;;
        MINGW* | CYGWIN* | Windows_NT*) echo "windows" ;;
        *)
          # Last resort: check for WSL
          if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "linux"   # WSL — behaves like Linux
          else
            echo "unknown"
          fi ;;
      esac ;;
  esac
}

OS="$(detect_os)"

# ── Java discovery ────────────────────────────────────────────────────────────
find_java() {
  # 1. JAVA_HOME explicitly set
  if [[ -n "${JAVA_HOME:-}" ]]; then
    if [[ "$OS" == "windows" ]]; then
      echo "$JAVA_HOME/bin/java.exe"
    else
      echo "$JAVA_HOME/bin/java"
    fi
    return
  fi

  # 2. java already on PATH
  if command -v java >/dev/null 2>&1; then
    echo "java"
    return
  fi

  # 3. Windows common install locations (Git Bash paths)
  if [[ "$OS" == "windows" ]]; then
    local candidates=(
      "/c/Program Files/Eclipse Adoptium/jdk-21"*"/bin/java.exe"
      "/c/Program Files/Eclipse Adoptium/jdk-25"*"/bin/java.exe"
      "/c/Program Files/Java/jdk-21"*"/bin/java.exe"
      "/c/Program Files/Java/jdk-25"*"/bin/java.exe"
      "/c/Program Files/Microsoft/jdk-21"*"/bin/java.exe"
      "/c/Program Files/Microsoft/jdk-25"*"/bin/java.exe"
    )
    for pattern in "${candidates[@]}"; do
      # shellcheck disable=SC2086
      for candidate in $pattern; do
        [[ -x "$candidate" ]] && { echo "$candidate"; return; }
      done
    done
  fi

  err "JDK 21+ not found. Install from https://adoptium.net and set JAVA_HOME."
}

JAVA="$(find_java)"

# ── CUDA runtime on PATH (GPU mode) ─────────────────────────────────────────────
# When using GPU, prepend CUDA toolkit bin so libcudart / jnicudart load.
prepend_cuda_bin_to_path_if_gpu() {
  local use_gpu="$1"
  [[ "$use_gpu" == "true" ]] || return 0
  local cuda_bin=""
  if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/bin" ]]; then
    cuda_bin="${CUDA_PATH}/bin"
  elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
    cuda_bin="${CUDA_HOME}/bin"
  fi
  if [[ -n "$cuda_bin" ]]; then
    export PATH="${cuda_bin}:${PATH}"
  fi
}

# ── Java version check ────────────────────────────────────────────────────────
check_java_version() {
  local ver
  ver=$("$JAVA" -version 2>&1 | awk -F'"' '/version/{print $2}' | cut -d. -f1)
  [[ "${ver:-0}" -ge 21 ]] || err "JDK 21+ required (found: $ver).  JAVA_HOME=$JAVA_HOME"
}

# ── Jar existence check ───────────────────────────────────────────────────────
require_jar() {
  local jar="$1" label="$2"
  if [[ ! -f "$jar" ]]; then
    err "$label jar not found: $jar\n  Build first: mvn clean package -DskipTests\n"
  fi
}

# ── Common JVM flags ──────────────────────────────────────────────────────────
# These suppress Guava/Netty sun.misc.Unsafe warnings and enable preview APIs.
JVM_BASE=(
  --enable-preview
  --enable-native-access=ALL-UNNAMED
  --add-opens java.base/java.lang=ALL-UNNAMED
  --add-opens java.base/java.nio=ALL-UNNAMED
  -XX:+UseG1GC
  -XX:+AlwaysPreTouch
)

# ── Commands ──────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# cluster — 3-node distributed cluster + interactive REPL (forked JVM nodes)
#           Use this for real GPU deployments and multi-node scenarios.
# ---------------------------------------------------------------------------
cmd_cluster() {
  local model="${MODEL_PATH:-}"
  local dtype="${DTYPE:-FLOAT16}"
  local byte_order="${BYTE_ORDER:-BE}"
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.6}"
  local top_k="${TOP_K:-20}"
  local top_p="${TOP_P:-0.95}"
  local heap="${HEAP:-4g}"
  local verbose="false"
  local ptype="pipeline"
  local jfr_duration=""
  local lora_play="${LORA_PLAY_PATH:-}"
  local health="false"
  local health_port="${HEALTH_PORT:-8081}"
  local api_port="${API_PORT:-}"
  local use_gpu="true"
  if [[ -n "${USE_GPU:-}" ]]; then
    case "${USE_GPU}" in
      false|0|no|NO) use_gpu="false" ;;
      *)             use_gpu="true"  ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)       model="$2";        shift 2 ;;
      --pType | --ptype)  ptype="$2";        shift 2 ;;
      --dtype)            dtype="$2";        shift 2 ;;
      --byteOrder | --byteorder | --byte-order) byte_order="${2^^}"; shift 2 ;;
      --max-tokens)       max_tokens="$2";   shift 2 ;;
      --temperature)      temperature="$2";  shift 2 ;;
      --top-k)            top_k="$2";        shift 2 ;;
      --top-p)            top_p="$2";        shift 2 ;;
      --heap)             heap="$2";         shift 2 ;;
      --jfr)              jfr_duration="$2"; shift 2 ;;
      --lora-play)        lora_play="$2";    shift 2 ;;
      --float16 | --fp16) dtype="FLOAT16";   shift   ;;
      --float32)          dtype="FLOAT32";   shift   ;;
      --int8)             dtype="INT8";      shift   ;;
      --gpu)              use_gpu="true";    shift   ;;
      --cpu)              use_gpu="false";   shift   ;;
      --health)           health="true";     shift   ;;
      --health-port)      health_port="$2";  shift 2 ;;
      --api-port)         api_port="$2";     shift 2 ;;
      --verbose | -v)     verbose="true";    shift   ;;
      --help)
        echo ""
        echo "  Usage: $0 cluster --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 cluster [flags]"
        echo ""
        echo "  Starts a 3-node cluster (one forked JVM per node) and an interactive"
        echo "  REPL. Each node serves gRPC on localhost:19092-19094."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH          Path to a GGUF model file"
        echo "                               (or set MODEL_PATH env var)"
        echo ""
        echo "  Parallelism:"
        echo "    --pType pipeline           pipeline-parallel: contiguous layer blocks,"
        echo "                               serial activation flow  (default)"
        echo "    --pType tensor             tensor-parallel: all layers on every node,"
        echo "                               weight-matrix slices, parallel AllReduce"
        echo "                               Constraint: numHeads % 3 == 0"
        echo ""
        echo "  Activation dtype:"
        echo "    --dtype FLOAT32|FLOAT16|INT8  wire format between nodes (default FLOAT16)"
        echo "    --float16 / --fp16            shorthand — 2x smaller gRPC payloads"
        echo "    --float32                     lossless, for debugging / reference runs"
        echo "    --int8                        ~4x smaller, ~1% relative error"
        echo ""
        echo "  Generation:"
        echo "    --max-tokens N             max tokens per response   (default 200)"
        echo "    --temperature F            sampling temperature       (default 0.6)"
        echo "    --top-k N                  top-K sampling cutoff     (default 20, 0=disabled)"
        echo "    --top-p F                  top-p nucleus sampling    (default 0.95, 0=disabled)"
        echo "    --api-port N               start REST API server on port N"
        echo "                               (includes OpenAI-compatible /v1/chat/completions)"
        echo ""
        echo "  Backend:"
        echo "    --gpu                      use GPU when available (default)"
        echo "    --cpu                      use CPU only"
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE                JVM heap  e.g. 4g 8g 16g  (default 4g)"
        echo "    --jfr DURATION             Enable Java Flight Recording for DURATION"
        echo "                               e.g. 5m 30s 1h — records from JVM start,"
        echo "                               writes juno-<timestamp>.jfr on exit"
        echo ""
        echo "  Logging:"
        echo "    --verbose / -v             show gRPC and node logs"
        echo ""
        echo "  Environment overrides:"
        echo "    MODEL_PATH  DTYPE  PTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  USE_GPU"
        echo ""
        echo "  Examples:"
        echo "    $0 cluster --model-path /models/tiny.gguf"
        echo "    $0 cluster --model-path /models/tiny.gguf --pType tensor"
        echo "    $0 cluster --model-path /models/tiny.gguf --jfr 5m"
        echo "    PTYPE=tensor MODEL_PATH=/models/tiny.gguf $0"
        echo ""
        exit 0 ;;
      *) err "Unknown cluster flag: $1.  Run: $0 cluster --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 cluster --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 cluster"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$JUNO_PLAYER_JAR" "juno-player"
  check_java_version

  warn "Starting 3-node cluster  (pType=${ptype}  dtype=${dtype}  byteOrder=${byte_order}  max_tokens=${max_tokens}  temperature=${temperature}  heap=${heap}  gpu=${use_gpu}  os=${OS})"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON"
  warn "Ctrl-C to stop all nodes and exit"
  echo ""

  local verbose_flag=""
  [[ "$verbose" == "true" ]] && verbose_flag="--verbose"

  local gpu_flag="--gpu"
  [[ "$use_gpu" == "false" ]] && gpu_flag="--cpu"
  prepend_cuda_bin_to_path_if_gpu "$use_gpu"

  # In cluster mode, ConsoleMain manages JFR programmatically via jdk.jfr.Recording,
  # exactly as it does in local mode.  Forward --jfr DURATION as a ConsoleMain
  # argument rather than a JVM flag so startClusterJfr() can own the lifecycle and
  # auto-extract metrics on exit.
  local jfr_arg=""
  [[ -n "$jfr_duration" ]] && jfr_arg="--jfr $jfr_duration" && \
    warn "JFR enabled — duration=${jfr_duration}  (programmatic recording, metrics auto-printed on exit)"

  local lora_play_arg=""
  [[ -n "$lora_play" ]] && { lora_play_arg="--lora-play $lora_play"; warn "LoRA inference overlay: ${lora_play}"; }
  local health_flag=""
  if [[ "$health" == "true" ]]; then
    health_flag="--health"
    warn "Health sidecar enabled on :${health_port} — dashboard at http://localhost:${health_port}/"
  fi
  local api_port_arg=""
  [[ -n "$api_port" ]] && api_port_arg="--api-port $api_port"

  # shellcheck disable=SC2086
  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    "-Djuno.node.heap=${heap}" \
    "-Djuno.byteOrder=${byte_order}" \
    -jar "$JUNO_PLAYER_JAR" \
    --model-path "$model" \
    --dtype "$dtype" \
    --byteOrder "$byte_order" \
    --max-tokens "$max_tokens" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --pType "$ptype" \
    "$gpu_flag" \
    ${jfr_arg} \
    ${lora_play_arg} \
    ${api_port_arg} \
    ${health_flag} \
    ${verbose_flag}
}

# ---------------------------------------------------------------------------
# console — single-JVM in-process REPL (no forked nodes, fastest startup)
#           Use this for interactive sessions and everyday model experimentation.
# ---------------------------------------------------------------------------
cmd_local() {
  local model="${MODEL_PATH:-}"
  local dtype="${DTYPE:-FLOAT16}"
  local byte_order="${BYTE_ORDER:-BE}"
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.6}"
  local heap="${HEAP:-4g}"
  local top_k="${TOP_K:-20}"
  local top_p="${TOP_P:-0.95}"
  local nodes="${NODES:-3}"
  local verbose="false"
  local jfr_duration=""
  local lora_play="${LORA_PLAY_PATH:-}"
  local health="false"
  local health_port="${HEALTH_PORT:-8081}"
  local api_port="${API_PORT:-}"
  local use_gpu="true"
  if [[ -n "${USE_GPU:-}" ]]; then
    case "${USE_GPU}" in
      false|0|no|NO) use_gpu="false" ;;
      *)             use_gpu="true"  ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)       model="$2";        shift 2 ;;
      --dtype)            dtype="$2";        shift 2 ;;
      --byteOrder | --byteorder | --byte-order) byte_order="${2^^}"; shift 2 ;;
      --max-tokens)       max_tokens="$2";   shift 2 ;;
      --temperature)      temperature="$2";  shift 2 ;;
      --top-k)            top_k="$2";        shift 2 ;;
      --top-p)            top_p="$2";        shift 2 ;;
      --heap)             heap="$2";         shift 2 ;;
      --nodes)            nodes="$2";        shift 2 ;;
      --jfr)              jfr_duration="$2"; shift 2 ;;
      --lora-play)        lora_play="$2";    shift 2 ;;
      --float16 | --fp16) dtype="FLOAT16";   shift   ;;
      --float32)          dtype="FLOAT32";   shift   ;;
      --int8)             dtype="INT8";      shift   ;;
      --gpu)              use_gpu="true";    shift   ;;
      --cpu)              use_gpu="false";   shift   ;;
      --health)           health="true";     shift   ;;
      --health-port)      health_port="$2";  shift 2 ;;
      --api-port)         api_port="$2";     shift 2 ;;
      --verbose | -v)     verbose="true";    shift   ;;
      --help)
        echo ""
        echo "  Usage: $0 local --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 local [flags]"
        echo ""
        echo "  Runs all transformer nodes in-process in a single JVM — no forking,"
        echo "  no gRPC sockets. Fastest startup. Use this for everyday experimentation."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH          Path to a GGUF model file"
        echo "                               (or set MODEL_PATH env var)"
        echo ""
        echo "  Activation dtype:"
        echo "    --dtype FLOAT32|FLOAT16|INT8  (default FLOAT16)"
        echo "    --float16 / --fp16"
        echo "    --float32"
        echo "    --int8"
        echo ""
        echo "  Generation:"
        echo "    --max-tokens N             (default 200)"
        echo "    --temperature F            (default 0.6)"
        echo "    --top-k N                  top-K sampling cutoff     (default 20, 0=disabled)"
        echo "    --top-p F                  top-p nucleus sampling    (default 0.95, 0=disabled)"
        echo ""
        echo "  Pipeline:"
        echo "    --nodes N                  number of in-process shards  (default 3)"
        echo "    --api-port N               start local REST API server on port N"
        echo "                               (includes OpenAI-compatible /v1/chat/completions)"
        echo ""
        echo "  Backend:"
        echo "    --gpu                      use GPU when available (default)"
        echo "    --cpu                      use CPU only"
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE                e.g. 4g 8g 16g               (default 4g)"
        echo "    --jfr DURATION             Enable Java Flight Recording for DURATION"
        echo "                               e.g. 5m 30s 1h — writes juno-<timestamp>.jfr"
        echo ""
        echo "  Logging:"
        echo "    --verbose / -v"
        echo ""
        echo "  LoRA:"
        echo "    --lora-play PATH           Apply a .lora file at inference (read-only, no training)"
        echo "                               (or set LORA_PLAY_PATH env var)"
        echo ""
        exit 0 ;;
      *) err "Unknown local flag: $1.  Run: $0 local --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 local --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 local"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$JUNO_PLAYER_JAR" "juno-player"
  check_java_version

  info "Starting local in-process REPL  (dtype=${dtype}  byteOrder=${byte_order}  max_tokens=${max_tokens}  temperature=${temperature}  nodes=${nodes}  heap=${heap}  gpu=${use_gpu}  os=${OS})"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON"
  echo ""

  local verbose_flag=""
  [[ "$verbose" == "true" ]] && verbose_flag="--verbose"

  local gpu_flag="--gpu"
  [[ "$use_gpu" == "false" ]] && gpu_flag="--cpu"
  prepend_cuda_bin_to_path_if_gpu "$use_gpu"

  # In local mode, ConsoleMain manages JFR programmatically via jdk.jfr.Recording.
  # Forward --jfr DURATION as a ConsoleMain argument rather than a JVM flag.
  local jfr_arg=""
  [[ -n "$jfr_duration" ]] && jfr_arg="--jfr $jfr_duration" && \
    warn "JFR enabled — duration=${jfr_duration}  (programmatic recording, metrics auto-printed on exit)"

  local lora_play_arg=""
  [[ -n "$lora_play" ]] && { lora_play_arg="--lora-play $lora_play"; warn "LoRA inference overlay: ${lora_play}"; }
  local health_flag=""
  if [[ "$health" == "true" ]]; then
    health_flag="--health"
    warn "Health sidecar enabled on :${health_port} — dashboard at http://localhost:${health_port}/"
  fi
  local api_port_arg=""
  [[ -n "$api_port" ]] && api_port_arg="--api-port $api_port"

  # shellcheck disable=SC2086
  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    "-Djuno.byteOrder=${byte_order}" \
    -jar "$JUNO_PLAYER_JAR" \
    --model-path "$model" \
    --dtype "$dtype" \
    --byteOrder "$byte_order" \
    --max-tokens "$max_tokens" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --nodes "$nodes" \
    --local \
    "$gpu_flag" \
    ${jfr_arg} \
    ${lora_play_arg} \
    ${api_port_arg} \
    ${health_flag} \
    ${verbose_flag}
}

# ---------------------------------------------------------------------------
# lora — LoRA fine-tuning REPL (single in-process JVM, adapter kept separate)
#        Use this to fine-tune a model locally and chat with the result.
#        Adapter is persisted as a separate .lora file — base GGUF untouched.
# ---------------------------------------------------------------------------
cmd_lora() {
  local model="${MODEL_PATH:-}"
  local lora_path="${LORA_PATH:-}"
  local lora_rank="${LORA_RANK:-8}"
  local lora_alpha="${LORA_ALPHA:-}"           # default = rank (set below)
  local lora_lr="${LORA_LR:-0.0001}"
  local lora_steps="${LORA_STEPS:-50}"
  local lora_steps_qa="${LORA_STEPS_QA:-10}"
  local lora_early_stop="${LORA_EARLY_STOP:-0.25}"
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.6}"
  local top_k="${TOP_K:-20}"
  local top_p="${TOP_P:-0.95}"
  local heap="${HEAP:-4g}"
  local verbose="false"
  local jfr_duration=""
  local health="false"
  local health_port="${HEALTH_PORT:-8081}"
  local use_gpu="true"
  if [[ -n "${USE_GPU:-}" ]]; then
    case "${USE_GPU}" in
      false|0|no|NO) use_gpu="false" ;;
      *)             use_gpu="true"  ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)   model="$2";       shift 2 ;;
      --lora-path)    lora_path="$2";   shift 2 ;;
      --lora-rank)    lora_rank="$2";   shift 2 ;;
      --lora-alpha)   lora_alpha="$2";  shift 2 ;;
      --lora-lr)      lora_lr="$2";     shift 2 ;;
      --lora-steps)   lora_steps="$2";    shift 2 ;;
      --lora-steps-qa) lora_steps_qa="$2"; shift 2 ;;
      --lora-early-stop) lora_early_stop="$2"; shift 2 ;;
      --max-tokens)   max_tokens="$2";  shift 2 ;;
      --temperature)  temperature="$2"; shift 2 ;;
      --top-k)        top_k="$2";       shift 2 ;;
      --top-p)        top_p="$2";       shift 2 ;;
      --heap)         heap="$2";        shift 2 ;;
      # --pType is accepted but ignored: lora always runs single in-process node
      --pType | --ptype) shift 2 ;;
      --jfr)          jfr_duration="$2"; shift 2 ;;
      --health)       health="true";     shift   ;;
      --health-port)  health_port="$2";  shift 2 ;;
      --gpu)          use_gpu="true";    shift   ;;
      --cpu)          use_gpu="false";   shift   ;;
      --verbose | -v) verbose="true";   shift   ;;
      --help)
        echo ""
        echo "  Usage: $0 lora --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 lora [flags]"
        echo ""
        echo "  Runs a LoRA fine-tuning REPL in a single in-process JVM."
        echo "  Adapter weights are saved to a separate .lora file."
        echo "  The base GGUF is never modified."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH       Path to a GGUF model file"
        echo "                            (or set MODEL_PATH env var)"
        echo ""
        echo "  LoRA adapter:"
        echo "    --lora-path PATH        Checkpoint file (default: <model>.lora)"
        echo "                            Loaded automatically if it exists."
        echo "    --lora-rank N           Low-rank bottleneck dimension (default: 8)"
        echo "                            4=minimal  8=standard  16=expressive"
        echo "    --lora-alpha F          Scaling alpha, default = rank (scale = alpha/rank)"
        echo "    --lora-lr F             Adam learning rate (default: 1e-4)"
        echo "    --lora-steps N          Gradient steps per /train command (default: 50)"
        echo ""
        echo "  Generation (used for chat inference):"
        echo "    --max-tokens N          (default 200)"
        echo "    --temperature F         (default 0.6)"
        echo "    --top-k N               (default 20)"
        echo "    --top-p F               (default 0.95)"
        echo ""
        echo "  Backend:"
        echo "    --gpu                   use GPU when available (default)"
        echo "    --cpu                   use CPU only"
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE             e.g. 4g 8g 16g  (default 4g)"
        echo "                            LoRA loads the full model in one JVM."
        echo "                            Tip: use at least 2× the model file size."
        echo ""
        echo "  Profiling:"
        echo "    --jfr DURATION          Java Flight Recording e.g. 30s 5m 1h"
        echo "                            Writes juno-<timestamp>.jfr on exit."
        echo "                            Open in JDK Mission Control, search for"
        echo "                            juno.LoraTrainStep to see per-step breakdown."
        echo ""
        echo "  Logging:"
        echo "    --verbose / -v"
        echo ""
        echo "  REPL commands once inside:"
        echo "    /train <text>           Fine-tune on inline text"
        echo "    /train-file <path>      Fine-tune on a text file (auto-chunked)"
        echo "    /save                   Save adapter to --lora-path"
        echo "    /reset                  Reinitialise adapters (clears training)"
        echo "    /status                 Show adapter info and training stats"
        echo "    /merge-hint             How to bake weights into a new GGUF"
        echo "    Regular input           Chat with the current adapter applied"
        echo ""
        echo "  Environment overrides:"
        echo "    MODEL_PATH  LORA_PATH  LORA_RANK  LORA_ALPHA  LORA_LR  LORA_STEPS"
        echo "    MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  USE_GPU"
        echo ""
        echo "  Examples:"
        echo "    $0 lora --model-path /models/tinyllama.gguf"
        echo "    $0 lora --model-path /models/tinyllama.gguf --lora-rank 16 --heap 8g"
        echo "    $0 lora --model-path /models/tinyllama.gguf --lora-path ./my.lora"
        echo "    MODEL_PATH=/models/tiny.gguf LORA_RANK=4 $0 lora"
        echo ""
        exit 0 ;;
      *) err "Unknown lora flag: $1.  Run: $0 lora --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 lora --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 lora"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$JUNO_PLAYER_JAR" "juno-player"
  check_java_version

  # Default alpha = rank when not explicitly set
  [[ -n "$lora_alpha" ]] || lora_alpha="$lora_rank"

  info "Starting LoRA fine-tuning REPL  (rank=${lora_rank}  alpha=${lora_alpha}  lr=${lora_lr}  steps=${lora_steps}  heap=${heap}  gpu=${use_gpu}  os=${OS})"
  [[ -n "$lora_path" ]] && info "Adapter file: ${lora_path}"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON"
  echo ""

  local verbose_flag=""
  [[ "$verbose" == "true" ]] && verbose_flag="--verbose"

  local gpu_flag="--gpu"
  [[ "$use_gpu" == "false" ]] && gpu_flag="--cpu"
  prepend_cuda_bin_to_path_if_gpu "$use_gpu"

  local lora_path_flag=""
  [[ -n "$lora_path" ]] && lora_path_flag="--lora-path $lora_path"

  local health_flag=""
  if [[ "$health" == "true" ]]; then
    health_flag="--health"
    warn "Health sidecar enabled on :${health_port} — dashboard at http://localhost:${health_port}/"
  fi

  local jfr_flag=""
  if [[ -n "$jfr_duration" ]]; then
    local model_name model_stem
    model_name="$(basename "$model")"
    model_stem="${model_name%.*}"
    local jfr_file="juno-${model_stem}-$(date +%Y%m%d-%H%M%S).jfr"
    jfr_flag="-XX:StartFlightRecording=duration=${jfr_duration},filename=${jfr_file},settings=profile,dumponexit=true"
    warn "JFR enabled — duration=${jfr_duration}  output=${jfr_file}"
    warn "After exit: open ${jfr_file} in JDK Mission Control → Event Browser → juno.LoraTrainStep"
  fi

  # shellcheck disable=SC2086
  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    ${jfr_flag:+"$jfr_flag"} \
    -jar "$JUNO_PLAYER_JAR" \
    --model-path "$model" \
    --lora \
    --lora-rank  "$lora_rank" \
    --lora-alpha "$lora_alpha" \
    --lora-lr    "$lora_lr" \
    --lora-steps "$lora_steps" \
    --lora-steps-qa "$lora_steps_qa" \
    --lora-early-stop "$lora_early_stop" \
    --max-tokens  "$max_tokens" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    "$gpu_flag" \
    ${lora_path_flag} \
    ${health_flag} \
    ${verbose_flag}
}

# ---------------------------------------------------------------------------
# live — ModelLiveRunner: 8 real-model smoke checks, exits 0/1
#        Use this as a quick regression check after any code change.
# ---------------------------------------------------------------------------
cmd_test() {
  local model="${MODEL_PATH:-}"
  local heap="${HEAP:-4g}"
  local ptype="${PTYPE:-all}"
  local jfr_duration=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)       model="$2";        shift 2 ;;
      --heap)             heap="$2";         shift 2 ;;
      --pType | --ptype)  ptype="$2";        shift 2 ;;
      --jfr)              jfr_duration="$2"; shift 2 ;;
      --help)
        echo ""
        echo "  Usage: $0 test --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 test [flags]"
        echo "     or: $0 test /path/to/model.gguf"
        echo ""
        echo "  Runs ModelLiveRunner — 8 automated real-model checks:"
        echo "    Pipeline-parallel (tests 1-6):"
        echo "    1. hello greeting coherence"
        echo "    2. no raw SentencePiece ▁ markers in output"
        echo "    3. question response is non-empty"
        echo "    4. greedy sampling is deterministic (temperature=0)"
        echo "    5. multi-turn conversation accumulates context (>20 prompt tokens)"
        echo "    6. FLOAT16 pipeline produces non-empty output"
        echo "    Tensor-parallel (tests 7-8):"
        echo "    7. tensor-parallel generation via AllReduce (3-node cluster)"
        echo "    8. tensor-parallel greedy determinism"
        echo ""
        echo "  Exits 0 if all 8 pass, 1 if any fail."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH  or  MODEL_PATH env var  or  positional arg"
        echo ""
        echo "  Parallelism:"
        echo "    --pType pipeline|tensor|all  filter which cluster tests to run"
        echo "                                 (default: all — runs both suites)"
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE        e.g. 4g 8g 16g  (default 4g)"
        echo "    --jfr DURATION     Enable Java Flight Recording for DURATION"
        echo "                       e.g. 5m 30s 1h — writes juno-<timestamp>.jfr"
        echo ""
        exit 0 ;;
      # positional model path
      *)
        if [[ -z "$model" && -f "$1" ]]; then
          model="$1"; shift
        else
          err "Unknown test flag: $1.  Run: $0 test --help"
        fi ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: MODEL_PATH=/path/to/model.gguf $0 test\n     or: $0 test /path/to/model.gguf\n     or: $0 test --model-path /path/to/model.gguf"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$LIVE_JAR" "juno-master"
  check_java_version

  info "Running ModelLiveRunner  (model=$(basename "$model")  pType=${ptype}  heap=${heap}  os=${OS})"
  echo ""

  local jfr_flag=""
  if [[ -n "$jfr_duration" ]]; then
    local model_name model_stem
    model_name="$(basename "$model")"
    model_stem="${model_name%.*}"
    local jfr_file="juno-${model_stem}-$(date +%Y%m%d-%H%M%S).jfr"
    jfr_flag="-XX:StartFlightRecording=duration=${jfr_duration},filename=${jfr_file},settings=profile,dumponexit=true"
    warn "JFR enabled — duration=${jfr_duration}  output=${jfr_file}"
  fi

  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    "-DpType=${ptype}" \
    "-Djuno.node.heap=${heap}" \
    ${jfr_flag:+"$jfr_flag"} \
    -jar "$LIVE_JAR" \
    "$model"
}

# ── Health server ─────────────────────────────────────────────────────────────
# health — standalone health-monitor HTTP server (no model required)
# Accepts node health probes via POST /health/probe and exposes a cluster
# overview via GET /health and per-node circuit states via GET /health/circuits.
cmd_health() {
  local port="${HEALTH_PORT:-8081}"
  local stale_ms="${HEALTH_STALE_MS:-15000}"
  local warn="${HEALTH_WARN:-0.90}"
  local critical="${HEALTH_CRITICAL:-0.98}"
  local heap="${HEAP:-512m}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)       port="$2";     shift 2 ;;
      --stale-ms)   stale_ms="$2"; shift 2 ;;
      --warn)       warn="$2";     shift 2 ;;
      --critical)   critical="$2"; shift 2 ;;
      --heap)       heap="$2";     shift 2 ;;
      --help)
        echo ""
        echo "  Usage: $0 health [flags]"
        echo ""
        echo "  Starts the Juno health-monitor HTTP server."
        echo "  No model file required.  Nodes push probes; the coordinator polls."
        echo ""
        echo "  API:"
        echo "    POST /health/probe           Accept a NodeHealth snapshot"
        echo "    GET  /health                 Cluster overview (status + all nodes)"
        echo "    GET  /health/nodes/{nodeId}  Single-node detail"
        echo "    GET  /health/circuits        Per-node circuit-breaker states"
        echo ""
        echo "  Flags:"
        echo "    --port N          HTTP listen port                (default: 8081)"
        echo "    --stale-ms N      ms before a node is stale       (default: 15000)"
        echo "    --warn F          VRAM warning threshold 0-1      (default: 0.90)"
        echo "    --critical F      VRAM critical threshold 0-1     (default: 0.98)"
        echo "    --heap SIZE       JVM heap e.g. 256m 512m 1g      (default: 512m)"
        echo ""
        echo "  Environment variable equivalents:"
        echo "    HEALTH_PORT  HEALTH_STALE_MS  HEALTH_WARN  HEALTH_CRITICAL  HEAP"
        echo ""
        echo "  Examples:"
        echo "    $0 health"
        echo "    $0 health --port 9090 --stale-ms 30000"
        echo "    HEALTH_PORT=9090 $0 health"
        echo ""
        exit 0 ;;
      *) err "Unknown health flag: $1.  Run: $0 health --help" ;;
    esac
  done

  require_jar "$HEALTH_JAR" "juno-health"
  check_java_version

  info "Starting Juno health server  (port=${port}  stale_ms=${stale_ms}  warn=${warn}  critical=${critical}  heap=${heap}  os=${OS})"
  echo ""

  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms64m "-Xmx${heap}" \
    -jar "$HEALTH_JAR" \
    --port     "$port" \
    --stale-ms "$stale_ms" \
    --warn     "$warn" \
    --critical "$critical"
}

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  echo ""
  echo -e "${CYAN}juno runtime launcher${NC}  (no Maven — uses pre-built jars)"
  echo -e "  OS detected: ${DIM}${OS}${NC}"
  echo -e "  Java:        ${DIM}${JAVA}${NC}"
  echo -e "  juno-player jar:  ${DIM}${JUNO_PLAYER_JAR}${NC}"
  echo -e "  juno-master jar: ${DIM}${LIVE_JAR}${NC}"
  echo -e "  health jar:  ${DIM}${HEALTH_JAR}${NC}"
  echo ""
  echo "  Build jars first (one time):"
  echo "    mvn clean package -DskipTests"
``  echo ""
  echo -e "  ${GREEN}$0${NC} --model-path PATH           3-node cluster + REPL  ${DIM}(default, forked JVM nodes)${NC}"
  echo    "  $0 cluster --help                  all cluster flags  (cluster keyword still works)"
  echo ""
  echo -e "  ${GREEN}$0 local${NC} --model-path PATH      in-process REPL  (single JVM, fast startup)"
  echo    "  $0 local --help                    all local flags"
  echo -e "  ${GREEN}$0 lora${NC} --model-path PATH       LoRA fine-tuning REPL  (single JVM, adapter separate)"
  echo    "  $0 lora --help                     all lora flags + REPL command reference"
  echo ""
  echo -e "  ${GREEN}$0 test${NC} --model-path PATH       8 real-model smoke checks, exits 0/1"
  echo    "  $0 test /path/to/model.gguf        model as positional arg"
  echo    "  $0 test --pType tensor             tensor-parallel checks only"
  echo    "  $0 test --help                     all test flags"
  echo ""
  echo -e "  ${GREEN}$0 health${NC}                        standalone health-monitor HTTP server"
  echo    "  $0 health --port 9090              listen on a custom port (default 8081)"
  echo    "  $0 health --help                   all health flags + API reference"
  echo ""
  echo "  Flags common to default (cluster), local, and lora:"
  echo "    --pType pipeline|tensor        parallelism type         (default pipeline)"
  echo "    --dtype FLOAT32|FLOAT16|INT8   activation wire format   (default FLOAT16)"
  echo "    --byteOrder BE|LE              activation codec endianness (default BE)"
  echo "    --float16 / --fp16             shorthand"
  echo "    --float32                      lossless reference / debug"
  echo "    --int8                         maximum compression"
  echo "    --max-tokens N                 max tokens per response  (default 200)"
  echo "    --temperature F                sampling temperature      (default 0.6)"
  echo "    --top-k N                      top-K sampling cutoff     (default 20, 0=disabled)"
  echo "    --top-p F                      top-p nucleus sampling    (default 0.95, 0=disabled)"
  echo "    --heap SIZE                    JVM heap e.g. 4g 8g      (default 4g)"
  echo "    --jfr DURATION                 Java Flight Recording     e.g. 5m 30s 1h"
  echo "    --gpu                          use GPU when available (default)"
  echo "    --cpu                          use CPU only"
  echo "    --verbose / -v                 show gRPC / node logs"
  echo ""
  echo "  local only:"
  echo "    --nodes N                      in-process shard count   (default 3)"
  echo ""
  echo "  lora only:"
  echo "    --lora-path PATH               adapter checkpoint file  (default <model>.lora)"
  echo "    --lora-rank N                  low-rank dimension       (default 8)"
  echo "    --lora-alpha F                 alpha scaling            (default = rank)"
  echo "    --lora-lr F                    Adam learning rate       (default 1e-4)"
  echo "    --lora-steps N                 gradient steps/train cmd (default 50)"
  echo ""
  echo "  Environment overrides (equivalent to their flag counterparts):"
  echo "    MODEL_PATH  DTYPE  PTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  NODES  USE_GPU"
  echo "    LORA_PATH  LORA_RANK  LORA_ALPHA  LORA_LR  LORA_STEPS"
  echo ""
  echo "  Examples:"
  echo "    MODEL_PATH=/models/tiny.gguf $0               # default = cluster (pipeline)"
  echo "    MODEL_PATH=/models/tiny.gguf $0 --pType tensor # tensor-parallel cluster"
  echo "    MODEL_PATH=/models/tiny.gguf $0 --float32 --heap 8g --verbose"
  echo "    MODEL_PATH=/models/tiny.gguf $0 local --temperature 0.3 --max-tokens 512"
  echo "    MODEL_PATH=/models/tiny.gguf $0 local --nodes 1"
  echo "    MODEL_PATH=/models/tiny.gguf $0 lora"
  echo "    MODEL_PATH=/models/tiny.gguf $0 lora --lora-rank 16 --lora-steps 100 --heap 8g"
  echo "    MODEL_PATH=/models/tiny.gguf $0 lora --lora-path ./finetune.lora"
  echo "    MODEL_PATH=/models/tiny.gguf $0 test"
  echo "    $0 test /models/tiny.gguf --heap 8g"
  echo "    MODEL_PATH=/models/tiny.gguf $0 --health                    # cluster + health sidecar on :8081"
  echo "    MODEL_PATH=/models/tiny.gguf $0 local --health --health-port 9090"
  echo "    $0 health --port 8081                                        # standalone health server"
  echo ""
  echo "  Custom Java:"
  echo "    JAVA_HOME=/path/to/jdk $0 cluster --model-path /models/tiny.gguf"
  echo ""
}

# ---------------------------------------------------------------------------
# merge — bake a .lora adapter into a new standalone GGUF
# ---------------------------------------------------------------------------
cmd_merge() {
  local model="${MODEL_PATH:-}"
  local lora=""
  local output=""
  local heap="${HEAP:-4g}"
  local use_gpu="${JUNO_USE_GPU:-false}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path) model="$2";  shift 2 ;;
      --lora-path)  lora="$2";   shift 2 ;;
      --output)     output="$2"; shift 2 ;;
      --heap)       heap="$2";   shift 2 ;;
      --help|-h)
        echo ""
        echo "  Usage: $0 merge --model-path /path/to/model.gguf [options]"
        echo ""
        echo "  Options:"
        echo "    --model-path PATH    Source GGUF or llamafile (required)"
        echo "    --lora-path PATH     Trained .lora checkpoint (default: <model>.lora)"
        echo "    --output PATH        Output GGUF path (default: <model>-merged.gguf)"
        echo "    --heap SIZE          JVM heap, e.g. 4g (default: 4g)"
        echo ""
        echo "  Example:"
        echo "    $0 merge --model-path /models/tinyllama.gguf"
        echo "    $0 merge --model-path /models/tinyllama.gguf --lora-path ./my.lora --output ./merged.gguf"
        echo ""
        return 0
        ;;
      *) err "Unknown merge flag: $1.  Run: $0 merge --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 merge --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 merge"

  shopt -s nullglob
  local candidates=( "$DIR/juno-player/target/"juno-player-*-shaded.jar "$DIR/juno-player/target/juno-player.jar" )
  shopt -u nullglob
  local juno_player_jar=""
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] || continue
    juno_player_jar="$f"
    break
  done
  [[ -n "$juno_player_jar" ]] || err "juno-player jar not found — build first with: mvn clean package -DskipTests"

  prepend_cuda_bin_to_path_if_gpu "$use_gpu"

  info "Starting LoRA merge  (heap=${heap})"

  "$JAVA" -Xmx${heap} \
    -cp "$juno_player_jar" \
    cab.ml.juno.player.LoraMergeMain \
    ${model:+--model-path "$model"} \
    ${lora:+--lora-path  "$lora"} \
    ${output:+--output   "$output"}
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in
  local)   cmd_local   "$@" ;;
  lora)    cmd_lora    "$@" ;;
  merge)   cmd_merge   "$@" ;;
  test)    cmd_test    "$@" ;;
  health)  cmd_health  "$@" ;;
  cluster) cmd_cluster "$@" ;;
  *)       cmd_cluster ${CMD:+"$CMD"} "$@" ;;
esac