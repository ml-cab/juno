#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# hyper-stack-4j — runtime launcher  (no Maven required)
# Uses pre-built shade jars from target/.  Build first with:
#   mvn clean package -DskipTests   or   ./hyper.sh build
#
# Requires: JDK 21+
# Runs on:  Linux · macOS · Windows (Git Bash / WSL)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLAYER_JAR="$DIR/player/target/player.jar"
LIVE_JAR="$DIR/integration/target/integration.jar"

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
    err "$label jar not found: $jar\n  Build first: mvn clean package -DskipTests\n           or: ./hyper.sh build"
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
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.6}"
  local top_k="${TOP_K:-20}"
  local top_p="${TOP_P:-0.95}"
  local heap="${HEAP:-4g}"
  local verbose="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)       model="$2";       shift 2 ;;
      --dtype)            dtype="$2";       shift 2 ;;
      --max-tokens)       max_tokens="$2";  shift 2 ;;
      --temperature)      temperature="$2"; shift 2 ;;
      --top-k)            top_k="$2";       shift 2 ;;
      --top-p)            top_p="$2";       shift 2 ;;
      --heap)             heap="$2";        shift 2 ;;
      --float16 | --fp16) dtype="FLOAT16";  shift   ;;
      --float32)          dtype="FLOAT32";  shift   ;;
      --int8)             dtype="INT8";     shift   ;;
      --verbose | -v)     verbose="true";   shift   ;;
      --help)
        echo ""
        echo "  Usage: $0 cluster --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 cluster [flags]"
        echo ""
        echo "  Starts a 3-node cluster (one forked JVM per node) and an interactive"
        echo "  REPL. Each node serves gRPC on localhost:19092-19094. Use this for"
        echo "  real GPU deployments and pipeline-parallel inference."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH          Path to a GGUF model file"
        echo "                               (or set MODEL_PATH env var)"
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
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE                JVM heap  e.g. 4g 8g 16g  (default 4g)"
        echo ""
        echo "  Logging:"
        echo "    --verbose / -v             show gRPC and node logs"
        echo ""
        exit 0 ;;
      *) err "Unknown cluster flag: $1.  Run: $0 cluster --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 cluster --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 cluster"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$PLAYER_JAR" "player"
  check_java_version

  warn "Starting 3-node cluster  (dtype=${dtype}  max_tokens=${max_tokens}  temperature=${temperature}  heap=${heap}  os=${OS})"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON"
  warn "Ctrl-C to stop all nodes and exit"
  echo ""

  local verbose_flag=""
  [[ "$verbose" == "true" ]] && verbose_flag="--verbose"

  # shellcheck disable=SC2086
  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    -jar "$PLAYER_JAR" \
    --model-path "$model" \
    --dtype "$dtype" \
    --max-tokens "$max_tokens" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    ${verbose_flag}
}

# ---------------------------------------------------------------------------
# console — single-JVM in-process REPL (no forked nodes, fastest startup)
#           Use this for interactive sessions and everyday model experimentation.
# ---------------------------------------------------------------------------
cmd_console() {
  local model="${MODEL_PATH:-}"
  local dtype="${DTYPE:-FLOAT16}"
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.6}"
  local heap="${HEAP:-4g}"
  local top_k="${TOP_K:-20}"
  local top_p="${TOP_P:-0.95}"
  local nodes="${NODES:-3}"
  local verbose="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)       model="$2";       shift 2 ;;
      --dtype)            dtype="$2";       shift 2 ;;
      --max-tokens)       max_tokens="$2";  shift 2 ;;
      --temperature)      temperature="$2"; shift 2 ;;
      --top-k)            top_k="$2";       shift 2 ;;
      --top-p)            top_p="$2";       shift 2 ;;
      --heap)             heap="$2";        shift 2 ;;
      --nodes)            nodes="$2";       shift 2 ;;
      --float16 | --fp16) dtype="FLOAT16";  shift   ;;
      --float32)          dtype="FLOAT32";  shift   ;;
      --int8)             dtype="INT8";     shift   ;;
      --verbose | -v)     verbose="true";   shift   ;;
      --help)
        echo ""
        echo "  Usage: $0 console --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 console [flags]"
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
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE                e.g. 4g 8g 16g               (default 4g)"
        echo ""
        echo "  Logging:"
        echo "    --verbose / -v"
        echo ""
        exit 0 ;;
      *) err "Unknown console flag: $1.  Run: $0 console --help" ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: $0 console --model-path /path/to/model.gguf\n     or: MODEL_PATH=/path/to/model.gguf $0 console"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$PLAYER_JAR" "player"
  check_java_version

  info "Starting local in-process console  (dtype=${dtype}  max_tokens=${max_tokens}  temperature=${temperature}  nodes=${nodes}  heap=${heap}  os=${OS})"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON"
  echo ""

  local verbose_flag=""
  [[ "$verbose" == "true" ]] && verbose_flag="--verbose"

  # shellcheck disable=SC2086
  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    -jar "$PLAYER_JAR" \
    --model-path "$model" \
    --dtype "$dtype" \
    --max-tokens "$max_tokens" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --nodes "$nodes" \
    --local \
    ${verbose_flag}
}

# ---------------------------------------------------------------------------
# live — ModelLiveRunner: 6 real-model smoke checks, exits 0/1
#        Use this as a quick regression check after any code change.
# ---------------------------------------------------------------------------
cmd_live() {
  local model="${MODEL_PATH:-}"
  local heap="${HEAP:-4g}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)    model="$2"; shift 2 ;;
      --heap)          heap="$2";  shift 2 ;;
      --help)
        echo ""
        echo "  Usage: $0 live --model-path /path/to/model.gguf [flags]"
        echo "     or: MODEL_PATH=/path/to/model.gguf $0 live [flags]"
        echo "     or: $0 live /path/to/model.gguf"
        echo ""
        echo "  Runs ModelLiveRunner — 6 automated real-model checks:"
        echo "    1. hello greeting coherence (>= 3 tokens, >= 2 greeting words)"
        echo "    2. no raw SentencePiece ▁ markers in output"
        echo "    3. question response is non-empty"
        echo "    4. greedy sampling is deterministic (temperature=0)"
        echo "    5. multi-turn conversation accumulates context (>20 prompt tokens)"
        echo "    6. FLOAT16 first token matches FLOAT32 first token"
        echo ""
        echo "  Exits 0 if all 6 pass, 1 if any fail."
        echo ""
        echo "  Required:"
        echo "    --model-path PATH  or  MODEL_PATH env var  or  positional arg"
        echo ""
        echo "  JVM:"
        echo "    --heap SIZE        e.g. 4g 8g 16g  (default 4g)"
        echo ""
        exit 0 ;;
      # positional model path
      *)
        if [[ -z "$model" && -f "$1" ]]; then
          model="$1"; shift
        else
          err "Unknown live flag: $1.  Run: $0 live --help"
        fi ;;
    esac
  done

  [[ -n "$model" ]] || err "Model path is required.\n  Usage: MODEL_PATH=/path/to/model.gguf $0 live\n     or: $0 live /path/to/model.gguf\n     or: $0 live --model-path /path/to/model.gguf"
  [[ -f "$model" ]] || err "Model file not found: $model"

  require_jar "$LIVE_JAR" "integration"
  check_java_version

  info "Running ModelLiveRunner  (model=$(basename "$model")  heap=${heap}  os=${OS})"
  echo ""

  exec "$JAVA" \
    "${JVM_BASE[@]}" \
    -Xms512m "-Xmx${heap}" \
    -jar "$LIVE_JAR" \
    "$model"
}

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  echo ""
  echo -e "${CYAN}hyper-stack-4j runtime launcher${NC}  (no Maven — uses pre-built jars)"
  echo -e "  OS detected: ${DIM}${OS}${NC}"
  echo -e "  Java:        ${DIM}${JAVA}${NC}"
  echo -e "  player jar:  ${DIM}${PLAYER_JAR}${NC}"
  echo -e "  live jar:    ${DIM}${LIVE_JAR}${NC}"
  echo ""
  echo "  Build jars first (one time):"
  echo "    mvn clean package -DskipTests"
  echo "    ./hyper.sh build"
  echo ""
  echo -e "  ${GREEN}$0 cluster${NC} --model-path PATH    3-node cluster + REPL  (forked JVM nodes, GPU)"
  echo    "  $0 cluster --help                  all cluster flags"
  echo ""
  echo -e "  ${GREEN}$0 console${NC} --model-path PATH    in-process REPL  (single JVM, fast startup)"
  echo    "  $0 console --help                  all console flags"
  echo ""
  echo -e "  ${GREEN}$0 live${NC} --model-path PATH       6 real-model smoke checks, exits 0/1"
  echo    "  $0 live /path/to/model.gguf        model as positional arg"
  echo    "  $0 live --help                     all live flags"
  echo ""
  echo "  Flags common to cluster and console:"
  echo "    --dtype FLOAT32|FLOAT16|INT8   activation wire format   (default FLOAT16)"
  echo "    --float16 / --fp16             shorthand"
  echo "    --float32                      lossless reference / debug"
  echo "    --int8                         maximum compression"
  echo "    --max-tokens N                 max tokens per response  (default 200)"
  echo "    --temperature F                sampling temperature      (default 0.6)"
  echo "    --top-k N                      top-K sampling cutoff     (default 20, 0=disabled)"
  echo "    --top-p F                      top-p nucleus sampling    (default 0.95, 0=disabled)"
  echo "    --heap SIZE                    JVM heap e.g. 4g 8g      (default 4g)"
  echo "    --verbose / -v                 show gRPC / node logs"
  echo ""
  echo "  console only:"
  echo "    --nodes N                      in-process shard count   (default 3)"
  echo ""
  echo "  Environment overrides (equivalent to their flag counterparts):"
  echo "    MODEL_PATH  DTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  NODES"
  echo ""
  echo "  Examples:"
  echo "    MODEL_PATH=/models/tiny.gguf $0 cluster"
  echo "    MODEL_PATH=/models/tiny.gguf $0 cluster --float32 --heap 8g --verbose"
  echo "    MODEL_PATH=/models/tiny.gguf $0 console --temperature 0.3 --max-tokens 512"
  echo "    MODEL_PATH=/models/tiny.gguf $0 console --nodes 1"
  echo "    MODEL_PATH=/models/tiny.gguf $0 live"
  echo "    $0 live /models/tiny.gguf --heap 8g"
  echo ""
  echo "  Custom Java:"
  echo "    JAVA_HOME=/path/to/jdk $0 cluster --model-path /models/tiny.gguf"
  echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in
  cluster) cmd_cluster "$@" ;;
  console) cmd_console "$@" ;;
  live)    cmd_live    "$@" ;;
  *)       usage ;;
esac