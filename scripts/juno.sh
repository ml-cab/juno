#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# juno — dev runner
# Requires: JDK 21+  ·  Maven 3.8+
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MVN="${MVN:-mvn}"     # override: MVN=/path/to/mvn ./hyper.sh test
PORT="${PORT:-8080}"  # coordinator REST port

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'
info() { echo -e "${CYAN}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✔ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✖ $*${NC}" >&2; exit 1; }

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
  command -v java    >/dev/null 2>&1 || err "JDK 21+ not found. Install from https://adoptium.net"
  command -v "$MVN"  >/dev/null 2>&1 || err "Maven not found.   brew install maven  or  sudo apt install maven"
  JAVA_VER=$(java -version 2>&1 | awk -F'"' '/version/{print $2}' | cut -d. -f1)
  [[ "${JAVA_VER:-0}" -ge 21 ]] || err "JDK 21+ required (found: $JAVA_VER)"
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_cluster() {
  # ── Parse optional flags ──────────────────────────────────────────────────
  local dtype="${DTYPE:-FLOAT16}"      # FLOAT16 default: halves gRPC payload vs FLOAT32
  local max_tokens="${MAX_TOKENS:-200}"
  local temperature="${TEMPERATURE:-0.7}"
  local heap="${HEAP:-4g}"             # -Xmx override: 4g default (2g was too tight for real models)
  local verbose="false"
  local skip_build="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dtype)        dtype="$2";        shift 2 ;;
      --max-tokens)   max_tokens="$2";   shift 2 ;;
      --temperature)  temperature="$2";  shift 2 ;;
      --heap)         heap="$2";         shift 2 ;;
      --float16|--fp16) dtype="FLOAT16"; shift ;;
      --float32)        dtype="FLOAT32"; shift ;;
      --int8)           dtype="INT8";    shift ;;
      --verbose|-v)     verbose="true";  shift ;;
      --skip-build|-B)  skip_build="true"; shift ;;
      --help)
        echo ""
        echo "  Usage: $0 cluster [flags]"
        echo ""
        echo "  --dtype FLOAT32|FLOAT16|INT8   activation wire format (default FLOAT16)"
        echo "  --float16 / --fp16             shorthand for --dtype FLOAT16 (default)"
        echo "  --float32                      shorthand for --dtype FLOAT32 (debug/reference)"
        echo "  --int8                         shorthand for --dtype INT8"
        echo "  --max-tokens N                 max generated tokens    (default 200)"
        echo "  --temperature F                sampling temperature     (default 0.7)"
        echo "  --heap SIZE                    JVM heap size e.g. 4g 8g (default 4g)"
        echo "  --skip-build / -B              skip mvn compile (use last build)"
        echo "  --verbose / -v                 show full gRPC + Maven logs"
        echo ""
        exit 0 ;;
      *) err "Unknown cluster flag: $1.  Run: $0 cluster --help" ;;
    esac
  done

  # ── Build ─────────────────────────────────────────────────────────────────
  if [[ "$skip_build" == "true" ]]; then
    warn "Skipping build (-B / --skip-build)"
  elif [[ "$verbose" == "true" ]]; then
    info "Building player module (compile)..."
    "$MVN" compile -pl player -am --no-transfer-progress
    ok "Build OK"
  else
    # Hide OpenAPI generator banner and Maven noise — only show on error
    build_log=$(mktemp)
    if ! "$MVN" compile -pl player -am -q --no-transfer-progress \
         > "$build_log" 2>&1; then
      cat "$build_log"
      rm -f "$build_log"
      err "Build failed"
    fi
    rm -f "$build_log"
    ok "Build OK"
  fi
  echo ""

  warn "Starting 3-node cluster  (dtype=${dtype}  max_tokens=${max_tokens}  temperature=${temperature}  heap=${heap})"
  [[ "$verbose" == "true" ]] && warn "Verbose mode ON — gRPC logs visible"
  warn "Ctrl-C to stop all nodes and exit"
  echo ""

  local hyper_verbose_flag=""
  [[ "$verbose" == "true" ]] && hyper_verbose_flag="-DHYPER_VERBOSE=true"

  # On Windows, mvn.cmd routes arguments through cmd.exe which interprets
  # %classpath as an environment variable and strips it — Maven never sees its
  # own %classpath expansion token, so the JVM launches with an empty classpath
  # and throws NoClassDefFoundError immediately.
  # Fix: escape as %%classpath so cmd.exe outputs a literal %classpath for Maven.
  # On Linux/macOS bash does not interpret %, so %classpath works unmodified.
  local cp_token="%classpath"
  case "$OSTYPE" in msys*|cygwin*|win32*) cp_token="%%classpath" ;; esac

  exec "$MVN" exec:exec \
    -pl player \
    -Dexec.executable=java \
    -Dexec.classpathScope=compile \
    -Dexec.args="--enable-preview --enable-native-access=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED -Xms512m -Xmx${heap} -XX:+UseG1GC -XX:+AlwaysPreTouch ${hyper_verbose_flag} -DDTYPE=${dtype} -DMAX_TOKENS=${max_tokens} -DTEMPERATURE=${temperature} -classpath ${cp_token} cab.ml.juno.player.ConsoleMain" \
    --no-transfer-progress \
    -q
}

cmd_unit() {
  info "Running all unit tests (skipping integration)..."
  cd "$DIR"
  "$MVN" test -DskipITs --no-transfer-progress
  ok "All unit tests passed"
}

cmd_unit_module() {
  local mod="${1:-}"
  [[ -n "$mod" ]] || err "Usage: $0 test-module <module>  e.g. coordinator  health  kvcache"
  [[ -d "$DIR/$mod" ]] || err "Module not found: $mod"
  info "Running tests for module: $mod"
  cd "$DIR"
  "$MVN" test -pl "$mod" -am --no-transfer-progress
  ok "$mod tests passed"
}

cmd_unit_fault() {
  info "Running fault tolerance tests only..."
  cd "$DIR"
  "$MVN" test -pl coordinator -am \
    -Dtest="FaultTolerantPipelineTest,HealthReactorTest,RetryPolicyTest" \
    --no-transfer-progress
  ok "Fault tolerance tests passed"
}

cmd_integration() {
  warn "Integration tests fork 3 JVM processes — takes ~30s"
  info "Running full integration suite..."
  cd "$DIR"
  "$MVN" verify -pl integration --no-transfer-progress
  ok "Integration tests passed"
}

cmd_integration_fast() {
  info "Running fast in-process cluster test only (~250ms)..."
  cd "$DIR"
  "$MVN" verify -pl integration -Dit.test=InProcessClusterIT --no-transfer-progress
  ok "InProcessClusterIT passed"
}

cmd_test() {
  # ── Resolve model path ────────────────────────────────────────────────────
  local model="${MODEL_PATH:-}"
  local heap="${HEAP:-4g}"
  local skip_build="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --heap)          heap="$2";          shift 2 ;;
      --skip-build|-B) skip_build="true";  shift ;;
      --help)
        echo ""
        echo "  Usage: MODEL_PATH=/path/to/model.gguf $0 test [flags]"
        echo ""
        echo "  Runs ModelLiveRunner — 6 real-model smoke checks with coloured output."
        echo "  Exits 0 if all pass, 1 if any fail."
        echo ""
        echo "  MODEL_PATH   path to a GGUF file (required — env var or first arg)"
        echo "  --heap SIZE  JVM heap size e.g. 4g 8g (default 4g)"
        echo "  --skip-build / -B  skip mvn compile (use last build)"
        echo ""
        exit 0 ;;
      # allow model path as positional arg too
      *)
        if [[ -z "$model" && -f "$1" ]]; then
          model="$1"; shift
        else
          err "Unknown test flag: $1.  Run: $0 test --help"
        fi ;;
    esac
  done

  if [[ -z "$model" ]]; then
    err "MODEL_PATH is not set.\n  Usage: MODEL_PATH=/path/to/model.gguf $0 test\n     or: $0 test /path/to/model.gguf"
  fi
  if [[ ! -f "$model" ]]; then
    err "Model file not found: $model"
  fi

  # ── Build ─────────────────────────────────────────────────────────────────
  cd "$DIR"
  if [[ "$skip_build" == "true" ]]; then
    warn "Skipping build (-B / --skip-build)"
  else
    build_log=$(mktemp)
    if ! "$MVN" compile -pl integration -am -q --no-transfer-progress \
         > "$build_log" 2>&1; then
      cat "$build_log"
      rm -f "$build_log"
      err "Build failed"
    fi
    rm -f "$build_log"
    ok "Build OK"
  fi
  echo ""

  info "Running ModelLiveRunner  (model=$(basename "$model")  heap=${heap})"
  echo ""

  local cp_token="%classpath"
  case "$OSTYPE" in msys*|cygwin*|win32*) cp_token="%%classpath" ;; esac

  exec "$MVN" exec:exec \
    -pl integration \
    -Dexec.executable=java \
    -Dexec.classpathScope=compile \
    -Dexec.args="--enable-preview --enable-native-access=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED -Xms512m -Xmx${heap} -XX:+UseG1GC -XX:+AlwaysPreTouch -classpath ${cp_token} cab.ml.juno.integration.ModelLiveRunner ${model}" \
    --no-transfer-progress \
    -q
}

cmd_build() {
  info "Compiling all modules (no tests)..."
  cd "$DIR"
  "$MVN" compile -DskipTests --no-transfer-progress
  ok "Build succeeded"
}

cmd_clean() {
  info "Cleaning all build artefacts..."
  cd "$DIR"
  "$MVN" clean --no-transfer-progress
  ok "Clean done"
}

cmd_verify() {
  info "Full verify: compile + unit tests + integration tests..."
  cd "$DIR"
  "$MVN" verify --no-transfer-progress
  ok "Full verify passed"
}

cmd_health_demo() {
  info "Fault tolerance wiring overview"
  cat <<'JAVA'

── What HealthReactor wires together ────────────────────────────────────────────

  FaultTolerantPipeline pipeline = new FaultTolerantPipeline(
      List.of(
          NodePipeline.of("node-1", grpcPipelineToNode1),
          NodePipeline.of("node-2", grpcPipelineToNode2)
      ),
      RetryPolicy.once()  // 2 attempts, 50ms backoff
  );

  HealthReactor reactor = new HealthReactor(
      HealthThresholds.defaults(),   // warning=90%, critical=98%, stale=15s
      pipeline,
      scheduler                      // shut down if all nodes go unavailable
  );

  // Hazelcast IMap listener (each GPU node publishes NodeHealth every 5s):
  nodeHealthMap.addEntryListener(event -> {
      reactor.onHealthProbe(event.getValue());   // <── one call drives everything
  }, true);

── Run the tests to see every scenario live ─────────────────────────────────────

  ./hyper.sh test-fault

JAVA
}

cmd_curl_demo() {
  info "Example REST API commands (coordinator must be running on :${PORT})"
  cat <<CURL

  # Blocking inference
  curl -s -X POST http://localhost:${PORT}/v1/inference \\
       -H 'Content-Type: application/json' \\
       -d '{
             "modelId":  "tinyllama",
             "messages": [{"role": "user", "content": "Hello!"}]
           }' | python3 -m json.tool

  # Streaming inference — tokens arrive as Server-Sent Events
  curl -sN -X POST http://localhost:${PORT}/v1/inference/stream \\
       -H 'Content-Type: application/json' \\
       -d '{
             "modelId":  "tinyllama",
             "messages": [{"role": "user", "content": "Count to five"}]
           }'

  # List registered models
  curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool

  # Cluster health
  curl -s http://localhost:${PORT}/v1/cluster/health | python3 -m json.tool

CURL
}

cmd_watch() {
  local mod="${1:-coordinator}"
  info "Watching $mod for changes (Ctrl-C to stop)..."
  command -v fswatch >/dev/null 2>&1 || err "fswatch not found — brew install fswatch"
  fswatch -o "$DIR/$mod/src" | while read -r; do
    echo ""
    info "Change detected — re-running $mod tests..."
    "$MVN" test -pl "$mod" -am -q --no-transfer-progress 2>&1 | tail -20
    ok "Done at $(date +%H:%M:%S)"
  done
}

usage() {
  echo ""
  echo -e "${CYAN}juno dev runner${NC}"
  echo ""
  echo -e "  ${GREEN}$0${NC}                           Boot 3-node cluster + interactive console  ${DIM}(default)${NC}"
  echo    "  $0 --dtype FLOAT32            Use FLOAT32 activations (debug; default is FLOAT16)"
  echo    "  $0 --dtype INT8               Use INT8  compressed activations"
  echo    "  $0 --max-tokens 512           Override max generation tokens (default 200)"
  echo    "  $0 --temperature 0.9          Override sampling temperature  (default 0.7)"
  echo    "  $0 --heap 8g                  Override JVM heap size         (default 4g)"
  echo    "  $0 --skip-build / -B          Skip mvn compile (use last build)"
  echo    "  $0 --verbose                  Show full gRPC + Maven logs"
  echo    "  $0 cluster --help             All cluster flags  (cluster keyword still works)"
  echo ""
  echo -e "  ${GREEN}$0 test${NC}                      Run ModelLiveRunner — 6 real-model smoke checks"
  echo    "  $0 test /path/to/model.gguf  Model path as positional arg (or set MODEL_PATH)"
  echo    "  $0 test --heap 8g            Override JVM heap size (default 4g)"
  echo    "  $0 test --skip-build / -B    Skip mvn compile (use last build)"
  echo    "  $0 test --help               All test flags"
  echo ""
  echo    "  $0 unit                   Unit tests — all modules, skip integration (~10s)"
  echo    "  $0 unit-module <mod>      Unit tests for one module  e.g. coordinator  health"
  echo    "  $0 unit-fault             Fault tolerance tests only"
  echo    "  $0 integration            Full integration suite — forks 3 JVMs (~30s)"
  echo    "  $0 integration-fast       InProcessClusterIT only (~250ms)"
  echo    "  $0 build                  Compile only, no tests"
  echo    "  $0 clean                  Remove all target/ directories"
  echo    "  $0 verify                 Full: compile + unit + integration"
  echo    "  $0 health-demo            Fault tolerance wiring walkthrough"
  echo    "  $0 curl-demo              Example REST API curl commands"
  echo    "  $0 watch [mod]            Auto-rerun tests on file changes (requires fswatch)"
  echo ""
  echo    "  Environment overrides:"
  echo    "    MVN=/path/to/mvn ./hyper.sh test"
  echo    "    PORT=9090 ./hyper.sh curl-demo"
  echo    "    DTYPE=FLOAT16 MAX_TOKENS=512 HEAP=8g ./hyper.sh cluster"
  echo    "    MODEL_PATH=/path/to/model.gguf ./hyper.sh test"
  echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
check_deps

CMD="${1:-}"
shift || true   # drop $1 so remaining args are available as $@

case "$CMD" in
  test)              cmd_test    "$@" ;;
  cluster)           cmd_cluster "$@" ;;
  unit)              cmd_unit ;;
  unit-module)       cmd_unit_module "${1:-}" ;;
  unit-fault)        cmd_unit_fault ;;
  integration)       cmd_integration ;;
  integration-fast)  cmd_integration_fast ;;
  build)             cmd_build ;;
  clean)             cmd_clean ;;
  verify)            cmd_verify ;;
  health-demo)       cmd_health_demo ;;
  curl-demo)         cmd_curl_demo ;;
  watch)             cmd_watch "${1:-coordinator}" ;;
  *)                 cmd_cluster ${CMD:+"$CMD"} "$@" ;;
esac