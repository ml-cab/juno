#!/usr/bin/env bash
# performance-test.sh — AWS perf runner and matrix generator
#
# No arguments: start detached worker via nohup. Each queue item (l1, l9, c1, c9)
# gets its own deploy → HTTP test → SIGINT on deploy monitor → wait JFR → teardown.
#
#   ./scripts/performance-test.sh              # nohup worker (pending matrix cells)
#   ./scripts/performance-test.sh --foreground # worker in foreground
#   ./scripts/performance-test.sh --parse      # parse test-scenario.txt -> HTML matrix
#   ./scripts/performance-test.sh --status     # tail nohup log / show worker pid
#
# See performance.md and perf/scenarios.yaml.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO="${ROOT}/test-scenario.txt"
MATRIX="${ROOT}/perf/matrix.tsv"
HTML="${ROOT}/docs/juno_test_matrix.html"
WORKDIR="${ROOT}/target/perf"
RUN_DIR="${WORKDIR}/runs"
NOHUP_LOG="${WORKDIR}/nohup.log"
PID_FILE="${WORKDIR}/worker.pid"
QUEUE_FILE="${WORKDIR}/queue.tsv"
PARSED="${WORKDIR}/parsed.tsv"
MERGED="${WORKDIR}/matrix.tsv"
ROWS_JS="${WORKDIR}/rows.js"

MODE="run"
FOREGROUND=0
ROW_FILTER=""
COL_FILTER=""
QUEUE_EXPLICIT=0

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \?//'
    cat <<EOF

Usage:
  $(basename "$0")                   Start nohup worker (default)
  $(basename "$0") --foreground       Run worker in foreground
  $(basename "$0") --parse             Parse test-scenario.txt, update HTML matrix
  $(basename "$0") --status            Show worker pid and log tail
  $(basename "$0") --row ID --col COL  Run one matrix cell (e.g. --row 1 --col l1)

Options:
  --matrix FILE     Matrix TSV (default: perf/matrix.tsv)
  --scenario FILE   Scenario log for --parse (default: test-scenario.txt)
  --html FILE       Output HTML (default: docs/juno_test_matrix.html)
  --queue FILE      Explicit queue TSV: row_id<TAB>column
  -n, --dry-run     Parse mode only: preview HTML rows
  -h, --help        This help
EOF
}

log() { printf '[perf] %s\n' "$*"; }
warn() { printf '[perf] warn: %s\n' "$*" >&2; }
die() { printf '[perf] error: %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker|--foreground) FOREGROUND=1; shift ;;
        --parse) MODE="parse"; shift ;;
        --status) MODE="status"; shift ;;
        --row) ROW_FILTER="$2"; shift 2 ;;
        --col) COL_FILTER="$2"; shift 2 ;;
        --matrix) MATRIX="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --html) HTML="$2"; shift 2 ;;
        --queue) QUEUE_FILE="$2"; QUEUE_EXPLICIT=1; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

DRY_RUN="${DRY_RUN:-0}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/perf-lib.sh"

list_run_commands() {
    grep -E '^\s*\./launcher\.sh juno-deploy\.sh setup' "$SCENARIO" \
        | sed 's/^[[:space:]]*//' \
        | sort -u
}

parse_and_merge() {
    log "parsing ${SCENARIO}..."
    awk -f "${ROOT}/scripts/parse-scenarios.awk" "$SCENARIO" | sort -u > "$PARSED"
    local count
    count="$(wc -l < "$PARSED" | tr -d ' ')"
    log "extracted ${count} coordinator TPS measurements"
    [[ "$count" -gt 0 ]] || die "no metrics parsed from ${SCENARIO}"

    log "merging into matrix..."
    awk -f "${ROOT}/scripts/merge-matrix.awk" -v matrix="$MATRIX" "$PARSED" > "$MERGED"
}

patch_html() {
    local generated tmp
    generated="$(date +%Y-%m-%d)"

    awk -f "${ROOT}/scripts/render-matrix-js.awk" "$MERGED" > "$ROWS_JS"

    tmp="$(mktemp)"
    awk -v gen="$generated" -v jsfile="$ROWS_JS" '
      BEGIN {
        while ((getline l < jsfile) > 0) js = js l "\n"
        close(jsfile)
      }
      /^const rows = \[/ { print js; skip = 1; next }
      skip && /^\];/ { skip = 0; next }
      skip { next }
      /generated [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/ {
        sub(/generated [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/, "generated " gen)
      }
      { print }
    ' "$HTML" > "$tmp"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "dry-run: would write ${HTML}"
        head -n 5 "$ROWS_JS" | sed 's/^/  /'
        rm -f "$tmp"
        return
    fi

    mv "$tmp" "$HTML"
    cp "$MERGED" "$MATRIX"
    log "updated ${HTML}"
    log "updated ${MATRIX}"
}

build_queue() {
    if [[ -n "$ROW_FILTER" && -n "$COL_FILTER" ]]; then
        printf '%s\t%s\n' "$ROW_FILTER" "$COL_FILTER" > "$QUEUE_FILE"
        return
    fi
    if [[ "$QUEUE_EXPLICIT" -eq 1 ]]; then
        [[ -f "$QUEUE_FILE" ]] || die "queue file not found: $QUEUE_FILE"
        return
    fi
    awk -f "${ROOT}/scripts/perf-queue.awk" "$MATRIX" > "$QUEUE_FILE"
}

worker_main() {
    require_cmd curl
    require_cmd jq
    require_cmd awk
    require_cmd sed
    require_cmd grep

    [[ -f "$MATRIX" ]] || die "matrix file not found: $MATRIX"
    [[ -x "${ROOT}/scripts/aws/launcher.sh" ]] || die "launcher not executable: scripts/aws/launcher.sh"

    perf_load_scenarios
    mkdir -p "$RUN_DIR"
    build_queue

    local count row_id column
    count="$(grep -cve '^[[:space:]]*$' "$QUEUE_FILE" 2>/dev/null || echo 0)"
    if [[ "$count" -eq 0 ]]; then
        log "queue empty — no P/A cells in ${MATRIX}"
        log "use --row ID --col COL to run a single test"
        exit 0
    fi

    log "queue: ${count} test(s) -> ${QUEUE_FILE}"
    log "nohup log: ${NOHUP_LOG}"
    log "run artifacts: ${RUN_DIR}"

    while IFS=$'\t' read -r row_id column; do
        [[ -z "$row_id" ]] && continue
        log "queue item: row=${row_id} column=${column} (fresh deploy → test → SIGINT/JFR → teardown)"
        if ! perf_run_single_test "$row_id" "$column"; then
            warn "test failed: row=${row_id} column=${column} (continuing queue)"
        fi
        sleep 5
    done < "$QUEUE_FILE"

    log "worker finished ${count} queued test(s)"
    rm -f "$PID_FILE"
}

launch_nohup_worker() {
    mkdir -p "$WORKDIR"
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid="$(cat "$PID_FILE")"
        if kill -0 "$old_pid" 2>/dev/null; then
            die "worker already running (pid ${old_pid}); see ${NOHUP_LOG}"
        fi
        rm -f "$PID_FILE"
    fi

    local extra=()
    [[ -n "$ROW_FILTER" ]] && extra+=(--row "$ROW_FILTER")
    [[ -n "$COL_FILTER" ]] && extra+=(--col "$COL_FILTER")
    [[ -n "${QUEUE_FILE:-}" && "$QUEUE_FILE" != "${WORKDIR}/queue.tsv" ]] && extra+=(--queue "$QUEUE_FILE")

    nohup "$0" --foreground "${extra[@]}" >>"$NOHUP_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    log "started worker pid $(cat "$PID_FILE")"
    log "monitor: tail -f ${NOHUP_LOG}"
}

show_status() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            log "worker running pid ${pid}"
        else
            log "stale pid file (${pid} not running)"
        fi
    else
        log "no worker pid file"
    fi
    if [[ -f "$NOHUP_LOG" ]]; then
        log "last 20 lines of ${NOHUP_LOG}:"
        tail -n 20 "$NOHUP_LOG"
    else
        log "no log yet: ${NOHUP_LOG}"
    fi
}

case "$MODE" in
    parse)
        require_cmd awk
        require_cmd sed
        require_cmd grep
        require_cmd jq
        require_cmd date
        [[ -f "$SCENARIO" ]] || die "scenario file not found: $SCENARIO"
        [[ -f "$MATRIX" ]] || die "matrix file not found: $MATRIX"
        [[ -f "$HTML" ]] || die "html template not found: $HTML"
        mkdir -p "$WORKDIR"
        parse_and_merge
        patch_html
        log "done"
        ;;
    status)
        show_status
        ;;
    run)
        if [[ "$FOREGROUND" -eq 1 ]]; then
            worker_main
        else
            launch_nohup_worker
        fi
        ;;
esac
