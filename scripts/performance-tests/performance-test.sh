#!/usr/bin/env bash
# performance-test.sh — AWS perf runner and matrix generator
#
# Each matrix cell (l1, l9, c1, c9) runs: deploy (--detach) → HTTP test → finish (JFR + teardown).
# Selection is read directly from scripts/performance-tests/matrix.tsv (no separate queue file).
#
#   ./scripts/performance-tests/performance-test.sh --foreground --all
#   ./scripts/performance-tests/performance-test.sh --foreground --row 2 --col l1
#   ./scripts/performance-tests/performance-test.sh --foreground --from 1 --to 2 --all
#   ./scripts/performance-tests/performance-test.sh --parse
#   ./scripts/performance-tests/performance-test.sh --status
#
# See docs/performance.md and scripts/performance-tests/scenarios.yaml.

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
SCENARIO="${ROOT}/test-scenario.txt"
MATRIX="${PERF_SCRIPTS}/matrix.tsv"
SCENARIOS="${PERF_SCRIPTS}/scenarios.yaml"
HTML="${ROOT}/docs/juno_test_matrix.html"
WORKDIR="${ROOT}/target/perf"
RUN_DIR="${WORKDIR}/runs"
NOHUP_LOG="${WORKDIR}/nohup.log"
SCREEN_NAME="${PERF_SCREEN_NAME:-juno-perf}"
SCREEN_SESSION_FILE="${WORKDIR}/screen.session"
PID_FILE="${WORKDIR}/worker.pid"
PARSED="${WORKDIR}/parsed.tsv"
MERGED="${WORKDIR}/matrix.tsv"
ROWS_JS="${WORKDIR}/rows.js"

MODE="run"
FOREGROUND=0
ROW_FILTER=""
COL_FILTER=""
ROW_FROM=""
ROW_TO=""
PERF_GIT_REF=""
PERF_LORA_PLAY=""
PERF_ALL=0
PERF_PENDING=0

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \?//'
    cat <<EOF

Usage:
  $(basename "$0")                   Start screen worker (pending cells, default)
  $(basename "$0") --foreground       Run worker in foreground
  $(basename "$0") --attach           Attach to running screen worker
  $(basename "$0") --parse             Parse test-scenario.txt, update HTML matrix
  $(basename "$0") --status            Show screen session and log tail
  $(basename "$0") --list              Print selected cells and exit

Selection (from matrix.tsv in this directory; combine as needed):
  --all                 Every applicable cell (non-NA), including done (D)
  --pending             Only pending (P) or suggested (A) cells [default with no filter]
  --row ID              Limit to matrix row id
  --col COL             Limit to column: l1, l9, c1, c9
  --from ID --to ID     Inclusive row id range

Examples:
  $(basename "$0") --foreground --all
  $(basename "$0") --foreground --row 2 --col l1
  $(basename "$0") --foreground --from 1 --to 2 --all
  $(basename "$0") --list --from 1 --to 2 --all

Options:
  --matrix FILE     Matrix TSV (default: scripts/performance-tests/matrix.tsv)
  --scenario FILE   Scenario log for --parse (default: test-scenario.txt)
  --html FILE       Output HTML (default: docs/juno_test_matrix.html)
  --git REF         Git branch, tag, or commit for juno-deploy.sh (default: main)
  --lora-play PATH  LoRA adapter for matrix rows with lo=on (default: models/...lora)
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

perf_screen_running() {
    screen -list 2>/dev/null | grep -qE "[[:space:]][0-9]+\\.${SCREEN_NAME}[[:space:]]"
}

perf_worker_args() {
    PERF_WORKER_ARGS=(--foreground)
    [[ "$PERF_ALL" -eq 1 ]] && PERF_WORKER_ARGS+=(--all)
    [[ "$PERF_PENDING" -eq 1 ]] && PERF_WORKER_ARGS+=(--pending)
    [[ -n "$ROW_FILTER" ]] && PERF_WORKER_ARGS+=(--row "$ROW_FILTER")
    [[ -n "$COL_FILTER" ]] && PERF_WORKER_ARGS+=(--col "$COL_FILTER")
    [[ -n "$ROW_FROM" ]] && PERF_WORKER_ARGS+=(--from "$ROW_FROM")
    [[ -n "$ROW_TO" ]] && PERF_WORKER_ARGS+=(--to "$ROW_TO")
    [[ -n "$PERF_GIT_REF" ]] && PERF_WORKER_ARGS+=(--git "$PERF_GIT_REF")
    [[ -n "$PERF_LORA_PLAY" ]] && PERF_WORKER_ARGS+=(--lora-play "$PERF_LORA_PLAY")
    [[ "$MATRIX" != "${PERF_SCRIPTS}/matrix.tsv" ]] && PERF_WORKER_ARGS+=(--matrix "$MATRIX")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker|--foreground) FOREGROUND=1; shift ;;
        --parse) MODE="parse"; shift ;;
        --status) MODE="status"; shift ;;
        --attach) MODE="attach"; shift ;;
        --row) ROW_FILTER="$2"; shift 2 ;;
        --col) COL_FILTER="$2"; shift 2 ;;
        --from) ROW_FROM="$2"; shift 2 ;;
        --to) ROW_TO="$2"; shift 2 ;;
        --all) PERF_ALL=1; shift ;;
        --pending) PERF_PENDING=1; shift ;;
        --list) MODE="list"; shift ;;
        --matrix) MATRIX="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --html) HTML="$2"; shift 2 ;;
        --git) PERF_GIT_REF="$2"; shift 2 ;;
        --lora-play) PERF_LORA_PLAY="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --queue) die "--queue removed; select cells from ${MATRIX} with --all, --row, --col, --from/--to" ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

DRY_RUN="${DRY_RUN:-0}"

# shellcheck source=/dev/null
source "${PERF_SCRIPTS}/perf-lib.sh"

list_run_commands() {
    grep -E '^\s*\./launcher\.sh juno-deploy\.sh setup' "$SCENARIO" \
        | sed 's/^[[:space:]]*//' \
        | sort -u
}

parse_and_merge() {
    log "parsing ${SCENARIO}..."
    awk -f "${PERF_SCRIPTS}/parse-scenarios.awk" "$SCENARIO" | sort -u > "$PARSED"
    local count
    count="$(wc -l < "$PARSED" | tr -d ' ')"
    log "extracted ${count} coordinator TPS measurements"
    [[ "$count" -gt 0 ]] || die "no metrics parsed from ${SCENARIO}"

    log "merging into matrix..."
    awk -f "${PERF_SCRIPTS}/merge-matrix.awk" -v matrix="$MATRIX" "$PARSED" > "$MERGED"
}

perf_select_mode() {
    if [[ "$PERF_ALL" -eq 1 && "$PERF_PENDING" -eq 1 ]]; then
        die "use only one of --all or --pending"
    fi
    if [[ "$PERF_ALL" -eq 1 ]]; then
        printf '%s' all
        return
    fi
    if [[ "$PERF_PENDING" -eq 1 ]]; then
        printf '%s' pending
        return
    fi
    if [[ -n "$ROW_FILTER" || -n "$ROW_FROM" || -n "$ROW_TO" || -n "$COL_FILTER" ]]; then
        printf '%s' all
        return
    fi
    printf '%s' pending
}

perf_select_matrix_cells() {
    local mode
    mode="$(perf_select_mode)"
    awk -f "${PERF_SCRIPTS}/select-matrix-cells.awk" \
        -v mode="$mode" \
        -v row="$ROW_FILTER" \
        -v col="$COL_FILTER" \
        -v from="$ROW_FROM" \
        -v to="$ROW_TO" \
        "$MATRIX"
}

perf_validate_selection_args() {
    if [[ -n "$ROW_FROM" && -n "$ROW_FILTER" ]]; then
        die "use --row or --from/--to, not both"
    fi
    if [[ -n "$ROW_FROM" && ! "$ROW_FROM" =~ ^[0-9]+$ ]]; then
        die "--from must be a numeric row id"
    fi
    if [[ -n "$ROW_TO" && ! "$ROW_TO" =~ ^[0-9]+$ ]]; then
        die "--to must be a numeric row id"
    fi
    if [[ -n "$ROW_FILTER" && ! "$ROW_FILTER" =~ ^[0-9]+$ ]]; then
        die "--row must be a numeric row id"
    fi
    if [[ -n "$ROW_FROM" && -n "$ROW_TO" && "$ROW_FROM" -gt "$ROW_TO" ]]; then
        die "--from must be <= --to"
    fi
    if [[ -n "$COL_FILTER" ]]; then
        case "$COL_FILTER" in
            l1|l9|c1|c9) ;;
            *) die "unknown column: ${COL_FILTER} (use l1, l9, c1, c9)" ;;
        esac
    fi
}

list_selected_cells() {
    perf_validate_selection_args
    [[ -f "$MATRIX" ]] || die "matrix file not found: $MATRIX"
    local mode line count=0
    mode="$(perf_select_mode)"
    log "selection from ${MATRIX} (mode=${mode})"
    while IFS=$'\t' read -r line; do
        [[ -z "$line" ]] && continue
        printf '%s\n' "$line"
        count=$((count + 1))
    done < <(perf_select_matrix_cells)
    log "${count} cell(s) selected"
}

worker_main() {
    require_cmd curl
    require_cmd jq
    require_cmd awk
    require_cmd sed
    require_cmd grep

    [[ -f "$MATRIX" ]] || die "matrix file not found: $MATRIX"
    [[ -x "${ROOT}/scripts/aws/launcher.sh" ]] || die "launcher not executable: scripts/aws/launcher.sh"

    perf_validate_selection_args
    perf_load_scenarios
    mkdir -p "$RUN_DIR"

    local -a selected=()
    local mode count row_id column
    mode="$(perf_select_mode)"
    mapfile -t selected < <(perf_select_matrix_cells)
    count="${#selected[@]}"

    if [[ "$count" -eq 0 ]]; then
        log "no cells selected from ${MATRIX} (mode=${mode})"
        log "use --all for every non-NA cell, or --row / --from/--to to target specific rows"
        exit 0
    fi

    log "selected ${count} cell(s) from ${MATRIX} (mode=${mode})"
    log "worker log: ${NOHUP_LOG}"
    log "run artifacts: ${RUN_DIR}"

    # Job control + TTY (screen or foreground) lets deploy SIGINT reach juno-deploy.sh.
    if [[ -t 0 || -n "${STY:-}" ]]; then
        set -m
    fi

    for line in "${selected[@]}"; do
        [[ -z "$line" ]] && continue
        IFS=$'\t' read -r row_id column <<< "$line"
        log "matrix cell: row=${row_id} column=${column} (deploy --detach → test → finish)"
        if ! perf_run_single_test "$row_id" "$column"; then
            warn "test failed: row=${row_id} column=${column} (continuing)"
        fi
        sleep 5
    done

    log "worker finished ${count} cell(s)"
    rm -f "$PID_FILE" "$SCREEN_SESSION_FILE"
}

launch_screen_worker() {
    require_cmd screen
    mkdir -p "$WORKDIR"

    if perf_screen_running; then
        die "worker already running in screen session ${SCREEN_NAME}; attach: screen -r ${SCREEN_NAME}"
    fi
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid="$(cat "$PID_FILE")"
        if kill -0 "$old_pid" 2>/dev/null; then
            die "worker already running (pid ${old_pid}); stop it first or use --foreground"
        fi
        rm -f "$PID_FILE"
    fi

    perf_worker_args
    local perf_script="${PERF_SCRIPTS}/performance-test.sh"
    local -a inner_cmd=("$perf_script" "${PERF_WORKER_ARGS[@]}")
    local screen_cmd
    screen_cmd="$(printf 'cd %q && exec ' "$ROOT")"
    screen_cmd+=$(printf '%q ' "${inner_cmd[@]}")
    screen_cmd+=$(printf '>> %q 2>&1' "$NOHUP_LOG")

    screen -dmS "$SCREEN_NAME" bash -lc "$screen_cmd"
    sleep 0.3
    if ! perf_screen_running; then
        die "failed to start screen session ${SCREEN_NAME}"
    fi

    echo "$SCREEN_NAME" > "$SCREEN_SESSION_FILE"
    log "started worker in screen session ${SCREEN_NAME}"
    log "attach:  screen -r ${SCREEN_NAME}"
    log "monitor: tail -f ${NOHUP_LOG}"
}

show_status() {
    if perf_screen_running; then
        log "worker running in screen session ${SCREEN_NAME}"
        log "attach: screen -r ${SCREEN_NAME}"
    elif [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            log "worker running pid ${pid} (no screen session)"
        else
            log "stale pid file (${pid} not running)"
        fi
    else
        log "no worker screen session (${SCREEN_NAME})"
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
        patch_html "$MERGED"
        log "done"
        ;;
    status)
        show_status
        ;;
    attach)
        require_cmd screen
        perf_screen_running || die "no running screen session ${SCREEN_NAME} (try --status)"
        exec screen -r "$SCREEN_NAME"
        ;;
    list)
        list_selected_cells
        ;;
    run)
        if [[ "$FOREGROUND" -eq 1 ]]; then
            worker_main
        else
            launch_screen_worker
        fi
        ;;
esac
