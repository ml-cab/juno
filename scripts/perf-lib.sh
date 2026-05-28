# perf-lib.sh — helpers for scripts/performance-test.sh (source, do not execute)

perf_load_scenarios() {
    local scenarios="${ROOT}/perf/scenarios.yaml"
    local key val conv_n=0
    [[ -f "$scenarios" ]] || die "missing ${scenarios}"

    while IFS=$'\t' read -r key val; do
        [[ -z "$key" ]] && continue
        case "$key" in
            MODEL_ID) MODEL_ID="$val" ;;
            MODEL_URL) MODEL_URL="$val" ;;
            CPU_MAX_TOKENS) CPU_MAX_TOKENS="$val" ;;
            GPU_MAX_TOKENS) GPU_MAX_TOKENS="$val" ;;
            LONG_PROMPT) LONG_PROMPT="$val" ;;
            CONV_MSG)
                conv_n=$((conv_n + 1))
                printf -v "CONV_MSG${conv_n}" '%s' "$val"
                ;;
        esac
    done < <(awk -f "${ROOT}/scripts/read-scenarios.awk" "$scenarios")

    : "${MODEL_ID:?MODEL_ID missing from scenarios.yaml}"
    : "${MODEL_URL:?MODEL_URL missing from scenarios.yaml}"
    : "${CPU_MAX_TOKENS:=50}"
    : "${GPU_MAX_TOKENS:=200}"
    : "${LONG_PROMPT:?LONG_PROMPT missing from scenarios.yaml}"
    : "${CONV_MSG1:?CONV_MSG1 missing from scenarios.yaml}"
    : "${CONV_MSG2:?CONV_MSG2 missing from scenarios.yaml}"
    : "${CONV_MSG3:?CONV_MSG3 missing from scenarios.yaml}"
    LORA_ADAPTER="${ROOT}/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.lora"
}

perf_matrix_row() {
    local id="$1"
    awk -F'\t' -v id="$id" '$1 == id { print; exit }' "$MATRIX"
}

perf_build_deploy_args() {
    local id="$1"
    local line hw pt n co dt bo lo inst dtype_arg
    line="$(perf_matrix_row "$id")"
    [[ -n "$line" ]] || die "matrix row not found: id=${id}"

    IFS=$'\t' read -r _rid hw pt n co dt bo lo _rest <<< "$line"

    case "$hw" in
        cpu) inst="m7i-flex.large" ;;
        gpu) inst="g4dn.2xlarge" ;;
        *) die "unknown hw: ${hw}" ;;
    esac

    DEPLOY_ARGS=(
        setup
        --instance-type "$inst"
        --node-count "$n"
        --model-url "$MODEL_URL"
        --ptype "$pt"
        --jfr 1h
    )

    [[ "$co" == "separate" ]] && DEPLOY_ARGS+=(--coordinator separate)

    case "$dt" in
        FP16) ;;
        FP32) DEPLOY_ARGS+=(--dtype FLOAT32) ;;
        INT8) DEPLOY_ARGS+=(--dtype INT8) ;;
        *) die "unknown dtype: ${dt}" ;;
    esac

    [[ "$bo" == "LE" ]] && DEPLOY_ARGS+=(--byteOrder LE)

    if [[ "$lo" == "on" ]]; then
        [[ -f "$LORA_ADAPTER" ]] || die "lora adapter missing: ${LORA_ADAPTER}"
        DEPLOY_ARGS+=(--lora-play "$LORA_ADAPTER")
    fi

    if [[ -n "${PERF_GIT_REF:-}" ]]; then
        DEPLOY_ARGS+=(--git "$PERF_GIT_REF")
    fi

    if [[ "${PERF_USE_DEPLOY_DETACH:-1}" == "1" ]]; then
        DEPLOY_ARGS+=(--detach --no-browser)
    fi

    PERF_ROW_HW="$hw"
    PERF_ROW_ID="$id"
}

perf_wait_for_log_line() {
    local logfile="$1" pattern="$2" timeout="$3"
    local start now
    start="$(date +%s)"
    while true; do
        if grep -q "$pattern" "$logfile" 2>/dev/null; then
            return 0
        fi
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            return 1
        fi
        if [[ -n "${DEPLOY_PID:-}" ]] && ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
            log "deploy process exited before ${pattern}"
            return 1
        fi
        sleep 5
    done
}

PERF_DEPLOY_JFR_WAIT="${PERF_DEPLOY_JFR_WAIT:-600}"
PERF_DEPLOY_EXIT_WAIT="${PERF_DEPLOY_EXIT_WAIT:-600}"
PERF_PRE_SIGINT_WAIT="${PERF_PRE_SIGINT_WAIT:-90}"
PERF_PRE_SIGINT_SETTLE_SEC="${PERF_PRE_SIGINT_SETTLE_SEC:-3}"
# 1 = setup --detach --no-browser, then juno-deploy finish (no SIGINT on monitor loop)
PERF_USE_DEPLOY_DETACH="${PERF_USE_DEPLOY_DETACH:-1}"

# juno-deploy.sh: log "  JFR Cluster Metrics" → "[juno]   JFR Cluster Metrics" after ANSI strip.
PERF_JFR_METRICS_BANNER_RE='\[juno\][[:space:]]+JFR Cluster Metrics'

# Extract JSON printed by juno-deploy after "JFR Cluster Metrics" (deploy monitor stdout).
perf_extract_jfr_json_from_deploy_log() {
    local logfile="$1" outfile="$2"
    perf_strip_ansi <"$logfile" | awk '
        /JFR Cluster Metrics/ { want=1; next }
        want && !json && /^=+/ { next }
        want && !json && /^\{/ {
            json=1
            depth=0
        }
        json {
            line=$0
            print line
            for (i = 1; i <= length(line); i++) {
                c = substr(line, i, 1)
                if (c == "{") depth++
                if (c == "}") {
                    depth--
                    if (depth == 0) exit
                }
            }
        }
    ' >"$outfile"
}

perf_deploy_log_has_jfr_banner() {
    local logfile="$1"
    [[ -f "$logfile" ]] || return 1
    perf_strip_ansi <"$logfile" | grep -qE "$PERF_JFR_METRICS_BANNER_RE"
}

perf_deploy_log_has_jfr_json() {
    local logfile="$1" tmp
    [[ -f "$logfile" ]] || return 1
    perf_deploy_log_has_jfr_banner "$logfile" || return 1
    tmp="$(mktemp)"
    perf_extract_jfr_json_from_deploy_log "$logfile" "$tmp"
    if [[ -s "$tmp" ]] && command -v jq >/dev/null 2>&1; then
        jq -e '.models | length > 0' "$tmp" >/dev/null 2>&1
    elif [[ -s "$tmp" ]]; then
        grep -q '"models"' "$tmp"
    else
        rm -f "$tmp"
        return 1
    fi
    local rc=$?
    rm -f "$tmp"
    return "$rc"
}

perf_count_log_matches() {
    local logfile="$1" pattern="$2"
    grep -c "$pattern" "$logfile" 2>/dev/null || true
}

# Point 4: after HTTP, wait for a fresh monitor refresh line (sleep window, not mid-SSH probe) before SIGINT.
perf_wait_for_monitor_idle_before_sigint() {
    local logfile="$1" timeout="${2:-$PERF_PRE_SIGINT_WAIT}"
    local pattern='Refreshing every 20s'
    local start now baseline current

    [[ -f "$logfile" ]] || return 1
    baseline="$(perf_count_log_matches "$logfile" "$pattern")"
    start="$(date +%s)"
    log "waiting for monitor idle (${pattern}) before SIGINT (up to ${timeout}s)…"

    while true; do
        current="$(perf_count_log_matches "$logfile" "$pattern")"
        if (( current > baseline )); then
            log "monitor refresh seen — settling ${PERF_PRE_SIGINT_SETTLE_SEC}s before SIGINT"
            sleep "$PERF_PRE_SIGINT_SETTLE_SEC"
            return 0
        fi
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            warn "timed out waiting for ${pattern} in ${logfile} — sending SIGINT anyway"
            return 1
        fi
        if [[ -n "${DEPLOY_PID:-}" ]] && ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
            warn "deploy exited while waiting for monitor idle"
            return 1
        fi
        sleep 2
    done
}

# Wait for deploy log banner "[juno]   JFR Cluster Metrics" and valid JSON after SIGINT.
perf_wait_for_deploy_jfr() {
    local logfile="$1" timeout="${2:-$PERF_DEPLOY_JFR_WAIT}"
    local start now grace=0
    local -a fail_patterns=(
        "MetricsMain failed"
        "No JFR files collected"
        "Not a valid Flight Recorder file"
    )

    start="$(date +%s)"
    while true; do
        if perf_deploy_log_has_jfr_json "$logfile"; then
            log "saw [juno]   JFR Cluster Metrics banner and metrics JSON in ${logfile}"
            return 0
        fi

        now="$(date +%s)"
        if (( now - start >= timeout )); then
            warn "timed out waiting for [juno]   JFR Cluster Metrics in ${logfile}"
            return 1
        fi

        for p in "${fail_patterns[@]}"; do
            if grep -q "$p" "$logfile" 2>/dev/null; then
                warn "JFR gather failed (${p}) — see ${logfile}"
                return 1
            fi
        done

        if [[ -n "${DEPLOY_PID:-}" ]] && ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
            while (( grace < 60 )); do
                if perf_deploy_log_has_jfr_json "$logfile"; then
                    log "saw [juno]   JFR Cluster Metrics banner and metrics JSON in ${logfile}"
                    return 0
                fi
                sleep 2
                grace=$((grace + 2))
            done
            for p in "${fail_patterns[@]}"; do
                if grep -q "$p" "$logfile" 2>/dev/null; then
                    warn "JFR gather failed (${p}) — see ${logfile}"
                    return 1
                fi
            done
            if grep -q "Caught exit signal" "$logfile" 2>/dev/null; then
                warn "deploy monitor exited without [juno]   JFR Cluster Metrics in ${logfile}"
            else
                warn "deploy exited before JFR gather started"
            fi
            return 1
        fi

        if grep -q "Gathering JFR metrics" "$logfile" 2>/dev/null \
            && ! perf_deploy_log_has_jfr_banner "$logfile"; then
            log "deploy monitor gathering JFR (waiting for [juno]   JFR Cluster Metrics in ${logfile})…"
        fi
        sleep 5
    done
}

perf_collect_metrics_json() {
    local row_id="$1" column="$2"
    local dest="${RUN_DIR}/metrics-${row_id}-${column}.json"

    if [[ ! -f "${DEPLOY_LOG:-}" ]]; then
        warn "deploy log missing: ${DEPLOY_LOG:-?}"
        return 1
    fi

    perf_extract_jfr_json_from_deploy_log "$DEPLOY_LOG" "$dest"
    if [[ -s "$dest" ]] && command -v jq >/dev/null 2>&1 && jq -e '.models | length > 0' "$dest" >/dev/null 2>&1; then
        log "metrics JSON: ${dest} (from deploy monitor console output)"
        return 0
    fi
    if [[ -s "$dest" ]] && grep -q '"models"' "$dest"; then
        log "metrics JSON: ${dest} (from deploy monitor console output)"
        return 0
    fi

    rm -f "$dest"
    warn "no [juno]   JFR Cluster Metrics JSON in ${DEPLOY_LOG}"
    return 1
}

perf_log_deploy_jfr_tail() {
    local logfile="$1"
    [[ -f "$logfile" ]] || return 0
    log "deploy log tail (JFR / exit):"
    perf_strip_ansi <"$logfile" | awk '
        /Caught exit signal|Gathering JFR|JFR Cluster Metrics|MetricsMain|Flight Recorder|No JFR files/ { show=1 }
        show { print }
    ' | tail -n 25
}

perf_strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

perf_extract_console_url() {
    local logfile="$1"
    local clean url

    clean="$(perf_strip_ansi < "$logfile")"

    url="$(printf '%s\n' "$clean" | grep -E 'Console\s+:' | tail -1 | grep -oE 'http://[^[:space:]()]+' | head -1)"
    if [[ -z "$url" ]]; then
        url="$(printf '%s\n' "$clean" | grep -E 'Web console\s+:' | tail -1 | grep -oE 'http://[^[:space:]()]+' | head -1)"
    fi
    if [[ -z "$url" ]]; then
        url="$(printf '%s\n' "$clean" | grep -oE 'http://[0-9.]+:8080' | tail -1)"
    fi
    [[ -n "$url" ]] || return 1
    printf '%s\n' "$url"
}

perf_wait_for_api() {
    local base="$1" timeout="$2"
    local start now health_url="${base%/}/v1/cluster/health"
    start="$(date +%s)"
    log "waiting for API health: ${health_url}"
    while true; do
        if curl -sf "$health_url" >/dev/null 2>&1; then
            log "API healthy"
            return 0
        fi
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            warn "API health check timed out after ${timeout}s: ${health_url}"
            return 1
        fi
        if (( (now - start) % 30 < 5 )); then
            log "still waiting for API ($(( now - start ))s)…"
        fi
        sleep 5
    done
}

perf_max_tokens_for_hw() {
    case "$1" in
        cpu) echo "$CPU_MAX_TOKENS" ;;
        gpu) echo "$GPU_MAX_TOKENS" ;;
        *) echo "$CPU_MAX_TOKENS" ;;
    esac
}

perf_column_workload() {
    case "$1" in
        l1|l9) echo "long" ;;
        c1|c9) echo "conv" ;;
        *) die "unknown column: $1" ;;
    esac
}

perf_column_sessions() {
    case "$1" in
        l1|c1) echo 1 ;;
        l9|c9) echo 9 ;;
        *) echo 1 ;;
    esac
}

perf_curl_timeout() {
    local max_tokens="$1" sessions="$2"
    # CPU long/s9 can run many minutes; cap at 2h per request batch.
    local est=$(( max_tokens * sessions * 3 + 120 ))
    (( est < 600 )) && est=600
    (( est > 7200 )) && est=7200
    echo "$est"
}

perf_wait_pids() {
    local pid status=0
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    return "$status"
}

perf_curl_chat() {
    local base="$1" max_tokens="$2" prompt="$3" outfile="$4"
    local url="${base%/}/v1/chat/completions"
    local timeout="${5:-3600}"
    local errfile="${outfile%.json}.err"
    local http_code

    http_code="$(curl -sS --max-time "$timeout" -o "$outfile" -w '%{http_code}' -X POST "$url" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg model "$MODEL_ID" --arg p "$prompt" --argjson m "$max_tokens" \
            '{model:$model,messages:[{role:"user",content:$p}],max_tokens:$m}')" \
        2>"$errfile")" || http_code="000"

    if [[ "$http_code" != "200" ]]; then
        warn "chat completion failed: http=${http_code} file=${outfile} ($(tr '\n' ' ' < "$errfile" 2>/dev/null | head -c 200))"
        return 1
    fi
    return 0
}

perf_curl_chat_history() {
    local base="$1" max_tokens="$2" history_json="$3" outfile="$4"
    local url="${base%/}/v1/chat/completions"
    local timeout="${5:-3600}"
    local errfile="${outfile%.json}.err"
    local http_code

    http_code="$(curl -sS --max-time "$timeout" -o "$outfile" -w '%{http_code}' -X POST "$url" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg model "$MODEL_ID" --argjson hist "$history_json" --argjson m "$max_tokens" \
            '{model:$model,messages:$hist,max_tokens:$m}')" \
        2>"$errfile")" || http_code="000"

    if [[ "$http_code" != "200" ]]; then
        warn "chat completion failed: http=${http_code} file=${outfile} ($(tr '\n' ' ' < "$errfile" 2>/dev/null | head -c 200))"
        return 1
    fi
    return 0
}

perf_run_long_test() {
    local base="$1" max_tokens="$2" sessions="$3" outdir="$4"
    local i timeout pids=() status=0
    timeout="$(perf_curl_timeout "$max_tokens" "$sessions")"
    mkdir -p "$outdir"
    log "curl timeout ${timeout}s per request (sessions=${sessions}, max_tokens=${max_tokens})"
    if [[ "$sessions" -eq 1 ]]; then
        perf_curl_chat "$base" "$max_tokens" "$LONG_PROMPT" "${outdir}/long.json" "$timeout"
        return $?
    fi
    for i in $(seq 1 "$sessions"); do
        perf_curl_chat "$base" "$max_tokens" "$LONG_PROMPT" "${outdir}/long-${i}.json" "$timeout" &
        pids+=($!)
        log "started session ${i}/${sessions} (pid ${pids[-1]})"
    done
    perf_wait_pids "${pids[@]}" || status=1
    return "$status"
}

perf_run_conv_test() {
    local base="$1" max_tokens="$2" sessions="$3" outdir="$4"
    local i t1 t2 t3 hist timeout pids=() status=0
    timeout="$(perf_curl_timeout "$max_tokens" "$sessions")"
    mkdir -p "$outdir"
    log "curl timeout ${timeout}s per turn (sessions=${sessions}, max_tokens=${max_tokens})"

    run_one_conv() {
        local sid="$1"
        t1="${outdir}/conv-${sid}-t1.json"
        t2="${outdir}/conv-${sid}-t2.json"
        t3="${outdir}/conv-${sid}-t3.json"

        perf_curl_chat "$base" "$max_tokens" "$CONV_MSG1" "$t1" "$timeout" || return 1

        hist="$(jq -nc \
            --arg u1 "$CONV_MSG1" \
            --arg a1 "$(jq -r '.choices[0].message.content // ""' "$t1")" \
            --arg u2 "$CONV_MSG2" \
            '[{role:"user",content:$u1},{role:"assistant",content:$a1},{role:"user",content:$u2}]')"
        perf_curl_chat_history "$base" "$max_tokens" "$hist" "$t2" "$timeout" || return 1

        hist="$(jq -nc \
            --arg u1 "$CONV_MSG1" \
            --arg a1 "$(jq -r '.choices[0].message.content // ""' "$t1")" \
            --arg u2 "$CONV_MSG2" \
            --arg a2 "$(jq -r '.choices[0].message.content // ""' "$t2")" \
            --arg u3 "$CONV_MSG3" \
            '[{role:"user",content:$u1},{role:"assistant",content:$a1},{role:"user",content:$u2},{role:"assistant",content:$a2},{role:"user",content:$u3}]')"
        perf_curl_chat_history "$base" "$max_tokens" "$hist" "$t3" "$timeout" || return 1
    }

    if [[ "$sessions" -eq 1 ]]; then
        run_one_conv 1
        return $?
    fi
    for i in $(seq 1 "$sessions"); do
        run_one_conv "$i" &
        pids+=($!)
        log "started conv session ${i}/${sessions} (pid ${pids[-1]})"
    done
    perf_wait_pids "${pids[@]}" || status=1
    return "$status"
}

# Resolve the bash process running juno-deploy.sh (not the launcher wrapper).
perf_resolve_deploy_pid() {
    local pid="${DEPLOY_PID:-}" child
    [[ -n "$pid" ]] || return 1

    while child="$(pgrep -P "$pid" -f 'juno-deploy\.sh' 2>/dev/null | head -1)"; do
        [[ -n "$child" ]] || break
        pid="$child"
    done

    if ps -p "$pid" -o args= 2>/dev/null | grep -q 'juno-deploy\.sh'; then
        printf '%s\n' "$pid"
        return 0
    fi

    printf '%s\n' "${DEPLOY_PID}"
}

# Send a signal to deploy monitor. Background bash ignores direct SIGINT while
# sleeping unless the whole process group is signaled (requires set -m at start).
perf_signal_deploy() {
    local sig="$1"
    local pid pgid child worker_pgid

    [[ -n "${DEPLOY_PID:-}" ]] || return 0
    pid="$(perf_resolve_deploy_pid)"
    worker_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
    pgid="$(ps -o pgid= -p "$DEPLOY_PID" 2>/dev/null | tr -d ' ')"

    if [[ -n "$pgid" && "$pgid" != "$worker_pgid" ]]; then
        kill "-$sig" "-$pgid" 2>/dev/null || true
    elif [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
        kill "-$sig" "-$pgid" 2>/dev/null || true
    else
        kill "-$sig" "$pid" 2>/dev/null || true
        kill "-$sig" "${DEPLOY_PID}" 2>/dev/null || true
    fi

    while read -r child; do
        [[ -z "$child" ]] && continue
        kill "-$sig" "$child" 2>/dev/null || true
    done < <(pgrep -P "$pid" 2>/dev/null || true)
}

perf_start_deploy() {
    local launcher="${ROOT}/scripts/aws/launcher.sh"
    local -a deploy_cmd

    deploy_cmd=("$launcher" juno-deploy.sh "${DEPLOY_ARGS[@]}")
    log "deploy: ${deploy_cmd[*]}"

    # Own process group (requires monitor mode) so SIGINT reaches juno-deploy monitor.
    case $- in
        *m*) ;;
        *) [[ -t 0 || -n "${STY:-}" ]] && set -m ;;
    esac
    (
        cd "${ROOT}/scripts/aws"
        exec "${deploy_cmd[@]}"
    ) >"$DEPLOY_LOG" 2>&1 &
    DEPLOY_PID=$!
}

perf_wait_for_deploy_exit() {
    local timeout="${1:-300}"
    local start now
    [[ -n "${DEPLOY_PID:-}" ]] || return 0
    start="$(date +%s)"
    while kill -0 "$DEPLOY_PID" 2>/dev/null; do
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            warn "deploy pid ${DEPLOY_PID} still running after ${timeout}s — sending SIGKILL"
            perf_signal_deploy KILL
            wait "$DEPLOY_PID" 2>/dev/null || true
            DEPLOY_PID=""
            return 1
        fi
        sleep 2
    done
    wait "$DEPLOY_PID" 2>/dev/null || true
    DEPLOY_PID=""
}

perf_interrupt_deploy_for_jfr() {
    local target_pid
    [[ -n "${DEPLOY_PID:-}" ]] || return 0
    if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
        DEPLOY_PID=""
        return 0
    fi

    target_pid="$(perf_resolve_deploy_pid)"
    pgid="$(ps -o pgid= -p "$target_pid" 2>/dev/null | tr -d ' ')"
    log "sending SIGINT (Ctrl+C) to deploy monitor pgid=${pgid:-?} pid=${target_pid}…"
    log "waiting for [juno]   JFR Cluster Metrics in ${DEPLOY_LOG} (deploy monitor console)"
    perf_signal_deploy INT

    if perf_wait_for_deploy_jfr "$DEPLOY_LOG" "$PERF_DEPLOY_JFR_WAIT"; then
        log "[juno]   JFR Cluster Metrics captured from deploy monitor"
        perf_wait_for_deploy_exit "$PERF_DEPLOY_EXIT_WAIT"
        return 0
    fi

    perf_log_deploy_jfr_tail "$DEPLOY_LOG"
    warn "letting deploy monitor finish stop (up to ${PERF_DEPLOY_EXIT_WAIT}s)…"
    perf_wait_for_deploy_exit "$PERF_DEPLOY_EXIT_WAIT"
    return 1
}

perf_save_jfr_fragment() {
    local row_id="$1" column="$2"
    local metrics_file="${RUN_DIR}/metrics-${row_id}-${column}.json"

    {
        echo "===== row=${row_id} column=${column} hw=${PERF_ROW_HW} ====="
        if [[ -f "$metrics_file" ]]; then
            cat "$metrics_file"
        elif [[ -f "${DEPLOY_LOG:-}" ]]; then
            awk '/JFR Cluster Metrics/{show=1} show{print}' "$DEPLOY_LOG"
        fi
        echo ""
    } >> "${RUN_DIR}/metrics-fragments.log"
}

perf_teardown_cluster() {
    local launcher="${ROOT}/scripts/aws/launcher.sh"
    log "teardown cluster (independent cycle end)…"
    (cd "${ROOT}/scripts/aws" && "$launcher" juno-deploy.sh teardown) \
        || warn "teardown failed or cluster already gone"
}

perf_stop_deploy_no_jfr() {
    [[ -n "${DEPLOY_PID:-}" ]] || return 0
    if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
        DEPLOY_PID=""
        return 0
    fi
    log "stopping deploy monitor without JFR collection…"
    perf_signal_deploy KILL
    wait "$DEPLOY_PID" 2>/dev/null || true
    DEPLOY_PID=""
}

perf_abort_test_cycle() {
    perf_stop_deploy_no_jfr
    perf_teardown_cluster
}

perf_run_deploy_finish() {
    local metrics_out="$1"
    local launcher="${ROOT}/scripts/aws/launcher.sh"
    local rc=0

    log "juno-deploy finish (JFR gather + teardown) → ${metrics_out}"
    set +e
    (
        export JUNO_DEPLOY_METRICS_OUT="$metrics_out"
        cd "${ROOT}/scripts/aws"
        "$launcher" juno-deploy.sh finish
    ) 2>&1 | tee -a "${DEPLOY_LOG:-/dev/null}"
    rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$rc" -ne 0 ]]; then
        warn "juno-deploy finish exited with ${rc}"
        return 1
    fi
    if [[ -s "$metrics_out" ]] && command -v jq >/dev/null 2>&1 \
        && jq -e '.models | length > 0' "$metrics_out" >/dev/null 2>&1; then
        log "[juno]   JFR Cluster Metrics captured via finish"
        return 0
    fi
    if perf_deploy_log_has_jfr_json "${DEPLOY_LOG:-}"; then
        log "[juno]   JFR Cluster Metrics captured via finish (deploy log)"
        return 0
    fi
    warn "finish completed but metrics missing: ${metrics_out}"
    return 1
}

perf_finish_test_cycle() {
    local row_id="$1" column="$2"
    local metrics_dest="${RUN_DIR}/metrics-${row_id}-${column}.json"

    if [[ "${PERF_USE_DEPLOY_DETACH:-1}" == "1" ]]; then
        if perf_run_deploy_finish "$metrics_dest"; then
            perf_save_jfr_fragment "$row_id" "$column"
        else
            warn "JFR gather incomplete for row=${row_id} column=${column}"
            perf_teardown_cluster
        fi
        return
    fi

    if ! perf_interrupt_deploy_for_jfr; then
        warn "JFR gather incomplete for row=${row_id} column=${column}"
    elif perf_collect_metrics_json "$row_id" "$column"; then
        perf_save_jfr_fragment "$row_id" "$column"
    else
        warn "JFR metrics not saved for row=${row_id} column=${column}"
    fi

    perf_teardown_cluster
}

perf_run_single_test() {
    local row_id="$1" column="$2"
    local attempt max_attempts="${PERF_HTTP_MAX_ATTEMPTS:-3}"

    for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
        if (( attempt > 1 )); then
            log "retrying row=${row_id} column=${column} (${attempt}/${max_attempts}) after HTTP failure…"
            sleep 10
        fi
        if perf_run_single_test_attempt "$row_id" "$column"; then
            return 0
        fi
    done

    warn "test failed after ${max_attempts} attempts: row=${row_id} column=${column}"
    return 1
}

perf_run_single_test_attempt() {
    local row_id="$1" column="$2"
    local workload sessions max_tokens console_url outdir http_status=0

    perf_build_deploy_args "$row_id"
    workload="$(perf_column_workload "$column")"
    sessions="$(perf_column_sessions "$column")"
    max_tokens="$(perf_max_tokens_for_hw "$PERF_ROW_HW")"

    DEPLOY_LOG="${RUN_DIR}/deploy-${row_id}-${column}.log"
    outdir="${RUN_DIR}/http-${row_id}-${column}"
    DEPLOY_PID=""
    rm -f "$DEPLOY_LOG"
    rm -rf "$outdir"
    mkdir -p "$outdir"

    log "=== independent test: row=${row_id} column=${column} hw=${PERF_ROW_HW} sessions=${sessions} workload=${workload} ==="
    log "deploy log: ${DEPLOY_LOG}"

    perf_start_deploy

    if [[ "${PERF_USE_DEPLOY_DETACH:-1}" == "1" ]]; then
        if ! perf_wait_for_log_line "$DEPLOY_LOG" "Coordinator is healthy" 5400; then
            warn "coordinator health not seen — checking deploy log tail"
            tail -n 30 "$DEPLOY_LOG" >&2 || true
            perf_abort_test_cycle
            return 1
        fi
        if ! perf_wait_for_deploy_exit 600; then
            warn "deploy setup did not exit cleanly"
            perf_abort_test_cycle
            return 1
        fi
    elif ! perf_wait_for_log_line "$DEPLOY_LOG" "JUNO CLUSTER MONITOR" 5400; then
        warn "monitor banner not seen — checking deploy log tail"
        tail -n 30 "$DEPLOY_LOG" >&2 || true
        perf_abort_test_cycle
        return 1
    fi

    console_url="$(perf_extract_console_url "$DEPLOY_LOG")" \
        || { perf_abort_test_cycle; die "could not parse coordinator URL from ${DEPLOY_LOG}"; }

    if [[ "${PERF_USE_DEPLOY_DETACH:-1}" == "1" ]]; then
        log "cluster ready (detach) — console ${console_url}"
    else
        log "cluster monitor up — console ${console_url}"
    fi

    if ! perf_wait_for_api "$console_url" 300; then
        warn "API health check timed out for ${console_url}"
        perf_abort_test_cycle
        return 1
    fi

    log "running HTTP workload (${workload}, sessions=${sessions})…"
    case "$workload" in
        long) perf_run_long_test "$console_url" "$max_tokens" "$sessions" "$outdir" || http_status=1 ;;
        conv) perf_run_conv_test "$console_url" "$max_tokens" "$sessions" "$outdir" || http_status=1 ;;
    esac

    if (( http_status != 0 )); then
        warn "HTTP workload failed — aborting without JFR collection"
        perf_abort_test_cycle
        return 1
    fi

    log "HTTP test finished — responses in ${outdir}"

    if [[ "${PERF_USE_DEPLOY_DETACH:-1}" == "1" ]]; then
        log "juno-deploy finish → wait for [juno]   JFR Cluster Metrics → teardown"
    else
        perf_wait_for_monitor_idle_before_sigint "$DEPLOY_LOG" "$PERF_PRE_SIGINT_WAIT" \
            || warn "monitor idle wait incomplete — continuing with SIGINT"
        log "Ctrl+C equivalent on deploy monitor → wait for [juno]   JFR Cluster Metrics → teardown"
    fi

    perf_finish_test_cycle "$row_id" "$column"
    log "=== test complete: row=${row_id} column=${column} (cluster torn down) ==="
    return 0
}
