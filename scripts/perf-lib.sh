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
            log "still waiting for API (${now - start}s)…"
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
    local i timeout pids=()
    timeout="$(perf_curl_timeout "$max_tokens" "$sessions")"
    mkdir -p "$outdir"
    log "curl timeout ${timeout}s per request (sessions=${sessions}, max_tokens=${max_tokens})"
    if [[ "$sessions" -eq 1 ]]; then
        perf_curl_chat "$base" "$max_tokens" "$LONG_PROMPT" "${outdir}/long.json" "$timeout"
        return
    fi
    for i in $(seq 1 "$sessions"); do
        perf_curl_chat "$base" "$max_tokens" "$LONG_PROMPT" "${outdir}/long-${i}.json" "$timeout" &
        pids+=($!)
        log "started session ${i}/${sessions} (pid ${pids[-1]})"
    done
    perf_wait_pids "${pids[@]}"
}

perf_run_conv_test() {
    local base="$1" max_tokens="$2" sessions="$3" outdir="$4"
    local i t1 t2 t3 hist timeout pids=()
    timeout="$(perf_curl_timeout "$max_tokens" "$sessions")"
    mkdir -p "$outdir"
    log "curl timeout ${timeout}s per turn (sessions=${sessions}, max_tokens=${max_tokens})"

    run_one_conv() {
        local sid="$1"
        t1="${outdir}/conv-${sid}-t1.json"
        t2="${outdir}/conv-${sid}-t2.json"
        t3="${outdir}/conv-${sid}-t3.json"

        perf_curl_chat "$base" "$max_tokens" "$CONV_MSG1" "$t1" "$timeout"

        hist="$(jq -nc \
            --arg u1 "$CONV_MSG1" \
            --arg a1 "$(jq -r '.choices[0].message.content // ""' "$t1")" \
            --arg u2 "$CONV_MSG2" \
            '[{role:"user",content:$u1},{role:"assistant",content:$a1},{role:"user",content:$u2}]')"
        perf_curl_chat_history "$base" "$max_tokens" "$hist" "$t2" "$timeout"

        hist="$(jq -nc \
            --arg u1 "$CONV_MSG1" \
            --arg a1 "$(jq -r '.choices[0].message.content // ""' "$t1")" \
            --arg u2 "$CONV_MSG2" \
            --arg a2 "$(jq -r '.choices[0].message.content // ""' "$t2")" \
            --arg u3 "$CONV_MSG3" \
            '[{role:"user",content:$u1},{role:"assistant",content:$a1},{role:"user",content:$u2},{role:"assistant",content:$a2},{role:"user",content:$u3}]')"
        perf_curl_chat_history "$base" "$max_tokens" "$hist" "$t3" "$timeout"
    }

    if [[ "$sessions" -eq 1 ]]; then
        run_one_conv 1
        return
    fi
    for i in $(seq 1 "$sessions"); do
        run_one_conv "$i" &
        pids+=($!)
        log "started conv session ${i}/${sessions} (pid ${pids[-1]})"
    done
    perf_wait_pids "${pids[@]}"
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
            kill -KILL "$DEPLOY_PID" 2>/dev/null || true
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
    [[ -n "${DEPLOY_PID:-}" ]] || return 0
    if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
        DEPLOY_PID=""
        return 0
    fi

    log "sending SIGINT (Ctrl+C) to deploy monitor pid ${DEPLOY_PID} to gather JFR…"
    kill -INT "$DEPLOY_PID" 2>/dev/null || true

    if ! perf_wait_for_log_line "$DEPLOY_LOG" "JFR Cluster Metrics" 600; then
        warn "JFR Cluster Metrics not seen in ${DEPLOY_LOG} within 600s"
        perf_wait_for_deploy_exit 120
        return 1
    fi

    log "JFR metrics captured in deploy log"
    perf_wait_for_deploy_exit 300
    return 0
}

perf_save_jfr_fragment() {
    local row_id="$1" column="$2"
    [[ -f "${DEPLOY_LOG:-}" ]] || return 0
    {
        echo "===== row=${row_id} column=${column} hw=${PERF_ROW_HW} ====="
        awk '/JFR Cluster Metrics/{show=1} show{print}' "$DEPLOY_LOG"
        echo ""
    } >> "${RUN_DIR}/metrics-fragments.log"
}

perf_teardown_cluster() {
    local launcher="${ROOT}/scripts/aws/launcher.sh"
    log "teardown cluster (independent cycle end)…"
    (cd "${ROOT}/scripts/aws" && "$launcher" juno-deploy.sh teardown) \
        || warn "teardown failed or cluster already gone"
}

perf_finish_test_cycle() {
    local row_id="$1" column="$2"

    if ! perf_interrupt_deploy_for_jfr; then
        warn "JFR gather incomplete for row=${row_id} column=${column}"
    else
        perf_save_jfr_fragment "$row_id" "$column"
    fi

    perf_teardown_cluster
}

perf_run_single_test() {
    local row_id="$1" column="$2"
    local launcher="${ROOT}/scripts/aws/launcher.sh"
    local workload sessions max_tokens console_url outdir
    local -a deploy_cmd

    perf_build_deploy_args "$row_id"
    workload="$(perf_column_workload "$column")"
    sessions="$(perf_column_sessions "$column")"
    max_tokens="$(perf_max_tokens_for_hw "$PERF_ROW_HW")"

    DEPLOY_LOG="${RUN_DIR}/deploy-${row_id}-${column}.log"
    outdir="${RUN_DIR}/http-${row_id}-${column}"
    DEPLOY_PID=""
    rm -f "$DEPLOY_LOG"
    mkdir -p "$outdir"

    log "=== independent test: row=${row_id} column=${column} hw=${PERF_ROW_HW} sessions=${sessions} workload=${workload} ==="
    log "deploy log: ${DEPLOY_LOG}"

    deploy_cmd=("$launcher" juno-deploy.sh "${DEPLOY_ARGS[@]}")
    log "deploy: ${deploy_cmd[*]}"

    (
        cd "${ROOT}/scripts/aws"
        exec "${deploy_cmd[@]}"
    ) >"$DEPLOY_LOG" 2>&1 &
    DEPLOY_PID=$!

    if ! perf_wait_for_log_line "$DEPLOY_LOG" "JUNO CLUSTER MONITOR" 5400; then
        warn "monitor banner not seen — checking deploy log tail"
        tail -n 30 "$DEPLOY_LOG" >&2 || true
        perf_interrupt_deploy_for_jfr || true
        perf_teardown_cluster
        return 1
    fi

    console_url="$(perf_extract_console_url "$DEPLOY_LOG")" \
        || { perf_interrupt_deploy_for_jfr || true; perf_teardown_cluster; die "could not parse coordinator URL from ${DEPLOY_LOG}"; }

    log "cluster monitor up — console ${console_url}"

    if ! perf_wait_for_api "$console_url" 300; then
        warn "API health check timed out for ${console_url}"
        perf_interrupt_deploy_for_jfr || true
        perf_teardown_cluster
        return 1
    fi

    log "running HTTP workload (${workload}, sessions=${sessions})…"
    case "$workload" in
        long) perf_run_long_test "$console_url" "$max_tokens" "$sessions" "$outdir" ;;
        conv) perf_run_conv_test "$console_url" "$max_tokens" "$sessions" "$outdir" ;;
    esac

    log "HTTP test finished — responses in ${outdir}"
    log "Ctrl+C equivalent on deploy monitor → wait JFR → teardown"

    perf_finish_test_cycle "$row_id" "$column"
    log "=== test complete: row=${row_id} column=${column} (cluster torn down) ==="
}
