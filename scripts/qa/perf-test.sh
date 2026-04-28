#!/usr/bin/env bash
# =============================================================================
# perf-test.sh — Juno performance test runner
#
# Reads scripts/qa/perf-test.yaml, provisions AWS clusters in parallel
# (GPU + CPU in separate regions via screen sessions), fires OpenAI-compatible
# curl scenarios via perf-runner.sh, and prints a formatted per-suite report.
#
# Invocation (always via launcher.sh for credentials):
#   ./launcher.sh perf-test.sh --suite <id|number>
#   ./launcher.sh perf-test.sh --list
#   ./launcher.sh perf-test.sh --help
#
# Region assignment:
#   GPU -> eu-north-1  (L-DB2E81BA quota checked; swaps to us-east-1 on miss)
#   CPU -> us-east-1   (swaps to eu-north-1 when GPU took it)
#
# Each deployment runs with an isolated HOME so state files and SSH keys
# do not collide between parallel GPU/CPU setups.
# Teardown is guaranteed via EXIT trap.
# =============================================================================

set -uo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_DIR="${SCRIPT_DIR}/../qa"
YAML_FILE="${QA_DIR}/perf-test.yaml"
RUNNER="${QA_DIR}/perf-runner.sh"
YAML_PARSER="${QA_DIR}/yaml-parse.sh"
RESULTS_BASE="${SCRIPT_DIR}/../../target/perf-results"
LOG_BASE="/tmp/juno-perf"
HTTP_PORT=8080
SETUP_TIMEOUT=900    # 15 min: AMI pull + jar build
READY_TIMEOUT=600    # 10 min: coordinator HTTP readiness
GPU_VCPUS_NEEDED=8   # g4dn.2xlarge

GPU_REGION=""
CPU_REGION=""
GPU_HOME="${LOG_BASE}/gpu-home"
CPU_HOME="${LOG_BASE}/cpu-home"

# ---------- Colours -----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[perf]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
die()  { echo -e "${RED}[error]${RESET} $*" >&2; exit 1; }
hdr()  { printf "\n${BOLD}${CYAN}%-80s${RESET}\n" "=== $* ==="; }
sep()  { printf '%*s\n' 80 '' | tr ' ' '-'; }

# =============================================================================
# JSON HELPERS (no jq / no python3)
# =============================================================================

# Extract a scalar string value: "key":"value"
_json_str() {
    local key="$1" json="$2"
    printf '%s' "$json" | awk -v k="\"${key}\"" '
    {
        p = index($0, k ":")
        if (!p) next
        rest = substr($0, p + length(k) + 1)
        gsub(/^[ \t]*/, "", rest)
        if (substr(rest,1,1) != "\"") next
        rest = substr(rest, 2)
        val = ""
        for (i = 1; i <= length(rest); i++) {
            c = substr(rest, i, 1)
            if (c == "\\") { i++; val = val substr(rest, i, 1); continue }
            if (c == "\"") break
            val = val c
        }
        print val; exit
    }'
}

# Extract a scalar non-string value (number, bool, null): "key":value
_json_val() {
    local key="$1" json="$2"
    printf '%s' "$json" | sed -n \
        "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\([^,}\"[:space:]][^,}\"[:space:]]*\).*/\1/p" \
        | head -1
}

# Emit config object at 0-based index from a "configs":[...] JSON array.
_json_config_at() {
    local idx="$1" json="$2"
    printf '%s' "$json" | awk -v want="$idx" '
    BEGIN { depth=0; obj=""; cnt=-1; in_obj=0 }
    {
        line = $0
        for (i = 1; i <= length(line); i++) {
            c = substr(line, i, 1)
            if (c == "{") {
                depth++
                if (depth == 2) { cnt++; if (cnt == want) { in_obj=1; obj="" } }
            }
            if (in_obj) obj = obj c
            if (c == "}") {
                if (depth == 2 && in_obj) { print obj; exit }
                depth--
            }
        }
    }'
}

# Count objects in configs array.
_json_configs_len() {
    local json="$1"
    printf '%s' "$json" | awk '
    BEGIN { depth=0; cnt=0; in_arr=0 }
    {
        for (i=1; i<=length($0); i++) {
            c = substr($0,i,1)
            if (!in_arr && substr($0,i,9)=="\"configs\"") in_arr=1
            if (!in_arr) continue
            if (c == "[") depth++
            else if (c == "]") { depth--; if (depth==0) { print cnt; exit } }
            else if (c == "{" && depth == 1) cnt++
        }
    }'
}

# Extract a JSON array as newline-separated items (strings without quotes).
_json_arr_raw() {
    local key="$1" json="$2"
    printf '%s' "$json" | awk -v k="\"${key}\"" '
    {
        p = index($0, k ":")
        if (!p) next
        rest = substr($0, p + length(k) + 1)
        gsub(/^[ \t]*/, "", rest)
        if (substr(rest,1,1) != "[") next
        rest = substr(rest, 2)
        item=""; depth=0; in_str=0
        for (i=1; i<=length(rest); i++) {
            c=substr(rest,i,1)
            if (in_str) {
                if (c=="\\") { i++; continue }
                if (c=="\"") { in_str=0 }
                else { item=item c }
                continue
            }
            if (c=="\"") { in_str=1; continue }
            if (c=="{"||c=="[") { depth++; item=item c; continue }
            if (c=="}"||c=="]") {
                if (depth==0) {
                    gsub(/^[ \t]+|[ \t]+$/,"",item)
                    if (item!="") print item
                    break
                }
                depth--; item=item c; continue
            }
            if (c=="," && depth==0) {
                gsub(/^[ \t]+|[ \t]+$/,"",item)
                if (item!="") print item
                item=""; continue
            }
            item=item c
        }
    }'
}

# =============================================================================
# PREREQUISITES
# =============================================================================
check_deps() {
    local missing=()
    for cmd in aws screen curl bash awk sed; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    [[ ${#missing[@]} -eq 0 ]] || die "Missing dependencies: ${missing[*]}"
    [[ -f "$YAML_FILE"    ]] || die "Not found: $YAML_FILE"
    [[ -f "$RUNNER"       ]] || die "Not found: $RUNNER"
    [[ -f "$YAML_PARSER"  ]] || die "Not found: $YAML_PARSER"
    chmod +x "$RUNNER" "$YAML_PARSER"
}

# =============================================================================
# YAML HELPERS  (delegated to yaml-parse.sh)
# =============================================================================

_source_parser() {
    # shellcheck disable=SC1090
    source "$YAML_PARSER"
}

list_suites() {
    _source_parser
    yaml_list_suites "$YAML_FILE"
}

parse_suite() {
    local selector="$1"
    _source_parser
    yaml_parse_suite "$selector" "$YAML_FILE"
}

# =============================================================================
# REGION DETECTION
# =============================================================================
detect_regions() {
    local preferred_gpu="eu-north-1"
    local preferred_cpu="us-east-1"
    local quota_code="L-DB2E81BA"

    log "Checking GPU quota in ${preferred_gpu} (code ${quota_code})..."
    local quota
    quota=$(aws service-quotas get-service-quota \
        --region "$preferred_gpu" \
        --service-code ec2 \
        --quota-code "$quota_code" \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "0")

    local has_quota
    has_quota=$(awk "BEGIN{print (${quota:-0} >= ${GPU_VCPUS_NEEDED}) ? 1 : 0}")
    if [[ "$has_quota" == "1" ]]; then
        GPU_REGION="$preferred_gpu"
        CPU_REGION="$preferred_cpu"
    else
        warn "GPU quota in ${preferred_gpu}: ${quota:-0} vCPUs (need ${GPU_VCPUS_NEEDED}). Swapping regions."
        GPU_REGION="$preferred_cpu"
        CPU_REGION="$preferred_gpu"
    fi
    log "GPU region: ${GPU_REGION}   CPU region: ${CPU_REGION}"
}

# =============================================================================
# DEPLOYMENT MANAGEMENT
# =============================================================================

init_hw_home() {
    local hw_home="$1"
    mkdir -p "${hw_home}/.ssh"
    local real_key="${HOME}/.ssh/juno-deploy-key.pem"
    if [[ -f "$real_key" ]]; then
        cp "$real_key" "${hw_home}/.ssh/juno-deploy-key.pem" 2>/dev/null || true
        chmod 600 "${hw_home}/.ssh/juno-deploy-key.pem" 2>/dev/null || true
    fi
}

start_deploy_screen() {
    local hw="$1"
    local setup_args="$2"
    local hw_home="$3"
    local region="$4"
    local log_file="$5"

    local sname="juno-${hw}"
    screen -S "$sname" -X quit 2>/dev/null || true

    local cmd
    cmd="export HOME='${hw_home}'
export AWS_DEFAULT_REGION='${region}'
export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID:-}'
export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY:-}'
cd '${SCRIPT_DIR}'
echo '[screen/${hw}] juno-deploy.sh setup starting in ${region}'
./juno-deploy.sh setup ${setup_args} --region '${region}' 2>&1 | tee '${log_file}'
echo '__SETUP_DONE__' | tee -a '${log_file}'"

    screen -dmS "$sname" bash -c "$cmd"
    log "Screen '${sname}' started — log: ${log_file}"
}

wait_setup_done() {
    local log_file="$1"
    local label="$2"
    local timeout="${3:-$SETUP_TIMEOUT}"
    local elapsed=0 interval=20

    log "Waiting for ${label} setup (timeout ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if grep -q '__SETUP_DONE__' "$log_file" 2>/dev/null; then
            log "${label} setup complete."
            return 0
        fi
        if grep -qE '^\[error\]|die |Insufficient quota|setup failed' "$log_file" 2>/dev/null; then
            warn "${label} setup log contains error — stopping wait early."
            return 1
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
        log "  ${label}: still running (${elapsed}s)"
    done
    warn "${label} setup timed out after ${timeout}s."
    return 1
}

get_coordinator_ip() {
    local hw_home="$1"
    local region="$2"
    local state_file="${hw_home}/.juno-deploy-state"

    [[ -f "$state_file" ]] || { echo ""; return; }

    local coord_mode coord_id node_ids
    coord_mode=$(grep '^COORDINATOR_MODE=' "$state_file" | head -1 | sed 's/.*=//;s/"//g')
    coord_id=$(  grep '^COORDINATOR_INSTANCE_ID=' "$state_file" | head -1 | sed 's/.*=//;s/"//g')
    node_ids=$(  grep '^INSTANCE_IDS=' "$state_file" | head -1 | sed 's/INSTANCE_IDS=//;s/"//g')

    local target_id
    if [[ "$coord_mode" == "separate" && -n "$coord_id" && "$coord_id" != '""' ]]; then
        target_id="$coord_id"
    else
        target_id=$(echo "$node_ids" | awk '{print $1}')
    fi

    [[ -z "$target_id" ]] && { echo ""; return; }

    aws ec2 describe-instances \
        --region "$region" \
        --instance-ids "$target_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo ""
}

wait_api_ready() {
    local host="$1"
    local label="$2"
    local timeout="${3:-$READY_TIMEOUT}"
    local elapsed=0 interval=20
    local url="http://${host}:${HTTP_PORT}/v1/models"

    [[ -z "$host" || "$host" == "None" ]] && {
        warn "${label}: no coordinator IP — skipping readiness wait."
        return 1
    }
    log "${label}: polling ${url} (timeout ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf --connect-timeout 5 --max-time 10 "$url" -o /dev/null 2>/dev/null; then
            log "${label}: API ready."
            return 0
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
        log "  ${label}: not ready yet (${elapsed}s)"
    done
    warn "${label}: API not ready after ${timeout}s."
    return 1
}

teardown_deploy() {
    local hw="$1"
    local hw_home="$2"
    local region="$3"
    [[ -z "$region" ]] && return 0

    log "Tearing down ${hw} cluster in ${region}..."
    (
        export HOME="$hw_home"
        export AWS_DEFAULT_REGION="$region"
        export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
        export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
        cd "$SCRIPT_DIR"
        ./juno-deploy.sh teardown --region "$region" \
            > "${LOG_BASE}/${hw}-teardown.log" 2>&1 || true
    )
    screen -S "juno-${hw}" -X quit 2>/dev/null || true
    log "${hw} teardown done."
}

# =============================================================================
# SUITE ORCHESTRATION
# =============================================================================

build_setup_args() {
    local config_json="$1"
    local hw="$2"
    local model_url="$3"
    local jfr="$4"

    # Extract instance type from the hw_defs sub-object for this hw target
    local itype
    itype=$(printf '%s' "$config_json" | awk -v hw="$hw" '
    {
        p = index($0, "\"" hw "\":{")
        if (!p) next
        rest = substr($0, p)
        pi = index(rest, "\"instance_type\":\"")
        if (!pi) next
        rest = substr(rest, pi + 17)
        gsub(/".*/, "", rest)
        print rest; exit
    }')
    itype="${itype:-$([ "$hw" == "cpu" ] && echo "m7i-flex.large" || echo "g4dn.2xlarge")}"

    local nc;    nc=$(   _json_val "node_count"     "$config_json"); nc="${nc:-1}"
    local coord; coord=$(_json_str "coordinator"    "$config_json"); coord="${coord:-separate}"
    local ptype; ptype=$(_json_str "ptype"          "$config_json"); ptype="${ptype:-pipeline}"
    local dtype; dtype=$(_json_str "dtype"          "$config_json"); dtype="${dtype:-FLOAT16}"
    local bord;  bord=$( _json_str "byte_order"     "$config_json"); bord="${bord:-BE}"
    local lora;  lora=$( _json_val "lora"           "$config_json"); lora="${lora:-false}"
    local lpath; lpath=$(_json_str "lora_play_path" "$config_json")

    local parts=(
        "--instance-type ${itype}"
        "--node-count ${nc}"
        "--coordinator ${coord}"
        "--model-url ${model_url}"
        "--ptype ${ptype}"
        "--dtype ${dtype}"
        "--byteOrder ${bord}"
        "--jfr ${jfr}"
    )

    if [[ "$lora" == "true" && -n "$lpath" && \
          "$lpath" != "null" && \
          "$lpath" != "/path/to/tinyllama-1.1b-chat-v1.0-q4_k_m.lora" ]]; then
        parts+=("--lora-play ${lpath}")
    fi

    printf '%s ' "${parts[@]}"
}

run_hw_target() {
    local hw="$1"
    local hw_home="$2"
    local region="$3"
    local config_json="$4"
    local model_url="$5"
    local model_name="$6"
    local jfr="$7"
    local results_dir="$8"

    mkdir -p "$results_dir"
    local result_file="${results_dir}/result.json"
    local log_file="${LOG_BASE}/${hw}-setup.log"

    local setup_args
    setup_args=$(build_setup_args "$config_json" "$hw" "$model_url" "$jfr") || {
        printf '{"hw":"%s","region":"%s","error":"setup args build failed"}\n' \
            "$hw" "$region" > "$result_file"; return; }

    init_hw_home "$hw_home"
    start_deploy_screen "$hw" "$setup_args" "$hw_home" "$region" "$log_file"

    if ! wait_setup_done "$log_file" "$hw"; then
        printf '{"hw":"%s","region":"%s","error":"setup failed or timed out"}\n' \
            "$hw" "$region" > "$result_file"; return; fi

    local coord_ip
    coord_ip=$(get_coordinator_ip "$hw_home" "$region")
    if [[ -z "$coord_ip" || "$coord_ip" == "None" ]]; then
        printf '{"hw":"%s","region":"%s","error":"coordinator IP unavailable"}\n' \
            "$hw" "$region" > "$result_file"; return; fi
    log "[${hw}] Coordinator at ${coord_ip}:${HTTP_PORT}"

    if ! wait_api_ready "$coord_ip" "$hw"; then
        printf '{"hw":"%s","region":"%s","error":"API readiness timeout"}\n' \
            "$hw" "$region" > "$result_file"; return; fi

    # Extract runner parameters
    local max_tokens sessions streaming scenarios_arg session_id_mode

    max_tokens=$(printf '%s' "$config_json" | awk -v hw="$hw" '
    {
        p = index($0, "\"" hw "\":{")
        if (!p) next
        rest = substr($0, p)
        pm = index(rest, "\"max_tokens\":")
        if (!pm) next
        rest = substr(rest, pm + 13)
        gsub(/[^0-9].*/, "", rest)
        print rest; exit
    }')
    max_tokens="${max_tokens:-50}"

    sessions=$(         _json_val "sessions"          "$config_json"); sessions="${sessions:-1}"
    streaming=$(        _json_val "streaming"         "$config_json"); streaming="${streaming:-false}"
    session_id_mode=$(  _json_val "x_juno_session_id" "$config_json"); session_id_mode="${session_id_mode:-false}"

    scenarios_arg=$(_json_arr_raw "scenarios" "$config_json" | tr '\n' ',' | sed 's/,$//')
    [[ -z "$scenarios_arg" ]] && scenarios_arg="s1"

    # Extract request_variants JSON object if present
    local variants_json
    variants_json=$(printf '%s' "$config_json" | awk '
    BEGIN { depth=0; in_rv=0; rv="" }
    {
        for (i=1; i<=length($0); i++) {
            c=substr($0,i,1)
            if (!in_rv) {
                if (substr($0,i,18)=="\"request_variants\"") {
                    p=index(substr($0,i),":{")
                    if (p) { in_rv=1; skip=p-1; i+=skip; depth=1; rv="{"; continue }
                }
                continue
            }
            if (c=="{") depth++
            else if (c=="}") {
                depth--
                if (depth==0) { rv=rv c; print rv; exit }
            }
            rv=rv c
        }
    }')
    [[ -z "$variants_json" ]] && variants_json="null"

    # Build config summary JSON for the report
    local cfg_itype cfg_nc cfg_ptype cfg_dtype cfg_bord cfg_lora cfg_coord
    cfg_itype=$(printf '%s' "$config_json" | awk -v hw="$hw" '
    { p=index($0,"\"" hw "\":{"); if(!p) next
      rest=substr($0,p); pi=index(rest,"\"instance_type\":\"")
      if(!pi) next; rest=substr(rest,pi+17); gsub(/".*$/,"",rest); print rest; exit }')
    cfg_itype="${cfg_itype:-?}"
    cfg_nc=$(   _json_val "node_count"   "$config_json"); cfg_nc="${cfg_nc:-?}"
    cfg_ptype=$(_json_str "ptype"        "$config_json"); cfg_ptype="${cfg_ptype:-?}"
    cfg_dtype=$(_json_str "dtype"        "$config_json"); cfg_dtype="${cfg_dtype:-?}"
    cfg_bord=$( _json_str "byte_order"   "$config_json"); cfg_bord="${cfg_bord:-?}"
    cfg_lora=$( _json_val "lora"         "$config_json"); cfg_lora="${cfg_lora:-?}"
    cfg_coord=$(_json_str "coordinator"  "$config_json"); cfg_coord="${cfg_coord:-?}"

    local config_summary
    config_summary=$(printf \
        '{"instance_type":"%s","node_count":%s,"ptype":"%s","dtype":"%s","byte_order":"%s","lora":%s,"coordinator":"%s"}' \
        "$cfg_itype" "$cfg_nc" "$cfg_ptype" "$cfg_dtype" "$cfg_bord" "$cfg_lora" "$cfg_coord")

    bash "$RUNNER" run \
        --host        "$coord_ip" \
        --port        "$HTTP_PORT" \
        --model       "$model_name" \
        --hw          "$hw" \
        --max-tokens  "$max_tokens" \
        --sessions    "$sessions" \
        --scenarios   "$scenarios_arg" \
        --streaming   "$streaming" \
        --session-id  "$session_id_mode" \
        --variants    "$variants_json" \
        --output-dir  "$results_dir" \
        --result-file "$result_file" \
        --hw-region   "$region" \
        --config      "$config_summary" \
        2>&1 | tee "${LOG_BASE}/${hw}-runner.log"
}

run_suite() {
    local suite_json="$1"
    local suite_id model_url model_name jfr n_configs

    suite_id=$(  _json_str "suite_id"   "$suite_json")
    model_url=$( _json_str "model_url"  "$suite_json")
    model_name=$(_json_str "model_name" "$suite_json")
    jfr=$(       _json_str "jfr"        "$suite_json")
    n_configs=$( _json_configs_len      "$suite_json")

    local ts; ts=$(date +%Y%m%d-%H%M%S)
    local results_dir="${RESULTS_BASE}/${suite_id}-${ts}"
    mkdir -p "$results_dir"

    hdr "SUITE: ${suite_id}   [${n_configs} config(s)]"

    local all_result_files=()

    for (( cfg_idx=0; cfg_idx<n_configs; cfg_idx++ )); do
        hdr "Config $(( cfg_idx+1 )) / ${n_configs}"

        local raw_cfg hw_defs_json config_json
        raw_cfg=$(_json_config_at "$cfg_idx" "$suite_json")

        # Extract hw_defs object from suite_json
        hw_defs_json=$(printf '%s' "$suite_json" | awk '
        BEGIN { in_hw=0; depth=0; obj="" }
        {
            for (i=1; i<=length($0); i++) {
                c=substr($0,i,1)
                if (!in_hw && substr($0,i,9)=="\"hw_defs\"") {
                    p=index(substr($0,i),":{")
                    if (p) { in_hw=1; i+=p-1; depth=1; obj="{"; continue }
                }
                if (in_hw) {
                    if (c=="{") depth++
                    else if (c=="}") {
                        depth--
                        if (depth==0) { obj=obj c; print obj; exit }
                    }
                    obj=obj c
                }
            }
        }')

        # Merge: strip trailing } from raw_cfg, append hw_defs field
        config_json="${raw_cfg%\}},\"hw_defs\":${hw_defs_json}}"

        local hw_list=()
        mapfile -t hw_list < <(_json_arr_raw "hardware" "$config_json")
        [[ ${#hw_list[@]} -eq 0 ]] && hw_list=("cpu")

        log "Hardware targets: ${hw_list[*]}"

        local hw_pids=()
        for hw in "${hw_list[@]}"; do
            local region hw_home cfg_results_dir
            if [[ "$hw" == "gpu" ]]; then
                region="$GPU_REGION"; hw_home="$GPU_HOME"
            else
                region="$CPU_REGION"; hw_home="$CPU_HOME"
            fi
            cfg_results_dir="${results_dir}/cfg${cfg_idx}/${hw}"

            (
                run_hw_target "$hw" "$hw_home" "$region" \
                    "$config_json" "$model_url" "$model_name" "$jfr" \
                    "$cfg_results_dir"
                teardown_deploy "$hw" "$hw_home" "$region"
            ) &
            hw_pids+=($!)
        done

        for pid in "${hw_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        for hw in "${hw_list[@]}"; do
            local rf="${results_dir}/cfg${cfg_idx}/${hw}/result.json"
            [[ -f "$rf" ]] && all_result_files+=("$rf")
        done

        sep
    done

    hdr "RESULTS: ${suite_id}"
    if [[ ${#all_result_files[@]} -gt 0 ]]; then
        local joined; joined=$(IFS=,; echo "${all_result_files[*]}")
        bash "$RUNNER" report \
            --suite-id     "$suite_id" \
            --timestamp    "$ts" \
            --result-files "$joined" \
            --report-out   "${results_dir}/report.json"
    else
        warn "No result files found — all deployments may have failed."
    fi
}

# =============================================================================
# MAIN
# =============================================================================

usage() {
    cat <<USAGE
Usage: ./launcher.sh perf-test.sh [options]

Options:
  --suite <id|number>   Run a specific suite by id or 1-based number
  --list                List all available suites
  --help                Show this help

Examples:
  ./launcher.sh perf-test.sh --list
  ./launcher.sh perf-test.sh --suite 1
  ./launcher.sh perf-test.sh --suite baseline
  ./launcher.sh perf-test.sh --suite dtype_sweep
USAGE
}

main() {
    local action="" suite_selector=""

    [[ $# -eq 0 ]] && { usage; exit 0; }

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --suite)   action="run"; suite_selector="${2:-}"; shift 2 ;;
            --list)    action="list"; shift ;;
            --help|-h) usage; exit 0 ;;
            *)         die "Unknown argument: $1" ;;
        esac
    done

    check_deps
    mkdir -p "$LOG_BASE" "$RESULTS_BASE"

    if [[ "$action" == "list" ]]; then
        list_suites; exit 0
    fi

    [[ "$action" == "run" ]]   || die "Specify --suite or --list."
    [[ -n "$suite_selector" ]] || die "--suite requires an id or number."

    local suite_json
    suite_json=$(parse_suite "$suite_selector")

    local err_val
    err_val=$(_json_str "error" "$suite_json")
    [[ -n "$err_val" ]] && die "$err_val"

    detect_regions

    trap '
        log "Caught exit signal — tearing down clusters..."
        teardown_deploy gpu "$GPU_HOME" "$GPU_REGION" 2>/dev/null || true
        teardown_deploy cpu "$CPU_HOME" "$CPU_REGION" 2>/dev/null || true
    ' EXIT INT TERM

    run_suite "$suite_json"
}

main "$@"