#!/bin/bash
# =============================================================
#  juno-deploy.sh — Provision, deploy, and monitor a Juno cluster on AWS
#
#  Replaces juno-infra.sh (GPU) and juno-infra-ft.sh (CPU).
#  Hardware is auto-detected inside the bootstrap: GPU nodes get
#  CUDA drivers and JUNO_USE_GPU=true; CPU nodes skip them.
#
#  Usage:
#    ./launcher.sh juno-deploy.sh setup   [options]
#    ./launcher.sh juno-deploy.sh start
#    ./launcher.sh juno-deploy.sh stop
#    ./launcher.sh juno-deploy.sh teardown
#    ./launcher.sh juno-deploy.sh status
#    ./launcher.sh juno-deploy.sh scan-regions
#
#  Core options (setup only):
#    --instance-type TYPE     EC2 instance type (default: g4dn.xlarge)
#                             GPU examples : g4dn.xlarge  g4dn.2xlarge  g4dn.4xlarge
#                             CPU examples : m7i-flex.large  c7i-flex.large  t3.medium
#    --node-count N           Number of inference nodes (default: 3)
#    --coordinator separate   Extra t3.medium coordinator instance  (nodeCount+1 total)
#                 node1       Coordinator JVM co-located on node 1  (default)
#    --model-url URL          Model to download during bootstrap
#                             (default: TinyLlama Q4_K_M from HuggingFace)
#    --ptype pipeline|tensor  Parallelism type passed to nodes (default: pipeline)
#    --dtype FLOAT32|FLOAT16  Activation dtype (default: FLOAT16)
#    --git git branch, tag or other reference
#
#  State is persisted to ~/.juno-deploy-state so stop/start/teardown
#  can be called without repeating options.
#
#  After setup completes and all services are healthy, the script
#  opens the web console in your browser and enters the live dashboard.
#  Ctrl+C → auto-stop all instances (EBS + key pair retained).
# =============================================================

set -euo pipefail
export LC_NUMERIC=C

# ── DEFAULTS ──────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-eu-north-1}"
INSTANCE_TYPE="g4dn.xlarge"
NODE_COUNT=3
GIT=main
COORDINATOR_MODE="node1"          # "node1" | "separate"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
MODEL_FILENAME="TinyLlama.gguf"
PTYPE="pipeline"
DTYPE="FLOAT16"
BYTE_ORDER="BE"
KEY_NAME="juno-deploy-key"
SG_NAME="juno-deploy-sg"
JFR_DURATION=""
LORA_PLAY_PATH=""
STATE_FILE="${HOME}/.juno-deploy-state"
SSH_KEY_FILE="${HOME}/.ssh/juno-deploy-key.pem"
MONITOR_INTERVAL=20
GRPC_PORT=19092
HTTP_PORT=8080

# ── INSTANCE PRICING TABLE (on-demand, eu-north-1) ────────────
# Extend as needed; used only for the cost display in the dashboard.
declare -A INSTANCE_PRICE=(
  [g4dn.xlarge]=0.526
  [g4dn.2xlarge]=1.052
  [g4dn.4xlarge]=1.686
  [g4dn.8xlarge]=2.971
  [m7i-flex.large]=0.0479
  [m7i-flex.xlarge]=0.0958
  [c7i-flex.large]=0.0408
  [c7i-flex.xlarge]=0.0816
  [t3.medium]=0.0416
  [t3.large]=0.0832
)
declare -A INSTANCE_VCPUS=(
  [g4dn.xlarge]=4
  [g4dn.2xlarge]=8
  [g4dn.4xlarge]=16
  [g4dn.8xlarge]=32
  [m7i-flex.large]=2
  [m7i-flex.xlarge]=4
  [c7i-flex.large]=2
  [c7i-flex.xlarge]=4
  [t3.medium]=2
  [t3.large]=2
)

# ── COLOURS ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[juno]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
die()  { echo -e "${RED}[error]${RESET} $*"; exit 1; }

require_cmd() {
  command -v "$1" &>/dev/null && return 0
  local install_hint="$2"
  if [[ "$install_hint" == apt:* ]]; then
    local apt_pkg="${install_hint#apt:}"
    warn "'$1' not found — auto-installing ${apt_pkg}…"
    sudo apt-get update -qq && sudo apt-get install -y -qq "$apt_pkg" \
      && log "  OK $apt_pkg installed" \
      || die "Could not install $apt_pkg — run: sudo apt install $apt_pkg"
  else
    die "'$1' not found — run: $install_hint"
  fi
}

# ── ARGUMENT PARSING ──────────────────────────────────────────
parse_options() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --instance-type)   INSTANCE_TYPE="$2";  shift 2 ;;
      --node-count)      NODE_COUNT="$2";     shift 2 ;;
      --git)             GIT="$2";            shift 2 ;;
      --coordinator)     COORDINATOR_MODE="$2"; shift 2 ;;
      --model-url)       MODEL_URL="$2";
                         MODEL_FILENAME="$(basename "$MODEL_URL" | cut -d'?' -f1)"
                         shift 2 ;;
      --ptype)           PTYPE="$2";          shift 2 ;;
      --dtype)           DTYPE="$2";          shift 2 ;;
      --byteOrder | --byte-order | --byteorder) BYTE_ORDER="${2^^}"; shift 2 ;;
      --region)          REGION="$2"; export AWS_DEFAULT_REGION="$2"; shift 2 ;;
      --jfr)             JFR_DURATION="$2";   shift 2 ;;
      --lora-play)       LORA_PLAY_PATH=$(realpath "$2" 2>/dev/null || echo "$2"); shift 2 ;;
      *)                 die "Unknown option: $1 (run without args for usage)" ;;
    esac
  done

  [[ "$COORDINATOR_MODE" =~ ^(node1|separate)$ ]] || \
    die "--coordinator must be 'node1' or 'separate', got: $COORDINATOR_MODE"
  [[ "$PTYPE" =~ ^(pipeline|tensor)$ ]] || \
    die "--ptype must be 'pipeline' or 'tensor', got: $PTYPE"
}

# ── STATE PERSISTENCE ─────────────────────────────────────────
save_state() {
  {
    echo "INSTANCE_IDS=\"${INSTANCE_IDS[*]}\""
    echo "SG_ID=\"$SG_ID\""
    echo "INSTANCE_TYPE=\"$INSTANCE_TYPE\""
    echo "NODE_COUNT=\"$NODE_COUNT\""
    echo "GIT=\"$GIT\""
    echo "COORDINATOR_MODE=\"$COORDINATOR_MODE\""
    echo "COORDINATOR_INSTANCE_ID=\"${COORDINATOR_INSTANCE_ID:-}\""
    echo "PTYPE=\"$PTYPE\""
    echo "DTYPE=\"$DTYPE\""
    echo "BYTE_ORDER=\"$BYTE_ORDER\""
    echo "MODEL_FILENAME=\"$MODEL_FILENAME\""
    echo "SETUP_TIME=\"$SETUP_TIME\""
    echo "JFR_DURATION=\"${JFR_DURATION:-}\""
    echo "LORA_PLAY_PATH=\"${LORA_PLAY_PATH:-}\""
  } > "$STATE_FILE"
  log "State saved → $STATE_FILE"
}

load_state() {
  [[ -f "$STATE_FILE" ]] || die "No state file. Run: ./launcher.sh juno-deploy.sh setup"
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"
}

instance_state() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null || echo "unknown"
}

private_ip() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PrivateIpAddress" \
    --output text 2>/dev/null || echo ""
}

public_ip() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text 2>/dev/null || echo ""
}

# ── STOP ──────────────────────────────────────────────────────
stop() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Stopping Juno cluster (instances preserved)…"
  log "═══════════════════════════════════════════"

  load_state 2>/dev/null || { warn "Nothing to stop."; return 0; }

  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  local running=()
  for ID in "${all_ids[@]}"; do
    local state; state=$(instance_state "$ID")
    [[ "$state" == "running" ]] && running+=("$ID") || \
      warn "  $ID is '$state' — skipping"
  done

  if [[ ${#running[@]} -eq 0 ]]; then
    warn "  No running instances to stop."
    return 0
  fi

  aws ec2 stop-instances --instance-ids "${running[@]}" \
    --region "$REGION" --output text &>/dev/null \
    && log "  ✅ Stop initiated" || warn "  Could not stop instances"

  aws ec2 wait instance-stopped --instance-ids "${running[@]}" \
    --region "$REGION" 2>/dev/null \
    && log "  ✅ All instances stopped" || warn "  Wait timed out"

  log ""
  log "  ✅  Cluster stopped. Run 'start' to resume."
  log "═══════════════════════════════════════════"
}

# ── START ─────────────────────────────────────────────────────
start() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Starting stopped Juno cluster…"
  log "═══════════════════════════════════════════"

  load_state

  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  local stopped=()
  for ID in "${all_ids[@]}"; do
    local state; state=$(instance_state "$ID")
    case "$state" in
      stopped) stopped+=("$ID") ;;
      running) warn "  $ID is already running — skipping" ;;
      *)       warn "  $ID is '$state' — skipping" ;;
    esac
  done

  [[ ${#stopped[@]} -eq 0 ]] && { warn "  No stopped instances to start."; return 0; }

  aws ec2 start-instances --instance-ids "${stopped[@]}" \
    --region "$REGION" --output text &>/dev/null \
    && log "  ✅ Start initiated" || die "  Could not start instances"

  aws ec2 wait instance-running --instance-ids "${stopped[@]}" \
    --region "$REGION" \
    && log "  ✅ All instances running" || die "  Instances did not reach 'running'"

  _fetch_ips_and_monitor
}

# ── TEARDOWN ──────────────────────────────────────────────────
teardown() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Tearing down Juno cluster…"
  log "═══════════════════════════════════════════"

  load_state 2>/dev/null || { warn "Nothing to tear down."; return 0; }

  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  if [[ ${#all_ids[@]} -gt 0 ]]; then
    log "Terminating ${#all_ids[@]} instances…"
    aws ec2 terminate-instances --instance-ids "${all_ids[@]}" \
      --region "$REGION" --output text &>/dev/null \
      && log "  ✅ Termination initiated" || warn "  Could not terminate"
    aws ec2 wait instance-terminated --instance-ids "${all_ids[@]}" \
      --region "$REGION" 2>/dev/null \
      && log "  ✅ Terminated" || warn "  Wait timed out"
  fi

  if [[ -n "${SG_ID:-}" ]]; then
    log "Deleting security group $SG_ID…"
    for i in 1 2 3 4 5; do
      aws ec2 delete-security-group --group-id "$SG_ID" \
        --region "$REGION" &>/dev/null && { log "  ✅ SG deleted"; break; }
      warn "  Retry $i/5 — waiting 10s…"; sleep 10
    done
  fi

  aws ec2 delete-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" &>/dev/null \
    && log "  ✅ Key pair deleted" || warn "  Key pair already gone"

  [[ -f "$SSH_KEY_FILE" ]] && rm -f "$SSH_KEY_FILE" && log "  ✅ Local key removed"
  rm -f "$STATE_FILE"

  log ""
  log "  ✅  Cluster fully torn down. No lingering AWS costs."
  log "═══════════════════════════════════════════"
}

# ── STATUS ────────────────────────────────────────────────────
status() {
  load_state 2>/dev/null || { warn "No state file. Run setup first."; return 0; }

  echo ""
  log "═══════════════════════════════════════════"
  log "  Juno cluster status"
  log "═══════════════════════════════════════════"
  printf "  %-22s %s\n" "Instance type:"   "$INSTANCE_TYPE"
  printf "  %-22s %s\n" "Node count:"      "$NODE_COUNT"
  printf "  %-22s %s\n" "Git reference:"   "$GIT"
  printf "  %-22s %s\n" "Coordinator:"     "$COORDINATOR_MODE"
  printf "  %-22s %s\n" "Parallelism:"     "$PTYPE"
  printf "  %-22s %s\n" "Dtype:"           "$DTYPE"
  printf "  %-22s %s\n" "Model:"           "$MODEL_FILENAME"
  echo ""

  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  local IDX=1
  for ID in "${INSTANCE_IDS[@]}"; do
    local st; st=$(instance_state "$ID")
    local pip; pip=$(public_ip "$ID")
    printf "  node-%-2d  %-22s  %-10s  %s\n" "$IDX" "$pip" "$st" "$ID"
    (( IDX++ ))
  done
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && {
    local st; st=$(instance_state "$COORDINATOR_INSTANCE_ID")
    local pip; pip=$(public_ip "$COORDINATOR_INSTANCE_ID")
    printf "  coord    %-22s  %-10s  %s\n" "$pip" "$st" "$COORDINATOR_INSTANCE_ID"
  }
  echo ""
}

# ── MONITOR DASHBOARD ─────────────────────────────────────────
_fetch_ips_and_monitor() {
  declare -gA INSTANCE_IPS
  declare -gA INSTANCE_PRIVATE_IPS

  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  for ID in "${all_ids[@]}"; do
    INSTANCE_IPS[$ID]="$(public_ip "$ID")"
    INSTANCE_PRIVATE_IPS[$ID]="$(private_ip "$ID")"
  done

  # Determine coordinator IP for web console URL
  local COORD_IP
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    COORD_IP="${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}"
  else
    # node1 is coordinator — use node1's public IP
    COORD_IP="${INSTANCE_IPS[${INSTANCE_IDS[0]}]}"
  fi
  local CONSOLE_URL="http://${COORD_IP}:${HTTP_PORT}"

  log ""
  log "═══════════════════════════════════════════"
  log "  Cluster is UP  [${INSTANCE_TYPE}  ×${NODE_COUNT}  ${PTYPE} git ref: ${GIT}]"
  for ID in "${INSTANCE_IDS[@]}"; do
    log "  • $ID  ${INSTANCE_IPS[$ID]}"
  done
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && \
    log "  • ${COORDINATOR_INSTANCE_ID}  ${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}  (coordinator)"
  log ""
  log "  Bootstrap may still be running on each node."
  log "  Bootstrap log : /var/log/juno-bootstrap.log"
  log "  SSH           : ssh -i ${SSH_KEY_FILE} ubuntu@<IP>"
  log ""
  log "  Web console   : ${CONSOLE_URL}  (available once juno-ready)"
  log "═══════════════════════════════════════════"
  log ""

  # ── Wait for all nodes to be juno-ready, then open browser ──
  _wait_for_ready_and_open "$CONSOLE_URL"

  trap '_on_exit' EXIT INT TERM
  log "Entering monitoring dashboard (Ctrl+C to exit & auto-stop)…"
  sleep 3

  while true; do
    clear
    local NOW ELAPSED HOURS MINS SECS ELAPSED_HRS TOTAL_COST HOURLY_RATE
    NOW=$(date +%s)
    ELAPSED=$(( NOW - SETUP_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    SECS=$(( ELAPSED % 60 ))
    ELAPSED_HRS=$(echo "scale=4; $ELAPSED / 3600" | bc)

    # Total cost: all instances (nodes + optional separate coordinator)
    local all_count=${#INSTANCE_IDS[@]}
    [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && (( all_count++ ))
    TOTAL_COST=$(echo "scale=4; $all_count * ${INSTANCE_PRICE[$INSTANCE_TYPE]:-0.5} * $ELAPSED_HRS" | bc)
    HOURLY_RATE=$(echo "scale=4; $all_count * ${INSTANCE_PRICE[$INSTANCE_TYPE]:-0.5}" | bc)

    echo -e "${BOLD}${CYAN}"
    echo "  ╔════════════════════════════════════════════════════╗"
    echo "  ║              JUNO CLUSTER MONITOR                  ║"
    printf "  ║  %-50s║\n" "${INSTANCE_TYPE}  ×${NODE_COUNT}  ${PTYPE}  ${DTYPE}${JFR_DURATION:+  JFR:${JFR_DURATION}}"
    printf "  ║  %-50s║\n" "git ref: ${GIT}"
    echo -e "  ╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    printf "  ${BOLD}Uptime      :${RESET}  %02d:%02d:%02d\n" $HOURS $MINS $SECS
    printf "  ${BOLD}Est. cost   :${RESET}  \$%.4f  (\$%.4f/hr × %d instances)\n" \
      "$TOTAL_COST" "${INSTANCE_PRICE[$INSTANCE_TYPE]:-0}" "$all_count"
    printf "  ${BOLD}Console     :${RESET}  ${CYAN}%s${RESET}\n" "$CONSOLE_URL"
    echo ""
    echo -e "  ${BOLD}Nodes:${RESET}"

    local IDX=1
    for ID in "${INSTANCE_IDS[@]}"; do
      local IP="${INSTANCE_IPS[$ID]}"
      local SYS_STATUS NODE_INFO IS_COORD=false
      # node1 is coordinator in co-located mode; separate mode uses COORDINATOR_INSTANCE_ID
      [[ "$COORDINATOR_MODE" == "node1" && "$IDX" -eq 1 ]] && IS_COORD=true

      SYS_STATUS=$(aws ec2 describe-instance-status \
        --instance-ids "$ID" --region "$REGION" \
        --query "InstanceStatuses[0].InstanceStatus.Status" \
        --output text 2>/dev/null || echo "unknown")

      NODE_INFO="$(_probe_node "$ID" "$IP" "$IDX" "$IS_COORD")"
      printf "  ${CYAN}  node-%-2d${RESET}  %-22s  sys:%-10s  %s\n" \
        "$IDX" "$IP" "$SYS_STATUS" "$NODE_INFO"
      (( IDX++ ))
    done

    # Coordinator row (separate instance only)
    if [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
      local CIP="${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}"
      local CST
      CST=$(aws ec2 describe-instance-status \
        --instance-ids "$COORDINATOR_INSTANCE_ID" --region "$REGION" \
        --query "InstanceStatuses[0].InstanceStatus.Status" \
        --output text 2>/dev/null || echo "unknown")
      local CINFO
      CINFO=$(_probe_coordinator "$COORDINATOR_INSTANCE_ID" "$CIP")
      printf "  ${CYAN}  coord  ${RESET}  %-22s  sys:%-10s  %s\n" \
        "$CIP" "$CST" "$CINFO"
    fi

    echo ""
    echo -e "  ${DIM}Refreshing every ${MONITOR_INTERVAL}s — Ctrl+C to exit & auto-stop${RESET}"
    echo ""
    sleep "$MONITOR_INTERVAL"
  done
}

# Probe one node — GPU stats if available, otherwise CPU/RAM + ready state
# $4 IS_COORD: true if this node also runs juno-coordinator (node1 in co-located mode)
_probe_node() {
  local ID="$1" IP="$2" IDX="$3" IS_COORD="${4:-false}"
  local OUT

  if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes \
         -i "$SSH_KEY_FILE" "ubuntu@$IP" \
         "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; \
          nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | head -1" \
         >/tmp/juno_probe_"$IDX".txt 2>/dev/null; then

    read -r USED TOTAL GPU_UTIL < <(tr ',' ' ' < /tmp/juno_probe_"$IDX".txt)
    if [[ -n "$USED" && -n "$TOTAL" && "$TOTAL" -gt 0 ]]; then
      local PCT FILLED BAR=""
      PCT=$(echo "scale=0; $USED * 100 / $TOTAL" | bc)
      FILLED=$(( USED * 20 / TOTAL ))
      for ((b=0; b<20; b++)); do
        [[ $b -lt $FILLED ]] && BAR+="█" || BAR+="░"
      done
      echo "VRAM ${USED}/${TOTAL} MiB (${PCT}%)  GPU:${GPU_UTIL}%  [${BAR}]"
      return
    fi
  fi

  # CPU fallback — includes service health + last journal lines on failure
  if OUT=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes \
             -i "$SSH_KEY_FILE" "ubuntu@$IP" \
             "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
              ready=\$([[ -f /opt/juno/.juno-ready ]] && echo yes || echo no);
              cpu=\$(awk '/^cpu /{print \$2,\$3,\$4,\$5,\$6; exit}' /proc/stat > /tmp/_juno_cpu1; sleep 1; awk '/^cpu /{print \$2,\$3,\$4,\$5,\$6; exit}' /proc/stat | awk 'NR==1{while((getline line < "/tmp/_juno_cpu1")>0){split(line,a); u1=a[1]+a[2]+a[3]; t1=a[1]+a[2]+a[3]+a[4]+a[5]}} {u2=\$1+\$2+\$3; t2=\$1+\$2+\$3+\$4+\$5; dt=t2-t1; printf \"%.0f\", (dt>0?(u2-u1)/dt*100:0)}')%;
              mem=\$(free -m | awk '/^Mem/{printf \"%d/%dMB\",\$3,\$2}');
              svc_node=\$(systemctl is-active juno-node 2>/dev/null); [[ -z \$svc_node ]] && svc_node=unknown;
              svc_coord=\$(systemctl is-active juno-coordinator 2>/dev/null); [[ -z \$svc_coord ]] && svc_coord=unknown;
              is_coord='${IS_COORD}';
              echo \"ready:\$ready cpu:\$cpu mem:\$mem node:\$svc_node coord:\$svc_coord\";
              if [[ \$svc_node != active ]]; then
                echo '--- juno-node journal ---';
                journalctl -u juno-node --no-pager -n 8 2>/dev/null | tail -8;
              fi;
              if [[ \$is_coord == true && \$svc_coord != active ]]; then
                echo '--- juno-coordinator journal ---';
                journalctl -u juno-coordinator --no-pager -n 8 2>/dev/null | tail -8;
              fi" 2>/dev/null); then
    echo "$OUT"
  else
    echo "unreachable"
  fi
}

# Probe coordinator — check if REST port is responding
_probe_coordinator() {
  local ID="$1" IP="$2"
  local STATUS

  if STATUS=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes \
                  -i "$SSH_KEY_FILE" "ubuntu@$IP" \
                  "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; \
                   curl -sf http://localhost:${HTTP_PORT}/v1/cluster/health 2>/dev/null \
                   | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[\"status\"],\"q=\"+str(d[\"queueDepth\"]))' \
                   2>/dev/null || \
                   ([[ -f /opt/juno/.juno-ready ]] && echo 'ready (juno not started yet)' \
                    || tail -1 /var/log/juno-bootstrap.log 2>/dev/null || echo 'bootstrapping')" \
                  2>/dev/null); then
    echo "$STATUS"
  else
    echo "unreachable"
  fi
}

# _wait_for_ready_and_open is defined below alongside
# _write_cluster_env_and_start_coordinator (single canonical definition).

# ── GATHER JFR METRICS ────────────────────────────────────────
# Called from _on_exit before stop() when JFR_DURATION is set.
# 1. SSH each node → SCP /opt/juno/jfr/*.jfr to a local temp dir
# 2. SCP all JFRs to coordinator's /opt/juno/  (MetricsMain scan root)
# 3. SSH coordinator → run MetricsMain jar to produce metrics.json
# 4. SCP metrics.json back → print to stdout
_gather_jfr_metrics() {
  [[ -n "${JFR_DURATION:-}" ]] || return 0

  log "Gathering JFR metrics from all nodes…"

  local TMP_DIR
  TMP_DIR=$(mktemp -d)
  trap "rm -rf $TMP_DIR" RETURN

  # Determine coordinator host
  local COORD_HOST
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    COORD_HOST="${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}"
  else
    COORD_HOST="${INSTANCE_IPS[${INSTANCE_IDS[0]}]}"
  fi

  local SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes -i $SSH_KEY_FILE"

  # Step 1 — stop services on each node so JFR files are flushed (dumponexit)
  local IDX=1
  for ID in "${INSTANCE_IDS[@]}"; do
    local IP="${INSTANCE_IPS[$ID]}"
    log "  Stopping juno-node on ${IP} to flush JFR…"
    ssh $SSH_OPTS "ubuntu@${IP}" \
      "/bin/sudo /bin/systemctl stop juno-node 2>/dev/null || true; /bin/sudo /bin/systemctl stop juno-coordinator 2>/dev/null || true" \
      2>/dev/null || warn "  Could not stop services on ${IP} (continuing)"
    (( IDX++ ))
  done
  # Also stop coordinator if separate
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    ssh $SSH_OPTS "ubuntu@${COORD_HOST}" \
      "/bin/sudo /bin/systemctl stop juno-coordinator 2>/dev/null || true" 2>/dev/null || true
  fi

  sleep 3   # brief pause to ensure dumponexit fires

  # Step 2 — SCP .jfr files from each node to local temp dir
  for ID in "${INSTANCE_IDS[@]}"; do
    local IP="${INSTANCE_IPS[$ID]}"
    log "  Fetching JFR files from node ${IP}…"
    scp $SSH_OPTS -r "ubuntu@${IP}:/opt/juno/jfr/*.jfr" "$TMP_DIR/" 2>/dev/null \
      || warn "  No JFR files found on ${IP}"
  done
  # Coordinator (separate mode) has its own recording
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    scp $SSH_OPTS -r "ubuntu@${COORD_HOST}:/opt/juno/jfr/*.jfr" "$TMP_DIR/" 2>/dev/null \
      || warn "  No JFR files found on coordinator ${COORD_HOST}"
  fi

  local JFR_COUNT
  JFR_COUNT=$(find "$TMP_DIR" -name "*.jfr" | wc -l)
  if [[ "$JFR_COUNT" -eq 0 ]]; then
    warn "No JFR files collected — skipping metrics extraction."
    return 0
  fi
  log "  Collected ${JFR_COUNT} JFR file(s) locally."

  # Step 3 — SCP all .jfr files to coordinator's home dir (ubuntu-writable; /opt/juno is root-owned)
  local REMOTE_COLLECT="/home/ubuntu/jfr-collect"
  log "  Uploading JFR files to coordinator for extraction…"
  ssh $SSH_OPTS "ubuntu@${COORD_HOST}" "/bin/mkdir -p ${REMOTE_COLLECT}" 2>/dev/null || true
  scp $SSH_OPTS "$TMP_DIR"/*.jfr "ubuntu@${COORD_HOST}:${REMOTE_COLLECT}/" \
    || { warn "  Could not upload JFR files to coordinator."; return 0; }

  # Step 4 — Run MetricsMain on coordinator from the collect dir (scans cwd for *.jfr)
  log "  Running MetricsMain on coordinator…"
  local MODEL_STEM="${MODEL_FILENAME%.*}"
  ssh $SSH_OPTS "ubuntu@${COORD_HOST}" \
    "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; \
     cd ${REMOTE_COLLECT} && \
     /bin/mkdir -p metrics/src/main/resources && \
     /usr/bin/jq --arg name '${MODEL_STEM}' --arg path '/opt/juno/models/${MODEL_FILENAME}' \
       'if (.models | map(.name) | index(\$name)) == null then .models += [{\"name\":\$name,\"path\":\$path}] else . end' \
       /opt/juno/metrics/src/main/resources/models.json \
       > metrics/src/main/resources/models.json && \
     /usr/bin/java -cp /opt/juno/metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain 2>&1" \
    || { warn "  MetricsMain failed on coordinator."; return 0; }

  # Step 5 — SCP metrics.json back and print
  local LOCAL_JSON="$TMP_DIR/metrics.json"
  scp $SSH_OPTS "ubuntu@${COORD_HOST}:${REMOTE_COLLECT}/target/metrics/metrics.json" "$LOCAL_JSON" \
    || { warn "  Could not retrieve metrics.json from coordinator."; return 0; }

  log ""
  log "══════════════════════════════════════════════════════════"
  log "  JFR Cluster Metrics"
  log "══════════════════════════════════════════════════════════"
  cat "$LOCAL_JSON"
  log "══════════════════════════════════════════════════════════"
  log ""
}

_on_exit() {
  echo ""
  warn "Caught exit signal — gathering metrics (if JFR enabled) then stopping…"
  trap - EXIT INT TERM
  _gather_jfr_metrics
  stop
}

# ── AMI RESOLUTION ────────────────────────────────────────────
# Checks if a Juno golden AMI already exists in the account for the given
# base OS + instance type.  If found, sets AMI_ID and returns.
# If not found, calls make-ami.sh to bake one (GPU bake takes ~30-40 min).
#
# The golden AMI has JDK 25, Maven, and (for GPU instances) CUDA 12.3 +
# nvidia-open pre-installed, shaving ~15-20 min off every bootstrap.
_resolve_ami() {
  local BASE="Ubuntu 22.04 LTS"
  local BASE_SLUG
  BASE_SLUG=$(echo "$BASE" | sed 's/[ .]/-/g')
  local AMI_NAME="Juno-golden-${BASE_SLUG}_${INSTANCE_TYPE}"

  log "Checking for golden AMI: ${AMI_NAME}…"
  AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners self \
    --filters "Name=name,Values=${AMI_NAME}" "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null || echo "")

  if [[ -n "$AMI_ID" && "$AMI_ID" != "None" ]]; then
    log "  OK Using golden AMI: $AMI_ID"
    return 0
  fi

  log "  Golden AMI not found — invoking make-ami.sh…"
  local IS_GPU=false
  [[ "$INSTANCE_TYPE" =~ ^g[0-9]|^p[0-9] ]] && IS_GPU=true
  if $IS_GPU; then
    log "  GPU instance — AMI bake takes ~30-40 min (CUDA DKMS compilation included)"
  fi

  local MAKE_AMI
  MAKE_AMI="$(dirname "${BASH_SOURCE[0]}")/make-ami.sh"
  [[ -f "$MAKE_AMI" ]] || die "make-ami.sh not found at: $MAKE_AMI"

  AMI_ID=$(bash "$MAKE_AMI" \
    --base "$BASE" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --key-file "$SSH_KEY_FILE" \
    --region "$REGION") || die "make-ami.sh exited with error"

  [[ -n "$AMI_ID" && "$AMI_ID" != "None" ]] || \
    die "make-ami.sh returned an empty AMI ID"
  log "  OK Golden AMI ready: $AMI_ID"
}

# ── SETUP ─────────────────────────────────────────────────────
setup() {
  require_cmd aws   "pip install awscli"
  require_cmd jq    "apt:jq"
  require_cmd bc    "apt:bc"
  require_cmd ssh   "apt:openssh-client"
  require_cmd curl  "apt:curl"

  # ── Resolve vCPU count for this instance type ────────────────
  local VCPUS_PER_NODE="${INSTANCE_VCPUS[$INSTANCE_TYPE]:-2}"

  # ── GPU quota check (only for G/P instance families) ─────────
  if [[ "$INSTANCE_TYPE" =~ ^g[0-9]|^p[0-9] ]]; then
    local VCPUS_NEEDED=$(( NODE_COUNT * VCPUS_PER_NODE ))
    log "Checking EC2 GPU quota (L-DB2E81BA)…"
    local QUOTA_VALUE
    QUOTA_VALUE=$(aws service-quotas get-service-quota \
      --service-code ec2 --quota-code L-DB2E81BA \
      --region "$REGION" --query "Quota.Value" --output text 2>/dev/null || echo "")

    if [[ -z "$QUOTA_VALUE" || "$QUOTA_VALUE" == "None" ]]; then
      warn "  Could not read GPU quota — continuing anyway."
    else
      local QUOTA_INT
      QUOTA_INT=$(echo "$QUOTA_VALUE" | awk '{printf "%d", $1}')
      if [[ "$QUOTA_INT" -eq 0 ]]; then
        die "GPU quota is 0 vCPUs in ${REGION}.\n\
  Request an increase: https://console.aws.amazon.com/servicequotas/\n\
  EC2 → Running On-Demand G and VT instances (L-DB2E81BA)\n\
  Minimum needed: ${VCPUS_NEEDED} vCPUs"
      elif [[ "$QUOTA_INT" -lt "$VCPUS_NEEDED" ]]; then
        die "GPU quota too low: have ${QUOTA_INT} vCPUs, need ${VCPUS_NEEDED} (${NODE_COUNT} x ${VCPUS_PER_NODE} vCPUs for ${INSTANCE_TYPE}).\n\
  Request an increase: https://console.aws.amazon.com/servicequotas/\n\
  EC2 -> Running On-Demand G and VT instances (L-DB2E81BA)\n\
  Target value: ${VCPUS_NEEDED} vCPUs"
      else
        log "  OK Quota: ${QUOTA_INT} vCPUs available for ${NODE_COUNT} nodes"
      fi
    fi
  fi

  # ── Smart resume: check existing state ───────────────────────
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"
    local all_running=true all_stopped=true has_valid=false
    local all_ids=("${INSTANCE_IDS[@]}")
    [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")
    for ID in "${all_ids[@]}"; do
      local st; st=$(instance_state "$ID")
      [[ "$st" == "running" ]] || all_running=false
      [[ "$st" == "stopped" ]] || all_stopped=false
      [[ "$st" =~ ^(running|stopped|pending|stopping)$ ]] && has_valid=true
    done
    if $has_valid && $all_running; then
      log "Cluster already running — entering dashboard."
      _fetch_ips_and_monitor; return 0
    fi
    if $has_valid && $all_stopped; then
      log "Found stopped cluster — resuming…"
      SETUP_TIME=$(date +%s); save_state; start; return 0
    fi
    warn "Stale state (inconsistent/terminated) — fresh setup…"
    rm -f "$STATE_FILE"; INSTANCE_IDS=(); SG_ID=""; COORDINATOR_INSTANCE_ID=""
  fi

  log "Account: $(aws sts get-caller-identity --query Arn --output text)"
  log "Region : $REGION"
  log "Nodes  : ${NODE_COUNT} × ${INSTANCE_TYPE}  coordinator=${COORDINATOR_MODE}"
  [[ -n "${JFR_DURATION:-}"   ]] && log "JFR    : duration=${JFR_DURATION}"
  if [[ -n "${LORA_PLAY_PATH:-}" ]]; then
    if [[ ! -f "$LORA_PLAY_PATH" ]]; then
      die "✖ --lora-play file not found: '${LORA_PLAY_PATH}'\n  Pass an absolute path or run the command from the directory containing the file."
    fi
    log "LoRA   : play-back=${LORA_PLAY_PATH}"
  fi
  echo ""

  # ── Key pair ─────────────────────────────────────────────────
  # Must be created before _resolve_ami so that make-ami.sh can reuse
  # the deploy key for SSH access to the bake instance (enables tracing).
  log "Creating key pair…"
  mkdir -p "$(dirname "$SSH_KEY_FILE")"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" &>/dev/null || true
  aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
    --query "KeyMaterial" --output text > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"
  log "  ✅ Key saved → $SSH_KEY_FILE"

  # ── Resolve AMI: use golden AMI if available, bake one if not ──
  _resolve_ami

  # ── Resolve subnet + VPC ─────────────────────────────────────
  # Collect ALL AZ→subnet pairs so _launch_nodes can fall back to the
  # next AZ on InsufficientInstanceCapacity instead of aborting.
  log "Resolving subnet for $INSTANCE_TYPE in $REGION…"
  SUBNET_ID="" VPC_ID="" CHOSEN_AZ=""
  # Global ordered list of "SubnetId:AZ" pairs for AZ fallback during launch.
  AZ_SUBNET_LIST=()
  local AZ_LIST
  AZ_LIST=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
    --region "$REGION" \
    --query "InstanceTypeOfferings[].Location" \
    --output text 2>/dev/null | tr '\t' '\n')
  [[ -z "$AZ_LIST" ]] && die "$INSTANCE_TYPE not available in $REGION. Run scan-regions."

  for AZ in $AZ_LIST; do
    local ROW SN VP
    ROW=$(aws ec2 describe-subnets --region "$REGION" \
      --filters "Name=availabilityZone,Values=$AZ" "Name=defaultForAz,Values=true" \
      --query "Subnets[0].[SubnetId,VpcId]" --output text 2>/dev/null)
    SN=$(awk '{print $1}' <<< "$ROW")
    VP=$(awk '{print $2}' <<< "$ROW")
    if [[ -n "$SN" && "$SN" != "None" ]]; then
      AZ_SUBNET_LIST+=("$SN:$AZ")
      if [[ -z "$SUBNET_ID" ]]; then
        # First hit — use this VPC for the security group
        SUBNET_ID="$SN"; VPC_ID="$VP"; CHOSEN_AZ="$AZ"
        log "  OK Subnet: $SUBNET_ID  VPC: $VPC_ID  (AZ: $AZ)"
      else
        log "  ALT Subnet: $SN  (AZ: $AZ) — fallback if primary AZ has no capacity"
      fi
    else
      warn "  No default subnet in $AZ — skipping…"
    fi
  done
  [[ -z "$SUBNET_ID" ]] && die "No default subnet found for $INSTANCE_TYPE in $REGION."

  # Resolve VPC CIDR so internal SG rules match whatever address space the VPC uses
  # (default VPC is 172.31.0.0/16; custom VPCs are often 10.x or 192.168.x)
  VPC_CIDR=$(aws ec2 describe-vpcs --region "$REGION" \
    --vpc-ids "$VPC_ID" \
    --query "Vpcs[0].CidrBlock" --output text 2>/dev/null || echo "")
  [[ -z "$VPC_CIDR" || "$VPC_CIDR" == "None" ]] && VPC_CIDR="0.0.0.0/0"
  log "  OK VPC CIDR: $VPC_CIDR"

  # ── Security group ───────────────────────────────────────────
  log "Creating security group…"
  local OLD_SG
  OLD_SG=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
  [[ "$OLD_SG" != "None" && -n "$OLD_SG" ]] && \
    aws ec2 delete-security-group --group-id "$OLD_SG" --region "$REGION" &>/dev/null || true

  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Juno deploy cluster" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query "GroupId" --output text)

  local MY_IP
  MY_IP=$(curl -sf https://checkip.amazonaws.com)

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=SSH}]" \
      "IpProtocol=tcp,FromPort=${GRPC_PORT},ToPort=$(( GRPC_PORT + NODE_COUNT + 1 )),IpRanges=[{CidrIp=${VPC_CIDR},Description=Juno-gRPC-internal}]" \
      "IpProtocol=tcp,FromPort=${HTTP_PORT},ToPort=${HTTP_PORT},IpRanges=[{CidrIp=0.0.0.0/0,Description=Juno-REST}]" \
      "IpProtocol=tcp,FromPort=8081,ToPort=8081,IpRanges=[{CidrIp=${VPC_CIDR},Description=Juno-health-sidecar-internal}]" \
      "IpProtocol=tcp,FromPort=5701,ToPort=5701,IpRanges=[{CidrIp=${VPC_CIDR},Description=Hazelcast}]" \
      &>/dev/null
  log "  OK Security group: $SG_ID  (SSH from ${MY_IP}, gRPC internal, REST public)"

  # ── Build bootstrap scripts ───────────────────────────────────
  # In separate mode the coordinator must be launched FIRST so its private IP
  # is known and can be baked into every node's JUNO_HEALTH_URL at creation
  # time. In node1 mode the order is irrelevant (no separate coordinator).
  _launch_coordinator_if_separate
  _launch_nodes

  SETUP_TIME=$(date +%s)
  save_state

  log "Waiting for instances to reach 'running' state…"
  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")
  aws ec2 wait instance-running --instance-ids "${all_ids[@]}" --region "$REGION"
  log "  ✅ All instances running"

  _fetch_ips_and_monitor
}

# ── LAUNCH NODE INSTANCES ──────────────────────────────────────
_launch_nodes() {
  log "Launching ${NODE_COUNT} × ${INSTANCE_TYPE} node(s)…"
  INSTANCE_IDS=()

  # Launch node-1 first so we can capture its private IP and bake
  # JUNO_HEALTH_URL into every other node's node.env at creation time.
  # No file writes, no restarts — coordinator URL is in RAM from first boot.
  # In separate mode the coordinator was already launched by _launch_coordinator_if_separate
  # and its private IP stored in COORDINATOR_PRIVATE_IP.  Seed COORD_PRIV_IP with it so
  # every node gets the correct JUNO_HEALTH_URL baked in from the start.
  local COORD_PRIV_IP="${COORDINATOR_PRIVATE_IP:-}"

  for (( i=1; i<=NODE_COUNT; i++ )); do
    local NODE_IDX=$i
    local IS_COORDINATOR_NODE=false
    [[ "$COORDINATOR_MODE" == "node1" && "$i" -eq 1 ]] && IS_COORDINATOR_NODE=true

    local USER_DATA
    USER_DATA=$(_build_node_userdata "$NODE_IDX" "$IS_COORDINATOR_NODE" "$COORD_PRIV_IP")
    local USER_DATA_FILE
    USER_DATA_FILE=$(mktemp /tmp/juno-userdata-XXXXXX.sh)
    printf '%s' "$USER_DATA" > "$USER_DATA_FILE"

    log "  Launching node $i / ${NODE_COUNT}…"
    log "  [TRACE] user-data size: $(wc -c < "$USER_DATA_FILE") bytes  first-line: $(head -1 "$USER_DATA_FILE")"

    # Try each AZ subnet in order; fall back on InsufficientInstanceCapacity.
    local LAUNCH_OUT="" LAUNCH_ERR LAUNCH_OK=false LAUNCH_SUBNET=""
    LAUNCH_ERR=$(mktemp)
    for AZ_ENTRY in "${AZ_SUBNET_LIST[@]}"; do
      LAUNCH_SUBNET="${AZ_ENTRY%%:*}"
      local TRY_AZ="${AZ_ENTRY##*:}"
      [[ "$LAUNCH_SUBNET" != "$SUBNET_ID" ]] &&         log "  [AZ-fallback] Retrying in $TRY_AZ (subnet $LAUNCH_SUBNET)…"
      > "$LAUNCH_ERR"
      LAUNCH_OUT=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --count 1 \
        --key-name "$KEY_NAME" \
        --subnet-id "$LAUNCH_SUBNET" \
        --security-group-ids "$SG_ID" \
        --associate-public-ip-address \
        --user-data "file://${USER_DATA_FILE}" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}" \
        --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=juno-node-${i}},{Key=Project,Value=juno}]" \
        --region "$REGION" \
        --output json 2>"$LAUNCH_ERR") && { LAUNCH_OK=true; break; }
      # If it is a capacity error, try the next AZ; otherwise abort immediately.
      if grep -q "InsufficientInstanceCapacity" "$LAUNCH_ERR"; then
        warn "  No capacity for $INSTANCE_TYPE in $TRY_AZ — trying next AZ…"
      else
        break   # Hard error — will be reported below
      fi
    done

    # Last resort: all explicit AZ subnets were capacity-exhausted.
    # Let AWS pick any AZ by omitting --subnet-id entirely (AWS recommendation).
    if ! $LAUNCH_OK && grep -q "InsufficientInstanceCapacity" "$LAUNCH_ERR"; then
      warn "  All AZs reported InsufficientInstanceCapacity — retrying with no AZ pin (AWS-managed placement)…"
      > "$LAUNCH_ERR"
      LAUNCH_OUT=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --count 1 \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --associate-public-ip-address \
        --user-data "file://${USER_DATA_FILE}" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}" \
        --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=juno-node-${i}},{Key=Project,Value=juno}]" \
        --region "$REGION" \
        --output json 2>"$LAUNCH_ERR") && LAUNCH_OK=true
    fi

    if ! $LAUNCH_OK; then
      echo ""
      echo -e "${RED}  Node $i launch FAILED:${RESET}"
      cat "$LAUNCH_ERR" | sed 's/^/    /'
      rm -f "$LAUNCH_ERR"
      [[ ${#INSTANCE_IDS[@]} -gt 0 ]] && {
        warn "Terminating ${#INSTANCE_IDS[@]} already-launched node(s)…"
        aws ec2 terminate-instances --instance-ids "${INSTANCE_IDS[@]}" \
          --region "$REGION" --output text &>/dev/null || true
      }
      rm -f "$LAUNCH_ERR" "$USER_DATA_FILE"
      die "Launch aborted."
    fi
    rm -f "$LAUNCH_ERR" "$USER_DATA_FILE"
    local IID
    IID=$(echo "$LAUNCH_OUT" | jq -r '.Instances[0].InstanceId')
    INSTANCE_IDS+=("$IID")
    log "  OK node-${i}: $IID"

    # After launching node-1, read its private IP immediately.
    # AWS assigns it at creation — no need to wait for "running".
    if [[ "$IS_COORDINATOR_NODE" == "true" && -z "$COORD_PRIV_IP" ]]; then
      COORD_PRIV_IP=$(aws ec2 describe-instances         --instance-ids "$IID" --region "$REGION"         --query "Reservations[0].Instances[0].PrivateIpAddress"         --output text 2>/dev/null || echo "")
      log "  OK coordinator private IP: ${COORD_PRIV_IP}"
    fi
  done
  log "  ✅ All nodes launched: ${INSTANCE_IDS[*]}"
}

# ── LAUNCH SEPARATE COORDINATOR INSTANCE ──────────────────────
_launch_coordinator_if_separate() {
  COORDINATOR_INSTANCE_ID=""
  COORDINATOR_PRIVATE_IP=""
  [[ "$COORDINATOR_MODE" != "separate" ]] && return 0
  local COORD_USERDATA
  COORD_USERDATA=$(_build_coordinator_userdata)
  local COORD_USERDATA_FILE
  COORD_USERDATA_FILE=$(mktemp /tmp/juno-userdata-coord-XXXXXX.sh)
  printf '%s' "$COORD_USERDATA" > "$COORD_USERDATA_FILE"
  log "  [TRACE] coordinator user-data size: $(wc -c < "$COORD_USERDATA_FILE") bytes"

  local LAUNCH_ERR; LAUNCH_ERR=$(mktemp)
  local LAUNCH_OUT
  LAUNCH_OUT=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "t3.medium" \
    --count 1 \
    --key-name "$KEY_NAME" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SG_ID" \
    --associate-public-ip-address \
    --user-data "file://${COORD_USERDATA_FILE}" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=juno-coordinator},{Key=Project,Value=juno}]" \
    --region "$REGION" \
    --output json 2>"$LAUNCH_ERR") || {
      echo -e "${RED}  Coordinator launch FAILED:${RESET}"
      cat "$LAUNCH_ERR" | sed 's/^/    /'; rm -f "$LAUNCH_ERR" "$COORD_USERDATA_FILE"
      die "Coordinator launch aborted."
    }
  rm -f "$LAUNCH_ERR" "$COORD_USERDATA_FILE"
  rm -f "$LAUNCH_ERR"
  COORDINATOR_INSTANCE_ID=$(echo "$LAUNCH_OUT" | jq -r '.Instances[0].InstanceId')
  # Fetch the coordinator's private IP immediately — AWS assigns it at creation,
  # no need to wait for "running". Stored in a global so _launch_nodes can bake
  # it into every node's JUNO_HEALTH_URL via __COORDINATOR_PRIVATE_IP__.
  COORDINATOR_PRIVATE_IP=$(aws ec2 describe-instances \
    --instance-ids "$COORDINATOR_INSTANCE_ID" --region "$REGION" \
    --query "Reservations[0].Instances[0].PrivateIpAddress" \
    --output text 2>/dev/null || echo "")
  log "  ✅ Coordinator: $COORDINATOR_INSTANCE_ID  private-ip: ${COORDINATOR_PRIVATE_IP}"
}

# ── NODE BOOTSTRAP SCRIPT ──────────────────────────────────────
#
# Runs on every compute node. Steps:
#   1. Install JDK 25 + Maven (always)
#   2. Detect GPU via lspci; install CUDA if found, else set JUNO_USE_GPU=false
#   3. Clone & build juno; download model
#   4. Write /etc/juno/env with JUNO_* env vars
#   5. Install juno-node.service (NodeMain) — starts immediately
#   6. If IS_COORDINATOR=true, also install juno-coordinator.service (CoordinatorMain)
#      but mark it as WantedBy=juno-node.service so it waits for the node first
#   7. touch .juno-ready
#
# The coordinator service on node1 needs private IPs of all other nodes, which
# are not available at bootstrap time. So the coordinator service is configured
# to poll /opt/juno/cluster-nodes.env, which the deploy script SSHes in after
# all nodes are running (see _write_cluster_env).
#
_build_node_userdata() {
  local NODE_IDX="$1"
  local IS_COORDINATOR="$2"
  local COORD_PRIV_IP_ARG="${3:-}"   # coordinator private IP; empty for node-1 (self-discovers)
  local PORT=$(( GRPC_PORT + NODE_IDX - 1 ))

  local MODEL_URL_VAL="$MODEL_URL"
  local MODEL_FILENAME_VAL="$MODEL_FILENAME"
  local MODEL_STEM_VAL="${MODEL_FILENAME%.*}"
  local JFR_DURATION_VAL="${JFR_DURATION:-}"
  local NODE_IDX_VAL="$NODE_IDX"
  local PORT_VAL="$PORT"
  local IS_COORD_VAL="$IS_COORDINATOR"
  local HTTP_PORT_VAL="$HTTP_PORT"
  local PTYPE_VAL="$PTYPE"
  local DTYPE_VAL="$DTYPE"
  local NODE_COUNT_VAL="$NODE_COUNT"
  local GIT_VAL="$GIT"
  # ── [TRACE] Show exactly what is being baked into the bootstrap script ──────
  # IMPORTANT: must go to stderr (>&2). This function is called as
  # USER_DATA=$(_build_node_userdata ...) so stdout is captured verbatim as the
  # cloud-init user-data script. Any log output on stdout would prepend before
  # #!/bin/bash, causing cloud-init to reject the script as non-multipart.
  log "  [TRACE] node-${NODE_IDX_VAL} bootstrap params:" >&2
  log "          grpc_port=${PORT_VAL}  is_coordinator=${IS_COORD_VAL}  dtype=${DTYPE_VAL}" >&2
  log "          model=${MODEL_FILENAME_VAL}  git=${GIT_VAL}" >&2
  log "          jfr=${JFR_DURATION_VAL:-<none>}  coord_ip=${COORD_PRIV_IP_ARG:-self-discover}" >&2

  sed -e "s|__NODE_IDX__|${NODE_IDX_VAL}|g" \
      -e "s|__PORT__|${PORT_VAL}|g" \
      -e "s|__IS_COORD__|${IS_COORD_VAL}|g" \
      -e "s|__HTTP_PORT__|${HTTP_PORT_VAL}|g" \
      -e "s|__PTYPE__|${PTYPE_VAL}|g" \
      -e "s|__DTYPE__|${DTYPE_VAL}|g" \
      -e "s|__BYTE_ORDER__|${BYTE_ORDER}|g" \
      -e "s|__NODE_COUNT__|${NODE_COUNT_VAL}|g" \
      -e "s|__GIT__|${GIT_VAL}|g" \
      -e "s|__MODEL_FILENAME__|${MODEL_FILENAME_VAL}|g" \
      -e "s|__MODEL_STEM__|${MODEL_STEM_VAL}|g" \
      -e "s|__JFR_DURATION__|${JFR_DURATION_VAL}|g" \
      -e "s|__LORA_PLAY_PATH__||g" \
      -e "s|__MODEL_URL__|${MODEL_URL_VAL}|g" \
      -e "s|__COORDINATOR_PRIVATE_IP__|${COORD_PRIV_IP_ARG}|g" \
      -e "s|__HTTP_PORT_VAL__|${HTTP_PORT_VAL}|g" \
      <<'EOF'
#!/bin/bash
exec > /var/log/juno-bootstrap.log 2>&1
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ── Prevent "held broken packages" on EC2 Ubuntu ──────────────
# unattended-upgrades holds dpkg locks during boot; kill it before
# any apt work so we never race against it.
systemctl stop unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
# Remove any stale lock files left by a previous or concurrent apt run.
rm -f /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/cache/apt/archives/lock \
      /var/lib/apt/lists/lock
# Finish configuring any packages that were interrupted (e.g. by cloud-init).
dpkg --configure -a 2>/dev/null || true
# ──────────────────────────────────────────────────────────────

NODE_IDX="__NODE_IDX__"
NODE_ID="node-${NODE_IDX}"
GRPC_PORT="__PORT__"
IS_COORDINATOR="__IS_COORD__"
HTTP_PORT="__HTTP_PORT__"
PTYPE="__PTYPE__"
DTYPE="__DTYPE__"
BYTE_ORDER="__BYTE_ORDER__"
NODE_COUNT="__NODE_COUNT__"
GIT="__GIT__"
MODEL_PATH="/opt/juno/models/__MODEL_FILENAME__"
MODEL_STEM="__MODEL_STEM__"
JFR_DURATION="__JFR_DURATION__"

echo "=== Bootstrap started $(date) ==="

apt-get update -qq
apt-get install -y -qq \
  openjdk-25-jdk maven git wget curl jq bc \
  numactl net-tools htop pciutils lsof

USE_GPU=false
if lspci | grep -qi nvidia; then
  # CUDA drivers and toolkit were pre-installed by make-ami.sh when the
  # golden AMI was baked.  _resolve_ami() guarantees the golden AMI always
  # exists before any instance is launched, so there is nothing to install.
  echo "GPU detected — CUDA pre-installed in golden AMI"
  USE_GPU=true
else
  echo "No GPU found — CPU-only mode"
fi

git clone https://github.com/ml-cab/juno /opt/juno
cd /opt/juno
git checkout ${GIT} --
mvn clean package -DskipTests -q
echo "Build complete"

mkdir -p /opt/juno/models
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Downloading model from __MODEL_URL__…"
  wget -q "__MODEL_URL__" -O "${MODEL_PATH}"
  echo "Model downloaded: $(du -sh ${MODEL_PATH} | cut -f1)"
else
  echo "Model already present: ${MODEL_PATH}"
fi

mkdir -p /etc/juno
# Coordinator private IP for health probes.
# node-1 (IS_COORDINATOR=true) self-discovers its own private IP via EC2 metadata.
# All other nodes have the coordinator IP embedded as __COORDINATOR_PRIVATE_IP__.
if [[ "${IS_COORDINATOR}" == "true" ]]; then
  _COORD_IP=$(curl -sf --retry 3 --retry-delay 1     http://169.254.169.254/latest/meta-data/local-ipv4 2>/dev/null || echo "127.0.0.1")
else
  _COORD_IP="__COORDINATOR_PRIVATE_IP__"
fi
cat > /etc/juno/node.env <<EOF2
JUNO_USE_GPU=${USE_GPU}
JUNO_MODEL_PATH=${MODEL_PATH}
JUNO_GRPC_PORT=${GRPC_PORT}
JUNO_BYTE_ORDER=${BYTE_ORDER}
NODE_ID=${NODE_ID}
JUNO_JFR_DURATION=${JFR_DURATION}
JUNO_MODEL_STEM=${MODEL_STEM}
JUNO_LORA_PLAY_PATH=__LORA_PLAY_PATH__
JUNO_HEALTH_URL=http://${_COORD_IP}:__HTTP_PORT_VAL__
EOF2

# Wrapper script — conditionally adds -XX:StartFlightRecording when JFR is enabled
mkdir -p /opt/juno/scripts
cat > /opt/juno/scripts/start-node.sh <<'EOF2'
#!/bin/bash
set -a   # auto-export every variable so exec'd java inherits them via System.getenv()
source /etc/juno/node.env
set +a
JFR_OPT=""
if [[ -n "${JUNO_JFR_DURATION:-}" ]]; then
  mkdir -p /opt/juno/jfr
  JFR_OPT="-XX:StartFlightRecording=duration=${JUNO_JFR_DURATION},\
filename=/opt/juno/jfr/juno-${JUNO_MODEL_STEM}-$(date +%Y%m%d)-$(date +%H%M%S).jfr,\
settings=profile,dumponexit=true"
fi
exec /usr/bin/java \
  --enable-preview --enable-native-access=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  -XX:+UseG1GC -XX:+AlwaysPreTouch -Xmx12g \
  ${JFR_OPT:+$JFR_OPT} \
  -DJUNO_USE_GPU=${JUNO_USE_GPU} \
  -Djuno.byteOrder=${JUNO_BYTE_ORDER:-BE} \
  -Dnode.id=${NODE_ID} \
  -Dnode.port=${JUNO_GRPC_PORT} \
  -Dmodel.path=${JUNO_MODEL_PATH} \
  ${JUNO_LORA_PLAY_PATH:+-Djuno.lora.play.path=${JUNO_LORA_PLAY_PATH}} \
  ${JUNO_HEALTH_URL:+-Djuno.health.url=${JUNO_HEALTH_URL}} \
  -jar /opt/juno/juno-node/target/juno-node.jar \
  cab.ml.juno.node.NodeMain
EOF2
chmod +x /opt/juno/scripts/start-node.sh

cat > /etc/systemd/system/juno-node.service <<EOF2
[Unit]
Description=Juno Node ${NODE_ID}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/juno/node.env
WorkingDirectory=/opt/juno
ExecStart=/opt/juno/scripts/start-node.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=juno-node

[Install]
WantedBy=multi-user.target
EOF2

systemctl daemon-reload
systemctl enable juno-node
systemctl start juno-node
echo "juno-node.service started"

if [[ "${IS_COORDINATOR}" == "true" ]]; then
  echo "Setting up coordinator service on this node…"

  cat > /opt/juno/scripts/start-coordinator.sh <<'EOF2'
#!/bin/bash
set -a   # auto-export every variable so exec'd java inherits them via System.getenv()
source /etc/juno/node.env
source /etc/juno/cluster-nodes.env 2>/dev/null || true
set +a
JFR_OPT=""
if [[ -n "${JUNO_JFR_DURATION:-}" ]]; then
  mkdir -p /opt/juno/jfr
  JFR_OPT="-XX:StartFlightRecording=duration=${JUNO_JFR_DURATION},\
filename=/opt/juno/jfr/juno-${JUNO_MODEL_STEM}-$(date +%Y%m%d)-$(date +%H%M%S).jfr,\
settings=profile,dumponexit=true"
fi
exec /usr/bin/java \
  --enable-preview --enable-native-access=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  -XX:+UseG1GC -XX:+AlwaysPreTouch -Xmx4g \
  ${JFR_OPT:+$JFR_OPT} \
  -DJUNO_HEALTH=true \
  -DJUNO_HEALTH_PORT=8081 \
  -jar /opt/juno/juno-master/target/juno-master.jar \
  --model-path ${JUNO_MODEL_PATH} \
  --pType ${JUNO_PTYPE} \
  --dtype ${JUNO_DTYPE}
EOF2
  chmod +x /opt/juno/scripts/start-coordinator.sh

  cat > /etc/systemd/system/juno-coordinator.service <<'EOF2'
[Unit]
Description=Juno Coordinator
After=juno-node.service network-online.target
Requires=juno-node.service

[Service]
Type=simple
WorkingDirectory=/opt/juno
ExecStartPre=/bin/bash -c 'for i in $(seq 1 120); do [[ -f /etc/juno/cluster-nodes.env ]] && exit 0; sleep 5; done; exit 1'
ExecStart=/opt/juno/scripts/start-coordinator.sh
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=juno-coordinator

[Install]
WantedBy=multi-user.target
EOF2

  systemctl daemon-reload
  systemctl enable juno-coordinator
  echo "juno-coordinator.service enabled (waiting for cluster-nodes.env)"
fi

touch /opt/juno/.juno-ready
echo "=== Bootstrap complete $(date) ==="
EOF
}

# ── SEPARATE COORDINATOR BOOTSTRAP ────────────────────────────
_build_coordinator_userdata() {
  local HTTP_PORT_VAL="$HTTP_PORT"
  local PTYPE_VAL="$PTYPE"
  local DTYPE_VAL="$DTYPE"
  local MODEL_URL_VAL="$MODEL_URL"
  local MODEL_FILENAME_VAL="$MODEL_FILENAME"
  local MODEL_STEM_VAL="${MODEL_FILENAME%.*}"
  local JFR_DURATION_VAL="${JFR_DURATION:-}"
  local GIT_VAL="$GIT"

  cat <<ENDCOORD
#!/bin/bash
exec > /var/log/juno-bootstrap.log 2>&1
set -euo pipefail

HTTP_PORT=${HTTP_PORT_VAL}
PTYPE=${PTYPE_VAL}
DTYPE=${DTYPE_VAL}
MODEL_PATH="/opt/juno/models/${MODEL_FILENAME_VAL}"
JUNO_JFR_DURATION="${JFR_DURATION_VAL}"
JUNO_MODEL_STEM="${MODEL_STEM_VAL}"

echo "=== Coordinator bootstrap started \$(date) ==="

export DEBIAN_FRONTEND=noninteractive
systemctl stop unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/cache/apt/archives/lock \
      /var/lib/apt/lists/lock
dpkg --configure -a 2>/dev/null || true

apt-get update -qq
apt-get install -y -qq openjdk-25-jdk maven git wget curl jq bc net-tools

git clone https://github.com/ml-cab/juno /opt/juno
cd /opt/juno
git checkout ${GIT_VAL}
mvn clean package -DskipTests -q
echo "Build complete"

mkdir -p /opt/juno/models
if [[ ! -f "\${MODEL_PATH}" ]]; then
  echo "Downloading model from ${MODEL_URL_VAL}…"
  wget -q "${MODEL_URL_VAL}" -O "\${MODEL_PATH}"
fi

mkdir -p /etc/juno /opt/juno/scripts

cat > /opt/juno/scripts/start-coordinator.sh <<'EOF'
#!/bin/bash
set -a   # auto-export every variable so exec'd java inherits them via System.getenv()
source /etc/juno/cluster-nodes.env 2>/dev/null || true
set +a
JFR_OPT=""
if [[ -n "\${JUNO_JFR_DURATION:-}" ]]; then
  mkdir -p /opt/juno/jfr
  JFR_OPT="-XX:StartFlightRecording=duration=\${JUNO_JFR_DURATION},\
filename=/opt/juno/jfr/juno-\${JUNO_MODEL_STEM}-\$(date +%Y%m%d)-\$(date +%H%M%S).jfr,\
settings=profile,dumponexit=true"
fi
exec /usr/bin/java \
  --enable-preview --enable-native-access=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  -XX:+UseG1GC -XX:+AlwaysPreTouch -Xmx4g \
  \${JFR_OPT:+\$JFR_OPT} \
  -DJUNO_HEALTH=true \
  -DJUNO_HEALTH_PORT=8081 \
  -jar /opt/juno/juno-master/target/juno-master.jar \
  --model-path \${JUNO_MODEL_PATH} \
  --pType \${JUNO_PTYPE} \
  --dtype \${JUNO_DTYPE}
EOF
chmod +x /opt/juno/scripts/start-coordinator.sh

cat > /etc/systemd/system/juno-coordinator.service <<'EOF'
[Unit]
Description=Juno Coordinator (separate instance)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/juno
ExecStartPre=/bin/bash -c 'for i in \$(seq 1 120); do [[ -f /etc/juno/cluster-nodes.env ]] && exit 0; sleep 5; done; exit 1'
ExecStart=/opt/juno/scripts/start-coordinator.sh
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=juno-coordinator

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable juno-coordinator
echo "juno-coordinator.service enabled (waiting for cluster-nodes.env)"

touch /opt/juno/.juno-ready
echo "=== Coordinator bootstrap complete \$(date) ==="
ENDCOORD
}

# ── WRITE CLUSTER-NODES ENV AND START COORDINATOR ─────────────
# Called from _wait_for_ready_and_open after all nodes are ready.
# SSHes into the coordinator host and writes /etc/juno/cluster-nodes.env
# with the private IPs of all nodes, then starts juno-coordinator.service.
#
# Design notes:
#   • File content is passed via SSH stdin (heredoc), not via echo '...'
#     inside a double-quoted remote command — avoids all shell-quoting
#     pitfalls with multi-line strings and special characters.
#   • systemctl start uses --no-block so the SSH session returns
#     immediately after dispatching the start request; it does not block
#     waiting for ExecStartPre (which itself polls for the file we just
#     wrote — no circular dependency, but blocking would needlessly hold
#     the SSH connection).
#   • Retries up to 5 times with exponential back-off (5 s, 10 s, 20 s…)
#     to handle transient SSH failures when the node is still under load
#     from DKMS compilation at the moment the bootstrap timeout fires.
#   • ConnectTimeout raised to 30 s; heavy DKMS builds saturate CPU/IO
#     and the SSH handshake can be slow.
#   • Errors are no longer swallowed via 2>/dev/null so failures are
#     visible in the deploy log.
_write_cluster_env_and_start_coordinator() {
  log "Writing cluster-nodes.env and starting coordinator…"

  # Build JUNO_NODE_ADDRESSES from private IPs of all node instances
  local ADDRS=""
  local IDX=1
  for ID in "${INSTANCE_IDS[@]}"; do
    local PRIV_IP="${INSTANCE_PRIVATE_IPS[$ID]}"
    local PORT=$(( GRPC_PORT + IDX - 1 ))
    [[ -n "$ADDRS" ]] && ADDRS+=","
    ADDRS+="${PRIV_IP}:${PORT}"
    (( IDX++ ))
  done

  local COORD_HOST
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    COORD_HOST="${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}"
  else
    COORD_HOST="${INSTANCE_IPS[${INSTANCE_IDS[0]}]}"
  fi

  local ENV_CONTENT
  ENV_CONTENT=$(cat <<EOF
JUNO_NODE_ADDRESSES=${ADDRS}
JUNO_MODEL_PATH=/opt/juno/models/${MODEL_FILENAME}
JUNO_HTTP_PORT=${HTTP_PORT}
JUNO_PTYPE=${PTYPE}
JUNO_DTYPE=${DTYPE}
JUNO_BYTE_ORDER=${BYTE_ORDER}
JUNO_MAX_QUEUE=1000
JUNO_JFR_DURATION=${JFR_DURATION:-}
JUNO_MODEL_STEM=${MODEL_FILENAME%.*}
JUNO_LORA_PLAY_PATH=${LORA_PLAY_PATH:-}
EOF
)

  # [TRACE] Show exactly what will be written to the coordinator
  log "  [TRACE] cluster-nodes.env to be written to ${COORD_HOST}:"
  while IFS= read -r line; do
    log "          ${line}"
  done <<< "${ENV_CONTENT}"

  local SSH_OPTS="-o ConnectTimeout=30 -o StrictHostKeyChecking=no \
-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
-i ${SSH_KEY_FILE}"

  local MAX_ATTEMPTS=5
  local ATTEMPT=0
  local BACKOFF=5

  while (( ATTEMPT < MAX_ATTEMPTS )); do
    # Use arithmetic assignment, not (( ATTEMPT++ )), because (( 0++ )) evaluates
    # to 0 which makes bash exit with set -e on the very first iteration.
    ATTEMPT=$(( ATTEMPT + 1 ))
    [[ $ATTEMPT -gt 1 ]] && warn "  Retry ${ATTEMPT}/${MAX_ATTEMPTS} in ${BACKOFF}s…" && sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF * 2 ))

    # Write the file via stdin — safe for multi-line content, no quoting issues.
    # Then start the coordinator with --no-block so SSH returns immediately
    # without waiting for ExecStartPre/ExecStart to complete.
    if ssh $SSH_OPTS "ubuntu@${COORD_HOST}" \
         "/bin/sudo /bin/mkdir -p /etc/juno && \
          /bin/sudo /usr/bin/tee /etc/juno/cluster-nodes.env > /dev/null && \
          /bin/sudo /bin/systemctl start --no-block juno-coordinator && \
          echo 'coordinator dispatched'" \
         <<< "${ENV_CONTENT}"; then
      log "  ✅ cluster-nodes.env written and coordinator started (nodes: ${ADDRS})"
      return 0
    fi

    warn "  SSH attempt ${ATTEMPT} failed — coordinator host: ${COORD_HOST}"
  done

  warn "  ✖ Could not deliver cluster-nodes.env after ${MAX_ATTEMPTS} attempts."
  warn "    SSH into ${COORD_HOST} and run:"
  warn "      /bin/sudo /bin/mkdir -p /etc/juno"
  warn "      /bin/sudo /usr/bin/tee /etc/juno/cluster-nodes.env <<'EOF'"
  while IFS= read -r line; do
    warn "      ${line}"
  done <<< "${ENV_CONTENT}"
  warn "      EOF"
  warn "      /bin/sudo /bin/systemctl start juno-coordinator"
}

# ── SCP .LORA FILE TO ALL NODES ───────────────────────────────
# Called after bootstrap completes and before the coordinator starts.
# Copies the local .lora file to /opt/juno/models/ on every node, then
# patches /etc/juno/node.env so JUNO_LORA_PLAY_PATH points to the correct
# remote absolute path and restarts juno-node.service so the new value
# takes effect before the coordinator sends any loadShard RPCs.
_scp_lora_to_nodes() {
  [[ -n "${LORA_PLAY_PATH:-}" ]] || return 0

  if [[ ! -f "$LORA_PLAY_PATH" ]]; then
    warn "  ✖ --lora-play file not found: '${LORA_PLAY_PATH}'"
    warn "    (path is resolved at parse time relative to your working directory)"
    warn "    Skipping LoRA deployment — nodes will run the base model."
    return 0
  fi

  local LORA_BASENAME
  LORA_BASENAME="$(basename "$LORA_PLAY_PATH")"
  local REMOTE_LORA_PATH="/opt/juno/models/${LORA_BASENAME}"
  local SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes \
-o ServerAliveInterval=10 -o ServerAliveCountMax=3 -i $SSH_KEY_FILE"

  log "  Deploying LoRA adapter to all nodes: ${LORA_BASENAME}"
  log "  [TRACE] local source  : ${LORA_PLAY_PATH}"
  log "  [TRACE] remote target : ${REMOTE_LORA_PATH}"

  # ── Per node: SCP → stop juno-node → patch node.env → start juno-node ───────
  # We stop BEFORE patching so the new start always picks up the correct env.
  # juno-node starts in ~2s (gRPC bind only; model loads on the coordinator's
  # loadShard RPC). So the synchronous stop+start completes quickly per node.
  for ID in "${INSTANCE_IDS[@]}"; do
    local IP="${INSTANCE_IPS[$ID]}"
    log "  [TRACE] deploying to ${IP}"

    # 1. Upload .lora file
    scp $SSH_OPTS "$LORA_PLAY_PATH" "ubuntu@${IP}:/tmp/${LORA_BASENAME}" 2>/dev/null \
      || { warn "  SCP failed to ${IP} — node will run without LoRA"; continue; }
    log "  [TRACE] ${IP}: SCP done"

    # 2. Stop juno-node (synchronous — ensures old process is gone before we patch env)
    ssh $SSH_OPTS "ubuntu@${IP}" "/bin/sudo /bin/systemctl stop juno-node" 2>/dev/null \
      || warn "  [TRACE] ${IP}: juno-node was not running (ok for first deploy)"
    log "  [TRACE] ${IP}: juno-node stopped"

    # 3. Move .lora into place + patch node.env
    ssh $SSH_OPTS "ubuntu@${IP}" \
      "/bin/sudo /bin/mv /tmp/${LORA_BASENAME} ${REMOTE_LORA_PATH} \
    && /bin/sudo /bin/chmod 644 ${REMOTE_LORA_PATH} \
    && if /bin/sudo /bin/grep -q '^JUNO_LORA_PLAY_PATH=' /etc/juno/node.env 2>/dev/null; then \
         /bin/sudo /bin/sed -i 's|^JUNO_LORA_PLAY_PATH=.*|JUNO_LORA_PLAY_PATH=${REMOTE_LORA_PATH}|' /etc/juno/node.env; \
       else \
         echo 'JUNO_LORA_PLAY_PATH=${REMOTE_LORA_PATH}' | /bin/sudo /usr/bin/tee -a /etc/juno/node.env >/dev/null; \
       fi \
    && echo '[TRACE] node.env JUNO_LORA_PLAY_PATH line:' \
    && /bin/sudo /bin/grep 'JUNO_LORA_PLAY_PATH' /etc/juno/node.env" \
      2>/dev/null \
      && log "  [TRACE] ${IP}: node.env patched" \
      || { warn "  Could not patch node.env on ${IP}"; continue; }

    # 4. Start juno-node (synchronous — returns once the service is active/failed)
    ssh $SSH_OPTS "ubuntu@${IP}" "/bin/sudo /bin/systemctl start juno-node" 2>/dev/null \
      && log "  ✅ ${IP}: juno-node started with LoRA" \
      || warn "  ${IP}: juno-node start failed — check journalctl -u juno-node"
  done

  log "  ✅ LoRA deployment complete — remote path: ${REMOTE_LORA_PATH}"
  log "  [TRACE] LORA_PLAY_PATH updated → ${REMOTE_LORA_PATH}"

  # Update global so _write_cluster_env_and_start_coordinator writes the remote path
  LORA_PLAY_PATH="$REMOTE_LORA_PATH"
}

scan_regions() {
  local VCPUS_PER_NODE="${INSTANCE_VCPUS[$INSTANCE_TYPE]:-2}"
  local VCPUS_NEEDED=$(( NODE_COUNT * VCPUS_PER_NODE ))
  local IS_GPU=false
  [[ "$INSTANCE_TYPE" =~ ^g[0-9]|^p[0-9] ]] && IS_GPU=true

  echo ""
  echo -e "${BOLD}${CYAN}Scanning all AWS regions for '${INSTANCE_TYPE}' availability…${RESET}"
  echo ""

  ALL_REGIONS=$(aws ec2 describe-regions \
    --all-regions \
    --query "Regions[?OptInStatus!='not-opted-in'].RegionName" \
    --output text | tr '\t' '\n' | sort)

  printf "  %-20s  %-12s  %-10s  %s\n" "REGION" "QUOTA/AZs" "AVAILABLE" "STATUS"
  printf "  %-20s  %-12s  %-10s  %s\n" "──────────────────" "────────────" "──────────" "──────"

  for R in $ALL_REGIONS; do
    local AZ_COUNT
    AZ_COUNT=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
      --region "$R" \
      --query "length(InstanceTypeOfferings)" \
      --output text 2>/dev/null || echo "0")
    [[ "$AZ_COUNT" == "None" || -z "$AZ_COUNT" ]] && AZ_COUNT=0

    local STATUS QUOTA_COL="–"
    if [[ "$AZ_COUNT" -eq 0 ]]; then
      STATUS="${DIM}unavailable${RESET}"
    elif $IS_GPU; then
      local QUOTA_RAW QUOTA_INT
      QUOTA_RAW=$(aws service-quotas get-service-quota \
        --service-code ec2 --quota-code L-DB2E81BA --region "$R" \
        --query "Quota.Value" --output text 2>/dev/null || echo "0")
      QUOTA_INT=$(echo "${QUOTA_RAW:-0}" | awk '{printf "%d", $1}')
      QUOTA_COL="${QUOTA_INT} vCPUs"
      if [[ "$QUOTA_INT" -eq 0 ]]; then
        STATUS="${RED}quota=0 (request increase)${RESET}"
      elif [[ "$QUOTA_INT" -lt "$VCPUS_NEEDED" ]]; then
        local MAX_N=$(( QUOTA_INT / VCPUS_PER_NODE ))
        STATUS="${YELLOW}partial — fits ${MAX_N} node(s)${RESET}"
      else
        STATUS="${GREEN}OK — can run ${NODE_COUNT} nodes${RESET}"
      fi
    else
      QUOTA_COL="${AZ_COUNT} AZs"
      STATUS="${GREEN}OK${RESET}"
    fi

    printf "  %-20s  %-12s  %-10s  " "$R" "$QUOTA_COL" "${AZ_COUNT} AZs"
    echo -e "$STATUS"
  done

  echo ""
  echo -e "${GREEN}To use a different region:${RESET}"
  echo "  export AWS_DEFAULT_REGION=<region>"
  echo "  ./launcher.sh juno-deploy.sh setup --instance-type $INSTANCE_TYPE --node-count $NODE_COUNT"
  echo ""
}

# ── WAIT FOR READY + OPEN BROWSER ─────────────────────────────
# Single canonical definition.  Waits for .juno-ready on every node,
# then writes cluster-nodes.env and starts the coordinator.
# Timeout is 1800 s (30 min) — GPU nodes with CUDA+DKMS regularly
# take 17-20 min; 15 min was too short and caused the coordinator to
# receive its env file while the instance was still compiling kernel
# modules, making the SSH write unreliable.
_wait_for_ready_and_open() {
  local CONSOLE_URL="$1"
  local all_ids=("${INSTANCE_IDS[@]}")
  [[ -n "${COORDINATOR_INSTANCE_ID:-}" ]] && all_ids+=("$COORDINATOR_INSTANCE_ID")

  log "Waiting for bootstrap to complete on all instances (~10-20 min for GPU nodes)…"
  local DEADLINE=$(( $(date +%s) + 1800 ))   # 30 min — covers CUDA DKMS build time

  while true; do
    local all_ready=true
    for ID in "${all_ids[@]}"; do
      local IP="${INSTANCE_IPS[$ID]}"
      local READY
      READY=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes \
                  -i "$SSH_KEY_FILE" "ubuntu@$IP" \
                  "[[ -f /opt/juno/.juno-ready ]] && echo yes || echo no" \
                  2>/dev/null || echo "no")
      [[ "$READY" != "yes" ]] && all_ready=false && break
    done

    if $all_ready; then
      log "  ✅ All instances bootstrapped"
      break
    fi
    if [[ $(date +%s) -ge $DEADLINE ]]; then
      warn "Bootstrap timed out after 30 min. Check /var/log/juno-bootstrap.log on each node."
      break
    fi
    printf "  ."
    sleep 15
  done
  echo ""

  # SCP .lora file to nodes and patch node.env (noop if --lora-play not set)
  _scp_lora_to_nodes

  # Write cluster-nodes.env + start coordinator (retries on transient SSH failures)
  _write_cluster_env_and_start_coordinator

  # Wait for coordinator REST to respond
  log "Waiting for coordinator REST server on ${CONSOLE_URL}/v1/cluster/health …"
  local COORD_IP
  if [[ "$COORDINATOR_MODE" == "separate" && -n "${COORDINATOR_INSTANCE_ID:-}" ]]; then
    COORD_IP="${INSTANCE_IPS[$COORDINATOR_INSTANCE_ID]}"
  else
    COORD_IP="${INSTANCE_IPS[${INSTANCE_IDS[0]}]}"
  fi

  local HEALTH_DEADLINE=$(( $(date +%s) + 180 ))
  while [[ $(date +%s) -lt $HEALTH_DEADLINE ]]; do
    if curl -sf "http://${COORD_IP}:${HTTP_PORT}/v1/cluster/health" &>/dev/null; then
      log "  ✅ Coordinator is healthy"
      break
    fi
    printf "  ."
    sleep 5
  done
  echo ""

  log ""
  log "  ╔══════════════════════════════════════════════════════╗"
  log "  ║  Web console  : ${CONSOLE_URL}"
  log "  ║  Health dash  : ${CONSOLE_URL}/health-ui"
  log "  ╚══════════════════════════════════════════════════════╝"
  log ""

  if command -v xdg-open &>/dev/null; then
    xdg-open "$CONSOLE_URL" &>/dev/null &
  elif command -v open &>/dev/null; then
    open "$CONSOLE_URL" &>/dev/null &
  fi
}

# ── ENTRYPOINT ────────────────────────────────────────────────
MODE="${1:-}"
shift || true

case "$MODE" in
  setup)
    parse_options "$@"
    setup
    ;;
  start)
    load_state
    SETUP_TIME=$(date +%s); save_state
    start
    ;;
  stop)
    stop
    ;;
  teardown)
    teardown
    ;;
  status)
    status
    ;;
  scan-regions)
    parse_options "$@"
    scan_regions
    ;;
  "")
    echo -e "${BOLD}Usage:${RESET}  ./launcher.sh juno-deploy.sh <command> [options]"
    echo ""
    echo "  Commands:"
    echo "    setup          Provision cluster, deploy Juno, open web console"
    echo "    start          Start a stopped cluster and re-enter dashboard"
    echo "    stop           Stop all instances (EBS + key pair retained)"
    echo "    teardown       Terminate everything — no lingering AWS costs"
    echo "    status         Show instance states and IPs without entering dashboard"
    echo "    scan-regions   List regions where the chosen instance type is available"
    echo ""
    echo "  Setup options:"
    echo "    --instance-type TYPE    g4dn.xlarge (default), m7i-flex.large, t3.medium, …"
    echo "    --node-count N          Number of inference nodes (default: 3)"
    echo "    --git REF               Git branch, tag, or commit to deploy (default: main)"
    echo "    --coordinator node1     Co-locate coordinator on node 1 (default, free)"
    echo "    --coordinator separate  Extra t3.medium coordinator instance"
    echo "    --model-url URL         Model to download (default: TinyLlama Q4_K_M)"
    echo "    --ptype pipeline|tensor Parallelism type (default: pipeline)"
    echo "    --dtype FLOAT16|FLOAT32 Activation dtype (default: FLOAT16)"
    echo "    --jfr DURATION          Enable JFR on all nodes + coordinator (e.g. 5m 30s 1h)"
    echo "                            Metrics are gathered and printed on Ctrl+C exit"
    echo "    --lora-play PATH        Apply a .lora adapter file at inference on every node"
    echo "                            The file must exist at PATH on each node instance"
    echo ""
    echo "  Examples:"
    echo "    # 3-node GPU cluster, TinyLlama, coordinator on node1"
    echo "    ./launcher.sh juno-deploy.sh setup"
    echo ""
    echo "    # 5-node CPU cluster with separate coordinator"
    echo "    ./launcher.sh juno-deploy.sh setup \\"
    echo "      --instance-type m7i-flex.large \\"
    echo "      --node-count 5 \\"
    echo "      --coordinator separate"
    echo ""
    echo "    # 4-node GPU cluster, custom model, tensor-parallel"
    echo "    ./launcher.sh juno-deploy.sh setup \\"
    echo "      --instance-type g4dn.2xlarge \\"
    echo "      --node-count 4 \\"
    echo "      --ptype tensor \\"
    echo "      --model-url https://huggingface.co/.../Phi-3.5-mini.Q4_K_M.gguf"
    echo ""
    exit 1
    ;;
  *)
    die "Unknown command: '$MODE'. Run without arguments for usage."
    ;;
esac