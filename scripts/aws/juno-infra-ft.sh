#!/bin/bash
# =============================================================
#  juno-infra-ft.sh — 3-node Juno CPU cluster, Free-Tier edition
#
#  Instance  : m7i-flex.large  (2 vCPU / 8 GB RAM, Intel Sapphire Rapids)
#              → AMX hardware acceleration for INT8/BF16 ML inference
#              → auto-falls back to c7i-flex.large if unavailable in region
#  Networking: all 3 nodes pinned to the same AZ/subnet (ENA, Up to 12.5 Gbps).
#              NOTE: m7i-flex.large does NOT support cluster placement groups —
#                    AWS rejects the launch. Same-AZ routing already keeps
#                    inter-node traffic on the same physical switching tier and
#                    saturates the 12.5 Gbps ENA envelope. No placement group needed.
#              NOTE: 25 Gbps requires non-flex M7i ≥ xlarge (not free-tier eligible).
#  No GPU quota checks, no NVIDIA/CUDA drivers.
#
#  Usage:
#    ./launcher.sh juno-infra-ft.sh setup
#    ./launcher.sh juno-infra-ft.sh teardown
#    ./launcher.sh juno-infra-ft.sh stop
#    ./launcher.sh juno-infra-ft.sh start
#    ./launcher.sh juno-infra-ft.sh scan-regions
#
#  IAM: no extra permissions needed beyond the base JunoAppPolicy.
# =============================================================

set -euo pipefail

# Force C locale for bc and printf so decimal separator is always '.'
# (systems with LC_NUMERIC=bg_BG or similar use ',' which breaks printf %f)
export LC_NUMERIC=C

# ── CONFIG ────────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-eu-north-1}"
INSTANCE_TYPE_PRIMARY="m7i-flex.large"   # 2 vCPU / 8 GB / AMX — preferred
INSTANCE_TYPE_FALLBACK="c7i-flex.large"  # 2 vCPU / 4 GB        — fallback
INSTANCE_TYPE=""                         # resolved at runtime
NODE_COUNT=3
KEY_NAME="juno-ft-key"
SG_NAME="juno-ft-sg"
STATE_FILE="${HOME}/.juno-ft-aws-state"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
# Verify prices at: https://aws.amazon.com/ec2/pricing/on-demand/
PRICE_M7I=0.0479   # m7i-flex.large on-demand, eu-north-1 ($/hr per node)
PRICE_C7I=0.0408   # c7i-flex.large on-demand, eu-north-1 ($/hr per node)
PRICE_PER_HOUR=0   # set after instance type is resolved
SSH_KEY_FILE="${HOME}/.ssh/juno-ft-key.pem"
MONITOR_INTERVAL=30

# ── COLOURS ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── HELPERS ───────────────────────────────────────────────────
log()  { echo -e "${GREEN}[juno-ft]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
die()  { echo -e "${RED}[error]${RESET} $*"; exit 1; }

require_cmd() { command -v "$1" &>/dev/null || die "'$1' not found — run: $2"; }

save_state() {
  {
    echo "INSTANCE_IDS=\"${INSTANCE_IDS[*]}\""
    echo "SG_ID=\"$SG_ID\""
    echo "INSTANCE_TYPE_SAVED=\"$INSTANCE_TYPE\""
    echo "SETUP_TIME=\"$SETUP_TIME\""
  } > "$STATE_FILE"
  log "State saved → $STATE_FILE"
}

load_state() {
  [[ -f "$STATE_FILE" ]] || die "No state file found. Run setup first."
  source "$STATE_FILE"
  read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"
  INSTANCE_TYPE="${INSTANCE_TYPE_SAVED:-$INSTANCE_TYPE_PRIMARY}"
  if [[ "$INSTANCE_TYPE" == "$INSTANCE_TYPE_PRIMARY" ]]; then
    PRICE_PER_HOUR=$PRICE_M7I
  else
    PRICE_PER_HOUR=$PRICE_C7I
  fi
}

instance_state() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null || echo "unknown"
}

# ── STOP ──────────────────────────────────────────────────────
stop() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Stopping Juno CPU cluster (instances preserved)…"
  log "═══════════════════════════════════════════"

  load_state 2>/dev/null || { warn "Nothing to stop."; return 0; }

  local running=()
  for ID in "${INSTANCE_IDS[@]}"; do
    local state
    state=$(instance_state "$ID")
    if [[ "$state" == "running" ]]; then
      running+=("$ID")
    else
      warn "  $ID is already in state '$state' — skipping"
    fi
  done

  if [[ ${#running[@]} -eq 0 ]]; then
    warn "  No running instances to stop."
    return 0
  fi

  log "Stopping ${#running[@]} instance(s): ${running[*]}"
  aws ec2 stop-instances \
    --instance-ids "${running[@]}" \
    --region "$REGION" --output text &>/dev/null \
    && log "  ✅ Stop initiated" \
    || warn "  Could not stop instances (check console)"

  log "Waiting for instances to reach 'stopped' state…"
  aws ec2 wait instance-stopped \
    --instance-ids "${running[@]}" \
    --region "$REGION" 2>/dev/null \
    && log "  ✅ Instances stopped" \
    || warn "  Wait timed out — check console"

  log ""
  log "  ✅  Cluster stopped. EBS volumes and key pair retained."
  log "      Run 'start' to resume, or 'teardown' to delete everything."
  log "═══════════════════════════════════════════"
}

# ── START ─────────────────────────────────────────────────────
start() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Starting stopped Juno CPU cluster…"
  log "═══════════════════════════════════════════"

  load_state

  local stopped=()
  for ID in "${INSTANCE_IDS[@]}"; do
    local state
    state=$(instance_state "$ID")
    case "$state" in
      stopped)   stopped+=("$ID") ;;
      running)   warn "  $ID is already running — skipping" ;;
      *)         warn "  $ID is in state '$state' — skipping" ;;
    esac
  done

  if [[ ${#stopped[@]} -eq 0 ]]; then
    warn "  No stopped instances to start."
    return 0
  fi

  log "Starting ${#stopped[@]} instance(s): ${stopped[*]}"
  aws ec2 start-instances \
    --instance-ids "${stopped[@]}" \
    --region "$REGION" --output text &>/dev/null \
    && log "  ✅ Start initiated" \
    || die "  Could not start instances"

  log "Waiting for instances to reach 'running' state…"
  aws ec2 wait instance-running \
    --instance-ids "${stopped[@]}" \
    --region "$REGION" \
    && log "  ✅ Instances running" \
    || die "  Instances did not reach 'running' in time"

  _fetch_ips_and_monitor
}

# ── TEARDOWN ──────────────────────────────────────────────────
teardown() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Tearing down Juno CPU cluster…"
  log "═══════════════════════════════════════════"

  load_state 2>/dev/null || { warn "Nothing to tear down."; return 0; }

  if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    log "Terminating ${#INSTANCE_IDS[@]} instances: ${INSTANCE_IDS[*]}"
    aws ec2 terminate-instances \
      --instance-ids "${INSTANCE_IDS[@]}" \
      --region "$REGION" --output text &>/dev/null \
      && log "  ✅ Termination initiated" \
      || warn "  Could not terminate (may already be gone)"

    log "Waiting for instances to terminate…"
    aws ec2 wait instance-terminated \
      --instance-ids "${INSTANCE_IDS[@]}" \
      --region "$REGION" 2>/dev/null \
      && log "  ✅ Instances terminated" \
      || warn "  Wait timed out — check console"
  fi

  # Delete security group (retry — needs instances to be fully gone first)
  if [[ -n "${SG_ID:-}" ]]; then
    log "Deleting security group: $SG_ID"
    for i in 1 2 3 4 5; do
      if aws ec2 delete-security-group --group-id "$SG_ID" \
           --region "$REGION" &>/dev/null; then
        log "  ✅ Security group deleted"
        break
      fi
      warn "  Retry $i/5 — waiting 10s for SG to be released…"
      sleep 10
    done
  fi

  log "Deleting key pair: $KEY_NAME"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" &>/dev/null \
    && log "  ✅ Key pair deleted" \
    || warn "  Key pair already gone"

  [[ -f "$SSH_KEY_FILE" ]] && rm -f "$SSH_KEY_FILE" \
    && log "  ✅ Local key file removed"

  rm -f "$STATE_FILE"

  log ""
  log "  ✅  Cluster fully torn down. No lingering AWS costs."
  log "═══════════════════════════════════════════"
}

# ── FETCH IPs & ENTER MONITOR ─────────────────────────────────
_fetch_ips_and_monitor() {
  declare -gA INSTANCE_IPS
  for ID in "${INSTANCE_IDS[@]}"; do
    local IP
    IP=$(aws ec2 describe-instances \
      --instance-ids "$ID" \
      --region "$REGION" \
      --query "Reservations[0].Instances[0].PublicIpAddress" \
      --output text)
    INSTANCE_IPS[$ID]="$IP"
  done

  log ""
  log "═══════════════════════════════════════════"
  log "  Juno CPU cluster is UP  [same-AZ: ENA, Up to 12.5 Gbps]"
  for ID in "${INSTANCE_IDS[@]}"; do
    log "  • $ID  →  ${INSTANCE_IPS[$ID]}"
  done
  log ""
  log "  Bootstrap running in background on each node."
  log "  SSH: ssh -i $SSH_KEY_FILE ubuntu@<IP>"
  log "  Bootstrap log: /var/log/juno-bootstrap.log"
  log "═══════════════════════════════════════════"
  log ""

  trap '_on_exit' EXIT INT TERM

  log "Entering monitoring dashboard (Ctrl+C to exit & auto-stop)…"
  sleep 5

  while true; do
    clear
    local NOW ELAPSED HOURS MINS SECS ELAPSED_HRS TOTAL_COST HOURLY_RATE
    NOW=$(date +%s)
    ELAPSED=$(( NOW - SETUP_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    SECS=$(( ELAPSED % 60 ))
    ELAPSED_HRS=$(echo "scale=4; $ELAPSED / 3600" | bc)
    TOTAL_COST=$(echo "scale=4; $NODE_COUNT * $PRICE_PER_HOUR * $ELAPSED_HRS" | bc)
    HOURLY_RATE=$(echo "scale=4; $NODE_COUNT * $PRICE_PER_HOUR" | bc)

    echo -e "${BOLD}${CYAN}"
    echo "  ╔════════════════════════════════════════════════════╗"
    echo "  ║          JUNO CPU CLUSTER MONITOR (FT)             ║"
    echo -e "  ╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    printf "  ${BOLD}Instance type  :${RESET}  %s  (same-AZ, ENA, Up to 12.5 Gbps)\n" \
      "$INSTANCE_TYPE"
    printf "  ${BOLD}Session uptime :${RESET}  %02d:%02d:%02d\n" $HOURS $MINS $SECS
    printf "  ${BOLD}Est. cost      :${RESET}  \$%.4f  (\$%.4f/hr × %d nodes)\n" \
      "$TOTAL_COST" "$PRICE_PER_HOUR" "$NODE_COUNT"
    printf "  ${BOLD}Hourly burn    :${RESET}  \$%.4f/hr\n" "$HOURLY_RATE"
    echo ""
    echo -e "  ${BOLD}Nodes:${RESET}"

    local IDX=1
    for ID in "${INSTANCE_IDS[@]}"; do
      local IP="${INSTANCE_IPS[$ID]}"
      local SYS_STATUS CPU_INFO MEM_INFO NODE_INFO

      SYS_STATUS=$(aws ec2 describe-instance-status \
        --instance-ids "$ID" \
        --region "$REGION" \
        --query "InstanceStatuses[0].InstanceStatus.Status" \
        --output text 2>/dev/null || echo "unknown")

      # Try to pull CPU + RAM via SSH (non-blocking, 3s timeout)
      NODE_INFO="connecting…"
      if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
             -o BatchMode=yes -i "$SSH_KEY_FILE" \
             "ubuntu@$IP" \
             "echo cpu:\$(grep 'cpu ' /proc/stat | awk '{u=\$2+\$4; t=\$2+\$3+\$4+\$5; printf \"%.0f\", (u/t)*100}')% mem:\$(free -m | awk '/^Mem/{printf \"%d/%dMB\",\$3,\$2}')" \
             >/tmp/juno_ft_node_$IDX.txt 2>/dev/null; then
        NODE_INFO=$(cat /tmp/juno_ft_node_$IDX.txt)
      else
        # Check bootstrap progress
        local READY
        READY=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
                    -o BatchMode=yes -i "$SSH_KEY_FILE" \
                    "ubuntu@$IP" \
                    "[[ -f /opt/juno/.juno-ready ]] && echo ready || tail -1 /var/log/juno-bootstrap.log 2>/dev/null || echo bootstrapping" \
                    2>/dev/null || echo "unreachable")
        NODE_INFO="[$READY]"
      fi

      printf "  ${CYAN}  Node %d${RESET}  %-22s  sys:%-10s  %s\n" \
        "$IDX" "$IP" "$SYS_STATUS" "$NODE_INFO"
      (( IDX++ ))
    done

    echo ""
    echo -e "  ${DIM}Refreshing every ${MONITOR_INTERVAL}s — press Ctrl+C to exit & auto-stop${RESET}"
    echo ""

    sleep "$MONITOR_INTERVAL"
  done
}

_on_exit() {
  echo ""
  warn "Caught exit signal — stopping cluster (instances preserved)…"
  trap - EXIT INT TERM
  stop
}

# ── SETUP ─────────────────────────────────────────────────────
setup() {

  require_cmd aws  "pip install awscli"
  require_cmd jq   "sudo apt install jq"
  require_cmd bc   "sudo apt install bc"
  require_cmd ssh  "sudo apt install openssh-client"

  # ── Resolve instance type ───────────────────────────────────
  log "Checking instance type availability in $REGION…"
  for TRY in "$INSTANCE_TYPE_PRIMARY" "$INSTANCE_TYPE_FALLBACK"; do
    COUNT=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=$TRY" \
      --region "$REGION" \
      --query "length(InstanceTypeOfferings)" \
      --output text 2>/dev/null || echo "0")
    if [[ "$COUNT" != "0" && "$COUNT" != "None" ]]; then
      INSTANCE_TYPE="$TRY"
      break
    fi
    warn "  $TRY not available in $REGION — trying fallback…"
  done

  if [[ -z "$INSTANCE_TYPE" ]]; then
    die "Neither $INSTANCE_TYPE_PRIMARY nor $INSTANCE_TYPE_FALLBACK is available in $REGION.
     Run: ./launcher.sh juno-infra-ft.sh scan-regions"
  fi

  if [[ "$INSTANCE_TYPE" == "$INSTANCE_TYPE_PRIMARY" ]]; then
    PRICE_PER_HOUR=$PRICE_M7I
    log "  ✅ Using $INSTANCE_TYPE  (2 vCPU / 8 GB / AMX,  Up to 12.5 Gbps)"
  else
    PRICE_PER_HOUR=$PRICE_C7I
    warn "  Using fallback $INSTANCE_TYPE  (2 vCPU / 4 GB,  Up to 12.5 Gbps)"
    warn "  m7i-flex.large was unavailable in $REGION — c7i-flex.large has half the RAM."
  fi

  # ── Existing state check ────────────────────────────────────
  if [[ -f "$STATE_FILE" ]]; then
    source "$STATE_FILE"
    read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"
    INSTANCE_TYPE="${INSTANCE_TYPE_SAVED:-$INSTANCE_TYPE}"

    local all_running=true all_stopped=true has_valid=false
    for ID in "${INSTANCE_IDS[@]}"; do
      local state
      state=$(instance_state "$ID")
      [[ "$state" == "running" ]]  || all_running=false
      [[ "$state" == "stopped" ]]  || all_stopped=false
      [[ "$state" =~ ^(running|stopped|pending|stopping)$ ]] && has_valid=true
    done

    if $has_valid && $all_running; then
      log "Cluster is already running. Nothing to do."
      for ID in "${INSTANCE_IDS[@]}"; do
        local IP
        IP=$(aws ec2 describe-instances \
          --instance-ids "$ID" --region "$REGION" \
          --query "Reservations[0].Instances[0].PublicIpAddress" \
          --output text 2>/dev/null || echo "unknown")
        log "  • $ID  →  $IP"
      done
      return 0
    fi

    if $has_valid && $all_stopped; then
      log "Found stopped cluster — resuming…"
      SETUP_TIME=$(date +%s)
      save_state
      start
      return 0
    fi

    warn "State file exists but instances are in inconsistent/terminated state."
    warn "Proceeding with fresh setup…"
    rm -f "$STATE_FILE"
    INSTANCE_IDS=()
    SG_ID=""
  fi

  log "Account: $(aws sts get-caller-identity --query Arn --output text)"
  log "Region : $REGION"
  log "Nodes  : $NODE_COUNT × $INSTANCE_TYPE  (same-AZ, ENA)"
  echo ""

  # ── Resolve AMI ─────────────────────────────────────────────
  log "Resolving Ubuntu 22.04 LTS AMI (Canonical)…"
  AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters \
      "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
      "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null)

  [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]] && \
    die "Could not resolve Ubuntu 22.04 AMI for $REGION"
  log "  OK AMI: $AMI_ID"

  # ── Key pair ────────────────────────────────────────────────
  log "Creating key pair…"
  mkdir -p "$(dirname "$SSH_KEY_FILE")"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" &>/dev/null || true
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query "KeyMaterial" \
    --output text > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"
  log "  ✅ Key saved → $SSH_KEY_FILE"

  # ── Resolve subnet — must pick ONE AZ and stick with it so all
  #    instances share the same physical switching tier              ──
  log "Resolving subnet for $INSTANCE_TYPE in $REGION…"
  SUBNET_ID=""
  VPC_ID=""
  CHOSEN_AZ=""

  AZ_LIST=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
    --region "$REGION" \
    --query "InstanceTypeOfferings[].Location" \
    --output text 2>/dev/null | tr '\t' '\n')

  [[ -z "$AZ_LIST" ]] && die "$INSTANCE_TYPE is not available in any AZ in $REGION."

  for AZ in $AZ_LIST; do
    ROW=$(aws ec2 describe-subnets \
      --region "$REGION" \
      --filters "Name=availabilityZone,Values=$AZ" "Name=defaultForAz,Values=true" \
      --query "Subnets[0].[SubnetId,VpcId]" \
      --output text 2>/dev/null)
    SN=$(echo "$ROW" | awk '{print $1}')
    VP=$(echo "$ROW" | awk '{print $2}')
    if [[ -n "$SN" && "$SN" != "None" ]]; then
      SUBNET_ID="$SN"
      VPC_ID="$VP"
      CHOSEN_AZ="$AZ"
      log "  OK Subnet: $SUBNET_ID  VPC: $VPC_ID  (AZ: $AZ — all nodes pinned here)"
      break
    else
      warn "  No default subnet in $AZ — trying next AZ…"
    fi
  done

  [[ -z "$SUBNET_ID" ]] && die "No default subnet found for $INSTANCE_TYPE in $REGION."

  # ── Security group ───────────────────────────────────────────
  log "Creating security group…"
  OLD_SG=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)
  [[ "$OLD_SG" != "None" && -n "$OLD_SG" ]] && \
    aws ec2 delete-security-group --group-id "$OLD_SG" --region "$REGION" &>/dev/null || true

  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Juno CPU 3-node cluster" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query "GroupId" \
    --output text)

  MY_IP=$(curl -s https://checkip.amazonaws.com)

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=SSH}]" \
      "IpProtocol=tcp,FromPort=19092,ToPort=19094,IpRanges=[{CidrIp=0.0.0.0/0,Description=Juno-gRPC}]" \
      "IpProtocol=tcp,FromPort=5701,ToPort=5701,IpRanges=[{CidrIp=0.0.0.0/0,Description=Hazelcast}]" \
      "IpProtocol=tcp,FromPort=7000,ToPort=7000,IpRanges=[{CidrIp=0.0.0.0/0,Description=Juno-REST}]" \
      &>/dev/null
  log "  OK Security group: $SG_ID in VPC $VPC_ID  (SSH from $MY_IP)"

  # ── User-data bootstrap (CPU-only, no NVIDIA/CUDA) ───────────
  USER_DATA=$(cat <<'BOOTSTRAP'
#!/bin/bash
exec > /var/log/juno-bootstrap.log 2>&1
set -e

apt-get update -qq
apt-get install -y -qq \
  openjdk-21-jdk maven git wget curl \
  numactl linux-tools-common linux-tools-generic \
  net-tools iperf3 htop

# ── Enable AMX on m7i-flex.large ─────────────────────────────
# The Linux kernel enables AMX tile registers via arch_prctl(ARCH_REQ_XCOMP_PERM).
# JVM and native code that calls into llama.cpp/GGML with AMX support will
# pick this up automatically; no manual configuration needed here.

# ── Clone & build Juno (CPU mode) ───────────────────────────
git clone https://github.com/ml-cab/juno /opt/juno
cd /opt/juno

# Build with CPU-only profile; skip CUDA check
mvn clean package -DskipTests -q -Pcpu-only 2>/dev/null \
  || mvn clean package -DskipTests -q

# ── Download TinyLlama model ─────────────────────────────────
mkdir -p /opt/juno/models
wget -q "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
     -O /opt/juno/models/TinyLlama.gguf

# Mark ready
touch /opt/juno/.juno-ready
echo "JUNO CPU BOOTSTRAP COMPLETE"
BOOTSTRAP
  )

  USER_DATA_B64=$(printf '%s' "$USER_DATA" | base64 -w 0)

  # ── Launch instances (one at a time) ────────────────────────
  log "Launching $NODE_COUNT × $INSTANCE_TYPE (same AZ: $CHOSEN_AZ)…"
  INSTANCE_IDS=()
  for (( i=1; i<=NODE_COUNT; i++ )); do
    log "  Launching node $i / $NODE_COUNT…"
    LAUNCH_ERR_FILE=$(mktemp)
    LAUNCH_OUTPUT=$(aws ec2 run-instances \
      --image-id "$AMI_ID" \
      --instance-type "$INSTANCE_TYPE" \
      --count 1 \
      --key-name "$KEY_NAME" \
      --subnet-id "$SUBNET_ID" \
      --security-group-ids "$SG_ID" \
      --associate-public-ip-address \
      --user-data "$USER_DATA_B64" \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}" \
      --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=juno-ft-node-${i}},{Key=Project,Value=juno-ft}]" \
      --region "$REGION" \
      --output json 2>"$LAUNCH_ERR_FILE") || {
        echo ""
        echo -e "${RED}  Node $i launch FAILED. Full AWS error:${RESET}"
        cat "$LAUNCH_ERR_FILE" | sed 's/^/    /'
        rm -f "$LAUNCH_ERR_FILE"
        if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
          warn "  Terminating ${#INSTANCE_IDS[@]} already-launched node(s)…"
          aws ec2 terminate-instances --instance-ids "${INSTANCE_IDS[@]}" \
            --region "$REGION" --output text &>/dev/null || true
        fi
        die "Launch aborted. See error above."
      }
    rm -f "$LAUNCH_ERR_FILE"
    IID=$(echo "$LAUNCH_OUTPUT" | jq -r '.Instances[0].InstanceId')
    INSTANCE_IDS+=("$IID")
    log "  OK Node $i: $IID"
  done
  log "  ✅ All nodes launched: ${INSTANCE_IDS[*]}"

  SETUP_TIME=$(date +%s)
  save_state

  log "Waiting for instances to reach 'running' state…"
  aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --region "$REGION"
  log "  ✅ All instances running"

  _fetch_ips_and_monitor
}

# ── SCAN REGIONS ─────────────────────────────────────────────
scan_regions() {
  echo ""
  echo -e "${BOLD}${CYAN}Scanning all AWS regions for free-tier CPU instance availability…${RESET}"
  echo ""

  ALL_REGIONS=$(aws ec2 describe-regions \
    --all-regions \
    --query "Regions[?OptInStatus!='not-opted-in'].RegionName" \
    --output text | tr '\t' '\n' | sort)

  printf "  %-20s  %-20s  %-20s\n" "REGION" "m7i-flex.large" "c7i-flex.large"
  printf "  %-20s  %-20s  %-20s\n" "──────────────────" "──────────────────" "──────────────────"

  for R in $ALL_REGIONS; do
    M7_AZS=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=m7i-flex.large" \
      --region "$R" \
      --query "length(InstanceTypeOfferings)" \
      --output text 2>/dev/null || echo "0")
    C7_AZS=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=c7i-flex.large" \
      --region "$R" \
      --query "length(InstanceTypeOfferings)" \
      --output text 2>/dev/null || echo "0")

    [[ "$M7_AZS" == "None" || -z "$M7_AZS" ]] && M7_AZS=0
    [[ "$C7_AZS" == "None" || -z "$C7_AZS" ]] && C7_AZS=0

    M7_STATUS="${DIM}unavailable${RESET}"
    C7_STATUS="${DIM}unavailable${RESET}"
    [[ "$M7_AZS" -gt 0 ]] && M7_STATUS="${GREEN}✓ ${M7_AZS} AZ(s)${RESET}"
    [[ "$C7_AZS" -gt 0 ]] && C7_STATUS="${GREEN}✓ ${C7_AZS} AZ(s)${RESET}"

    printf "  %-20s  " "$R"
    echo -en "$M7_STATUS"
    printf "%-20s  " ""  # spacing after color codes (approx)
    echo -e "$C7_STATUS"
  done

  echo ""
  echo -e "${GREEN}To use a different region:${RESET}"
  echo "  export AWS_DEFAULT_REGION=<region>"
  echo "  ./launcher.sh juno-infra-ft.sh setup"
  echo ""
}

# ── ENTRYPOINT ────────────────────────────────────────────────
MODE="${1:-}"

case "$MODE" in
  setup)
    setup
    ;;
  teardown)
    teardown
    ;;
  stop)
    stop
    ;;
  start)
    load_state
    SETUP_TIME=$(date +%s)
    save_state
    start
    ;;
  scan-regions)
    scan_regions
    ;;
  *)
    echo -e "${BOLD}Usage:${RESET}  ./launcher.sh juno-infra-ft.sh <setup|teardown|stop|start|scan-regions>"
    echo ""
    echo "  setup        — launch $NODE_COUNT × m7i-flex.large (or c7i-flex.large fallback)"
    echo "                 all in the same AZ for max inter-node bandwidth, bootstrap"
    echo "                 Juno CPU mode, hold console with live CPU/RAM dashboard."
    echo "                 Ctrl+C auto-stops."
    echo "  stop         — stop all running instances (EBS preserved)."
    echo "  start        — start previously stopped instances and re-enter dashboard."
    echo "  teardown     — terminate everything: instances, SG, key pair."
    echo "  scan-regions — list all regions with m7i-flex.large / c7i-flex.large availability."
    echo ""
    echo "  Instance     : m7i-flex.large  (2 vCPU / 8 GB / Intel Sapphire Rapids / AMX)"
    echo "  Networking   : same-AZ ENA — Up to 12.5 Gbps inter-node"
    echo "                 (m7i-flex.large does not support cluster placement groups)"
    echo "  Free-tier    : 750 hrs/mo shared across all 3 nodes (~250 hrs each)"
    exit 1
    ;;
esac
