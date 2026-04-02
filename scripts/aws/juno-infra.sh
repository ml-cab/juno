#!/bin/bash
# =============================================================
#  juno-infra.sh — Manage 3-node Juno GPU cluster on AWS
#  Usage:
#    ./launcher.sh juno-infra.sh setup
#    ./launcher.sh juno-infra.sh teardown
#    ./launcher.sh juno-infra.sh stop
#    ./launcher.sh juno-infra.sh start
#
#  On setup/start: holds the console, shows live VRAM / cost / uptime.
#  Ctrl+C or 'q' → auto-stop before exit.
# =============================================================

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-eu-north-1}"
INSTANCE_TYPE="g4dn.xlarge"
NODE_COUNT=3
KEY_NAME="juno-key"
SG_NAME="juno-sg"
STATE_FILE="${HOME}/.juno-aws-state"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
PRICE_PER_HOUR=0.526   # g4dn.xlarge on-demand, eu-north-1
SSH_KEY_FILE="${HOME}/.ssh/juno-key.pem"
MONITOR_INTERVAL=30    # seconds between dashboard refresh

# ── COLOURS ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── HELPERS ───────────────────────────────────────────────────
log()  { echo -e "${GREEN}[juno]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
die()  { echo -e "${RED}[error]${RESET} $*"; exit 1; }

require_cmd() { command -v "$1" &>/dev/null || die "'$1' not found — run: $2"; }

save_state() {
  echo "INSTANCE_IDS=\"${INSTANCE_IDS[*]}\"" > "$STATE_FILE"
  echo "SG_ID=\"$SG_ID\""                   >> "$STATE_FILE"
  echo "SETUP_TIME=\"$SETUP_TIME\""          >> "$STATE_FILE"
  log "State saved → $STATE_FILE"
}

load_state() {
  [[ -f "$STATE_FILE" ]] || die "No state file found. Run setup first."
  source "$STATE_FILE"
  read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"
}

# Returns the AWS state of an instance: running, stopped, terminated, etc.
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
  log "  Stopping Juno cluster (instances preserved)…"
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
  log "  Starting stopped Juno cluster…"
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

  # Refresh IPs (they change after stop/start)
  _fetch_ips_and_monitor
}

# ── TEARDOWN ──────────────────────────────────────────────────
teardown() {
  echo ""
  log "═══════════════════════════════════════════"
  log "  Tearing down Juno cluster…"
  log "═══════════════════════════════════════════"

  load_state 2>/dev/null || { warn "Nothing to tear down."; return 0; }

  # Terminate instances
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

  # Delete security group (retry a few times — AWS needs instance termination to propagate)
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

  # Delete key pair
  log "Deleting key pair: $KEY_NAME"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" &>/dev/null \
    && log "  ✅ Key pair deleted" \
    || warn "  Key pair already gone"

  # Clean local key file
  [[ -f "$SSH_KEY_FILE" ]] && rm -f "$SSH_KEY_FILE" \
    && log "  ✅ Local key file removed"

  rm -f "$STATE_FILE"

  log ""
  log "  ✅  Cluster fully torn down. No lingering AWS costs."
  log "═══════════════════════════════════════════"
}

# ── FETCH IPs & ENTER MONITOR (shared by setup and start) ─────
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
  log "  Juno cluster is UP"
  for ID in "${INSTANCE_IDS[@]}"; do
    log "  • $ID  →  ${INSTANCE_IPS[$ID]}"
  done
  log ""
  log "  Bootstrap running in background on each node."
  log "  SSH: ssh -i $SSH_KEY_FILE ubuntu@<IP>"
  log "  Bootstrap log: /var/log/juno-bootstrap.log"
  log "═══════════════════════════════════════════"
  log ""

  # Register stop (not teardown) on script exit
  trap '_on_exit' EXIT INT TERM

  # ── Live monitoring loop ───────────────────────────────────
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
    HOURLY_RATE=$(echo "scale=3; $NODE_COUNT * $PRICE_PER_HOUR" | bc)

    echo -e "${BOLD}${CYAN}"
    echo "  ╔════════════════════════════════════════════════════╗"
    echo "  ║             JUNO CLUSTER MONITOR                   ║"
    echo "  ╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    printf "  ${BOLD}Session uptime :${RESET}  %02d:%02d:%02d\n" $HOURS $MINS $SECS
    printf "  ${BOLD}Est. cost      :${RESET}  \$%.4f  (\$%.3f/hr × %d nodes)\n" \
      "$TOTAL_COST" "$PRICE_PER_HOUR" "$NODE_COUNT"
    printf "  ${BOLD}Hourly burn    :${RESET}  \$%.3f/hr\n" "$HOURLY_RATE"
    echo ""
    echo -e "  ${BOLD}Nodes:${RESET}"

    local IDX=1
    for ID in "${INSTANCE_IDS[@]}"; do
      local IP="${INSTANCE_IPS[$ID]}"
      local STATUS VRAM_INFO
      STATUS=$(aws ec2 describe-instance-status \
        --instance-ids "$ID" \
        --region "$REGION" \
        --query "InstanceStatuses[0].InstanceStatus.Status" \
        --output text 2>/dev/null || echo "unknown")

      # Try to get VRAM via SSH (non-blocking, 3s timeout)
      VRAM_INFO="connecting…"
      if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
             -o BatchMode=yes -i "$SSH_KEY_FILE" \
             "ubuntu@$IP" \
             "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
              --format=csv,noheader,nounits 2>/dev/null | head -1" \
             >/tmp/juno_gpu_$IDX.txt 2>/dev/null; then
        read USED TOTAL GPU_UTIL < <(cat /tmp/juno_gpu_$IDX.txt | tr ',' ' ')
        if [[ -n "$USED" && -n "$TOTAL" ]]; then
          local PCT BAR FILLED
          PCT=$(echo "scale=1; $USED * 100 / $TOTAL" | bc)
          VRAM_INFO="VRAM ${USED}/${TOTAL} MiB (${PCT}%)  GPU util: ${GPU_UTIL}%"
          # Draw a mini bar
          FILLED=$(( USED * 20 / TOTAL ))
          BAR=""
          for ((b=0; b<20; b++)); do
            [[ $b -lt $FILLED ]] && BAR+="█" || BAR+="░"
          done
          VRAM_INFO="$VRAM_INFO  [${BAR}]"
        fi
      else
        # Check if bootstrap is still running
        local READY
        READY=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
                    -o BatchMode=yes -i "$SSH_KEY_FILE" \
                    "ubuntu@$IP" \
                    "[[ -f /opt/juno/.juno-ready ]] && echo ready || echo bootstrapping" \
                    2>/dev/null || echo "unreachable")
        VRAM_INFO="[$READY]  (nvidia-smi unavailable)"
      fi

      printf "  ${CYAN}  Node %d${RESET}  %-22s  sys: %-10s  %s\n" \
        "$IDX" "$IP" "$STATUS" "$VRAM_INFO"
      (( IDX++ ))
    done

    echo ""
    echo -e "  ${DIM}Refreshing every ${MONITOR_INTERVAL}s — press Ctrl+C to exit & auto-stop${RESET}"
    echo ""

    sleep "$MONITOR_INTERVAL"
  done
}

# Trap handler: stop (not teardown) on exit
_on_exit() {
  echo ""
  warn "Caught exit signal — stopping cluster (instances preserved)…"
  # Disarm trap to avoid re-entrancy
  trap - EXIT INT TERM
  stop
}

# ── SETUP ─────────────────────────────────────────────────────
setup() {

  # ── Pre-flight checks ───────────────────────────────────────
  require_cmd aws   "pip install awscli"
  require_cmd jq    "sudo apt install jq"
  require_cmd bc    "sudo apt install bc"
  require_cmd ssh   "sudo apt install openssh-client"

  # ── GPU quota check ─────────────────────────────────────────
  # g4dn.xlarge = 4 vCPUs; quota L-DB2E81BA covers all G & VT on-demand instances
  local VCPUS_PER_NODE=4
  local VCPUS_NEEDED=$(( NODE_COUNT * VCPUS_PER_NODE ))

  log "Checking EC2 quota: Running On-Demand G and VT instances…"
  local QUOTA_VALUE
  QUOTA_VALUE=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-DB2E81BA \
    --region "$REGION" \
    --query "Quota.Value" \
    --output text 2>/dev/null || echo "")

  if [[ -z "$QUOTA_VALUE" || "$QUOTA_VALUE" == "None" ]]; then
    warn "  Could not read quota -- Service Quotas API unavailable or missing IAM permission."
    warn "  Continuing with NODE_COUNT=${NODE_COUNT}; AWS will reject the launch if quota is insufficient."
  else
    # Quota is returned as a float (e.g. "0.0" or "32.0") -- strip decimal with awk
    local QUOTA_INT
    QUOTA_INT=$(echo "$QUOTA_VALUE" | awk '{printf "%d", $1}')

    if [[ "$QUOTA_INT" -eq 0 ]]; then
      echo ""
      echo -e "${RED}  x  GPU quota is 0 vCPUs in ${REGION} -- cannot launch any instances.${RESET}"
      echo -e "${RED}     Request an increase at: https://console.aws.amazon.com/servicequotas/${RESET}"
      echo -e "${RED}     EC2 -> Running On-Demand G and VT instances (L-DB2E81BA)${RESET}"
      echo -e "${RED}     Minimum needed: ${VCPUS_NEEDED} vCPUs (${NODE_COUNT} x ${INSTANCE_TYPE} x ${VCPUS_PER_NODE} vCPUs)${RESET}"
      echo ""
      exit 1
    elif [[ "$QUOTA_INT" -lt "$VCPUS_NEEDED" ]]; then
      # Fit as many nodes as the quota allows
      local MAX_NODES=$(( QUOTA_INT / VCPUS_PER_NODE ))
      echo ""
      warn "  Quota is ${QUOTA_INT} vCPUs -- not enough for ${NODE_COUNT} nodes (need ${VCPUS_NEEDED} vCPUs)."
      warn "  Adjusting NODE_COUNT: ${NODE_COUNT} -> ${MAX_NODES} (using all available quota)."
      warn "  To run the full ${NODE_COUNT}-node cluster, request a quota increase to >= ${VCPUS_NEEDED} vCPUs:"
      warn "    https://console.aws.amazon.com/servicequotas/"
      warn "    EC2 -> Running On-Demand G and VT instances (L-DB2E81BA)"
      echo ""
      NODE_COUNT=$MAX_NODES
      VCPUS_NEEDED=$(( NODE_COUNT * VCPUS_PER_NODE ))
    else
      log "  OK Quota: ${QUOTA_INT} vCPUs available, ${VCPUS_NEEDED} needed for ${NODE_COUNT} nodes"
    fi
  fi

  # ── Smart entry: check existing state ───────────────────────
  if [[ -f "$STATE_FILE" ]]; then
    source "$STATE_FILE"
    read -ra INSTANCE_IDS <<< "$INSTANCE_IDS"

    # Determine live states for all known instances
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
      log "  • Use 'stop' to pause, 'teardown' to delete."
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
      log "Found stopped cluster — resuming with 'start' instead of reprovisioning…"
      SETUP_TIME=$(date +%s)
      save_state
      start
      return 0
    fi

    # Partial / mixed / terminated states — warn and fall through to fresh setup
    warn "State file exists but instances are in an inconsistent or terminated state."
    warn "Proceeding with fresh setup…"
    rm -f "$STATE_FILE"
    INSTANCE_IDS=()
    SG_ID=""
  fi

  log "Account: $(aws sts get-caller-identity --query Arn --output text)"
  log "Region : $REGION"
  log "Nodes  : $NODE_COUNT × $INSTANCE_TYPE"
  echo ""

  # ── Resolve plain Ubuntu 22.04 LTS AMI (Canonical) ─────────
  # We deliberately skip the AWS Deep Learning AMI here: it is a Marketplace
  # product that requires accepting subscription terms via the console before
  # programmatic launch. Launching it without an accepted subscription triggers
  # the misleading "not eligible for Free Tier / InvalidParameterCombination"
  # error. Plain Ubuntu + manual NVIDIA driver install is cleaner and has no
  # Marketplace dependency.
  log "Resolving Ubuntu 22.04 LTS AMI (Canonical)..."
  AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters \
      "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
      "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null)

  [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]] && die "Could not resolve Ubuntu 22.04 AMI for $REGION"
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

  # ── Resolve subnet first — VPC ID needed for the security group ──
  log "Resolving subnet for $INSTANCE_TYPE in $REGION..."
  SUBNET_ID=""
  VPC_ID=""
  AZ_LIST=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
    --region "$REGION" \
    --query "InstanceTypeOfferings[].Location" \
    --output text 2>/dev/null | tr '\t' '\n')

  if [[ -z "$AZ_LIST" ]]; then
    die "$INSTANCE_TYPE is not available in any AZ in $REGION. Run: scan-regions"
  fi

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
      log "  OK Subnet: $SUBNET_ID  VPC: $VPC_ID  (AZ: $AZ)"
      break
    else
      warn "  No default subnet in $AZ -- trying next AZ..."
    fi
  done

  if [[ -z "$SUBNET_ID" || -z "$VPC_ID" ]]; then
    die "No default subnet found in any AZ offering $INSTANCE_TYPE in $REGION.
       Fix: aws ec2 create-default-vpc --region $REGION
       Available AZs: $AZ_LIST"
  fi

  # ── Security group — created in the same VPC as the subnet ────
  log "Creating security group..."
  # Delete old one in the same VPC if it exists
  OLD_SG=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)
  [[ "$OLD_SG" != "None" && -n "$OLD_SG" ]] && \
    aws ec2 delete-security-group --group-id "$OLD_SG" --region "$REGION" &>/dev/null || true

  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Juno 3-node cluster" \
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
  log "  OK Security group: $SG_ID in VPC $VPC_ID (SSH from $MY_IP)"

  # ── User-data bootstrap ─────────────────────────────────────
  # ── User-data bootstrap ─────────────────────────────────────
  USER_DATA=$(cat <<'BOOTSTRAP'
#!/bin/bash
exec > /var/log/juno-bootstrap.log 2>&1
set -e

apt-get update -qq
apt-get install -y -qq openjdk-21-jdk maven git wget curl software-properties-common

# ── NVIDIA driver + CUDA via official repo ───────────────────
# This is the correct path for plain Ubuntu on g4dn (T4 GPU).
# The AWS Deep Learning AMI has Marketplace terms; we avoid it.
apt-get install -y -qq linux-headers-$(uname -r)

# Add CUDA keyring
DISTRO="ubuntu2204"
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
     -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update -qq

# Install driver + CUDA toolkit
apt-get install -y -qq --no-install-recommends cuda-drivers nvidia-utils-535 cuda-toolkit-12-3
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /etc/environment
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment

# ── Clone & build Juno ───────────────────────────────────────
git clone https://github.com/ml-cab/juno /opt/juno
cd /opt/juno
mvn clean package -DskipTests -q

# ── Download TinyLlama model ─────────────────────────────────
mkdir -p /opt/juno/models
wget -q "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
     -O /opt/juno/models/TinyLlama.gguf

# Mark ready
touch /opt/juno/.juno-ready
echo "JUNO BOOTSTRAP COMPLETE"
BOOTSTRAP
  )

  # Base64-encode for the --user-data flag
  USER_DATA_B64=$(printf '%s' "$USER_DATA" | base64 -w 0)

  # ── Launch instances one-by-one ─────────────────────────────
  # AWS new-account restriction: GPU instances must be launched individually.
  # Batching with --count N triggers a misleading "not eligible for Free Tier /
  # InvalidParameterCombination" error even when quota is approved and a
  # dry-run with count=1 passes cleanly.
  log "Launching $NODE_COUNT x $INSTANCE_TYPE (one at a time)..."
  INSTANCE_IDS=()
  for (( i=1; i<=NODE_COUNT; i++ )); do
    log "  Launching node $i / $NODE_COUNT..."
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
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}" \
      --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=juno-node-${i}},{Key=Project,Value=juno}]" \
      --region "$REGION" \
      --output json 2>"$LAUNCH_ERR_FILE") || {
        echo ""
        echo -e "${RED}  Node $i launch FAILED. Full AWS error:${RESET}"
        cat "$LAUNCH_ERR_FILE" | sed 's/^/    /'
        rm -f "$LAUNCH_ERR_FILE"
        if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
          warn "  Terminating ${#INSTANCE_IDS[@]} already-launched node(s)..."
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
  log "  OK All nodes launched: ${INSTANCE_IDS[*]}"

  SETUP_TIME=$(date +%s)
  save_state

  # ── Wait for running state ───────────────────────────────────
  log "Waiting for instances to reach 'running' state…"
  aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --region "$REGION"
  log "  ✅ All instances running"

  _fetch_ips_and_monitor
}

# ── SCAN REGIONS ─────────────────────────────────────────────
scan_regions() {
  local VCPUS_PER_NODE=4
  local VCPUS_NEEDED=$(( NODE_COUNT * VCPUS_PER_NODE ))

  echo ""
  echo -e "${BOLD}${CYAN}Scanning all AWS regions for $INSTANCE_TYPE quota (need >= ${VCPUS_NEEDED} vCPUs)...${RESET}"
  echo ""

  # Fetch all enabled regions
  ALL_REGIONS=$(aws ec2 describe-regions \
    --all-regions \
    --query "Regions[?OptInStatus!='not-opted-in'].RegionName" \
    --output text | tr '\t' '\n' | sort)

  local found_any=false

  printf "  %-20s  %-10s  %-10s  %s\n" "REGION" "QUOTA" "AZ COUNT" "STATUS"
  printf "  %-20s  %-10s  %-10s  %s\n" "──────────────────" "──────────" "──────────" "──────"

  for R in $ALL_REGIONS; do
    # Check quota
    local QUOTA_RAW QUOTA_INT
    QUOTA_RAW=$(aws service-quotas get-service-quota \
      --service-code ec2 \
      --quota-code L-DB2E81BA \
      --region "$R" \
      --query "Quota.Value" \
      --output text 2>/dev/null || echo "0")
    QUOTA_INT=$(echo "${QUOTA_RAW:-0}" | awk '{printf "%d", $1}')

    # Check AZ availability for the instance type
    local AZ_COUNT
    AZ_COUNT=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
      --region "$R" \
      --query "length(InstanceTypeOfferings)" \
      --output text 2>/dev/null || echo "0")
    [[ "$AZ_COUNT" == "None" || -z "$AZ_COUNT" ]] && AZ_COUNT=0

    local STATUS
    if [[ "$AZ_COUNT" -eq 0 ]]; then
      STATUS="${DIM}instance type not available${RESET}"
    elif [[ "$QUOTA_INT" -eq 0 ]]; then
      STATUS="${RED}quota = 0  (request increase)${RESET}"
    elif [[ "$QUOTA_INT" -lt "$VCPUS_NEEDED" ]]; then
      local MAX_NODES=$(( QUOTA_INT / VCPUS_PER_NODE ))
      STATUS="${YELLOW}partial — fits ${MAX_NODES} node(s)${RESET}"
      found_any=true
    else
      STATUS="${GREEN}OK — can run ${NODE_COUNT} nodes${RESET}"
      found_any=true
    fi

    printf "  %-20s  %-10s  %-10s  " "$R" "${QUOTA_INT} vCPUs" "$AZ_COUNT"
    echo -e "$STATUS"
  done

  echo ""
  if $found_any; then
    echo -e "${GREEN}To use a different region, set AWS_DEFAULT_REGION before running setup:${RESET}"
    echo "  export AWS_DEFAULT_REGION=<region>"
    echo "  ./launcher.sh juno-infra.sh setup"
  else
    echo -e "${YELLOW}No region has quota for $INSTANCE_TYPE yet.${RESET}"
    echo "  Request an increase in your preferred region:"
    echo "  aws service-quotas request-service-quota-increase \"
    echo "    --service-code ec2 --quota-code L-DB2E81BA \"
    echo "    --desired-value $VCPUS_NEEDED --region <region>"
  fi
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
    echo -e "${BOLD}Usage:${RESET}  ./launcher.sh juno-infra.sh <setup|teardown|stop|start|scan-regions>"
    echo ""
    echo "  setup        — launch $NODE_COUNT x g4dn.xlarge, bootstrap Juno, hold console"
    echo "                 with live VRAM / cost dashboard. If instances already exist"
    echo "                 and are stopped, resumes them. Ctrl+C auto-stops."
    echo "  stop         — stop all running instances (EBS + key pair preserved)."
    echo "                 Cheapest idle state -- no compute charge while stopped."
    echo "  start        — start previously stopped instances and re-enter dashboard."
    echo "  teardown     — terminate all instances, delete SG & key pair entirely."
    echo "  scan-regions — scan all AWS regions for g4dn.xlarge quota and availability."
    exit 1
    ;;
esac
