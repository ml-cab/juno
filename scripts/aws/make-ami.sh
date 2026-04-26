#!/bin/bash
# =============================================================
#  make-ami.sh — Build a Juno golden AMI for a given base OS + instance type.
#
#  The resulting AMI has JDK 25, Maven, and (for GPU instances) CUDA 12.3 +
#  nvidia-open drivers pre-installed.  Every juno-deploy.sh bootstrap that uses
#  a golden AMI skips ~15-20 min of package installation and DKMS compilation.
#
#  Usage:
#    ./launcher.sh make-ami.sh --instance-type TYPE [--base "Ubuntu 22.04 LTS"] [--region REGION]
#    ./make-ami.sh --instance-type g4dn.2xlarge
#
#  AMI name: Juno-golden-<base-slug>_<instance-type>
#    e.g.    Juno-golden-Ubuntu-22-04-LTS_g4dn.2xlarge
#
#  Exits 0 and prints the AMI ID on stdout in every success path
#  (whether the AMI already existed or was freshly baked).
#  All progress messages go to stderr.
# =============================================================

set -euo pipefail
export LC_NUMERIC=C

# ── DEFAULTS ──────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
BASE="Ubuntu 22.04 LTS"
INSTANCE_TYPE=""
# When --key-name / --key-file are supplied by juno-deploy.sh the script reuses
# the deploy key so the operator can SSH into the bake instance for tracing.
# When running standalone the script falls back to a fresh ephemeral key.
EXTERNAL_KEY_NAME=""
EXTERNAL_KEY_FILE=""

# ── COLOURS (stderr only) ────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

inf()  { echo -e "${GREEN}[make-ami]${RESET} $*" >&2; }
warn() { echo -e "${YELLOW}[make-ami]${RESET} $*" >&2; }
die()  { echo -e "${RED}[make-ami]${RESET} $*" >&2; exit 1; }

# ── ARGUMENT PARSING ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)          BASE="$2";                shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2";       shift 2 ;;
    --region)        REGION="$2"; export AWS_DEFAULT_REGION="$2"; shift 2 ;;
    --key-name)      EXTERNAL_KEY_NAME="$2";   shift 2 ;;
    --key-file)      EXTERNAL_KEY_FILE="$2";   shift 2 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -n "$INSTANCE_TYPE" ]] || die "--instance-type is required"

# ── AMI NAME ──────────────────────────────────────────────────
# Dots in the base string (e.g. "22.04") become dashes so the name is
# safe as an EC2 resource name and unambiguous in shell variables.
BASE_SLUG=$(echo "$BASE" | sed 's/[ .]/-/g')
AMI_NAME="Juno-golden-${BASE_SLUG}_${INSTANCE_TYPE}"

# ── GPU vs CPU DETECTION ──────────────────────────────────────
IS_GPU=false
[[ "$INSTANCE_TYPE" =~ ^g[0-9]|^p[0-9] ]] && IS_GPU=true

# For GPU baking we launch the smallest instance of the same family so the
# DKMS kernel modules and CUDA toolkit are built on matching silicon and kernel.
# A g4dn.xlarge AMI boots cleanly on g4dn.2xlarge / g4dn.4xlarge etc.
_smallest_in_family() {
  case "${1%%.*}" in
    g4dn) echo "g4dn.xlarge" ;;
    g5)   echo "g5.xlarge"   ;;
    g6)   echo "g6.xlarge"   ;;
    g6e)  echo "g6e.xlarge"  ;;
    p3)   echo "p3.2xlarge"  ;;
    p4d)  echo "p4d.24xlarge";;
    p5)   echo "p5.48xlarge" ;;
    *)    echo "$1"           ;;
  esac
}

if $IS_GPU; then
  BAKE_INSTANCE="$(_smallest_in_family "$INSTANCE_TYPE")"
  inf "GPU instance family — baking on ${BAKE_INSTANCE} (smallest in family)"
else
  BAKE_INSTANCE="t3.medium"
  inf "CPU instance — baking on ${BAKE_INSTANCE}"
fi

# ── CHECK FOR EXISTING AMI ────────────────────────────────────
inf "Checking for existing AMI: ${AMI_NAME} in ${REGION}…"
EXISTING=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners self \
  --filters "Name=name,Values=${AMI_NAME}" "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text 2>/dev/null || echo "")
if [[ -n "$EXISTING" && "$EXISTING" != "None" ]]; then
  inf "  Found existing AMI: ${EXISTING}"
  echo "$EXISTING"
  exit 0
fi

inf "AMI not found — proceeding with bake…"

# ── RESOLVE BASE UBUNTU AMI ───────────────────────────────────
inf "Resolving ${BASE} base AMI…"
BASE_AMI=$(aws ec2 describe-images \
  --region "$REGION" --owners 099720109477 \
  --filters \
    "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text 2>/dev/null || echo "")
[[ -z "$BASE_AMI" || "$BASE_AMI" == "None" ]] && die "Could not resolve ${BASE} AMI in ${REGION}"
inf "  Base AMI: ${BASE_AMI}"

# ── EPHEMERAL INFRA HANDLES ───────────────────────────────────
BAKE_KEY_NAME=""
BAKE_KEY_FILE=""
BAKE_KEY_EXTERNAL=false
BAKE_SG_ID=""
BAKE_INSTANCE_ID=""

_cleanup() {
  inf "Cleaning up ephemeral bake resources…"
  if [[ -n "$BAKE_INSTANCE_ID" ]]; then
    aws ec2 terminate-instances --instance-ids "$BAKE_INSTANCE_ID" \
      --region "$REGION" --output text &>/dev/null || true
    # Fire-and-forget: no wait.  GPU instances under active DKMS teardown can
    # take 10+ min to reach 'terminated'; blocking here is what caused the
    # "hang forever on cleanup" symptom.  SG deletion already has || true so
    # a still-attached ENI is non-fatal — AWS releases it once the instance
    # actually terminates on its own.
  fi
  if [[ -n "$BAKE_SG_ID" ]]; then
    aws ec2 delete-security-group --group-id "$BAKE_SG_ID" \
      --region "$REGION" &>/dev/null || true
  fi
  # Only delete the key pair when we created it ourselves (standalone mode).
  if [[ "${BAKE_KEY_EXTERNAL:-false}" == "false" ]]; then
    aws ec2 delete-key-pair --key-name "$BAKE_KEY_NAME" \
      --region "$REGION" &>/dev/null || true
    rm -f "$BAKE_KEY_FILE"
  fi
}
trap _cleanup EXIT

# ── KEY PAIR ──────────────────────────────────────────────────
if [[ -n "$EXTERNAL_KEY_NAME" && -n "$EXTERNAL_KEY_FILE" ]]; then
  # Reuse the deploy key supplied by juno-deploy.sh — enables SSH tracing.
  BAKE_KEY_NAME="$EXTERNAL_KEY_NAME"
  BAKE_KEY_FILE="$EXTERNAL_KEY_FILE"
  BAKE_KEY_EXTERNAL=true
  inf "Using external key pair: ${BAKE_KEY_NAME} (${BAKE_KEY_FILE})"
  [[ -f "$BAKE_KEY_FILE" ]] || die "External key file not found: ${BAKE_KEY_FILE}"
else
  # Standalone invocation — create a short-lived ephemeral key.
  BAKE_KEY_NAME="juno-bake-$(date +%s)"
  BAKE_KEY_FILE="/tmp/${BAKE_KEY_NAME}.pem"
  BAKE_KEY_EXTERNAL=false
  inf "Creating ephemeral key pair…"
  aws ec2 create-key-pair --key-name "$BAKE_KEY_NAME" --region "$REGION" \
    --query "KeyMaterial" --output text > "$BAKE_KEY_FILE"
  chmod 600 "$BAKE_KEY_FILE"
fi

# ── SECURITY GROUP ────────────────────────────────────────────
inf "Creating ephemeral security group…"
MY_IP=$(curl -sf https://checkip.amazonaws.com)

DEFAULT_VPC=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")
[[ -z "$DEFAULT_VPC" || "$DEFAULT_VPC" == "None" ]] && die "No default VPC in ${REGION}"

BAKE_SG_NAME="juno-bake-$(date +%s)"
BAKE_SG_ID=$(aws ec2 create-security-group \
  --group-name "$BAKE_SG_NAME" \
  --description "Juno AMI bake (ephemeral)" \
  --vpc-id "$DEFAULT_VPC" \
  --region "$REGION" \
  --query "GroupId" --output text)
aws ec2 authorize-security-group-ingress \
  --group-id "$BAKE_SG_ID" --region "$REGION" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" &>/dev/null
inf "  SG: ${BAKE_SG_ID}  (SSH from ${MY_IP})"

# ── RESOLVE SUBNET ────────────────────────────────────────────
# For GPU instances pick a subnet in an AZ that actually offers the instance type.
_resolve_bake_subnet() {
  if $IS_GPU; then
    local AZ
    AZ=$(aws ec2 describe-instance-type-offerings \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=${BAKE_INSTANCE}" \
      --region "$REGION" \
      --query "InstanceTypeOfferings[0].Location" \
      --output text 2>/dev/null || echo "")
    [[ -z "$AZ" || "$AZ" == "None" ]] && die "${BAKE_INSTANCE} is not available in ${REGION}"
    local SN
    SN=$(aws ec2 describe-subnets --region "$REGION" \
      --filters "Name=availabilityZone,Values=${AZ}" "Name=defaultForAz,Values=true" \
      --query "Subnets[0].SubnetId" --output text 2>/dev/null || echo "")
    [[ -z "$SN" || "$SN" == "None" ]] && die "No default subnet in AZ ${AZ}"
    echo "$SN"
  else
    aws ec2 describe-subnets --region "$REGION" \
      --filters "Name=vpc-id,Values=${DEFAULT_VPC}" "Name=defaultForAz,Values=true" \
      --query "Subnets[0].SubnetId" --output text 2>/dev/null
  fi
}

BAKE_SUBNET=$(_resolve_bake_subnet)
[[ -z "$BAKE_SUBNET" || "$BAKE_SUBNET" == "None" ]] && die "Could not resolve a subnet for ${BAKE_INSTANCE}"
inf "  Subnet: ${BAKE_SUBNET}"

# ── LAUNCH BAKE INSTANCE ──────────────────────────────────────
inf "Launching bake instance (${BAKE_INSTANCE})…"
LAUNCH_OUT=$(aws ec2 run-instances \
  --image-id "$BASE_AMI" \
  --instance-type "$BAKE_INSTANCE" \
  --count 1 \
  --key-name "$BAKE_KEY_NAME" \
  --subnet-id "$BAKE_SUBNET" \
  --security-group-ids "$BAKE_SG_ID" \
  --associate-public-ip-address \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=juno-bake},{Key=Project,Value=juno}]" \
  --region "$REGION" \
  --output json)
BAKE_INSTANCE_ID=$(echo "$LAUNCH_OUT" | jq -r '.Instances[0].InstanceId')
inf "  Instance: ${BAKE_INSTANCE_ID}"

# ── WAIT FOR RUNNING ──────────────────────────────────────────
inf "Waiting for instance running…"
aws ec2 wait instance-running --instance-ids "$BAKE_INSTANCE_ID" --region "$REGION"
BAKE_IP=$(aws ec2 describe-instances \
  --instance-ids "$BAKE_INSTANCE_ID" --region "$REGION" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
inf "  Public IP: ${BAKE_IP}"

# ── WAIT FOR SSH ──────────────────────────────────────────────
inf "Waiting for SSH…"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes -i ${BAKE_KEY_FILE}"
SSH_DEADLINE=$(( $(date +%s) + 300 ))
while [[ $(date +%s) -lt $SSH_DEADLINE ]]; do
  if ssh $SSH_OPTS "ubuntu@${BAKE_IP}" "true" &>/dev/null; then
    inf "  SSH ready"
    break
  fi
  printf "." >&2
  sleep 10
done
echo "" >&2
# Final check — die if SSH never came up.
ssh $SSH_OPTS "ubuntu@${BAKE_IP}" "true" || die "SSH did not become available within 5 min"

# ── INSTALLATION ──────────────────────────────────────────────
if $IS_GPU; then
  inf "Installing JDK 25 + Maven + CUDA 12.3 on ${BAKE_IP} (~20-40 min — DKMS build)…"
  inf "  SSH access while baking: ssh -i ${BAKE_KEY_FILE} ubuntu@${BAKE_IP}"
  # Redirect to stderr: this heredoc's stdout would otherwise be captured by
  # AMI_ID=$(bash make-ami.sh ...) in juno-deploy.sh, corrupting AMI_ID with
  # the full bake log instead of just the bare ami-... id.
  ssh $SSH_OPTS "ubuntu@${BAKE_IP}" 'sudo bash -s' >&2 <<'INSTALL'
set -euo pipefail
# Suppress perl/apt locale warnings caused by the client forwarding LC_* vars
# (e.g. bg_BG.UTF-8) that are not installed on this Ubuntu server.
export LC_ALL=C LANG=C LANGUAGE=C
export DEBIAN_FRONTEND=noninteractive
# cloud-init runs package updates in the background on first boot.
# Proceeding without this wait causes apt to hang on the dpkg lock
# — the root cause of the previous "hang forever" behaviour.
echo "[bake] Waiting for cloud-init to finish…"
cloud-init status --wait 2>/dev/null || true

# Kill unattended-upgrades immediately — do NOT use systemctl stop.
# stop waits for any in-progress apt transaction to finish (minutes).
# kill sends SIGKILL so we proceed right away.
systemctl kill --signal=SIGKILL unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true

# Release any stale locks left behind.
rm -f /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/cache/apt/archives/lock \
      /var/lib/apt/lists/lock
dpkg --configure -a 2>/dev/null || true

echo "[bake] Installing base packages…"
apt-get update -qq
apt-get install -y -qq \
  openjdk-25-jdk maven git wget curl jq bc \
  numactl net-tools htop pciutils lsof

# ── CUDA kernel headers ───────────────────────────────────────
KVER=$(uname -r)
echo "[bake] Kernel: ${KVER}"
apt-get install -y -qq gcc-12
# Only install the aws-flavoured headers package.
# The stripped variant (e.g. linux-headers-5.15.0-1052) does not exist for
# EC2 kernels and would cause apt to abort with a missing-package error.
apt-get install -y -qq \
  "linux-headers-${KVER}" \
  "linux-modules-extra-${KVER}" \
  software-properties-common

DISTRO="ubuntu2204"
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
  -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-mark unhold $(apt-mark showhold 2>/dev/null) 2>/dev/null || true
apt-get -f install -y -qq
apt-get update -qq

# DKMS compilation takes 20-40 min.  Run WITHOUT -qq so stdout/stderr
# proves the build is progressing rather than appearing to hang.
echo "[bake] Starting CUDA/DKMS install — takes 20-40 min, do not interrupt…"
apt-get install -y \
  -o Dpkg::Options::="--force-confdef" \
  -o Dpkg::Options::="--force-confold" \
  --no-install-recommends nvidia-open cuda-toolkit-12-3 || {
    echo "=== DKMS make.log ===" >&2
    cat /var/lib/dkms/nvidia/*/build/make.log 2>/dev/null >&2 || true
    exit 1
  }

echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /etc/environment
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment
echo "[bake] CUDA install complete."
INSTALL

else
  inf "Installing JDK 25 + Maven on ${BAKE_IP}…"
  ssh $SSH_OPTS "ubuntu@${BAKE_IP}" 'sudo bash -s' >&2 <<'INSTALL'
set -euo pipefail
export LC_ALL=C LANG=C LANGUAGE=C
export DEBIAN_FRONTEND=noninteractive

# Wait for cloud-init — same reasoning as GPU path above.
echo "[bake] Waiting for cloud-init to finish…"
cloud-init status --wait 2>/dev/null || true

systemctl kill --signal=SIGKILL unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/cache/apt/archives/lock \
      /var/lib/apt/lists/lock
dpkg --configure -a 2>/dev/null || true

echo "[bake] Installing base packages…"
apt-get update -qq
apt-get install -y -qq \
  openjdk-25-jdk maven git wget curl jq bc \
  numactl net-tools htop pciutils lsof
echo "[bake] Install complete."
INSTALL
fi

# ── VERIFY ────────────────────────────────────────────────────
# Non-login SSH shells have a minimal PATH that may omit /usr/bin.
# Explicitly set a full PATH and use awk instead of head (head -1 is the
# command that was not found, which caused the EXIT trap to fire before
# create-image was ever called — the root cause of "AMI never recorded").
_RENV='export LC_ALL=C LANG=C PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
inf "Verifying installation…"
JAVA_VER=$(ssh $SSH_OPTS "ubuntu@${BAKE_IP}" "${_RENV}; java -version 2>&1 | awk 'NR==1'")
MVN_VER=$(ssh  $SSH_OPTS "ubuntu@${BAKE_IP}" "${_RENV}; mvn  -version 2>&1 | awk 'NR==1'")
inf "  java : ${JAVA_VER}"
inf "  mvn  : ${MVN_VER}"

if $IS_GPU; then
  # nvidia-smi requires the kernel modules to be loaded; they auto-load on
  # first use after DKMS install.  A brief modprobe ensures they are present.
  NVID_OUT=$(ssh $SSH_OPTS "ubuntu@${BAKE_IP}" \
    "${_RENV}; sudo modprobe nvidia 2>/dev/null || true; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1")
  inf "  nvidia-smi: ${NVID_OUT}"
  # Fail fast if nvidia-smi reports no devices — the AMI would be unusable.
  echo "$NVID_OUT" | grep -qi "mib\|gib" || \
    die "nvidia-smi did not report VRAM — CUDA install may have failed"
fi

# ── CREATE AMI ────────────────────────────────────────────────
inf "Creating AMI: ${AMI_NAME}…"
GPU_SUFFIX=""
$IS_GPU && GPU_SUFFIX=" + CUDA 12.3 + nvidia-open"
CREATED_AMI=$(aws ec2 create-image \
  --instance-id "$BAKE_INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "Juno golden AMI: ${BASE} + JDK 25 + Maven${GPU_SUFFIX}" \
  --no-reboot \
  --region "$REGION" \
  --query "ImageId" --output text)
inf "  AMI ID: ${CREATED_AMI} — waiting for available state (5-15 min)…"

# ── WAIT FOR AMI AVAILABLE ────────────────────────────────────
# aws ec2 wait image-available polls every 30 s up to 40 attempts (20 min).
# GPU AMIs with large root volumes can take up to 15 min.
AMI_DEADLINE=$(( $(date +%s) + 2700 ))  # 45 min
while [[ $(date +%s) -lt $AMI_DEADLINE ]]; do
  STATE=$(aws ec2 describe-images --image-ids "$CREATED_AMI" --region "$REGION" \
    --query "Images[0].State" --output text 2>/dev/null || echo "pending")
  [[ "$STATE" == "available" ]] && break
  [[ "$STATE" == "failed" ]] && die "AMI ${CREATED_AMI} entered failed state"
  sleep 30
done
[[ "$STATE" == "available" ]] || die "AMI ${CREATED_AMI} did not reach 'available' within 45 min"
inf "  AMI available: ${CREATED_AMI}"

# _cleanup runs via EXIT trap — terminates bake instance and deletes SG + key.
echo "$CREATED_AMI"