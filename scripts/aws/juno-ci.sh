#!/bin/bash
# =============================================================
#  juno-ci.sh — Spin up a tiny Ubuntu Jenkins instance, bake it
#  (or reuse a cached AMI), run the Juno performance test matrix,
#  pull docs/juno_test_matrix.html back to the local repo, then
#  optionally tear down.
#
#  Modes:
#    startup    Provision the CI instance (or reuse an existing one),
#               install Jenkins via juno-jenkins.sh, trigger the
#               "Juno performance tests" job, wait for it, and pull
#               the updated docs/juno_test_matrix.html back here.
#
#    teardown   Terminate the CI instance, delete the key pair and
#               security group, and remove local state.
#
#  Usage (via launcher.sh):
#    ./launcher.sh juno-ci.sh startup   [options]
#    ./launcher.sh juno-ci.sh teardown
#
#  Options (startup only):
#    --instance-type TYPE   EC2 instance type (default: t3.medium)
#    --region REGION        AWS region        (default: $AWS_DEFAULT_REGION or us-east-1)
#    --port PORT            Jenkins HTTP port on the instance (default: 8080)
#    --repo URL             Git remote for Juno (default: https://github.com/ml-cab/juno.git)
#    --git-ref REF          Branch / tag the performance-test job checks out (default: main)
#    --perf-args ARGS       Arguments forwarded to performance-test.sh
#                           (default: --foreground --all)
#    --no-ami-cache         Skip golden AMI lookup; always bake a fresh one.
#    --keep                 Do not tear down after a successful run (useful for debugging).
#    --html-out PATH        Local path where juno_test_matrix.html is written
#                           (default: docs/juno_test_matrix.html relative to repo root)
#
#  Environment:
#    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — set by launcher.sh
#    AWS_DEFAULT_REGION                       — set by launcher.sh
#
#  State file: ~/.juno-ci-state
#    Contains instance ID, security group ID, key pair name, and
#    public IP so teardown does not need any repeated options.
#
#  AMI name:  Juno-CI-Ubuntu-22-04-LTS_<instance-type>
#    The CI AMI has JDK 17 (Jenkins), JDK 25 (Juno), Maven, and
#    the Jenkins installation baked in.  It is reused across runs
#    to avoid the ~5 min package installation on every startup.
#    Pass --no-ami-cache to force a fresh bake.
#
#  Exits 0 on success; all progress to stderr.
# =============================================================

set -euo pipefail
export LC_NUMERIC=C LC_ALL=C LANG=C LANGUAGE=C

# ── LOCATE REPO ROOT ──────────────────────────────────────────
# Works whether called as ./launcher.sh juno-ci.sh or directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── DEFAULTS ──────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="t3.medium"
JENKINS_PORT=8080
JUNO_REPO="https://github.com/ml-cab/juno.git"
JUNO_GIT_REF="main"
PERF_ARGS="--foreground --all"
NO_AMI_CACHE=false
KEEP=false
HTML_OUT="${REPO_ROOT}/docs/juno_test_matrix.html"
LORA_PLAY_LOCAL=""   # local path to .lora file; uploaded to CI instance if set

KEY_NAME="juno-ci-key"
SG_NAME="juno-ci-sg"
SSH_KEY_FILE="${HOME}/.ssh/juno-ci-key.pem"
STATE_FILE="${HOME}/.juno-ci-state"

# ── COLOURS ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

inf()  { echo -e "${GREEN}[juno-ci]${RESET} $*" >&2; }
warn() { echo -e "${YELLOW}[juno-ci]${RESET} $*" >&2; }
die()  { echo -e "${RED}[juno-ci]${RESET} $*" >&2; exit 1; }

# ── ARGUMENT PARSING ──────────────────────────────────────────
MODE="${1:-}"
shift || true

case "$MODE" in
  startup|teardown) ;;
  "")
    echo "Usage: ./launcher.sh juno-ci.sh <startup|teardown> [options]"
    echo ""
    echo "  startup  [--instance-type TYPE] [--region REGION] [--port PORT]"
    echo "           [--repo URL] [--git-ref REF] [--perf-args ARGS]"
    echo "           [--no-ami-cache] [--keep] [--html-out PATH]"
    echo "  teardown"
    exit 1
    ;;
  *) die "Unknown mode: '$MODE'. Use startup or teardown." ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-type) INSTANCE_TYPE="$2";  shift 2 ;;
    --region)        REGION="$2"; export AWS_DEFAULT_REGION="$2"; shift 2 ;;
    --port)          JENKINS_PORT="$2";   shift 2 ;;
    --repo)          JUNO_REPO="$2";      shift 2 ;;
    --git-ref)       JUNO_GIT_REF="$2";   shift 2 ;;
    --perf-args)     PERF_ARGS="$2";      shift 2 ;;
    --lora-play)     LORA_PLAY_LOCAL="$2"; shift 2 ;;
    --no-ami-cache)  NO_AMI_CACHE=true;   shift ;;
    --keep)          KEEP=true;           shift ;;
    --html-out)      HTML_OUT="$2";       shift 2 ;;
    *) die "Unknown option: $1" ;;
  esac
done

# ── HELPERS ───────────────────────────────────────────────────
_require_cmd() {
  command -v "$1" &>/dev/null && return 0
  die "'$1' not found — install it and retry"
}

_instance_state() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null || echo "unknown"
}

_public_ip() {
  aws ec2 describe-instances \
    --instance-ids "$1" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text 2>/dev/null || echo ""
}

# SSH options reused throughout — 30 s connect timeout to handle
# the window where the instance is "running" but sshd is not yet up.
SSH_OPTS="-o ConnectTimeout=30 -o StrictHostKeyChecking=no \
  -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -i ${SSH_KEY_FILE}"

_ssh()  { ssh  $SSH_OPTS "ubuntu@${CI_IP}" "$@"; }
_scp()  { scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
              -o BatchMode=yes -i "${SSH_KEY_FILE}" "$@"; }

# ── STATE PERSISTENCE ─────────────────────────────────────────
_save_state() {
  cat > "$STATE_FILE" <<EOF
CI_INSTANCE_ID="${CI_INSTANCE_ID}"
CI_SG_ID="${CI_SG_ID}"
CI_IP="${CI_IP}"
REGION="${REGION}"
INSTANCE_TYPE="${INSTANCE_TYPE}"
KEY_NAME="${KEY_NAME}"
JENKINS_PORT="${JENKINS_PORT}"
EOF
  inf "State saved -> ${STATE_FILE}"
}

_load_state() {
  [[ -f "$STATE_FILE" ]] || die "No state file found. Run: ./launcher.sh juno-ci.sh startup"
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

# ── AMI RESOLUTION ────────────────────────────────────────────
# Looks for a cached Juno-CI AMI.  Falls back to baking a new one
# by launching a fresh instance, running juno-jenkins.sh on it, then
# calling create-image.  The bake instance is then terminated; the CI
# run itself uses a fresh instance launched from the cached AMI.
#
# AMI baking is a one-time cost (~10 min).  Subsequent startups skip
# straight to instance launch (~90 s) and SSH provisioning.

_ami_name() {
  local BASE_SLUG="Ubuntu-22-04-LTS"
  echo "Juno-CI-${BASE_SLUG}_${INSTANCE_TYPE}"
}

_resolve_base_ami() {
  aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters \
      "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
      "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null || echo ""
}

# Bakes a new CI AMI.
# Launches a temporary t3.medium, runs juno-jenkins.sh on it,
# creates an AMI from the stopped instance, waits for it to be
# available, then terminates the bake instance.
# Prints the AMI ID to stdout; all progress to stderr.
_bake_ci_ami() {
  local AMI_NAME; AMI_NAME="$(_ami_name)"
  inf "Baking CI AMI: ${AMI_NAME} (~10 min)..."

  local BASE_AMI; BASE_AMI="$(_resolve_base_ami)"
  [[ -z "$BASE_AMI" || "$BASE_AMI" == "None" ]] && \
    die "Could not resolve Ubuntu 22.04 LTS base AMI in ${REGION}"
  inf "  Base AMI: ${BASE_AMI}"

  # ── Ephemeral bake state (script-level so _bake_cleanup can read them
  #    after the function returns — bash locals are gone at that point) ──
  _BAKE_KEY_NAME="juno-ci-bake-$(date +%s)"
  _BAKE_KEY_FILE="/tmp/${_BAKE_KEY_NAME}.pem"
  aws ec2 create-key-pair \
    --key-name "$_BAKE_KEY_NAME" \
    --region "$REGION" \
    --query "KeyMaterial" --output text > "$_BAKE_KEY_FILE"
  chmod 600 "$_BAKE_KEY_FILE"

  # ── Ephemeral bake SG ───────────────────────────────────────
  local MY_IP; MY_IP=$(curl -sf https://checkip.amazonaws.com)
  local DEFAULT_VPC; DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")
  [[ -z "$DEFAULT_VPC" || "$DEFAULT_VPC" == "None" ]] && \
    die "No default VPC in ${REGION}"

  local BAKE_SG_NAME="juno-ci-bake-$(date +%s)"
  _BAKE_SG_ID=$(aws ec2 create-security-group \
    --group-name "$BAKE_SG_NAME" \
    --description "Juno CI AMI bake (ephemeral)" \
    --vpc-id "$DEFAULT_VPC" \
    --region "$REGION" \
    --query "GroupId" --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$_BAKE_SG_ID" \
    --region "$REGION" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" &>/dev/null

  # ── Cleanup: _BAKE_CLEANED is script-level (not local) so the trap
  #    can read it after _bake_ci_ami returns. Guard ensures one run.
  _BAKE_ID=""
  _bake_cleanup() {
    [[ "${_BAKE_CLEANED:-false}" == "true" ]] && return 0
    _BAKE_CLEANED=true
    inf "Cleaning up bake resources..."
    [[ -n "${_BAKE_ID:-}" ]] && \
      aws ec2 terminate-instances --instance-ids "$_BAKE_ID" \
        --region "$REGION" --output text &>/dev/null || true
    [[ -n "${_BAKE_SG_ID:-}" ]] && \
      aws ec2 delete-security-group --group-id "$_BAKE_SG_ID" \
        --region "$REGION" &>/dev/null || true
    [[ -n "${_BAKE_KEY_NAME:-}" ]] && \
      aws ec2 delete-key-pair --key-name "$_BAKE_KEY_NAME" \
        --region "$REGION" &>/dev/null || true
    [[ -n "${_BAKE_KEY_FILE:-}" ]] && rm -f "$_BAKE_KEY_FILE"
  }
  trap _bake_cleanup RETURN EXIT

  # ── Launch bake instance (t3.medium — no GPU needed for CI) ─
  local DEFAULT_SUBNET; DEFAULT_SUBNET=$(aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" "Name=defaultForAz,Values=true" \
    --query "Subnets[0].SubnetId" --output text 2>/dev/null || echo "")
  [[ -z "$DEFAULT_SUBNET" || "$DEFAULT_SUBNET" == "None" ]] && \
    die "No default subnet found in ${REGION}"

  local BAKE_OUT; BAKE_OUT=$(aws ec2 run-instances \
    --image-id "$BASE_AMI" \
    --instance-type "t3.medium" \
    --count 1 \
    --key-name "$_BAKE_KEY_NAME" \
    --subnet-id "$DEFAULT_SUBNET" \
    --security-group-ids "$_BAKE_SG_ID" \
    --associate-public-ip-address \
    --block-device-mappings \
      "DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=juno-ci-bake},{Key=Project,Value=juno}]" \
    --region "$REGION" \
    --output json)
  _BAKE_ID=$(echo "$BAKE_OUT" | jq -r '.Instances[0].InstanceId')
  inf "  Bake instance: ${_BAKE_ID}"

  inf "  Waiting for bake instance to reach 'running'..."
  aws ec2 wait instance-running --instance-ids "$_BAKE_ID" --region "$REGION"
  local BAKE_IP; BAKE_IP=$(aws ec2 describe-instances \
    --instance-ids "$_BAKE_ID" --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  inf "  Bake IP: ${BAKE_IP}"

  # ── Wait for SSH on bake instance ───────────────────────────
  inf "  Waiting for SSH..."
  local BAKE_SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    -o BatchMode=yes -i ${_BAKE_KEY_FILE}"
  local SSH_DEADLINE=$(( $(date +%s) + 300 ))
  while [[ $(date +%s) -lt $SSH_DEADLINE ]]; do
    if ssh $BAKE_SSH_OPTS "ubuntu@${BAKE_IP}" "true" &>/dev/null; then
      inf "  SSH ready"
      break
    fi
    printf "." >&2
    sleep 10
  done
  echo "" >&2
  ssh $BAKE_SSH_OPTS "ubuntu@${BAKE_IP}" "true" \
    || die "SSH did not become available on bake instance within 5 min"

  # ── Upload juno-jenkins.sh and run it ───────────────────────
  local JENKINS_SCRIPT="${SCRIPT_DIR}/../ci/juno-jenkins.sh"
  [[ -f "$JENKINS_SCRIPT" ]] || \
    die "juno-jenkins.sh not found at: ${JENKINS_SCRIPT}"

  inf "  Uploading juno-jenkins.sh to bake instance..."
  scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o BatchMode=yes \
    -i "$_BAKE_KEY_FILE" \
    "$JENKINS_SCRIPT" "ubuntu@${BAKE_IP}:/tmp/juno-jenkins.sh"

  inf "  Running juno-jenkins.sh on bake instance (~5 min)..."
  # Run juno-jenkins.sh and capture its exit code explicitly.
  # Plain SSH exit-code propagation through sudo is unreliable on some
  # Ubuntu configurations; embedding __EXIT__:$? in the output is definitive.
  ssh $BAKE_SSH_OPTS "ubuntu@${BAKE_IP}" \
    "chmod +x /tmp/juno-jenkins.sh && \
     { sudo bash /tmp/juno-jenkins.sh \
         --port ${JENKINS_PORT} \
         --repo ${JUNO_REPO} \
         --git-ref ${JUNO_GIT_REF} 2>&1; echo \"__EXIT__:\$?\"; }" 2>&1 \
    | tee /tmp/juno-bake-$$.log \
    | grep -v "^__EXIT__:" >&2
  BAKE_EXIT=$(grep -o '__EXIT__:[0-9]*' /tmp/juno-bake-$$.log 2>/dev/null | tail -1 | cut -d: -f2)
  rm -f /tmp/juno-bake-$$.log
  [[ "${BAKE_EXIT:-1}" == "0" ]] \
    || die "juno-jenkins.sh failed on bake instance (exit ${BAKE_EXIT:-unknown})"

  # ── Stop instance before create-image ───────────────────────
  # Stopping (not just --no-reboot) gives the cleanest filesystem snapshot.
  inf "  Stopping bake instance for image creation..."
  aws ec2 stop-instances --instance-ids "$_BAKE_ID" \
    --region "$REGION" --output text &>/dev/null
  aws ec2 wait instance-stopped --instance-ids "$_BAKE_ID" \
    --region "$REGION"

  # ── Create AMI ───────────────────────────────────────────────
  inf "  Creating AMI: ${AMI_NAME}..."
  local CREATED_AMI; CREATED_AMI=$(aws ec2 create-image \
    --instance-id "$_BAKE_ID" \
    --name "$AMI_NAME" \
    --description "Juno CI: Ubuntu 22.04 + JDK 17 + JDK 25 + Maven + Jenkins LTS" \
    --region "$REGION" \
    --query "ImageId" --output text)
  inf "  AMI ID: ${CREATED_AMI} — waiting for 'available' state (~5 min)..."

  local AMI_DEADLINE=$(( $(date +%s) + 1800 ))
  local AMI_STATE=""
  while [[ $(date +%s) -lt $AMI_DEADLINE ]]; do
    AMI_STATE=$(aws ec2 describe-images \
      --image-ids "$CREATED_AMI" --region "$REGION" \
      --query "Images[0].State" --output text 2>/dev/null || echo "pending")
    [[ "$AMI_STATE" == "available" ]] && break
    [[ "$AMI_STATE" == "failed" ]] && die "AMI ${CREATED_AMI} entered failed state"
    sleep 30
  done
  [[ "$AMI_STATE" == "available" ]] || \
    die "AMI ${CREATED_AMI} did not reach 'available' within 30 min"

  inf "  AMI ready: ${CREATED_AMI}"
  # _bake_cleanup fires via RETURN trap — terminates bake instance, deletes SG + key.
  echo "$CREATED_AMI"
}

# ── RESOLVE CI AMI ────────────────────────────────────────────
# Returns the AMI ID via stdout.  Bakes a new one if none exists
# or --no-ami-cache was requested.
_resolve_ci_ami() {
  local AMI_NAME; AMI_NAME="$(_ami_name)"

  if ! $NO_AMI_CACHE; then
    inf "Checking for existing CI AMI: ${AMI_NAME}..."
    local EXISTING; EXISTING=$(aws ec2 describe-images \
      --region "$REGION" \
      --owners self \
      --filters "Name=name,Values=${AMI_NAME}" "Name=state,Values=available" \
      --query "sort_by(Images, &CreationDate)[-1].ImageId" \
      --output text 2>/dev/null || echo "")
    if [[ -n "$EXISTING" && "$EXISTING" != "None" ]]; then
      inf "  Reusing cached CI AMI: ${EXISTING}"
      echo "$EXISTING"
      return 0
    fi
    inf "  No cached CI AMI found — baking..."
  else
    inf "--no-ami-cache: forcing fresh AMI bake..."
  fi

  _bake_ci_ami
}

# ── JENKINS JOB TRIGGER ───────────────────────────────────────
# Triggers the job via Jenkins REST API (crumb-protected POST),
# waits for it to start, then polls until completion.
# Returns 0 on SUCCESS, 1 on FAILURE/ABORTED.
_trigger_and_wait_job() {
  local JOB_NAME="juno-performance-tests"
  local JENKINS_URL="http://localhost:${JENKINS_PORT}"
  local ADMIN_PASSWORD; ADMIN_PASSWORD=$(_ssh \
    "sudo cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null")
  [[ -n "$ADMIN_PASSWORD" ]] \
    || die "Could not read initialAdminPassword from CI instance"

  inf "Triggering Jenkins job: ${JOB_NAME}..."

  # ── Fetch crumb (CSRF token) ─────────────────────────────────
  local CRUMB_JSON; CRUMB_JSON=$(_ssh "curl -sf \
    --user 'admin:${ADMIN_PASSWORD}' \
    '${JENKINS_URL}/crumbIssuer/api/json' 2>/dev/null || echo '{}'")
  local CRUMB_FIELD; CRUMB_FIELD=$(echo "$CRUMB_JSON" | jq -r '.crumbRequestField // ""')
  local CRUMB_VAL;   CRUMB_VAL=$(echo   "$CRUMB_JSON" | jq -r '.crumb // ""')
  [[ -n "$CRUMB_FIELD" && -n "$CRUMB_VAL" ]] \
    || die "Failed to obtain Jenkins CSRF crumb — check admin credentials"

  # ── Build trigger URL with job parameters ────────────────────
  # Run the trigger as a remote bash script via SSH heredoc to avoid
  # all shell quoting interactions between local and remote shells.
  local QUEUE_LOCATION
  QUEUE_LOCATION=$(ssh $SSH_OPTS "ubuntu@${CI_IP}" bash << REMOTE
set -euo pipefail
PASS="${ADMIN_PASSWORD}"
JENKINS_URL="${JENKINS_URL}"
JOB_NAME="${JOB_NAME}"
PERF_ARGS="${PERF_ARGS}"
GIT_REF="${JUNO_GIT_REF}"

JAR=\$(mktemp)
CRUMB_JSON=\$(curl -sf -c "\$JAR" -b "\$JAR" \\
  --user "admin:\$PASS" \\
  "\$JENKINS_URL/crumbIssuer/api/json" 2>/dev/null || echo '{}')
CF=\$(echo "\$CRUMB_JSON" | jq -r '.crumbRequestField // ""')
CV=\$(echo "\$CRUMB_JSON" | jq -r '.crumb // ""')

ENCODED=\$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "\$PERF_ARGS" 2>/dev/null || printf '%s' "\$PERF_ARGS")
PARAMS="PERF_ARGS=\$ENCODED&GIT_REF=\$GIT_REF"

HDR=\$(mktemp)
curl -sf -c "\$JAR" -b "\$JAR" -D "\$HDR" -X POST \\
  --user "admin:\$PASS" \\
  -H "\$CF: \$CV" \\
  "\$JENKINS_URL/job/\$JOB_NAME/buildWithParameters?\$PARAMS" \\
  -o /dev/null 2>/dev/null || true

awk '/^[Ll]ocation:/{print \$2}' "\$HDR" | tr -d '\r\n'
rm -f "\$JAR" "\$HDR"
REMOTE
)

  [[ -n "$QUEUE_LOCATION" ]]     || die "Jenkins did not return a queue Location for job '${JOB_NAME}' — job may not exist or POST was rejected"
  inf "  Queued: ${QUEUE_LOCATION}"


  # ── Wait for build to leave the queue ───────────────────────
  inf "  Waiting for build to start..."
  local BUILD_URL=""
  local QUEUE_DEADLINE=$(( $(date +%s) + 120 ))
  while [[ -z "$BUILD_URL" && $(date +%s) -lt $QUEUE_DEADLINE ]]; do
    BUILD_URL=$(_ssh "curl -sf \
      --user admin:${ADMIN_PASSWORD} \
      '${QUEUE_LOCATION}api/json' 2>/dev/null \
      | jq -r '.executable.url // \"\"'" || echo "")
    [[ -z "$BUILD_URL" ]] && sleep 5
  done
  [[ -n "$BUILD_URL" ]] || die "Build did not leave Jenkins queue within 2 min"
  inf "  Build URL: ${BUILD_URL}"

  # ── Poll build result ────────────────────────────────────────
  # Performance tests can take hours.  Poll every 60 s with no hard timeout —
  # the operator should cancel the process manually if needed.
  inf "  Polling build result (this can take a long time)..."
  local RESULT=""
  while true; do
    RESULT=$(_ssh "curl -sf \
      --user admin:${ADMIN_PASSWORD} \
      '${BUILD_URL}api/json' 2>/dev/null \
      | jq -r '.result // \"IN_PROGRESS\"'" || echo "UNKNOWN")
    case "$RESULT" in
      SUCCESS)          inf "  Build SUCCEEDED."; return 0 ;;
      FAILURE|ABORTED)  warn "  Build ${RESULT}.";  return 1 ;;
      IN_PROGRESS|null|UNKNOWN)
        printf "  [polling] build in progress...\r" >&2
        sleep 60
        ;;
      *) warn "  Unexpected result: ${RESULT}"; sleep 60 ;;
    esac
  done
}

# ── PULL HTML ARTIFACT ────────────────────────────────────────
# Copies the updated juno_test_matrix.html from the Jenkins workspace
# back to the local repo.
_pull_html_artifact() {
  local REMOTE_HTML="/var/lib/jenkins/workspace/juno-performance-tests/docs/juno_test_matrix.html"
  inf "Pulling updated HTML matrix from CI instance..."
  mkdir -p "$(dirname "${HTML_OUT}")"
  _scp "ubuntu@${CI_IP}:${REMOTE_HTML}" "${HTML_OUT}" \
    || die "Could not scp ${REMOTE_HTML} from ${CI_IP}"
  inf "  HTML matrix written: ${HTML_OUT}"
}

# ── TEARDOWN ──────────────────────────────────────────────────
_teardown() {
  inf "Tearing down Juno CI instance..."
  _load_state

  if [[ -n "${CI_INSTANCE_ID:-}" ]]; then
    local st; st=$(_instance_state "$CI_INSTANCE_ID")
    if [[ "$st" != "terminated" && "$st" != "shutting-down" ]]; then
      inf "  Terminating ${CI_INSTANCE_ID}..."
      aws ec2 terminate-instances --instance-ids "$CI_INSTANCE_ID" \
        --region "$REGION" --output text &>/dev/null || true
      aws ec2 wait instance-terminated --instance-ids "$CI_INSTANCE_ID" \
        --region "$REGION" 2>/dev/null \
        && inf "  Terminated." || warn "  Wait timed out — instance may still be shutting down."
    else
      inf "  Instance ${CI_INSTANCE_ID} already ${st}."
    fi
  fi

  if [[ -n "${CI_SG_ID:-}" ]]; then
    inf "  Deleting security group ${CI_SG_ID}..."
    # Terminate any instances still attached to the SG before deleting it.
    # This handles the case where a failed provisioning run left an instance
    # running that wasn't recorded in the state file.
    local ATTACHED; ATTACHED=$(aws ec2 describe-instances \
      --region "$REGION" \
      --filters \
        "Name=instance.group-id,Values=${CI_SG_ID}" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[].Instances[].InstanceId" \
      --output text 2>/dev/null || echo "")
    if [[ -n "$ATTACHED" && "$ATTACHED" != "None" ]]; then
      warn "  Found instances still using SG: ${ATTACHED} — terminating..."
      aws ec2 terminate-instances \
        --instance-ids $ATTACHED \
        --region "$REGION" --output text &>/dev/null || true
      aws ec2 wait instance-terminated \
        --instance-ids $ATTACHED \
        --region "$REGION" 2>/dev/null || true
    fi
    for i in 1 2 3 4 5; do
      aws ec2 delete-security-group \
        --group-id "$CI_SG_ID" \
        --region "$REGION" &>/dev/null && { inf "  SG deleted."; break; }
      warn "  Retry ${i}/5 — waiting 10 s..."; sleep 10
    done
  fi

  aws ec2 delete-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" &>/dev/null \
    && inf "  Key pair deleted." || true
  [[ -f "$SSH_KEY_FILE" ]] && rm -f "$SSH_KEY_FILE" && inf "  Local key removed."
  rm -f "$STATE_FILE"

  inf "Teardown complete. No lingering AWS costs from the CI instance."
}

# ── STARTUP ───────────────────────────────────────────────────
_startup() {
  _require_cmd aws
  _require_cmd jq
  _require_cmd curl
  _require_cmd ssh
  _require_cmd scp

  inf "Account: $(aws sts get-caller-identity --query Arn --output text)"
  inf "Region:  ${REGION}"
  inf "Type:    ${INSTANCE_TYPE}"
  inf "Port:    ${JENKINS_PORT}"
  inf "Repo:    ${JUNO_REPO}  ref=${JUNO_GIT_REF}"
  inf "Args:    ${PERF_ARGS}"
  echo "" >&2

  # ── Resolve CI AMI ───────────────────────────────────────────
  local CI_AMI; CI_AMI="$(_resolve_ci_ami)"
  [[ -n "$CI_AMI" && "$CI_AMI" != "None" ]] || die "Could not resolve CI AMI"

  # ── Key pair ─────────────────────────────────────────────────
  inf "Creating SSH key pair..."
  mkdir -p "$(dirname "$SSH_KEY_FILE")"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" &>/dev/null || true
  aws ec2 create-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query "KeyMaterial" --output text > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"
  inf "  Key saved -> ${SSH_KEY_FILE}"

  # ── Security group ───────────────────────────────────────────
  inf "Creating security group..."
  local MY_IP; MY_IP=$(curl -sf https://checkip.amazonaws.com)
  local DEFAULT_VPC; DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")
  [[ -z "$DEFAULT_VPC" || "$DEFAULT_VPC" == "None" ]] && \
    die "No default VPC in ${REGION}"

  # Delete stale SG with the same name if it exists, then wait for
  # EC2 to fully propagate the deletion before creating the new one.
  local OLD_SG; OLD_SG=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${DEFAULT_VPC}" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")
  if [[ -n "$OLD_SG" && "$OLD_SG" != "None" ]]; then
    inf "  Deleting stale SG ${OLD_SG}..."
    aws ec2 delete-security-group \
      --group-id "$OLD_SG" --region "$REGION" &>/dev/null || true
    # Poll until the SG is gone — EC2 deletion is eventually consistent.
    local SG_DEADLINE=$(( $(date +%s) + 60 ))
    while [[ $(date +%s) -lt $SG_DEADLINE ]]; do
      local STILL; STILL=$(aws ec2 describe-security-groups \
        --region "$REGION" \
        --filters "Name=group-id,Values=${OLD_SG}" \
        --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")
      [[ -z "$STILL" || "$STILL" == "None" ]] && break
      sleep 3
    done
  fi

  CI_SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Juno CI Jenkins instance" \
    --vpc-id "$DEFAULT_VPC" \
    --region "$REGION" \
    --query "GroupId" --output text)

  aws ec2 authorize-security-group-ingress \
    --group-id "$CI_SG_ID" \
    --region "$REGION" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=SSH}]" \
      "IpProtocol=tcp,FromPort=${JENKINS_PORT},ToPort=${JENKINS_PORT},IpRanges=[{CidrIp=${MY_IP}/32,Description=Jenkins}]" \
    &>/dev/null
  inf "  SG: ${CI_SG_ID}  (SSH + Jenkins from ${MY_IP})"

  # ── Launch CI instance ───────────────────────────────────────
  inf "Launching CI instance (${INSTANCE_TYPE} from AMI ${CI_AMI})..."
  local DEFAULT_SUBNET; DEFAULT_SUBNET=$(aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" "Name=defaultForAz,Values=true" \
    --query "Subnets[0].SubnetId" --output text 2>/dev/null || echo "")
  [[ -z "$DEFAULT_SUBNET" || "$DEFAULT_SUBNET" == "None" ]] && \
    die "No default subnet found"

  # Attach the instance profile if it exists — this gives the Jenkins job
  # AWS CLI access via the metadata service without any stored credentials.
  local PROFILE_ARG=""
  if aws iam get-instance-profile \
      --instance-profile-name "juno-ci-profile" \
      --region "$REGION" &>/dev/null 2>&1; then
    PROFILE_ARG="--iam-instance-profile Name=juno-ci-profile"
    inf "  Attaching IAM instance profile: juno-ci-profile"
  fi

  local LAUNCH_OUT; LAUNCH_OUT=$(aws ec2 run-instances \
    --image-id "$CI_AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --count 1 \
    --key-name "$KEY_NAME" \
    --subnet-id "$DEFAULT_SUBNET" \
    --security-group-ids "$CI_SG_ID" \
    --associate-public-ip-address \
    ${PROFILE_ARG} \
    --block-device-mappings \
      "DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=juno-ci},{Key=Project,Value=juno}]" \
    --region "$REGION" \
    --output json)
  CI_INSTANCE_ID=$(echo "$LAUNCH_OUT" | jq -r '.Instances[0].InstanceId')
  inf "  Instance: ${CI_INSTANCE_ID}"

  # ── Wait for running ─────────────────────────────────────────
  inf "  Waiting for instance running..."
  aws ec2 wait instance-running \
    --instance-ids "$CI_INSTANCE_ID" \
    --region "$REGION"
  CI_IP=$(_public_ip "$CI_INSTANCE_ID")
  inf "  Public IP: ${CI_IP}"

  _save_state

  # ── Wait for SSH ─────────────────────────────────────────────
  inf "  Waiting for SSH..."
  local SSH_DEADLINE=$(( $(date +%s) + 300 ))
  while [[ $(date +%s) -lt $SSH_DEADLINE ]]; do
    if ssh $SSH_OPTS "ubuntu@${CI_IP}" "true" &>/dev/null; then
      inf "  SSH ready."
      break
    fi
    printf "." >&2
    sleep 10
  done
  echo "" >&2
  ssh $SSH_OPTS "ubuntu@${CI_IP}" "true" \
    || die "SSH did not become available within 5 min"

  # ── Check whether Jenkins is already running (from AMI bake) ─
  # If the AMI was baked with a running Jenkins, it may already be up.
  # If not (e.g. after a fresh base-AMI launch), run juno-jenkins.sh.
  inf "Checking Jenkins service status..."
  local JENKINS_STATUS; JENKINS_STATUS=$(_ssh \
    "systemctl is-active jenkins 2>/dev/null || echo inactive")

  if [[ "$JENKINS_STATUS" == "active" ]]; then
    inf "  Jenkins already active (AMI bake preserved the service)."
  else
    inf "  Jenkins not running — provisioning via juno-jenkins.sh..."
    local JENKINS_SCRIPT="${SCRIPT_DIR}/../ci/juno-jenkins.sh"
    [[ -f "$JENKINS_SCRIPT" ]] || \
      die "juno-jenkins.sh not found at: ${JENKINS_SCRIPT}"
    _scp "$JENKINS_SCRIPT" "ubuntu@${CI_IP}:/tmp/juno-jenkins.sh"
    # Capture exit code explicitly — sudo exit-code propagation through SSH
    # is unreliable on some Ubuntu configurations.
    local CI_PROV_LOG; CI_PROV_LOG=$(mktemp)
    _ssh "chmod +x /tmp/juno-jenkins.sh && \
          { sudo bash /tmp/juno-jenkins.sh \
              --port ${JENKINS_PORT} \
              --repo ${JUNO_REPO} \
              --git-ref ${JUNO_GIT_REF} 2>&1; echo \"__EXIT__:\$?\"; }" 2>&1 \
      | tee "$CI_PROV_LOG" \
      | grep -v "^__EXIT__:" >&2
    local CI_PROV_EXIT; CI_PROV_EXIT=$(grep -o '__EXIT__:[0-9]*' "$CI_PROV_LOG" 2>/dev/null | tail -1 | cut -d: -f2)
    rm -f "$CI_PROV_LOG"
    if [[ "${CI_PROV_EXIT:-1}" != "0" ]]; then
      # Provisioning failed.  The cached AMI may be stale (e.g. expired
      # Jenkins signing key baked in).  Deregister it so the next run
      # forces a clean rebake from the Ubuntu base AMI.
      warn "Jenkins provisioning failed."
      warn "Deregistering stale CI AMI ${CI_AMI} so next run rebakes..."
      local STALE_SNAP; STALE_SNAP=$(aws ec2 describe-images \
        --image-ids "$CI_AMI" --region "$REGION" \
        --query "Images[0].BlockDeviceMappings[0].Ebs.SnapshotId" \
        --output text 2>/dev/null || echo "")
      aws ec2 deregister-image \
        --image-id "$CI_AMI" --region "$REGION" &>/dev/null || true
      [[ -n "$STALE_SNAP" && "$STALE_SNAP" != "None" ]] && \
        aws ec2 delete-snapshot \
          --snapshot-id "$STALE_SNAP" --region "$REGION" &>/dev/null || true
      warn "Stale AMI cleared. Re-run startup — a fresh AMI will be baked."
      die "Provisioning failed — stale AMI deregistered."
    fi
  fi

  # ── Wait for Jenkins HTTP ─────────────────────────────────────
  inf "Waiting for Jenkins on http://${CI_IP}:${JENKINS_PORT}/..."
  local JENKINS_DEADLINE=$(( $(date +%s) + 180 ))
  while [[ $(date +%s) -lt $JENKINS_DEADLINE ]]; do
    if _ssh "curl -sf 'http://localhost:${JENKINS_PORT}/login' -o /dev/null" &>/dev/null; then
      inf "  Jenkins is up."
      break
    fi
    printf "." >&2
    sleep 5
  done
  echo "" >&2
  _ssh "curl -sf 'http://localhost:${JENKINS_PORT}/login' -o /dev/null" \
    || die "Jenkins did not respond on port ${JENKINS_PORT} within 3 min"

  inf "Jenkins UI: http://${CI_IP}:${JENKINS_PORT}/"
  inf "SSH:        ssh -i ${SSH_KEY_FILE} ubuntu@${CI_IP}"

  # ── Upload LoRA adapter if provided ──────────────────────────
  if [[ -n "$LORA_PLAY_LOCAL" ]]; then
    [[ -f "$LORA_PLAY_LOCAL" ]] \
      || die "--lora-play: file not found: ${LORA_PLAY_LOCAL}"
    local LORA_REMOTE="/var/lib/jenkins/juno/models/$(basename "${LORA_PLAY_LOCAL}")"
    inf "Uploading LoRA adapter to CI instance: $(basename "${LORA_PLAY_LOCAL}")..."
    _ssh "sudo mkdir -p /var/lib/jenkins/juno/models && sudo chown jenkins:jenkins /var/lib/jenkins/juno/models"
    _scp "$LORA_PLAY_LOCAL" "ubuntu@${CI_IP}:/tmp/$(basename "${LORA_PLAY_LOCAL}")"
    _ssh "sudo mv /tmp/$(basename "${LORA_PLAY_LOCAL}") ${LORA_REMOTE} && sudo chown jenkins:jenkins ${LORA_REMOTE}"
    inf "  LoRA adapter uploaded: ${LORA_REMOTE}"
    # Append --lora-play to perf args so the job picks it up
    PERF_ARGS="${PERF_ARGS} --lora-play ${LORA_REMOTE}"
  fi

  # ── Trigger job and wait ──────────────────────────────────────
  local JOB_RC=0
  _trigger_and_wait_job || JOB_RC=$?

  if [[ $JOB_RC -eq 0 ]]; then
    # ── Pull HTML back to local repo ────────────────────────────
    _pull_html_artifact
    inf ""
    inf "Performance test run complete."
    inf "Updated HTML matrix: ${HTML_OUT}"
  else
    warn "Jenkins job did not succeed (exit code ${JOB_RC})."
    warn "Jenkins build log: http://${CI_IP}:${JENKINS_PORT}/job/juno-performance-tests/lastBuild/console"
    warn "SSH: ssh -i ${SSH_KEY_FILE} ubuntu@${CI_IP}"
  fi

  # ── Teardown unless --keep or build failed ───────────────────
  if $KEEP; then
    inf ""
    inf "--keep: CI instance retained."
    inf "  Instance: ${CI_INSTANCE_ID}  IP: ${CI_IP}"
    inf "  Teardown: ./launcher.sh juno-ci.sh teardown"
  elif [[ $JOB_RC -ne 0 ]]; then
    inf ""
    inf "Build failed — CI instance retained for inspection."
    inf "  Jenkins log: http://${CI_IP}:${JENKINS_PORT}/job/juno-performance-tests/lastBuild/console"
    inf "  SSH:         ssh -i ${SSH_KEY_FILE} ubuntu@${CI_IP}"
    inf "  Teardown:    ./launcher.sh juno-ci.sh teardown"
  else
    inf ""
    inf "Tearing down CI instance..."
    _teardown
  fi

  # Propagate job failure after teardown so the calling process sees it.
  return $JOB_RC
}

# ── ENTRYPOINT ────────────────────────────────────────────────
case "$MODE" in
  startup)  _startup  ;;
  teardown) _teardown ;;
esac