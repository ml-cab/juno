#!/bin/bash
# =============================================================
#  juno-jenkins.sh — Provision Jenkins on the current Ubuntu host.
#
#  Installs Jenkins LTS, configures it to build on master only
#  (no agents), and creates a single freestyle job:
#
#    "Juno performance tests"
#      - executes: ./scripts/performance-tests/performance-test.sh
#      - archives:  docs/juno_test_matrix.html
#
#  Intended to run inside an AMI bake (see scripts/aws/make-ami.sh)
#  or directly on any Ubuntu 22.04 instance that already has
#  JDK 25 and Maven installed.
#
#  Usage:
#    sudo ./scripts/ci/juno-jenkins.sh [--port PORT] [--repo URL]
#                                      [--git-ref REF] [--juno-dir DIR]
#
#  Options:
#    --port PORT       Jenkins HTTP port (default: 8080)
#    --repo URL        Git remote Juno will be cloned from in the job
#                      (default: https://github.com/ml-cab/juno.git)
#    --git-ref REF     Branch / tag the job checks out (default: main)
#    --juno-dir DIR    Absolute workspace path Jenkins will use
#                      (default: /var/lib/jenkins/workspace/juno-perf)
#
#  After the script finishes:
#    - Jenkins is running at http://<host>:<PORT>/
#    - Initial admin password: /var/lib/jenkins/secrets/initialAdminPassword
#    - The job "Juno performance tests" is pre-seeded via Jenkins CLI.
#    - To bake into an AMI: call this script from make-ami.sh
#      INSTALL heredoc for a CPU/t3 instance, then create-image normally.
#
#  Exits 0 on success.  All progress goes to stderr; no stdout noise.
# =============================================================

set -euo pipefail
export LC_ALL=C LANG=C LANGUAGE=C DEBIAN_FRONTEND=noninteractive

# Trap any unexpected exit and print the line so we know exactly
# which command caused set -e to fire.
trap 'echo "[jenkins] FATAL at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ── DEFAULTS ──────────────────────────────────────────────────
JENKINS_PORT=8080
JUNO_REPO="https://github.com/ml-cab/juno.git"
JUNO_GIT_REF="main"
JUNO_DIR="/var/lib/jenkins/workspace/juno-perf"

# ── COLOURS (stderr only) ─────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

inf()  { echo -e "${GREEN}[jenkins]${RESET} $*" >&2; }
warn() { echo -e "${YELLOW}[jenkins]${RESET} $*" >&2; }
die()  { echo -e "${RED}[jenkins]${RESET} $*" >&2; exit 1; }

# ── ARGUMENT PARSING ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     JENKINS_PORT="$2";  shift 2 ;;
    --repo)     JUNO_REPO="$2";     shift 2 ;;
    --git-ref)  JUNO_GIT_REF="$2";  shift 2 ;;
    --juno-dir) JUNO_DIR="$2";      shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || die "must be run as root (use sudo)"

# ── WAIT FOR CLOUD-INIT ───────────────────────────────────────
inf "Waiting for cloud-init..."
cloud-init status --wait 2>/dev/null || true

systemctl kill --signal=SIGKILL unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/cache/apt/archives/lock \
      /var/lib/apt/lists/lock
dpkg --configure -a 2>/dev/null || true

# ── BASE PACKAGES ─────────────────────────────────────────────
inf "Installing base packages..."
apt-get update -qq
apt-get install -y -qq \
  curl wget gnupg2 ca-certificates git \
  openjdk-17-jdk-headless \
  fontconfig jq bc \
  awscli

# Jenkins LTS requires JDK 17 at minimum for its own JVM.
# Juno itself runs on JDK 25; JAVA_HOME in the job is set explicitly.
# Verify JDK 25 is present (expected to already be installed in the AMI).
if ! update-alternatives --list java 2>/dev/null | grep -q 'java-25\|jdk-25'; then
  inf "JDK 25 not found via update-alternatives; installing..."
  apt-get install -y -qq openjdk-25-jdk maven
fi

# ── JENKINS REPOSITORY ────────────────────────────────────────
# Jenkins rotated their signing key in December 2025 (weekly 2.543 /
# LTS 2.541.1).  The old jenkins.io-2023.key is expired; the current
# key is jenkins.io-2026.key.  It must be stored as armored ASCII
# (.asc) — NOT dearmored binary — for modern apt to accept it.
# See: https://www.jenkins.io/blog/2025/12/23/repository-signing-keys-changing/

inf "Adding Jenkins LTS repository (2026 signing key)..."

# Remove any stale keyring or list files left by previous install attempts
# (including those that may have been baked into a cached AMI).
rm -f /usr/share/keyrings/jenkins-keyring.gpg \
      /usr/share/keyrings/jenkins-keyring.asc \
      /etc/apt/sources.list.d/jenkins.list

JENKINS_KEYRING=/usr/share/keyrings/jenkins-keyring.asc
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2026.key \
  | tee "$JENKINS_KEYRING" > /dev/null

# Verify the key was written as a real PGP block, not an HTML error page.
grep -q "BEGIN PGP PUBLIC KEY BLOCK" "$JENKINS_KEYRING" \
  || die "Jenkins signing key download failed — got HTML instead of PGP key"

echo "deb [signed-by=${JENKINS_KEYRING}] \
  https://pkg.jenkins.io/debian-stable binary/" \
  > /etc/apt/sources.list.d/jenkins.list

apt-get update -qq

# ── INSTALL JENKINS ───────────────────────────────────────────
inf "Installing Jenkins LTS..."
apt-get install -y -qq jenkins

# ── CONFIGURE PORT ────────────────────────────────────────────
# Modern Jenkins (2.346+) reads HTTP_PORT from /etc/default/jenkins
# and from the systemd override.
inf "Configuring Jenkins on port ${JENKINS_PORT}..."

JENKINS_DEFAULTS=/etc/default/jenkins
if [[ -f "$JENKINS_DEFAULTS" ]]; then
  sed -i "s/^HTTP_PORT=.*/HTTP_PORT=${JENKINS_PORT}/" "$JENKINS_DEFAULTS" || true
  grep -q "^HTTP_PORT=" "$JENKINS_DEFAULTS" \
    || echo "HTTP_PORT=${JENKINS_PORT}" >> "$JENKINS_DEFAULTS"
fi

# Systemd override so the port change survives daemon-reload.
SYSTEMD_OVERRIDE_DIR=/etc/systemd/system/jenkins.service.d
mkdir -p "$SYSTEMD_OVERRIDE_DIR"
cat > "${SYSTEMD_OVERRIDE_DIR}/override.conf" <<EOF
[Service]
Environment="JENKINS_PORT=${JENKINS_PORT}"
EOF

systemctl daemon-reload

# ── DISABLE SETUP WIZARD ──────────────────────────────────────
# Skip interactive first-run wizard so the AMI is usable out of the box.
# The initial admin password is still written to the secrets file.
JENKINS_JAVA_OPTS="-Djenkins.install.runSetupWizard=false"

if [[ -f "$JENKINS_DEFAULTS" ]]; then
  if grep -q "^JAVA_ARGS=" "$JENKINS_DEFAULTS"; then
    sed -i "s|^JAVA_ARGS=.*|JAVA_ARGS=\"${JENKINS_JAVA_OPTS}\"|" "$JENKINS_DEFAULTS"
  else
    echo "JAVA_ARGS=\"${JENKINS_JAVA_OPTS}\"" >> "$JENKINS_DEFAULTS"
  fi
fi

cat >> "${SYSTEMD_OVERRIDE_DIR}/override.conf" <<EOF
Environment="JAVA_OPTS=${JENKINS_JAVA_OPTS}"
EOF
systemctl daemon-reload

# ── RESTRICT TO MASTER-ONLY BUILDS ───────────────────────────
# numExecutors=2 on built-in node; no remote agents configured.
# This is enforced via the init Groovy script that runs before Jenkins
# accepts the first connection, so it survives restarts.
INIT_GROOVY_DIR=/var/lib/jenkins/init.groovy.d
mkdir -p "$INIT_GROOVY_DIR"

cat > "${INIT_GROOVY_DIR}/00-master-only.groovy" <<'GROOVY'
import jenkins.model.Jenkins
import hudson.model.Node.Mode

def j = Jenkins.get()
j.setNumExecutors(2)
j.setMode(Mode.EXCLUSIVE)
j.save()
GROOVY

# Write jenkins.model.JenkinsLocationConfiguration.xml directly — this is
# simpler than Groovy and is read by Jenkins before the CLI handshake.
# Without a configured URL Jenkins returns 403 on every CLI call.
cat > /var/lib/jenkins/jenkins.model.JenkinsLocationConfiguration.xml <<XML
<?xml version='1.1' encoding='UTF-8'?>
<jenkins.model.JenkinsLocationConfiguration>
  <adminAddress>address not configured yet &lt;nobody@nowhere&gt;</adminAddress>
  <jenkinsUrl>http://localhost:${JENKINS_PORT}/</jenkinsUrl>
</jenkins.model.JenkinsLocationConfiguration>
XML
chown jenkins:jenkins /var/lib/jenkins/jenkins.model.JenkinsLocationConfiguration.xml



# ── START JENKINS (first time, to install plugins) ────────────
# Start Jenkins so the plugin directory is initialised and the CLI jar
# is available.  Plugins are installed via the Jenkins REST API,
# which works with basic auth and needs no Jenkins URL to be configured.
inf "Starting Jenkins to initialise plugin directory..."
# Reset any failed state from a prior install (e.g. re-provisioning a baked instance).
systemctl reset-failed jenkins 2>/dev/null || true
systemctl enable jenkins
systemctl start jenkins

inf "Waiting for Jenkins to accept connections on port ${JENKINS_PORT} (plugin init)..."
DEADLINE=$(( $(date +%s) + 180 ))
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if curl -sf "http://localhost:${JENKINS_PORT}/login" -o /dev/null 2>/dev/null; then
    inf "  Jenkins up."
    break
  fi
  printf "." >&2
  sleep 5
done
echo "" >&2
curl -sf "http://localhost:${JENKINS_PORT}/login" -o /dev/null \
  || die "Jenkins did not respond on port ${JENKINS_PORT} within 3 min"

ADMIN_PASSWORD=$(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || true)
[[ -n "$ADMIN_PASSWORD" ]] || die "initialAdminPassword not found — Jenkins did not initialise correctly"

# ── INSTALL PLUGINS (minimal set) ─────────────────────────────
# Use the REST API directly — avoids CLI URL-not-configured 403 entirely.
# POST to pluginManager/installNecessaryPlugins with a minimal plugins.xml.
inf "Installing Jenkins plugins (git, workflow-aggregator)..."

# Jenkins ties the CSRF crumb to the session cookie.
# Both the crumb fetch and every subsequent POST must share the same cookie jar.
COOKIE_JAR=$(mktemp)
CRUMB_FIELD=""
CRUMB_VAL=""
CRUMB_DEADLINE=$(( $(date +%s) + 60 ))
while [[ $(date +%s) -lt $CRUMB_DEADLINE ]]; do
  CRUMB_JSON=$(curl -sf \
    -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    --user "admin:${ADMIN_PASSWORD}" \
    "http://localhost:${JENKINS_PORT}/crumbIssuer/api/json" 2>/dev/null || echo '{}')
  CRUMB_FIELD=$(echo "$CRUMB_JSON" | jq -r '.crumbRequestField // ""' 2>/dev/null || echo "")
  CRUMB_VAL=$(echo   "$CRUMB_JSON" | jq -r '.crumb // ""'            2>/dev/null || echo "")
  [[ -n "$CRUMB_FIELD" && -n "$CRUMB_VAL" ]] && break
  inf "  Waiting for CSRF crumb endpoint..."
  sleep 5
done
if [[ -z "$CRUMB_FIELD" || -z "$CRUMB_VAL" ]]; then
  inf "  DEBUG crumbIssuer response: ${CRUMB_JSON:-<empty>}"
  inf "  DEBUG admin password length: ${#ADMIN_PASSWORD}"
  rm -f "$COOKIE_JAR"
  die "Failed to get CSRF crumb after 60 s — Jenkins security realm not ready"
fi
inf "  CSRF crumb obtained (field: ${CRUMB_FIELD})."

for plugin in git workflow-aggregator; do
  inf "  Installing ${plugin}..."
  PLUGIN_BODY=$(mktemp)
  HTTP=$(curl -sw "%{http_code}" -o "$PLUGIN_BODY" \
    -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -X POST \
    --user "admin:${ADMIN_PASSWORD}" \
    -H "${CRUMB_FIELD}: ${CRUMB_VAL}" \
    "http://localhost:${JENKINS_PORT}/pluginManager/install?plugin.${plugin}.default=on")
  if [[ "$HTTP" != "200" && "$HTTP" != "302" ]]; then
    inf "  DEBUG response body: $(head -c 500 "$PLUGIN_BODY")"
    rm -f "$PLUGIN_BODY" "$COOKIE_JAR"
    die "Plugin install POST for '${plugin}' failed — HTTP ${HTTP}"
  fi
  rm -f "$PLUGIN_BODY"
done
rm -f "$COOKIE_JAR"

# Wait for plugins to download and appear on disk (update centre is async).
# workflow-aggregator pulls ~50 dependencies; allow 5 min per plugin.
inf "  Waiting for plugins to land on disk..."
for plugin in git workflow-aggregator; do
  PLUGIN_DEADLINE=$(( $(date +%s) + 300 ))
  while [[ $(date +%s) -lt $PLUGIN_DEADLINE ]]; do
    [[ -f "/var/lib/jenkins/plugins/${plugin}.jpi" || \
       -f "/var/lib/jenkins/plugins/${plugin}.hpi" ]] && break
    sleep 5
  done
  [[ -f "/var/lib/jenkins/plugins/${plugin}.jpi" || \
     -f "/var/lib/jenkins/plugins/${plugin}.hpi" ]] \
    || die "Plugin '${plugin}' not on disk after 5 min — check update centre connectivity"
done
inf "  Plugins verified on disk."

# ── CREATE JOB CONFIG ─────────────────────────────────────────
# Freestyle job.  Jenkins reads config.xml from the job directory on startup.
# We use a shell build step (not Pipeline) to keep the job identical to what
# a developer would type locally.
#
# The job:
#   1. Clones / updates the Juno repo into JUNO_DIR.
#   2. Runs ./scripts/performance-tests/performance-test.sh --foreground --all
#      (full matrix run; override via PERF_ARGS parameter if needed).
#   3. Archives docs/juno_test_matrix.html as a build artifact.
#
# JAVA_HOME is forced to JDK 25 so `mvn` and Juno's JVM both use it.

JOB_DIR=/var/lib/jenkins/jobs/juno-performance-tests
mkdir -p "$JOB_DIR"
chown jenkins:jenkins "$JOB_DIR"

# Resolve JDK 25 home at provisioning time so the path is baked into config.xml.
JAVA25_HOME=$(update-java-alternatives -l 2>/dev/null \
  | awk '/java-1\.25\.|java-25/{print $3}' | head -1 || true)
if [[ -z "$JAVA25_HOME" ]]; then
  for candidate in \
      /usr/lib/jvm/java-25-openjdk-amd64 \
      /usr/lib/jvm/java-1.25.0-openjdk-amd64 \
      /usr/lib/jvm/temurin-25; do
    [[ -d "$candidate/bin/java" || -f "$candidate/bin/java" ]] \
      && { JAVA25_HOME="$candidate"; break; }
  done
fi
[[ -n "$JAVA25_HOME" ]] || die "JDK 25 home not found; ensure openjdk-25-jdk is installed"
inf "JDK 25 home: ${JAVA25_HOME}"

# Write config.xml with a QUOTED heredoc ('XML') so bash does NOT expand any
# variables inside it.  The only value that needs baking in is JAVA25_HOME,
# which we substitute with sed after writing.
cat > "${JOB_DIR}/config.xml" <<'XML'
<?xml version='1.1' encoding='UTF-8'?>
<project>
  <description>Run the Juno performance test matrix and publish juno_test_matrix.html.</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <hudson.model.ParametersDefinitionProperty>
      <parameterDefinitions>
        <hudson.model.StringParameterDefinition>
          <name>PERF_ARGS</name>
          <defaultValue>--foreground --all</defaultValue>
          <description>Arguments forwarded to performance-test.sh (default: full matrix run)</description>
          <trim>true</trim>
        </hudson.model.StringParameterDefinition>
        <hudson.model.StringParameterDefinition>
          <name>GIT_REF</name>
          <defaultValue>main</defaultValue>
          <description>Branch, tag, or commit to check out</description>
          <trim>true</trim>
        </hudson.model.StringParameterDefinition>
      </parameterDefinitions>
    </hudson.model.ParametersDefinitionProperty>
  </properties>
  <assignedNode>built-in</assignedNode>
  <canRoam>false</canRoam>
  <disabled>false</disabled>
  <blockBuildWhenDownstreamBuilding>false</blockBuildWhenDownstreamBuilding>
  <blockBuildWhenUpstreamBuilding>false</blockBuildWhenUpstreamBuilding>
  <triggers/>
  <concurrentBuild>false</concurrentBuild>
  <builders>
    <hudson.tasks.Shell>
      <command>#!/bin/bash
set -euo pipefail

export JAVA_HOME=__JAVA25_HOME__
export PATH="${JAVA_HOME}/bin:${PATH}"

JUNO_WORKSPACE=/var/lib/jenkins/juno
REPO=https://github.com/ml-cab/juno.git
REF=${GIT_REF:-main}

if [[ -d "${JUNO_WORKSPACE}/.git" ]]; then
  echo "[ci] Updating existing clone in ${JUNO_WORKSPACE}..."
  git -C "${JUNO_WORKSPACE}" fetch --prune origin
  git -C "${JUNO_WORKSPACE}" checkout "${REF}"
  git -C "${JUNO_WORKSPACE}" reset --hard "origin/${REF}" 2>/dev/null     || git -C "${JUNO_WORKSPACE}" reset --hard "${REF}"
else
  echo "[ci] Cloning ${REPO} into ${JUNO_WORKSPACE}..."
  git clone --depth 50 --branch "${REF}" "${REPO}" "${JUNO_WORKSPACE}"
fi

cd "${JUNO_WORKSPACE}"

echo "[ci] Building Juno (skip tests)..."
mvn -q clean package -DskipTests

echo "[ci] Running performance tests with args: ${PERF_ARGS}"
./scripts/performance-tests/performance-test.sh ${PERF_ARGS}

echo "[ci] Performance tests complete."
echo "[ci] HTML matrix: ${JUNO_WORKSPACE}/docs/juno_test_matrix.html"
</command>
    </hudson.tasks.Shell>
  </builders>
  <publishers>
    <hudson.tasks.ArtifactArchiver>
      <artifacts>docs/juno_test_matrix.html</artifacts>
      <allowEmptyArchive>false</allowEmptyArchive>
      <onlyIfSuccessful>false</onlyIfSuccessful>
      <fingerprint>false</fingerprint>
      <defaultExcludes>true</defaultExcludes>
      <caseSensitive>true</caseSensitive>
      <followSymlinks>false</followSymlinks>
    </hudson.tasks.ArtifactArchiver>
  </publishers>
  <buildWrappers/>
</project>
XML

# Substitute the bake-time JAVA25_HOME into the placeholder we left above.
sed -i "s|__JAVA25_HOME__|${JAVA25_HOME}|g" "${JOB_DIR}/config.xml"

[[ -s "${JOB_DIR}/config.xml" ]] || die "config.xml is empty after write — heredoc failed"
chown -R jenkins:jenkins "$JOB_DIR"

# ── RELOAD CONFIGURATION ──────────────────────────────────────
inf "Reloading Jenkins configuration..."
RELOAD_COOKIE_JAR=$(mktemp)
RELOAD_CRUMB=$(curl -sf \
  -c "$RELOAD_COOKIE_JAR" -b "$RELOAD_COOKIE_JAR" \
  --user "admin:${ADMIN_PASSWORD}" \
  "http://localhost:${JENKINS_PORT}/crumbIssuer/api/json" \
  | jq -r '"\(.crumbRequestField):\(.crumb)"' 2>/dev/null || echo "")

RELOAD_HTTP=$(curl -sw "%{http_code}" -o /dev/null \
  -c "$RELOAD_COOKIE_JAR" -b "$RELOAD_COOKIE_JAR" \
  -X POST \
  --user "admin:${ADMIN_PASSWORD}" \
  ${RELOAD_CRUMB:+-H "$RELOAD_CRUMB"} \
  "http://localhost:${JENKINS_PORT}/reload")
rm -f "$RELOAD_COOKIE_JAR"
[[ "$RELOAD_HTTP" == "200" || "$RELOAD_HTTP" == "302" ]] \
  || die "reload failed — HTTP ${RELOAD_HTTP}"

# Give Jenkins a moment to finish reloading.
sleep 5

# ── VERIFY JOB EXISTS ─────────────────────────────────────────
JOB_HTTP=$(curl -sw "%{http_code}" -o /dev/null \
  --user "admin:${ADMIN_PASSWORD}" \
  "http://localhost:${JENKINS_PORT}/job/juno-performance-tests/api/json")
[[ "$JOB_HTTP" == "200" ]] \
  || die "Job 'juno-performance-tests' not found after reload (HTTP ${JOB_HTTP}) — check config.xml"
inf "Job 'juno-performance-tests' verified via REST API."

# ── SUMMARY ───────────────────────────────────────────────────
inf "Jenkins setup complete."
inf "  URL:            http://0.0.0.0:${JENKINS_PORT}/"
inf "  Admin password: $(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo 'see /var/lib/jenkins/secrets/initialAdminPassword')"
inf "  Job:            Juno performance tests"
inf "  Artifact:       docs/juno_test_matrix.html"