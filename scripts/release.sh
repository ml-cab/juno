#!/usr/bin/env bash
# Juno release driver: Maven Release Plugin (prepare / perform) + optional dist bundle.
#
# Versions and secrets are supplied as Maven system properties (-Dname=value) and/or
# environment variables (env wins are merged; explicit -D on the command line overrides env).
#
# Required (one of each pair):
#   -DreleaseVersion=0.1.1              or  JUNO_RELEASE_VERSION=0.1.1
#   -DdevelopmentVersion=0.1.2-SNAPSHOT  or  JUNO_DEVELOPMENT_VERSION=0.1.2-SNAPSHOT
#
# Optional — same names as Maven user properties / plugin fields where applicable:
#   -DpushChanges=true|false           or  JUNO_PUSH_CHANGES
#   -DlocalCheckout=true|false         or  JUNO_LOCAL_CHECKOUT   (override pom default)
#   -DscmUsername=…                    or  JUNO_SCM_USERNAME  (Git user for HTTPS)
#   -DscmPassword=…                    or  JUNO_SCM_PASSWORD  (PAT or token; avoid shell history — prefer env)
#   -Dgpg.passphrase=…                 or  JUNO_GPG_PASSPHRASE / GPG_PASSPHRASE  (batch GPG + Maven -Prelease-sign)
#   -Dgpg.keyname=…                    or  JUNO_GPG_KEYNAME / GPG_KEYNAME
#   -DskipTests / -DskipITs            or  JUNO_SKIP_TESTS / JUNO_SKIP_ITS  (default true for release goals)
#
# Dist step (after perform): set JUNO_SKIP_DIST=1 to skip. GPG for ./scripts/cd/create-release-artifacts.sh
# uses JUNO_GPG_PASSPHRASE with --pinentry-mode loopback when set.
#
# Other:
#   MAVEN_BIN          default: mvn
#   MAVEN_SETTINGS     if set: passed as -s "$MAVEN_SETTINGS"
#   JUNO_GNUPGHOME     if set: exported as GNUPGHOME for dist signing
#   JUNO_MAVEN_ARGS    extra args appended to every mvn invocation (quoted string split on spaces is fragile — prefer multiple env vars or edit script)
#
# Phases:  all (default) = prepare + perform + dist
#          prepare | perform | dist
#
# Examples:
#   ./release.sh -DreleaseVersion=0.1.1 -DdevelopmentVersion=0.1.2-SNAPSHOT
#   JUNO_RELEASE_VERSION=0.2.0 JUNO_DEVELOPMENT_VERSION=0.2.1-SNAPSHOT JUNO_PUSH_CHANGES=true \
#     JUNO_SCM_USERNAME=x JUNO_SCM_PASSWORD="$GITHUB_TOKEN" ./release.sh all
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
	cat <<'USAGE'
Juno release.sh — see script header for -D properties and JUNO_* / GPG_* env vars.
Phases: all (default) | prepare | perform | dist
USAGE
}

# --- merge -D properties: env defaults, then explicit -D from argv (argv wins) ---
declare -A PROP=()

set_prop() {
	local k="$1" v="$2"
	[[ -z "$v" ]] && return 0
	PROP["$k"]="$v"
}

parse_d_arg() {
	local a="$1"
	[[ "$a" == -D*=* ]] || return 1
	local rest="${a#-D}"
	local k="${rest%%=*}"
	local v="${rest#*=}"
	[[ -n "$k" ]] || return 1
	PROP["$k"]="$v"
	return 0
}

# Env → properties (only if not empty)
set_prop releaseVersion "${JUNO_RELEASE_VERSION:-}"
set_prop developmentVersion "${JUNO_DEVELOPMENT_VERSION:-}"
set_prop pushChanges "${JUNO_PUSH_CHANGES:-}"
set_prop localCheckout "${JUNO_LOCAL_CHECKOUT:-}"
set_prop scmUsername "${JUNO_SCM_USERNAME:-}"
set_prop scmPassword "${JUNO_SCM_PASSWORD:-}"
set_prop gpg.passphrase "${JUNO_GPG_PASSPHRASE:-${GPG_PASSPHRASE:-}}"
set_prop gpg.keyname "${JUNO_GPG_KEYNAME:-${GPG_KEYNAME:-}}"

# Test skip flags for release build (defaults on unless explicitly disabled)
if [[ -n "${JUNO_SKIP_TESTS:-}" ]]; then
	set_prop skipTests "${JUNO_SKIP_TESTS}"
else
	set_prop skipTests "true"
fi
if [[ -n "${JUNO_SKIP_ITS:-}" ]]; then
	set_prop skipITs "${JUNO_SKIP_ITS}"
else
	set_prop skipITs "true"
fi

PHASE="all"
CLI=()
for arg in "$@"; do
	case "$arg" in
		all | prepare | perform | dist)
			PHASE="$arg"
			;;
		-h | --help | help)
			usage
			exit 0
			;;
		*)
			CLI+=("$arg")
			;;
	esac
done

for a in "${CLI[@]}"; do
	if [[ "$a" == -D*=* ]]; then
		parse_d_arg "$a" || true
	fi
done

# Recomputed after env + CLI merge
to_mvn_props() {
	local -a out=()
	local k
	for k in "${!PROP[@]}"; do
		[[ -z "${PROP[$k]:-}" ]] && continue
		out+=("-D${k}=${PROP[$k]}")
	done
	# deterministic order
	local IFS=$'\n'
	printf '%s\n' "${out[@]}" | LC_ALL=C sort
}

MVN_BIN="${MAVEN_BIN:-mvn}"
MVN_SETTINGS=()
[[ -n "${MAVEN_SETTINGS:-}" ]] && MVN_SETTINGS=(-s "$MAVEN_SETTINGS")

require_versions() {
	[[ -n "${PROP[releaseVersion]:-}" ]] || {
		echo "Missing releaseVersion (-DreleaseVersion=… or JUNO_RELEASE_VERSION)." >&2
		exit 1
	}
	[[ -n "${PROP[developmentVersion]:-}" ]] || {
		echo "Missing developmentVersion (-DdevelopmentVersion=… or JUNO_DEVELOPMENT_VERSION)." >&2
		exit 1
	}
}

# Forward only user-defined -D from CLI (Maven already has our merged list in MVN_PROPS)
# Also forward unknown flags (e.g. -B) from CLI
CLI_FILTERED=()
for a in "${CLI[@]}"; do
	case "$a" in
		-D*=*)
			# already merged into PROP
			;;
		*)
			CLI_FILTERED+=("$a")
			;;
	esac
done

rebuild_mvn_props() {
	mapfile -t MVN_PROPS < <(to_mvn_props)
}

run_mvn() {
	rebuild_mvn_props
	"$MVN_BIN" -B "${MVN_SETTINGS[@]}" "$@" "${MVN_PROPS[@]}" "${CLI_FILTERED[@]}"
}

do_prepare() {
	require_versions
	run_mvn release:clean release:prepare
}

do_perform() {
	require_versions
	run_mvn release:perform
}

do_dist() {
	require_versions
	local rv="${PROP[releaseVersion]}"
	local tag="v${rv}"
	local prior=""
	prior="$(git symbolic-ref -q --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || true)"
	export GNUPGHOME="${JUNO_GNUPGHOME:-${GNUPGHOME:-}}"
	# Non-interactive signing for create-release-artifacts.sh
	if [[ -n "${JUNO_GPG_PASSPHRASE:-${GPG_PASSPHRASE:-}}" ]]; then
		export JUNO_GPG_PASSPHRASE="${JUNO_GPG_PASSPHRASE:-${GPG_PASSPHRASE:-}}"
	fi
	if [[ -t 0 ]]; then
		export GPG_TTY="${GPG_TTY:-$(tty)}"
	fi
	git fetch --tags --quiet 2>/dev/null || true
	git checkout "$tag"
	"$MVN_BIN" clean package \
		-B \
		"${MVN_SETTINGS[@]}" \
		-DskipTests="${PROP[skipTests]:-true}" \
		-DskipITs="${PROP[skipITs]:-true}" \
		-q
	"$ROOT/scripts/cd/create-release-artifacts.sh" "$rv"
	if [[ -n "$prior" ]]; then
		git switch "$prior" 2>/dev/null || git checkout "$prior" 2>/dev/null || true
	fi
	echo "Dist layout: $ROOT/dist/juno-${rv}/"
}

case "$PHASE" in
	all)
		do_prepare
		do_perform
		if [[ "${JUNO_SKIP_DIST:-0}" != "1" ]]; then
			do_dist
		fi
		;;
	prepare) do_prepare ;;
	perform) do_perform ;;
	dist) do_dist ;;
	*)
		echo "Unknown phase: $PHASE (use all|prepare|perform|dist)" >&2
		exit 2
		;;
esac
