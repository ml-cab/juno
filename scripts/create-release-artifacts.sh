#!/usr/bin/env bash
# Collect every module JAR from target/ after a full "mvn clean package",
# copy into dist/juno-VERSION/, write SHA256SUMS, and detached-sign each JAR with GPG.
#
# Usage (from repo root, on the desired tag or release tree):
#   mvn clean package -DskipTests
#   ./scripts/create-release-artifacts.sh 0.1.0
#   ./scripts/create-release-artifacts.sh --checksums-only 0.1.0   # SHA-256 only (no GPG)
#
# GPG: requires a secret key and unlocked agent (or batch passphrase setup).
set -euo pipefail

CHECKSUMS_ONLY=0
if [[ "${1:-}" == "--checksums-only" ]]; then
	CHECKSUMS_ONLY=1
	shift
fi

VERSION="${1:?usage: $0 [--checksums-only] <version e.g. 0.1.0>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/dist/juno-${VERSION}"
mkdir -p "${OUT}"

mapfile -t jars < <(
	find "${ROOT}" -type f -path '*/target/*.jar' \
		! -name '*-sources.jar' \
		! -name '*-javadoc.jar' \
		! -name '*-tests.jar' \
		! -name '*-original.jar' \
		-print 2>/dev/null | LC_ALL=C sort
)

if [[ ${#jars[@]} -eq 0 ]]; then
	echo "No JARs found under */target/. Run: mvn clean package -DskipTests" >&2
	exit 1
fi

for j in "${jars[@]}"; do
	rel="${j#"${ROOT}/"}"
	safe="${rel//\//__}"
	cp -f "$j" "${OUT}/${safe}"
done

cd "${OUT}"
rm -f SHA256SUMS SHA256SUMS.asc SIGNING.log
sha256sum -- *.jar > SHA256SUMS

if [[ "$CHECKSUMS_ONLY" -eq 1 ]]; then
	echo "Wrote ${#jars[@]} JARs plus SHA256SUMS under ${OUT} (GPG skipped)"
	exit 0
fi

SIGN_OK=1
for j in *.jar; do
	[[ -f "$j" ]] || continue
	if ! gpg --batch --yes --detach-sign --armor --output "${j}.asc" "$j" 2>>SIGNING.log; then
		SIGN_OK=0
	fi
done
if ! gpg --batch --yes --detach-sign --armor --output SHA256SUMS.asc SHA256SUMS 2>>SIGNING.log; then
	SIGN_OK=0
fi

echo "Wrote ${#jars[@]} JARs plus SHA256SUMS and .asc signatures under ${OUT}"
if [[ "$SIGN_OK" -ne 1 ]]; then
	echo "GPG signing failed or no usable secret key (see ${OUT}/SIGNING.log)." >&2
	exit 2
fi
