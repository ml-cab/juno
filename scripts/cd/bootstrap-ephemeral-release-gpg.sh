#!/usr/bin/env bash
# Creates a throwaway GPG homedir under .release-gpg/ with one signing-only RSA key.
# Use only for local/CI artifact signing when no project key is configured.
#
#   source ./scripts/bootstrap-ephemeral-release-gpg.sh
#   ./scripts/create-release-artifacts.sh 0.1.0
#
# Or one-shot:
#   GNUPGHOME="$(pwd)/.release-gpg" ./scripts/bootstrap-ephemeral-release-gpg.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GNUPGHOME="${GNUPGHOME:-$ROOT/.release-gpg}"
mkdir -p "$GNUPGHOME"
chmod 700 "$GNUPGHOME"
if gpg --list-secret-keys --with-colons 2>/dev/null | grep -q '^sec:'; then
	echo "GNUPGHOME=$GNUPGHOME already has a secret key; skipping generation."
	exit 0
fi
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
cat >"$TMP" <<'EOF'
%no-protection
Key-Type: RSA
Key-Length: 3072
Subkey-Type: RSA
Subkey-Length: 3072
Name-Real: Juno Ephemeral Release
Name-Email: dev@ml.cab
Expire-Date: 7d
%commit
%echo done
EOF
gpg --batch --generate-key "$TMP"
gpg --export -a "ephemeral-release@juno.build" >"$GNUPGHOME/PUBLIC-KEY-ASCII.asc"
echo "Generated signing key in GNUPGHOME=$GNUPGHOME"
echo "Public key: $GNUPGHOME/PUBLIC-KEY-ASCII.asc"
