#!/usr/bin/env bash
# Waits for the CPU prefill matmul gate to be prepared, then runs it pinned.
#
# Start it now, in a terminal you leave open:   bash dist/cpu-prefill-jit-gate/wait-and-run.sh
# It asks for sudo once and keeps the timestamp alive while it waits. It starts run-gate.sh only when
#   1. dist/cpu-prefill-jit-gate/READY exists (written last, after the tests passed and both jars and
#      run-gate.sh are in place; it holds the two jar hashes, checked here before anything runs), and
#   2. no Maven or surefire process is running (no build or test may share the machine with a pinned run).
# Ctrl-C at any point stops it; nothing is pinned until the gate itself starts.
set -uo pipefail
cd "$(dirname "$0")/../.."
DIR=dist/cpu-prefill-jit-gate

sudo -v || { echo "sudo failed" >&2; exit 2; }
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

echo "$(date +%H:%M:%S) waiting for $DIR/READY ..."
until [[ -f $DIR/READY ]]; do sleep 30; done
echo "$(date +%H:%M:%S) READY found:"; cat "$DIR/READY"

while pgrep -f "org.codehaus.plexus.classworlds|surefirebooter|surefire.booter" >/dev/null; do
  echo "$(date +%H:%M:%S) a Maven or test process is still running; waiting"; sleep 30
done

for j in baseline candidate; do
  want=$(awk -v k="$j" '$1 == k {print $2}' "$DIR/READY")
  have=$(sha256sum "$DIR/$j-shaded.jar" 2>/dev/null | cut -c1-16)
  [[ -n $want && $want == "$have" ]] || { echo "$j jar hash $have does not match READY ($want); not running" >&2; exit 2; }
done

echo "$(date +%H:%M:%S) starting the gate"
bash "$DIR/run-gate.sh" 2>&1 | tee "$DIR/gate-$(date +%Y%m%dT%H%M%S).log"
exit "${PIPESTATUS[0]}"
