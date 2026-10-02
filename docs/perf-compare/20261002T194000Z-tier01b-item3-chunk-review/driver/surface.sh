#!/usr/bin/env bash
# usage: surface.sh LABEL PORT N_PROMPT -- <./juno subcommand and flags, without --api-port>
# Launches the engine in its own process group with stdin held open, waits for the API,
# runs ttft.py (2 warmup, 3 measured), prints LABEL + result, then stops the whole group.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/medion/Repo/juno
label="$1"; port="$2"; n_prompt="$3"; shift 4
log="$HERE/logs/${label}.log"; mkdir -p "$HERE/logs"
cd "$ROOT"
setsid bash -c 'sleep infinity | exec ./juno "$@"' _ "$@" --api-port "$port" >"$log" 2>&1 &
pgid=$!
ok=0
for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "$pgid" 2>/dev/null || break
  sleep 1
done
if (( ok )); then
  res="$(python3 "$HERE/ttft.py" "$port" "$n_prompt" 2 3 2>>"$log")"
  echo "$label $res"
else
  echo "$label FAILED_TO_START (see $log)"
fi
grep -m3 -i "Prefill chunk size\|prefill-batch\|chunk" "$log" | sed "s/^/  [$label log] /"
kill -TERM -- "-$pgid" 2>/dev/null
for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
kill -KILL -- "-$pgid" 2>/dev/null
sleep 2
