#!/usr/bin/env bash
# Selftest for perf-lib.sh's engine stdin keepalive (perf_engine_stdin_open, perf_engine_exec,
# perf_engine_stdin_release). Needs no model, GPU or build: the "engine" is a stub that reads
# stdin until end of file, which is exactly the property the console REPL has.
#
# Checks, per launch: the engine stays up while the pipe is held, sees end of file and exits
# on release, leaves no pipe on disk and no descriptor in this shell, and leaves no helper
# process behind; this shell's stderr still works afterwards; an engine whose launching
# script exits without stopping it is not orphaned. A negative control runs the old
# process-substitution pattern and requires the leftover-process check to catch it, so the
# check is shown able to fail.
#
# Usage: scripts/performance-tests/selftest-engine-stdin.sh     (exit 0 = all pass)

set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"

failures=0
pass() { printf '[engine-stdin-selftest] PASS: %s\n' "$*"; }
# stdout, not stderr: one defect this test exists to catch is a release that silences stderr.
fail() { printf '[engine-stdin-selftest] FAIL: %s\n' "$*"; failures=$((failures + 1)); }

TMPDIR="$(mktemp -d)"
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT

# A unique marker in the stub's argv lets pgrep find exactly this test's processes.
MARK="juno-stdin-selftest-$$"
STUB=(bash -c 'while IFS= read -r _; do :; done; exit 0' "$MARK")

# Leftover helpers: any sleep loop or stub still alive that carries this run's marker, or
# any `sleep 3600` whose parent is gone (reparented), started after this test began.
leftovers() {
  pgrep -f "$MARK" 2>/dev/null | wc -l
}
sleep_loops() {
  pgrep -f -x 'sleep 3600' -s 0 2>/dev/null | wc -l
}
fifo_count() {
  find "$TMPDIR" -type p 2>/dev/null | wc -l
}
own_fifo_fds() {
  local n=0 fd
  for fd in /proc/$$/fd/*; do
    [[ "$(readlink "$fd" 2>/dev/null)" == *juno-engine-stdin* ]] && n=$((n + 1))
  done
  echo "$n"
}
wait_gone() {
  local pid="$1" i
  for i in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
  return 1
}

stderr_target="$(readlink /proc/$$/fd/2)"
baseline_sleeps="$(sleep_loops)"
baseline_sleep_pids=" $(pgrep -f -x 'sleep 3600' -s 0 2>/dev/null | tr '\n' ' ') "

# 1. Launch, hold, release: repeated to show nothing accumulates.
for round in 1 2 3; do
  perf_engine_stdin_open || { fail "round ${round}: perf_engine_stdin_open failed"; continue; }
  ( perf_engine_exec "${STUB[@]}" ) &
  pid=$!
  sleep 0.5
  if kill -0 "$pid" 2>/dev/null; then pass "round ${round}: engine stays up while the pipe is held"
  else fail "round ${round}: engine exited while the pipe was held (saw end of file early)"; fi

  # The engine must hold exactly one descriptor on the pipe: its stdin, read side.
  n="$(find /proc/"$pid"/fd -lname '*juno-engine-stdin*' 2>/dev/null | wc -l)"
  if [[ "$n" == 1 ]]; then pass "round ${round}: engine holds only its stdin on the pipe"
  else fail "round ${round}: engine holds ${n} descriptors on the pipe (inherited writer not closed)"; fi

  perf_engine_stdin_release
  if wait_gone "$pid"; then pass "round ${round}: engine sees end of file and exits on release"
  else fail "round ${round}: engine still running 5s after release"; kill -KILL "$pid" 2>/dev/null; fi
  wait "$pid" 2>/dev/null

  [[ "$(fifo_count)" == 0 ]] && pass "round ${round}: no pipe left on disk" \
    || fail "round ${round}: $(fifo_count) pipe(s) left on disk"
  [[ "$(own_fifo_fds)" == 0 ]] && pass "round ${round}: no pipe descriptor left in this shell" \
    || fail "round ${round}: $(own_fifo_fds) pipe descriptor(s) left in this shell"
  [[ "$(leftovers)" == 0 ]] && pass "round ${round}: no process left carrying the run marker" \
    || fail "round ${round}: $(leftovers) process(es) left carrying the run marker"
done
[[ "$(sleep_loops)" == "$baseline_sleeps" ]] && pass "no sleep loop left after three launches" \
  || fail "sleep loops went from ${baseline_sleeps} to $(sleep_loops)"

# 2. Releasing must not silence this shell's stderr (a bare `exec {fd}>&- 2>/dev/null` does).
[[ "$(readlink /proc/$$/fd/2)" == "$stderr_target" ]] && pass "stderr still points where it did before any release" \
  || fail "stderr now points at $(readlink /proc/$$/fd/2), was ${stderr_target}"

# 3. Release with nothing open is a no-op.
perf_engine_stdin_release && perf_engine_stdin_release && pass "release is idempotent" \
  || fail "release with nothing open returned non-zero"

# 4. A launching script that exits without stopping its engine must not orphan it.
orphan_pid="$(
  bash -c '
    source "$1"; shift
    perf_engine_stdin_open
    ( perf_engine_exec "$@" ) >/dev/null 2>&1 &
    echo $!
  ' _ "${PERF_SCRIPTS}/perf-lib.sh" "${STUB[@]}"
)"
if [[ -n "$orphan_pid" ]] && wait_gone "$orphan_pid"; then
  pass "engine exits when its launching script exits without stopping it"
else
  fail "engine ${orphan_pid:-?} outlived its launching script"; kill -KILL "$orphan_pid" 2>/dev/null
fi
rm -f "$TMPDIR"/juno-engine-stdin.* 2>/dev/null

# 5. Negative control: the old pattern leaves a sleep loop behind, and the check must see it.
( exec "${STUB[@]}" < <(while true; do sleep 3600; done) ) &
old_pid=$!
sleep 0.5
kill -TERM "$old_pid" 2>/dev/null; wait "$old_pid" 2>/dev/null
sleep 0.2
if (( $(sleep_loops) > baseline_sleeps )); then
  pass "negative control: the leftover check catches the old process-substitution pattern"
else
  fail "negative control: the old pattern left no detectable sleep loop, so the check proves nothing"
fi
# Clean up the control's leftovers (the looping subshell and its sleep), and nothing that
# was already running before this test started.
for p in $(pgrep -f -x 'sleep 3600' -s 0 2>/dev/null); do
  [[ "$baseline_sleep_pids" == *" $p "* ]] && continue
  loop="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
  [[ -n "$loop" && "$loop" != 1 ]] && kill -KILL "$loop" 2>/dev/null
  kill -KILL "$p" 2>/dev/null
done

if (( failures > 0 )); then
  printf '[engine-stdin-selftest] %d check(s) failed\n' "$failures"
  exit 1
fi
printf '[engine-stdin-selftest] all checks passed\n'
