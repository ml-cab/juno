#!/usr/bin/env bash
# smoke-cpu-kernels.sh — the CPU smoke on two builds, with every greedy reply compared between them
#
# Runs smoke-cpu-hot-path.sh (CPU decode and a prompt of about 512 tokens on the four sweep models at two
# --threads values, and the moondream2 vision case, whose prefill is the B of about 741 width) once with
# the baseline jar and once with the candidate jar, then compares each greedy reply between the builds:
#   1. every check of both runs passes (each run's own exit status);
#   2. each reply (short decode, 512-token prefill, vision) is reported identical or different between the
#      builds. A different reply is printed with both texts and counted, not failed: the integer kernels
#      scale sub-block sums in float lanes, so their rounding differs from the baseline's by design, and a
#      greedy token may move. Model quality is scored by the NLL test; read a difference next to it.
#
# Environment overrides: as smoke-cpu-hot-path.sh (MODELS_DIR, SMOKE_HEAP, VISION_TEST_IMAGE); JUNO_JAR is
# set per run from the flags below. Writes baseline/, candidate/ and compare.json under
# target/cpu-kernels-smoke/<timestamp>/. Exits 0 when both runs pass, 1 otherwise.
#
# Usage:
#   ./scripts/performance-tests/smoke-cpu-kernels.sh --baseline-jar dist/q6k-ab/baseline.jar
#   ./scripts/performance-tests/smoke-cpu-kernels.sh --baseline-jar A.jar --candidate-jar B.jar \
#       --models tinyllama --threads "2 12" --no-vision
set -uo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/target/cpu-kernels-smoke/${RUN_ID}"
BASELINE_JAR=""
CANDIDATE_JAR=""
PASS_THROUGH=()

log() { printf '[cpu-kernels-smoke] %s\n' "$*"; }
die() { printf '[cpu-kernels-smoke] error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-jar) BASELINE_JAR="$2"; shift 2 ;;
    --candidate-jar) CANDIDATE_JAR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --models|--threads|--vision-max-s|--api-port) PASS_THROUGH+=("$1" "$2"); shift 2 ;;
    --no-vision) PASS_THROUGH+=("$1"); shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

[[ -n "$BASELINE_JAR" ]] || die "--baseline-jar is required (the build to compare against)"
[[ -f "$BASELINE_JAR" ]] || die "no such baseline jar: ${BASELINE_JAR}"
if [[ -z "$CANDIDATE_JAR" ]]; then
  jars=( "$ROOT"/juno-player/target/juno-player-*-shaded.jar )
  [[ -f "${jars[0]}" ]] || die "candidate jar missing; pass --candidate-jar or run: mvn package -DskipTests"
  CANDIDATE_JAR="${jars[0]}"
fi
[[ -f "$CANDIDATE_JAR" ]] || die "no such candidate jar: ${CANDIDATE_JAR}"
command -v jq >/dev/null 2>&1 || die "jq is required"
mkdir -p "$OUT"

run_status=()
for side in baseline candidate; do
  jar="$BASELINE_JAR"
  [[ "$side" == candidate ]] && jar="$CANDIDATE_JAR"
  log "${side}: $(sha256sum "$jar" | cut -c1-16) ${jar}"
  JUNO_JAR="$jar" "${PERF_SCRIPTS}/smoke-cpu-hot-path.sh" --out "${OUT}/${side}" "${PASS_THROUGH[@]}" \
    >"${OUT}/${side}.log" 2>&1
  status=$?
  run_status+=("$status")
  if (( status == 0 )); then log "${side}: every check passed"
  else log "${side}: checks failed (exit ${status}); see ${OUT}/${side}.log"; fi
done

# reply <file>: the reply text of a chat or vision response
reply() { jq -r '.choices[0].message.content // .content // ""' "$1" 2>/dev/null; }

identical=0
different=0
rows='[]'
for f in "${OUT}"/candidate/*.short.json "${OUT}"/candidate/*.long.json "${OUT}"/candidate/vision-*.json; do
  [[ -f "$f" ]] || continue
  name="$(basename "$f")"
  b="${OUT}/baseline/${name}"
  [[ -f "$b" ]] || { log "no baseline reply for ${name}"; continue; }
  rb="$(reply "$b")"
  rc="$(reply "$f")"
  if [[ "$rb" == "$rc" ]]; then
    identical=$((identical + 1)); same=true
    log "identical: ${name}"
  else
    different=$((different + 1)); same=false
    log "different: ${name}"
    log "  baseline:  ${rb}"
    log "  candidate: ${rc}"
  fi
  rows="$(jq -c --arg n "$name" --arg b "$rb" --arg c "$rc" --argjson s "$same" \
    '. + [{reply:$n, identical:$s, baseline:$b, candidate:$c}]' <<<"$rows")"
done

jq -n --arg run "$RUN_ID" --arg bj "$BASELINE_JAR" --arg cj "$CANDIDATE_JAR" \
  --argjson bs "${run_status[0]}" --argjson cs "${run_status[1]}" \
  --argjson same "$identical" --argjson diff "$different" --argjson rows "$rows" \
  '{run_id:$run, baseline_jar:$bj, candidate_jar:$cj, baseline_exit:$bs, candidate_exit:$cs,
    replies_identical:$same, replies_different:$diff, replies:$rows}' >"${OUT}/compare.json"
log "replies identical between builds: ${identical}, different: ${different} (${OUT}/compare.json)"
if (( run_status[0] == 0 && run_status[1] == 0 )); then exit 0; fi
exit 1
