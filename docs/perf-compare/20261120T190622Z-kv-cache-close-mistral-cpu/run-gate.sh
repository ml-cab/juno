#!/usr/bin/env bash
# KV cache tier close, owner decision 21 (b): the standing gate's CPU lane re-run on Mistral 7B alone, pinned. In the
# full gate (dist/kv-cache-close/, 2026-11-20) every other row met every bar; this row was not scored, because the
# harness's span check withheld Mistral 7B's CPU generation reading on all three baseline runs (the token span's
# residual a few ms past its -25 ms floor in 34-second requests).
#
#   baseline  = baseline-shaded.jar: the pre-tier jar (e16748ad, git archive, mvn package), sha256 21109a3a8fd317c8.
#   candidate = candidate-shaded.jar: the tier's final build, sha256 2ceed60ef9139b7a, the same jars as the full gate.
#               The tree's own shaded jar must equal it, so do not rebuild the tree before this has run.
#
#   Part C only, Mistral 7B only: A B A B A B, compare-llama-cpp.sh --cpu --pin-clocks --n-prompt 128, 64 generated
#   tokens, three cycles; the same scorer and bars as the full gate (tg and pp t/s B/A >= 0.95, allocation per token
#   <= 1.10x, GC pause in the token span <= 1.25x or <= 5 ms under a 5 ms baseline). When the span check withholds a
#   run, the scorer also prints the request-timed and decode-derived medians; the scored row is the harness's
#   token_gen_tps median over the runs it did not withhold.
#
# About 30 minutes. Normally started by wait-and-run.sh; by hand: sudo -v && bash dist/kv-cache-close-mistral-cpu/run-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AB=dist/kv-cache-close-mistral-cpu
N=3
OUT=$AB/runs
PS=scripts/performance-tests
SWEEP=mistral-7b-instruct-v0.1-q4_k_m
mkdir -p "$OUT"

for j in baseline candidate; do
  [[ -f "$AB/$j-shaded.jar" ]] || { echo "missing $AB/$j-shaded.jar" >&2; exit 2; }
  echo "$j jar sha256: $(sha256sum "$AB/$j-shaded.jar" | cut -c1-16)"
done
tree_jar=$(ls juno-player/target/juno-player-*-shaded.jar 2>/dev/null | head -1)
if [[ -z "$tree_jar" ]] || ! cmp -s "$tree_jar" "$AB/candidate-shaded.jar"; then
  echo "tree jar ($tree_jar) is not the candidate jar; rebuild (mvn package -DskipTests) and re-copy it" >&2
  exit 2
fi
bash dist/kv-cache-close/tree-hash.sh | tee "$OUT/tree.txt"

sudo -v || exit 2
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

status=0
median() { sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", (a > 0 ? b/a : 0)}'; }
ge() { awk -v r="$1" -v b="$2" 'BEGIN{exit !(r >= b)}'; }

# alternate TAG FLAGS...: A B A B A B, both jars, extra compare flags passed through.
alternate() {
  local tag="$1"; shift
  for i in $(seq 1 "$N"); do
    for side in baseline candidate; do
      echo "=== $tag: $side $i ($(date +%H:%M:%S))"
      "$PS/compare-llama-cpp.sh" "$@" --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
        --juno-jar "$AB/$side-shaded.jar" --out "$OUT/$tag-$side-$i" > "$OUT/$tag-$side-$i.log" 2>&1
      local rc=$?
      echo "    exit=$rc  (log: $OUT/$tag-$side-$i.log)"
      [[ $rc -eq 0 ]] || { echo "run failed; see the log" >&2; exit 1; }
    done
  done
}
# read_metric TAG SIDE MODEL JQ-PATH [FILE-SUFFIX]: median over the runs that have a reading. A reading the harness
# withheld (null: its span check refused the repetition) is left out, never counted as 0; with none left, "na".
read_metric() {
  local v suffix=${5:-juno}
  v=$(for i in $(seq 1 "$N"); do jq -r ".$4 // empty" "$OUT/$1-$2-$i/$3-$suffix.json"; done | median)
  echo "${v:-na}"
}
# count_metric TAG SIDE MODEL JQ-PATH: how many of the N runs have a reading.
count_metric() {
  for i in $(seq 1 "$N"); do jq -r ".$4 // empty" "$OUT/$1-$2-$i/$3-juno.json"; done | grep -c .
}
pinned_values() { # tag
  for side in baseline candidate; do for i in $(seq 1 "$N"); do
    jq -r '.clock_pinned' "$OUT/$1-$side-$i/host.json"
  done; done | sort -u | tr '\n' ' '
}

# score_throughput TAG MODELS: t/s B/A >= 0.95, then allocation and GC.
score_throughput() {
  local tag="$1" models="$2" m
  echo
  echo "$tag: clock_pinned across the six runs: $(pinned_values "$tag")(must be only: true)"
  printf '%-12s %10s %10s %7s   %8s %8s %7s   %12s %12s %6s   %7s %7s\n' model pp_A pp_B pp_B/A tg_A tg_B tg_B/A \
    alloc/tok_A alloc/tok_B B/A gc_A gc_B
  IFS=',' read -r -a list <<<"$models"
  for m in "${list[@]}"; do
    local ppa ppb tga tgb aa ab ga gb ppr tgr ar
    ppa=$(read_metric "$tag" baseline "$m" prompt_eval_tps); ppb=$(read_metric "$tag" candidate "$m" prompt_eval_tps)
    tga=$(read_metric "$tag" baseline "$m" token_gen_tps);   tgb=$(read_metric "$tag" candidate "$m" token_gen_tps)
    aa=$(read_metric "$tag" baseline "$m" jfr.allocated_bytes_per_token_in_recording)
    ab=$(read_metric "$tag" candidate "$m" jfr.allocated_bytes_per_token_in_recording)
    ga=$(read_metric "$tag" baseline "$m" jfr.gc_pause_total_ms_in_token_span)
    gb=$(read_metric "$tag" candidate "$m" jfr.gc_pause_total_ms_in_token_span)
    local tgna tgnb
    tgna=$(count_metric "$tag" baseline "$m" token_gen_tps); tgnb=$(count_metric "$tag" candidate "$m" token_gen_tps)
    if [[ $tga == na || $tgb == na ]]; then
      echo "  ${m%%-*}: generation withheld by the harness on every run of one side ($tgna and $tgnb of $N scorable);" \
        "request-timed api_token_gen_tps median: $(read_metric "$tag" baseline "$m" api_token_gen_tps) ->" \
        "$(read_metric "$tag" candidate "$m" api_token_gen_tps); decode-derived (forward-pass totals) median:" \
        "$(read_metric "$tag" baseline "$m" jfr.token_gen_tps_decode_derived) -> $(read_metric "$tag" candidate "$m" jfr.token_gen_tps_decode_derived)"
      tga=0; tgb=0
    elif [[ $tgna -lt $N || $tgnb -lt $N ]]; then
      echo "  ${m%%-*}: generation median over $tgna (baseline) and $tgnb (candidate) of $N runs; the rest were" \
        "withheld by the harness's span check. Request-timed api_token_gen_tps median of $N:" \
        "$(read_metric "$tag" baseline "$m" api_token_gen_tps) -> $(read_metric "$tag" candidate "$m" api_token_gen_tps);" \
        "decode-derived (forward-pass totals) median: $(read_metric "$tag" baseline "$m" jfr.token_gen_tps_decode_derived) ->" \
        "$(read_metric "$tag" candidate "$m" jfr.token_gen_tps_decode_derived)"
    fi
    if [[ $aa == na || $ab == na ]]; then
      echo "  ${m%%-*}: allocation per token in the recording has no reading on one side (not scorable)"
      aa=0; ab=0
    fi
    [[ $ga == na ]] && ga=0
    [[ $gb == na ]] && gb=0
    ppr=$(ratio "$ppa" "$ppb"); tgr=$(ratio "$tga" "$tgb"); ar=$(ratio "$aa" "$ab")
    printf '%-12s %10.2f %10.2f %7s   %8.2f %8.2f %7s   %12.0f %12.0f %6s   %7.1f %7.1f\n' "${m%%-*}" \
      "$ppa" "$ppb" "$ppr" "$tga" "$tgb" "$tgr" "$aa" "$ab" "$ar" "$ga" "$gb"
    ge "$ppr" 0.95 || { echo "  FAIL: ${m%%-*} prefill below 0.95x"; status=1; }
    ge "$tgr" 0.95 || { echo "  FAIL: ${m%%-*} generation below 0.95x"; status=1; }
    if awk -v a="$aa" 'BEGIN{exit !(a > 0)}'; then
      awk -v r="$ar" 'BEGIN{exit !(r <= 1.10)}' || { echo "  FAIL: ${m%%-*} allocation per token above 1.10x"; status=1; }
    else
      echo "  FAIL: ${m%%-*} allocation per token not scorable (a side without a reading, or a baseline of 0)"; status=1
    fi
    if awk -v a="$ga" 'BEGIN{exit !(a < 5)}'; then
      awk -v b="$gb" 'BEGIN{exit !(b <= 5)}' || { echo "  FAIL: ${m%%-*} GC pause in span above 5 ms"; status=1; }
    else
      awk -v a="$ga" -v b="$gb" 'BEGIN{exit !(b <= 1.25 * a)}' || { echo "  FAIL: ${m%%-*} GC pause in span above 1.25x"; status=1; }
    fi
  done
}

# reference_ratios TAG MODELS: the candidate's median Juno t/s over the median of the reference tool's t/s in the
# candidate's own runs (the reference tool ran once per invocation). Reported, never gated.
reference_ratios() {
  local tag="$1" models="$2" m
  echo
  echo "$tag: candidate against the reference tool (median of $N each, same runs), reported not gated"
  printf '%-12s %10s %10s %7s   %8s %8s %7s\n' model pp_juno pp_ref pp_x tg_juno tg_ref tg_x
  IFS=',' read -r -a list <<<"$models"
  for m in "${list[@]}"; do
    local pj pr tj tr
    pj=$(read_metric "$tag" candidate "$m" prompt_eval_tps); pr=$(read_metric "$tag" candidate "$m" prompt_eval_tps llama-cpp)
    tj=$(read_metric "$tag" candidate "$m" token_gen_tps);   tr=$(read_metric "$tag" candidate "$m" token_gen_tps llama-cpp)
    [[ $pj == na || $pr == na || $tj == na || $tr == na ]] && { echo "  ${m%%-*}: a reading is missing ($pj $pr $tj $tr)"; continue; }
    printf '%-12s %10.2f %10.2f %7s   %8.2f %8.2f %7s\n' "${m%%-*}" "$pj" "$pr" "$(ratio "$pr" "$pj")" "$tj" "$tr" "$(ratio "$tr" "$tj")"
  done
}

# greedy_report TAG MODELS: each candidate run's generated text against the baseline run of the same index, and the
# candidate's own runs against its run 1. Reported, not gated (see the header).
greedy_report() {
  local tag="$1" models="$2" m i a b c1
  echo
  echo "$tag: greedy output (temperature 0), reported not gated: candidate vs baseline per run; candidate run 1 vs runs 1..$N"
  IFS=',' read -r -a list <<<"$models"
  for m in "${list[@]}"; do
    local same=0 self=0
    c1=$(jq -r '.choices[0].message.content' "$OUT/$tag-candidate-1/$m-generate-rep1-juno-response.json")
    for i in $(seq 1 "$N"); do
      a=$(jq -r '.choices[0].message.content' "$OUT/$tag-baseline-$i/$m-generate-rep1-juno-response.json")
      b=$(jq -r '.choices[0].message.content' "$OUT/$tag-candidate-$i/$m-generate-rep1-juno-response.json")
      [[ "$a" == "$b" ]] && same=$((same + 1))
      [[ "$b" == "$c1" ]] && self=$((self + 1))
    done
    echo "  ${m%%-*}: candidate equals baseline in $same of $N runs; candidate equals its run 1 in $self of $N"
  done
}

hot_methods() { # TAG FIELD TITLE
  echo
  echo "$3, run 1 of each jar, reported not gated:"
  local m side
  IFS=',' read -r -a list <<<"$SWEEP"
  for m in "${list[@]}"; do
    for side in baseline candidate; do
      echo "--- ${m%%-*} $side"
      jq -r --arg f "$2" '.jfr[$f][0:10][]? | "\(.samples)\t\(.method)"' "$OUT/$1-$side-1/$m-juno.json"
    done
  done
}

alternate cpu-ab --cpu --pin-clocks --models "$SWEEP" --n-prompt 128
score_throughput cpu-ab "$SWEEP"
reference_ratios cpu-ab "$SWEEP"
greedy_report cpu-ab "$SWEEP"
hot_methods cpu-ab top_methods "CPU lane, top 10 hot methods (jdk.ExecutionSample)"

[[ $status -eq 0 ]] && echo "GATES MET" || echo "GATES MISSED (see FAIL lines)"
exit $status
