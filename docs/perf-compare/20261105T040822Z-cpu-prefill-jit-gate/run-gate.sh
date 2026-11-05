#!/usr/bin/env bash
# CPU prefill matmul, no loop-carried receivers: throughput gate (gap-closure plan, hardware validation kit tier,
# out-of-tier change, owner decision (a) of 2026-11-04). Owner run, pinned (prompt-free sudo). Started by
# wait-and-run.sh once READY exists.
#
#   baseline  = baseline-shaded.jar: HEAD 80ec949e (the tree with CpuIntegerGemm.java and its test taken from HEAD by
#               git show; jar 15083b9c320c1c36). Its four-row kernels call add, sub and lane on the loop-carried
#               accumulators f0 to f3, which C2 boxed in some JVMs.
#   candidate = candidate-shaded.jar: this tree (80ec949e plus the change, jar ea700f227c83b911). The accumulators are
#               only ever arguments (f0 = term.add(f0), f0 = p.neg().add(f0)); every output is bit for bit the
#               baseline's. The tree's own shaded jar must equal it.
#
#   Same-hour A/B, pinned: compare-llama-cpp.sh --cpu --pin-clocks --threads 12 against each jar (--juno-jar), A B A B
#   A B, on the four sweep models, two untimed warmups and one measured request per run; scored on the median of three.
#     cpu512  n_prompt 512 (the batched matmul at window width).
#             Gate: Juno CPU pp t/s B/A >= 0.95 on every model.
#     cpu128  n_prompt 128.
#             Gates: pp B/A >= 0.95 and tg B/A >= 0.95 on every model (decode kernels unchanged: a control);
#             allocation per token in the recording <= 1.10x (a side without a reading, or a baseline reading of 0,
#             fails as not scorable); GC pause in the token span <= 1.25x (<= 5 ms under 5 ms).
#     Both:   greedy output (temperature 0) equals the baseline's in every run on every model (the change is meant to
#             be bit for bit).
#   LoRA, pinned by hand (governor performance, turbo off): each tree's own compare-lora.sh, A B A B A B, GPU and CPU
#     (baseline-tree/ for the baseline; its models/ links to this tree's). Gates: training time per pass B/A <= 1.0526
#     (throughput >= 0.95; training's frozen forward runs the float kernels, unchanged: a control), playback
#     wall-clock t/s B/A >= 0.80 (the script's own limit). Medians of three.
#   Every gate compares unrounded values.
#
# About four and a half hours. By hand: sudo -v && bash dist/cpu-prefill-jit-gate/run-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AB=dist/cpu-prefill-jit-gate
N=3
OUT=$AB/runs
PS=scripts/performance-tests
TINY=tinyllama-1.1b-chat-v1.0.Q4_K_M
QWEN=qwen2.5-3b-instruct-q4_k_m
PHI=Phi-3.5-mini-instruct-Q4_K_M
MIS=mistral-7b-instruct-v0.1-q4_k_m
SWEEP=$TINY,$QWEN,$PHI,$MIS
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

[[ -d $AB/baseline-tree/juno-player/target && -e $AB/baseline-tree/models ]] \
  || { echo "missing the built $AB/baseline-tree or its models link" >&2; exit 2; }
OUT_ABS=$(cd "$OUT" && pwd -P)

sudo -v || exit 2
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
restore_clocks() {
  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1
  sudo cpupower frequency-set -g schedutil >/dev/null 2>&1
}
trap 'restore_clocks; kill $KEEPALIVE 2>/dev/null' EXIT
pin_by_hand() {
  sudo cpupower frequency-set -g performance >/dev/null || { echo "cannot set the performance governor" >&2; exit 2; }
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null || { echo "cannot turn turbo off" >&2; exit 2; }
}

status=0
median() { sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", (a > 0 ? b/a : 0)}'; }
le() { awk -v r="$1" -v b="$2" 'BEGIN{exit !(r <= b)}'; }
# Gate on B/A unrounded: ratio() rounds to three places for display only.
ratio_ge() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{exit !(a > 0 && b / a >= t)}'; }
ratio_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{exit !(a > 0 && b / a <= t)}'; }

# alternate TAG MODELS -- FLAGS...: A B A B A B, both jars, compare flags passed through.
alternate() {
  local tag="$1" models="$2"; shift 3
  for i in $(seq 1 "$N"); do
    for side in baseline candidate; do
      echo "=== $tag: $side $i ($(date +%H:%M:%S))"
      "$PS/compare-llama-cpp.sh" "$@" --models "$models" --juno-warmup 2 --juno-reps 1 --reps 1 \
        --no-tuned-lane --no-publish --juno-jar "$AB/$side-shaded.jar" --out "$OUT/$tag-$side-$i" \
        > "$OUT/$tag-$side-$i.log" 2>&1
      local rc=$?
      echo "    exit=$rc  (log: $OUT/$tag-$side-$i.log)"
      [[ $rc -eq 0 ]] || { echo "run failed; see the log" >&2; exit 1; }
    done
  done
}
# read_metric TAG SIDE MODEL JQ-PATH: median over the runs that have a reading.
read_metric() {
  local v
  v=$(for i in $(seq 1 "$N"); do jq -r ".$4 // empty" "$OUT/$1-$2-$i/$3-juno.json" 2>/dev/null; done | median)
  echo "${v:-na}"
}
spread() { # TAG SIDE MODEL JQ-PATH: the N readings, for the record
  for i in $(seq 1 "$N"); do jq -r ".$4 // \"na\"" "$OUT/$1-$2-$i/$3-juno.json" 2>/dev/null; done | tr '\n' ' '
}
pinned_values() {
  for side in baseline candidate; do for i in $(seq 1 "$N"); do
    jq -r '.clock_pinned' "$OUT/$1-$side-$i/host.json"
  done; done | sort -u | tr '\n' ' '
}
# score TAG MODELS FULL(1|0): pp B/A >= 0.95; with FULL also tg, allocation and GC.
score() {
  local tag="$1" models="$2" full="$3" m
  echo
  echo "$tag: clock_pinned across the runs: $(pinned_values "$tag")(must be only: true)"
  [[ $(pinned_values "$tag") == "true " ]] || { echo "  FAIL: $tag not pinned in every run"; status=1; }
  printf '%-12s %9s %9s %7s   %8s %8s %7s   %12s %12s %6s   %7s %7s\n' model pp_A pp_B B/A tg_A tg_B B/A \
    inrec/tok_A inrec/tok_B B/A gc_A gc_B
  IFS=',' read -r -a list <<<"$models"
  for m in "${list[@]}"; do
    local ppa ppb tga tgb aa ab ga gb ar
    ppa=$(read_metric "$tag" baseline "$m" prompt_eval_tps); ppb=$(read_metric "$tag" candidate "$m" prompt_eval_tps)
    tga=$(read_metric "$tag" baseline "$m" token_gen_tps);   tgb=$(read_metric "$tag" candidate "$m" token_gen_tps)
    aa=$(read_metric "$tag" baseline "$m" jfr.allocated_bytes_per_token_in_recording)
    ab=$(read_metric "$tag" candidate "$m" jfr.allocated_bytes_per_token_in_recording)
    ga=$(read_metric "$tag" baseline "$m" jfr.gc_pause_total_ms_in_token_span)
    gb=$(read_metric "$tag" candidate "$m" jfr.gc_pause_total_ms_in_token_span)
    [[ $ppa == na ]] && ppa=0; [[ $ppb == na ]] && ppb=0; [[ $tga == na ]] && tga=0; [[ $tgb == na ]] && tgb=0
    [[ $ga == na ]] && ga=0; [[ $gb == na ]] && gb=0
    ar=na; [[ $aa != na && $ab != na ]] && ar=$(ratio "$aa" "$ab")
    printf '%-12s %9.3f %9.3f %7s   %8.3f %8.3f %7s   %12s %12s %6s   %7.1f %7.1f\n' "${m%%-*}" \
      "$ppa" "$ppb" "$(ratio "$ppa" "$ppb")" "$tga" "$tgb" "$(ratio "$tga" "$tgb")" "${aa%.*}" "${ab%.*}" "$ar" "$ga" "$gb"
    echo "    runs: pp A $(spread "$tag" baseline "$m" prompt_eval_tps)| B $(spread "$tag" candidate "$m" prompt_eval_tps)"
    ratio_ge "$ppa" "$ppb" 0.95 || { echo "  FAIL: ${m%%-*} pp B/A below 0.95"; status=1; }
    if (( full )); then
      echo "    runs: tg A $(spread "$tag" baseline "$m" token_gen_tps)| B $(spread "$tag" candidate "$m" token_gen_tps)"
      ratio_ge "$tga" "$tgb" 0.95 || { echo "  FAIL: ${m%%-*} tg B/A below 0.95"; status=1; }
      if [[ $ar == na ]] || ! awk -v a="$aa" 'BEGIN{exit !(a > 0)}'; then
        echo "  FAIL: ${m%%-*} allocation in recording not scorable (A $aa, B $ab)"; status=1
      else
        ratio_le "$aa" "$ab" 1.10 || { echo "  FAIL: ${m%%-*} allocation per token in recording above 1.10x"; status=1; }
      fi
      if awk -v a="$ga" 'BEGIN{exit !(a < 5)}'; then
        le "$gb" 5 || { echo "  FAIL: ${m%%-*} GC pause in span above 5 ms"; status=1; }
      else
        awk -v a="$ga" -v b="$gb" 'BEGIN{exit !(b <= 1.25 * a)}' || { echo "  FAIL: ${m%%-*} GC pause in span above 1.25x"; status=1; }
      fi
    fi
  done
}
greedy() { # TAG MODELS: candidate text equals the baseline's in every run
  local tag="$1" models="$2" m i a b
  echo
  echo "$tag: greedy output (temperature 0), candidate against baseline per run"
  IFS=',' read -r -a list <<<"$models"
  for m in "${list[@]}"; do
    local same=0
    for i in $(seq 1 "$N"); do
      a=$(jq -r '.choices[0].message.content' "$OUT/$tag-baseline-$i/$m-generate-rep1-juno-response.json" 2>/dev/null)
      b=$(jq -r '.choices[0].message.content' "$OUT/$tag-candidate-$i/$m-generate-rep1-juno-response.json" 2>/dev/null)
      [[ -n "$a" && "$a" == "$b" ]] && same=$((same + 1))
    done
    echo "  ${m%%-*}: equal in $same of $N runs"
    (( same == N )) || { echo "  FAIL: ${m%%-*} greedy output differs from the baseline"; status=1; }
  done
}

echo
echo "=== CPU same-hour A/B, pinned ($(date +%H:%M:%S))"
alternate cpu512 "$SWEEP" -- --cpu --pin-clocks --threads 12 --n-prompt 512
score cpu512 "$SWEEP" 0
greedy cpu512 "$SWEEP"
alternate cpu128 "$SWEEP" -- --cpu --pin-clocks --threads 12 --n-prompt 128
score cpu128 "$SWEEP" 1
greedy cpu128 "$SWEEP"

echo
echo "=== LoRA, pinned by hand ($(date +%H:%M:%S))"
pin_by_hand
for lane in gpu cpu; do
  for i in $(seq 1 "$N"); do
    for side in baseline candidate; do
      tree=$([[ $side == baseline ]] && echo "$AB/baseline-tree" || echo ".")
      echo "--- lora $lane: $side $i ($(date +%H:%M:%S))"
      ( cd "$tree" && scripts/performance-tests/compare-lora.sh --$lane --reps 1 --skip-build --no-publish \
          --out "$OUT_ABS/lora-$lane-$side-$i" ) > "$OUT/lora-$lane-$side-$i.log" 2>&1 \
        || { echo "    run failed; see $OUT/lora-$lane-$side-$i.log"; status=1; }
    done
  done
  lora() { for i in $(seq 1 "$N"); do jq -r ".$3 // empty" "$OUT/lora-$lane-$1-$i/current.json" 2>/dev/null; done | median; }
  ta=$(lora baseline "$lane" train.ms_per_pass); tb=$(lora candidate "$lane" train.ms_per_pass)
  pa=$(lora baseline "$lane" playback.tps); pb=$(lora candidate "$lane" playback.tps)
  echo "lora $lane: train ms/pass A ${ta:-na} B ${tb:-na} B/A $(ratio "${ta:-0}" "${tb:-0}"); playback t/s A ${pa:-na} B ${pb:-na} B/A $(ratio "${pa:-0}" "${pb:-0}")"
  ratio_le "${ta:-0}" "${tb:-0}" 1.0526 || { echo "  FAIL: lora $lane training time per pass B/A above 1.0526"; status=1; }
  ratio_ge "${pa:-0}" "${pb:-0}" 0.80 || { echo "  FAIL: lora $lane playback B/A below 0.80"; status=1; }
done
restore_clocks

echo
if (( status == 0 )); then echo "GATE: PASS"; else echo "GATE: FAIL"; fi
exit $status
