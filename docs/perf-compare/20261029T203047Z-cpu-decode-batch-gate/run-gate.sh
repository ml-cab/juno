#!/usr/bin/env bash
# CPU decode-batch matmul: throughput gate (gap-closure plan, hardware validation kit tier, out-of-tier change). Owner
# run, pinned (prompt-free sudo). Started by wait-and-run.sh once READY exists.
#
#   baseline  = baseline-shaded.jar: HEAD 06d6d22d, built from git archive (jar 758520d7ad418890). A CPU decode batch
#               runs the prefill gemm (CpuIntegerGemm), whose rows differ from the single-request kernel's in the last
#               bits.
#   candidate = candidate-shaded.jar: this tree (06d6d22d plus the uncommitted decode-batch matmul, jar
#               1c181de29e52d249). A CPU decode batch runs CpuIntegerDecodeBatch and the serial-head attention, bit for
#               bit the single-request decode. The tree's own shaded jar must equal it.
#
#   Same-hour A/B, pinned by hand (governor performance, turbo off): this tree's compare-parallel.sh against each jar
#   (--juno-jar), A B A B A B, on TinyLlama and Qwen2.5-3B at --parallel 1, 2, 4 and 8 with as many concurrent sessions,
#   64 tokens each, greedy, two untimed warmup rounds per server.
#   Gates: CPU aggregate token-generation t/s, median of three, B/A >= 0.90 at --parallel 2, 4 and 8 on both models
#   (unrounded); every candidate reply at --parallel 2, 4 and 8 equals the candidate's single-request reply.
#   Reported only: --parallel 1 (both builds run the single-request path there), the baseline's replies that differ.
#
# About 2.5 hours. By hand: sudo -v && bash dist/cpu-decode-batch-gate/run-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AB=dist/cpu-decode-batch-gate
N=3
OUT=$AB/runs
PS=scripts/performance-tests
MODELS=(tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf qwen2.5-3b-instruct-q4_k_m.gguf)
declare -A HEAP=([tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=4g [qwen2.5-3b-instruct-q4_k_m.gguf]=6g)
PARALLEL=(1 2 4 8)
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd -P)

for j in baseline candidate; do
  [[ -f "$AB/$j-shaded.jar" ]] || { echo "missing $AB/$j-shaded.jar" >&2; exit 2; }
  echo "$j jar sha256: $(sha256sum "$AB/$j-shaded.jar" | cut -c1-16)"
done
tree_jar=$(ls juno-player/target/juno-player-*-shaded.jar 2>/dev/null | head -1)
if [[ -z "$tree_jar" ]] || ! cmp -s "$tree_jar" "$AB/candidate-shaded.jar"; then
  echo "tree jar ($tree_jar) is not the candidate jar; rebuild (mvn package -DskipTests) and re-copy it" >&2
  exit 2
fi

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
clock_state() {
  echo "$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | tr '\n' ',')no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
}

status=0
median() { sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", (a > 0 ? b/a : 0)}'; }
# Gate on B/A unrounded: ratio() rounds to three places for display only.
ratio_ge() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{exit !(a > 0 && b / a >= t)}'; }
short() { echo "${1%%-*}"; }

echo
echo "=== CPU decode-batch A/B, pinned by hand ($(date +%H:%M:%S))"
pin_by_hand
: > "$OUT/clocks.txt"
for model in "${MODELS[@]}"; do
  m=$(short "$model")
  for i in $(seq 1 "$N"); do
    for p in "${PARALLEL[@]}"; do
      for side in baseline candidate; do
        dir="$OUT_ABS/$m-p$p-$side-$i"
        echo "--- $m --parallel $p: $side $i ($(date +%H:%M:%S))"
        echo "$m-p$p-$side-$i $(clock_state)" >> "$OUT/clocks.txt"
        COMPARE_HEAP="${HEAP[$model]}" "$PS/compare-parallel.sh" --cpu --model "$model" --sessions "$p" \
          --parallel-values "$p" --max-tokens 64 --warmup-rounds 2 --juno-jar "$AB/$side-shaded.jar" --no-publish \
          --out "$dir" > "$dir.log" 2>&1
        rc=$?
        ok=$(jq -r --argjson n $((64 * p)) \
          'if .status == "success" and .sessions_failed == 0 and .total_completion_tokens == $n then "ok" else "bad" end' \
          "$dir/parallel-$p.json" 2>/dev/null)
        echo "    exit=$rc reading=${ok:-none} tps=$(jq -r '.aggregate_token_gen_tps' "$dir/parallel-$p.json" 2>/dev/null)"
        [[ $rc -eq 0 && $ok == ok ]] || { echo "run failed; see $dir.log" >&2; exit 1; }
      done
    done
  done
done
restore_clocks
echo "clock states recorded before each run (must be only: performance,no_turbo=1):"
awk '{print $2}' "$OUT/clocks.txt" | sort | uniq -c

tps() { # MODEL P SIDE: the N readings
  for i in $(seq 1 "$N"); do jq -r '.aggregate_token_gen_tps' "$OUT/$1-p$2-$3-$i/parallel-$2.json"; done
}
reply() { jq -r '.choices[0].message.content' "$1"; }

echo
printf '%-10s %-4s %-26s %-26s %8s %6s\n' model P baseline_runs candidate_runs B/A gate
for model in "${MODELS[@]}"; do
  m=$(short "$model")
  for p in "${PARALLEL[@]}"; do
    a=$(tps "$m" "$p" baseline | median); b=$(tps "$m" "$p" candidate | median)
    gate=$([[ $p == 1 ]] && echo report || echo 0.90)
    printf '%-10s %-4s %-26s %-26s %8s %6s\n' "$m" "$p" "$(tps "$m" "$p" baseline | tr '\n' ' ')" \
      "$(tps "$m" "$p" candidate | tr '\n' ' ')" "$(ratio "$a" "$b")" "$gate"
    if [[ $p != 1 ]]; then
      ratio_ge "$a" "$b" 0.90 || { echo "  FAIL: $m --parallel $p aggregate t/s B/A $(ratio "$a" "$b"), below 0.90"; status=1; }
    fi
  done
done

echo
echo "Replies against each build's single-request reply (--parallel 1, run 1):"
for model in "${MODELS[@]}"; do
  m=$(short "$model")
  for side in baseline candidate; do
    single=$(reply "$OUT/$m-p1-$side-1/parallel-1-responses/resp-1.json")
    line="  $m $side:"
    for p in 2 4 8; do
      diff=0; total=0
      for f in "$OUT"/"$m"-p"$p"-"$side"-*/parallel-"$p"-responses/resp-*.json; do
        total=$((total + 1)); [[ "$(reply "$f")" == "$single" ]] || diff=$((diff + 1))
      done
      line+=" --parallel $p $diff of $total differ;"
      if [[ $side == candidate && $diff -ne 0 ]]; then
        echo "  FAIL: $m candidate --parallel $p: $diff of $total replies differ from its single-request reply"
        status=1
      fi
    done
    echo "$line"
  done
done

[[ $status -eq 0 ]] && echo "GATES MET" || echo "GATES MISSED (see FAIL lines)"
exit $status
