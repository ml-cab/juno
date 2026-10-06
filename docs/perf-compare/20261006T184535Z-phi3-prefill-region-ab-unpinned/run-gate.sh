#!/usr/bin/env bash
# Same-hour A/B: Phi-3's prefill-window RoPE and attention moved inside the device region.
# A = baseline jar (5e4c913, Phi-3 rotates and attends on the host side of the region),
# B = candidate jar (this working tree), A B A B A B, Phi-3.5-mini only, n_prompt=512,
# GPU attention and the prefill region at their defaults. Then, Juno t/s medians of three:
#   pp B/A >= 0.95 (the tier's no-regression bound, read here for the model the change touches)
#   tg B/A >= 0.95 (the change does not touch decode; read to confirm it)
# Pinned by default; PIN=0 runs unpinned (informational, not scorable against the bound).
# Lives under dist/ (not target/) so a root `mvn clean` does not delete it, the jars or the runs.
# Run from anywhere: bash dist/phi3-prefill-region-ab/run-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AB=dist/phi3-prefill-region-ab
PIN=${PIN:-1}
TAG=phi3-prefill-region-ab$([[ $PIN == 1 ]] || echo -unpinned)
PINFLAG=$([[ $PIN == 1 ]] && echo --pin-clocks)
OUT=$AB/runs
MODEL=Phi-3.5-mini-instruct-Q4_K_M
mkdir -p "$OUT"

for side in baseline candidate; do
  [[ -f "$AB/$side-shaded.jar" ]] || { echo "missing $AB/$side-shaded.jar" >&2; exit 2; }
  echo "$side jar sha256: $(sha256sum "$AB/$side-shaded.jar" | cut -c1-16)"
done

if [[ $PIN == 1 ]]; then
  sudo -v || exit 2
  ( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
  KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null' EXIT
fi

for i in 1 2 3; do
  for side in baseline candidate; do
    echo "=== $side $i ($(date +%H:%M:%S))"
    scripts/performance-tests/compare-llama-cpp.sh --gpu $PINFLAG --models "$MODEL" \
      --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
      --juno-jar "$AB/$side-shaded.jar" --out "$OUT/$TAG-$side-$i" \
      > "$OUT/$TAG-$side-$i.log" 2>&1
    rc=$?
    echo "    exit=$rc  (log: $OUT/$TAG-$side-$i.log)"
    [[ $rc -eq 0 ]] || { echo "run failed; see the log" >&2; exit 1; }
  done
done

median() { sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
read_metric() { # side field
  for i in 1 2 3; do jq -r ".$2" "$OUT/$TAG-$1-$i/$MODEL-juno.json"; done | median
}
all() { # side field: the three readings
  for i in 1 2 3; do jq -r ".$2" "$OUT/$TAG-$1-$i/$MODEL-juno.json"; done | tr '\n' ' '
}
pinned() {
  for side in baseline candidate; do for i in 1 2 3; do
    jq -r '.clock_pinned' "$OUT/$TAG-$side-$i/host.json"
  done; done | sort -u | tr '\n' ' '
}

echo
echo "clock_pinned values across all six runs: $(pinned)(must be only: true for a gate reading)"
status=0
for f in prompt_eval_tps token_gen_tps; do
  a=$(read_metric baseline $f); b=$(read_metric candidate $f)
  r=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.3f", b/a}')
  printf '%-16s baseline %8.2f [%s]  candidate %8.2f [%s]  B/A %s (bound 0.95)\n' "$f" "$a" "$(all baseline $f)" "$b" "$(all candidate $f)" "$r"
  awk -v r="$r" 'BEGIN{exit !(r >= 0.95)}' || { echo "  BELOW 0.95: $f"; status=1; }
done
[[ $status -eq 0 ]] && echo "BOUND MET" || echo "BOUND MISSED"
exit $status
