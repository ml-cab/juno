#!/usr/bin/env bash
# Same-hour A/B: the whole decode layer inside the residency region on the LLaMA family, Phi-3 and Qwen3.
# One jar (the candidate), the flag alternated: A = --gpu-residency off, B = --gpu-residency on,
# A B A B A B, GPU attention at its default (auto: on under CUDA). Then scores, Juno t/s medians:
#   tg B/A >= 1.00 on every model where the region runs (TinyLlama, Mistral 7B, Phi-3.5-mini, Qwen3-1.7B)
#   tg B/A >= 0.95 everywhere (Qwen2.5-3B declines the region: split-half RoPE and Q/K/V biases in
#   the LLaMA-family handler; it reads the flag's cost where it does nothing)
#   pp B/A recorded, not gated (the region is decode-only)
# Pinned by default; PIN=0 runs unpinned (informational, not scorable against the gate).
# Lives under dist/ (not target/) so a root `mvn clean` does not delete it, the jar or the runs.
# Run from anywhere: bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AB=dist/gpu-residency-phi3-qwen3-ab
PIN=${PIN:-1}
TAG=residency-phi3-qwen3-ab$([[ $PIN == 1 ]] || echo -unpinned)
PINFLAG=$([[ $PIN == 1 ]] && echo --pin-clocks)
OUT=$AB/runs
JAR=$AB/candidate-shaded.jar
REGION_MODELS=(tinyllama-1.1b-chat-v1.0.Q4_K_M mistral-7b-instruct-v0.1-q4_k_m Phi-3.5-mini-instruct-Q4_K_M Qwen3-1.7B-Q4_K_M)
OTHER_MODELS=(qwen2.5-3b-instruct-q4_k_m)
MODELS=$(IFS=,; echo "${REGION_MODELS[*]},${OTHER_MODELS[*]}")
mkdir -p "$OUT"

[[ -f "$JAR" ]] || { echo "missing $JAR" >&2; exit 2; }
echo "candidate jar sha256: $(sha256sum "$JAR" | cut -c1-16)"

if [[ $PIN == 1 ]]; then
  # --pin-clocks needs sudo for the whole run (about 20-30 minutes for five models); keep the credential warm.
  sudo -v || exit 2
  ( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
  KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null' EXIT
fi

for i in 1 2 3; do
  for side in off on; do
    echo "=== region $side $i ($(date +%H:%M:%S))"
    scripts/performance-tests/compare-llama-cpp.sh --gpu $PINFLAG --models "$MODELS" \
      --n-prompt 128 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
      --gpu-residency "$side" --juno-jar "$JAR" --out "$OUT/$TAG-$side-$i" \
      > "$OUT/$TAG-$side-$i.log" 2>&1
    rc=$?
    echo "    exit=$rc  (log: $OUT/$TAG-$side-$i.log)"
    [[ $rc -eq 0 ]] || { echo "run failed; see the log" >&2; exit 1; }
  done
done

median() { sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
read_metric() { # side model field
  for i in 1 2 3; do jq -r ".$3" "$OUT/$TAG-$1-$i/$2-juno.json"; done | median
}
pinned() { # every run must say it was pinned
  for side in off on; do for i in 1 2 3; do
    jq -r '.clock_pinned' "$OUT/$TAG-$side-$i/host.json"
  done; done | sort -u | tr '\n' ' '
}

echo
echo "clock_pinned values across all six runs: $(pinned)(must be only: true for a gate reading)"
printf '%-12s %10s %10s %7s   %8s %8s %7s %6s\n' model pp_off pp_on pp_on/off tg_off tg_on tg_on/off bound
status=0
for m in "${REGION_MODELS[@]}" "${OTHER_MODELS[@]}"; do
  bound=0.95
  [[ " ${REGION_MODELS[*]} " == *" $m "* ]] && bound=1.00
  ppa=$(read_metric off "$m" prompt_eval_tps); ppb=$(read_metric on "$m" prompt_eval_tps)
  tga=$(read_metric off "$m" token_gen_tps);   tgb=$(read_metric on "$m" token_gen_tps)
  ppr=$(awk -v a="$ppa" -v b="$ppb" 'BEGIN{printf "%.3f", b/a}')
  tgr=$(awk -v a="$tga" -v b="$tgb" 'BEGIN{printf "%.3f", b/a}')
  printf '%-12s %10.2f %10.2f %7s   %8.2f %8.2f %7s %6s\n' "${m%%-*}" "$ppa" "$ppb" "$ppr" "$tga" "$tgb" "$tgr" "$bound"
  awk -v r="$tgr" -v b="$bound" 'BEGIN{exit !(r >= b)}' || { echo "  FAIL: ${m%%-*} generation below ${bound}x"; status=1; }
done
[[ $status -eq 0 ]] && echo "GATE MET" || echo "GATE MISSED"
exit $status
