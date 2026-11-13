#!/usr/bin/env bash
# Same-session A/B, unpinned: TinyLlama continuous lanes (F16 and q8_0 KV) through this tree's compare
# script with each side's jar, then compare-lora.sh CPU and GPU from each side's tree. A B A B A B.
set -uo pipefail
S=<scratch>
CAND_TREE=<repo>
BASE_TREE=$S/6ad5c728
OUT=$S/ab
mkdir -p "$OUT"
declare -A JAR=([baseline]=$BASE_TREE/juno-player/target/juno-player-0.1.2-shaded.jar
                [candidate]=$CAND_TREE/juno-player/target/juno-player-0.1.2-shaded.jar)
declare -A TREE=([baseline]=$BASE_TREE [candidate]=$CAND_TREE)
cd "$CAND_TREE"
for lane in f16; do
  extra=(); [[ $lane == q8_0 ]] && extra=(--cache-type-k q8_0 --cache-type-v q8_0)
  for i in 1 2 3; do
    for side in baseline candidate; do
      echo "[ab] continuous-$lane $side $i $(date -u +%H:%M:%S)"
      scripts/performance-tests/compare-llama-cpp.sh --cpu --models tinyllama-1.1b-chat-v1.0.Q4_K_M --n-prompt 512 --schedule continuous \
        "${extra[@]}" --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
        --juno-jar "${JAR[$side]}" --out "$OUT/continuous-$lane-$side-$i" > "$OUT/continuous-$lane-$side-$i.log" 2>&1
      echo "[ab]   exit $?"
    done
  done
done
for dev in cpu gpu; do
  for i in 1 2 3; do
    for side in baseline candidate; do
      echo "[ab] lora-$dev $side $i $(date -u +%H:%M:%S)"
      (cd "${TREE[$side]}" && scripts/performance-tests/compare-lora.sh --$dev --skip-build --no-publish \
        --out "$OUT/lora-$dev-$side-$i" > "$OUT/lora-$dev-$side-$i.log" 2>&1)
      echo "[ab]   exit $?"
    done
  done
done
echo "[ab] done $(date -u +%H:%M:%S)"
