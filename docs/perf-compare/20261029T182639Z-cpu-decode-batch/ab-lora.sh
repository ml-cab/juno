#!/usr/bin/env bash
# LoRA no-regression gate, each tree's own compare-lora.sh, alternating A B (B A on even runs), GPU lane then CPU lane.
# Usage: ab-lora.sh OUT_DIR TREE_A TREE_B [RUNS]
set -uo pipefail
OUT="$1"; TREE_A="$2"; TREE_B="$3"; RUNS="${4:-3}"
mkdir -p "$OUT"
for lane in gpu cpu; do
  for run in $(seq 1 "$RUNS"); do
    if (( run % 2 == 1 )); then order="A B"; else order="B A"; fi
    for side in $order; do
      tree="$TREE_A"; [[ "$side" == B ]] && tree="$TREE_B"
      dir="$OUT/lora-$lane-$side-$run"
      (cd "$tree" && scripts/performance-tests/compare-lora.sh --$lane --skip-build --no-publish --out "$dir") > "$dir.log" 2>&1 \
        || echo "$(date -u +%H:%M:%S) FAILED $dir" | tee -a "$OUT/progress.txt"
      echo "$(date -u +%H:%M:%S) lora $lane $side run$run done" | tee -a "$OUT/progress.txt"
    done
  done
done
echo done >> "$OUT/progress.txt"
