#!/usr/bin/env bash
# Same-hour interleaved A/B of CPU --parallel aggregate throughput: before jar (A) against candidate jar (B).
# Usage: JUNO_ROOT=<repo> ab-parallel.sh OUT_DIR JAR_A JAR_B [REPS]
set -euo pipefail
OUT="$1"; JAR_A="$2"; JAR_B="$3"; REPS="${4:-3}"
ROOT="${JUNO_ROOT:?set JUNO_ROOT to the repository root}"
mkdir -p "$OUT"
declare -A HEAP=( [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=4g [qwen2.5-3b-instruct-q4_k_m.gguf]=6g )
for model in tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf qwen2.5-3b-instruct-q4_k_m.gguf; do
  for rep in $(seq 1 "$REPS"); do
    if (( rep % 2 == 1 )); then order="A B"; else order="B A"; fi
    for p in 1 2 4 8; do
      for side in $order; do
        jar="$JAR_A"; [[ "$side" == B ]] && jar="$JAR_B"
        dir="$OUT/${model%%-*}/p$p/$side-rep$rep"
        mkdir -p "$(dirname "$dir")"
        COMPARE_HEAP="${HEAP[$model]}" "$ROOT/scripts/performance-tests/compare-parallel.sh" --cpu --model "$model" \
          --sessions "$p" --parallel-values "$p" --max-tokens 64 --warmup-rounds 2 --juno-jar "$jar" --no-publish --out "$dir" \
          > "$dir.log" 2>&1 || echo "FAILED $dir"
        tps=$(jq -r '.aggregate_token_gen_tps' "$dir/parallel-$p.json" 2>/dev/null || echo null)
        echo "$(date -u +%H:%M:%S) ${model%%-*} p=$p $side rep$rep tps=$tps" | tee -a "$OUT/progress.txt"
      done
    done
  done
done
echo done >> "$OUT/progress.txt"
