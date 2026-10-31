#!/usr/bin/env bash
# GPU, one build: 8 concurrent clients at --parallel 1 (each request decoded on its own) against --parallel 8 (one
# decode batch). Alternates the flag, 1 8 on odd repetitions and 8 1 on even, three repetitions per model.
# Usage: JUNO_ROOT=<repo> concurrency-parallel-gpu.sh OUT_DIR JAR [REPS]
set -uo pipefail
OUT="$1"; JAR="$2"; REPS="${3:-3}"
ROOT="${JUNO_ROOT:?set JUNO_ROOT to the repository root}"
mkdir -p "$OUT"
declare -A HEAP=( [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=4g [qwen2.5-3b-instruct-q4_k_m.gguf]=6g )
for model in tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf qwen2.5-3b-instruct-q4_k_m.gguf; do
  for rep in $(seq 1 "$REPS"); do
    if (( rep % 2 == 1 )); then order="1 8"; else order="8 1"; fi
    for p in $order; do
      dir="$OUT/${model%%-*}/p$p-rep$rep"
      mkdir -p "$(dirname "$dir")"
      COMPARE_HEAP="${HEAP[$model]}" "$ROOT/scripts/performance-tests/compare-parallel.sh" --gpu --model "$model" \
        --sessions 8 --parallel-values "$p" --max-tokens 64 --warmup-rounds 2 --juno-jar "$JAR" --no-publish \
        --out "$dir" > "$dir.log" 2>&1 || echo "FAILED $dir"
      line=$(jq -r '"tps=\(.aggregate_token_gen_tps) mean_latency_ms=\(.mean_per_session_latency_ms) wall_ms=\(.wall_ms) failed=\(.sessions_failed) tokens=\(.total_completion_tokens)"' "$dir/parallel-$p.json" 2>/dev/null || echo "no reading")
      echo "$(date -u +%H:%M:%S) ${model%%-*} p=$p rep$rep $line" | tee -a "$OUT/progress.txt"
    done
  done
done
echo done >> "$OUT/progress.txt"
