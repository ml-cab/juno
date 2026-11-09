#!/usr/bin/env bash
# Every configuration of this reading: TinyLlama 1.1B and Qwen2.5-3B, N = 1, 2, 4 and 8, each once untraced (JFR) and
# once CUDA-traced (nsys), 128 generated tokens per request at a 50-word prompt. Then the trace split per step
# (batch_trace.py, at the batch's last logits row) and the JFR step (batch_jfr.py).
# Usage: run-all.sh OUT_DIR   (JAR may point at another shaded jar; NS="9 16" runs other batch sizes)
set -uo pipefail
OUT="$1"; T="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/runs"
declare -A HEAP=( [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=4g [qwen2.5-3b-instruct-q4_k_m.gguf]=6g )
declare -A LAYERS=( [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=22 [qwen2.5-3b-instruct-q4_k_m.gguf]=36 )
declare -A VOCAB=( [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=32000 [qwen2.5-3b-instruct-q4_k_m.gguf]=151936 )
for model in tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf qwen2.5-3b-instruct-q4_k_m.gguf; do
  for n in ${NS:-1 2 4 8}; do
    p="$OUT/runs/${model%%-*}-n$n"
    TRACE=0 "$T/nsys-batch-decode.sh" "$model" "${HEAP[$model]}" 50 128 "$n" "$p-untraced" > "$p-untraced.out" 2>&1 \
      || echo "FAILED $p-untraced"
    TRACE=1 "$T/nsys-batch-decode.sh" "$model" "${HEAP[$model]}" 50 128 "$n" "$p-traced" > "$p-traced.out" 2>&1 \
      || echo "FAILED $p-traced"
    # nsys can be shut down before it converts its raw stream; convert it here when no report was written.
    [[ ! -f "$p-traced.nsys-rep" && -f "$p-traced.qdstrm" ]] && \
      /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i "$p-traced.qdstrm" >/dev/null 2>&1 && rm -f "$p-traced.qdstrm"
    [[ -f "$p-traced.nsys-rep" ]] && nsys export --type sqlite --force-overwrite true -o "$p-traced.sqlite" \
      "$p-traced.nsys-rep" >/dev/null 2>&1
    # Up to 8 rows the batch downloads its logits one row at a time; wider batches download one N-row block.
    if (( n <= 8 )); then split=(--logit-bytes $(( ${VOCAB[$model]} * 4 )) --every "$n")
    else split=(--logit-bytes $(( n * ${VOCAB[$model]} * 4 )) --every 1); fi
    python3 -I "$T/batch_trace.py" "$p-traced.sqlite" "${LAYERS[$model]}" \
      "${split[@]}" --json "$p-trace.json" > "$p-trace.txt" 2>&1 \
      || echo "TRACE SPLIT FAILED $p"
    python3 -I "$T/batch_jfr.py" "$p-untraced.jfr" "$n" --json "$p-step.json" > "$p-step.txt" 2>&1 \
      || echo "JFR STEP FAILED $p"
    echo "$(date -u +%H:%M:%S) ${model%%-*} n=$n $(jq -c '{step_ms, tokens_per_s}' "$p-step.json" 2>/dev/null) $(jq -c '{wall_ms, device_busy_ms, kernel_ms, copy_ms}' "$p-trace.json" 2>/dev/null)" \
      | tee -a "$OUT/progress.txt"
  done
done
echo done >> "$OUT/progress.txt"
