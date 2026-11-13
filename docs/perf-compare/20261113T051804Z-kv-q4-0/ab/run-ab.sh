#!/usr/bin/env bash
# Unpinned same-session A/B for the q4_0 KV change: TinyLlama static, n_prompt 512.
set -u
cd <home>/Repo/juno
S=<scratch>/ab
A=dist/kv-q4-0/baseline-shaded.jar
B=dist/kv-q4-0/candidate-shaded.jar
M=tinyllama-1.1b-chat-v1.0.Q4_K_M
run() { # lane side i jar types...
  local lane=$1 side=$2 i=$3 jar=$4; shift 4
  echo "== $(date -u +%H:%M:%S) $lane $side $i"
  scripts/performance-tests/compare-llama-cpp.sh --cpu --models "$M" --n-prompt 512 "$@" \
    --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
    --juno-jar "$jar" --out "$S/$lane-$side-$i" > "$S/$lane-$side-$i.log" 2>&1
  echo "   exit=$?"
}
for i in 1 2 3; do
  run f16 A $i $A
  run f16 B $i $B
  run q8 A $i $A --cache-type-k q8_0 --cache-type-v q8_0
  run q8 B $i $B --cache-type-k q8_0 --cache-type-v q8_0
  run q4 B $i $B --cache-type-k q4_0 --cache-type-v q4_0
done
echo "== done $(date -u +%H:%M:%S)"
