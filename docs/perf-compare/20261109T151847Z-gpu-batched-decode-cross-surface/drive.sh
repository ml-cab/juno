#!/usr/bin/env bash
# Batched GPU decode, cross-surface runs on the final jar, strictly one GPU workload at a time. Every step runs even if
# an earlier one fails; each exit code is recorded in progress.txt.
set -uo pipefail
R=<repo>; X=$(cd "$(dirname "$0")" && pwd); B=$X/lora-baseline-tree; cd "$R"
P=$X/progress.txt; : > "$P"
MODEL=$R/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
PROBES=gpu.consistency.tinyllama,gpu.parallel.tinyllama,gpu.schedule.tinyllama,gpu.mixed-prefill.tinyllama,gpu.live.tinyllama,gpu.live.qwen2.5-3b,gpu.live.phi-3.5-mini,gpu.live.mistral-7b,gpu.live.qwen3-1.7b,gpu.residency.tinyllama,gpu.lora.tinyllama,gpu.vision.moondream2,gpu.unit-tests,consistency.tinyllama,parallel.tinyllama
step() { local n=$1; shift; echo "$(date -u +%H:%M:%S) start $n" >> "$P"; "$@"; local rc=$?; echo "$(date -u +%H:%M:%S) end   $n exit=$rc" >> "$P"; sleep 8; }
loraside() { # side i tree
  ( cd "$3" && scripts/performance-tests/compare-lora.sh --gpu --reps 1 --skip-build --no-publish --out "$X/lora/$1-$2" ) > "$X/lora-$1-$2.log" 2>&1
}
mkdir -p "$X/lora"
for i in 1 2 3; do
  step lora-baseline-$i loraside baseline $i "$B"
  step lora-candidate-$i loraside candidate $i "$R"
done
playadapter=$(find "$X/lora/candidate-1" -name '*.lora' | head -1)
echo "adapter: $playadapter" >> "$P"
step lora-play-batch python3 -I "$X/lora-play-batch.py" "$R" "$R/juno-player/target/juno-player-0.1.2-shaded.jar" "$playadapter" "$X/lora-play" 18095
schedrun() { scripts/performance-tests/compare-schedule.sh --gpu --no-publish --out "$X/schedule" > "$X/schedule.log" 2>&1; }
step schedule schedrun
gpuit() { mvn -o verify -pl juno-master -am -Pgpu -Dit.model.path="$MODEL" -Dtest=NoSuchUnitTest -Dsurefire.failIfNoSpecifiedTests=false > "$X/it-gpu.log" 2>&1; }
step it-gpu gpuit
mkdir -p "$X/it-gpu-reports"; cp juno-master/target/failsafe-reports/*.txt "$X/it-gpu-reports/" 2>/dev/null
stubit() { mvn -o verify -pl juno-master -am -Dtest=NoSuchUnitTest -Dsurefire.failIfNoSpecifiedTests=false > "$X/it-stub.log" 2>&1; }
step it-stub stubit
mkdir -p "$X/it-stub-reports"; cp juno-master/target/failsafe-reports/*.txt "$X/it-stub-reports/" 2>/dev/null
kitrun() { ./juno-kit.sh --skip-build --yes --probes "$PROBES" --models-dir models --out "$X/kit" > "$X/kit.log" 2>&1; }
step kit kitrun
echo "$(date -u +%H:%M:%S) ALL DONE" >> "$P"
