#!/usr/bin/env bash
# measure-reserve.sh <label> <jar> <model-path> <nodes> <heap> <decode-tokens> <out-dir>
# One engine: per-process peak VRAM during a short-prompt decode and then during a
# 512-token prefill (max_tokens 1), plus the load-time log lines that matter.
set -uo pipefail
label="$1" jar="$2" model="$3" nodes="$4" heap="$5" ntok="$6" out="$7"
ROOT=<repo>
PORT=18093
mkdir -p "$out"
stem="${out}/${label}"
rm -f "${stem}".*
source "${ROOT}/scripts/performance-tests/perf-lib.sh"
perf_engine_stdin_open || exit 1
(
  cd "$ROOT"
  perf_engine_exec java --enable-preview --enable-native-access=ALL-UNNAMED \
    --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
    --add-modules jdk.incubator.vector -XX:+UseG1GC -Xms"$heap" -Xmx"$heap" -Djuno.byteOrder=BE \
    -jar "$jar" --model-path "$model" --dtype FLOAT16 --byteOrder BE \
    --max-tokens 64 --temperature 0 --top-k 0 --top-p 0 --nodes "$nodes" --local --gpu \
    --api-port "$PORT" --schedule static --verbose
) >"${stem}.server.log" 2>&1 &
pid=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:${PORT}/v1/cluster/health" >/dev/null 2>&1; do
  kill -0 "$pid" 2>/dev/null || { echo "server exited early"; perf_engine_stdin_release; exit 1; }
  (( $(date +%s) - t0 < 1800 )) || { echo "not healthy"; kill "$pid"; exit 1; }
  sleep 2
done
jpid=$(pgrep -f -- "--api-port ${PORT}" | head -1)
( while true; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p="$jpid" '$1==p{print $2}')
    echo "$(date +%s.%N) ${m:-0}"
    sleep 0.03
  done ) >"${stem}.vram.csv" &
smi=$!
mark() { echo "$1 $(date +%s.%N)" >>"${stem}.marks"; }
chat() {
  jq -n --arg u "$1" --argjson n "$2" '{messages:[{role:"user",content:$u}],temperature:0,max_tokens:$n}' |
    curl -sS --max-time 3600 -o "$3" -w '%{http_code} %{time_total}\n' -H 'Content-Type: application/json' \
      "http://127.0.0.1:${PORT}/v1/chat/completions" -d @-
}
long_prompt() {  # 18 notes: about 515 prompt tokens with the chat template
  python3 - <<'PY'
places = "harbour orchard library bridge market lighthouse mill chapel".split()
colours = "red green blue white yellow grey black orange".split()
print("Read the notes below, then answer the question at the end.")
for i in range(1, 19):
    print(f"Note {i}: the {places[i % 8]} in district {i * 7 % 31} was painted {colours[i * 3 % 8]} in the year {1800 + i * 13}.")
print("Question: according to note 1, what colour was the orchard painted? Answer in one sentence.")
PY
}
sleep 1; mark idle_end
mark decode_start
echo "decode $(chat 'Write a short story about a cat.' "$ntok" "${stem}.decode.json")" >>"${stem}.timing"
mark decode_end
sleep 1
mark prefill_start
echo "prefill $(chat "$(long_prompt)" 1 "${stem}.prefill.json")" >>"${stem}.timing"
mark prefill_end
sleep 1
kill "$smi" 2>/dev/null
kill -TERM "$pid" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -KILL "$pid" 2>/dev/null
perf_engine_stdin_release
python3 - "$stem" <<'PY'
import sys, json
stem = sys.argv[1]
rows = [(float(a), int(b)) for a, b in (l.split() for l in open(stem + ".vram.csv")) if b.isdigit()]
marks = {k: float(v) for k, v in (l.split() for l in open(stem + ".marks"))}
def peak(a, b):
    v = [m for t, m in rows if marks[a] <= t <= marks[b]]
    return max(v) if v else None
idle = [m for t, m in rows if t <= marks["idle_end"]]
r = {"after_load_mib": max(idle) if idle else None, "decode_peak_mib": peak("decode_start", "decode_end"),
     "prefill_peak_mib": peak("prefill_start", "prefill_end"), "samples": len(rows)}
if r["decode_peak_mib"] and r["prefill_peak_mib"]:
    r["prefill_over_decode"] = round(r["prefill_peak_mib"] / r["decode_peak_mib"], 4)
print(json.dumps(r))
json.dump(r, open(stem + ".vram.json", "w"))
PY
grep -h -E "stopping GPU upload|Prefill chunk size resolved|out of device memory|run on the device|gpu-layers|GPU upload|fell back|resolved to" "${stem}.server.log" | head -30
cat "${stem}.timing"
jq -c '.usage' "${stem}.decode.json" "${stem}.prefill.json"
