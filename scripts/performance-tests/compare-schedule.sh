#!/usr/bin/env bash
# compare-schedule.sh — continuous vs static schedule bake-off (local / in-process)
#
# Workloads:
#   tps    — multi-session blocking chat (same recipe as compare-parallel.sh)
#   sse    — concurrent stream:true clients; wall TTFT/TPOT + JFR ContinuousStep
#   prefix — sequential shared-system-prompt; prefixLookups/Hits/HitRate + TTFT
#   all    — run tps, sse, prefix (default)
#
# Usage:
#   ./scripts/performance-tests/compare-schedule.sh --gpu
#   ./scripts/performance-tests/compare-schedule.sh --gpu --mode tps --sessions 8

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-schedule/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
SESSIONS=8
MAX_TOKENS=64
BATCH_WINDOW_MS=50
PARALLEL=8
API_PORT=18092
USE_GPU=0
MODE="all"
PROMPT_TEXT="could you please write me a short poem about love and war"
SYSTEM_PROMPT="You are a helpful assistant that answers briefly and accurately."
PUBLISH=1
JFR_DURATION="30m"

log()  { printf '[schedule] %s\n' "$*"; }
warn() { printf '[schedule] warn: %s\n' "$*" >&2; }
die()  { printf '[schedule] error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Continuous vs static bake-off for local serving.

Options:
  --mode tps|sse|prefix|all   Workload (default: all)
  --model FILE                GGUF basename under models/
  --sessions N                Concurrent / sequential clients (default: ${SESSIONS})
  --max-tokens N              Tokens per session (default: ${MAX_TOKENS})
  --parallel N                Running-set / static batch cap (default: ${PARALLEL})
  --batch-window-ms M         Batch window (default: ${BATCH_WINDOW_MS})
  --api-port N                REST port (default: ${API_PORT})
  --cpu / --gpu               Backend (default: CPU)
  --prompt TEXT               User prompt (tps/sse)
  --system-prompt TEXT        Shared system prompt (prefix mode)
  --out DIR                   Output directory
  --no-publish                Skip docs/perf-compare copy
  -h, --help                  This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --batch-window-ms) BATCH_WINDOW_MS="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --prompt) PROMPT_TEXT="$2"; shift 2 ;;
    --system-prompt) SYSTEM_PROMPT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --no-publish) PUBLISH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$MODE" in
  tps|sse|prefix|all) ;;
  *) die "--mode must be tps|sse|prefix|all" ;;
esac

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
require_cmd curl
require_cmd jq
require_cmd awk
require_cmd python3

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH}"

find_juno_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/juno-player/target/"juno-player-*-shaded.jar )
  shopt -u nullglob
  [[ ${#jars[@]} -gt 0 ]] && { printf '%s' "${jars[0]}"; return; }
  [[ -f "$ROOT/juno-player/target/juno-player.jar" ]] || die "build jar: mvn package -DskipTests -pl juno-player -am"
  printf '%s' "$ROOT/juno-player/target/juno-player.jar"
}

find_java() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    printf '%s' "${JAVA_HOME}/bin/java"; return
  fi
  command -v java || die "java not found"
}

find_metrics_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/metrics/target/"metrics-*.jar )
  shopt -u nullglob
  [[ ${#jars[@]} -gt 0 ]] && { printf '%s' "${jars[-1]}"; return; }
  printf ''
}

setup_cuda_env() {
  [[ "$USE_GPU" -eq 1 ]] || return 0
  if [[ -e /usr/lib/x86_64-linux-gnu/libcudart.so.12 ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export JUNO_USE_GPU=true
}

wait_for_api() {
  local port="$1" timeout="${2:-300}" pid="${3:-}"
  local start now; start="$(date +%s)"
  while true; do
    curl -sf "http://127.0.0.1:${port}/v1/cluster/health" >/dev/null 2>&1 && return 0
    [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && return 1
    now="$(date +%s)"
    (( now - start >= timeout )) && return 1
    sleep 2
  done
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  JUNO_PID=""
}

extract_jfr_metrics() {
  local stem="$1"
  local metrics_jar
  metrics_jar="$(find_metrics_jar)"
  [[ -n "$metrics_jar" ]] || { warn "metrics jar missing — skip JFR extract"; return 0; }
  mkdir -p "${ROOT}/target/metrics"
  (
    cd "$ROOT"
    "$java_bin" -cp "$metrics_jar" cab.ml.juno.metrics.MetricsMain >/dev/null 2>&1 || true
  )
  if [[ -f "${ROOT}/target/metrics/metrics.json" ]]; then
    cp -a "${ROOT}/target/metrics/metrics.json" "${OUT_ROOT}/${stem}-jfr-metrics.json"
  fi
  # Capture newest *.jfr next to cwd / target
  local newest
  newest="$(find "$ROOT" -maxdepth 2 -name '*.jfr' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "$newest" && -f "$newest" ]]; then
    cp -a "$newest" "${OUT_ROOT}/${stem}.jfr" 2>/dev/null || true
  fi
}

start_juno() {
  local schedule="$1"
  local stem="$2"
  local logf="${OUT_ROOT}/${stem}.log"
  local with_jfr="${3:-0}"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} in use — stop other juno or pass --api-port"
  fi

  local jar heap backend_flag
  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  heap="${COMPARE_HEAP:-4g}"
  backend_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && backend_flag="--gpu"

  local -a java_args=(
    --enable-preview --enable-native-access=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED
    --add-opens java.base/java.nio=ALL-UNNAMED
    -XX:+UseG1GC -XX:+AlwaysPreTouch -Xms512m -Xmx"${heap}"
    -Djuno.byteOrder=BE
    -jar "$jar"
    --model-path "$MODEL_PATH"
    --dtype FLOAT16 --byteOrder BE
    --max-tokens "$MAX_TOKENS"
    --temperature 0 --top-k 0 --top-p 0
    --nodes 1 --local "$backend_flag"
    --api-port "$API_PORT"
    --parallel "$PARALLEL"
    --batch-window-ms "$BATCH_WINDOW_MS"
    --schedule "$schedule"
  )
  if [[ "$with_jfr" == "1" ]]; then
    java_args+=(--jfr "$JFR_DURATION")
  fi

  log "start juno schedule=${schedule} parallel=${PARALLEL} sessions=${SESSIONS} ${backend_flag#--} jfr=${with_jfr}"
  : >"$logf"
  (
    cd "$ROOT"
    # Remove stale JFRs so MetricsMain maps this run
    rm -f "$ROOT"/*.jfr "$ROOT"/target/*.jfr 2>/dev/null || true
    exec "$java_bin" "${java_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  JUNO_LOG="$logf"

  if ! wait_for_api "$API_PORT" 600 "$JUNO_PID"; then
    stop_juno
    die "API failed for schedule=${schedule} — see ${logf}"
  fi

  local health
  health="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health")"
  local got
  got="$(echo "$health" | jq -r '.schedule // empty')"
  [[ "$got" == "$schedule" ]] || warn "health.schedule=${got} expected ${schedule}"
}

resolve_model_id() {
  local model_id
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  [[ -n "$model_id" ]] || model_id="${MODEL%.gguf}"
  printf '%s' "$model_id"
}

run_tps() {
  local schedule="$1"
  local stem="tps-${schedule}"
  local resp_dir="${OUT_ROOT}/${stem}-responses"
  mkdir -p "$resp_dir"

  start_juno "$schedule" "$stem" 0
  local model_id; model_id="$(resolve_model_id)"

  local start_ns end_ns wall_ms
  start_ns="$(date +%s%N)"
  local i pids=() rc=0
  for i in $(seq 1 "$SESSIONS"); do
    (
      set +e
      http="$(curl -sS --max-time 7200 -o "${resp_dir}/resp-${i}.json" -w '%{http_code}' \
        -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$model_id" --arg p "$PROMPT_TEXT" --argjson n "$MAX_TOKENS" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')")"
      echo "$http" >"${resp_dir}/resp-${i}.http"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local total_completion=0 total_prompt=0 total_latency=0 ok=0 fail=0
  for i in $(seq 1 "$SESSIONS"); do
    local resp="${resp_dir}/resp-${i}.json"
    local http; http="$(cat "${resp_dir}/resp-${i}.http" 2>/dev/null || echo 000)"
    if [[ "$http" == "200" && -s "$resp" ]]; then
      ok=$((ok + 1))
      total_completion=$((total_completion + $(jq -r '.usage.completion_tokens // 0' "$resp")))
      total_prompt=$((total_prompt + $(jq -r '.usage.prompt_tokens // 0' "$resp")))
      total_latency=$((total_latency + $(jq -r '.x_juno_latency_ms // 0' "$resp")))
    else
      fail=$((fail + 1))
    fi
  done

  local health
  health="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" || echo '{}')"
  stop_juno
  sleep 2

  local agg_tps="null" mean_latency="null"
  if (( wall_ms > 0 && total_completion > 0 )); then
    agg_tps="$(awk -v t="$total_completion" -v ms="$wall_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
  fi
  if (( ok > 0 )); then
    mean_latency="$(awk -v l="$total_latency" -v n="$ok" 'BEGIN { printf "%.2f", l / n }')"
  fi

  jq -n \
    --arg run_id "$RUN_ID" \
    --arg model "$MODEL" \
    --arg backend "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" \
    --arg schedule "$schedule" \
    --argjson parallel "$PARALLEL" \
    --argjson sessions "$SESSIONS" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson wall_ms "$wall_ms" \
    --argjson ok "$ok" \
    --argjson fail "$fail" \
    --argjson total_completion_tokens "$total_completion" \
    --argjson total_prompt_tokens "$total_prompt" \
    --argjson aggregate_tps "${agg_tps:-null}" \
    --argjson mean_latency_ms "${mean_latency:-null}" \
    --argjson prefix_lookups "$(echo "$health" | jq -r '.prefixLookups // 0')" \
    --argjson prefix_hits "$(echo "$health" | jq -r '.prefixHits // 0')" \
    --argjson prefix_hit_rate "$(echo "$health" | jq -r '.prefixHitRate // 0')" \
    --arg prompt "$PROMPT_TEXT" \
  '{
    run_id:$run_id, tool:"compare-schedule", workload:"tps", status:(if $fail==0 then "success" else "partial_failure" end),
    model:$model, backend:$backend, schedule:$schedule, parallel:$parallel, sessions:$sessions, max_tokens:$max_tokens,
    wall_ms:$wall_ms, sessions_ok:$ok, sessions_failed:$fail,
    total_completion_tokens:$total_completion_tokens, total_prompt_tokens:$total_prompt_tokens,
    aggregate_token_gen_tps:$aggregate_tps, mean_per_session_latency_ms:$mean_latency_ms,
    prefix_lookups:$prefix_lookups, prefix_hits:$prefix_hits, prefix_hit_rate:$prefix_hit_rate, prompt:$prompt
  }' >"${OUT_ROOT}/${stem}.json"

  log "tps schedule=${schedule}: agg_tps=${agg_tps:-?} ok=${ok}/${SESSIONS} wall_ms=${wall_ms}"
}

# Parse SSE: write per-client JSON with ttft_ms, tpot_mean_ms, token_count
parse_sse_client() {
  python3 - "$1" "$2" <<'PY'
import json, sys, time
path, out = sys.argv[1], sys.argv[2]
text = open(path, "r", encoding="utf-8", errors="replace").read()
# file format: first line WALL_START_NS, then raw SSE body
lines = text.splitlines()
if not lines:
    json.dump({"ok": False, "error": "empty"}, open(out, "w")); sys.exit(0)
wall_start_ns = int(lines[0])
body = "\n".join(lines[1:])
token_times_ms = []
for line in body.splitlines():
    if not line.startswith("data:"):
        continue
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    choices = obj.get("choices") or []
    if not choices:
        continue
    delta = (choices[0].get("delta") or {})
    content = delta.get("content")
    if content:
        # Approximate arrival: use sequential index * we don't have per-chunk wall;
        # caller embeds CHUNK_WALL_MS lines — see below.
        pass

# Prefer instrumented CHUNK lines from curl wrapper
chunk_ms = []
for line in lines[1:]:
    if line.startswith("CHUNK_MS "):
        try:
            chunk_ms.append(float(line.split()[1]))
        except Exception:
            pass

if not chunk_ms:
    # Fallback: count content deltas without timing precision
    count = 0
    for line in body.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
            if raw and raw != "[DONE]":
                try:
                    obj = json.loads(raw)
                    c = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
                    if c:
                        count += 1
                except Exception:
                    pass
    json.dump({"ok": count > 0, "token_count": count, "ttft_ms": None, "tpot_mean_ms": None,
               "note": "no CHUNK_MS instrumentation"}, open(out, "w"))
    sys.exit(0)

ttft = chunk_ms[0]
tpots = [chunk_ms[i] - chunk_ms[i-1] for i in range(1, len(chunk_ms))]
tpot_mean = sum(tpots)/len(tpots) if tpots else None
json.dump({
    "ok": True,
    "token_count": len(chunk_ms),
    "ttft_ms": ttft,
    "tpot_mean_ms": tpot_mean,
    "tpot_p95_ms": sorted(tpots)[int(0.95*(len(tpots)-1))] if tpots else None,
}, open(out, "w"))
PY
}

run_sse() {
  local schedule="$1"
  local stem="sse-${schedule}"
  local resp_dir="${OUT_ROOT}/${stem}-responses"
  mkdir -p "$resp_dir"

  local with_jfr=0
  [[ "$schedule" == "continuous" ]] && with_jfr=1
  start_juno "$schedule" "$stem" "$with_jfr"
  local model_id; model_id="$(resolve_model_id)"

  local start_ns end_ns wall_ms
  start_ns="$(date +%s%N)"
  local i pids=()
  for i in $(seq 1 "$SESSIONS"); do
    (
      set +e
      raw="${resp_dir}/raw-${i}.txt"
      echo "$(date +%s%N)" >"$raw"
      # Stream SSE; stamp wall ms after each data line with content via python tee
      curl -sS --max-time 7200 -N \
        -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -H 'Accept: text/event-stream' \
        -d "$(jq -nc --arg m "$model_id" --arg p "$PROMPT_TEXT" --argjson n "$MAX_TOKENS" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:true}')" \
        | python3 -c '
import sys, time, json
t0 = time.time() * 1000.0
for line in sys.stdin:
    sys.stdout.write(line)
    sys.stdout.flush()
    if line.startswith("data:"):
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
            c = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if c:
                print(f"CHUNK_MS {time.time()*1000.0 - t0:.3f}", file=sys.stderr)
        except Exception:
            pass
' >>"$raw" 2>>"${resp_dir}/chunks-${i}.txt"
      # Merge CHUNK_MS into raw for parser
      cat "${resp_dir}/chunks-${i}.txt" >>"$raw"
      parse_sse_client "$raw" "${resp_dir}/client-${i}.json"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || true
  done
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local health
  health="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" || echo '{}')"
  stop_juno
  sleep 2
  if [[ "$with_jfr" == "1" ]]; then
    extract_jfr_metrics "$stem"
  fi

  # Aggregate client metrics
  python3 - "$OUT_ROOT" "$stem" "$SESSIONS" "$schedule" "$RUN_ID" "$MODEL" \
    "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" "$PARALLEL" "$MAX_TOKENS" "$wall_ms" \
    "$(echo "$health" | jq -c '.')" <<'PY'
import json, glob, os, sys
out_root, stem, sessions, schedule, run_id, model, backend, parallel, max_tokens, wall_ms, health = sys.argv[1:]
sessions = int(sessions); parallel = int(parallel); max_tokens = int(max_tokens); wall_ms = int(wall_ms)
health = json.loads(health)
clients = []
for i in range(1, sessions+1):
    p = os.path.join(out_root, f"{stem}-responses", f"client-{i}.json")
    if os.path.isfile(p):
        clients.append(json.load(open(p)))
ok = [c for c in clients if c.get("ok")]
ttfts = [c["ttft_ms"] for c in ok if c.get("ttft_ms") is not None]
tpots = [c["tpot_mean_ms"] for c in ok if c.get("tpot_mean_ms") is not None]
def mean(xs):
    return sum(xs)/len(xs) if xs else None
jfr = {}
jfr_path = os.path.join(out_root, f"{stem}-jfr-metrics.json")
if os.path.isfile(jfr_path):
    try:
        snap = json.load(open(jfr_path))
        # metrics.json shape varies; try nested metrics map
        metrics = snap.get("metrics") or snap
        if isinstance(metrics, dict) and "models" in metrics:
            models = metrics["models"]
            if models:
                metrics = models[0].get("metrics") or models[0]
        for k in ("juno.ContinuousStep.count","juno.ContinuousStep.shared_steps",
                  "juno.ContinuousStep.max_decode_batch","juno.ContinuousStep.max_running_set",
                  "juno.TokenProduced.tps"):
            if isinstance(metrics, dict) and k in metrics:
                jfr[k] = metrics[k]
            elif isinstance(snap, dict) and k in snap:
                jfr[k] = snap[k]
    except Exception as e:
        jfr["parse_error"] = str(e)

shared = jfr.get("juno.ContinuousStep.shared_steps")
max_batch = jfr.get("juno.ContinuousStep.max_decode_batch")
if schedule != "continuous":
    proof = "n/a_static_isolated"
elif (isinstance(max_batch, (int, float)) and max_batch >= 2) or (isinstance(shared, (int, float)) and shared >= 1):
    proof = "pass"
else:
    proof = "fail_or_missing_jfr"

result = {
  "run_id": run_id, "tool": "compare-schedule", "workload": "sse",
  "status": "success" if len(ok)==sessions else "partial_failure",
  "model": model, "backend": backend, "schedule": schedule,
  "parallel": parallel, "sessions": sessions, "max_tokens": max_tokens,
  "wall_ms": wall_ms, "sessions_ok": len(ok), "sessions_failed": sessions-len(ok),
  "mean_ttft_ms": mean(ttfts), "mean_tpot_ms": mean(tpots),
  "median_ttft_ms": sorted(ttfts)[len(ttfts)//2] if ttfts else None,
  "prefix_lookups": health.get("prefixLookups", 0),
  "prefix_hits": health.get("prefixHits", 0),
  "prefix_hit_rate": health.get("prefixHitRate", 0),
  "jfr": jfr,
  "shared_step_proof": proof,
}
json.dump(result, open(os.path.join(out_root, f"{stem}.json"), "w"), indent=2)
print(f"sse schedule={schedule}: mean_ttft={result['mean_ttft_ms']} mean_tpot={result['mean_tpot_ms']} proof={proof}")
PY
}

run_prefix() {
  local schedule="$1"
  local stem="prefix-${schedule}"
  local resp_dir="${OUT_ROOT}/${stem}-responses"
  mkdir -p "$resp_dir"

  start_juno "$schedule" "$stem" 0
  local model_id; model_id="$(resolve_model_id)"
  local session_id="bakeoff-prefix-${schedule}-${RUN_ID}"

  # Multi-turn same x_juno_session_id + fixed system prompt: prefix trie + KV reuse
  # (stateless shared-system across different request ids is lookups-only until KV pin).
  local i ok=0 fail=0
  local first_ttft="" later_ttfts=()
  local history_json
  history_json="$(jq -nc --arg s "$SYSTEM_PROMPT" '[{role:"system",content:$s}]')"

  for i in $(seq 1 "$SESSIONS"); do
    local user="Turn ${i}: name one color."
    history_json="$(jq -nc --argjson h "$history_json" --arg u "$user" \
      '$h + [{role:"user",content:$u}]')"
    local t0 t1 latency http
    t0="$(date +%s%N)"
    http="$(curl -sS --max-time 7200 -o "${resp_dir}/resp-${i}.json" -w '%{http_code}' \
      -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg m "$model_id" --argjson msgs "$history_json" --arg sid "$session_id" --argjson n "$MAX_TOKENS" \
        '{model:$m,messages:$msgs,max_tokens:$n,temperature:0,stream:false,x_juno_session_id:$sid}')")"
    t1="$(date +%s%N)"
    latency=$(( (t1 - t0) / 1000000 ))
    echo "$http" >"${resp_dir}/resp-${i}.http"
    echo "$latency" >"${resp_dir}/resp-${i}.ttft_wall_ms"
    if [[ "$http" == "200" ]]; then
      ok=$((ok + 1))
      local api_lat assistant
      api_lat="$(jq -r '.x_juno_latency_ms // empty' "${resp_dir}/resp-${i}.json" 2>/dev/null || true)"
      [[ -n "$api_lat" ]] && latency="$api_lat"
      assistant="$(jq -r '.choices[0].message.content // empty' "${resp_dir}/resp-${i}.json" 2>/dev/null || true)"
      if [[ -n "$assistant" ]]; then
        history_json="$(jq -nc --argjson h "$history_json" --arg a "$assistant" \
          '$h + [{role:"assistant",content:$a}]')"
      fi
      if [[ -z "$first_ttft" ]]; then
        first_ttft="$latency"
      else
        later_ttfts+=("$latency")
      fi
    else
      fail=$((fail + 1))
    fi
  done

  local health
  health="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" || echo '{}')"
  stop_juno
  sleep 2

  local mean_later="null"
  if (( ${#later_ttfts[@]} > 0 )); then
    mean_later="$(python3 -c "xs=[float(x) for x in '''${later_ttfts[*]}'''.split()]; print(f'{sum(xs)/len(xs):.2f}')")"
  fi

  jq -n \
    --arg run_id "$RUN_ID" \
    --arg model "$MODEL" \
    --arg backend "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" \
    --arg schedule "$schedule" \
    --arg system "$SYSTEM_PROMPT" \
    --arg session_id "$session_id" \
    --argjson sessions "$SESSIONS" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson ok "$ok" \
    --argjson fail "$fail" \
    --argjson first_latency_ms "${first_ttft:-null}" \
    --argjson mean_later_latency_ms "${mean_later:-null}" \
    --argjson prefix_lookups "$(echo "$health" | jq -r '.prefixLookups // 0')" \
    --argjson prefix_hits "$(echo "$health" | jq -r '.prefixHits // 0')" \
    --argjson prefix_hit_rate "$(echo "$health" | jq -r '.prefixHitRate // 0')" \
  '{
    run_id:$run_id, tool:"compare-schedule", workload:"prefix",
    status:(if $fail==0 then "success" else "partial_failure" end),
    model:$model, backend:$backend, schedule:$schedule, sessions:$sessions, max_tokens:$max_tokens,
    sessions_ok:$ok, sessions_failed:$fail, system_prompt:$system, session_id:$session_id,
    first_request_latency_ms:$first_latency_ms, mean_later_request_latency_ms:$mean_later_latency_ms,
    prefix_lookups:$prefix_lookups, prefix_hits:$prefix_hits, prefix_hit_rate:$prefix_hit_rate,
    note:"Multi-turn x_juno_session_id + shared system. Latency is full-response wall (non-stream)."
  }' >"${OUT_ROOT}/${stem}.json"

  log "prefix schedule=${schedule}: lookups=$(echo "$health" | jq -r '.prefixLookups') hits=$(echo "$health" | jq -r '.prefixHits') rate=$(echo "$health" | jq -r '.prefixHitRate')"
}

mkdir -p "$OUT_ROOT"
setup_cuda_env
java_bin="$(find_java)"

run_modes() {
  local m
  for m in "$@"; do
    case "$m" in
      tps)
        run_tps static
        run_tps continuous
        ;;
      sse)
        run_sse continuous
        run_sse static
        ;;
      prefix)
        run_prefix continuous
        run_prefix static
        ;;
    esac
  done
}

if [[ "$MODE" == "all" ]]; then
  run_modes tps sse prefix
else
  run_modes "$MODE"
fi

# INDEX.md
index_md="${OUT_ROOT}/INDEX.md"
{
  echo "# Continuous vs static schedule — ${RUN_ID}"
  echo
  echo "Model: \`${MODEL}\` · sessions=${SESSIONS} · max_tokens=${MAX_TOKENS} · parallel=${PARALLEL} · backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)"
  echo
  echo "## Multi-session TPS (blocking)"
  echo
  echo "| schedule | aggregate tg t/s | mean latency ms | wall ms | ok/fail |"
  echo "|----------|----------------:|----------------:|--------:|--------:|"
  for f in "${OUT_ROOT}"/tps-static.json "${OUT_ROOT}"/tps-continuous.json; do
    [[ -f "$f" ]] || continue
    jq -r '"| \(.schedule) | \(.aggregate_token_gen_tps // "-") | \(.mean_per_session_latency_ms // "-") | \(.wall_ms) | \(.sessions_ok)/\(.sessions) |"' "$f"
  done
  if [[ -f "${OUT_ROOT}/tps-static.json" && -f "${OUT_ROOT}/tps-continuous.json" ]]; then
    echo
    echo -n "Continuous / static aggregate TPS: "
    jq -n --slurpfile s "${OUT_ROOT}/tps-static.json" --slurpfile c "${OUT_ROOT}/tps-continuous.json" \
      'if ($s[0].aggregate_token_gen_tps // 0) > 0 then (($c[0].aggregate_token_gen_tps) / ($s[0].aggregate_token_gen_tps)) else null end'
  fi
  echo
  echo "## Concurrent SSE TTFT / TPOT"
  echo
  echo "| schedule | mean TTFT ms | mean TPOT ms | shared-step proof | ContinuousStep max_decode_batch |"
  echo "|----------|-------------:|-------------:|-------------------|--------------------------------:|"
  for f in "${OUT_ROOT}"/sse-continuous.json "${OUT_ROOT}"/sse-static.json; do
    [[ -f "$f" ]] || continue
    jq -r '"| \(.schedule) | \(.mean_ttft_ms // "-") | \(.mean_tpot_ms // "-") | \(.shared_step_proof) | \(.jfr["juno.ContinuousStep.max_decode_batch"] // "-") |"' "$f"
  done
  echo
  echo "## Prefix cache (shared system prompt, multi-turn session)"
  echo
  echo "| schedule | lookups | hits | hit rate | first latency ms | mean later latency ms |"
  echo "|----------|--------:|-----:|---------:|-----------------:|----------------------:|"
  for f in "${OUT_ROOT}"/prefix-continuous.json "${OUT_ROOT}"/prefix-static.json; do
    [[ -f "$f" ]] || continue
    jq -r '"| \(.schedule) | \(.prefix_lookups) | \(.prefix_hits) | \(.prefix_hit_rate) | \(.first_request_latency_ms // "-") | \(.mean_later_request_latency_ms // "-") |"' "$f"
  done
  echo
  echo "Script: \`scripts/performance-tests/compare-schedule.sh\`"
  echo
  echo "Notes:"
  echo "- TPS: synchronized 8-way blocking arrival (same recipe as compare-parallel)."
  echo "- SSE continuous: JFR \`juno.ContinuousStep\` with decodeBatchSize≥2 proves shared steps."
  echo "- Prefix: \`x_juno_session_id\` multi-turn; counters from \`GET /v1/cluster/health\`."
} >"$index_md"

if [[ "$PUBLISH" -eq 1 ]]; then
  dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-continuous"
  mkdir -p "$dest"
  cp -a "$OUT_ROOT"/* "$dest/"
  log "published → docs/perf-compare/${RUN_ID}-continuous/"
fi

log "done: ${OUT_ROOT}"
cat "$index_md"
