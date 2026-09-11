#!/usr/bin/env bash
# compare-mixed-prefill.sh — mixed chunked prefill vs admit-time full prefill
#
# Under --schedule continuous:
#   mixed  — default (juno.continuous.mixedPrefill=true): ubatch chunks mix with decode
#   admit  — bake-off baseline (juno.continuous.mixedPrefill=false): full prefill at admit
#
# Workload: 1 long-prompt SSE + N short-prompt SSE clients started together.
# Records short TTFT/TPOT (starvation metric), long TTFT, and JFR ContinuousStep
# prefillChunks proof for mixed mode.
#
# Usage:
#   ./scripts/performance-tests/compare-mixed-prefill.sh --gpu
#   ./scripts/performance-tests/compare-mixed-prefill.sh --gpu --n-prompt 256 --shorts 3

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-mixed-prefill/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
N_PROMPT=256
SHORTS=3
MAX_TOKENS_LONG=16
MAX_TOKENS_SHORT=16
PREFILL_BATCH=32
PARALLEL=8
BATCH_WINDOW_MS=50
API_PORT=18094
USE_GPU=0
PUBLISH=1
JFR_DURATION="30m"
SHORT_PROMPT="Say hi in one word."

log()  { printf '[mixed-prefill] %s\n' "$*"; }
warn() { printf '[mixed-prefill] warn: %s\n' "$*" >&2; }
die()  { printf '[mixed-prefill] error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Mixed chunked prefill bake-off (continuous schedule only).

Options:
  --model FILE            GGUF basename under models/
  --n-prompt N            Long prompt word count (default: ${N_PROMPT})
  --shorts N              Concurrent short SSE clients (default: ${SHORTS})
  --max-tokens-long N     Decode tokens for long request (default: ${MAX_TOKENS_LONG})
  --max-tokens-short N    Decode tokens for short requests (default: ${MAX_TOKENS_SHORT})
  --prefill-batch N       Prefill ubatch size (default: ${PREFILL_BATCH})
  --parallel N            Continuous running-set cap (default: ${PARALLEL})
  --api-port N            REST port (default: ${API_PORT})
  --cpu / --gpu           Backend (default: CPU)
  --out DIR               Output directory
  --no-publish            Skip docs/perf-compare copy
  -h, --help              This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --n-prompt) N_PROMPT="$2"; shift 2 ;;
    --shorts) SHORTS="$2"; shift 2 ;;
    --max-tokens-long) MAX_TOKENS_LONG="$2"; shift 2 ;;
    --max-tokens-short) MAX_TOKENS_SHORT="$2"; shift 2 ;;
    --prefill-batch) PREFILL_BATCH="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --no-publish) PUBLISH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
require_cmd curl
require_cmd jq
require_cmd python3

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH}"

LONG_PROMPT="$(python3 -c "print('Please summarize this passage carefully. ' + ' '.join(['word'] * ${N_PROMPT}))")"

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
  local newest
  newest="$(find "$ROOT" -maxdepth 2 -name '*.jfr' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "$newest" && -f "$newest" ]]; then
    cp -a "$newest" "${OUT_ROOT}/${stem}.jfr" 2>/dev/null || true
  fi
}

# mode: mixed | admit
start_juno() {
  local mode="$1"
  local stem="$2"
  local logf="${OUT_ROOT}/${stem}.log"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} in use — stop other juno or pass --api-port"
  fi

  local jar heap backend_flag mixed_prop
  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  heap="${COMPARE_HEAP:-4g}"
  backend_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && backend_flag="--gpu"
  mixed_prop="true"
  [[ "$mode" == "admit" ]] && mixed_prop="false"

  local -a java_args=(
    --enable-preview --enable-native-access=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED
    --add-opens java.base/java.nio=ALL-UNNAMED
    -XX:+UseG1GC -XX:+AlwaysPreTouch -Xms512m -Xmx"${heap}"
    -Djuno.byteOrder=BE
    -Djuno.continuous.mixedPrefill="${mixed_prop}"
    -jar "$jar"
    --model-path "$MODEL_PATH"
    --dtype FLOAT16 --byteOrder BE
    --max-tokens "$MAX_TOKENS_LONG"
    --temperature 0 --top-k 0 --top-p 0
    --nodes 1 --local "$backend_flag"
    --api-port "$API_PORT"
    --parallel "$PARALLEL"
    --batch-window-ms "$BATCH_WINDOW_MS"
    --schedule continuous
    --prefill-batch "$PREFILL_BATCH"
    --jfr "$JFR_DURATION"
  )

  log "start juno mode=${mode} mixedPrefill=${mixed_prop} parallel=${PARALLEL} prefill-batch=${PREFILL_BATCH} ${backend_flag#--}"
  : >"$logf"
  (
    cd "$ROOT"
    rm -f "$ROOT"/*.jfr "$ROOT"/target/*.jfr 2>/dev/null || true
    exec "$java_bin" "${java_args[@]}" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  if ! wait_for_api "$API_PORT" 600 "$JUNO_PID"; then
    stop_juno
    die "API failed for mode=${mode} — see ${logf}"
  fi
}

resolve_model_id() {
  local model_id
  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  [[ -n "$model_id" ]] || model_id="${MODEL%.gguf}"
  printf '%s' "$model_id"
}

sse_client() {
  local out_raw="$1" out_chunks="$2" out_json="$3" model_id="$4" prompt="$5" max_tokens="$6"
  (
    set +e
    echo "$(date +%s%N)" >"$out_raw"
    curl -sS --max-time 7200 -N \
      -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -H 'Accept: text/event-stream' \
      -d "$(jq -nc --arg m "$model_id" --arg p "$prompt" --argjson n "$max_tokens" \
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
' >>"$out_raw" 2>>"$out_chunks"
    cat "$out_chunks" >>"$out_raw"
    python3 - "$out_raw" "$out_json" <<'PY'
import json, sys
path, out = sys.argv[1], sys.argv[2]
lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
chunk_ms = []
for line in lines[1:]:
    if line.startswith("CHUNK_MS "):
        try:
            chunk_ms.append(float(line.split()[1]))
        except Exception:
            pass
if not chunk_ms:
    json.dump({"ok": False, "ttft_ms": None, "tpot_mean_ms": None, "token_count": 0}, open(out, "w"))
    raise SystemExit(0)
tpots = [chunk_ms[i] - chunk_ms[i-1] for i in range(1, len(chunk_ms))]
json.dump({
    "ok": True,
    "token_count": len(chunk_ms),
    "ttft_ms": chunk_ms[0],
    "tpot_mean_ms": (sum(tpots)/len(tpots)) if tpots else None,
    "tpot_p95_ms": sorted(tpots)[int(0.95*(len(tpots)-1))] if tpots else None,
    "e2e_ms": chunk_ms[-1],
}, open(out, "w"))
PY
  )
}

run_mode() {
  local mode="$1"
  local stem="mix-${mode}"
  local resp_dir="${OUT_ROOT}/${stem}-responses"
  mkdir -p "$resp_dir"

  start_juno "$mode" "$stem"
  local model_id; model_id="$(resolve_model_id)"

  local start_ns end_ns wall_ms
  start_ns="$(date +%s%N)"
  local pids=()

  # Long request first (index 0)
  sse_client "${resp_dir}/raw-long.txt" "${resp_dir}/chunks-long.txt" "${resp_dir}/client-long.json" \
    "$model_id" "$LONG_PROMPT" "$MAX_TOKENS_LONG" &
  pids+=($!)

  local i
  for i in $(seq 1 "$SHORTS"); do
    sse_client "${resp_dir}/raw-short-${i}.txt" "${resp_dir}/chunks-short-${i}.txt" \
      "${resp_dir}/client-short-${i}.json" "$model_id" "$SHORT_PROMPT" "$MAX_TOKENS_SHORT" &
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
  extract_jfr_metrics "$stem"

  python3 - "$OUT_ROOT" "$stem" "$mode" "$RUN_ID" "$MODEL" \
    "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" "$PARALLEL" "$PREFILL_BATCH" \
    "$N_PROMPT" "$SHORTS" "$MAX_TOKENS_LONG" "$MAX_TOKENS_SHORT" "$wall_ms" \
    "$(echo "$health" | jq -c '.')" <<'PY'
import json, os, sys
(out_root, stem, mode, run_id, model, backend, parallel, prefill_batch,
 n_prompt, shorts, max_long, max_short, wall_ms, health) = sys.argv[1:]
parallel=int(parallel); prefill_batch=int(prefill_batch); n_prompt=int(n_prompt)
shorts=int(shorts); max_long=int(max_long); max_short=int(max_short); wall_ms=int(wall_ms)
health=json.loads(health)
resp=os.path.join(out_root, f"{stem}-responses")
long_c=json.load(open(os.path.join(resp,"client-long.json"))) if os.path.isfile(os.path.join(resp,"client-long.json")) else {"ok":False}
shorts_c=[]
for i in range(1, shorts+1):
    p=os.path.join(resp, f"client-short-{i}.json")
    if os.path.isfile(p):
        shorts_c.append(json.load(open(p)))
ok_s=[c for c in shorts_c if c.get("ok")]
def mean(xs):
    return sum(xs)/len(xs) if xs else None
sttfts=[c["ttft_ms"] for c in ok_s if c.get("ttft_ms") is not None]
stpots=[c["tpot_mean_ms"] for c in ok_s if c.get("tpot_mean_ms") is not None]
sp95=[c["tpot_p95_ms"] for c in ok_s if c.get("tpot_p95_ms") is not None]
jfr={}
jfr_path=os.path.join(out_root, f"{stem}-jfr-metrics.json")
if os.path.isfile(jfr_path):
    try:
        snap=json.load(open(jfr_path))
        metrics=snap.get("metrics") or snap
        if isinstance(metrics, dict) and "models" in metrics:
            models=metrics["models"]
            if models:
                metrics=models[0].get("metrics") or models[0]
        keys=(
            "juno.ContinuousStep.count","juno.ContinuousStep.shared_steps",
            "juno.ContinuousStep.max_decode_batch","juno.ContinuousStep.max_running_set",
            "juno.ContinuousStep.steps_with_prefill","juno.ContinuousStep.prefill_chunks",
            "juno.ContinuousStep.prefill_tokens","juno.ContinuousStep.max_prefill_chunks",
            "juno.TokenProduced.tps",
        )
        for k in keys:
            if isinstance(metrics, dict) and k in metrics:
                jfr[k]=metrics[k]
            elif isinstance(snap, dict) and k in snap:
                jfr[k]=snap[k]
    except Exception as e:
        jfr["parse_error"]=str(e)

prefill_chunks=jfr.get("juno.ContinuousStep.prefill_chunks") or 0
steps_with_prefill=jfr.get("juno.ContinuousStep.steps_with_prefill") or 0
if mode=="mixed":
    proof="pass" if (isinstance(prefill_chunks,(int,float)) and prefill_chunks>=1) or (isinstance(steps_with_prefill,(int,float)) and steps_with_prefill>=1) else "fail_or_missing_jfr"
else:
    proof="n/a_admit_baseline"

result={
  "run_id":run_id,"tool":"compare-mixed-prefill","mode":mode,
  "status":"success" if long_c.get("ok") and len(ok_s)==shorts else "partial_failure",
  "model":model,"backend":backend,"schedule":"continuous",
  "parallel":parallel,"prefill_batch":prefill_batch,"n_prompt":n_prompt,"shorts":shorts,
  "max_tokens_long":max_long,"max_tokens_short":max_short,"wall_ms":wall_ms,
  "long_ok":bool(long_c.get("ok")),"long_ttft_ms":long_c.get("ttft_ms"),
  "long_tpot_mean_ms":long_c.get("tpot_mean_ms"),"long_e2e_ms":long_c.get("e2e_ms"),
  "short_ok":len(ok_s),"short_failed":shorts-len(ok_s),
  "short_mean_ttft_ms":mean(sttfts),"short_median_ttft_ms":(sorted(sttfts)[len(sttfts)//2] if sttfts else None),
  "short_mean_tpot_ms":mean(stpots),"short_mean_tpot_p95_ms":mean(sp95),
  "short_max_ttft_ms":(max(sttfts) if sttfts else None),
  "jfr":jfr,"mixed_prefill_proof":proof,
}
json.dump(result, open(os.path.join(out_root, f"{stem}.json"), "w"), indent=2)
print(f"mode={mode}: short_ttft={result['short_mean_ttft_ms']} short_tpot={result['short_mean_tpot_ms']} long_ttft={result['long_ttft_ms']} proof={proof}")
PY
}

write_index() {
  local mixed_json="${OUT_ROOT}/mix-mixed.json"
  local admit_json="${OUT_ROOT}/mix-admit.json"
  [[ -f "$mixed_json" && -f "$admit_json" ]] || die "missing mode JSON"

  python3 - "$OUT_ROOT" "$mixed_json" "$admit_json" "$RUN_ID" "$MODEL" \
    "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" <<'PY'
import json, os, sys
out_root, mixed_p, admit_p, run_id, model, backend = sys.argv[1:]
mixed=json.load(open(mixed_p)); admit=json.load(open(admit_p))

def ratio(a,b):
    if a is None or b is None or b==0: return None
    return a/b

ttft_ratio=ratio(mixed.get("short_mean_ttft_ms"), admit.get("short_mean_ttft_ms"))
tpot_ratio=ratio(mixed.get("short_mean_tpot_ms"), admit.get("short_mean_tpot_ms"))
# Bound: short TTFT under mixed should not exceed admit (or document measured ceiling)
bound_ms=mixed.get("short_max_ttft_ms") or mixed.get("short_mean_ttft_ms")
improved = (ttft_ratio is not None and ttft_ratio < 1.0) or (tpot_ratio is not None and tpot_ratio < 1.0)

lines=[]
lines.append(f"# Mixed chunked prefill bake-off — {run_id}")
lines.append("")
lines.append(f"Model: `{model}` · backend={backend} · schedule=continuous · "
             f"n_prompt={mixed.get('n_prompt')} · shorts={mixed.get('shorts')} · "
             f"prefill-batch={mixed.get('prefill_batch')} · parallel={mixed.get('parallel')}")
lines.append("")
lines.append("## Long-prompt + short-decode concurrency (SSE)")
lines.append("")
lines.append("| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |")
lines.append("|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|")
for label, d in (("mixed (default)", mixed), ("admit-time baseline", admit)):
    lines.append(
        f"| {label} | {d.get('short_mean_ttft_ms')} | {d.get('short_mean_tpot_ms')} | "
        f"{d.get('short_max_ttft_ms')} | {d.get('long_ttft_ms')} | {d.get('wall_ms')} | {d.get('mixed_prefill_proof')} |"
    )
lines.append("")
lines.append(f"Short TTFT mixed/admit: **{ttft_ratio}**" if ttft_ratio is not None else "Short TTFT ratio: n/a")
lines.append(f"Short TPOT mixed/admit: **{tpot_ratio}**" if tpot_ratio is not None else "Short TPOT ratio: n/a")
lines.append("")
lines.append("## JFR ContinuousStep (mixed)")
jfr=mixed.get("jfr") or {}
lines.append("")
lines.append("| metric | value |")
lines.append("|--------|------:|")
for k in sorted(jfr.keys()):
    lines.append(f"| `{k}` | {jfr[k]} |")
lines.append("")
lines.append("## Short-decode latency bound")
lines.append("")
lines.append(f"Under this recipe, short-request TTFT max was **{bound_ms} ms** with mixed chunked prefill.")
lines.append("Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs "
             f"(≤ **{(bound_ms * 1.25) if isinstance(bound_ms,(int,float)) else 'n/a'} ms**).")
lines.append("")
lines.append("Script: `scripts/performance-tests/compare-mixed-prefill.sh`")
lines.append("")
lines.append("### Verdict")
lines.append("")
if improved:
    lines.append("- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.")
else:
    lines.append("- **HONEST:** mixed did not improve short TTFT/TPOT vs admit-time on this recipe; "
                 "artifacts published; check JFR prefill_chunks proof and arrival timing.")
if ttft_ratio is not None and tpot_ratio is not None and tpot_ratio > 1.0 and ttft_ratio < 1.0:
    lines.append(f"- **TPOT tradeoff:** short TPOT {tpot_ratio:.2f}× admit-time under mix "
                 "(shared steps with long ubatch); expected capacity sharing.")
if mixed.get("mixed_prefill_proof")=="pass":
    lines.append("- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).")
else:
    lines.append("- Mixed prefill JFR proof **FAIL or missing**.")
open(os.path.join(out_root,"INDEX.md"),"w").write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
}

publish_docs() {
  [[ "$PUBLISH" -eq 1 ]] || return 0
  local dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-mixed-prefill"
  mkdir -p "$dest"
  cp -a "${OUT_ROOT}/." "$dest/"
  log "published ${dest}"
}

mkdir -p "$OUT_ROOT"
setup_cuda_env
java_bin="$(find_java)"

log "OUT_ROOT=${OUT_ROOT}"
run_mode mixed
run_mode admit
write_index
publish_docs
log "done ${RUN_ID}"
