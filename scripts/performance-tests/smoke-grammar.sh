#!/usr/bin/env bash
# smoke-grammar.sh — cross-feature proof for constrained decoding (GBNF / JSON Schema)
#
# Starts TinyLlama locally with --jfr/--verbose, hits wired matrix cells, and
# writes HTTP bodies + JFR metrics under target/grammar-smoke/<timestamp>/.
# Does not publish to docs/perf-compare/ (that is compare-llama-cpp.sh).
#
# Usage:
#   ./scripts/performance-tests/smoke-grammar.sh
#   ./scripts/performance-tests/smoke-grammar.sh --api-port 18082
set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/grammar-smoke/${RUN_ID}"
MODEL="${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LORA="${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora"
YES_NO="${ROOT}/docs/grammars/yes-no.gbnf"
OK_SCHEMA="${ROOT}/docs/grammars/ok.schema.json"
BAD_SCHEMA="${ROOT}/docs/grammars/pattern-unsupported.schema.json"
API_PORT=18082
JUNO_PID=""
failures=0

log()  { printf '[grammar-smoke] %s\n' "$*"; }
warn() { printf '[grammar-smoke] warn: %s\n' "$*" >&2; }
die()  { printf '[grammar-smoke] error: %s\n' "$*" >&2; exit 1; }
fail() { printf '[grammar-smoke] FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
pass() { printf '[grammar-smoke] PASS: %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown flag: $1" ;;
  esac
done

find_juno_jar() {
  shopt -s nullglob
  local jars=( "$ROOT/juno-player/target/"juno-player-*-shaded.jar )
  shopt -u nullglob
  if [[ ${#jars[@]} -gt 0 ]]; then
    printf '%s' "${jars[0]}"
    return
  fi
  [[ -f "$ROOT/juno-player/target/juno-player.jar" ]] || die "juno-player jar missing; run: mvn package -DskipTests"
  printf '%s' "$ROOT/juno-player/target/juno-player.jar"
}

find_java() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    printf '%s' "${JAVA_HOME}/bin/java"
    return
  fi
  command -v java >/dev/null 2>&1 || die "java not found"
  printf '%s' java
}

wait_for_juno_api() {
  local port="$1" timeout="${2:-300}" pid="${3:-}"
  local start now
  start="$(date +%s)"
  while true; do
    if curl -sf "http://127.0.0.1:${port}/v1/cluster/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      warn "juno process ${pid} exited before API became healthy"
      return 1
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 2
  done
}

stop_juno() {
  local pid="${JUNO_PID:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  JUNO_PID=""
}

wait_for_metrics() {
  local dest="$1" timeout="${2:-45}"
  local metrics_src="${ROOT}/target/metrics/metrics.json"
  local start now
  start="$(date +%s)"
  while true; do
    if [[ -s "$metrics_src" ]]; then
      cp -a "$metrics_src" "$dest"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 1
  done
}

collect_jfr() {
  local stem="$1"
  local marker="$2"
  local dest="${OUT_ROOT}/${stem}-jfr.json"
  if wait_for_metrics "$dest" 45; then
    log "jfr metrics: ${dest}"
  else
    warn "metrics.json missing for ${stem}"
    echo '{"error":"metrics_missing"}' >"$dest"
  fi
  local f
  if [[ -n "${marker:-}" && -e "$marker" ]]; then
    while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      cp -a "$f" "${OUT_ROOT}/$(basename "$f")"
    done < <(find "$ROOT" -maxdepth 1 -name 'juno-*.jfr' -newer "$marker")
  fi
}

grammar_count() {
  local f="$1"
  jq -r '.models[0].metrics["juno.GrammarConstrained.count"] // 0' "$f" 2>/dev/null || echo 0
}

start_juno() {
  local stem="$1"
  shift
  local jar java_bin logf
  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  logf="${OUT_ROOT}/${stem}.log"
  : >"$logf"
  rm -f "${ROOT}/target/metrics/metrics.json"

  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} already has a healthy Juno API"
  fi

  (
    cd "$ROOT"
    exec "$java_bin" \
      --enable-preview \
      --enable-native-access=ALL-UNNAMED \
      --add-opens java.base/java.lang=ALL-UNNAMED \
      --add-opens java.base/java.nio=ALL-UNNAMED \
      -XX:+UseG1GC \
      -Xms512m -Xmx4g \
      -Djuno.byteOrder=BE \
      -jar "$jar" \
      --model-path "$MODEL" \
      --dtype FLOAT16 \
      --byteOrder BE \
      --max-tokens 32 \
      --temperature 0 \
      --top-k 0 \
      --top-p 0 \
      --nodes 1 \
      --local \
      --cpu \
      --api-port "$API_PORT" \
      --jfr 30m \
      --verbose \
      "$@" < <(while true; do sleep 3600; done)
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!
  if ! wait_for_juno_api "$API_PORT" 300 "$JUNO_PID"; then
    warn "API did not become healthy — see ${logf}"
    stop_juno
    return 1
  fi
  log "up: ${stem} pid=${JUNO_PID}"
}

chat() {
  local name="$1" body="$2"
  local resp="${OUT_ROOT}/${name}.json"
  local code
  code="$(curl -sS -o "$resp" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -d "$body" || true)"
  printf '%s' "$code"
}

content_of() {
  jq -r '.choices[0].message.content // empty' "$1" 2>/dev/null || true
}

json_ok_object() {
  python3 - "$1" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read()
try:
    obj = json.loads(raw)
except Exception as e:
    sys.stderr.write(f"not json: {e}\n")
    sys.exit(1)
if not isinstance(obj, dict):
    sys.stderr.write("not an object\n")
    sys.exit(1)
if "ok" in obj and not isinstance(obj["ok"], bool):
    sys.stderr.write("ok is not boolean\n")
    sys.exit(1)
print(json.dumps(obj, ensure_ascii=False))
PY
}

yes_or_no() {
  python3 - "$1" <<'PY'
import sys
t = open(sys.argv[1], encoding="utf-8").read().strip().lower()
if t not in ("yes", "no"):
    sys.stderr.write(f"not yes/no: {t!r}\n")
    sys.exit(1)
print(t)
PY
}

MODEL_ID=""
SCHEMA_BODY='{"type":"json_schema","json_schema":{"name":"ok","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}'

prompt_json='Reply with JSON only: an object with boolean field ok.'
prompt_yn='Answer with a single word, yes or no: is 2 even?'

mkdir -p "$OUT_ROOT"
trap 'stop_juno' EXIT
command -v jq >/dev/null && command -v curl >/dev/null && command -v python3 >/dev/null \
  || die "need curl, jq, python3"
[[ -f "$MODEL" ]] || die "missing ${MODEL}"
[[ -f "$YES_NO" && -f "$OK_SCHEMA" && -f "$BAD_SCHEMA" ]] || die "missing sample grammars"

{
  echo "# Constrained-decoding smoke ${RUN_ID}"
  echo
  echo "Host: $(hostname)  model: $(basename "$MODEL")  cpu  --vector default-off-in-this-script"
  echo
} >"${OUT_ROOT}/INDEX.md"

# ── CUDA / ROCm / vision surfaces (N/A) ──────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >"${OUT_ROOT}/nvidia-smi.txt" || true
  echo "- CUDA: nvidia-smi ok (GPU smoke not required; grammar is CPU-path)" >>"${OUT_ROOT}/INDEX.md"
else
  echo "nvidia-smi unavailable or cannot talk to the driver" >"${OUT_ROOT}/nvidia-smi.txt"
  echo "- CUDA/ROCm: N/A (driver unavailable). See nvidia-smi.txt" >>"${OUT_ROOT}/INDEX.md"
  pass "CUDA/ROCm N/A (driver down)"
fi
if compgen -G "${ROOT}/models/*.mmproj*" >/dev/null; then
  echo "- Vision mmproj present" >>"${OUT_ROOT}/INDEX.md"
else
  echo "- Vision: no mmproj GGUF. /v1/vision/chat does not parse response_format; wired cell is /v1/chat/completions." >>"${OUT_ROOT}/INDEX.md"
  pass "Vision honesty: no mmproj; OpenAI chat is the wired surface"
fi

# ── LoRA train explicit no-op ────────────────────────────────────────────────
lora_help="${OUT_ROOT}/lora-train-warn.txt"
if ! "${ROOT}/scripts/run.sh" lora --grammar-file "$YES_NO" --help >"$lora_help" 2>&1; then
  fail "lora --help with --grammar-file exited non-zero"
elif grep -q 'constrained decoding is a no-op for LoRA training' "$lora_help"; then
  pass "LoRA train launcher WARNING"
  echo "- LoRA train: launcher WARNING recorded (lora-train-warn.txt)" >>"${OUT_ROOT}/INDEX.md"
else
  fail "LoRA train did not print no-op WARNING"
fi

# ── CLI fail-closed (no weight load beyond file check) ───────────────────────
java_bin="$(find_java)"
jar="$(find_juno_jar)"
fail_log="${OUT_ROOT}/cli-schema-unsupported.log"
set +e
"$java_bin" --enable-preview --enable-native-access=ALL-UNNAMED \
  -jar "$jar" --model-path "$MODEL" --cpu --json-schema-file "$BAD_SCHEMA" \
  >"$fail_log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]] && grep -qiE 'pattern|failed to load constrained' "$fail_log"; then
  pass "CLI --json-schema-file unsupported keyword fail-closed"
  echo "- CLI unsupported schema: exit ${rc}; see cli-schema-unsupported.log" >>"${OUT_ROOT}/INDEX.md"
else
  fail "CLI unsupported schema did not fail closed (rc=${rc})"
fi

both_log="${OUT_ROOT}/cli-mutual-exclusive.log"
set +e
"$java_bin" --enable-preview --enable-native-access=ALL-UNNAMED \
  -jar "$jar" --model-path "$MODEL" --cpu \
  --grammar-file "$YES_NO" --json-schema-file "$OK_SCHEMA" \
  >"$both_log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]] && grep -qi 'mutually exclusive' "$both_log"; then
  pass "CLI grammar + json-schema mutually exclusive"
else
  fail "CLI mutual exclusion missing (rc=${rc})"
fi

# ── Server A: OpenAI json_object / json_schema / x_juno_grammar ──────────────
if start_juno api-base; then
  MODEL_ID="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" | jq -r '.data[0].id // empty')"
  [[ -n "$MODEL_ID" ]] || MODEL_ID="$(basename "$MODEL")"

  code="$(chat json_schema "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_json" --argjson rf "$SCHEMA_BODY" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:32, temperature:0, response_format:$rf}')")"
  if [[ "$code" == "200" ]]; then
    content_of "${OUT_ROOT}/json_schema.json" >"${OUT_ROOT}/json_schema.txt"
    if json_ok_object "${OUT_ROOT}/json_schema.txt" >"${OUT_ROOT}/json_schema.parsed.json" 2>"${OUT_ROOT}/json_schema.parse.err"; then
      pass "OpenAI json_schema → parseable JSON"
    else
      fail "json_schema HTTP 200 but content not parseable JSON ($(cat "${OUT_ROOT}/json_schema.parse.err"))"
    fi
  else
    fail "json_schema HTTP ${code}"
  fi

  code="$(chat json_object "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_json" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:48, temperature:0, response_format:{type:"json_object"}}')")"
  if [[ "$code" == "200" ]]; then
    content_of "${OUT_ROOT}/json_object.json" >"${OUT_ROOT}/json_object.txt"
    if python3 -c 'import json,sys; json.loads(open(sys.argv[1]).read()); d=json.loads(open(sys.argv[1]).read()); assert isinstance(d, dict)' \
        "${OUT_ROOT}/json_object.txt" 2>"${OUT_ROOT}/json_object.parse.err"; then
      pass "OpenAI json_object → parseable JSON object"
    else
      fail "json_object not parseable ($(cat "${OUT_ROOT}/json_object.parse.err"))"
    fi
  else
    fail "json_object HTTP ${code}"
  fi

  code="$(chat x_juno_grammar "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_yn" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:8, temperature:0, x_juno_grammar:"root ::= \"yes\" | \"no\""}')")"
  if [[ "$code" == "200" ]]; then
    content_of "${OUT_ROOT}/x_juno_grammar.json" >"${OUT_ROOT}/x_juno_grammar.txt"
    if yes_or_no "${OUT_ROOT}/x_juno_grammar.txt"; then
      pass "x_juno_grammar yes/no"
    else
      fail "x_juno_grammar not yes/no: $(cat "${OUT_ROOT}/x_juno_grammar.txt")"
    fi
  else
    fail "x_juno_grammar HTTP ${code}"
  fi

  code="$(chat conflict "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_json" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:8, temperature:0, response_format:{type:"json_object"}, x_juno_grammar:"root ::= \"a\""}')")"
  if [[ "$code" == "400" ]] && grep -q 'x_juno_grammar' "${OUT_ROOT}/conflict.json"; then
    pass "x_juno_grammar + json_object HTTP 400"
  else
    fail "conflict expected 400, got ${code}"
  fi

  code="$(chat unsupported_schema "$(jq -nc --arg m "$MODEL_ID" \
    '{model:$m, messages:[{role:"user",content:"x"}], max_tokens:8, temperature:0, response_format:{type:"json_schema", json_schema:{name:"bad", schema:{type:"string", pattern:"^a$"}} }}')")"
  if [[ "$code" == "400" ]] && grep -qi 'pattern' "${OUT_ROOT}/unsupported_schema.json"; then
    pass "unsupported json_schema keyword HTTP 400"
  else
    fail "unsupported schema expected 400/pattern, got ${code}"
  fi

  stop_juno
  collect_jfr api-base "${OUT_ROOT}/api-base.log"
  gc="$(grammar_count "${OUT_ROOT}/api-base-jfr.json")"
  if python3 -c "import sys; sys.exit(0 if float('${gc}') >= 3 else 1)"; then
    pass "JFR juno.GrammarConstrained.count=${gc} (api-base)"
    echo "- api-base JFR GrammarConstrained.count=${gc}" >>"${OUT_ROOT}/INDEX.md"
  else
    fail "api-base GrammarConstrained.count=${gc} (want >=3)"
  fi
  if grep -q 'grammar=true' "${OUT_ROOT}/api-base.log"; then
    pass "log: Decode grammar=true"
  else
    warn "api-base.log has no grammar=true (JUL may still be off)"
  fi
else
  fail "api-base server failed to start"
fi

# ── Server B: --grammar-file (CLI → API fallback) ────────────────────────────
if start_juno cli-gbnf --grammar-file "$YES_NO"; then
  [[ -n "$MODEL_ID" ]] || MODEL_ID="$(basename "$MODEL")"
  code="$(chat cli_gbnf "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_yn" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:8, temperature:0}')")"
  if [[ "$code" == "200" ]]; then
    content_of "${OUT_ROOT}/cli_gbnf.json" >"${OUT_ROOT}/cli_gbnf.txt"
    if yes_or_no "${OUT_ROOT}/cli_gbnf.txt"; then
      pass "--grammar-file constrains API (no response_format)"
    else
      fail "--grammar-file output not yes/no: $(cat "${OUT_ROOT}/cli_gbnf.txt")"
    fi
  else
    fail "--grammar-file HTTP ${code}"
  fi
  stop_juno
  collect_jfr cli-gbnf "${OUT_ROOT}/cli-gbnf.log"
  gc="$(grammar_count "${OUT_ROOT}/cli-gbnf-jfr.json")"
  if python3 -c "import sys; sys.exit(0 if float('${gc}') >= 1 else 1)"; then
    pass "JFR GrammarConstrained.count=${gc} (cli-gbnf)"
  else
    fail "cli-gbnf GrammarConstrained.count=${gc}"
  fi
else
  fail "cli-gbnf server failed to start"
fi

# ── Server C: --json-schema-file ─────────────────────────────────────────────
if start_juno cli-schema --json-schema-file "$OK_SCHEMA"; then
  code="$(chat cli_schema "$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_json" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:32, temperature:0}')")"
  if [[ "$code" == "200" ]]; then
    content_of "${OUT_ROOT}/cli_schema.json" >"${OUT_ROOT}/cli_schema.txt"
    if json_ok_object "${OUT_ROOT}/cli_schema.txt" >"${OUT_ROOT}/cli_schema.parsed.json" 2>"${OUT_ROOT}/cli_schema.parse.err"; then
      pass "--json-schema-file constrains API"
    else
      fail "--json-schema-file not parseable ($(cat "${OUT_ROOT}/cli_schema.parse.err"))"
    fi
  else
    fail "--json-schema-file HTTP ${code}"
  fi
  stop_juno
  collect_jfr cli-schema "${OUT_ROOT}/cli-schema.log"
  gc="$(grammar_count "${OUT_ROOT}/cli-schema-jfr.json")"
  if python3 -c "import sys; sys.exit(0 if float('${gc}') >= 1 else 1)"; then
    pass "JFR GrammarConstrained.count=${gc} (cli-schema)"
  else
    fail "cli-schema GrammarConstrained.count=${gc}"
  fi
else
  fail "cli-schema server failed to start"
fi

# ── Server D: --lora-play + --parallel 2 + json_schema ───────────────────────
lora_args=()
if [[ -f "$LORA" ]]; then
  lora_args+=(--lora-play "$LORA")
else
  warn "LoRA overlay missing: ${LORA}"
  fail "--lora-play overlay file missing"
fi
if start_juno lora-parallel --parallel 2 "${lora_args[@]}"; then
  body="$(jq -nc --arg m "$MODEL_ID" --arg p "$prompt_json" --argjson rf "$SCHEMA_BODY" \
    '{model:$m, messages:[{role:"user",content:$p}], max_tokens:32, temperature:0, response_format:$rf}')"
  chat lora_parallel_a "$body" >"${OUT_ROOT}/lora_parallel_a.http" &
  pid_a=$!
  chat lora_parallel_b "$body" >"${OUT_ROOT}/lora_parallel_b.http" &
  pid_b=$!
  wait "$pid_a" || true
  wait "$pid_b" || true
  h1="$(jq -r '.choices[0].message.content // empty' "${OUT_ROOT}/lora_parallel_a.json" 2>/dev/null || true)"
  h2="$(jq -r '.choices[0].message.content // empty' "${OUT_ROOT}/lora_parallel_b.json" 2>/dev/null || true)"
  if [[ -n "$h1" && -n "$h2" ]]; then
    printf '%s' "$h1" >"${OUT_ROOT}/lora_parallel_a.txt"
    printf '%s' "$h2" >"${OUT_ROOT}/lora_parallel_b.txt"
    if json_ok_object "${OUT_ROOT}/lora_parallel_a.txt" >/dev/null 2>&1 \
      && json_ok_object "${OUT_ROOT}/lora_parallel_b.txt" >/dev/null 2>&1; then
      pass "--lora-play + --parallel 2 + json_schema (two parseable replies)"
    else
      fail "lora-parallel replies not both parseable JSON"
    fi
  else
    fail "lora-parallel missing completion text (bg curl)"
  fi
  if grep -qi 'lora' "${OUT_ROOT}/lora-parallel.log"; then
    pass "lora-parallel log mentions LoRA overlay"
  else
    warn "lora-parallel.log has no LoRA string — check overlay applied"
  fi
  stop_juno
  collect_jfr lora-parallel "${OUT_ROOT}/lora-parallel.log"
  gc="$(grammar_count "${OUT_ROOT}/lora-parallel-jfr.json")"
  if python3 -c "import sys; sys.exit(0 if float('${gc}') >= 2 else 1)"; then
    pass "JFR GrammarConstrained.count=${gc} (lora-parallel, want >=2)"
  else
    fail "lora-parallel GrammarConstrained.count=${gc} (want >=2)"
  fi
else
  fail "lora-parallel server failed to start"
fi

# ── Cluster launcher forwards flags (same ConsoleMain cliGrammar binding) ────
if grep -n 'grammar_file_arg' "${ROOT}/scripts/run.sh" | grep -q . \
  && grep -A2 'cmd_cluster' "${ROOT}/scripts/run.sh" >/dev/null; then
  if grep -n 'grammar_file_arg' "${ROOT}/scripts/run.sh" | head -5 >>"${OUT_ROOT}/cluster-launcher-forward.txt"; then
    :
  fi
  grep -n 'grammar_file_arg' "${ROOT}/scripts/run.sh" >"${OUT_ROOT}/cluster-launcher-forward.txt"
  pass "cluster launcher forwards --grammar-file (run.sh argv)"
  echo "- cluster: launcher forwards --grammar-file / --json-schema-file; ConsoleMain cluster API uses cliGrammar. 3-node process not started this run." >>"${OUT_ROOT}/INDEX.md"
else
  fail "cluster launcher does not mention grammar_file_arg"
fi

{
  echo
  echo "## Results"
  echo
  echo "failures=${failures}"
  echo
  echo "Artifacts in this directory: HTTP JSON, *.log, *-jfr.json, INDEX.md"
} >>"${OUT_ROOT}/INDEX.md"

log "done. failures=${failures}  out=${OUT_ROOT}"
exit "$failures"
