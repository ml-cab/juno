#!/usr/bin/env bash
# compare-lora.sh — LoRA train-qa + playback benchmark
#
# Runs the canonical TinyLlama name-recall scenario (fresh adapter → /train-qa → playback)
# via ./juno lora --jfr and writes JSON under target/perf-compare-lora/.
#
# Usage:
#   ./scripts/performance-tests/compare-lora.sh --gpu
#   ./scripts/performance-tests/compare-lora.sh --gpu --baseline release-0.1.2
#   ./scripts/performance-tests/compare-lora.sh --cpu --no-publish
#
# Regression gate (default): train ms/pass or total ms > 1.25× baseline → exit 1.

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare-lora/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

MODEL="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
QUESTION="${LORA_PERF_QUESTION:-What is your name?}"
ANSWER="${LORA_PERF_ANSWER:-My name is Juno}"
LOSS_TARGET="${LORA_PERF_LOSS_TARGET:-1.2}"
USE_GPU=1
HEAP="${COMPARE_HEAP:-4g}"
JFR_DURATION="${JFR_DURATION:-10m}"
PUBLISH=1
BASELINE_REF=""
CURRENT_REF=""
REGRESSION_RATIO="${LORA_PERF_REGRESSION_RATIO:-1.25}"
DRY_RUN=0
SKIP_BUILD=0

log()  { printf '[lora-perf] %s\n' "$*"; }
warn() { printf '[lora-perf] warn: %s\n' "$*" >&2; }
die()  { printf '[lora-perf] error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \?//'
  cat <<EOF

Options:
  --model FILE          GGUF basename under models/ (default: ${MODEL})
  --gpu / --cpu         Train + playback backend (default: gpu)
  --heap SIZE           JVM -Xmx (default: ${HEAP})
  --jfr DURATION        JFR recording duration (default: ${JFR_DURATION})
  --out DIR             Output directory (default: target/perf-compare-lora/<timestamp>)
  --baseline REF        Git ref to compare against (e.g. release-0.1.2)
  --current REF         Git ref for current run (default: HEAD)
  --regression-ratio F  Fail when current/baseline > F (default: ${REGRESSION_RATIO})
  --skip-build          Skip mvn package (use existing jar)
  --no-publish          Skip docs/perf-compare copy
  -n, --dry-run         Print planned steps only
  -h, --help            This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --gpu) USE_GPU=1; shift ;;
    --cpu) USE_GPU=0; shift ;;
    --heap) HEAP="$2"; shift 2 ;;
    --jfr) JFR_DURATION="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --baseline) BASELINE_REF="$2"; shift 2 ;;
    --current) CURRENT_REF="$2"; shift 2 ;;
    --regression-ratio) REGRESSION_RATIO="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }
require_cmd git
require_cmd jq

MODEL_PATH="${MODELS_DIR}/${MODEL}"
[[ -f "$MODEL_PATH" ]] || die "model not found: ${MODEL_PATH} (download TinyLlama Q4_K_M first)"

setup_cuda_env() {
  if [[ -f "${ROOT}/scripts/set_cuda_env.sh" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${ROOT}/scripts/set_cuda_env.sh" || true
    set -u
  fi
  if [[ "$USE_GPU" -eq 1 ]]; then
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]]; then
      export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}

strip_ansi() {
  sed 's/\x1b\[[0-9;]*m//g'
}

build_juno() {
  local dir="$1"
  [[ "$SKIP_BUILD" -eq 1 ]] && return 0
  log "building juno-player in ${dir}…"
  (cd "$dir" && mvn -q package -DskipTests -pl juno-player -am)
}

wait_for_jfr_metrics() {
  local src="$1" dest="$2" timeout="${3:-60}"
  local start now
  start="$(date +%s)"
  while true; do
    if [[ -s "$src" ]]; then
      cp -a "$src" "$dest"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 1
  done
}

# Parse REPL log after strip_ansi. Sets globals used by write_bench_json.
parse_lora_bench_log() {
  local logf="$1"
  local clean train_line playback_line reply

  clean="$(strip_ansi <"$logf")"

  train_line="$(grep -E 'train-loss=[0-9.,]+.*passes=[0-9]+.*opt-updates=[0-9]+.*[0-9]+s total' <<<"$clean" | tail -1 || true)"
  [[ -n "$train_line" ]] || return 1

  PARSED_TRAIN_LOSS="$(sed -nE 's/.*train-loss=([0-9.,]+).*/\1/p' <<<"$train_line" | tr ',' '.')"
  PARSED_TRAIN_PASSES="$(sed -nE 's/.*passes=([0-9]+).*/\1/p' <<<"$train_line")"
  PARSED_TRAIN_UPDATES="$(sed -nE 's/.*opt-updates=([0-9]+).*/\1/p' <<<"$train_line")"
  PARSED_TRAIN_TOTAL_S="$(sed -nE 's/.* ([0-9]+)s total .*/\1/p' <<<"$train_line")"

  playback_line="$(grep -E '\[[0-9]+ tokens · [0-9]+ ms ·' <<<"$clean" | tail -1 || true)"
  PARSED_PLAY_TOKENS=0
  PARSED_PLAY_MS=0
  if [[ -n "$playback_line" ]]; then
    PARSED_PLAY_TOKENS="$(sed -nE 's/.*\[([0-9]+) tokens · ([0-9]+) ms ·.*/\1/p' <<<"$playback_line")"
    PARSED_PLAY_MS="$(sed -nE 's/.*\[([0-9]+) tokens · ([0-9]+) ms ·.*/\2/p' <<<"$playback_line")"
  fi

  reply="$(grep 'bot>' <<<"$clean" | tail -1 | sed -E 's/^.*bot>[[:space:]]*//' | tr -d '\r' || true)"
  PARSED_REPLY="${reply:-}"
  PARSED_RECALL_OK=0
  if grep -qi 'juno' <<<"${PARSED_REPLY}"; then
    PARSED_RECALL_OK=1
  fi

  if grep -q 'target reached' <<<"$clean"; then
    PARSED_STOP_REASON="TARGET_REACHED"
    PARSED_TARGET_REACHED=1
  elif grep -q 'max iters reached' <<<"$clean"; then
    PARSED_STOP_REASON="MAX_ITERATIONS"
    PARSED_TARGET_REACHED=0
  elif grep -q 'overfit guard' <<<"$clean"; then
    PARSED_STOP_REASON="LOW_LOSS_GUARD"
    PARSED_TARGET_REACHED=0
  elif grep -q 'validation patience exhausted' <<<"$clean"; then
    PARSED_STOP_REASON="PATIENCE_EXHAUSTED"
    PARSED_TARGET_REACHED=0
  else
    PARSED_STOP_REASON="UNKNOWN"
    PARSED_TARGET_REACHED=0
  fi

  return 0
}

write_bench_json() {
  local out_json="$1"
  local ref="$2"
  local commit="$3"
  local host="$4"
  local adapter="$5"
  local jfr_metrics="$6"
  local logf="$7"

  local train_total_ms train_ms_per_pass play_tps play_tps_jfr jfr_block
  train_total_ms=$(( PARSED_TRAIN_TOTAL_S * 1000 ))
  if (( PARSED_TRAIN_PASSES > 0 )); then
    train_ms_per_pass="$(awk -v ms="$train_total_ms" -v p="$PARSED_TRAIN_PASSES" 'BEGIN { printf "%.6f", ms / p }')"
  else
    train_ms_per_pass="null"
  fi

  play_tps="null"
  if (( PARSED_PLAY_MS > 0 && PARSED_PLAY_TOKENS > 0 )); then
    play_tps="$(awk -v t="$PARSED_PLAY_TOKENS" -v ms="$PARSED_PLAY_MS" 'BEGIN { printf "%.6f", t / (ms / 1000.0) }')"
  fi

  jfr_block="null"
  play_tps_jfr="null"
  if [[ -f "$jfr_metrics" ]]; then
    jfr_block="$(jq -c '.models[0] // null' "$jfr_metrics" 2>/dev/null || echo null)"
    play_tps_jfr="$(jq -r '.models[0].metrics["juno.TokenProduced.tps"] // empty' "$jfr_metrics" 2>/dev/null || true)"
    if [[ -n "$play_tps_jfr" && "$play_tps_jfr" != "null" ]]; then
      play_tps="$play_tps_jfr"
    fi
  fi

  local status="success"
  [[ "$PARSED_RECALL_OK" -eq 1 ]] || status="recall_failed"

  jq -n \
    --arg tool "compare-lora+jfr" \
    --arg run_id "$RUN_ID" \
    --arg git_ref "$ref" \
    --arg git_commit "$commit" \
    --arg host "$host" \
    --arg model "$MODEL" \
    --arg model_path "$MODEL_PATH" \
    --arg adapter_path "$adapter" \
    --arg backend "$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu)" \
    --arg heap "$HEAP" \
    --arg jfr_duration "$JFR_DURATION" \
    --arg question "$QUESTION" \
    --arg answer "$ANSWER" \
    --argjson loss_target "$LOSS_TARGET" \
    --argjson train_passes "$PARSED_TRAIN_PASSES" \
    --argjson train_updates "$PARSED_TRAIN_UPDATES" \
    --argjson train_total_ms "$train_total_ms" \
    --arg train_ms_per_pass "$train_ms_per_pass" \
    --arg train_loss "$PARSED_TRAIN_LOSS" \
    --argjson target_reached "$PARSED_TARGET_REACHED" \
    --arg stop_reason "$PARSED_STOP_REASON" \
    --arg reply "$PARSED_REPLY" \
    --argjson recall_ok "$([ "$PARSED_RECALL_OK" -eq 1 ] && echo true || echo false)" \
    --argjson play_tokens "$PARSED_PLAY_TOKENS" \
    --argjson play_ms "$PARSED_PLAY_MS" \
    --arg play_tps "$play_tps" \
    --arg play_tps_jfr "${play_tps_jfr:-null}" \
    --arg log "$logf" \
    --argjson jfr_model "$jfr_block" \
    --arg status "$status" \
    '{
      tool: $tool,
      run_id: $run_id,
      git_ref: $git_ref,
      git_commit: $git_commit,
      host: $host,
      model: $model,
      model_path: $model_path,
      adapter_path: $adapter_path,
      model_type: null,
      backend: $backend,
      train_device: "auto",
      heap: $heap,
      use_jfr: true,
      jfr_duration: $jfr_duration,
      scenario: {
        question: $question,
        answer: $answer,
        loss_target: $loss_target,
        max_iters: 50
      },
      config: {
        rank: 8,
        alpha: 8,
        lr: 0.0001,
        microbatch: 8,
        targets: "qv"
      },
      train: {
        passes: $train_passes,
        final_loss: ($train_loss | tonumber),
        target_reached: $target_reached,
        stop_reason: $stop_reason,
        optimizer_updates: $train_updates,
        total_ms: $train_total_ms,
        ms_per_pass: (if $train_ms_per_pass == "null" then null else ($train_ms_per_pass | tonumber) end)
      },
      playback: {
        reply: $reply,
        recall_ok: $recall_ok,
        tokens: $play_tokens,
        latency_ms: $play_ms,
        wall_ms: $play_ms,
        tps: (if $play_tps == "null" then null else ($play_tps | tonumber) end),
        tps_jfr: (if $play_tps_jfr == "null" or $play_tps_jfr == "" then null else ($play_tps_jfr | tonumber) end)
      },
      jfr: $jfr_model,
      log: $log,
      status: $status
    }' >"$out_json"
}

run_bench_at() {
  local workdir="$1"
  local ref="$2"
  local out_json="$3"
  local logf="${out_json%.json}.log"
  local jfr_metrics="${out_json%.json}-jfr.json"
  local adapter="${OUT_ROOT}/adapter-${ref//\//_}.lora"
  local juno="${workdir}/juno"
  local metrics_src="${workdir}/target/metrics/metrics.json"
  local backend_flag="--gpu"
  [[ "$USE_GPU" -eq 0 ]] && backend_flag="--cpu"

  local commit host
  commit="$(git -C "$workdir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  host="$(hostname 2>/dev/null || echo unknown)"

  log "benchmark ref=${ref} commit=${commit} backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu) jfr=${JFR_DURATION}"

  local -a juno_args=(
    lora
    --model-path "$MODEL_PATH"
    --lora-path "$adapter"
    --lora-loss-target-qa "$LOSS_TARGET"
    --temperature 0
    --heap "$HEAP"
    --jfr "$JFR_DURATION"
    "$backend_flag"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: (cd ${workdir} && printf ... | ${juno} ${juno_args[*]})"
  else
    [[ -x "$juno" ]] || die "juno launcher missing in ${workdir} — build failed?"
    rm -f "$adapter"
    rm -f "$metrics_src"
    mkdir -p "${workdir}/target/metrics"

    setup_cuda_env
    (
      cd "$workdir"
      printf '%s\n' \
        "/train-qa ${QUESTION} A: ${ANSWER}" \
        "$QUESTION" \
        /save \
        quit \
      | "$juno" "${juno_args[@]}"
    ) >"$logf" 2>&1

    if ! parse_lora_bench_log "$logf"; then
      die "could not parse training summary from ${logf}"
    fi

    if ! wait_for_jfr_metrics "$metrics_src" "$jfr_metrics" 60; then
      warn "JFR metrics.json not found — continuing with REPL log metrics only"
      : >"$jfr_metrics"
      echo '{"models":[]}' >"$jfr_metrics"
    fi

    write_bench_json "$out_json" "$ref" "$commit" "$host" "$adapter" "$jfr_metrics" "$logf"

    jq -e '.status == "success"' "$out_json" >/dev/null \
      || die "benchmark failed for ref=${ref} — see ${logf} and ${out_json}"
  fi
}

stage_worktree() {
  local ref="$1"
  local dest="$2"
  git -C "$ROOT" worktree add --detach "$dest" "$ref" >/dev/null
  mkdir -p "$dest/scripts/performance-tests"
  cp "${PERF_SCRIPTS}/compare-lora.sh" "$dest/scripts/performance-tests/"
}

cleanup_worktree() {
  local dest="$1"
  git -C "$ROOT" worktree remove --force "$dest" 2>/dev/null || rm -rf "$dest"
}

run_at_ref() {
  local ref="$1"
  local label="$2"
  local out_json="${OUT_ROOT}/${label}.json"

  if [[ "$ref" == "HEAD" || "$ref" == "$(git -C "$ROOT" rev-parse HEAD)" ]]; then
    build_juno "$ROOT"
    run_bench_at "$ROOT" "$ref" "$out_json"
    return 0
  fi

  local wt
  wt="$(mktemp -d "${ROOT}/target/lora-perf-wt-XXXXXX")"
  trap 'cleanup_worktree "$wt"' RETURN
  stage_worktree "$ref" "$wt"
  build_juno "$wt"
  run_bench_at "$wt" "$ref" "$out_json"
  cleanup_worktree "$wt"
  trap - RETURN
}

write_compare_json() {
  local baseline_json="$1"
  local current_json="$2"
  local compare_json="$3"

  jq -s --argjson ratio "$REGRESSION_RATIO" '
    def num($x): if ($x | type) == "number" then $x else null end;
    .[0] as $base | .[1] as $cur |
    {
      tool: "compare-lora",
      run_id: ($cur.run_id // $base.run_id),
      model: ($cur.model // $base.model),
      backend: ($cur.backend // $base.backend),
      use_jfr: true,
      baseline: {
        git_ref: $base.git_ref,
        git_commit: $base.git_commit,
        train_total_ms: $base.train.total_ms,
        train_ms_per_pass: $base.train.ms_per_pass,
        train_passes: $base.train.passes,
        playback_tps: $base.playback.tps,
        playback_latency_ms: $base.playback.latency_ms,
        recall_ok: $base.playback.recall_ok
      },
      current: {
        git_ref: $cur.git_ref,
        git_commit: $cur.git_commit,
        train_total_ms: $cur.train.total_ms,
        train_ms_per_pass: $cur.train.ms_per_pass,
        train_passes: $cur.train.passes,
        playback_tps: $cur.playback.tps,
        playback_latency_ms: $cur.playback.latency_ms,
        recall_ok: $cur.playback.recall_ok
      },
      ratios: {
        train_total_ms: (if ($base.train.total_ms // 0) > 0 then ($cur.train.total_ms / $base.train.total_ms) else null end),
        train_ms_per_pass: (if ($base.train.ms_per_pass // 0) > 0 then ($cur.train.ms_per_pass / $base.train.ms_per_pass) else null end),
        playback_tps: (if ($base.playback.tps // 0) > 0 then ($cur.playback.tps / $base.playback.tps) else null end),
        playback_latency_ms: (if ($base.playback.latency_ms // 0) > 0 then ($cur.playback.latency_ms / $base.playback.latency_ms) else null end)
      },
      regression_ratio_limit: $ratio,
      regressions: [
        (if ($base.train.total_ms // 0) > 0 and ($cur.train.total_ms / $base.train.total_ms) > $ratio
         then "train_total_ms" else empty end),
        (if ($base.train.ms_per_pass // 0) > 0 and ($cur.train.ms_per_pass / $base.train.ms_per_pass) > $ratio
         then "train_ms_per_pass" else empty end),
        (if ($base.playback.tps // 0) > 0 and ($cur.playback.tps / $base.playback.tps) < (1 / $ratio)
         then "playback_tps" else empty end),
        (if ($base.playback.recall_ok == true and $cur.playback.recall_ok != true)
         then "recall_ok" else empty end)
      ],
      status: (
        if ($base.playback.recall_ok != true or $cur.playback.recall_ok != true) then "recall_failed"
        elif (
          (($base.train.total_ms // 0) > 0 and ($cur.train.total_ms / $base.train.total_ms) > $ratio) or
          (($base.train.ms_per_pass // 0) > 0 and ($cur.train.ms_per_pass / $base.train.ms_per_pass) > $ratio) or
          (($base.playback.tps // 0) > 0 and ($cur.playback.tps / $base.playback.tps) < (1 / $ratio))
        ) then "regression"
        else "ok"
        end
      )
    }
  ' "$baseline_json" "$current_json" >"$compare_json"
}

write_index_md() {
  local index_md="$1"
  local baseline_json="$2"
  local current_json="$3"
  local compare_json="$4"

  {
    echo "# LoRA perf compare — ${RUN_ID}"
    echo
    echo "Model: \`${MODEL}\` · backend=$([ "$USE_GPU" -eq 1 ] && echo gpu || echo cpu) · scenario: train-qa name recall · JFR \`${JFR_DURATION}\`"
    echo
    echo "| ref | commit | train ms | ms/pass | passes | playback tps | recall |"
    echo "|-----|--------|----------:|--------:|-------:|-------------:|:------:|"
    for f in "$baseline_json" "$current_json"; do
      [[ -f "$f" ]] || continue
      jq -r '"| \(.git_ref // "?") | \(.git_commit // "?") | \(.train.total_ms) | \(.train.ms_per_pass) | \(.train.passes) | \(.playback.tps) | \(.playback.recall_ok) |"' "$f"
    done
    echo
    if [[ -f "$compare_json" ]]; then
      echo "## Comparison"
      echo
      jq -r '
        "- baseline: \(.baseline.git_ref) (\(.baseline.git_commit))",
        "- current: \(.current.git_ref) (\(.current.git_commit))",
        "- train_total_ms ratio: \(.ratios.train_total_ms // "-")",
        "- train_ms_per_pass ratio: \(.ratios.train_ms_per_pass // "-")",
        "- playback_tps ratio: \(.ratios.playback_tps // "-")",
        "- status: **\(.status)**",
        (if (.regressions | length) > 0 then "- regressions: " + (.regressions | join(", ")) else empty end)
      ' "$compare_json"
    fi
    echo
    echo "Train/playback timings from REPL log; playback tps prefers \`juno.TokenProduced.tps\` from JFR when present."
  } >"$index_md"
}

mkdir -p "$OUT_ROOT"

CURRENT_REF="${CURRENT_REF:-HEAD}"

if [[ -n "$BASELINE_REF" ]]; then
  log "comparing baseline=${BASELINE_REF} vs current=${CURRENT_REF}"
  run_at_ref "$BASELINE_REF" "baseline"
  run_at_ref "$CURRENT_REF" "current"
  write_compare_json "${OUT_ROOT}/baseline.json" "${OUT_ROOT}/current.json" "${OUT_ROOT}/compare.json"
  write_index_md "${OUT_ROOT}/INDEX.md" "${OUT_ROOT}/baseline.json" "${OUT_ROOT}/current.json" "${OUT_ROOT}/compare.json"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    status="$(jq -r '.status' "${OUT_ROOT}/compare.json")"
    log "comparison status: ${status}"
    jq -r '"train_total_ms ratio=\(.ratios.train_total_ms // "-") ms_per_pass ratio=\(.ratios.train_ms_per_pass // "-") playback_tps ratio=\(.ratios.playback_tps // "-")"' \
      "${OUT_ROOT}/compare.json"
    [[ "$status" == "ok" ]] || exit 1
  fi
else
  run_at_ref "$CURRENT_REF" "current"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    cp "${OUT_ROOT}/current.json" "${OUT_ROOT}/lora-perf.json"
  fi
  write_index_md "${OUT_ROOT}/INDEX.md" "${OUT_ROOT}/current.json" "${OUT_ROOT}/current.json" ""
fi

if [[ "$PUBLISH" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}-lora"
  mkdir -p "$dest"
  cp -a "$OUT_ROOT"/* "$dest/"
  log "published → docs/perf-compare/${RUN_ID}-lora/"
fi

log "done: ${OUT_ROOT}"
