#!/usr/bin/env bash
# compare-llama-cpp.sh — local perf compare: llama.cpp vs Juno (CPU or GPU)
#
# For each selected GGUF under models/, runs:
#   1) llama-bench  (prompt eval + token gen tokens/s)
#   2) juno local API with --jfr (pp/tg from JFR ForwardPass + TokenProduced; API latency kept too)
# and writes one result JSON per engine per model under target/perf-compare/<run-id>/.
# On success, also publishes metrics into docs/perf-compare/<run-id>/.
#
# Baselines: docs/perf-compare/README.md
#
# Usage:
#   ./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 0
#   ./scripts/performance-tests/compare-llama-cpp.sh --gpu
#   ./scripts/performance-tests/compare-llama-cpp.sh --gpu --models tinyllama,qwen2.5-3b
#   ./scripts/performance-tests/compare-llama-cpp.sh --list
#   ./scripts/performance-tests/compare-llama-cpp.sh --no-publish
#
# Env:
#   LLAMA_CPP_BIN   dir containing llama-bench (default: CUDA build if --gpu and present,
#                   else ../llama.cpp-bin/llama-b9551)
#   JUNO_USE_VECTOR 1 = keep jdk.incubator.vector (default); 0 = scalar kernels
#   COMPARE_HEAP    JVM -Xmx for juno (default: auto from model size)
#   CUDA_HOME / CUDA_PATH  optional; prepended to LD_LIBRARY_PATH for Juno GPU

set -euo pipefail

PERF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perf-lib.sh
source "${PERF_SCRIPTS}/perf-lib.sh"
ROOT="$(cd "${PERF_SCRIPTS}/../.." && pwd)"
MODELS_DIR="${ROOT}/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${ROOT}/target/perf-compare/${RUN_ID}"
DOCS_PUBLISH_ROOT="${ROOT}/docs/perf-compare"

N_PROMPT=128
N_GEN=64
REPS=3
# Juno's own warmup and repetition counts, which are not the reference tool's.
# llama-bench runs its own warmup and -r repetitions internally; Juno was measured
# from a single request on a just-started JVM, so the compilation of the whole
# forward pass sat inside the measurement window and every published Juno number
# was a cold one. These two close that asymmetry: JUNO_WARMUP requests are issued
# and discarded before the measured one, and JUNO_REPS whole cycles are measured
# with the median published and the spread recorded next to it.
JUNO_WARMUP=2
JUNO_REPS=3
# Generation-length parity. The reference tool generates its requested token count
# whatever the model would rather do; Juno stops at a stop token unless asked not
# to. Defaulting this to the requested count makes the two comparable, which is
# resolved after flag parsing since it follows --n-gen. Set to 0 to measure Juno
# as a caller would experience it, at the cost of a comparable generation ratio.
JUNO_MIN_TOKENS=""
# The generation lane prompt: as short as the chat template allows, so decode runs
# at the shallow context the reference tool measures its own generation at.
GENERATE_LANE_PROMPT="Hi"
N_THREADS="$(nproc 2>/dev/null || echo 6)"
# Juno's CPU kernels dispatch on the common fork-join pool, and the calling thread joins
# the work as one more worker. Juno is launched with the pool's parallelism set to
# N_THREADS - 1 (resolved after the flags are parsed), so its hot path runs the same
# N_THREADS threads the reference tool is given with -t. Recorded as juno_threads.
JUNO_POOL_PARALLELISM=""
JUNO_THREADS=""
NGL=""
API_PORT=18080
JUNO_USE_VECTOR="${JUNO_USE_VECTOR:-1}"
PROMPT_TEXT="could you please write me a short poem about love and war"
# On by default: the reference tool prefills N_PROMPT tokens, so Juno has to be
# given a prompt of about the same length or the prefill columns compare two
# different amounts of work. Turn it off only for a Juno-only measurement.
RAW_PROMPT=1
# A prefill ratio is only published when Juno's real prompt_tokens is within this
# fraction of n_prompt. Beyond it the two engines did measurably different work.
PROMPT_PARITY_TOLERANCE=0.10
# How far the repetitions of one reading may spread before the reading is not worth
# scoring. Anchored to this host's own measurement floor: two sweeps taken minutes
# apart with identical flags moved a reference figure by 14%, so a spread wider than
# this says the number is not stable at the resolution a gate would read it at.
REP_SPREAD_TOLERANCE=0.15
JUNO_GPU_LAYERS=""
JUNO_MMQ=""
JUNO_GPU_ATTENTION=""
JUNO_GPU_RESIDENCY=""
JUNO_SCHEDULE=""
JUNO_CACHE_TYPE_K=""
JUNO_CACHE_TYPE_V=""
JUNO_KV_PAGE_SIZE=""
MODEL_FILTER=""
TUNED_LANE=1
DRY_RUN=0
LIST_ONLY=0
SELFTEST_ONLY=0
PUBLISH=1
USE_GPU=0
LLAMA_CPP_BIN_EXPLICIT=""
JUNO_JAR_EXPLICIT=""
# Clock pinning for gate runs. Recording the governor, turbo state and GPU clocks
# tells a reader a run was taken at a moving clock; it does not stop the clock from
# moving, and this host's reference tool has moved -14% and +16% between two sweeps
# eight minutes apart. --pin-clocks fixes the CPU governor at performance and turns
# turbo off for the run (restored on exit), and locks the GPU graphics clock where
# the driver allows it. It needs sudo without a prompt (run `sudo -v` first) and
# refuses to start a run it could not pin, rather than publishing an unpinned run
# that believes it was pinned.
PIN_CLOCKS=0
PIN_GPU_MHZ=""
CLOCK_PIN_STATE="not requested"
PINNED_PREV_GOVERNOR=""
PINNED_PREV_NO_TURBO=""
PINNED_GPU=0
# The JVM flags every Juno cycle runs with, recorded so a run states what it ran
# under. The heap is pinned per model (-Xms equal to -Xmx) so the collector does not
# resize it inside a measurement.
JUNO_JVM_FLAGS=(-XX:+UseG1GC -XX:+AlwaysPreTouch)
# JFR reads the operating-system clock where the kernel has rejected the CPU timestamp
# counter; otherwise a span crossing two cores with disagreeing counters is misread
# (perf-lib.sh, "JFR timestamp source"). PERF_JFR_OS_CLOCK=0 keeps the JVM default.
# shellcheck disable=SC2206
JUNO_JVM_FLAGS+=($(perf_jfr_clock_jvm_flags))
USE_JFR=1
JFR_DURATION="${JFR_DURATION:-30m}"
# The one measurement configuration every recording in this project names, so two
# runs carry the same instrumentation overhead and stay comparable.
JFR_SETTINGS_FILE="${JUNO_JFR_SETTINGS_FILE:-${PERF_SCRIPTS}/juno-perf.jfc}"
# --device-spans layers juno-perf-spans.jfc over it, enabling juno.DeviceStaging,
# juno.WeightDequant and juno.DeviceCompute: the host-device copy, dequantization and
# device kernel totals a prefill breakdown reads. Off by default because counting and timing tens of thousands of
# copies per prefill window cost 4% to 7% of TinyLlama prefill; bytes are exact
# either way, so a spans lane is for attribution, not for a throughput figure.
DEVICE_SPANS=0
JFR_SPANS_FILE="${PERF_SCRIPTS}/juno-perf-spans.jfc"
JFR_RECORDING_NAME=juno-compare
# Discarded; runs over the last warmup request only (run_juno_rep).
JFR_WARMUP_RECORDING_NAME=juno-compare-warmup
# Set per rep: a rep whose recording could not be started is measured without one
# rather than abandoned, and says so.
USE_JFR_THIS_REP=0

# Default model set: text GGUFs both engines load locally (skip huge / vision / llamafile /
# architectures Juno cannot load yet, e.g. Qwen3.5).
DEFAULT_MODELS=(
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  qwen2.5-3b-instruct-q4_k_m.gguf
  Phi-3.5-mini-instruct-Q4_K_M.gguf
  mistral-7b-instruct-v0.1-q4_k_m.gguf
)

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
  cat <<EOF

Options:
  --models LIST     Comma-separated basenames or substrings (default: curated set)
  --all-gguf        Every *.gguf under models/ (except skipped patterns)
  --n-prompt N      llama-bench prompt tokens (default: ${N_PROMPT})
  --n-gen N         gen tokens for both engines (default: ${N_GEN})
  --reps N          llama-bench repetitions (default: ${REPS})
  --juno-warmup N   Juno requests issued and discarded before the measured one,
                    so the measured request runs on already-compiled code
                    (default: ${JUNO_WARMUP})
  --juno-reps N     Measured Juno cycles per lane; the median is published and the
                    min/max spread is recorded beside it (default: ${JUNO_REPS})
  --juno-min-tokens N  Tokens Juno must generate before the model's own end
                    signals (end-of-sequence, chat turn markers) may end the
                    request (default: the same as --n-gen, which is what the
                    reference tool generates). 0 lets Juno stop early, which makes
                    the generation ratio incomparable and is reported as such
  --threads N       CPU threads (default: ${N_THREADS})
  --cpu             Force CPU (llama -ngl 0, juno --cpu) [default]
  --gpu             GPU mode (llama -ngl 99 unless --ngl set, juno --gpu)
  --ngl N           llama.cpp GPU layers (overrides --cpu/--gpu default)
  --gpu-layers N|all|auto  Juno --gpu-layers (default: all in GPU mode)
  --mmq on|off|auto Juno --mmq (packed Q4_K device GEMV; default off)
  --gpu-attention on|off|auto  Juno --gpu-attention (GPU-resident attention kernel). Not passed
                    unless set, so a lane runs the engine's own default (auto: on under CUDA
                    for Llama-family handlers, scalar elsewhere); each result records the
                    value the engine actually resolved
  --gpu-residency on|off|auto  Juno --gpu-residency (device-resident decode region: norm, Q/K/V
                    projection and RoPE with one upload and one download; default off)
  --schedule static|continuous  Juno --schedule (default static)
  --cache-type-k f16|q8_0  Juno --cache-type-k (default f16)
  --cache-type-v f16|q8_0  Juno --cache-type-v (default f16)
  --kv-page-size N  Juno --kv-page-size (default 16, schedule=continuous only)
  --prefill-batch N Juno --prefill-batch: the prefill window width (default: the engine's own
                    choice; also read from JUNO_PREFILL_BATCH). Recorded in host.json, as are
                    JUNO_SPEC_TYPE, JUNO_SPEC_NGRAM_N, JUNO_SPEC_NGRAM_M and JUNO_MODEL_DRAFT
  --raw-prompt      Repeat a minimal token pattern (~1 tok/word) for prompt-length parity
                    (default: on; a prefill ratio needs it)
  --no-raw-prompt   Use the fixed prompt sentence instead. Juno then prefills far
                    fewer tokens than the reference tool, so prefill ratios from
                    such a run are withheld rather than published
  --api-port N      Juno REST port (default: ${API_PORT})
  --out DIR         Output directory (default: target/perf-compare/<timestamp>)
  --llama-bin DIR   Directory with llama-bench
  --juno-jar PATH   Run this Juno shaded jar instead of this tree's juno-player build, for an
                    A/B against another build from the same checkout (so the reference tool
                    and models resolve as usual). Implies --no-publish: a published sweep is
                    always this tree's build.
  --pin-clocks      Pin the clocks for the run: CPU governor performance, turbo off,
                    GPU graphics clock locked where the driver supports it; all restored
                    on exit. Needs prompt-free sudo (run sudo -v first); the run refuses
                    to start if the CPU cannot be pinned. Required for any gate tighter
                    than this host's 15% noise floor
  --device-spans    Also record juno.DeviceStaging, juno.WeightDequant and
                    juno.DeviceCompute (host-device copy, dequantization and device
                    kernel totals per site and phase; results gain a device_staging
                    object). Costs 4% to 7% of TinyLlama prefill, so a
                    spans run is for attribution and staged bytes, not for a ratio
  --pin-gpu-mhz N   GPU graphics clock to lock at with --pin-clocks (default: the card's
                    maximum graphics clock)
  --vector 0|1     Pass jdk.incubator.vector to Juno (default: ${JUNO_USE_VECTOR})
  --jfr DURATION    Enable Juno JFR for DURATION (default: ${JFR_DURATION}; on by default)
  --no-jfr          Skip Juno --jfr (API latency only for Juno tg)
  --publish         Copy metrics JSON+INDEX into docs/perf-compare/ (default)
  --no-publish      Skip docs/perf-compare publish
  --no-tuned-lane   Skip the extra per-model tuned lane (--mmq auto --gpu-attention auto
                    --gpu-layers auto) that GPU runs add automatically for every selected
                    model (default: on). The vanilla default-flags lane always runs
                    regardless; this only controls the second, additional run. See
                    docs/infra-plan/PLAN-Infra-Review-Fixes.md item 8 and
                    docs/infra-plan/PLAN-Infra-Tier18.md — the default-flags numbers
                    understate what Juno's already-shipped auto modes do, so the regression
                    sweep also measures the tuned config.
  --list            List selected models and exit
  --selftest        Check this script's own aggregation arithmetic and exit
                    (needs only jq; runs no model)
  -n, --dry-run     Print commands only
  -h, --help        This help
EOF
}

log()  { printf '[compare] %s\n' "$*"; }
warn() { printf '[compare] warn: %s\n' "$*" >&2; }
die()  { printf '[compare] error: %s\n' "$*" >&2; exit 1; }

# ── selftest ─────────────────────────────────────────────────────────────────
# The arithmetic this script publishes a verdict from is worth checking without
# spending an hour of model runs to reach it. One bug of exactly this class
# already shipped here: the withheld-ratio reason list read `.prompt_parity.ok //
# true`, and jq treats false as absent for //, so the reason could never print.
# Run with --selftest; it needs nothing but jq.

SELFTEST_FAILURES=0

selftest_expect() {
  local what="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    printf '  ok   %s = %s\n' "$what" "$actual"
  else
    printf '  FAIL %s: expected %s, got %s\n' "$what" "$expected" "$actual"
    SELFTEST_FAILURES=$((SELFTEST_FAILURES + 1))
  fi
}

# Writes a minimal per-rep Juno result carrying only the fields the aggregation
# reads, so a case states its inputs instead of hiding them in a big fixture.
selftest_rep() {
  local path="$1" status="$2" pp="$3" tg="$4" api_tg="$5" latency="$6" gc_max="$7" alloc_per_tok="$8"
  jq -n --arg status "$status" --argjson pp "$pp" --argjson tg "$tg" --argjson api_tg "$api_tg" \
    --argjson latency "$latency" --argjson gc_max "$gc_max" --argjson alloc "$alloc_per_tok" \
    '{ engine: "juno", status: $status, model: "selftest.gguf",
       prompt_eval_tps: $pp, token_gen_tps: $tg, api_token_gen_tps: $api_tg, latency_ms: $latency,
       jfr: { gc_pause_max_ms: $gc_max, allocated_bytes_per_token: $alloc } }' >"$path"
}

run_selftest() {
  require_cmd jq
  local d
  d="$(mktemp -d)"
  trap 'rm -rf "$d"' RETURN
  log "selftest: rep aggregation, spread, noise aggregate, fixed heap"

  # Odd rep count: the median is the middle reading, not the mean, so one wild
  # run cannot drag the published number with it.
  selftest_rep "$d/r1.json" success 10 5.0 4.0 1000 12 100
  selftest_rep "$d/r2.json" success 20 6.0 5.0 2000 300 200
  selftest_rep "$d/r3.json" success 90 7.0 6.0 3000 30 300
  aggregate_juno_reps_json "$d/odd.json" "$d/r1.json" "$d/r2.json" "$d/r3.json"
  selftest_expect "median of 3 pp readings" 20 "$(jq -r '.prompt_eval_tps' "$d/odd.json")"
  selftest_expect "median of 3 tg readings" 6.0 "$(jq -r '.token_gen_tps' "$d/odd.json")"
  selftest_expect "min recorded next to the median" 10 "$(jq -r '.reps.prompt_eval_tps.min' "$d/odd.json")"
  selftest_expect "max recorded next to the median" 90 "$(jq -r '.reps.prompt_eval_tps.max' "$d/odd.json")"
  selftest_expect "every rep reading kept" "10,20,90" "$(jq -r '.reps.prompt_eval_tps.values | join(",")' "$d/odd.json")"
  selftest_expect "rep count published" 3 "$(jq -r '.juno_reps' "$d/odd.json")"
  selftest_expect "all-success aggregate status" success "$(jq -r '.status' "$d/odd.json")"
  # A noise indicator is aggregated by its worst reading, not its median: the
  # re-run rule fires on any single pause over its threshold, and a median would
  # hide exactly the outlier the rule exists to catch.
  selftest_expect "worst GC pause across reps, not the median" 300 "$(jq -r '.jfr.gc_pause_max_ms' "$d/odd.json")"
  selftest_expect "GC pause median still recorded" 30 "$(jq -r '.reps.gc_pause_max_ms.median' "$d/odd.json")"

  # Even rep count: the two middle readings average, matching the median the LoRA
  # comparison already publishes.
  selftest_rep "$d/e1.json" success 10 5.0 4.0 1000 0 100
  selftest_rep "$d/e2.json" success 20 8.0 6.0 2000 0 200
  aggregate_juno_reps_json "$d/even.json" "$d/e1.json" "$d/e2.json"
  selftest_expect "median of 2 pp readings" 15 "$(jq -r '.prompt_eval_tps' "$d/even.json")"
  selftest_expect "median of 2 tg readings" 6.5 "$(jq -r '.token_gen_tps' "$d/even.json")"

  # A rep that produced no reading must not count as a zero: a withheld prefill
  # ratio is null, and averaging null in as 0 would publish a number nobody measured.
  selftest_rep "$d/n1.json" success null 5.0 4.0 1000 0 100
  selftest_rep "$d/n2.json" success 20 7.0 6.0 2000 0 200
  selftest_rep "$d/n3.json" success 30 9.0 8.0 3000 0 300
  aggregate_juno_reps_json "$d/nulls.json" "$d/n1.json" "$d/n2.json" "$d/n3.json"
  selftest_expect "null readings dropped, not read as zero" 25 "$(jq -r '.prompt_eval_tps' "$d/nulls.json")"
  selftest_expect "dropped reading still visible in the spread" "null,20,30" \
    "$(jq -r '.reps.prompt_eval_tps.values | map(tostring) | join(",")' "$d/nulls.json")"

  # No reading at all stays null rather than becoming 0.
  selftest_rep "$d/z1.json" failure null null null 0 0 0
  selftest_rep "$d/z2.json" failure null null null 0 0 0
  aggregate_juno_reps_json "$d/allnull.json" "$d/z1.json" "$d/z2.json"
  selftest_expect "no reading anywhere stays null" null "$(jq -r '.prompt_eval_tps' "$d/allnull.json")"
  selftest_expect "a failed rep fails the aggregate" failure "$(jq -r '.status' "$d/allnull.json")"

  # One failed rep among successes is still a failure: a median over a partly
  # broken set is not a measurement.
  selftest_rep "$d/m1.json" success 10 5.0 4.0 1000 0 100
  selftest_rep "$d/m2.json" failure null null null 0 0 0
  aggregate_juno_reps_json "$d/mixed.json" "$d/m1.json" "$d/m2.json"
  selftest_expect "one failed rep fails the aggregate" failure "$(jq -r '.status' "$d/mixed.json")"
  selftest_expect "per-rep status list published" "success,failure" \
    "$(jq -r '.rep_status | join(",")' "$d/mixed.json")"

  # A generation ratio needs tokens to have been generated. A model that emits a
  # stop token immediately produces no reading at all, and publishing that as 0
  # says "infinitely slower than the reference" when it means "never measured".
  jq -n '{ status: "success", prompt_eval_tps: 1000, token_gen_tps: 64, raw_json: "ref.raw.json" }' >"$d/ref.json"
  jq -n '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 50, token_gen_tps: 0,
           n_gen: 64, completion_tokens: 0, prompt_tokens: 128, n_prompt: 128, finish_reason: "stop",
           prompt_token_deviation: 0, prompt_parity_tolerance: 0.10, latency_ms: 100,
           response_json: "resp.json", use_jfr: 0 }' >"$d/zero-juno.json"
  cp "$d/ref.json" "$d/zerogen-llama-cpp.json"; cp "$d/zero-juno.json" "$d/zerogen-juno.json"
  OUT_ROOT="$d" write_pair_summary zerogen
  selftest_expect "no generated tokens withholds the tg ratio" null \
    "$(jq -r '.ratio_juno_over_llamacpp_tg' "$d/zerogen-compare.json")"
  selftest_expect "and states why" false \
    "$(jq -r '.generation_parity.ok' "$d/zerogen-compare.json")"
  selftest_expect "while the prefill ratio is unaffected" 0.05 \
    "$(jq -r '.ratio_juno_over_llamacpp_pp' "$d/zerogen-compare.json")"

  # Generating fewer tokens than the reference is reported, not withheld: decode
  # cost per token is roughly steady, so the reading is comparable, but it is taken
  # at slightly shorter context and a reader has to be able to see that.
  jq -n '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 50, token_gen_tps: 32,
           n_gen: 64, completion_tokens: 22, prompt_tokens: 128, n_prompt: 128, finish_reason: "stop",
           prompt_token_deviation: 0, prompt_parity_tolerance: 0.10, latency_ms: 100,
           response_json: "resp.json", use_jfr: 0 }' >"$d/short-juno.json"
  cp "$d/ref.json" "$d/shortgen-llama-cpp.json"; cp "$d/short-juno.json" "$d/shortgen-juno.json"
  OUT_ROOT="$d" write_pair_summary shortgen
  selftest_expect "a short generation still publishes a tg ratio" 0.5 \
    "$(jq -r '.ratio_juno_over_llamacpp_tg' "$d/shortgen-compare.json")"
  selftest_expect "and records the token shortfall" 22 \
    "$(jq -r '.generation_parity.juno_completion_tokens' "$d/shortgen-compare.json")"

  # The two Juno lanes mirror the reference tool two benchmarks: one measures
  # prefill at the requested prompt length, the other measures generation from a
  # minimal prompt. Merging takes each figure from the lane that measured it.
  jq -n '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 50, token_gen_tps: 999,
           n_gen: 1, completion_tokens: 1, prompt_tokens: 128, n_prompt: 128, finish_reason: "length",
           prompt_token_deviation: 0, prompt_parity_tolerance: 0.10, latency_ms: 2000,
           response_json: "p.json", use_jfr: 1, juno_reps: 3,
           reps: { prompt_eval_tps: { median: 50, min: 48, max: 52, values: [48,50,52] },
                   token_gen_tps: { median: 999, min: 999, max: 999, values: [999] } },
           jfr: { gc_pause_max_ms: 12, allocated_bytes_per_token: 100 } }' >"$d/lane-prefill.json"
  jq -n '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 777, token_gen_tps: 32,
           n_gen: 64, completion_tokens: 64, prompt_tokens: 21, n_prompt: 0, finish_reason: "length",
           prompt_token_deviation: null, prompt_parity_tolerance: 0.10, latency_ms: 4000,
           response_json: "g.json", use_jfr: 1, juno_reps: 3,
           reps: { prompt_eval_tps: { median: 777, min: 777, max: 777, values: [777] },
                   token_gen_tps: { median: 32, min: 31, max: 33, values: [31,32,33] } },
           jfr: { gc_pause_max_ms: 300, allocated_bytes_per_token: 200 } }' >"$d/lane-generate.json"
  merge_juno_lanes_json "$d/merged.json" "$d/lane-prefill.json" "$d/lane-generate.json"
  selftest_expect "prefill figure comes from the prefill lane" 50 \
    "$(jq -r '.prompt_eval_tps' "$d/merged.json")"
  selftest_expect "generation figure comes from the generation lane" 32 \
    "$(jq -r '.token_gen_tps' "$d/merged.json")"
  selftest_expect "prompt parity is judged on the prefill lane" 128 \
    "$(jq -r '.prompt_tokens' "$d/merged.json")"
  selftest_expect "generation parity is judged on the generation lane" 64 \
    "$(jq -r '.completion_tokens' "$d/merged.json")"
  selftest_expect "and against what that lane requested" 64 "$(jq -r '.n_gen' "$d/merged.json")"
  selftest_expect "the prefill lane spread survives" "48,50,52" \
    "$(jq -r '.reps.prompt_eval_tps.values | join(",")' "$d/merged.json")"
  selftest_expect "the generation lane spread survives" "31,32,33" \
    "$(jq -r '.reps.token_gen_tps.values | join(",")' "$d/merged.json")"
  # A pause in either lane disqualifies the row, so the worst of the two is kept.
  selftest_expect "the worse collection pause of the two lanes is kept" 300 \
    "$(jq -r '.jfr.gc_pause_max_ms' "$d/merged.json")"
  selftest_expect "the generation lane prompt length is visible" 21 \
    "$(jq -r '.lanes.generate.prompt_tokens' "$d/merged.json")"
  selftest_expect "both lanes must succeed" success "$(jq -r '.status' "$d/merged.json")"

  jq -n '{ status: "failure", model: "selftest.gguf" }' >"$d/lane-broken.json"
  merge_juno_lanes_json "$d/merged-bad.json" "$d/lane-prefill.json" "$d/lane-broken.json"
  selftest_expect "a failed lane fails the row" failure "$(jq -r '.status' "$d/merged-bad.json")"

  # What decides whether a row is scorable is whether its own repetitions agree.
  # That is the direct evidence, and it is measured rather than inferred: a reading
  # whose three cycles land within a few percent of each other was not disturbed,
  # whatever any counter says.
  jq -n '{ status: "success", prompt_eval_tps: 1000, token_gen_tps: 64, raw_json: "r.json" }' >"$d/disp-llama-cpp.json"
  selftest_rep_set() {
    jq -n --argjson med "$2" --argjson lo "$3" --argjson hi "$4" --argjson gc "$5" \
      '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 50, token_gen_tps: $med,
         n_gen: 64, completion_tokens: 64, prompt_tokens: 128, n_prompt: 128, finish_reason: "length",
         prompt_token_deviation: 0, prompt_parity_tolerance: 0.10, latency_ms: 1134,
         response_json: "resp.json", use_jfr: 1, juno_reps: 3,
         reps: { token_gen_tps: { median: $med, min: $lo, max: $hi, values: [$lo,$med,$hi] } },
         jfr: { gc_pause_max_ms: $gc, allocated_bytes_per_token: 100 } }' >"$1"
  }

  # The real case: three cycles within 0.7%, one of them reporting a 634 ms pause.
  selftest_rep_set "$d/agree-juno.json" 56.82 56.42 56.82 634
  cp "$d/disp-llama-cpp.json" "$d/agree-llama-cpp.json"
  OUT_ROOT="$d" write_pair_summary agree
  selftest_expect "repetitions that agree are scorable despite a reported pause" true \
    "$(jq -r '.noise.ok' "$d/agree-compare.json")"
  selftest_expect "and the pause is still reported" 634 \
    "$(jq -r '.noise.gc_pause_max_ms' "$d/agree-compare.json")"

  # The real counter-case: a clean pause counter over readings spanning 31%.
  selftest_rep_set "$d/scatter-juno.json" 25.79 21.00 29.12 7
  cp "$d/disp-llama-cpp.json" "$d/scatter-llama-cpp.json"
  OUT_ROOT="$d" write_pair_summary scatter
  selftest_expect "readings that scatter are not scorable" false \
    "$(jq -r '.noise.ok' "$d/scatter-compare.json")"
  selftest_expect "and the condition names dispersion" rep_spread \
    "$(jq -r '.noise.condition' "$d/scatter-compare.json")"
  selftest_expect "with the spread as a fraction of the median" 0.31 \
    "$(jq -r '.noise.rep_spread | . * 100 | round / 100' "$d/scatter-compare.json")"

  # A single reading offers no dispersion evidence either way, and says so rather
  # than passing itself as verified.
  jq -n '{ status: "success", model: "selftest.gguf", prompt_eval_tps: 50, token_gen_tps: 32,
           n_gen: 64, completion_tokens: 64, prompt_tokens: 128, n_prompt: 128, finish_reason: "length",
           prompt_token_deviation: 0, prompt_parity_tolerance: 0.10, latency_ms: 1000,
           response_json: "resp.json", use_jfr: 1, juno_reps: 1,
           jfr: { gc_pause_max_ms: 4, allocated_bytes_per_token: 100 } }' >"$d/single-juno.json"
  cp "$d/disp-llama-cpp.json" "$d/single-llama-cpp.json"
  OUT_ROOT="$d" write_pair_summary single
  selftest_expect "a single reading is not judged by dispersion" "not assessable" \
    "$(jq -r '.noise.basis' "$d/single-compare.json")"

  # Collection pauses are reported, not gated on, and the share of them that landed
  # inside the measured token span is reported beside the whole-recording figure so
  # a reader can tell the two apart.
  selftest_rep_set "$d/paused-juno.json" 56.82 56.42 56.82 634
  jq '.jfr.gc_pause_max_ms_in_token_span = 634' "$d/paused-juno.json" >"$d/inspan-juno.json"
  cp "$d/disp-llama-cpp.json" "$d/inspan-llama-cpp.json"
  OUT_ROOT="$d" write_pair_summary inspan
  selftest_expect "the in-span pause figure is carried through" 634 \
    "$(jq -r '.noise.gc_pause_max_ms_in_token_span' "$d/inspan-compare.json")"
  selftest_expect "a pause alone does not disqualify a stable reading" true \
    "$(jq -r '.noise.ok' "$d/inspan-compare.json")"

  # A first rep that died before writing a full record must not strip the aggregate.
  jq -n '{ engine: "juno", status: "failure", model: "selftest.gguf" }' >"$d/stub.json"
  selftest_rep "$d/good.json" success 40 9.0 8.0 4000 0 400
  aggregate_juno_reps_json "$d/stubfirst.json" "$d/stub.json" "$d/good.json"
  selftest_expect "aggregate built on the rep that succeeded" 40 \
    "$(jq -r '.prompt_eval_tps' "$d/stubfirst.json")"
  selftest_expect "a stub rep still fails the aggregate" failure \
    "$(jq -r '.status' "$d/stubfirst.json")"

  # The heap is fixed per model rather than derived from the file size, so the
  # collector is doing the same amount of work in a run as in the baseline it is
  # compared against, whatever the derivation would say today.
  selftest_expect "fixed heap for a sweep model" 6g \
    "$(COMPARE_HEAP= heap_for_model "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf")"
  selftest_expect "sweep model heap marked fixed" fixed \
    "$(COMPARE_HEAP= heap_source_for_model "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf")"
  selftest_expect "explicit COMPARE_HEAP still wins" 11g \
    "$(COMPARE_HEAP=11g heap_for_model "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf")"
  selftest_expect "explicit heap marked explicit" explicit \
    "$(COMPARE_HEAP=11g heap_source_for_model "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf")"
  # An off-table model still has to run, so it keeps the size derivation — but it
  # is labelled derived, because a derived heap is not comparable with a baseline
  # taken at a fixed one.
  selftest_expect "off-table model heap marked derived" derived \
    "$(COMPARE_HEAP= heap_source_for_model "${MODELS_DIR}/Qwen3-1.7B-Q4_K_M.gguf")"

  # Every measured repetition is checked against its own request. The cases are the
  # shapes of the 2026-09-27 sweeps (20260927T091155Z, 20260927T232837Z): JFR spans
  # read 633 ms short or long when they crossed CPU0, whose timestamp counter runs
  # ahead of the other cores on that host, while the request latency (the engine
  # clock) stayed right. A reading taken off such spans is withheld.
  log "selftest: span check against the request, full prefill"
  selftest_metrics() {  # path prefill_ms decode_ms [prefill_tokens min_start]
    jq -n --argjson p "$2" --argjson dec "$3" --argjson t "${4:-null}" --argjson st "${5:-null}" \
      '{ models: [ { metrics: ({ "juno.ForwardPass.prefill.total_ms": $p,
                                  "juno.ForwardPass.decode.total_ms": $dec }
                               + (if $t == null then {} else { "juno.PrefillBatch.tokens": $t,
                                  "juno.PrefillBatch.min_start_position": $st } end)) } ] }' >"$1"
  }
  local sj
  selftest_metrics "$d/short.json" 142 25 127 0
  sj="$(jfr_summary_json "$d/short.json" 128 1 808)"
  selftest_expect "a prefill span 633 ms short of its request is caught" false \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_expect "its 898 t/s prefill reading is withheld" null "$(jq -r '.prompt_eval_tps' <<<"$sj")"
  selftest_metrics "$d/long.json" 1451 23 127 0
  sj="$(jfr_summary_json "$d/long.json" 128 1 850)"
  selftest_expect "a prefill span longer than its whole request is caught" false \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_metrics "$d/ok.json" 750 23 127 0
  sj="$(jfr_summary_json "$d/ok.json" 128 1 782)"
  selftest_expect "a span that accounts for its request passes" true \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_expect "and its prefill reading is published" 170.67 \
    "$(jq -r '.prompt_eval_tps * 100 | round / 100' <<<"$sj")"
  selftest_expect "a whole-prompt prefill from position 0 is a full prefill" true \
    "$(jq -r '.span_check.full_prefill' <<<"$sj")"
  selftest_metrics "$d/gen.json" 266 2003 10 0
  sj="$(jfr_summary_json "$d/gen.json" 11 64 2411)"
  selftest_expect "64 tokens of sampling overhead (142 ms) is not flagged" true \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_metrics "$d/gen-long.json" 334 3796 10 0
  sj="$(jfr_summary_json "$d/gen-long.json" 11 64 2948)"
  selftest_expect "decode spans 1184 ms over their request are caught" false \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_expect "and the generation reading is withheld" null "$(jq -r '.token_gen_tps' <<<"$sj")"
  # CPU shape (2026-09-28, operating-system clock): 8 tokens, 314 ms per decode step,
  # so the first step alone leaves 484 ms outside the token span. Not a misread.
  jq -n '{ models: [ { metrics: { "juno.ForwardPass.prefill.total_ms": 3293,
      "juno.ForwardPass.decode.total_ms": 2510, "juno.ForwardPass.decode.count": 8,
      "juno.TokenProduced.elapsed_seconds": 2.075 } } ] }' >"$d/cpu-gen.json"
  sj="$(jfr_summary_json "$d/cpu-gen.json" 11 8 5853)"
  selftest_expect "a slow first CPU decode step is not read as a misread span" true \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  jq '.models[0].metrics["juno.TokenProduced.elapsed_seconds"] = 1.414' "$d/gen.json" >"$d/gen-tok.json"
  sj="$(jfr_summary_json "$d/gen-tok.json" 11 64 2361)"
  selftest_expect "a token span 651 ms short, with sound forward passes, is caught" false \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  jq '.models[0].metrics["juno.TokenProduced.elapsed_seconds"] = 2.065' "$d/gen.json" >"$d/gen-tok-ok.json"
  sj="$(jfr_summary_json "$d/gen-tok-ok.json" 11 64 2361)"
  selftest_expect "a token span that leaves a first-token gap passes" true \
    "$(jq -r '.span_check.timestamps_consistent' <<<"$sj")"
  selftest_metrics "$d/reuse.json" 200 23 27 100
  sj="$(jfr_summary_json "$d/reuse.json" 128 1 240)"
  selftest_expect "a prefill resumed at position 100 is not a full prefill" false \
    "$(jq -r '.span_check.full_prefill' <<<"$sj")"
  selftest_expect "and its prefill reading is withheld" null "$(jq -r '.prompt_eval_tps' <<<"$sj")"
  selftest_expect "and the reason names the start position" true \
    "$(jq -r '.span_check.reason | test("starting at position 100")' <<<"$sj")"
  selftest_metrics "$d/oldjar.json" 750 23
  sj="$(jfr_summary_json "$d/oldjar.json" 128 1 782)"
  selftest_expect "a build without prefill-token keys is not judged on them" null \
    "$(jq -r '.span_check.full_prefill' <<<"$sj")"
  selftest_expect "and still publishes its reading" 170.67 \
    "$(jq -r '.prompt_eval_tps * 100 | round / 100' <<<"$sj")"
  selftest_expect "a build without staging keys reports no staging figures" null \
    "$(jq -r '.device_staging' <<<"$sj")"
  jq '.models[0].metrics += { "juno.DeviceStaging.H2D.prefill.bytes": 1000000,
      "juno.DeviceStaging.D2H.prefill.bytes": 3000000, "juno.DeviceStaging.H2D.prefill.count": 7,
      "juno.DeviceStaging.D2H.prefill.count": 7 }' "$d/ok.json" >"$d/staging.json"
  sj="$(DEVICE_SPANS=1 jfr_summary_json "$d/staging.json" 128 1 782)"
  selftest_expect "a run without --device-spans reports no staging figures" null \
    "$(jq -r '.device_staging' <<<"$(DEVICE_SPANS=0 jfr_summary_json "$d/staging.json" 128 1 782)")"
  selftest_expect "prefill staging is surfaced per prompt token" 31496.06 \
    "$(jq -r '.device_staging.prefill_bytes_per_token * 100 | round / 100' <<<"$sj")"
  selftest_expect "a build without compute keys reports no compute figure" null \
    "$(jq -r '.device_staging.prefill_compute_ms' <<<"$sj")"
  jq '.models[0].metrics += { "juno.DeviceCompute.prefill.total_ms": 410.5,
      "juno.DeviceCompute.site.gemm_half.prefill.total_ms": 370.5,
      "juno.DeviceCompute.site.gqa_attention.prefill.total_ms": 40,
      "juno.DeviceCompute.site.gemm_half.decode.total_ms": 0,
      "juno.DeviceStaging.HOST.prefill.total_ms": 55.25 }' "$d/staging.json" >"$d/compute.json"
  sj="$(DEVICE_SPANS=1 jfr_summary_json "$d/compute.json" 128 1 782)"
  selftest_expect "prefill kernel time is surfaced" 410.5 "$(jq -r '.device_staging.prefill_compute_ms' <<<"$sj")"
  selftest_expect "and split per kernel site, prefill only" '{"gemm_half":370.5,"gqa_attention":40}' \
    "$(jq -c '.device_staging.prefill_compute_ms_by_site' <<<"$sj")"
  selftest_expect "host FP16 packing is surfaced apart from the copies" 55.25 \
    "$(jq -r '.device_staging.prefill_pack_host_ms' <<<"$sj")"

  log "selftest: Juno's hot path runs the reference tool's -t"
  local saved_threads="$N_THREADS" saved_pool="$JUNO_POOL_PARALLELISM" saved_juno_threads="$JUNO_THREADS"
  local saved_flags=("${JUNO_JVM_FLAGS[@]}")
  N_THREADS=4
  JUNO_JVM_FLAGS=(-XX:+UseG1GC)
  resolve_thread_parity
  selftest_expect "the common pool is given -t minus the calling thread" \
    "-Djava.util.concurrent.ForkJoinPool.common.parallelism=3" "${JUNO_JVM_FLAGS[-1]}"
  local hj
  hj="$(host_meta_json)"
  selftest_expect "host.json records juno_threads" 4 "$(jq -r '.juno_threads' <<<"$hj")"
  selftest_expect "host.json records the pool property" 3 "$(jq -r '.juno_common_pool_parallelism' <<<"$hj")"
  selftest_expect "and the JVM flags it ran with carry it" true \
    "$(jq -r '.juno_jvm_flags | test("common.parallelism=3")' <<<"$hj")"

  # Settings set only through the environment reach the engine's command line;
  # without these fields a run's window width or speculation mode would be
  # recorded nowhere but the directory name it was published under.
  local saved_pb="${JUNO_PREFILL_BATCH:-}" saved_st="${JUNO_SPEC_TYPE:-}" saved_sn="${JUNO_SPEC_NGRAM_N:-}"
  local saved_sm="${JUNO_SPEC_NGRAM_M:-}" saved_md="${JUNO_MODEL_DRAFT:-}"
  JUNO_PREFILL_BATCH=9 JUNO_SPEC_TYPE=ngram JUNO_SPEC_NGRAM_N=3 JUNO_SPEC_NGRAM_M=4 JUNO_MODEL_DRAFT=draft.gguf
  hj="$(host_meta_json)"
  selftest_expect "host.json records the prefill window width" 9 "$(jq -r '.juno_prefill_batch' <<<"$hj")"
  selftest_expect "host.json records the speculation settings" "ngram,3,4,draft.gguf" \
    "$(jq -r '[.juno_spec_type, .juno_spec_ngram_n, .juno_spec_ngram_m, .juno_model_draft] | join(",")' <<<"$hj")"
  JUNO_PREFILL_BATCH="" JUNO_SPEC_TYPE="" JUNO_SPEC_NGRAM_N="" JUNO_SPEC_NGRAM_M="" JUNO_MODEL_DRAFT=""
  hj="$(host_meta_json)"
  selftest_expect "an unset width is recorded as the engine default, not omitted" '""' \
    "$(jq -c '.juno_prefill_batch' <<<"$hj")"
  JUNO_PREFILL_BATCH="$saved_pb" JUNO_SPEC_TYPE="$saved_st" JUNO_SPEC_NGRAM_N="$saved_sn"
  JUNO_SPEC_NGRAM_M="$saved_sm" JUNO_MODEL_DRAFT="$saved_md"
  N_THREADS=1
  JUNO_JVM_FLAGS=()
  resolve_thread_parity
  selftest_expect "-t 1 still leaves the pool one worker" 1 "$JUNO_POOL_PARALLELISM"
  N_THREADS="$saved_threads"
  JUNO_POOL_PARALLELISM="$saved_pool"
  JUNO_THREADS="$saved_juno_threads"
  JUNO_JVM_FLAGS=("${saved_flags[@]}")

  log "selftest: GPU attention as the engine resolved it, not as the flag asked"
  # Read off the engine log: the published default lane passed no flag at all, and
  # handlers without the kernel resolve to scalar attention without saying so.
  printf 'INFO: GPU-resident attention path active (gpu-attention=auto)\n' >"$d/attn-on.log"
  printf 'WARNING: --gpu-attention requested on ROCm backend — not yet implemented there, falling back to scalar CPU attention\n' >"$d/attn-fallback.log"
  printf 'INFO: Phi3TransformerHandler loaded\n' >"$d/attn-silent.log"
  # The console turns library logging off unless --verbose, so a silent log is the
  # normal case and proves nothing. With --device-spans the attention kernel leaves
  # its own copy sites (memcpy_gqa_*) in the recording, which is positive evidence
  # either way; without spans and without a log line the answer is unknown.
  jq -n '{ models: [ { metrics: { "juno.DeviceStaging.site.memcpy_gqa_qbatch_h2d.prefill.count": 22,
                                  "juno.DeviceStaging.site.memcpy_k_row_h2d.prefill.count": 11242 } } ] }' >"$d/attn-gqa.json"
  jq -n '{ models: [ { metrics: { "juno.DeviceStaging.site.cudamemcpyasync_xh_h2d_q4k_batched_gemm.prefill.count": 128 } } ] }' >"$d/attn-nogqa.json"
  jq -n '{ models: [ { metrics: { "juno.DeviceStaging.H2D.count": 0 } } ] }' >"$d/attn-nospans.json"
  # The prefill-window device region runs attention inside itself: no memcpy_gqa_* copies,
  # only its own compute site.
  jq -n '{ models: [ { metrics: { "juno.DeviceCompute.site.gqa_attention_region.prefill.count": 22,
                                  "juno.DeviceStaging.site.upload_resident_activation.prefill.count": 22 } } ] }' \
    >"$d/attn-region.json"
  selftest_expect "an activation line reads on" on "$(gpu_attention_resolved_from_log "$d/attn-on.log" 1)"
  selftest_expect "a backend fallback reads off, naming it" off-backend-fallback \
    "$(gpu_attention_resolved_from_log "$d/attn-fallback.log" 1)"
  selftest_expect "a silent log with no recording is unknown, not off" unknown \
    "$(gpu_attention_resolved_from_log "$d/attn-silent.log" 1)"
  selftest_expect "a silent log without spans is unknown, not off" unknown \
    "$(DEVICE_SPANS=0 gpu_attention_resolved_from_log "$d/attn-silent.log" 1 "$d/attn-nospans.json")"
  selftest_expect "attention-kernel copy sites in a spans run read on" on \
    "$(DEVICE_SPANS=1 gpu_attention_resolved_from_log "$d/attn-silent.log" 1 "$d/attn-gqa.json")"
  selftest_expect "attention inside the prefill-window device region reads on" on \
    "$(DEVICE_SPANS=1 gpu_attention_resolved_from_log "$d/attn-silent.log" 1 "$d/attn-region.json")"
  selftest_expect "a spans run without attention-kernel copies reads off" off \
    "$(DEVICE_SPANS=1 gpu_attention_resolved_from_log "$d/attn-silent.log" 1 "$d/attn-nogqa.json")"
  selftest_expect "a CPU run has no GPU attention to resolve" n/a "$(gpu_attention_resolved_from_log "$d/attn-on.log" 0)"
  selftest_rep "$d/a1.json" success 10 5.0 4.0 1000 0 100
  selftest_rep "$d/a2.json" success 20 6.0 5.0 2000 0 200
  jq '.gpu_attention_resolved = "on"' "$d/a1.json" >"$d/a1r.json"
  jq '.gpu_attention_resolved = "on"' "$d/a2.json" >"$d/a2r.json"
  jq '.gpu_attention_resolved = "off"' "$d/a2.json" >"$d/a2o.json"
  aggregate_juno_reps_json "$d/attn-agree.json" "$d/a1r.json" "$d/a2r.json"
  aggregate_juno_reps_json "$d/attn-mixed.json" "$d/a1r.json" "$d/a2o.json"
  selftest_expect "reps that agree carry the resolved value" on "$(jq -r '.gpu_attention_resolved' "$d/attn-agree.json")"
  selftest_expect "reps that disagree read mixed" mixed "$(jq -r '.gpu_attention_resolved' "$d/attn-mixed.json")"
  merge_juno_lanes_json "$d/attn-lanes.json" "$d/attn-agree.json" "$d/attn-mixed.json"
  selftest_expect "the prefill lane keeps its own resolved value" on \
    "$(jq -r '.lanes.prefill.gpu_attention_resolved' "$d/attn-lanes.json")"
  selftest_expect "lanes that disagree read mixed at the top" mixed \
    "$(jq -r '.gpu_attention_resolved' "$d/attn-lanes.json")"

  if (( SELFTEST_FAILURES > 0 )); then
    die "selftest: ${SELFTEST_FAILURES} check(s) failed"
  fi
  log "selftest: all checks passed"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

json_escape() {
  # minimal JSON string escape
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --all-gguf) MODEL_FILTER="__ALL__"; shift ;;
    --n-prompt) N_PROMPT="$2"; shift 2 ;;
    --n-gen) N_GEN="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --juno-warmup) JUNO_WARMUP="$2"; shift 2 ;;
    --juno-reps) JUNO_REPS="$2"; shift 2 ;;
    --juno-min-tokens) JUNO_MIN_TOKENS="$2"; shift 2 ;;
    --selftest) SELFTEST_ONLY=1; shift ;;
    --threads) N_THREADS="$2"; shift 2 ;;
    --cpu) USE_GPU=0; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --ngl) NGL="$2"; shift 2 ;;
    --gpu-layers) JUNO_GPU_LAYERS="$2"; shift 2 ;;
    --mmq) JUNO_MMQ="$2"; shift 2 ;;
    --gpu-attention) JUNO_GPU_ATTENTION="$2"; shift 2 ;;
    --gpu-residency) JUNO_GPU_RESIDENCY="$2"; shift 2 ;;
    --schedule) JUNO_SCHEDULE="$2"; shift 2 ;;
    --cache-type-k) JUNO_CACHE_TYPE_K="$2"; shift 2 ;;
    --cache-type-v) JUNO_CACHE_TYPE_V="$2"; shift 2 ;;
    --kv-page-size) JUNO_KV_PAGE_SIZE="$2"; shift 2 ;;
    --prefill-batch) JUNO_PREFILL_BATCH="$2"; shift 2 ;;
    --raw-prompt) RAW_PROMPT=1; shift ;;
    --no-raw-prompt) RAW_PROMPT=0; shift ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --llama-bin) LLAMA_CPP_BIN_EXPLICIT="$2"; shift 2 ;;
    --juno-jar) JUNO_JAR_EXPLICIT="$2"; shift 2 ;;
    --pin-clocks) PIN_CLOCKS=1; shift ;;
    --device-spans) DEVICE_SPANS=1; shift ;;
    --pin-gpu-mhz) PIN_GPU_MHZ="$2"; shift 2 ;;
    --vector) JUNO_USE_VECTOR="$2"; shift 2 ;;
    --jfr) USE_JFR=1; JFR_DURATION="$2"; shift 2 ;;
    --no-jfr) USE_JFR=0; shift ;;
    --publish) PUBLISH=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    --no-tuned-lane) TUNED_LANE=0; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

if [[ -n "$JUNO_JAR_EXPLICIT" ]]; then
  [[ -f "$JUNO_JAR_EXPLICIT" ]] || die "--juno-jar: no such file: $JUNO_JAR_EXPLICIT"
  JUNO_JAR_EXPLICIT="$(cd "$(dirname "$JUNO_JAR_EXPLICIT")" && pwd)/$(basename "$JUNO_JAR_EXPLICIT")"
  if [[ "$PUBLISH" -eq 1 ]]; then
    PUBLISH=0
    printf '[compare] --juno-jar given: not publishing (a published sweep is always this tree'"'"'s build)\n' >&2
  fi
fi

# Thread parity with the reference tool's -t (see JUNO_POOL_PARALLELISM above).
resolve_thread_parity() {
  JUNO_POOL_PARALLELISM=$(( N_THREADS - 1 ))
  (( JUNO_POOL_PARALLELISM < 1 )) && JUNO_POOL_PARALLELISM=1
  JUNO_THREADS=$(( JUNO_POOL_PARALLELISM + 1 ))
  JUNO_JVM_FLAGS+=("-Djava.util.concurrent.ForkJoinPool.common.parallelism=${JUNO_POOL_PARALLELISM}")
}
resolve_thread_parity

# Generation parity defaults to the requested token count; resolved here because it
# follows --n-gen, which may itself have been set on the command line.
if [[ -z "$JUNO_MIN_TOKENS" ]]; then
  JUNO_MIN_TOKENS="$N_GEN"
fi

# Resolve backend defaults after flag parse.
if [[ -z "$NGL" ]]; then
  if [[ "$USE_GPU" -eq 1 ]]; then
    NGL=99
  else
    NGL=0
  fi
else
  # Explicit --ngl implies GPU intent when > 0.
  if [[ "$NGL" != "0" ]]; then
    USE_GPU=1
  fi
fi

resolve_llama_bin() {
  if [[ -n "$LLAMA_CPP_BIN_EXPLICIT" ]]; then
    LLAMA_CPP_BIN="$LLAMA_CPP_BIN_EXPLICIT"
    return
  fi
  if [[ -n "${LLAMA_CPP_BIN:-}" ]]; then
    return
  fi
  local cuda_build="${ROOT}/../llama.cpp/build-cuda/bin"
  local stock="${ROOT}/../llama.cpp-bin/llama-b9551"
  if [[ "$USE_GPU" -eq 1 && -x "${cuda_build}/llama-bench" ]]; then
    LLAMA_CPP_BIN="$(cd "$cuda_build" && pwd)"
  elif [[ -x "${stock}/llama-bench" ]]; then
    LLAMA_CPP_BIN="$(cd "$stock" && pwd)"
  else
    LLAMA_CPP_BIN=""
  fi
}

setup_cuda_env() {
  [[ "$USE_GPU" -eq 1 ]] || return 0
  local cuda_root=""
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
    cuda_root="${CUDA_HOME}"
  elif [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]]; then
    cuda_root="${CUDA_PATH}"
  fi
  if [[ -n "$cuda_root" ]]; then
    export PATH="${cuda_root}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  # Distro CUDA packages often land in /usr/lib/x86_64-linux-gnu.
  if [[ -e /usr/lib/x86_64-linux-gnu/libcudart.so.12 ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export JUNO_USE_GPU=true
}

backend_label() {
  if [[ "$USE_GPU" -eq 1 ]]; then
    printf 'gpu'
  else
    printf 'cpu'
  fi
}

select_models() {
  local -a out=()
  local f base
  if [[ "$MODEL_FILTER" == "__ALL__" ]]; then
    for f in "${MODELS_DIR}"/*.gguf; do
      [[ -f "$f" ]] || continue
      base="$(basename "$f")"
      case "$base" in
        *mmproj*|moondream*|llama-1-30b*) continue ;;
      esac
      out+=("$base")
    done
  elif [[ -n "$MODEL_FILTER" ]]; then
    local IFS=',' part
    for part in $MODEL_FILTER; do
      part="$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      [[ -n "$part" ]] || continue
      if [[ -f "${MODELS_DIR}/${part}" ]]; then
        out+=("$part")
        continue
      fi
      local matched=0
      for f in "${MODELS_DIR}"/*.gguf; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        if [[ "$base" == *"$part"* ]]; then
          out+=("$base")
          matched=1
        fi
      done
      [[ "$matched" -eq 1 ]] || die "no model matching: ${part}"
    done
  else
    for base in "${DEFAULT_MODELS[@]}"; do
      [[ -f "${MODELS_DIR}/${base}" ]] || { warn "skip missing default model: ${base}"; continue; }
      out+=("$base")
    done
  fi
  # unique preserve order
  local -A seen=()
  SELECTED_MODELS=()
  for base in "${out[@]}"; do
    [[ -n "${seen[$base]:-}" ]] && continue
    seen[$base]=1
    SELECTED_MODELS+=("$base")
  done
  [[ ${#SELECTED_MODELS[@]} -gt 0 ]] || die "no models selected"
}

# A fixed heap per sweep model, so the collector does the same amount of work in
# a run as in the baseline that run is compared against. A heap derived from the
# file size moves with the model — and would move again if the derivation were
# ever tuned — which shows up as a GC difference nobody intended to measure. The
# values below are the ones the derivation produced when it was replaced, so the
# fixed table starts from what the published runs were already taken at.
declare -A FIXED_HEAP=(
  [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]=4g
  [tinyllama-1.1b-chat-v1.0.Q2_K.gguf]=4g
  [qwen2.5-3b-instruct-q4_k_m.gguf]=5g
  [Phi-3.5-mini-instruct-Q4_K_M.gguf]=6g
  [mistral-7b-instruct-v0.1-q4_k_m.gguf]=9g
  [llama-1-30b.Q4_K_M.gguf]=30g
)

derived_heap_for_model() {
  local path="$1" bytes heap_g
  bytes="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")"
  # Rough: file_size * 1.5 + 2 GiB headroom, clamp 4g..48g
  heap_g=$(( (bytes * 3 / 2 + 2 * 1024 * 1024 * 1024 + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024) ))
  (( heap_g < 4 )) && heap_g=4
  (( heap_g > 48 )) && heap_g=48
  printf '%sg' "$heap_g"
}

heap_for_model() {
  local path="$1" base
  if [[ -n "${COMPARE_HEAP:-}" ]]; then
    printf '%s' "$COMPARE_HEAP"
    return
  fi
  base="$(basename "$path")"
  if [[ -n "${FIXED_HEAP[$base]:-}" ]]; then
    printf '%s' "${FIXED_HEAP[$base]}"
    return
  fi
  derived_heap_for_model "$path"
}

# Where a run's heap came from, recorded in the run metadata. A derived heap is
# not comparable with a baseline taken at a fixed one, so the distinction has to
# survive into the published result rather than living only in this script.
heap_source_for_model() {
  local path="$1" base
  if [[ -n "${COMPARE_HEAP:-}" ]]; then
    printf 'explicit'
    return
  fi
  base="$(basename "$path")"
  if [[ -n "${FIXED_HEAP[$base]:-}" ]]; then
    printf 'fixed'
  else
    printf 'derived'
  fi
}

find_juno_jar() {
  if [[ -n "$JUNO_JAR_EXPLICIT" ]]; then
    printf '%s' "$JUNO_JAR_EXPLICIT"
    return
  fi
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
  command -v java >/dev/null 2>&1 || die "java not found (need JDK 25+)"
  printf '%s' java
}

# Clock state, so a thermally throttled or differently-governed run can be told
# apart from a code regression. Without it the two look identical in the result.
cpu_governor_state() {
  local govs
  govs="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | paste -sd, - || true)"
  printf '%s' "${govs:-unknown}"
}

cpu_turbo_state() {
  if [[ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    if [[ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" == "0" ]]; then
      printf 'enabled'
    else
      printf 'disabled'
    fi
  elif [[ -r /sys/devices/system/cpu/cpufreq/boost ]]; then
    if [[ "$(cat /sys/devices/system/cpu/cpufreq/boost)" == "1" ]]; then
      printf 'enabled'
    else
      printf 'disabled'
    fi
  else
    printf 'unknown'
  fi
}

# One clock reading per run, taken from the same source the vendor tool reports.
gpu_clock_json() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    printf 'null'
    return
  fi
  local graphics sm mem throttle
  graphics="$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  sm="$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  mem="$(nvidia-smi --query-gpu=clocks.current.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  throttle="$(nvidia-smi -q -d PERFORMANCE 2>/dev/null \
    | awk -F: '/Clocks (Event )?Reasons/{f=1;next} f&&/:/{gsub(/^[ \t]+|[ \t]+$/,"",$1);gsub(/^[ \t]+|[ \t]+$/,"",$2); if ($2=="Active") print $1} f&&/^$/{exit}' \
    | paste -sd, - || true)"
  printf '{"graphics_mhz": %s, "sm_mhz": %s, "memory_mhz": %s, "active_throttle_reasons": "%s"}' \
    "${graphics:-null}" "${sm:-null}" "${mem:-null}" "$(json_escape "${throttle:-none}")"
}

pin_clocks() {
  [[ "$PIN_CLOCKS" -eq 1 ]] || return 0
  sudo -n true 2>/dev/null \
    || die "--pin-clocks needs sudo without a prompt: run 'sudo -v' first, or drop --pin-clocks (the run is then recorded as unpinned)"
  local govs=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
  [[ -e "${govs[0]}" ]] || die "--pin-clocks: no cpufreq governor on this host; cannot pin the CPU clock"
  PINNED_PREV_GOVERNOR="$(cat "${govs[0]}")"
  printf 'performance\n' | sudo -n tee "${govs[@]}" >/dev/null \
    || die "--pin-clocks: could not set the performance governor"
  local detail="cpu governor performance (was ${PINNED_PREV_GOVERNOR})"
  if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    PINNED_PREV_NO_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
    printf '1\n' | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null \
      || die "--pin-clocks: could not turn turbo off"
    detail+=", turbo off"
  elif [[ -e /sys/devices/system/cpu/cpufreq/boost ]]; then
    PINNED_PREV_NO_TURBO="boost:$(cat /sys/devices/system/cpu/cpufreq/boost)"
    printf '0\n' | sudo -n tee /sys/devices/system/cpu/cpufreq/boost >/dev/null \
      || die "--pin-clocks: could not turn boost off"
    detail+=", boost off"
  else
    die "--pin-clocks: no turbo or boost control found; cannot pin the CPU clock"
  fi
  if [[ "$USE_GPU" -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local mhz="$PIN_GPU_MHZ"
    [[ -n "$mhz" ]] || mhz="$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$mhz" ]] && sudo -n nvidia-smi -lgc "${mhz},${mhz}" >/dev/null 2>&1; then
      PINNED_GPU=1
      detail+=", gpu graphics clock locked at ${mhz} MHz"
    else
      # Locking graphics clocks is not available on every card and driver (it is
      # refused on this project's Pascal GeForce). The CPU half still stands; the GPU
      # clock stays recorded rather than fixed, and the run says so.
      detail+=", gpu clock lock refused by the driver (recorded, not fixed)"
    fi
  fi
  CLOCK_PIN_STATE="pinned: ${detail}"
  log "clocks ${CLOCK_PIN_STATE}"
}

restore_clocks() {
  [[ -n "$PINNED_PREV_GOVERNOR" ]] || return 0
  local govs=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
  printf '%s\n' "$PINNED_PREV_GOVERNOR" | sudo -n tee "${govs[@]}" >/dev/null 2>&1 \
    || warn "could not restore the CPU governor to ${PINNED_PREV_GOVERNOR}"
  if [[ "$PINNED_PREV_NO_TURBO" == boost:* ]]; then
    printf '%s\n' "${PINNED_PREV_NO_TURBO#boost:}" | sudo -n tee /sys/devices/system/cpu/cpufreq/boost >/dev/null 2>&1 \
      || warn "could not restore the boost setting"
  elif [[ -n "$PINNED_PREV_NO_TURBO" ]]; then
    printf '%s\n' "$PINNED_PREV_NO_TURBO" | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1 \
      || warn "could not restore the turbo setting"
  fi
  if [[ "$PINNED_GPU" -eq 1 ]]; then
    sudo -n nvidia-smi -rgc >/dev/null 2>&1 || warn "could not reset the GPU clock lock"
  fi
  PINNED_PREV_GOVERNOR=""
}

# Build identity, so a published run names the code and toolchain it measured
# rather than leaving a reader to infer them from the date.
juno_commit() {
  git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown'
}

juno_tree_dirty() {
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    printf 'true'
  else
    printf 'false'
  fi
}

# Hashes the jar the engine is actually launched from (find_juno_jar), not a fixed
# name the build does not produce: that recorded `missing` on every run until 2026-09-30.
juno_jar_sha256() {
  local jar
  jar="$(find_juno_jar 2>/dev/null || true)"
  if [[ -n "$jar" && -f "$jar" ]]; then
    sha256sum "$jar" 2>/dev/null | cut -c1-16
  else
    printf 'missing'
  fi
}

java_version_line() {
  local j
  j="${JAVA_HOME:+${JAVA_HOME}/bin/}java"
  "$j" -version 2>&1 | sed -n 2p || printf 'unknown'
}

gpu_driver_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || printf 'unknown'
  else
    printf 'none'
  fi
}

# The reference tool reports its own build in every result; read it from the first
# one this run wrote. Two builds are in use on this host (one per backend), so a run
# that does not name its build cannot be compared with one that does.
llama_build_commit() {
  local f
  for f in "${OUT_ROOT}"/*-llama-cpp.json; do
    [[ -f "$f" ]] || continue
    jq -r '[.. | objects | select(has("build_commit")) | .build_commit][0] // empty' "$f" 2>/dev/null && return 0
  done
  printf ''
}

host_meta_json() {
  local cpu mem
  cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || uname -m)"
  mem="$(awk '/MemTotal/ {printf "%.1f GiB", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo unknown)"
  cat <<EOF
{
  "run_id": "$(json_escape "$RUN_ID")",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(json_escape "$(hostname 2>/dev/null || echo unknown)")",
  "cpu": "$(json_escape "$cpu")",
  "mem_total": "$(json_escape "$mem")",
  "n_threads": ${N_THREADS},
  "juno_threads": ${JUNO_THREADS},
  "juno_common_pool_parallelism": ${JUNO_POOL_PARALLELISM},
  "cpu_governor": "$(json_escape "$(cpu_governor_state)")",
  "cpu_turbo": "$(json_escape "$(cpu_turbo_state)")",
  "gpu_clocks": $(gpu_clock_json),
  "clock_pinned": $([[ "$CLOCK_PIN_STATE" == pinned:* ]] && echo true || echo false),
  "clock_pin_state": "$(json_escape "$CLOCK_PIN_STATE")",
  "juno_commit": "$(json_escape "$(juno_commit)")",
  "juno_tree_dirty": $(juno_tree_dirty),
  "juno_jar": "$(json_escape "$(find_juno_jar 2>/dev/null || echo missing)")",
  "juno_jar_sha256_16": "$(json_escape "$(juno_jar_sha256)")",
  "java_version": "$(json_escape "$(java_version_line)")",
  "juno_jvm_flags": "$(json_escape "${JUNO_JVM_FLAGS[*]} -Xms=-Xmx (fixed per model)")",
  "jfr_timestamp_source": "$(json_escape "$(perf_jfr_clock_label)")",
  "device_spans": ${DEVICE_SPANS},
  "gpu_driver": "$(json_escape "$(gpu_driver_version)")",
  "llama_build_commit": "$(json_escape "$(llama_build_commit)")",
  "n_prompt": ${N_PROMPT},
  "n_gen": ${N_GEN},
  "reps": ${REPS},
  "juno_warmup": ${JUNO_WARMUP},
  "juno_reps": ${JUNO_REPS},
  "juno_min_tokens": ${JUNO_MIN_TOKENS},
  "ngl": ${NGL},
  "backend": "$(backend_label)",
  "use_gpu": ${USE_GPU},
  "prompt": "$(json_escape "$PROMPT_TEXT")",
  "raw_prompt": ${RAW_PROMPT},
  "juno_gpu_layers": "$(json_escape "${JUNO_GPU_LAYERS:-}")",
  "juno_mmq": "$(json_escape "${JUNO_MMQ:-}")",
  "juno_gpu_attention": "$(json_escape "${JUNO_GPU_ATTENTION:-}")",
  "juno_gpu_residency": "$(json_escape "${JUNO_GPU_RESIDENCY:-}")",
  "juno_schedule": "$(json_escape "${JUNO_SCHEDULE:-}")",
  "juno_cache_type_k": "$(json_escape "${JUNO_CACHE_TYPE_K:-}")",
  "juno_cache_type_v": "$(json_escape "${JUNO_CACHE_TYPE_V:-}")",
  "juno_kv_page_size": "$(json_escape "${JUNO_KV_PAGE_SIZE:-}")",
  "juno_prefill_batch": "$(json_escape "${JUNO_PREFILL_BATCH:-}")",
  "juno_spec_type": "$(json_escape "${JUNO_SPEC_TYPE:-}")",
  "juno_spec_ngram_n": "$(json_escape "${JUNO_SPEC_NGRAM_N:-}")",
  "juno_spec_ngram_m": "$(json_escape "${JUNO_SPEC_NGRAM_M:-}")",
  "juno_model_draft": "$(json_escape "${JUNO_MODEL_DRAFT:-}")",
  "juno_use_vector": ${JUNO_USE_VECTOR},
  "juno_jfr": ${USE_JFR},
  "jfr_duration": "$(json_escape "${JFR_DURATION}")",
  "llama_cpp_bin": "$(json_escape "${LLAMA_CPP_BIN:-}")"
}
EOF
}

run_llama_bench() {
  local model_path="$1" stem="$2"
  local out_json="${OUT_ROOT}/${stem}-llama-cpp.json"
  local raw="${OUT_ROOT}/${stem}-llama-cpp.raw.json"
  local bench="${LLAMA_CPP_BIN}/llama-bench"
  local logf="${OUT_ROOT}/${stem}-llama-cpp.log"

  [[ -x "$bench" ]] || die "llama-bench not found/executable: ${bench}"

  log "llama.cpp: ${stem} (pp=${N_PROMPT} tg=${N_GEN} reps=${REPS} ngl=${NGL})"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: LD_LIBRARY_PATH=${LLAMA_CPP_BIN} ${bench} -m ${model_path} -p ${N_PROMPT} -n ${N_GEN} -r ${REPS} -t ${N_THREADS} -ngl ${NGL} -o json"
    return 0
  fi

  local start_ns end_ns elapsed_ms rc=0
  start_ns="$(date +%s%N)"
  set +e
  (
    cd "$LLAMA_CPP_BIN"
    export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ./llama-bench -m "$model_path" -p "$N_PROMPT" -n "$N_GEN" -r "$REPS" \
      -t "$N_THREADS" -ngl "$NGL" -o json
  ) >"$raw" 2>"$logf"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

  local pp_tps tg_tps build_number model_type
  pp_tps="$(jq -r '[.[] | select(.n_prompt > 0 and .n_gen == 0) | .avg_ts] | if length>0 then .[0] else empty end' "$raw" 2>/dev/null || true)"
  tg_tps="$(jq -r '[.[] | select(.n_gen > 0 and .n_prompt == 0) | .avg_ts] | if length>0 then .[0] else empty end' "$raw" 2>/dev/null || true)"
  build_number="$(jq -r '.[0].build_number // empty' "$raw" 2>/dev/null || true)"
  model_type="$(jq -r '.[0].model_type // empty' "$raw" 2>/dev/null || true)"

  cat >"$out_json" <<EOF
{
  "engine": "llama.cpp",
  "tool": "llama-bench",
  "status": "$([ "$rc" -eq 0 ] && echo success || echo failure)",
  "exit_code": ${rc},
  "model": "$(json_escape "$(basename "$model_path")")",
  "model_path": "$(json_escape "$model_path")",
  "model_type": "$(json_escape "${model_type:-}")",
  "build_number": ${build_number:-null},
  "n_prompt": ${N_PROMPT},
  "n_gen": ${N_GEN},
  "repetitions": ${REPS},
  "threads": ${N_THREADS},
  "n_gpu_layers": ${NGL},
  "prompt_eval_tps": ${pp_tps:-null},
  "token_gen_tps": ${tg_tps:-null},
  "wall_ms": ${elapsed_ms},
  "raw_json": "$(json_escape "$raw")",
  "log": "$(json_escape "$logf")",
  "host": $(host_meta_json)
}
EOF
  # Attach llama-bench samples when raw output is valid JSON.
  if [[ -s "$raw" ]] && jq -e 'type == "array"' "$raw" >/dev/null 2>&1; then
    jq --slurpfile raw "$raw" '. + {llama_bench: $raw[0]}' "$out_json" >"${out_json}.tmp" \
      && mv "${out_json}.tmp" "$out_json"
  fi

  if [[ "$rc" -ne 0 ]]; then
    warn "llama-bench failed for ${stem} (rc=${rc}) — see ${logf}"
    return 1
  fi
  log "llama.cpp result: ${out_json}  (pp=${pp_tps:-?} tg=${tg_tps:-?} t/s)"
  return 0
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
  if [[ -z "$pid" ]]; then
    perf_engine_stdin_release
    return 0
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  JUNO_PID=""
  perf_engine_stdin_release
}

# One request. Callers that discard it read nothing back but its success.
juno_chat_request() {
  local model_id="$1" prompt="$2" max_tokens="$3" dest="$4" logf="$5" min_tokens="${6:-0}" code
  code="$(curl -sS --max-time 7200 -o "$dest" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$model_id" --arg p "$prompt" --argjson n "$max_tokens" \
      --argjson min "$min_tokens" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}
       + (if $min > 0 then {min_tokens:$min} else {} end)')" \
    2>>"$logf" || echo 000)"
  [[ "$code" == "200" ]]
}

# A prompt of N words, the minimal-token pattern the raw-prompt mode uses.
raw_prompt_of_words() {
  local words="$1"
  (( words < 1 )) && words=1
  printf 'x'
  local i
  for (( i = 1; i < words; i++ )); do printf ' x'; done
}

# Picks the word count whose real prefill lands on n_prompt.
#
# Asking for n_prompt words does not produce n_prompt prefill tokens: the chat
# template wraps every request in role and control tokens, which on this model is
# about 19 tokens of fixed overhead. That overhead is what made a 32-word prompt
# prefill 50 tokens, 56% over the request, and at n_prompt=128 it would still be
# 15% over — past the parity tolerance, so every prefill ratio in the sweep would
# be withheld and the one metric this comparison exists to move would go
# unpublished. Nothing here assumes a particular template: the engine is asked
# what it actually tokenized, and the word count is corrected by the difference.
# The parity gate downstream still has the final say.
calibrate_raw_prompt_words() {
  local model_id="$1" logf="$2" dest="$3"
  local probe measured overhead words
  probe="$(raw_prompt_of_words "$N_PROMPT")"
  # Printing nothing on failure, rather than the uncalibrated count: a caller that
  # got a plausible number back would report the prompt as calibrated when it was
  # not, and the deviation would then be blamed on the engine.
  if ! juno_chat_request "$model_id" "$probe" 1 "$dest" "$logf"; then
    return 1
  fi
  measured="$(jq -r '.usage.prompt_tokens // 0' "$dest" 2>/dev/null || echo 0)"
  if ! [[ "$measured" =~ ^[0-9]+$ ]] || (( measured <= 0 )); then
    return 1
  fi
  overhead=$(( measured - N_PROMPT ))
  words=$(( N_PROMPT - overhead ))
  (( words < 1 )) && words=1
  printf '%s' "$words"
}

# Turns the recording this script scoped into the metrics JSON the rest of the
# script reads. The engine does this itself for its own --jfr recording, but that
# recording covers the whole process lifetime; a harness that owns its window has
# to name both the recording and where the metrics land.
extract_jfr_metrics() {
  local jfr_file="$1" dest="$2" model_stem="$3" model_filename="$4" logf="$5"
  local java_bin jar
  java_bin="$(find_java)"
  jar="$(find_juno_jar)"
  [[ -s "$jfr_file" ]] || return 1
  "$java_bin" -cp "$jar" cab.ml.juno.metrics.JfrMetricsCli \
    "$jfr_file" "$dest" "$model_stem" "$model_filename" >>"$logf" 2>&1 || return 1
  [[ -s "$dest" ]]
}

# Derive pp/tg from JFR metrics + API token counts; writes compact JSON object to stdout.
jfr_summary_json() {
  local jfr_file="$1" prompt_tokens="$2" completion_tokens="$3" latency_ms="${4:-0}"
  [[ -f "$jfr_file" ]] || { echo null; return 0; }
  jq -nc --arg f "$jfr_file" \
    --argjson pt "${prompt_tokens:-0}" --argjson ct "${completion_tokens:-0}" \
    --argjson latency_ms "${latency_ms:-0}" --argjson spans "${DEVICE_SPANS:-0}" \
    --slurpfile raw "$jfr_file" '
    ($raw[0].models[0].metrics // {}) as $m |
    ($m."juno.ForwardPass.prefill.total_ms" // 0) as $prefill_ms |
    ($m."juno.ForwardPass.decode.total_ms" // 0) as $decode_ms |
    ($m."juno.TokenProduced.tps" // null) as $token_tps |
    (if $decode_ms > 0 and $ct > 0 then ($ct / ($decode_ms / 1000.0)) else null end) as $decode_derived_tps |
    (if $prefill_ms > 0 and $pt > 0 then ($pt / ($prefill_ms / 1000.0))
     elif ($latency_ms > $decode_ms and $pt > 0)
     then ($pt / (($latency_ms - $decode_ms) / 1000.0))
     else null end) as $pp_tps |
    (if $prefill_ms > 0 then "jfr_prefill_total_ms"
     elif ($latency_ms > $decode_ms and $pt > 0) then "wall_minus_decode"
     else null end) as $pp_source |
    # The spans must account for the request that contains them. The engine latency
    # (read off System.nanoTime) minus the prefill and decode spans leaves a small
    # positive overhead, 9 to 170 ms on the reference sweeps; a span misread by a
    # CPU timestamp counter that disagrees between cores leaves hundreds of ms either
    # side of that. Spans exceeding their request are impossible, so the lower bound
    # is tight; the upper bound allows per-token sampling and detokenization.
    (if $latency_ms > 0 and ($prefill_ms + $decode_ms) > 0
     then ($latency_ms - $prefill_ms - $decode_ms) else null end) as $resid |
    (300 + 3 * $ct) as $resid_max |
    # The generation figure is read off the first-to-last token span, which is a
    # separate pair of timestamps and is misread separately. What the request leaves
    # around it after prefill is the first decode step (the span starts at the first
    # token) plus overhead; one mean decode step is taken off, so a slow CPU step is
    # not read as a misread span. 22 to 106 ms before that subtraction on the GPU
    # reference sweeps.
    ($m."juno.TokenProduced.elapsed_seconds" // null) as $tok_span_s |
    ($m."juno.ForwardPass.decode.count" // 0) as $decode_n |
    (if $decode_n > 0 then ($decode_ms / $decode_n) else 0 end) as $decode_step_ms |
    (if $ct >= 2 and $tok_span_s != null and $tok_span_s > 0 and $latency_ms > 0
     then ($latency_ms - $prefill_ms - ($tok_span_s * 1000) - $decode_step_ms) else null end) as $tok_resid |
    (if $resid == null and $tok_resid == null then null
     else (($resid == null or ($resid >= -25 and $resid <= $resid_max))
           and ($tok_resid == null or ($tok_resid >= -25 and $tok_resid <= $resid_max))) end) as $clock_ok |
    # A measured request prefills its whole prompt from position 0: the batched
    # windows cover every prompt token but the last, which the first forward pass
    # takes. A later start means it resumed from KV it did not write.
    ($m."juno.PrefillBatch.tokens" // null) as $pb_tokens |
    ($m."juno.PrefillBatch.min_start_position" // null) as $pb_start |
    (if $pb_tokens == null or $pb_tokens == 0 or $pt <= 1 then null
     else ($pb_start == 0 and $pb_tokens >= ($pt - 1)) end) as $full_prefill |
    {
      metrics_file: $f,
      jfr_file: ($raw[0].models[0].jfrFile // null),
      # A reading taken off spans that fail the check below is withheld, not published.
      prompt_eval_tps: (if $clock_ok == false or $full_prefill == false then null else $pp_tps end),
      prompt_eval_tps_source: $pp_source,
      token_gen_tps: (if $clock_ok == false then null
                      elif $token_tps != null and $token_tps > 0 then $token_tps
                      else $decode_derived_tps end),
      span_check: {
        timestamps_consistent: $clock_ok,
        span_residual_ms: $resid,
        token_span_residual_ms: $tok_resid,
        span_residual_min_ms: -25,
        span_residual_max_ms: $resid_max,
        prefill_tokens: $pb_tokens,
        prefill_min_start_position: $pb_start,
        full_prefill: $full_prefill,
        reason: (if $clock_ok == false then
                   ("the recorded spans leave \($resid // 0 | floor) ms (forward passes) and "
                    + "\($tok_resid // 0 | floor) ms (token span) of the \($latency_ms) ms request "
                    + "unaccounted for (allowed -25 to \($resid_max)): JFR timestamps disagree "
                    + "with the engine clock, readings withheld")
                 elif $full_prefill == false then
                   ("the request prefilled \($pb_tokens) of \($pt) prompt tokens starting at "
                    + "position \($pb_start): it did not prefill its whole prompt, prefill reading withheld")
                 else null end)
      },
      token_gen_tps_source: (if $token_tps != null and $token_tps > 0 then "TokenProduced.tps" else "ForwardPass.decode.total_ms" end),
      token_gen_tps_decode_derived: $decode_derived_tps,
      forward_pass_prefill_total_ms: $prefill_ms,
      forward_pass_decode_total_ms: $decode_ms,
      forward_pass_prefill_p95_ms: ($m."juno.ForwardPass.prefill.p95_ms" // null),
      forward_pass_decode_p95_ms: ($m."juno.ForwardPass.decode.p95_ms" // null),
      forward_pass_prefill_count: ($m."juno.ForwardPass.prefill.count" // null),
      forward_pass_decode_count: ($m."juno.ForwardPass.decode.count" // null),
      token_produced_count: ($m."juno.TokenProduced.count" // null),
      token_produced_tps: $token_tps,
      token_produced_elapsed_s: ($m."juno.TokenProduced.elapsed_seconds" // null),
      # A short measurement window holding one long collection pause reports a
      # throughput drop that looks exactly like a code regression. Recording the
      # pauses is what lets the two be told apart after the run.
      gc_pause_count: ($m."jdk.GCPhasePause.count" // null),
      gc_pause_max_ms: ($m."jdk.GCPhasePause.max_ms" // null),
      gc_pause_total_ms: ($m."jdk.GCPhasePause.total_ms" // null),
      # The share of those pauses that overlapped the first-token-to-last-token
      # span, which is the window the generation figure is measured over. A pause
      # outside it cannot have slowed that figure.
      gc_pause_count_in_token_span: ($m."jdk.GCPhasePause.in_token_span.count" // null),
      gc_pause_max_ms_in_token_span: ($m."jdk.GCPhasePause.in_token_span.max_ms" // null),
      gc_pause_total_ms_in_token_span: ($m."jdk.GCPhasePause.in_token_span.total_ms" // null),
      allocated_bytes_total: ($m."jdk.ThreadAllocationStatistics.bytes_total" // null),
      allocated_bytes_per_token:
        (($m."jdk.ThreadAllocationStatistics.bytes_total" // null) as $bytes
         | if $bytes != null and $ct > 0 then ($bytes / $ct) else null end),
      execution_sample_count: ($m."jdk.ExecutionSample.count" // null),
      top_methods: [ $m | to_entries[]
                     | select(.key | startswith("jdk.ExecutionSample.top_methods."))
                     | { method: (.key | ltrimstr("jdk.ExecutionSample.top_methods.") | rtrimstr(".samples")),
                         samples: .value } ]
                   | sort_by(-.samples),
      top_allocation_sites: [ $m | to_entries[]
                     | select(.key | startswith("jdk.ObjectAllocationSample.top_sites."))
                     | { site: (.key | ltrimstr("jdk.ObjectAllocationSample.top_sites.") | rtrimstr(".bytes")),
                         bytes: .value } ]
                   | sort_by(-.bytes),
      monitor_enter_total_ms: ($m."jdk.JavaMonitorEnter.total_ms" // null),
      thread_park_total_ms: ($m."jdk.ThreadPark.total_ms" // null),
      # Host-device copies and weight dequantization, which the matmul span otherwise
      # hides inside itself. The prefill figures are the bytes a residency change is
      # scored on; durations are measured on the device for asynchronous copies. A
      # build that predates the events, or a run without --device-spans (the events
      # were not recorded), reports null, not zero.
      device_staging: (($m."juno.DeviceStaging.H2D.prefill.bytes" // null) as $h2d
        | ($m."juno.DeviceStaging.D2H.prefill.bytes" // null) as $d2h
        | if $h2d == null or $spans == 0 then null else {
            prefill_h2d_bytes: $h2d,
            prefill_d2h_bytes: $d2h,
            # Small copies are timed one in sixteen; the estimate scales the mean of each site
            # to all of its copies (older builds: the measured sum).
            prefill_h2d_ms: ($m."juno.DeviceStaging.H2D.prefill.estimated_total_ms"
                             // $m."juno.DeviceStaging.H2D.prefill.total_ms" // null),
            prefill_d2h_ms: ($m."juno.DeviceStaging.D2H.prefill.estimated_total_ms"
                             // $m."juno.DeviceStaging.D2H.prefill.total_ms" // null),
            prefill_copies: (($m."juno.DeviceStaging.H2D.prefill.count" // 0)
                             + ($m."juno.DeviceStaging.D2H.prefill.count" // 0)),
            prefill_bytes_per_token: (if $pb_tokens != null and $pb_tokens > 0
                                      then (($h2d + $d2h) / $pb_tokens) else null end),
            decode_h2d_bytes: ($m."juno.DeviceStaging.H2D.decode.bytes" // null),
            decode_d2h_bytes: ($m."juno.DeviceStaging.D2H.decode.bytes" // null),
            decode_h2d_ms: ($m."juno.DeviceStaging.H2D.decode.total_ms" // null),
            decode_d2h_ms: ($m."juno.DeviceStaging.D2H.decode.total_ms" // null),
            dequant_device_count: ($m."juno.WeightDequant.device.count" // null),
            dequant_device_ms: ($m."juno.WeightDequant.device.total_ms" // null),
            # Host FP16 packing of each activation window: host work done only to stage
            # the upload, kept apart from the bytes that cross the bus (null before it existed).
            prefill_pack_host_ms: ($m."juno.DeviceStaging.HOST.prefill.total_ms" // null),
            # Device kernels timed at prefill width: the GEMM term, read rather than inferred.
            prefill_compute_ms: ($m."juno.DeviceCompute.prefill.total_ms" // null),
            prefill_compute_ms_by_site: ([$m | to_entries[]
                | select(.key | test("^juno\\.DeviceCompute\\.site\\..*\\.prefill\\.total_ms$"))
                | { key: (.key | sub("^juno\\.DeviceCompute\\.site\\."; "") | sub("\\.prefill\\.total_ms$"; "")),
                    value: .value }] | if length == 0 then null else from_entries end)
          } end)
    }
  '
}

# One measured Juno cycle: start the engine, discard JUNO_WARMUP requests, record
# and measure exactly one request, stop.
#
# The recording is started by this script through jcmd rather than by passing
# --jfr to the engine, and that is what makes warmup possible at all. The engine's
# own --jfr recording runs for the whole process lifetime, so the requests that
# must not be measured would land in the same window as the one that must, and the
# aggregates every published number is read off — the token-to-token span above
# all — would mix them. Starting the recording after the warmup requests have
# returned puts exactly the measured request inside it. Verified on a real run:
# with two warmups and an 8-token measured request the recording holds 8
# juno.TokenProduced events, not 24.
# What the engine actually ran for attention, rather than the flag this script
# passed: the default lane passes no --gpu-attention at all, and a handler without
# the GPU kernel resolves to scalar attention without saying so. Evidence, in order:
# LlamaTransformerHandler's activation or fallback line in the engine log (only
# written under --verbose; the console turns library logging off otherwise), then,
# on a --device-spans run, the attention kernel's own copy sites (memcpy_gqa_*) in
# the recording, or its launches inside the prefill-window device region
# (juno.DeviceCompute site gqa_attention_region), which copy nothing. on,
# off-backend-fallback, off (spans run, no kernel evidence), unknown (nothing
# observable: a silent log is the normal case), or n/a on CPU.
gpu_attention_resolved_from_log() {
  local logf="$1" use_gpu="$2" jfr_json="${3:-}"
  local gqa_copies
  if [[ "$use_gpu" -ne 1 ]]; then
    echo "n/a"
  elif grep -q 'GPU-resident attention path active' "$logf" 2>/dev/null; then
    echo "on"
  elif grep -q 'falling back to scalar CPU attention' "$logf" 2>/dev/null; then
    echo "off-backend-fallback"
  elif [[ "${DEVICE_SPANS:-0}" == "1" && -f "$jfr_json" ]]; then
    gqa_copies="$(jq -r '[.models[0].metrics // {} | to_entries[]
                          | select(.key | test("^juno\\.DeviceStaging\\.site\\.memcpy_gqa_.*\\.count$")
                                          or test("^juno\\.DeviceCompute\\.site\\.gqa_attention_region\\..*\\.count$"))
                          | .value]
                         | add // 0' "$jfr_json" 2>/dev/null || echo 0)"
    if awk -v n="$gqa_copies" 'BEGIN { exit !(n > 0) }'; then
      echo "on"
    else
      echo "off"
    fi
  else
    echo "unknown"
  fi
}

run_juno_rep() {
  local model_path="$1" stem="$2" rep_label="$3" lane="${4:-generate}"
  local out_json="${OUT_ROOT}/${rep_label}-juno.json"
  local resp="${OUT_ROOT}/${rep_label}-juno-response.json"
  local logf="${OUT_ROOT}/${rep_label}-juno.log"
  local jar java_bin heap heap_source
  local -a java_args=()

  # The shape of this lane request, decided before the engine starts because the
  # launch log states it. The prefill lane asks for one token, since its figure is
  # the prefill and generation after it is cost without a reading; the generation
  # lane asks for the full count and insists on it.
  local lane_max_tokens="$N_GEN" lane_min_tokens="$JUNO_MIN_TOKENS" lane_n_prompt="$N_PROMPT"
  if [[ "$lane" == "prefill" ]]; then
    lane_max_tokens=1
    lane_min_tokens=0
  else
    lane_n_prompt=0
  fi

  jar="$(find_juno_jar)"
  java_bin="$(find_java)"
  heap="$(heap_for_model "$model_path")"
  heap_source="$(heap_source_for_model "$model_path")"

  local juno_backend_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && juno_backend_flag="--gpu"

  java_args=(
    --enable-preview
    --enable-native-access=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED
    --add-opens java.base/java.nio=ALL-UNNAMED
  )
  if [[ "$JUNO_USE_VECTOR" == "1" ]]; then
    java_args+=(--add-modules jdk.incubator.vector)
  fi
  java_args+=(
    "${JUNO_JVM_FLAGS[@]}"
    -Xms"${heap}"
    -Xmx"${heap}"
    -Djuno.byteOrder=BE
    -jar "$jar"
    --model-path "$model_path"
    --dtype FLOAT16
    --byteOrder BE
    --max-tokens "$N_GEN"
    --temperature 0
    --top-k 0
    --top-p 0
    --nodes 1
    --local
    "$juno_backend_flag"
    --api-port "$API_PORT"
  )
  if [[ -n "$JUNO_GPU_LAYERS" ]]; then
    java_args+=(--gpu-layers "$JUNO_GPU_LAYERS")
  fi
  if [[ -n "$JUNO_MMQ" ]]; then
    java_args+=(--mmq "$JUNO_MMQ")
  fi
  if [[ -n "$JUNO_GPU_RESIDENCY" ]]; then
    java_args+=(--gpu-residency "$JUNO_GPU_RESIDENCY")
  fi
  if [[ -n "$JUNO_GPU_ATTENTION" ]]; then
    java_args+=(--gpu-attention "$JUNO_GPU_ATTENTION")
  fi
  if [[ -n "${JUNO_PREFILL_BATCH:-}" ]]; then
    java_args+=(--prefill-batch "$JUNO_PREFILL_BATCH")
  fi
  if [[ -n "${JUNO_SPEC_TYPE:-}" ]]; then
    java_args+=(--spec-type "$JUNO_SPEC_TYPE")
  fi
  if [[ -n "${JUNO_SPEC_NGRAM_N:-}" ]]; then
    java_args+=(--spec-ngram-n "$JUNO_SPEC_NGRAM_N")
  fi
  if [[ -n "${JUNO_SPEC_NGRAM_M:-}" ]]; then
    java_args+=(--spec-ngram-m "$JUNO_SPEC_NGRAM_M")
  fi
  if [[ -n "${JUNO_MODEL_DRAFT:-}" ]]; then
    java_args+=(--model-draft "$JUNO_MODEL_DRAFT")
  fi
  if [[ -n "$JUNO_SCHEDULE" ]]; then
    java_args+=(--schedule "$JUNO_SCHEDULE")
  fi
  if [[ -n "$JUNO_CACHE_TYPE_K" ]]; then
    java_args+=(--cache-type-k "$JUNO_CACHE_TYPE_K")
  fi
  if [[ -n "$JUNO_CACHE_TYPE_V" ]]; then
    java_args+=(--cache-type-v "$JUNO_CACHE_TYPE_V")
  fi
  if [[ -n "$JUNO_KV_PAGE_SIZE" ]]; then
    java_args+=(--kv-page-size "$JUNO_KV_PAGE_SIZE")
  fi

  log "juno: ${rep_label} (lane=${lane} backend=$(backend_label) max_tokens=${lane_max_tokens} heap=${heap}/${heap_source} vector=${JUNO_USE_VECTOR} warmup=${JUNO_WARMUP} jfr=${USE_JFR} port=${API_PORT} gpu_layers=${JUNO_GPU_LAYERS:-default})"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: ${java_bin} ${java_args[*]}"
    return 0
  fi

  # Ensure port free
  if curl -sf "http://127.0.0.1:${API_PORT}/v1/cluster/health" >/dev/null 2>&1; then
    die "port ${API_PORT} already has a healthy Juno API — stop it or pass --api-port"
  fi

  : >"$logf"
  # Keep stdin open: the console REPL exits on EOF, which would tear down the API
  # mid-benchmark when launched non-interactively (perf-lib.sh, "Engine stdin keepalive").
  perf_engine_stdin_open || die "cannot create the engine stdin pipe"
  (
    cd "$ROOT"
    perf_engine_exec "$java_bin" "${java_args[@]}"
  ) >>"$logf" 2>&1 &
  JUNO_PID=$!

  local start_ns end_ns wall_ms load_ms gen_ms rc=0
  local completion_tokens=0 prompt_tokens=0 latency_ms=0 tps=0
  local model_id="" finish_reason=""
  local jfr_metrics="${OUT_ROOT}/${rep_label}-juno-jfr.json"
  local jfr_recording="${OUT_ROOT}/${rep_label}-measured.jfr"
  local jfr_block="null"
  local jfr_pp_tps="null" jfr_tg_tps="null" span_reason="" span_withheld=0

  load_start="$(date +%s%N)"
  if ! wait_for_juno_api "$API_PORT" 600 "$JUNO_PID"; then
    warn "juno API did not become healthy — see ${logf}"
    stop_juno
    local err_snip
    err_snip="$(grep -E 'Exception|Error|Tensor not found|Unsupported' "$logf" 2>/dev/null | tail -3 | tr '\n' ' ' | head -c 400 || true)"
    cat >"$out_json" <<EOF
{
  "engine": "juno",
  "tool": "local-api",
  "status": "failure",
  "error": "api_health_timeout_or_crash",
  "error_detail": "$(json_escape "${err_snip:-}")",
  "model": "$(json_escape "$(basename "$model_path")")",
  "model_path": "$(json_escape "$model_path")",
  "log": "$(json_escape "$logf")",
  "juno_use_vector": ${JUNO_USE_VECTOR},
  "backend": "$(backend_label)",
  "use_gpu": ${USE_GPU},
  "heap": "$(json_escape "$heap")",
  "host": $(host_meta_json)
}
EOF
    return 1
  fi
  load_end="$(date +%s%N)"
  load_ms=$(( (load_end - load_start) / 1000000 ))

  model_id="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" \
    | jq -r '.data[0].id // .models[0].modelId // empty' 2>/dev/null || true)"
  if [[ -z "$model_id" ]]; then
    # OpenAI list shape uses .data[].id; native may differ — fall back to filename stem
    model_id="$(basename "$model_path")"
    model_id="${model_id%.gguf}"
  fi

  # Each lane measures one figure, so each shapes its own request. The prefill lane
  # is given the requested prompt length and asked for a single token, because its
  # figure is the prefill and any generation after it is cost without a reading. The
  # generation lane is given the shortest prompt the chat template allows and asked
  # for the full token count, so it decodes at the shallow context the reference
  # tool measures its own generation at.
  local measured_prompt="$PROMPT_TEXT" calibrated_words=0
  if [[ "$lane" == "prefill" ]]; then
    if [[ "$RAW_PROMPT" -eq 1 ]]; then
      calibrated_words="$(calibrate_raw_prompt_words "$model_id" "$logf" \
        "${OUT_ROOT}/${rep_label}-juno-calibration.json" || true)"
      if [[ "$calibrated_words" =~ ^[0-9]+$ ]] && (( calibrated_words > 0 )); then
        measured_prompt="$(raw_prompt_of_words "$calibrated_words")"
        log "prompt calibration: ${calibrated_words} words to reach n_prompt=${N_PROMPT}"
      else
        warn "prompt calibration failed; using the uncalibrated ${N_PROMPT}-word prompt"
        calibrated_words=0
      fi
    fi
  else
    # The reference tool generates from an empty context. Juno cannot: the chat
    # template wraps every request, so the shortest reachable prompt is that
    # wrapper plus one word. The residual is recorded rather than hidden.
    measured_prompt="$GENERATE_LANE_PROMPT"
  fi

  # Discarded requests, so the measured one runs on compiled code rather than
  # paying for the compilation of the whole forward pass inside the window. They
  # use the measured prompt, so they compile the shapes the measured request runs.
  #
  # The last warmup runs under a throwaway recording with the measurement settings.
  # Starting the first recording in a process instruments the event classes and
  # takes every branch that only runs while one records; code compiled during an
  # unrecorded warmup has never seen those paths, so the first recording
  # deoptimizes it and the measured request would pay for the recompilation (seen:
  # the first two of 22 layers of a TinyLlama prefill window running two to three
  # times slower, then every layer identical). The throwaway recording moves that
  # into the warmup; its data is discarded.
  local w warmup_rc=0 warmup_recording=0
  local -a jfr_settings_args=("settings=${JFR_SETTINGS_FILE}")
  (( DEVICE_SPANS )) && jfr_settings_args+=("settings=${JFR_SPANS_FILE}")
  for (( w = 1; w <= JUNO_WARMUP; w++ )); do
    if [[ "$USE_JFR" -eq 1 && "$w" -eq "$JUNO_WARMUP" ]]; then
      if jcmd "$JUNO_PID" JFR.start name="$JFR_WARMUP_RECORDING_NAME" "${jfr_settings_args[@]}" \
           >>"$logf" 2>&1; then
        warmup_recording=1
      else
        warn "could not start the warmup recording — the measured request may pay for recompilation; see ${logf}"
      fi
    fi
    log "juno warmup ${w}/${JUNO_WARMUP}: ${rep_label}"
    if ! juno_chat_request "$model_id" "$measured_prompt" "$lane_max_tokens" \
         "${OUT_ROOT}/${rep_label}-juno-warmup${w}.json" "$logf" "$lane_min_tokens"; then
      warn "juno warmup request ${w} failed — see ${logf}"
      warmup_rc=1
      break
    fi
  done
  if (( warmup_recording )); then
    jcmd "$JUNO_PID" JFR.stop name="$JFR_WARMUP_RECORDING_NAME" >>"$logf" 2>&1 \
      || warn "could not stop the warmup recording — see ${logf}"
  fi

  # From here to JFR.stop is the measurement window, and nothing else is in it.
  if [[ "$USE_JFR" -eq 1 && "$warmup_rc" -eq 0 ]]; then
    # --jfr DURATION is an upper bound on the window, not the window itself: the
    # window is the measured request, and the bound only stops a recording whose
    # request never returned. The dump lands in the file named here either way, so
    # a recording that hit the bound is still readable.
    if ! jcmd "$JUNO_PID" JFR.start name="$JFR_RECORDING_NAME" "${jfr_settings_args[@]}" \
         "duration=${JFR_DURATION}" "filename=${jfr_recording}" >>"$logf" 2>&1; then
      warn "could not start the measurement recording — see ${logf}"
      USE_JFR_THIS_REP=0
    else
      USE_JFR_THIS_REP=1
    fi
  else
    USE_JFR_THIS_REP=0
  fi

  start_ns="$(date +%s%N)"
  set +e
  curl -sS --max-time 7200 -o "$resp" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$model_id" --arg p "$measured_prompt" --argjson n "$lane_max_tokens" \
      --argjson min "$lane_min_tokens" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}
       + (if $min > 0 then {min_tokens:$min} else {} end)')" \
    >"${OUT_ROOT}/${rep_label}-juno.http" 2>>"$logf"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  local http_code
  http_code="$(cat "${OUT_ROOT}/${rep_label}-juno.http" 2>/dev/null || echo 000)"

  if [[ "$rc" -eq 0 && "$http_code" == "200" && -s "$resp" ]]; then
    completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$resp")"
    prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$resp")"
    latency_ms="$(jq -r '.x_juno_latency_ms // 0' "$resp")"
    finish_reason="$(jq -r '.choices[0].finish_reason // empty' "$resp")"
    if [[ "$latency_ms" =~ ^[0-9]+$ ]] && (( latency_ms > 0 && completion_tokens > 0 )); then
      tps="$(awk -v t="$completion_tokens" -v ms="$latency_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
    elif (( wall_ms > 0 && completion_tokens > 0 )); then
      tps="$(awk -v t="$completion_tokens" -v ms="$wall_ms" 'BEGIN { printf "%.4f", t / (ms/1000.0) }')"
      latency_ms="$wall_ms"
    fi
    gen_ms="$latency_ms"
  else
    warn "juno chat failed http=${http_code} rc=${rc} — see ${logf} ${resp}"
    rc=1
  fi

  # Close the window before the engine goes away: the recording has to be dumped
  # from a live process, and it must not pick up shutdown work either.
  if [[ "${USE_JFR_THIS_REP:-0}" -eq 1 ]]; then
    # Stop may legitimately fail when the bound above already stopped and dumped
    # the recording, so the dump file, not the exit code, is what decides.
    jcmd "$JUNO_PID" JFR.stop name="$JFR_RECORDING_NAME" >>"$logf" 2>&1 || true
    if [[ -s "$jfr_recording" ]]; then
      if extract_jfr_metrics "$jfr_recording" "$jfr_metrics" "$stem" "$(basename "$model_path")" "$logf"; then
        jfr_block="$(jfr_summary_json "$jfr_metrics" "$prompt_tokens" "$completion_tokens" "${gen_ms:-0}")"
        jfr_pp_tps="$(jq -r '.prompt_eval_tps // empty' <<<"$jfr_block" 2>/dev/null || true)"
        jfr_tg_tps="$(jq -r '.token_gen_tps // empty' <<<"$jfr_block" 2>/dev/null || true)"
        span_reason="$(jq -r '.span_check.reason // empty' <<<"$jfr_block" 2>/dev/null || true)"
        if [[ -n "$span_reason" ]]; then
          warn "${rep_label}: ${span_reason}"
        fi
        if [[ "$(jq -r '.span_check.timestamps_consistent' <<<"$jfr_block" 2>/dev/null)" == "false" ]]; then
          span_withheld=1
        fi
      else
        warn "could not extract metrics from ${jfr_recording} — see ${logf}"
      fi
    else
      warn "the measurement recording produced no dump — see ${logf}"
    fi
  fi

  stop_juno
  sleep 2

  # Prefer JFR tg for compare when available; keep API tg separately. A zero is not
  # a slow reading, it is the absence of one: a model that emitted a stop token
  # immediately generated nothing to time.
  local compare_pp="$jfr_pp_tps" compare_tg="$tps"
  if [[ -n "$jfr_tg_tps" && "$jfr_tg_tps" != "null" && "$jfr_tg_tps" != "0" ]]; then
    compare_tg="$jfr_tg_tps"
  fi
  if [[ "${completion_tokens:-0}" -eq 0 ]]; then
    compare_tg=null
    tps=null
  fi
  # Spans that failed the timestamp check withhold the generation reading too, rather
  # than letting it fall back to the API figure, which is a different measurement.
  (( span_withheld == 1 )) && compare_tg=null
  [[ -z "$compare_pp" ]] && compare_pp=null
  [[ -z "$compare_tg" ]] && compare_tg=null
  [[ -z "$tps" ]] && tps=null
  [[ -z "${gen_ms:-}" ]] && gen_ms=0

  jq -n \
    --arg engine "juno" \
    --arg tool "local-api+jfr" \
    --arg status "$([ "$rc" -eq 0 ] && echo success || echo failure)" \
    --argjson exit_code "$rc" \
    --arg http_code "$http_code" \
    --arg model "$(basename "$model_path")" \
    --arg model_path "$model_path" \
    --arg model_id "$model_id" \
    --argjson n_gen "$lane_max_tokens" \
    --argjson prompt_tokens "${prompt_tokens:-0}" \
    --argjson completion_tokens "${completion_tokens:-0}" \
    --arg finish_reason "${finish_reason:-}" \
    --argjson load_ms "$load_ms" \
    --argjson latency_ms "${gen_ms:-0}" \
    --argjson wall_ms "$wall_ms" \
    --argjson api_token_gen_tps "${tps:-null}" \
    --argjson prompt_eval_tps "${compare_pp:-null}" \
    --argjson token_gen_tps "${compare_tg:-null}" \
    --argjson juno_use_vector "$JUNO_USE_VECTOR" \
    --arg backend "$(backend_label)" \
    --argjson use_gpu "$USE_GPU" \
    --arg heap "$heap" \
    --arg heap_source "$heap_source" \
    --argjson threads_hint "$N_THREADS" \
    --argjson use_jfr "$USE_JFR" \
    --argjson jfr_scoped_to_measured_request "${USE_JFR_THIS_REP:-0}" \
    --argjson juno_warmup "$JUNO_WARMUP" \
    --argjson calibrated_prompt_words "${calibrated_words:-0}" \
    --arg rep_label "$rep_label" \
    --arg lane "$lane" \
    --argjson n_prompt "$lane_n_prompt" \
    --argjson prompt_parity_tolerance "$PROMPT_PARITY_TOLERANCE" \
    --arg jfr_duration "$JFR_DURATION" \
    --arg response_json "$resp" \
    --arg log "$logf" \
    --arg gpu_attention_requested "${JUNO_GPU_ATTENTION:-engine default}" \
    --arg gpu_attention_resolved "$(gpu_attention_resolved_from_log "$logf" "$USE_GPU" "$jfr_metrics")" \
    --argjson jfr "$jfr_block" \
    --argjson host "$(host_meta_json)" \
    '{
      engine: $engine,
      tool: (if $use_jfr == 1 then $tool else "local-api" end),
      status: $status,
      exit_code: $exit_code,
      http_code: $http_code,
      model: $model,
      model_path: $model_path,
      model_id: $model_id,
      n_gen: $n_gen,
      n_prompt: $n_prompt,
      prompt_tokens: $prompt_tokens,
      # How far the real Juno prefill was from what the reference tool was asked
      # for. A prefill ratio is published only while this stays within tolerance.
      prompt_token_deviation:
        (if $n_prompt > 0 and $prompt_tokens != null
         then (($prompt_tokens - $n_prompt) / $n_prompt | fabs)
         else null end),
      prompt_parity_tolerance: $prompt_parity_tolerance,
      completion_tokens: $completion_tokens,
      finish_reason: $finish_reason,
      load_ms: $load_ms,
      latency_ms: $latency_ms,
      wall_ms: $wall_ms,
      api_token_gen_tps: $api_token_gen_tps,
      prompt_eval_tps: $prompt_eval_tps,
      token_gen_tps: $token_gen_tps,
      juno_use_vector: $juno_use_vector,
      # The flag as passed (engine default when none was), and what the engine
      # log shows it ran: see gpu_attention_resolved_from_log.
      gpu_attention_requested: $gpu_attention_requested,
      gpu_attention_resolved: $gpu_attention_resolved,
      backend: $backend,
      use_gpu: $use_gpu,
      heap: $heap,
      # fixed (a per-model table), explicit (COMPARE_HEAP) or derived (file size).
      # A derived heap is not comparable with a baseline taken at a fixed one.
      heap_source: $heap_source,
      threads_hint: $threads_hint,
      use_jfr: $use_jfr,
      rep_label: $rep_label,
      # prefill or generate: which of the two figures this record measured.
      lane: $lane,
      juno_warmup: $juno_warmup,
      # Words in the measured prompt after calibrating against the real token
      # count, or 0 when the prompt was not calibrated.
      calibrated_prompt_words: $calibrated_prompt_words,
      # 1 when the recording for this rep held only the measured request, which is
      # what makes the warmup requests above safe to issue into the same process.
      jfr_scoped_to_measured_request: $jfr_scoped_to_measured_request,
      jfr_duration: $jfr_duration,
      jfr: (if $use_jfr == 1 then $jfr else null end),
      response_json: $response_json,
      log: $log,
      host: $host
    }' >"$out_json"

  if [[ "$rc" -ne 0 ]]; then
    return 1
  fi
  log "juno result: ${out_json}  (jfr pp=${compare_pp:-?} tg=${compare_tg:-?} t/s; api tg=${tps:-?} t/s)"
  return 0
}

# Every measured cycle for one lane, then the median of them.
#
# A rep is a whole cycle rather than another request in the same process: model
# load, page-cache state and device residency all sit inside a cycle, and a
# repetition that reused them would only re-measure the cheapest part of the run.
# This matches the repetition discipline the LoRA comparison already follows.
# Every measured cycle of one lane, then the median of them.
run_juno_lane() {
  local model_path="$1" stem="$2" lane="$3" out_json="$4"
  local -a rep_jsons=()
  local i rc=0 rep_label

  for (( i = 1; i <= JUNO_REPS; i++ )); do
    rep_label="${stem}-${lane}-rep${i}"
    log "=== juno ${lane} lane, rep ${i}/${JUNO_REPS}: ${stem} ==="
    run_juno_rep "$model_path" "$stem" "$rep_label" "$lane" || rc=1
    rep_jsons+=("${OUT_ROOT}/${rep_label}-juno.json")
  done

  [[ "$DRY_RUN" -eq 1 ]] && return 0

  local -a present=()
  for i in "${!rep_jsons[@]}"; do
    [[ -f "${rep_jsons[$i]}" ]] && present+=("${rep_jsons[$i]}")
  done
  if (( ${#present[@]} == 0 )); then
    warn "no juno ${lane} result produced for ${stem}"
    return 1
  fi

  aggregate_juno_reps_json "$out_json" "${present[@]}"
  # The representative recording is the first rep's, so a reader following
  # result_files lands on a real measurement rather than on an aggregate.
  if [[ -f "${OUT_ROOT}/${stem}-${lane}-rep1-juno-jfr.json" ]]; then
    cp -a "${OUT_ROOT}/${stem}-${lane}-rep1-juno-jfr.json" "${OUT_ROOT}/${stem}-juno-jfr.json"
  fi
  log "aggregated ${#present[@]} ${lane} rep(s) (median, spread recorded) -> ${out_json}"
  return "$rc"
}

# Both Juno lanes for one model, then the one record the rest of the script reads.
#
# A cycle is a whole engine start rather than another request in the same process,
# because model load, page-cache state and device residency all sit inside one; and
# there are two lanes because the reference tool measures prefill and generation as
# two benchmarks and a single request cannot reproduce that (see
# merge_juno_lanes_json).
run_juno() {
  local model_path="$1" stem="$2"
  local out_json="${OUT_ROOT}/${stem}-juno.json"
  local prefill_json="${OUT_ROOT}/${stem}-juno-prefill.json"
  local generate_json="${OUT_ROOT}/${stem}-juno-generate.json"
  local rc=0

  run_juno_lane "$model_path" "$stem" prefill "$prefill_json" || rc=1
  run_juno_lane "$model_path" "$stem" generate "$generate_json" || rc=1

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: would merge the prefill and generate lanes into ${out_json}"
    return 0
  fi

  if [[ ! -f "$prefill_json" || ! -f "$generate_json" ]]; then
    warn "a juno lane produced no result for ${stem}; cannot merge"
    return 1
  fi

  merge_juno_lanes_json "$out_json" "$prefill_json" "$generate_json"
  log "merged prefill and generate lanes -> ${out_json}"
  return "$rc"
}

# Collapses the per-rep Juno results into the one result the rest of the script
# reads, publishing the median and recording the spread beside it.
#
# The median, not the mean: this host's own measurement floor moved a reference
# reading by 14% between two runs eight minutes apart, and a mean lets one such
# run drag the published number with it. The spread is kept because a median with
# no min/max cannot be read against that floor — a 5% difference between two
# medians whose reps span 30% is not a result.
#
# Two fields are deliberately not medianed. Collection-pause maximum and
# allocation per token are noise indicators, and the re-run rule fires on the
# worst single pause in a run, so the aggregate keeps the worst reading across
# reps; a median would hide precisely the outlier the rule exists to catch.
aggregate_juno_reps_json() {
  local out_json="$1"
  shift
  local -a rep_jsons=("$@")

  jq -s --argjson reps "${#rep_jsons[@]}" --argjson warmup "$JUNO_WARMUP" '
    def median: sort as $s | ($s | length) as $n |
      if $n == 0 then null
      elif ($n % 2) == 1 then $s[($n - 1) / 2]
      else (($s[$n / 2 - 1] + $s[$n / 2]) / 2)
      end;
    # A rep that produced no reading contributes nothing rather than a zero: a
    # withheld or failed reading averaged in as 0 publishes a number nobody measured.
    def spread(f): (map(f)) as $all | ($all | map(select(. != null))) as $v |
      { median: ($v | median), min: ($v | min), max: ($v | max), values: $all };

    . as $all_reps |
    (spread(.prompt_eval_tps)) as $pp |
    (spread(.token_gen_tps)) as $tg |
    (spread(.api_token_gen_tps)) as $api_tg |
    (spread(.latency_ms)) as $latency |
    (spread(.jfr.gc_pause_max_ms)) as $gc |
    (spread(.jfr.gc_pause_max_ms_in_token_span)) as $gc_in_span |
    (spread(.jfr.allocated_bytes_per_token)) as $alloc |
    (map(.status)) as $rep_status |
    # The record everything else is grafted onto is the first rep that succeeded.
    # A rep that died before its first request writes only a stub, and building the
    # aggregate on that would drop the model id, the recording and the flags from a
    # result whose other reps measured fine.
    (((map(select(.status == "success")) | first) // .[0])
      | .prompt_eval_tps = $pp.median
      | .token_gen_tps = $tg.median
      | .api_token_gen_tps = $api_tg.median
      | .latency_ms = $latency.median
      | .juno_reps = $reps
      | .juno_warmup = $warmup
      | .rep_status = $rep_status
      | .gpu_attention_resolved = ($all_reps | map(.gpu_attention_resolved) | unique
          | if length == 1 then .[0] else "mixed" end)
      | .reps = { prompt_eval_tps: $pp, token_gen_tps: $tg, api_token_gen_tps: $api_tg,
                  latency_ms: $latency, gc_pause_max_ms: $gc, allocated_bytes_per_token: $alloc }
      | (if .jfr != null then .jfr.gc_pause_max_ms = $gc.max else . end)
      | (if .jfr != null then .jfr.gc_pause_max_ms_in_token_span = $gc_in_span.max else . end)
      | (if .jfr != null then .jfr.allocated_bytes_per_token = $alloc.max else . end)
      | .status = (if ($rep_status | all(. == "success")) then "success" else "failure" end))
  ' "${rep_jsons[@]}" >"$out_json"
}

# Combines the two Juno measurement lanes into the one record the rest of the
# script reads: prefill from the lane that measured prefill, generation from the
# lane that measured generation.
#
# The reference tool runs these as two benchmarks — prompt tokens with no
# generation, then generation from an empty context — and measuring both in one
# Juno request cannot reproduce that. A single request generates immediately after
# its own prefill, so its generation figure is taken at the prompt length while the
# reference takes it at nearly zero, and decode cost grows with context depth. That
# understated Juno generation once prompt-token parity raised the prompt to the
# requested length. Two lanes, one figure each, is the only shape that compares
# like with like.
merge_juno_lanes_json() {
  local out_json="$1" prefill_json="$2" generate_json="$3"

  jq -n --slurpfile p "$prefill_json" --slurpfile g "$generate_json" '
    ($p[0]) as $pre | ($g[0]) as $gen |
    $gen
    # Generation is the headline figure, so the generation lane supplies the base
    # record and the prefill lane contributes the one number it measured.
    | .prompt_eval_tps = $pre.prompt_eval_tps
    # Prompt parity is a statement about the prefill lane, which is the only lane
    # asked for a particular prompt length.
    | .prompt_tokens = $pre.prompt_tokens
    | .n_prompt = $pre.n_prompt
    | .prompt_token_deviation = $pre.prompt_token_deviation
    | .prompt_parity_tolerance = ($pre.prompt_parity_tolerance // 0.10)
    | .gpu_attention_resolved = (if $pre.gpu_attention_resolved == $gen.gpu_attention_resolved
                                 then $gen.gpu_attention_resolved else "mixed" end)
    | .reps = { prompt_eval_tps: ($pre.reps.prompt_eval_tps // null),
                token_gen_tps: ($gen.reps.token_gen_tps // null),
                api_token_gen_tps: ($gen.reps.api_token_gen_tps // null),
                latency_ms: ($gen.reps.latency_ms // null),
                gc_pause_max_ms: ($gen.reps.gc_pause_max_ms // null),
                allocated_bytes_per_token: ($gen.reps.allocated_bytes_per_token // null) }
    # A pause in either lane disqualifies the row, so the worse of the two stands.
    | (if .jfr != null then
         .jfr.gc_pause_max_ms = ([($pre.jfr.gc_pause_max_ms // 0), ($gen.jfr.gc_pause_max_ms // 0)] | max)
       else . end)
    # The generation lane is the one whose token span the in-span figure describes.
    | (if .jfr != null then
         .jfr.gc_pause_max_ms_in_token_span = ($gen.jfr.gc_pause_max_ms_in_token_span // null)
       else . end)
    | .lanes = { prefill: { prompt_tokens: $pre.prompt_tokens, n_prompt: $pre.n_prompt,
                            n_gen: $pre.n_gen, completion_tokens: $pre.completion_tokens,
                            prompt_eval_tps: $pre.prompt_eval_tps, latency_ms: $pre.latency_ms,
                            status: $pre.status, gc_pause_max_ms: ($pre.jfr.gc_pause_max_ms // null),
                            gpu_attention_resolved: ($pre.gpu_attention_resolved // null) },
                 generate: { prompt_tokens: $gen.prompt_tokens, n_prompt: $gen.n_prompt,
                             n_gen: $gen.n_gen, completion_tokens: $gen.completion_tokens,
                             token_gen_tps: $gen.token_gen_tps, latency_ms: $gen.latency_ms,
                             status: $gen.status, gc_pause_max_ms: ($gen.jfr.gc_pause_max_ms // null),
                             gpu_attention_resolved: ($gen.gpu_attention_resolved // null) } }
    | .status = (if ($pre.status == "success" and $gen.status == "success") then "success" else "failure" end)
  ' >"$out_json"
}

run_tuned_lane() {
  # Default-flags (--mmq off, --gpu-attention unset so the engine default, --gpu-layers unset) lanes
  # sit well below what Juno's own shipped auto modes already do (see
  # docs/infra-plan/PLAN-Infra-Review-Fixes.md item 8 and
  # docs/infra-plan/PLAN-Infra-Tier18.md) — a config nobody would actually run
  # in production. Add a second lane per model, using the three flags' own
  # auto resolution together, so the standing GPU regression sweep measures
  # both, not just the unrepresentative default. auto falls back to off/serial
  # per-architecture where a flag is not wired (e.g. --gpu-attention on Phi-3)
  # — this is reported, not silently implied as "fully tuned" for every model.
  local model_path="$1" stem="$2"
  local tuned_stem="${stem}-tuned"
  local saved_mmq="$JUNO_MMQ" saved_gpu_attention="$JUNO_GPU_ATTENTION" saved_gpu_layers="$JUNO_GPU_LAYERS"

  JUNO_MMQ="auto"
  JUNO_GPU_ATTENTION="auto"
  JUNO_GPU_LAYERS="auto"
  log "=== tuned lane: ${stem} (--mmq auto --gpu-attention auto --gpu-layers auto) ==="
  run_juno "$model_path" "$tuned_stem"
  local rc=$?
  JUNO_MMQ="$saved_mmq"
  JUNO_GPU_ATTENTION="$saved_gpu_attention"
  JUNO_GPU_LAYERS="$saved_gpu_layers"

  # Reuse the vanilla lane's llama.cpp reference numbers — the reference engine
  # doesn't read Juno's flags, so re-running llama-bench would just add noise.
  if [[ -f "${OUT_ROOT}/${stem}-llama-cpp.json" ]]; then
    cp -a "${OUT_ROOT}/${stem}-llama-cpp.json" "${OUT_ROOT}/${tuned_stem}-llama-cpp.json"
  fi
  write_pair_summary "$tuned_stem"
  STEMS+=("$tuned_stem")
  return "$rc"
}

write_pair_summary() {
  local stem="$1"
  local llama_f="${OUT_ROOT}/${stem}-llama-cpp.json"
  local juno_f="${OUT_ROOT}/${stem}-juno.json"
  local out="${OUT_ROOT}/${stem}-compare.json"
  [[ -f "$llama_f" && -f "$juno_f" ]] || return 0
  if ! command -v jq >/dev/null 2>&1; then
    return 0
  fi
  jq -n --slurpfile l "$llama_f" --slurpfile j "$juno_f" \
    --arg llama_result "$llama_f" --arg juno_result "$juno_f" \
    --argjson tolerance "${REP_SPREAD_TOLERANCE}" '
    def num(x): if x == null then null else x end;
    {
      model: (( $l[0].model ) // ( $j[0].model )),
      llama_cpp: {
        status: $l[0].status,
        prompt_eval_tps: num($l[0].prompt_eval_tps),
        token_gen_tps: num($l[0].token_gen_tps)
      },
      juno: {
        status: $j[0].status,
        prompt_eval_tps: num($j[0].prompt_eval_tps),
        token_gen_tps: num($j[0].token_gen_tps),
        api_token_gen_tps: num($j[0].api_token_gen_tps),
        latency_ms: $j[0].latency_ms,
        n_prompt: $j[0].n_prompt,
        prompt_tokens: $j[0].prompt_tokens,
        prompt_token_deviation: $j[0].prompt_token_deviation,
        completion_tokens: $j[0].completion_tokens,
        juno_use_vector: $j[0].juno_use_vector,
        use_jfr: $j[0].use_jfr,
        gc_pause_count: ($j[0].jfr.gc_pause_count // null),
        gc_pause_max_ms: ($j[0].jfr.gc_pause_max_ms // null),
        gc_pause_total_ms: ($j[0].jfr.gc_pause_total_ms // null),
        allocated_bytes_total: ($j[0].jfr.allocated_bytes_total // null),
        allocated_bytes_per_token: ($j[0].jfr.allocated_bytes_per_token // null),
        # The published readings are medians of this many cycles, each preceded by
        # this many discarded requests. The spread belongs next to the median: a
        # difference between two medians whose own reps span more than it is not a
        # result on this host.
        juno_reps: ($j[0].juno_reps // 1),
        lanes: ($j[0].lanes // null),
        juno_warmup: ($j[0].juno_warmup // 0),
        rep_status: ($j[0].rep_status // null),
        heap: $j[0].heap,
        heap_source: ($j[0].heap_source // null),
        reps: ($j[0].reps // null)
      },
      # A prefill ratio compares two engines only when both prefilled about the
      # same number of tokens. Where they did not, the number is withheld and the
      # reason is stated, rather than published as though it meant something.
      prompt_parity:
        (($j[0].prompt_token_deviation) as $dev
         | ($j[0].prompt_parity_tolerance // 0.10) as $tol
         | if $dev == null then { ok: false, reason: "juno prompt_tokens unknown" }
           elif $dev > $tol then
             { ok: false,
               deviation: $dev,
               tolerance: $tol,
               reason: ("juno prefilled \($j[0].prompt_tokens) tokens against n_prompt "
                        + "\($j[0].n_prompt): \(($dev * 100) | floor)% off, over the "
                        + "\(($tol * 100) | floor)% tolerance") }
           else { ok: true, deviation: $dev, tolerance: $tol } end),
      ratio_juno_over_llamacpp_pp:
        (($j[0].prompt_token_deviation) as $dev
         | ($j[0].prompt_parity_tolerance // 0.10) as $tol
         | if ($dev == null or $dev > $tol) then null
           elif ($l[0].prompt_eval_tps != null and $j[0].prompt_eval_tps != null and $l[0].prompt_eval_tps > 0)
           then ($j[0].prompt_eval_tps / $l[0].prompt_eval_tps)
           else null end),
      # Whether this run is fit to be scored at all, decided by whether its own
      # repetitions agree. That is the question the plan is really asking, and the
      # repetitions answer it directly instead of by proxy.
      #
      # The plan originally gated on the longest collection pause. That was tried and
      # withdrawn, because on this host the pause counter does not measure time the
      # application was stopped. Three rows of the reference sweep reported a pause
      # of about 635 ms and lost nothing: tinyllama-tuned produced 64 tokens across
      # token spans of 1110, 1120 and 1107 ms while its three pause readings were
      # 633, 5 and 4 ms. A 633 ms stop-the-world inside a 1110 ms span would have
      # left 477 ms for 64 tokens, which is over twice the rate the model reaches, so
      # the duration cannot be stop-the-world time. Gating on it rejected three rows
      # whose readings agreed to within 1%, and passed a row whose readings spanned
      # 31%. Dispersion gets both right, and a collection pause that does cost time
      # shows up in it anyway, as one slow repetition.
      #
      # Both pause figures are still published, including the share of them that fell
      # inside the token span, because they are worth reading even when they are not
      # worth gating on.
      noise:
        (($j[0].reps.token_gen_tps) as $tg
         | ($j[0].jfr.gc_pause_max_ms // null) as $gc_max
         | (if ($tg != null and $tg.median != null and $tg.median > 0
                and $tg.min != null and $tg.max != null)
            then (($tg.max - $tg.min) / $tg.median) else null end) as $spread
         | if $spread == null then
             { ok: true, basis: "not assessable",
               reason: "a single reading offers no dispersion evidence either way",
               rep_spread: null, rep_spread_tolerance: $tolerance,
               gc_pause_max_ms: $gc_max,
               gc_pause_max_ms_in_token_span: ($j[0].jfr.gc_pause_max_ms_in_token_span // null) }
           else
             { ok: ($spread <= $tolerance),
               basis: "repetition dispersion",
               condition: (if $spread > $tolerance then "rep_spread" else null end),
               rep_spread: $spread,
               rep_spread_tolerance: $tolerance,
               gc_pause_max_ms: $gc_max,
               gc_pause_max_ms_in_token_span: ($j[0].jfr.gc_pause_max_ms_in_token_span // null),
               reason: (if $spread > $tolerance
                        then ("the \($j[0].juno_reps // 0) generation readings span "
                              + "\(($spread * 100) | floor)% of their median, over the "
                              + "\(($tolerance * 100) | floor)% this host can resolve: re-run "
                              + "rather than score this row")
                        else null end) }
           end),
      # The reference tool generates n_gen tokens whatever the model would rather do;
      # Juno stops at a stop token. Where Juno generated nothing there is no reading
      # to publish, and a 0 would read as "infinitely slower" instead of "unmeasured".
      # Where it generated fewer tokens than asked, the ratio is published and the
      # shortfall stated: decode cost per token is roughly steady, so the reading is
      # comparable, but it is an average taken over a shorter and slightly cheaper
      # span of context.
      generation_parity:
        (($j[0].completion_tokens // 0) as $ct | ($j[0].n_gen // 0) as $want
         | if $ct == 0 then
             { ok: false, juno_completion_tokens: $ct, requested: $want,
               reason: ("juno generated no tokens (finish_reason "
                        + ($j[0].finish_reason // "unknown")
                        + "): there is no generation reading to compare") }
           elif $ct < $want then
             { ok: true, juno_completion_tokens: $ct, requested: $want,
               note: ("juno stopped after \($ct) of \($want) tokens; the reference tool "
                      + "generated \($want) regardless, so this ratio compares averages "
                      + "taken over different spans") }
           else { ok: true, juno_completion_tokens: $ct, requested: $want } end),
      ratio_juno_over_llamacpp_tg:
        (if (($j[0].completion_tokens // 0) == 0) then null
         elif ($l[0].token_gen_tps != null and $j[0].token_gen_tps != null and $l[0].token_gen_tps > 0)
         then ($j[0].token_gen_tps / $l[0].token_gen_tps)
         else null end),
      result_files: {
        llama_cpp_raw: ($l[0].raw_json | values),
        juno_response: ($j[0].response_json | values),
        llama_result: $llama_result,
        juno_result: $juno_result,
        juno_jfr: (if ($j[0].use_jfr // false) then ($juno_result | sub("-juno\\.json$"; "-juno-jfr.json")) else null end)
      }
    }
  ' >"$out" || { warn "compare json failed for ${stem}"; return 0; }
  log "compare: ${out}"
}

write_run_index() {
  local index="${OUT_ROOT}/INDEX.md"
  local juno_flag="--cpu"
  [[ "$USE_GPU" -eq 1 ]] && juno_flag="--gpu"
  {
    echo "# llama.cpp vs Juno - ${RUN_ID} ($(backend_label))"
    echo
    echo "| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |"
    echo "|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|"
    local stem llama_f juno_f cmp pp tg jpp jt ratio_pp ratio_tg ptok gcmax allocpt tgspread gtok noise
    local -a parity_notes=()
    local -a gen_notes=()
    local -a noise_notes=()
    local -a withheld_notes=()
    local repf wreason
    for stem in "${STEMS[@]}"; do
      for repf in "${OUT_ROOT}/${stem}"-*rep*-juno.json; do
        [[ -f "$repf" ]] || continue
        wreason="$(jq -r '.jfr.span_check.reason // empty' "$repf" 2>/dev/null || true)"
        [[ -n "$wreason" ]] && withheld_notes+=("$(basename "$repf" -juno.json): ${wreason}")
      done
      llama_f="${OUT_ROOT}/${stem}-llama-cpp.json"
      juno_f="${OUT_ROOT}/${stem}-juno.json"
      cmp="${OUT_ROOT}/${stem}-compare.json"
      pp="$(jq -r 'if .prompt_eval_tps == null then "-" else .prompt_eval_tps end' "$llama_f" 2>/dev/null || echo -)"
      tg="$(jq -r 'if .token_gen_tps == null then "-" else .token_gen_tps end' "$llama_f" 2>/dev/null || echo -)"
      jpp="$(jq -r 'if .prompt_eval_tps == null then "-" else .prompt_eval_tps end' "$juno_f" 2>/dev/null || echo -)"
      jt="$(jq -r 'if .token_gen_tps == null then "-" else .token_gen_tps end' "$juno_f" 2>/dev/null || echo -)"
      ptok="$(jq -r '[(.prompt_tokens // "?"), "/", (.n_prompt // "?")] | map(tostring) | join("")' "$juno_f" 2>/dev/null || echo -)"
      gtok="$(jq -r '[(.completion_tokens // "?"), "/", (.n_gen // "?")] | map(tostring) | join("")' "$juno_f" 2>/dev/null || echo -)"
      gcmax="$(jq -r 'if (.jfr.gc_pause_max_ms // null) == null then "-" else (.jfr.gc_pause_max_ms | floor) end' "$juno_f" 2>/dev/null || echo -)"
      tgspread="$(jq -r 'if (.reps.token_gen_tps // null) == null then "-"
                         elif (.reps.token_gen_tps.min == null) then "-"
                         else ((.reps.token_gen_tps.min * 100 | round / 100 | tostring) + " / "
                               + (.reps.token_gen_tps.max * 100 | round / 100 | tostring)) end' \
                 "$juno_f" 2>/dev/null || echo -)"
      allocpt="$(jq -r 'if (.jfr.allocated_bytes_per_token // null) == null then "-" else (.jfr.allocated_bytes_per_token | floor) end' "$juno_f" 2>/dev/null || echo -)"
      if [[ -f "$cmp" ]]; then
        ratio_pp="$(jq -r 'if .ratio_juno_over_llamacpp_pp == null then "withheld" else .ratio_juno_over_llamacpp_pp end' "$cmp" 2>/dev/null || echo -)"
        ratio_tg="$(jq -r 'if .ratio_juno_over_llamacpp_tg == null then
                              (if (.generation_parity.ok == false) then "withheld" else "-" end)
                            else .ratio_juno_over_llamacpp_tg end' "$cmp" 2>/dev/null || echo -)"
        if [[ "$(jq -r 'if .prompt_parity.ok == false then "false" else "true" end' "$cmp" 2>/dev/null || echo true)" == "false" ]]; then
          parity_notes+=("${stem}: $(jq -r '.prompt_parity.reason // "prompt parity not established"' "$cmp")")
        fi
        if [[ "$(jq -r 'if .generation_parity.ok == false then "false" else "true" end' "$cmp" 2>/dev/null || echo true)" == "false" ]]; then
          gen_notes+=("${stem}: $(jq -r '.generation_parity.reason // "no generation reading"' "$cmp")")
        elif [[ -n "$(jq -r '.generation_parity.note // empty' "$cmp" 2>/dev/null || true)" ]]; then
          gen_notes+=("${stem}: $(jq -r '.generation_parity.note' "$cmp")")
        fi
        if [[ "$(jq -r 'if .noise.ok == false then "false" else "true" end' "$cmp" 2>/dev/null || echo true)" == "false" ]]; then
          noise="NOISY"
          noise_notes+=("${stem}: $(jq -r '.noise.reason // "noise condition fired"' "$cmp")")
        else
          noise="yes"
        fi
      else
        ratio_pp="-"
        ratio_tg="-"
        noise="-"
      fi
      printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s-*.json |\n' \
        "$stem" "$pp" "$tg" "$jpp" "$jt" "$tgspread" "$ratio_pp" "$ratio_tg" "$ptok" "$gtok" "$gcmax" "$allocpt" \
        "$noise" "$stem"
    done
    echo
    echo "GPU attention per row, as the engine resolved it (read from its log, not from the flag):"
    for stem in "${STEMS[@]}"; do
      juno_f="${OUT_ROOT}/${stem}-juno.json"
      [[ -f "$juno_f" ]] || continue
      echo "- ${stem}: $(jq -r '"\(.gpu_attention_resolved // "not recorded") (requested: \(.gpu_attention_requested // "not recorded"))"' "$juno_f" 2>/dev/null || echo "not recorded")"
    done
    echo
    echo "Host meta: see any *-llama-cpp.json .host field."
    echo
    echo "Notes:"
    echo "- llama.cpp metrics from llama-bench (avg_ts)."
    echo "- Prompt tokens column is Juno actual / requested. A prefill ratio is published"
    echo "  only when the two are within $(awk -v t="$PROMPT_PARITY_TOLERANCE" 'BEGIN{printf "%d", t*100}')%; otherwise it reads \`withheld\`, because the two"
    echo "  engines then prefilled measurably different amounts of work."
    if (( ${#parity_notes[@]} > 0 )); then
      echo "- Prefill ratios withheld this run:"
      local note
      for note in "${parity_notes[@]}"; do
        echo "  - ${note}"
      done
    fi
    echo "- Prefill and generation are measured in two separate Juno runs per model, mirroring the"
    echo "  reference tool, which benchmarks prompt processing and generation separately. The prefill"
    echo "  run is given the requested prompt length and asked for one token; the generation run is"
    echo "  given the shortest prompt the chat template allows and asked for the full count, so it"
    echo "  decodes at a shallow context like the reference does. Measuring both in one request would"
    echo "  time Juno generation at the prompt length while the reference times it near zero, and"
    echo "  decode slows as context grows."
    if (( ${#STEMS[@]} > 0 )) && [[ -f "${OUT_ROOT}/${STEMS[0]}-compare.json" ]]; then
      local gen_ctx
      gen_ctx="$(jq -r '.juno.lanes.generate.prompt_tokens // "?"' "${OUT_ROOT}/${STEMS[0]}-compare.json" 2>/dev/null || echo "?")"
      echo "  Residual: Juno cannot reach an empty context, because the chat template wraps every"
      echo "  request; the generation run prefilled ${gen_ctx} tokens against the reference 0."
    fi
    echo "- Generated-token column is Juno actual / requested. The reference tool generates the"
    echo "  requested count whatever the model would rather do, while Juno stops at a stop token."
    echo "  Where Juno generated nothing there is no reading and the ratio reads \`-\`; where it"
    echo "  generated fewer tokens, the ratio is published and the shortfall noted, since it is then"
    echo "  an average over a shorter and slightly cheaper span of context."
    if (( ${#gen_notes[@]} > 0 )); then
      echo "- Generation notes this run:"
      local gnote
      for gnote in "${gen_notes[@]}"; do
        echo "  - ${gnote}"
      done
    fi
    echo "- Juno readings are the median of ${JUNO_REPS} measured cycle(s), each preceded by"
    echo "  ${JUNO_WARMUP} discarded request(s) so the measured request runs on compiled code. The"
    echo "  min/max column is that median's own spread; a difference smaller than the spread"
    echo "  is not a result. Each cycle records only its measured request: the recording is"
    echo "  started after the warmup requests return and stopped before the engine exits."
    echo "  The last warmup ran under a discarded recording with the same settings, so the"
    echo "  measured request does not pay for the first recording's recompilation."
    echo "- Build: Juno \`$(juno_commit)\`$([[ "$(juno_tree_dirty)" == true ]] && echo ' plus uncommitted changes'), jar sha256 \`$(juno_jar_sha256)\`,"
    echo "  $(java_version_line); JVM ${JUNO_JVM_FLAGS[*]}, -Xms equal to -Xmx per model."
    echo "  Reference tool build \`$(llama_build_commit)\` from ${LLAMA_CPP_BIN:-?}; GPU driver $(gpu_driver_version)."
    echo "  A different reference build is a measurement boundary: its ratios are not comparable with this run."
    if [[ "$CLOCK_PIN_STATE" == pinned:* ]]; then
      echo "- Clocks ${CLOCK_PIN_STATE}."
    else
      echo "- Clocks not pinned (${CLOCK_PIN_STATE}). This run can be read against the 15% noise floor"
      echo "  only; it is not usable for a gate tighter than that (see --pin-clocks)."
    fi
    echo "- Clock state this run: CPU governor $(cpu_governor_state), turbo $(cpu_turbo_state),"
    echo "  GPU clocks $(gpu_clock_json). A throttled run and a regression look the same"
    echo "  without this."
    if (( JUNO_THREADS == N_THREADS )); then
      echo "- Thread counts are matched: the reference tool ran with -t ${N_THREADS}, and Juno's CPU kernels"
      echo "  ran on ${JUNO_THREADS} threads (common pool parallelism ${JUNO_POOL_PARALLELISM} plus the calling thread)."
    else
      echo "- Thread counts are not matched: the reference tool ran with -t ${N_THREADS}, while Juno's CPU"
      echo "  kernels ran on ${JUNO_THREADS} threads (common pool parallelism ${JUNO_POOL_PARALLELISM} plus the calling"
      echo "  thread; the pool needs at least one worker)."
    fi
    if (( ${#noise_notes[@]} > 0 )); then
      echo "- Rows marked NOISY are not scorable and should be re-run:"
      local nnote
      for nnote in "${noise_notes[@]}"; do
        echo "  - ${nnote}"
      done
    fi
    if (( ${#withheld_notes[@]} > 0 )); then
      echo "- Repetitions whose readings were withheld (the median is taken over the rest):"
      local wnote
      for wnote in "${withheld_notes[@]}"; do
        echo "  - ${wnote}"
      done
    fi
    echo "- JFR timestamps: $(perf_jfr_clock_label). Every repetition's spans are checked against"
    echo "  the engine's own request latency and its prefill against its prompt; a failing"
    echo "  repetition is withheld and listed above."
    echo "- The Scorable column asks whether a row's own repetitions agree: a generation reading"
    echo "  whose cycles span more than $(awk -v t="$REP_SPREAD_TOLERANCE" 'BEGIN{printf "%d", t*100}')% of their median is not stable at the resolution a"
    echo "  gate would read it at, and should be re-run rather than scored. Collection pauses and"
    echo "  lock/park totals are recorded in every result JSON but are not gated on. Neither"
    echo "  measures lost time reliably here: the park figure sums every thread, so an idle worker"
    echo "  pool exceeds wall time on a healthy run, and the ~635 ms pauses that runs before"
    echo "  2026-09-28 reported on rows that lost no time were JFR timestamps misread across"
    echo "  CPU cores whose counters disagree, not pauses. A pause that does cost time appears in"
    echo "  the dispersion anyway."
    echo "- GC max ms and Alloc B/tok come from the recording taken alongside each run. A"
    echo "  result whose GC max is a large fraction of its measurement window should be"
    echo "  re-run rather than scored: one long pause looks exactly like a regression."
    if [[ "$USE_JFR" -eq 1 ]]; then
      echo "- Juno pp/tg from JFR (--jfr ${JFR_DURATION}): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms)."
    else
      echo "- Juno tg from POST /v1/chat/completions (x_juno_latency_ms / completion_tokens); JFR disabled (--no-jfr)."
    fi
    echo "- Backend: $(backend_label) (llama -ngl ${NGL} / juno ${juno_flag}), temperature 0, max_tokens=${N_GEN}."
    if [[ "$JUNO_USE_VECTOR" == "0" ]]; then
      echo "- Juno ran without jdk.incubator.vector (scalar kernels)."
    else
      echo "- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0."
    fi
  } >"$index"
  log "index: ${index}"
}

publish_results() {
  [[ "$PUBLISH" -eq 1 ]] || { log "publish skipped (--no-publish)"; return 0; }
  [[ "$DRY_RUN" -eq 1 ]] && { log "dry-run: would publish ${OUT_ROOT} → ${DOCS_PUBLISH_ROOT}"; return 0; }
  [[ -d "$OUT_ROOT" ]] || return 0

  local docs_dest="${DOCS_PUBLISH_ROOT}/${RUN_ID}"
  mkdir -p "$docs_dest"

  # Metrics only (skip large logs / response dumps).
  local f
  for f in "$OUT_ROOT"/INDEX.md "$OUT_ROOT"/host.json \
           "$OUT_ROOT"/*-llama-cpp.json "$OUT_ROOT"/*-juno.json "$OUT_ROOT"/*-juno-jfr.json "$OUT_ROOT"/*-compare.json; do
    [[ -e "$f" ]] || continue
    cp -a "$f" "$docs_dest/"
  done

  log "published: ${docs_dest}"
  if [[ -f "${DOCS_PUBLISH_ROOT}/README.md" ]]; then
    log "summary doc: ${DOCS_PUBLISH_ROOT}/README.md (update manually for narrative baselines)"
  fi
}

# ── main ─────────────────────────────────────────────────────────────────────

require_cmd jq
require_cmd curl
require_cmd awk
require_cmd date

if [[ "$SELFTEST_ONLY" -eq 1 ]]; then
  run_selftest
  exit 0
fi

# The harness now owns the recording, so the two things it needs for that are
# checked before a model is loaded rather than warned about per repetition. A run
# that cannot state which settings produced its recording must not be compared
# against one that can.
if [[ "$USE_JFR" -eq 1 ]]; then
  require_cmd jcmd
  [[ -f "$JFR_SETTINGS_FILE" ]] \
    || die "JFR settings not found: ${JFR_SETTINGS_FILE} (pass --no-jfr for an API-only measurement)"
  if (( DEVICE_SPANS )); then
    [[ -f "$JFR_SPANS_FILE" ]] || die "--device-spans: settings not found: ${JFR_SPANS_FILE}"
  fi
fi

resolve_llama_bin
select_models

if [[ "$RAW_PROMPT" -eq 1 ]]; then
  # The starting point only. Each rep then calibrates the word count against the
  # token count the engine reports, because the chat template adds tokens this
  # word count knows nothing about.
  PROMPT_TEXT="$(raw_prompt_of_words "$N_PROMPT")"
  log "raw-prompt: ${N_PROMPT}-word starting prompt, calibrated per rep to reach n_prompt=${N_PROMPT} prefill tokens"
fi

if [[ "$LIST_ONLY" -eq 1 ]]; then
  log "selected ${#SELECTED_MODELS[@]} model(s) (backend=$(backend_label)):"
  printf '  %s\n' "${SELECTED_MODELS[@]}"
  exit 0
fi

[[ -n "${LLAMA_CPP_BIN:-}" && -d "$LLAMA_CPP_BIN" ]] \
  || die "llama.cpp bin dir not found (set LLAMA_CPP_BIN or --llama-bin). Tried: ${LLAMA_CPP_BIN:-<empty>}"
[[ -x "${LLAMA_CPP_BIN}/llama-bench" ]] || die "llama-bench missing in ${LLAMA_CPP_BIN}"

setup_cuda_env
if [[ "$USE_GPU" -eq 1 ]]; then
  log "GPU mode: ngl=${NGL} juno=--gpu llama_bin=${LLAMA_CPP_BIN}"
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi failed — GPU run may fall back to CPU"
  else
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  fi
fi

mkdir -p "$OUT_ROOT"
trap 'stop_juno; restore_clocks' EXIT
pin_clocks
host_meta_json >"${OUT_ROOT}/host.json"
log "output: ${OUT_ROOT}"
[[ -n "$JUNO_JAR_EXPLICIT" ]] && log "juno jar (--juno-jar): $JUNO_JAR_EXPLICIT"
log "models: ${#SELECTED_MODELS[@]}"

STEMS=()
failures=0

for base in "${SELECTED_MODELS[@]}"; do
  model_path="${MODELS_DIR}/${base}"
  stem="${base%.gguf}"
  stem="${stem//\//_}"
  STEMS+=("$stem")
  log "=== model: ${base} ==="
  run_llama_bench "$model_path" "$stem" || failures=$((failures + 1))
  run_juno "$model_path" "$stem" || failures=$((failures + 1))
  write_pair_summary "$stem"
  if [[ "$USE_GPU" -eq 1 && "$TUNED_LANE" -eq 1 ]]; then
    run_tuned_lane "$model_path" "$stem" || failures=$((failures + 1))
  fi
done

# The reference build is only known once the reference tool has reported it.
if [[ -n "$(llama_build_commit)" ]]; then
  jq --arg c "$(llama_build_commit)" '.llama_build_commit = $c' "${OUT_ROOT}/host.json" >"${OUT_ROOT}/host.json.tmp" \
    && mv "${OUT_ROOT}/host.json.tmp" "${OUT_ROOT}/host.json"
fi
write_run_index
publish_results
log "done. failures=${failures}  results in ${OUT_ROOT}"
exit "$failures"
