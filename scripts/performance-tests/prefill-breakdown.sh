#!/usr/bin/env bash
# prefill-breakdown.sh — per-term breakdown of prefill wall time from a compare run.
#
# Reads the prefill repetitions of a compare-llama-cpp.sh run taken with
# --device-spans (*-prefill-rep*-juno-jfr.json) and splits juno.ForwardPass prefill
# time into terms that do not overlap, so the terms plus the residue add up to the
# whole window:
#
#   top-level spans inside the window   nested terms subtracted from their parent
#   ---------------------------------   -----------------------------------------
#   juno.WindowStep projection and      GEMM compute (juno.DeviceCompute gemm sites),
#   juno.WindowStep device_layer          device weight dequant, matmul and region
#   (the prefill-window device region)    staging copies, host FP16 packing, the
#                                         region's elementwise kernels (rms_norm,
#                                         convert_fp16, bias_add, rope, kv_append,
#                                         swiglu, residual_add) and its attention
#                                         (gqa_attention_region); what is left is the
#                                         host's side of both: dispatch, copy-out and
#                                         waiting (projection_and_region_host)
#   juno.WindowStep kv_write            KV mirror copies (memcpy_k/v_row, k/v_window);
#                                         what is left is the host KV write
#   juno.Attention                      gqa_attention compute and gqa_* copies
#   juno.RmsNorm, Rope, ResidualAdd, SwiGlu, WindowStep embed, bias_add, lm_head
#
# residue = ForwardPass prefill - every top-level term. A staging site that is not a
# gqa_*, a KV mirror copy or the host packing is counted under matmul staging; the
# per-site list in the JSON shows which sites that was. The host part of the
# projection spans was called projection_dispatch_and_copy_out before the device
# region existed; breakdowns published before 2026-10-02 carry that key.
#
# Each term is the median over the run's repetitions. A model whose residue is above
# --max-residue-pct (default 5) is flagged and the script exits 1.
#
# Usage:
#   ./scripts/performance-tests/prefill-breakdown.sh RUN_DIR [--lane default|tuned]
#        [--json OUT] [--max-residue-pct N]
#   ./scripts/performance-tests/prefill-breakdown.sh --selftest

set -euo pipefail

MAX_RESIDUE_PCT=5
LANE=default
JSON_OUT=""
RUN_DIR=""

# One repetition's metrics object in, its breakdown (ms) out.
JQ_TERMS='
def g($k): (.[$k] // 0);
def sites($ev; $suffix):
  to_entries
  | map(select(.key | test("^juno\\." + $ev + "\\.site\\..+\\.prefill\\." + $suffix + "$")))
  | map({site: (.key | capture("site\\.(?<s>.+)\\.prefill\\.").s), ms: .value});
. as $m
| (sites("DeviceStaging"; "estimated_total_ms")) as $staging
| (sites("DeviceCompute"; "total_ms")) as $compute
| "^(rms_norm|convert_fp16|bias_add|rope|kv_append|swiglu|residual_add)$" as $elementwise_sites
| ([$staging[] | select(.site | test("k_row|v_row|k_window|v_window")) | .ms] | add // 0) as $kv_copy
| ([$staging[] | select(.site | test("gqa")) | .ms] | add // 0) as $gqa_copy
| ([$staging[] | select(.site == "pack_fp16_host") | .ms] | add // 0) as $pack
| ([$staging[] | select(.site | test("k_row|v_row|k_window|v_window|gqa|pack_fp16_host") | not) | .ms] | add // 0) as $mm_copy
| ([$compute[] | select(.site == "gqa_attention") | .ms] | add // 0) as $gqa_compute
| ([$compute[] | select(.site == "gqa_attention_region") | .ms] | add // 0) as $region_attn
| ([$compute[] | select(.site | test($elementwise_sites)) | .ms] | add // 0) as $elementwise
| ([$compute[] | select(.site | test("^gqa_attention|" + $elementwise_sites) | not) | .ms] | add // 0) as $gemm
| ($m | g("juno.WeightDequant.device.total_ms")) as $dequant
| ($m | g("juno.WindowStep.projection.prefill.total_ms")
   + g("juno.WindowStep.device_layer.prefill.total_ms")) as $proj
| ($m | g("juno.WindowStep.kv_write.prefill.total_ms")) as $kv
| ($m | g("juno.Attention.prefill.total_ms")) as $attn
| ($m | g("juno.ForwardPass.prefill.total_ms")) as $fp
| {
    forward_pass_ms: $fp,
    terms: {
      gemm_compute: $gemm,
      weight_dequant: $dequant,
      matmul_staging: $mm_copy,
      host_fp16_pack: $pack,
      projection_and_region_host: ($proj - $gemm - $dequant - $mm_copy - $pack - $elementwise - $region_attn),
      device_elementwise: $elementwise,
      region_attention_compute: $region_attn,
      attention_compute: $gqa_compute,
      attention_copies: $gqa_copy,
      attention_host: ($attn - $gqa_compute - $gqa_copy),
      kv_mirror_copies: $kv_copy,
      kv_host_write: ($kv - $kv_copy),
      swiglu_host: ($m | g("juno.SwiGlu.prefill.total_ms")),
      rmsnorm_host: ($m | g("juno.RmsNorm.prefill.total_ms")),
      rope_host: ($m | g("juno.Rope.prefill.total_ms")),
      residual_add_host: ($m | g("juno.ResidualAdd.prefill.total_ms")),
      bias_add_host: ($m | g("juno.WindowStep.bias_add.prefill.total_ms")),
      embed: ($m | g("juno.WindowStep.embed.prefill.total_ms")),
      lm_head: ($m | g("juno.WindowStep.lm_head.prefill.total_ms"))
    },
    staging_sites: $staging,
    compute_sites: $compute,
    spanned: ($proj + $kv + $attn + ([
      "juno.SwiGlu.prefill.total_ms", "juno.RmsNorm.prefill.total_ms", "juno.Rope.prefill.total_ms",
      "juno.ResidualAdd.prefill.total_ms", "juno.WindowStep.bias_add.prefill.total_ms",
      "juno.WindowStep.embed.prefill.total_ms", "juno.WindowStep.lm_head.prefill.total_ms"
      ] | map(. as $k | $m | g($k)) | add))
  }
| .terms.residue = (.forward_pass_ms - .spanned)
| del(.spanned)
'

# Several repetitions' breakdowns in (an array), the per-term median out, with shares.
JQ_MEDIAN='
def median: sort | if length == 0 then 0 elif length % 2 == 1 then .[length / 2 | floor]
  else (.[length / 2 - 1] + .[length / 2]) / 2 end;
. as $reps
| ($reps | map(.forward_pass_ms) | median) as $fp
| {
    repetitions: ($reps | length),
    forward_pass_ms: $fp,
    terms: ($reps[0].terms | keys_unsorted | map(. as $t | {key: $t, value: {
      ms: ($reps | map(.terms[$t]) | median),
      pct: (if $fp > 0 then (($reps | map(.terms[$t]) | median) * 100 / $fp) else 0 end)
    }}) | from_entries),
    staging_sites: $reps[0].staging_sites,
    compute_sites: $reps[0].compute_sites
  }
'

breakdown_files() { # files... -> median breakdown JSON
  local f out=()
  for f in "$@"; do
    out+=("$(jq -c '.models[0].metrics' "$f" | jq -c "$JQ_TERMS")")
  done
  printf '%s\n' "${out[@]}" | jq -s -c "$JQ_MEDIAN"
}

run() {
  [ -d "$RUN_DIR" ] || { echo "prefill-breakdown: no such run directory: $RUN_DIR" >&2; exit 2; }
  local models model files f prefix result all="{}" failed=0
  if [ "$LANE" = tuned ]; then prefix='-tuned-prefill-rep'; else prefix='-prefill-rep'; fi
  models=$(cd "$RUN_DIR" && ls -- *"${prefix}"[0-9]*-juno-jfr.json 2>/dev/null \
    | sed -E "s/${prefix}[0-9]+-juno-jfr\\.json$//" | grep -v -- '-tuned$' | sort -u || true)
  [ "$LANE" = tuned ] && models=$(cd "$RUN_DIR" && ls -- *"${prefix}"[0-9]*-juno-jfr.json 2>/dev/null \
    | sed -E "s/${prefix}[0-9]+-juno-jfr\\.json$//" | sort -u || true)
  [ -n "$models" ] || { echo "prefill-breakdown: no prefill repetitions in $RUN_DIR" >&2; exit 2; }

  printf '| Model | Prefill ms | Reps | Term | ms | Share |\n|---|---|---|---|---|---|\n'
  for model in $models; do
    files=()
    for f in "$RUN_DIR/${model}${prefix}"[0-9]*-juno-jfr.json; do files+=("$f"); done
    result=$(breakdown_files "${files[@]}")
    all=$(jq -c --arg k "$model" --argjson v "$result" '. + {($k): $v}' <<<"$all")
    jq -r --arg model "$model" '
      . as $r | $r.terms | to_entries[]
      | "| \($model) | \($r.forward_pass_ms * 10 | round / 10) | \($r.repetitions) | \(.key) | \(.value.ms * 10 | round / 10) | \(.value.pct * 10 | round / 10)% |"
    ' <<<"$result"
    if jq -e --argjson max "$MAX_RESIDUE_PCT" '.terms.residue.pct > $max' <<<"$result" >/dev/null; then
      echo "RESIDUE ${model}: $(jq -r '.terms.residue.pct * 10 | round / 10' <<<"$result")% of prefill > ${MAX_RESIDUE_PCT}%" >&2
      failed=1
    fi
  done
  [ -n "$JSON_OUT" ] && jq . <<<"$all" >"$JSON_OUT"
  return $failed
}

selftest() {
  local d pass=0 fail=0 got
  d=$(mktemp -d)
  trap 'rm -rf "$d"' RETURN
  expect() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1: expected $2, got $3" >&2; fi; }
  # A 1000 ms window: projection 400 holding 100 compute, 20 dequant, 30 copies, 10 packing;
  # kv_write 60 holding 40 mirror copies; attention 200 holding 150 compute and 10 copies;
  # elementwise 250; embed 5, lm_head 15; 70 ms unspanned.
  rep() {
    jq -n --argjson fp "$1" '{models: [{metrics: {
      "juno.ForwardPass.prefill.total_ms": $fp,
      "juno.WindowStep.projection.prefill.total_ms": 400,
      "juno.WindowStep.kv_write.prefill.total_ms": 60,
      "juno.WindowStep.embed.prefill.total_ms": 5,
      "juno.WindowStep.lm_head.prefill.total_ms": 15,
      "juno.WindowStep.bias_add.prefill.total_ms": 0,
      "juno.Attention.prefill.total_ms": 200,
      "juno.SwiGlu.prefill.total_ms": 200, "juno.RmsNorm.prefill.total_ms": 30,
      "juno.Rope.prefill.total_ms": 10, "juno.ResidualAdd.prefill.total_ms": 10,
      "juno.WeightDequant.device.total_ms": 20,
      "juno.DeviceCompute.site.gemm_half.prefill.total_ms": 100,
      "juno.DeviceCompute.site.gqa_attention.prefill.total_ms": 150,
      "juno.DeviceStaging.site.cudamemcpyasync_xh_h2d_q4k_batched_gemm.prefill.estimated_total_ms": 30,
      "juno.DeviceStaging.site.pack_fp16_host.prefill.estimated_total_ms": 10,
      "juno.DeviceStaging.site.memcpy_k_row_h2d.prefill.estimated_total_ms": 20,
      "juno.DeviceStaging.site.memcpy_v_row_h2d.prefill.estimated_total_ms": 20,
      "juno.DeviceStaging.site.memcpy_gqa_qbatch_h2d.prefill.estimated_total_ms": 10
    }}]}'
  }
  mkdir -p "$d/run"
  rep 1000 >"$d/run/m-prefill-rep1-juno-jfr.json"
  rep 1000 >"$d/run/m-prefill-rep2-juno-jfr.json"
  rep 2000 >"$d/run/m-prefill-rep3-juno-jfr.json"
  got=$(breakdown_files "$d"/run/m-prefill-rep*-juno-jfr.json)
  expect "median window" 1000 "$(jq -r '.forward_pass_ms' <<<"$got")"
  expect "residue is the unspanned time" 70 "$(jq -r '.terms.residue.ms' <<<"$got")"
  expect "residue share" 7 "$(jq -r '.terms.residue.pct' <<<"$got")"
  expect "projection keeps only its host part" 240 "$(jq -r '.terms.projection_and_region_host.ms' <<<"$got")"
  expect "GEMM compute excludes attention" 100 "$(jq -r '.terms.gemm_compute.ms' <<<"$got")"
  expect "KV host write is kv_write minus mirror copies" 20 "$(jq -r '.terms.kv_host_write.ms' <<<"$got")"
  expect "attention host part" 40 "$(jq -r '.terms.attention_host.ms' <<<"$got")"
  expect "terms plus residue add up to the window" 1000 "$(jq -r '[.terms[].ms] | add' <<<"$got")"

  # A device-region window: device_layer 900 holding 200 GEMM, 20 dequant, 30 activation copies,
  # 50 elementwise kernels and 100 attention; kv_write 40 (host write only); embed 5, lm_head 15;
  # 40 ms unspanned.
  jq -n '{models: [{metrics: {
    "juno.ForwardPass.prefill.total_ms": 1000,
    "juno.WindowStep.device_layer.prefill.total_ms": 900,
    "juno.WindowStep.kv_write.prefill.total_ms": 40,
    "juno.WindowStep.embed.prefill.total_ms": 5,
    "juno.WindowStep.lm_head.prefill.total_ms": 15,
    "juno.WeightDequant.device.total_ms": 20,
    "juno.DeviceCompute.site.gemm_half.prefill.total_ms": 200,
    "juno.DeviceCompute.site.swiglu.prefill.total_ms": 30,
    "juno.DeviceCompute.site.rms_norm.prefill.total_ms": 10,
    "juno.DeviceCompute.site.residual_add.prefill.total_ms": 10,
    "juno.DeviceCompute.site.gqa_attention_region.prefill.total_ms": 100,
    "juno.DeviceStaging.site.upload_resident_activation.prefill.estimated_total_ms": 20,
    "juno.DeviceStaging.site.materialize_resident_activation.prefill.estimated_total_ms": 10
  }}]}' >"$d/region.json"
  got=$(breakdown_files "$d/region.json")
  expect "region: GEMM compute excludes the region's elementwise and attention kernels" 200 "$(jq -r '.terms.gemm_compute.ms' <<<"$got")"
  expect "region: elementwise kernels are their own term" 50 "$(jq -r '.terms.device_elementwise.ms' <<<"$got")"
  expect "region: attention inside the region is its own term" 100 "$(jq -r '.terms.region_attention_compute.ms' <<<"$got")"
  expect "region: host part of the region spans" 500 "$(jq -r '.terms.projection_and_region_host.ms' <<<"$got")"
  expect "region: residue is the unspanned time" 40 "$(jq -r '.terms.residue.ms' <<<"$got")"
  expect "region: terms plus residue add up to the window" 1000 "$(jq -r '[.terms[].ms] | add' <<<"$got")"
  RUN_DIR="$d/run"
  if run >/dev/null 2>&1; then got=0; else got=$?; fi
  expect "residue over 5% fails the run" 1 "$got"
  MAX_RESIDUE_PCT=10
  if run >/dev/null 2>&1; then got=0; else got=$?; fi
  expect "residue under the bound passes" 0 "$got"
  echo "prefill-breakdown selftest: $pass passed, $fail failed"
  [ "$fail" -eq 0 ]
}

while [ $# -gt 0 ]; do
  case "$1" in
    --selftest) selftest; exit $? ;;
    --lane) LANE="$2"; shift 2 ;;
    --json) JSON_OUT="$2"; shift 2 ;;
    --max-residue-pct) MAX_RESIDUE_PCT="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) echo "prefill-breakdown: unknown option $1" >&2; exit 2 ;;
    *) RUN_DIR="$1"; shift ;;
  esac
done
[ -n "$RUN_DIR" ] || { echo "usage: $0 RUN_DIR [--lane default|tuned] [--json OUT] [--max-residue-pct N] | --selftest" >&2; exit 2; }
run
