# Prefill weight dequantization and GEMM by window width - 20261003T060301Z (gpu)

## Purpose

Where a GPU prefill window's time goes at each window width, before a packed K-quant GEMM is written.
Today a prefill window wider than 8 rows expands every packed Q4_K/Q5_K/Q6_K weight matrix to FP16 on
the device (`juno.WeightDequant`), then multiplies with a cuBLAS FP16 GEMM at FP32 compute
(`juno.DeviceCompute`, site `gemm_half`). This run measures both terms, and everything else in the
window, at window widths 9, 16, 32, 64, 128 and 512 on the four sweep models. It answers two questions:

1. What share of a 512-token prefill is the GEMM on each model?
2. At what width does the dequantization pass stop being a material share of the window?

**Not pinned, and not a ratio reference.** Clocks were recorded, not fixed (CPU governor `schedutil`,
turbo on). The device spans themselves cost 4% to 7% of TinyLlama prefill. Read the shares and the
per-width trend from this run. Do not read absolute t/s against a pinned run, or the pp ratios against
the reference column.

## Command

One `compare-llama-cpp.sh` run per width, the same build, one session (06:03 to 07:14 UTC):

```bash
for W in 512 128 64 32 16 9; do
  JUNO_PREFILL_BATCH=$W scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans \
    --n-prompt 512 --juno-reps 3 --no-tuned-lane --no-publish --out target/perf-compare/kquant-width/w$W
  scripts/performance-tests/prefill-breakdown.sh target/perf-compare/kquant-width/w$W \
    --json target/perf-compare/kquant-width/w$W/prefill-breakdown.json
done
```

`JUNO_PREFILL_BATCH` passes `--prefill-batch W` to the engine. The harness does not record it in
`host.json`, so the width is the subdirectory name. The engine's own counts confirm it: TinyLlama ran
57, 32, 16, 8, 4 and 1 prefill forward passes, and dequantized 8,624, 4,928, 2,464, 1,232, 616 and 154
matrices (154 per full window). At width 9 the 512-token prompt is 56 nine-row windows plus one
8-row remainder. The remainder goes to the per-row GEMV path, which does not dequantize.

- Build: Juno `f02bdae5a62c`, tree clean, jar sha256 `ddeda5b20878a847` (a single build; no A/B).
- Every row: 512 of 512 prompt tokens, GPU attention resolved `on`, scorable (repetitions within 15%),
  `prefill-breakdown.sh` exit 0, residue 0.4% to 3.9% (limit 5%).
- Subdirectories `w009` to `w512` hold each run's harness output (`INDEX.md`, `host.json`, result
  JSON), `prefill-breakdown.md` and `prefill-breakdown.json`. Local paths are rewritten relative to
  the repository root.

## Results

Device terms are medians of three repetitions. Shares are of `juno.ForwardPass` prefill time for the
whole 512-token prompt.

**Share of prefill: GEMM compute (`gemm_half`) / weight dequantization / the two together**

| Model | w9 | w16 | w32 | w64 | w128 | w512 |
|---|---|---|---|---|---|---|
| tinyllama-1.1b | 45.3 / 36.2 / 81.5% | 43.9 / 34.9 / 78.8% | 39.2 / 30.2 / 69.4% | 33.6 / 24.3 / 57.9% | 32.2 / 15.9 / 48.1% | **32.9** / 5.1 / 38.0% |
| qwen2.5-3b | 48.2 / 39.4 / 87.6% | 48.2 / 37.9 / 86.1% | 44.9 / 34.4 / 79.3% | 40.1 / 29.3 / 69.4% | 39.1 / 20.5 / 59.6% | **49.3** / 6.3 / 55.6% |
| Phi-3.5-mini | 50.3 / 35.7 / 86.0% | 49.0 / 32.4 / 81.4% | 43.1 / 28.0 / 71.1% | 34.9 / 21.5 / 56.4% | 31.8 / 13.5 / 45.3% | **37.8** / 3.2 / 41.0% |
| mistral-7b | 47.4 / 45.3 / 92.7% | 47.4 / 43.3 / 90.7% | 47.3 / 38.3 / 85.6% | 44.5 / 33.1 / 77.6% | 45.9 / 23.7 / 69.6% | **53.3** / 7.9 / 61.2% |

**Milliseconds for the whole prompt: prefill / GEMM / dequantization; Juno prefill t/s (min to max)**

| Model | Width | Prefill ms | GEMM ms | Dequant ms | Attention share | Juno pp t/s |
|---|---|---|---|---|---|---|
| tinyllama-1.1b | 9 | 4,378 | 1,985 | 1,584 | 7.3% | 116.9 (116.8 to 117.0) |
| | 16 | 2,618 | 1,148 | 914 | 12.4% | 195.5 (195.5 to 195.6) |
| | 32 | 1,504 | 589 | 455 | 21.3% | 340.3 (339.8 to 340.4) |
| | 64 | 939 | 315 | 227 | 32.1% | 545.1 (542.2 to 546.3) |
| | 128 | 686 | 221 | 109 | 40.6% | 746.2 (719.1 to 749.0) |
| | 512 | 493 | 162 | 25 | 50.5% | 1,038.5 (1,009.1 to 1,040.1) |
| qwen2.5-3b | 9 | 12,048 | 5,803 | 4,745 | 4.7% | 42.5 (41.8 to 43.6) |
| | 16 | 6,885 | 3,319 | 2,610 | 7.3% | 74.4 (74.3 to 74.9) |
| | 32 | 3,796 | 1,705 | 1,305 | 13.5% | 134.9 (134.8 to 135.0) |
| | 64 | 2,228 | 893 | 652 | 22.9% | 229.8 (229.2 to 229.8) |
| | 128 | 1,556 | 609 | 319 | 31.3% | 329.0 (327.7 to 332.2) |
| | 512 | 1,171 | 577 | 73 | 35.6% | 437.2 (436.4 to 437.3) |
| Phi-3.5-mini | 9 | 18,112 | 9,116 | 6,470 | 4.1% | 28.3 (28.1 to 29.1) |
| | 16 | 10,613 | 5,197 | 3,436 | 6.9% | 48.2 (48.2 to 48.3) |
| | 32 | 6,131 | 2,640 | 1,714 | 11.2% | 83.5 (83.4 to 83.5) |
| | 64 | 3,949 | 1,377 | 847 | 16.4% | 129.6 (129.4 to 130.2) |
| | 128 | 3,098 | 983 | 417 | 20.2% | 165.3 (164.4 to 165.7) |
| | 512 | 2,939 | 1,112 | 95 | 26.5% | 174.2 (172.4 to 174.2) |
| mistral-7b | 9 | 29,927 | 14,177 | 13,553 | 3.5% | 17.1 (16.7 to 17.4) |
| | 16 | 17,144 | 8,120 | 7,431 | 6.1% | 29.9 (29.6 to 30.8) |
| | 32 | 8,521 | 4,033 | 3,263 | 10.4% | 60.1 (59.9 to 60.5) |
| | 64 | 4,929 | 2,192 | 1,633 | 17.2% | 103.9 (103.9 to 103.9) |
| | 128 | 3,441 | 1,580 | 816 | 23.9% | 148.8 (148.3 to 149.0) |
| | 512 | 2,554 | 1,362 | 203 | 31.6% | 200.5 (199.2 to 201.4) |

The attention share is the GPU attention kernel (`gqa_attention_region` on the region models,
`gqa_attention` on Phi-3.5-mini). Matmul staging is 0.2% to 2.7% at every width; host-side terms and
the residue make up the rest (each model's `prefill-breakdown.md` lists every term).

## How to read it

- **The GEMM is the largest single term at 512 tokens on three of four models**: 53.3% on Mistral 7B,
  49.3% on Qwen2.5-3B, 37.8% on Phi-3.5-mini. On TinyLlama attention is larger (50.5% against 32.9%).
  The Mistral 7B figure matches the last published breakdown on an earlier build (53.1%,
  `20261002T050741Z`).
- **Dequantization never stops being material on the region models.** It is at least 5% of the window
  at every width on TinyLlama, Qwen2.5-3B and Mistral 7B, including a full 512-token window (5.1%, 6.3%,
  7.9%). Only Phi-3.5-mini at 512 falls below (3.2%). The pass costs about the same per window whatever the
  width (Mistral 7B: 242 ms per window at width 9, 203 ms at 512), so it scales with the number of windows. At widths 9 to 32, where chunked prefill and
  mixed continuous steps run, it is 28% to 45% of the window.
- **Narrow FP16 GEMMs are bound by the weight read, not by compute.** The 512-token prompt carries the
  same arithmetic at every width, yet GEMM time for the whole prompt is 10.4x (Mistral 7B) to 12.3x
  (TinyLlama) larger at width 9 than at 512. Every window re-reads every FP16 weight matrix, which is
  about 3.6x the bytes of a packed Q4_K matrix (2.4x for the Q6_K tensors a Q4_K_M file mixes in). At width 9, dequantization plus GEMM
  is 81.5% to 92.7% of the window. A GEMM reading the packed weights removes the dequantization pass
  and cuts the bytes per window. This is the regime where it has the most room.
- **Phi-3.5-mini's GEMM is faster at width 128 than at 512** (983 against 1,112 ms for the same work).
  This was observed, not investigated. Its fused QKV and gate-up matrices are the widest in the sweep,
  and the cuBLAS algorithm choice at N=512 is the likely cause.
