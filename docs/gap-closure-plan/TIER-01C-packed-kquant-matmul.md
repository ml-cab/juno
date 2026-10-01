# Tier 01C: Packed K-quant prefill matmul

Status: not started
Gap analysis refs: none directly — adjacent to §1.1. Split out of
[Tier 04C](TIER-04C-packed-weight-matmul.md) on 2026-09-30 by the plan review; see "Why this tier, why
now".

Line numbers cited from Tier 04C were a snapshot at HEAD `1f90b68`. Re-verify every one against
current source before acting on it (README, opening paragraph).

## Objective

Give the CUDA prefill path a tiled integer matmul over still-packed Q4_K, Q5_K and Q6_K weights, so a
batch wider than `HALF_SGEMM_BATCH_MAX = 8` stops expanding each weight matrix to FP16 and running
`cublasGemmEx` at FP32 compute. On this host the kernel is the one route by which Juno's prefill GEMM
can beat the FP32-compute ceiling. That ceiling is the limit the README's "Post-plan anchor" roofline
puts at roughly 0.5x to 0.7x of the reference tool on mistral-7b.

## Why this tier, why now

Tier 04C described three regimes for quantized weights on CUDA. Regime B is the prefill path for every
sweep model: Q4_K/Q5_K/Q6_K at batch > 8 calls `Q4KMmqKernel.launchDequant` to expand the whole packed
matrix into a device FP16 scratch (`Q4KDequantScratch`), then `CudaFp16GemmOps.gemmHalf`
(`cublasGemmEx`, `CUDA_R_16F` operands, `CUBLAS_COMPUTE_32F`). The fused kernels in `q4k_gemv.cu` are
all GEMV. Juno has the decode end (`Q4KMmqKernel.launch`, `dp4a` over packed bytes) and the dequant
fallback, but not the tiled middle the reference tool uses as its default prefill route on Pascal.

It was moved from Tier 04C's position (after Tiers 02, 03, 04 and 04B) to directly after Tier 01B for
three reasons:

- **Prompt processing is the largest gap in the program.** At the step 2 reference GPU pp is about
  0.06x to 0.10x and GPU tg about 0.36x to 0.62x. Tier 01B is the only other tier working on pp.
- **Tier 01B's own data says its items cannot carry the program past the FP32 ceiling.** Removing
  staging and dequantization entirely is worth at most about 1.13x to 1.18x (01B step 2). Moving the
  host elementwise work (01B item 6) removes the largest term but leaves the GEMM. Once that is gone,
  the GEMM is the floor, and on GP104 only `dp4a` lowers it: native FP16 arithmetic runs at 1/64 of
  FP32 rate and `dp4a` at about 4x.
- **It needs nothing from Tier 04.** Every sweep model is Q4_K_M, and the packed device matrix
  (`DeviceQ4KMatrix`) already covers exactly Q4_K, Q5_K and Q6_K. Tier 04C's argument for waiting
  ("written once against the complete set of packed types") costs every tier between here and 04C a pp
  ratio nobody is working on. Writing it against three types and widening later is the cheaper error.

What stays in [Tier 04C](TIER-04C-packed-weight-matmul.md): extending this tier's kernel to the
formats Tier 04 adds, widening packed residency to them, the explicit and budgeted FP16 fallback, ROCm
packed residency, and the `--mmq` documentation audit.

**Read Tier 01B's breakdown before starting.** This tier's throughput threshold is conditioned on the
GEMM's measured share of prefill after Tier 01B, read from `juno.DeviceCompute` (01B item 1a-ii). If
01B closed partial-complete with the GEMM named as the dominant term, this tier is its successor; if 01B
found the GEMM negligible at every width Juno runs, this tier re-scopes to the VRAM argument (see the
threshold) and says so, rather than proceeding on a throughput premise the data did not support.

## Scope

### In scope

1. **Measure first.** Per sweep model, at batch widths 9, 16, 32, 64, 128 and 512, the split between
   `launchDequant` (`juno.WeightDequant`, device timing), `gemmHalf` (`juno.DeviceCompute`,
   `site=gemm_half`), and the staging around them (`juno.DeviceStaging`), from a `--device-spans` run.
   Publish it before writing kernel code. Record the width at which the dequant pass stops being
   material, and the GEMM's share of prefill at 512 on every sweep model.
2. **A tiled packed-weight GEMM for Q4_K, Q5_K and Q6_K.** A `mul_mat_q`-shaped kernel in the
   `q4k_gemv.cu` family: quantize the activation batch to Q8_1 once, tile over K, integer-dot with
   `dp4a`, accumulate in FP32, write FP32. `launchPacked` already establishes the packed-pointer launch
   shape for one column; this is its batched counterpart. Route `sgemm(DeviceQ4KMatrix, ...)` for
   batch > `HALF_SGEMM_BATCH_MAX` to it, and take `Q4KDequantScratch` off the default path for these
   three formats. Keep the dequant entry points: `--mmq off` and Tier 04C's fallback use them.
   It reads and writes through Tier 01B's prefill-window region (the operand residency of 01B items 2
   and 6), not through the host-staging `sgemm` contract, so the gain is not paid back in copies.
3. **Shrink the forward-pass reserve.** `DeviceScratchBudget` (shipped in `c91f879`) reserves a
   weight-shaped FP16 scratch for the widest matmul. Once item 2 is the default for every device
   matrix a model has, that term goes. Shrink the reserve to what the packed path needs, and prove the
   shrink with a model that fits under the new budget and not the old one.
4. **Host-specific findings marked.** Every conclusion this tier reaches is taken on a GP104 with no
   tensor cores and 1/64-rate FP16. The choice of an integer GEMM over `cublasGemmEx`, the crossover
   width, and any claim that the packed path wins on compute carry a one-line `host-specific` or
   `expected-general` marker in the execution record (the obligation Tier 04C item 2 stated, moved
   here with the kernel).

### Out of scope

- Formats other than Q4_K, Q5_K and Q6_K, and packed residency for them — Tier 04C, after Tier 04.
- ROCm. `RocmMatVec` has no `sgemm` and no packed residency; the ROCm tiled GEMM is Tier 10 item 1 and
  ROCm packed residency is Tier 04C item 3. This tier must not leave the ROCm path worse than it found it.
- Phi-2 and Qwen3-MoE, which have no device weight path until Tier 08 item 6.
- CPU packed matmul (Tier 10) and LoRA training residency (exempt by design; Tier 12).
- Activation formats below Q8_1.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference (scalar) | N/A for the kernel; the `CpuMatVec` FP32 path is the correctness oracle |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | N/A for the kernel (CUDA-only this tier); must not regress, verified by `RocmMatVecTest` compiling and its non-hardware cases passing; `NEEDS-AMD-HARDWARE` for anything else |
| 4 | Static schedule | primary target: prefill windows always cross batch 8 |
| 5 | Continuous schedule | mixed prefill and decode steps sit at widths 9 to 64, where the dequant amortizes worst; measure there specifically |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | each shard runs the kernel on its own layers; verify output against local mode and that the reserve is computed per shard |
| 8 | Tensor-parallel cluster | same; nodes still load the whole model until Tier 09 |
| 9 | LoRA training | exempt (FP32-resident frozen weights, `LoraMmqPolicy`); re-verify the exemption and its notice |
| 10 | LoRA playback | MMQ is allowed in playback; the delta-add must compose against the packed batched path at every width, within the numerical-quality threshold |
| 11 | Vision | the CLIP encoder is pinned to `CpuMatVec.INSTANCE`; verify that still holds. moondream2's text half is Phi-2 (CPU). `compare-vision.sh` is still a required gate |
| 12 | OpenAI REST surface | N/A directly; verified through TTFT |
| 13 | Native REST surface | same |
| 14 | CLI | `--mmq off` now differs from `auto` in prefill as well as decode; the `--help` text and `docs/howto.md` say so |
| 15 | JVM embedding facade | N/A: no new embedder-invocable capability |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Publish item 1's per-width breakdown and record the GEMM share at 512 per sweep model. Decide here,
   in writing, which branch of the throughput threshold applies.
3. Write the tiled kernel against Q4_K, validated against the CPU FP32 oracle at every width in the
   test matrix; then Q5_K and Q6_K in one pass.
4. Switch `sgemm(DeviceQ4KMatrix, ...)` for batch > 8, re-measure immediately (same-hour pinned A/B
   against the pre-switch build).
5. Shrink the reserve (item 3) as a separate, separately measured change.
6. Full cross-surface smoke matrix, including the vision gate.

## Tests to write/upgrade before implementation

- **Tiled packed GEMM correctness**: against the `CpuMatVec` FP32 oracle for Q4_K, Q5_K and Q6_K at
  batch widths 1, 2, 8, 9, 16, 32, 64, 128, 512 and 741, square and FFN-shaped, including a `cols` of a
  single super-block and one that is not a multiple of the tile.
- **Numerical quality**: mean and max relative error against the FP32 oracle for the packed path and
  for the dequant-to-FP16 path, same matrices, published side by side.
- **Greedy-decode parity**: same prompt and seed through the packed and FP16 paths, token-identical or
  characterised.
- **Device memory**: no leak across repeated windows (Tier 01's `memGetInfo` assertion); peak device
  bytes during a 512-token prefill asserted against the decode-time figure.
- **Reserve**: a `DeviceScratchBudgetTest` case for the shrunk reserve, and a device-OOM regression
  test that a model fitting only under the new budget survives a prefill window wider than 8.
- **`ModelLiveRunnerIT`**: a 512-token prefill check on both schedules with the packed path active.
- **New bash smoke script**: `scripts/performance-tests/smoke-packed-kquant-matmul.sh` (no tier number
  in the name, as Tier 01 established for scripts `docs/howto.md` cites) — drives `./juno local` on both
  schedules for every sweep model, asserts correct output, records peak VRAM and TTFT.
- **Perf gate (required)**: a MatVec and forward-pass change. `compare-lora.sh --reps 3`,
  `compare-vision.sh`, and `compare-llama-cpp.sh --gpu --pin-clocks` at `n_prompt` 128 and 512;
  publish under `docs/perf-compare/<timestamp>-packed-kquant-matmul/`.

  **Threshold, throughput (item 2).** Read as Juno absolute prefill t/s from a same-hour interleaved
  A/B with pinned clocks against the pre-switch build (README, "No-regression gates tighter than the
  floor"). Which branch applies is decided in step 2 and written down before the kernel exists:
  - if the GEMM is **>= 30%** of prefill at `n_prompt=512` on mistral-7b after Tier 01B: prefill
    **>= 1.30x** on mistral-7b at 512, and **>= 1.10x** on tinyllama at 512;
  - otherwise, the VRAM branch: prefill **>= 0.95x** at 512 on both models (a no-regression floor),
    with the throughput premise recorded as unsupported and the tier justified by the VRAM threshold
    alone;
  - either branch: **>= 1.10x** at batch widths 16 to 64 on tinyllama and mistral-7b, where the
    dequant amortizes worst and the continuous schedule runs.

  **Threshold, VRAM.** Peak device memory during a 512-token prefill **<= 1.05x** peak during decode on
  mistral-7b and llama-1-30b Q4_K_M (the weight-shaped scratch term is gone, not reduced).

  **Threshold, numerical quality.** Mean relative error against the FP32 oracle **<= 1.0x** that of the
  dequant-to-FP16 path on every tested shape; greedy decode agrees with the FP16 path on **>= 99%** of
  the first 512 tokens at temperature 0.

  **Threshold, no regression elsewhere.** Juno tg t/s **>= 0.95x** the pre-tier build on every sweep
  model (same-hour pinned A/B; decode stays on the GEMV path and should not move); `compare-lora.sh`
  train **>= 0.95x** and wall-clock playback **>= 0.80x**; vision `latency_ms` **<= 1.25x** and decode
  tps **>= 0.80x** its baseline.

  **Milestone (README milestone table, added 2026-09-30).** GPU pp ratio at `n_prompt=512` **>= 0.20x**
  on every sweep model, read from this tier's closing `compare-llama-cpp.sh --gpu --pin-clocks` sweep.
  Reported, not gated below the noise floor (README): 0.20x is about 2x the Tier 01B milestone, well
  outside it. The closing sweep also runs at `n_prompt=2048` and publishes the first parity-corrected
  2048 reading, which becomes the reference for Tier 02's milestone row.

  The pp ratio on every sweep model is recorded against Tier 01B's milestone rows, this tier's
  milestone, the end-of-plan target (`>= 0.25x`) and the README's post-plan anchor, met or missed with
  the number.

## Models needed

The four sweep models (all Q4_K_M), `llama-1-30b.Q4_K_M.gguf` for the VRAM and reserve thresholds, and
`moondream2-q5_k.llamafile` for the vision gate. All present. Q4_K_M files mix in Q6_K for some
tensors, so real Q6_K matrices are on disk; check with `./juno gguf-info` whether any file carries Q5_K
tensors, and if none does, test Q5_K on synthetic matrices encoded by the existing golden-value path.
A whole Q5_K_M model is not required.

## Exit criteria

- [ ] Item 1's per-width breakdown published, with the GEMM's share at 512 per sweep model and the
      throughput-threshold branch chosen in writing before the kernel was built.
- [ ] A tiled packed GEMM for Q4_K, Q5_K and Q6_K is the default for batch > `HALF_SGEMM_BATCH_MAX`,
      reading and writing through Tier 01B's prefill-window region; `Q4KDequantScratch` is off the
      default path for these formats; `--mmq off` still gives the FP16 baseline.
- [ ] The forward-pass reserve shrunk, proven by the fits-only-under-the-new-budget test.
- [ ] Every threshold above met, or reported missed with its number.
- [ ] The milestone (GPU pp >= 0.20x at `n_prompt=512` on every sweep model) read from the closing
      sweep and reported met or missed with the number; the first `n_prompt=2048` sweep published and
      the README's Tier 02 milestone reference cell filled from it in the same change.
- [ ] Every conclusion in the execution record carries a `host-specific` or `expected-general` marker.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published; `compare-llama-cpp.sh` pp and tg ratios recorded in this file against the
      program target, Tier 01B's milestone rows and the post-plan anchor.
- [ ] Docs (`docs/agent-arch.txt`, `docs/howto.md`, `docs/performance.md`) updated in Juno-native
      language; the `--mmq` help text says `off` and `auto` now differ in prefill.
- [ ] `CHANGELOG.md` entry added.
