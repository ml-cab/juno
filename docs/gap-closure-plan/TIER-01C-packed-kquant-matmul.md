# Tier 01C: Packed K-quant prefill matmul

Status: **complete** (2026-10-04). Every exit criterion is checked. The tiled packed K-quant GEMM
is the default prefill matmul on CUDA. On the owner's pinned gates: prefill 1.23x to 1.55x at 512 tokens
against the FP16-expansion build; GPU pp 0.226x (Phi-3.5-mini, binding) to 0.468x of the reference engine
at 512 tokens, meeting the `>= 0.20x` milestone; decode, vision and LoRA not regressed. The first complete
2048 reading (0.099x to 0.180x) is Tier 02's starting point. Four out-of-tier fixes, each by owner decision:
a race in the GPU attention kernel, the decode GEMV scratch held at upload, out-of-memory requests failing
instead of hanging, and recorded prompt encoding. Carried to later tiers: playback prefill on the tiled kernel
(Tier 12 item 5), the quadratic SentencePiece encoder (Tier 04B item 5), and the host KV footprint (Tier 03
item 6). See "Execution record".
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
| 10 | LoRA playback | MMQ is allowed in playback; the delta-add must compose against the packed batched path at every width, within the numerical-quality threshold. *Corrected 2026-10-04 (step 6 plan-versus-code check): playback never reaches the batched packed path. `LoraTrainableHandler.forwardBatch` multiplies through `LoraResidentWeights.matVecBatch`, which runs one packed decode GEMV per row whenever the weights are Q4-resident ("Q4 residency always uses sequential GEMV"); it holds `ResidentQ4KWeight`, not the `CudaMatVec.sgemm(DeviceQ4KMatrix, ...)` route this tier switched. Playback numerics and speed are therefore unchanged by this tier, and the row is resolved as unchanged, guarded by `compare-lora.sh` playback and `LoraQ4KPlaybackParityTest`. Whether playback prefill should take the tiled kernel is raised with the owner (step 6 record).* |
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

  **Threshold, numerical quality** (restated 2026-10-03, owner decision; see the step 3 record).
  Mean relative error against the FP32 oracle **<= 1.001x** that of the fused decode GEMV (the same
  Q8_1 activation rounding; the 0.1% admits float accumulation order only) and **<= 0.5%** absolute,
  on every tested shape of `KQuantGemmQualityTest`. End to end, greedy decode at temperature 0 is
  deterministic: two identical requests in one process give the same 512 tokens, on every sweep model and
  both schedules (`smoke-packed-kquant-matmul.sh`, **100%** of cells). The first divergence from the FP16
  route over 512 tokens is recorded per model and schedule and the outputs read for coherence; it is not
  a gate. *Restated 2026-10-04, owner decision 3 (a). Was: "greedy decode agrees with the FP16 path on
  >= 99% of the first 512 tokens". No route meets that against any other: the FP16 route parts from
  per-token prefill at token 0 to 244 and the packed route at 24 to 512
  (`docs/perf-compare/20261004T050618Z-packed-kquant-smoke`). Over 512 greedy tokens, any change in
  rounding flips a near-tie somewhere, so the clause measured where, not which route is more accurate.
  The kernel-level clause above stays the binding accuracy gate.* *Was: mean relative error <= 1.0x that of the
  dequant-to-FP16 path. No integer-activation kernel can meet that: 8-bit activations put the packed
  route at about 14x the FP16 route's error by construction (0.37% against 0.026%,
  `docs/perf-compare/20261003T171753Z-packed-kquant-kernel`), the same rounding decode already ships
  for every generated token.*

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

## Execution record

### 2026-10-03: implementation steps 1 and 2, the per-width breakdown and the branch decision

**Step 1.** `scripts/performance-tests/check-plan-thresholds.sh`: ok (19 tier files, milestone table
checked).

**Plan against code (re-verified at HEAD `f02bdae`, tree clean).** Every claim this step depends on
holds. `CudaMatVec.HALF_SGEMM_BATCH_MAX = 8`. `sgemmInto(DeviceQ4KMatrix, ...)` runs serial GEMVs up to
8 rows; above that, `sgemmQ4KBatchedGemm` calls `Q4KMmqKernel.launchDequant` into `Q4KDequantScratch`
and then `CudaFp16GemmOps.gemmHalf`. Tier 01B's prefill-window region takes the same route on device
operands: `PrefillWindowRegion` calls `CudaMatVec.gemmOnStream(DeviceQ4KMatrix, ...)`, which calls
`dequantOnStream` then `gemmHalfOnStream`. The region opens for windows wider than
`PrefillWindowRegion.MAX_HOST_WINDOW = 8`. No line number in this file needed correcting, because it
cites none. All models the tier needs are on disk (`INVENTORY.md`).

**Method.** The breakdown needs fixed window widths. `--n-prompt 512` with `--prefill-batch W`
(the harness's `JUNO_PREFILL_BATCH` pass-through) splits the same 512-token prompt into windows of
width W. The engine's own counts confirm the split, for example TinyLlama: 57, 32, 16, 8, 4 and 1
prefill passes, 154 matrices dequantized per full window. At width 9 the last window is 8 rows and
takes the GEMV path. One `--device-spans --juno-reps 3 --no-tuned-lane` run per width, one build
(jar `ddeda5b20878a847`), one session, unpinned. The tier asks for this as a `--device-spans`
attribution run, not as a gate. Shares are what the decision reads.

Published as [`20261003T060301Z-packed-kquant-width-breakdown`](../perf-compare/20261003T060301Z-packed-kquant-width-breakdown/INDEX.md).
Every row 512 of 512 prompt tokens, GPU attention on, scorable, residue 0.4% to 3.9%.

| Model | GEMM share at 512 | Dequant share w9 / w16 / w32 / w64 / w128 / w512 | Dequant + GEMM at w9 | GEMM ms for the prompt, w9 over w512 |
|---|---|---|---|---|
| tinyllama-1.1b | 32.9% | 36.2 / 34.9 / 30.2 / 24.3 / 15.9 / 5.1% | 81.5% | 12.3x |
| qwen2.5-3b | 49.3% | 39.4 / 37.9 / 34.4 / 29.3 / 20.5 / 6.3% | 87.6% | 10.1x |
| Phi-3.5-mini | 37.8% | 35.7 / 32.4 / 28.0 / 21.5 / 13.5 / 3.2% | 86.0% | 8.2x |
| mistral-7b | **53.3%** | 45.3 / 43.3 / 38.3 / 33.1 / 23.7 / 7.9% | 92.7% | 10.4x |

- **The width at which dequantization stops being material: none that Juno runs, on the region
  models.** It is at least 5% of the window at every width on TinyLlama, Qwen2.5-3B and Mistral 7B,
  a full 512-token window included. Only Phi-3.5-mini at 512 falls below (3.2%). `expected-general`:
  the pass reads the packed matrix and writes an FP16 one once per window on any device. Its share
  depends on the bandwidth-to-compute balance, but it does not vanish.
- **Narrow FP16 GEMMs are bound by the weight read.** The prompt's arithmetic is the same at every
  width, yet the GEMM takes 8x to 12x longer in total at width 9 than at 512, because every window
  re-reads every FP16 matrix. `expected-general` for the direction: any device re-reads the weights
  per window. `host-specific` for the size of the factor.
- **Phi-3.5-mini's GEMM is faster at width 128 than at 512** (983 against 1,112 ms). Observed, not
  investigated; likely the cuBLAS algorithm choice for its fused matrices. `host-specific`.

**The branch decision (step 2, written before any kernel code).** The GEMM is **53.3%** of a
512-token prefill on Mistral 7B, at or above the 30% line. It read 53.1% on the earlier build in
`20261002T050741Z`, so the decision does not rest on one run. **The throughput branch applies**:

- prefill **>= 1.30x** on mistral-7b and **>= 1.10x** on tinyllama at `n_prompt=512`;
- **>= 1.10x** at batch widths 16 to 64 on tinyllama and mistral-7b (either branch);
- the VRAM, numerical-quality and no-regression thresholds unchanged.

All of these are read from same-hour pinned A/Bs against the pre-switch build (step 4), never from
this run. `host-specific`: the 30% line and the shares behind it come from GP104's FP32-compute
cuBLAS rate and this build's attention kernel.

**Decomposition of the ask** (Tier 01B's rule: write down each item's expected contribution
before implementing). A removed or replaced term changes the window and nothing else. The device
spans cost 4% to 7%, so these are estimates to check against the A/B, not predictions of its
figure:

| Threshold | Window now | Window needed | Dequant removed | What the packed GEMM must then do |
|---|---|---|---|---|
| mistral-7b, 512, >= 1.30x | 2,554 ms | <= 1,965 ms | -203 ms (1.09x alone) | <= 976 ms against cuBLAS's 1,362: **at least 1.40x faster than cuBLAS at N=512**, Q8_1 activation quantization included |
| tinyllama, 512, >= 1.10x | 493 ms | <= 448 ms | -25 ms (1.05x alone) | <= 142 ms against 162: at least 1.14x faster |
| mistral-7b, w16 / w64, >= 1.10x | 17,144 / 4,929 ms | <= 15,585 / 4,481 ms | 1.76x / 1.49x alone | may be up to 1.72x / 1.54x *slower* than cuBLAS and still pass |
| tinyllama, w16 / w64, >= 1.10x | 2,618 / 939 ms | <= 2,380 / 854 ms | 1.54x / 1.32x alone | may be up to 1.59x / 1.45x slower and still pass |

The width rows follow from removing the dequantization pass. The binding item is the 512-token
Mistral 7B row. It needs the integer kernel to sustain about 7.4 TOPS on the window's 7.2 TFLOP-equivalent
of matmul work (README "Post-plan anchor"). That is about 21% of GP104's `dp4a` peak (2,560 cores, 4
int8 MACs each, about 1.7 GHz: about 35 TOPS). cuBLAS sustains about 5.3 TFLOPS today, about 60% of the
FP32 peak. Reachable on paper; this is where the tier can miss, and if it does the miss is reported
with its number. `host-specific`: on a device with FP16 or int8 tensor cores cuBLAS would move first,
and the comparison would have to be redone there.

**Observation for step 4 (not fixed here, outside this step).** `compare-llama-cpp.sh` passes
`JUNO_PREFILL_BATCH` to the engine but does not write it to `host.json`, so a run's window width is
recorded only in its directory name. Step 4's width-16-to-64 A/B depends on it. Raised with the owner.
*Resolved 2026-10-03 (owner decision: record it).* See the entry below.

### 2026-10-03: the harness records the prefill window width

`compare-llama-cpp.sh` gains `--prefill-batch N` (the `JUNO_PREFILL_BATCH` environment variable still
works) and writes `juno_prefill_batch` to `host.json`. The same edit records the other settings that
reached the engine only through the environment: `juno_spec_type`, `juno_spec_ngram_n`,
`juno_spec_ngram_m` and `juno_model_draft`, which Tier 06's runs will need. Each is empty when unset,
so "engine default" reads differently from "not recorded".

- Tests first: three `--selftest` cases (width recorded, speculation settings recorded, an unset width
  recorded as empty rather than missing). They failed on the script without the fields (`null`, `,,,`,
  `null`) and pass with them; the whole `--selftest` exits 0.
- Live check: dry runs with `--prefill-batch 16` and with `JUNO_PREFILL_BATCH=9` both put
  `--prefill-batch` on the engine's command line, and the second wrote `"juno_prefill_batch":"9"` to
  its `host.json`.
- `docs/howto.md` (prefill breakdown section) documents the option and the recorded fields.
- **Not a measurement boundary.** The engine's command line is unchanged for any given setting; only
  the metadata gains fields. (Tooling only; this entry states no performance conclusion to mark.) The step 2 run predates it. Its width is in its directory names, and its
  INDEX gives the pass counts that confirm it.

### 2026-10-03: implementation step 3, the tiled kernel for Q4_K, Q5_K and Q6_K

**Plan against code (re-verified at HEAD `f02bdae`).** Unchanged from the step 1 entry:
`HALF_SGEMM_BATCH_MAX = 8`, `sgemmQ4KBatchedGemm` dequantizes then calls `gemmHalf`, and the region's
`gemmOnStream(DeviceQ4KMatrix, ...)` calls `dequantOnStream` then `gemmHalfOnStream`. `./juno gguf-info`
on the sweep models: TinyLlama, Qwen2.5-3B and Mistral 7B carry Q4_K and Q6_K; **Phi-3.5-mini also
carries 32 Q5_K tensors**, so Q5_K gets real-weight coverage once step 4 routes prefill through the
kernel. Step 3 tests Q5_K on synthetic matrices from `GgufKQuantCodec`, as "Models needed" allows.

**What was built.** `node/src/main/cuda/kquant_gemm.cu` compiles to `kquant_gemm.ptx`. It is a new file,
not added to `q4k_gemv.cu`, so the decode GEMV's PTX is not regenerated by a different compiler; that
would be a decode measurement boundary for no reason. `KQuantGemmKernel` loads and launches it.
`KernelParams` gains a two-dimensional grid launch. Shape of the kernel: the activation batch is
quantized to Q8_1 once by the GEMV module's `quantize_q8_1` (contiguous `batch * cols` values). Each
256-thread block takes 64 weight rows and a column tile of 16, 32 or 64 activation rows (the narrowest
that covers the batch). For each 256-element super-block, the block unpacks the packed weights to int8
in shared memory, stores the scales as float pairs, and integer-dots them with `dp4a`. Each lane owns
two weight rows and each warp owns a slice of the columns, so the activation reads from shared memory
are broadcasts. Output is FP32 at a caller-chosen row stride. No FP16 copy of the weights exists at
any point. ptxas: 77 to 112 registers, no spills, 27 to 42 KB shared memory, two to three blocks per SM.

**Tests first.** `KQuantGemmParityTest` was written against a stub `KQuantGemmKernel` that had no PTX
module, and failed for the right reason: the "tiled K-quant GEMM kernel loads" assertion. With the
kernel in place it passes, 31 cases: Q4_K, Q5_K and Q6_K at widths 1, 2, 8, 9, 16, 32, 64, 128, 512 and
741, each on four shapes (512 x 512 square; 1376 x 512 and 512 x 1536, both FFN orientations; 100 x 256,
one super-block with a row count that is not a multiple of the tile), plus an output-stride case that
checks nothing outside the matrix's column range is written. Each output is checked two ways:
- **against the fused GEMV, within 2e-5 relative.** Both quantize to the same Q8_1 bytes and compute the
  same integer products, so only the float accumulation order differs. This is the check that catches a
  layout error.
- **against the CPU FP32 oracle**, within six standard deviations of the row's Q8_1 rounding noise
  (`||w_r|| / 127 / sqrt(12)`), or the GEMV tests' `q8Tol`, whichever is wider. The first run used
  `q8Tol` alone and failed by 0.15 to 0.19 on the 1536-wide shape for all three types. That is about
  3.5 sigma of the activation rounding the GEMV path shares, on millions of samples, so the band was
  wrong, not the kernel. The band was corrected, not loosened: it now scales with the noise it is meant
  to admit, and the tight GEMV check runs first.

**Numerical quality, measured** (`KQuantGemmQualityTest`, published in
[`20261003T171753Z-packed-kquant-kernel`](../perf-compare/20261003T171753Z-packed-kquant-kernel/INDEX.md)).
Against the FP32 oracle, the packed route's mean relative error is 0.37% and the FP16 dequant route's is
0.026%, a ratio of 14.4 on every type and shape. `expected-general`: this is the 8-bit activation
rounding, identical to the decode GEMV's, and any `dp4a` kernel carries it.
**This fails the tier's numerical-quality threshold as written ("mean relative error <= 1.0x that of
the dequant-to-FP16 path"), and no integer-activation kernel can meet it.** Raised with the owner; the
threshold is not edited here. The options put to the owner:
(a) restate it against the fused GEMV, so the packed GEMM is no worse than what decode already ships,
with an absolute ceiling and the unchanged greedy-decode >= 99% as the end-to-end guard (recommended);
(b) keep it, which means the tier cannot close on any `dp4a` kernel;
(c) drop the comparative clause and keep only the greedy-decode agreement.
*Owner decision 2026-10-03: (a).* The threshold now reads against the fused GEMV with a 0.5% absolute
ceiling; `KQuantGemmQualityTest` asserts both.

**Speed, measured** (unpinned unit-test microbenchmark, same directory, median of three; not a gate).
These are Mistral 7B Q4_K shapes, per matmul: packed against dequant + FP16 GEMM, 2.3x to 3.0x faster at
width 512 and 3.7x to 9.3x at widths 16 to 64. At width 512 the 4096 x 4096 matmul sustains about 10 to
11 TOPS-equivalent, against the about 7.4 that the step 2 decomposition needs for the 1.30x Mistral 7B
threshold. `host-specific`: GP104 holds cuBLAS to FP32 compute. Whether it carries end to end is step
4's pinned A/B.

**Regression runs.** `mvn test -pl node`: 864 run, 0 failures, 44 skipped. A first full run had three
failures in `PrefillRegionHandlerParityTest`'s device-wide free-VRAM check (0.1 to 53 MiB short, with the
starting free memory falling between cases). The evidence that this is the known environment-sensitive
check and not this change:
- the class passed alone (4 of 4);
- it passed with the three new test classes run first in the same JVM;
- pristine HEAD passed the same suite (831 run, 0 failures);
- the re-run on this tree passed (864, 0 failures).

The region path does not load the new module, and the only shared edit, `KernelParams.launch`, now
passes a grid height of 1 and is otherwise unchanged. GPU-tagged parity tests touching the changed
code pass: `GemmOnStreamTest`, `PrefillWindowRegionResidualTest`, `KQuantMmqParityTest`, `Q4KMmqParityTest`,
`Q4KDequantParityTest`, `CudaSgemmBatchedPrefillParityTest`. No ROCm code changed.

**Dormant until step 4 (README feature-complete rule).** `KQuantGemmKernel` is called only by its tests.
Nothing routes to it: `CudaMatVec.sgemmInto(DeviceQ4KMatrix, ...)` and `gemmOnStream` still dequantize.
Step 4 wires it into both, behind `--mmq` (`off` keeps the dequant route). No `CHANGELOG.md` entry yet,
because nothing user-visible changed. It comes with step 4.

**Out-of-tier changes.** `docs/agent-arch.txt` had lost the header line of the
`CudaDriverBindings / Q4KMmqKernel / DeviceQ4KMatrix` entry, which left an orphaned continuation and a
duplicated `ResidentQ4KWeight` line. The header is restored from commit `d45d2b0` in the same edit that
adds `KQuantGemmKernel`. Documentation only, not a measurement boundary.


### 2026-10-03: implementation step 4, the switch (implemented; the pinned A/B is owed to the owner)

**Owner decision applied first.** The numerical-quality threshold is restated against the fused decode
GEMV (option (a)); see the threshold block. `KQuantGemmQualityTest` now asserts it: packed mean error /
GEMV mean error reads 1.0000 on all twelve type and shape cases (limit 1.001), and 0.37% against the
0.5% ceiling.

**What changed.** `CudaMatVec` routes every batched K-quant matmul wider than `HALF_SGEMM_BATCH_MAX`
to the tiled kernel, on both prefill paths. The host-staged `sgemmInto(DeviceQ4KMatrix, ...)` still
stages the window as FP16. The region's `gemmOnStream(DeviceQ4KMatrix, ...)` reads its FP16 `xh`. Both
pack that FP16 window as Q8_1 with a new `quantize_q8_1_half` kernel (`kquant_gemm.cu`), whose
operations match `quantize_q8_1` step for step, into a new `Q8WindowScratch`, then launch the tiled
kernel. The two paths therefore stay bit-identical to each other (`GemmOnStreamTest`,
`PrefillRegionHandlerParityTest`). Device time is one new `juno.DeviceCompute` site, `gemm_kquant`
(Q8_1 packing included). It is added to the metrics module's always-written site list.
`prefill-breakdown.sh` counts it as GEMM without change. `Q4KDequantScratch` is off the default path. The
dequantizing route stays as `dequantOnStream`/`gemmHalfOnStream`, used when the tiled module fails to
load (warned once). `--mmq off` loads FP16 weights, never `DeviceQ4KMatrix`, so it still gives the FP16
baseline. A package-private `CudaMatVec.dequantizeBatchedKQuant` forces the dequant route for parity
tests (below); it is not a flag. Help text, `docs/howto.md`, `docs/agent-arch.txt` and `CHANGELOG.md`
are updated.

**Tests first.** These were updated or added before the routing and seen failing for the right reason:
- `KQuantGemmParityTest.batchedSgemmRunsThisKernel`: `sgemm` output equals the tiled kernel on the
  FP16-rounded window bit for bit, 3 types x widths 9, 64, 512. 9 of 9 failed on value mismatch;
- `DeviceComputeSpansTest`: a `gemm_kquant` site and no `gemm_half`;
- `DeviceStagingSpansTest`: no `juno.WeightDequant`;
- `CudaMatVecScratchLifetimeTest`: scratch smaller than one FP16 weight matrix;
- `JfrMetricsExtractorDeviceComputeTest`: `gemm_kquant` always written.

All pass after it.

**Two existing tests needed changes, recorded rather than smoothed over.**
- `PrefillWindowRegionResidualTest`'s two out-of-memory recovery cases left 2 MiB of device memory and
  relied on the first packed GEMM growing a whole FP16 weight matrix. That growth is gone; the first
  allocation is now a 46 KB Q8_1 window, which fits. The recovery logic is unchanged. The ballast now
  fills the device down to 4 KiB chunks with no headroom, so the Q8_1 scratch growth is the allocation
  that fails. Both cases pass.
- `GpuAttentionHandlerParityTest` (attention kernel on against off, relative L2 <= 0.025, calibrated with
  FP16 activations) failed on Qwen3-1.7B's second multi-decode stream at 0.0379. Measured against pristine
  HEAD, the prefill sites moved from 0.0003-0.0004 to 0.008-0.009 on both models. That is the 8-bit
  activation rounding turning a 3e-4 attention difference into whole rounding steps. A sqrt(3e-4 x 8e-3)
  estimate per matmul, compounded over 28 layers, gives about 0.008, as measured. It is not an attention
  fault. `expected-general`. The bound was not raised. Both of the test's backends now take the dequant
  route (`dequantizeBatchedKQuant`), which reproduces the pre-switch numbers exactly (0.000291 to 0.014879
  on Qwen3, identical to HEAD), and it passes. Pinning it to `--mmq off` was not possible: FP16 weights
  for Phi-3.5-mini do not fit the 8 GiB card.

**Greedy decode against the pre-switch build.** `smoke-long-prompt-prefill.sh --baseline-jar` (run
unmodified, candidate `5393faf01b594879` against baseline `c9f8e7a74c4187b7`): exit 0, and **all 12 greedy
answers identical** (TinyLlama and Mistral 7B, `static` and `continuous`, 128, 512 and 2048 tokens). That
is consistent with the >= 99% greedy-agreement threshold, but not its scored reading, which needs 512
generated tokens against the FP16 path and is owed at step 6.

**Unpinned smoke of the gate invocation** (single readings, not scorable): prefill at 512, candidate over
baseline, 1.26x on TinyLlama and 1.65x on Mistral 7B; generation 0.96x and 1.00x.

**Owed to the owner: the pinned same-hour A/B** (README "No-regression gates tighter than the floor").
`dist/packed-kquant-ab/` holds `baseline-shaded.jar` (HEAD `f02bdae`, `c9f8e7a74c4187b7`, built from
`git archive`), `candidate-shaded.jar` (this tree, `5393faf01b594879`) and `run-gate.sh`.
- Part P: all four sweep models at `n_prompt=512`. Prefill B/A >= 1.30 on mistral-7b and >= 1.10 on
  tinyllama; generation B/A >= 0.95 on every model.
- Part W: tinyllama and mistral-7b at `--prefill-batch` 16 and 64. Prefill B/A >= 1.10 at each.

Command: `sudo -v && bash dist/packed-kquant-ab/run-gate.sh`.


### 2026-10-03: step 4's pinned A/B (owner run), prefill met, generation missed on one model

Owner run of `dist/packed-kquant-ab/run-gate.sh`, 19:59 to 20:40 UTC, all runs pinned. Published as
[`20261003T195923Z-packed-kquant-ab`](../perf-compare/20261003T195923Z-packed-kquant-ab/INDEX.md).
Candidate over baseline, median of three:

| Threshold | Reading | Result |
|---|---|---|
| prefill mistral-7b at 512, >= 1.30x | 1.554x | met |
| prefill tinyllama at 512, >= 1.10x | 1.226x | met |
| prefill at width 16, >= 1.10x | tinyllama 3.157x, mistral-7b 5.140x | met |
| prefill at width 64, >= 1.10x | tinyllama 1.824x, mistral-7b 2.527x | met |
| generation, every sweep model, >= 0.95x | tinyllama 0.975x, **qwen2.5-3b 0.945x**, Phi-3.5-mini 0.951x, mistral-7b 1.017x | **missed (qwen2.5-3b)** |

Qwen2.5-3B and Phi-3.5-mini prefill (recorded, not gated): 1.413x and 1.368x.

Against the decomposition in the step 2 record, Mistral 7B at 512 came in at 1.55x against the 1.30x it
asked for. The kernel beat cuBLAS by more than the 1.40x that row needed. `host-specific`.

**The generation miss.** The change does not reach decode: a generated token is one activation row and
never takes the batched path, and the decode kernels and their PTX are unchanged. Per request, GC pauses
(8 to 15 ms peak) and allocation are equal on both builds to within 1%. The baseline's own Qwen2.5-3B
readings span 8.4% (28.14 to 30.61, its run 2 low on every model), more than the 5.5% gap.

An **unpinned** probe afterwards read generation B/A 0.986x (TinyLlama) and 0.994x (Qwen2.5-3B) at 512
prompt tokens, and 1.00x at an 8-token prompt. It was four alternations, published beside the A/B as
`unpinned-decode-probe.txt`. That argues against a reproducible decode cost. By the README's rule an
unpinned run does not score a gate, so **the miss stands**.

Decision raised with the owner, options:
- (a) a confirmation run declared before it starts:
  `sudo -v && PART=P N=5 RUN_LABEL=-confirm bash dist/packed-kquant-ab/run-gate.sh`. That is five
  alternations, scored on the median of five, published beside this run. If any model's generation
  reads below 0.95x again, it is treated as a real regression and investigated before step 5.
  Recommended.
- (b) treat it as a regression now and profile decode on Qwen2.5-3B before any re-run.

Exit criterion "every threshold met, or reported missed with its number" stays unticked until this is
resolved.


### 2026-10-03: step 4's confirmation run (owner decision (a)), gate met

Owner decision: option (a), the confirmation run with its rule fixed before it started. The owner ran
`PART=P N=5 RUN_LABEL=-confirm bash dist/packed-kquant-ab/run-gate.sh`, 21:48 to 22:21 UTC, all ten runs
pinned, same jars. Published in the same directory
([`20261003T195923Z-packed-kquant-ab`](../perf-compare/20261003T195923Z-packed-kquant-ab/INDEX.md),
`ab-readings-confirm.json`). Median of five, candidate over baseline:

| Threshold | Reading | Result |
|---|---|---|
| generation, every sweep model, >= 0.95x | tinyllama 0.989x, qwen2.5-3b 0.999x, Phi-3.5-mini 1.009x, mistral-7b 1.016x | met |
| prefill mistral-7b at 512, >= 1.30x | 1.487x | met |
| prefill tinyllama at 512, >= 1.10x | 1.227x | met |

Qwen2.5-3B and Phi-3.5-mini prefill (recorded): 1.494x and 1.447x.

**Step 4's gate is met under the declared rule.** The first run's 0.945x stays on record. Across both
runs generation sat between 0.945x and 1.017x with no model consistently below 1.0, at a host
run-to-run spread of up to about 15% across runs. The host's absolute generation also drifted down over
the session on both builds alike (Qwen2.5-3B 30.1 to 25.7 t/s). `host-specific`.

Width thresholds (16 and 64) were met in the first run and not re-run; the declared confirmation covered
part P only.

Still owed before the tier closes:
- the VRAM threshold and the reserve (step 5);
- the scored greedy-decode >= 99% over 512 tokens;
- the full cross-surface matrix, the vision and LoRA gates, and the closing `n_prompt` 128/512/2048
  sweeps (step 6).


### 2026-10-04: implementation step 5, the reserve sized from the prefill window

**Step 1 check.** `check-plan-thresholds.sh`: ok (19 tier files, milestone table checked).

**Plan against code (re-verified at HEAD `5a77981`).** Batched K-quant matmuls take the tiled kernel
on both prefill paths, and `Q4KDequantScratch` is reached only when the tiled module fails to load (or
through the test-only `dequantizeBatchedKQuant`). `DeviceScratchBudget.reserveBytes` still reserved the
widest FP16 weight matrix plus 40%, on every upload path, including `--mmq off` and FP32 residency, which
never dequantize. Two claims did not hold, and they changed what this step could deliver:
- **The reserve's javadoc said staging and activations were "much smaller" than the weight term.**
  False since Tier 01B's prefill-window region. Worked out from `PrefillWindowRegion.Window`'s buffers, a
  512-row window with its attention scores needs 88 / 101 / 162 / 253 MiB (TinyLlama, Qwen2.5-3B,
  Mistral 7B, 30B), against the old reserve's 31 / 60 / 157 / 319 MiB. "Remove the weight term" shrinks
  the reserve only if it promises a window narrower than 512.
- **`PrefillBatchOptions.ADAPTIVE_BYTES_PER_TOKEN` (64 KiB, documented as a conservative worst case)
  is 1.7x to 6.3x below a region window's cost per row** (114 to 412 KB before scores), so the adaptive
  chunk could choose a window the free memory could not hold.

Raised with the owner, options: (a) reserve the region's narrowest window (64 rows) and correct the chunk
sizer to the real footprint; (b) the same reserve without the sizer fix; (c) reserve a 512-row window
(barely a shrink); (d) stop and re-plan. *Owner decision 2026-10-03: (a).* The sizer fix is outside
step 5's literal scope and is recorded under "Out-of-tier changes" below.

**What changed.**
- `PrefillWindowFootprint` (new): device bytes of a region window of `rows` rows, term for term with the
  `Window` constructor at its capacity (rows rounded up to 64), plus attention scores (rows x heads x
  context x 4) and the backend's Q8_1 copy at the widest matmul input. `PrefillWindowRegion.open` takes
  its capacity from it, and `Window.deviceBytes()` reports what a window holds.
- `DeviceScratchBudget`: `reserveBytes(windowBytes, dequantScratchBytes)` = (window + dequant) x 1.4 +
  `ALLOCATOR_HOLDBACK_BYTES`. The window term is `RESERVED_WINDOW_ROWS = 64` rows at the KV mirror's
  initial 64 positions. `dequantScratchBytes` (the old weight-shaped term, no margin) is added only when
  K-quant weights are uploaded packed and `KQuantGemmKernel.tryLoad()` fails. The KV mirror term is
  unchanged.
- `ForwardPassHandler.prefillWindowDeviceBytes(rows)` (default 0). Llama, Phi-3 and Qwen3 report their
  region's footprint, and `VisionAwareForwardPassHandler` delegates.
- `PrefillChunkDefaults.resolve(..., handlers)` and `PrefillBatchOptions.adaptiveChunkSize(free,
  windowBytes)`: the widest window (32 to 65536 rows, binary search) whose summed footprint fits half
  of free VRAM. With no shard on the region, the 64 KiB a token still applies. `ConsoleMain` local mode
  and `JunoPlayer` pass their handlers.

**Found by the GPU test: the allocator withholds memory the free-memory query reports.** The first run
of `PrefillReserveDeviceTest` filled the device down to the shrunk reserve, and the window's first
256 KB allocation failed. With the device filled to allocation failure, `memGetInfo` still reported 44,
52 and 54 MiB free in three processes (stable within one process), and not even 64 KiB could be
allocated. The upload stop rule reads that query, so a window-sized reserve would have left nothing
allocatable after the upload. The old 157 to 319 MiB reserve hid this. Hence
`ALLOCATOR_HOLDBACK_BYTES = 64 MiB`. `host-specific`: measured on this GTX 1080, driver 580.173.02,
with a desktop session.

**Tests first.**
- `PrefillWindowFootprintTest` (7, new): hand-computed Mistral 7B figures, capacity rounding, the fused
  Q/K/V term, quadratic scores, monotonicity. 5 of 7 failed against stubs, on value.
- `DeviceScratchBudgetTest` (19, rewritten): the window-shaped reserve, the dequant term only for the
  fallback, the holdback, and **`aLayerThatFitsOnlyUnderTheShrunkReserveIsUploaded`**: a 30B layer that
  the packed reserve uploads and the dequant reserve does not. 9 failed against stubs; the holdback case
  was added after the GPU finding and failed against a zero constant.
- `PrefillBatchOptionsTest` (+7) and `PrefillChunkDefaultsTest` (+5). On the first run they never
  executed: node's failures stopped the reactor. They were then run against the stub behaviour put back
  temporarily, and 9 failed on value; the implementation was restored.
- `PrefillReserveDeviceTest` (3, new, `gpu`):
  - **The footprint equals what a window allocates**, at 9, 64, 65 and 512 rows, separate and fused.
    It passed on its first run against the stubs, because both sides returned 0, so it was not seen
    failing.
  - **What the reserve guarantees beyond the holdback** (a hole of that size, the rest of the device
    filled, the hole freed) runs a 64-row window through two packed layers bit-identical to an
    unconstrained run, while the dequant route in the same state runs out of memory. This is the
    device-OOM regression test the tier asks for.
  - **With the device full, the query reports no more than the holdback.**
  - Measuring by filling down to a `memGetInfo` target was tried first and dropped: the leftover came
    back in fragments smaller than 2 MiB. The hole method passed 3 of 3 runs.

**Measured** (unpinned; VRAM and layer counts, not timings). HEAD `5a77981` (`ce4db246795fd33f`)
against this change (`2d55d27b07a8a564`), one engine at a time, published as
[`20261003T235146Z-packed-kquant-reserve`](../perf-compare/20261003T235146Z-packed-kquant-reserve/INDEX.md).
Peak per-process VRAM during a 512-token prefill over peak during a short-prompt decode, same process:

| Model, nodes | GPU layers (base / cand) | Kept free | Chunk (base / cand) | Prefill / decode peak (base / cand) |
|---|---|---|---|---|
| tinyllama-1.1b, 1 | 22 / 22 | not binding | 52202 / 4738 | 1.098 / 1.098 |
| qwen2.5-3b, 1 | 36 / 36 | not binding | 40266 / 5213 | 1.045 / 1.045 |
| Phi-3.5-mini, 1 | 32 / 32 | not binding | 38848 / 3659 | 1.125 / 1.126 |
| mistral-7b, 1 | 32 / 32 | not binding | 23880 / 2601 | **1.043 / 1.043** |
| llama-1-30b, 1 | 22 / **23** | 416 / 197 MiB | 3471 / 320 | **1.034 / 1.012** |
| llama-1-30b, 3 | 23 / 23 | 351 / 132 MiB | 2480 / 32 | 1.007 / 1.001 |

- **VRAM threshold (`<= 1.05x` on mistral-7b and llama-1-30b): met**, on both builds. The baseline
  already met it, because step 4 removed the dequant scratch from prefill. `expected-general`: what a
  prefill adds over decode is now window-shaped on any device.
- **The shrink is proven on a real model.** The 30B on one node holds 23 layers instead of 22.
  `host-specific` for the count. On three nodes the total stays 23: the third shard's first layer,
  uploaded before its per-layer cost is measured, meets the allocator's refusal (handled) instead of
  the stop rule. That behaviour predates this change; it is now visible because less is held back.
- **Models that fit whole: no change** in layers, peaks or fallbacks. Their chunk is now 2601 to 5213
  rows, so every 128-, 512- and 2048-token prompt is still one window. No sweep reading moves, so no
  pinned A/B was owed for this step.
- **Fallbacks on the 30B remain, on both builds.** A 508-token prompt's KV mirror on 23 layers needs
  about 311 MiB, which no reserve on this card holds. The mirror grows at run time and was never
  reserved for, by design. Each fallback lands on a correct path. The 30B wall times (1 node: prefill
  2467 / 2118 s, decode 189 / 168 s; 3 nodes: 2450 / 1967 s and 167 / 196 s) are single unpinned readings
  dominated by the 37 CPU layers. They are recorded, not scored.

**Out-of-tier changes.** The adaptive prefill chunk sizer (`coordinator`: `PrefillBatchOptions`,
`PrefillChunkDefaults`; call sites in `ConsoleMain` and `JunoPlayer`) is a batching change outside step
5's literal scope, made by owner decision. It is a measurement boundary only for prompts longer than the
new adaptive width (2601 rows or more on the sweep models), and for partially offloaded models. It
invalidates no published baseline: every published sweep and smoke prompt is 2048 tokens or shorter, and
no prior 30B reading is used as a reference.

**Regression runs.**
- Unit suite, all eleven modules: 2,058 run, 0 failures, 49 skipped, BUILD SUCCESS. The node module's
  888 include every GPU-tagged test, and the known free-VRAM check in
  `PrefillRegionHandlerParityTest` passed this time.
- `smoke-long-prompt-prefill.sh --baseline-jar` (the HEAD `5a77981` jar), run unmodified: exit 0, 48 PASS
  checks. Every greedy answer is identical to the baseline's on TinyLlama and Mistral 7B, `static` and
  `continuous`, at 128, 512 and 2048 tokens.

**Docs.** `docs/agent-arch.txt` (`DeviceScratchBudget`, `PrefillWindowFootprint`, `CudaMatVec`
scratch, `GpuContext`, `PrefillChunkDefaults`), `docs/howto.md` (device memory with `--gpu-layers
auto`, the `--prefill-batch` row, the facade), `docs/performance.md` (the superseded per-token figure)
and `CHANGELOG.md` (Session 108).

**Raised with the owner, not acted on.**
- The KV mirror's run-time growth is outside both the reserve and the chunk sizer. On a card filled to
  the reserve, it falls back for any prompt longer than the reserve's 64 positions.
- `ALLOCATOR_HOLDBACK_BYTES` is a constant measured on one card and driver.

*Owner decisions 2026-10-04, both as recommended:* the KV mirror's growth on a nearly full card moves to
[Tier 02](TIER-02-attention-long-context.md) (a dated scope note after its item 8), and
`ALLOCATOR_HOLDBACK_BYTES` stays at 64 MiB, to be re-measured in
[Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md) (item 10) and on any other GPU.


### 2026-10-04: before step 6, greedy output is not reproducible on the batched prefill path, raised with the owner

**Step 1 check.** `check-plan-thresholds.sh`: ok (19 tier files, milestone table checked).

**Plan against code (re-verified at HEAD `59c53cc`, tree: only the untracked `.github/`).** Step 5's state
holds: `CudaMatVec.sgemmInto(DeviceQ4KMatrix, ...)` and `gemmOnStream` both reach `gemmPackedOnStream`, which
runs the tiled kernel and dequantizes only when the module fails to load or under the test-only
`dequantizeBatchedKQuant`; `--mmq` help text says `off` and `auto` differ in prefill; the CLIP encoder is
built on `CpuMatVec.INSTANCE` (`LlavaHandlerFactory`). One claim was false: checklist row 10 assumed LoRA
playback composes its delta against the batched packed path. It does not reach that path at all (row 10
corrected in place). Not a scope change for this step; whether playback should take the tiled kernel is a
decision below.

**What was built.** `scripts/performance-tests/smoke-packed-kquant-matmul.sh` (the tier's smoke script):
the four sweep models, both schedules, a 512-token prompt, exactly 512 greedy tokens (`min_tokens` =
`max_tokens`). It asserts HTTP 200, prompt length within 10%, 512 tokens with one stream chunk each,
streamed text equal to unstreamed, and the server log showing packed residency with the tiled kernel loaded
and no dequantizing fallback. It records TTFT and the engine's peak device memory. With `--baseline-jar` it
scores token agreement with that build (agreeing prefix over 512, `--min-agreement` 0.99). Test first: run
against the pre-switch jar posing as the candidate (TinyLlama, static), the packed-path assertion failed
for the right reason (no tiled module in that build).

**Found by that first run: two identical greedy requests in one process can give different text.**
Streamed and unstreamed answers parted at about token 50 on the pre-switch build. Both requests prefilled
the same 506-row window from position 0, so this is not prefix reuse. A determinism study followed:
TinyLlama, 512-token prompt, 512 greedy tokens, 2 fresh servers x 5 identical requests per configuration,
unpinned (raw outputs in `dist/packed-kquant-close/determinism/`, git-ignored):

| Configuration | Distinct outputs of 10 | First difference (tokens) |
|---|---|---|
| candidate `59c53cc` (packed GEMM, GPU attention) | 4 (7 identical) | 24, 24, 94 |
| pre-switch `f02bdae` (FP16 dequant GEMM, GPU attention) | 3 (8 identical) | 46, 46 |
| candidate, `--prefill-batch 1` (per-token prefill) | 1 | none |
| candidate, `--gpu-attention off` | 1 | none |

- **The variance predates this tier and is not the packed kernel.** Both builds show it at similar rates.
  It disappears with per-token prefill, and with GPU attention off while the window still runs batched.
  It sits in the GPU attention step of the batched prefill window (Tier 01B's region), which per-token
  prefill does not use. No CUDA source in `node/src/main/cuda/` uses atomics, so an unsynchronized read
  (for example of the window's K/V or score buffers) is the likely mechanism. Not yet located.
  `expected-general` for "a race gives run-to-run variance"; `host-specific` for the rates.
- **Consequence for this tier.** The greedy threshold ("agrees with the FP16 path on >= 99% of the
  first 512 tokens") cannot be scored as written: the FP16 route disagrees with itself at token 46 in 2
  of 10 requests, so a candidate-against-FP16 reading measures the shared variance, not the packed
  route. The new smoke script's streamed-against-unstreamed and agreement checks inherit the same flake.
  Step 4's 12-of-12 identical answers (16 tokens) and step 5's (16 tokens) stopped before the earliest
  divergence seen here (token 24).

**Raised with the owner** (step 6 not continued past this point):
1. The prefill-window attention variance. (a) Locate and fix it now as an out-of-tier change (rule 9),
   then score the greedy threshold unchanged (recommended: it is a likely race in shipped code, on the
   default GPU path of every sweep model); (b) move it to Tier 02 (attention) and restate this tier's
   greedy threshold against noise: candidate-against-FP16 agreement no lower than FP16-against-FP16
   agreement over the same repeats, both with GPU attention off; (c) score the threshold with
   `--gpu-attention off` on both sides and record the variance as a Tier 02 item.
2. LoRA playback prefill on the tiled kernel. (a) Leave playback on per-row GEMV; record the missed
   speed-up as a Tier 12 item (recommended: playback numerics stay what `compare-lora.sh` measures, and
   this tier's scope named the `DeviceQ4KMatrix` route only); (b) route `matVecBatch` for Q4-resident
   weights through the tiled kernel in this tier.

Prepared for the owner, not yet run: `dist/packed-kquant-close/run-gate.sh` (part L: LoRA train and
playback, same-hour A/B against `f02bdae` with governor and turbo pinned; part B: closing sweeps at
`n_prompt` 128, 512 and 2048). Its baseline tree and candidate jar (`425285a3daef839e`, HEAD) are staged.
Not to be run until decision 1 is made, because a fix would change the candidate.

*Owner decisions 2026-10-04: 1 (a), locate and fix the prefill-window attention variance now as an
out-of-tier change, then score the greedy threshold unchanged; 2 (a), LoRA playback stays on per-row GEMV,
and the tiled-kernel route for Q4-resident playback is recorded as a Tier 12 item.*

### 2026-10-04: the prefill-window attention variance located and fixed (owner decision 1 (a))

Decision 2 (a) is recorded as [Tier 12](TIER-12-lora.md) scope item 5.

**Cause.** `gqa_attention.cu`'s `block_reduce` is called twice per block, for the softmax maximum and
then the sum of exponents, and both calls go through one `__shared__ warpVals` array. The function ended
with every thread reading `warpVals[0]` after a barrier, with no barrier after that read. Warp 0 could run
through pass 2 and write its partial sum into `warpVals[0]` before a slower warp had read the maximum.
That warp then exponentiated its scores against a partial sum, so its share of the softmax was scaled
inconsistently with the others'. The outcome depends on warp scheduling. A 512-row window launches
512 x heads blocks per layer, which is why it showed there; per-token prefill and decode launch
`heads` blocks and never showed it in the study. `expected-general`: a shared-memory reuse race with no
barrier is a defect on any CUDA device; `host-specific`: how often it fires.

**Tests first.** `GqaAttentionReproducibilityTest` (new, `gpu`) launches a TinyLlama-shaped 512-row
window 100 times with fixed inputs, asserts every output is bit-identical to the first, and checks
sampled rows against the `GqaMath` CPU oracle. Before the fix it failed for the right reason: the output
differed from launch 46 on. After the fix it passed 6 of 6 runs. `GqaAttentionKernelParityTest` passes
(2 of 2).

**Fix.** `block_reduce` reads the result into a register, then runs a second `__syncthreads()` before
returning. The PTX was regenerated with the compiler that built the committed file (CUDA 12.0,
`CL-32267302`). Recompiling the unmodified source first reproduced the committed PTX byte for byte, so
the PTX diff is exactly two added `bar.sync 0` instructions, one per call.

**End to end** (TinyLlama, the step's determinism setup: 2 servers x 5 identical requests, 512-token
prompt, 512 greedy tokens; outputs in `dist/packed-kquant-close/determinism/`):

| Build | Distinct outputs of 10 (before the fix) | After the fix |
|---|---|---|
| candidate (packed GEMM) | 4 | **1** (`cff55ad0807c6ea7`) |
| pre-switch `f02bdae` (FP16 GEMM) | 3 | **1** (`f02bdae` plus only this fix, `463af3e52bc9801d`) |

**What this does to the greedy threshold.** With the variance gone, the packed route's output parts from
the FP16 route's at token **24 of 512** on TinyLlama, static. Agreement is 0.047 against `>= 0.99`.
Both are now deterministic, so this is the routes' real difference, and every numerically different pair
of routes parts early on this prompt:

| TinyLlama, 512 greedy tokens, first difference at token | |
|---|---|
| packed route against FP16 route | 24 |
| packed route against per-token prefill (decode kernels throughout) | 24 |
| FP16 route against per-token prefill | 63 |
| FP16 route against CPU attention (`--gpu-attention off`) | 46 |

So the pre-tier FP16 path does not meet ">= 99% of 512 tokens" against its own per-token oracle either.
The scored reading on every model and schedule comes from `smoke-packed-kquant-matmul.sh --baseline-jar`
(the fixed reference), below.

**Scored reading: the greedy threshold is missed on every cell.** `smoke-packed-kquant-matmul.sh
--baseline-jar` (candidate `cff55ad0807c6ea7` against `f02bdae` plus the fix, `463af3e52bc9801d`),
published as
[`20261004T050618Z-packed-kquant-smoke`](../perf-compare/20261004T050618Z-packed-kquant-smoke/INDEX.md).
All 48 candidate checks pass (packed path active, 512 tokens, streamed equal to unstreamed, VRAM and TTFT
recorded). Agreement with the FP16 route, as the agreeing prefix of 512 tokens (threshold `>= 0.99`):

| Model | static | continuous | packed vs per-token (static) | FP16 vs per-token (static) |
|---|---|---|---|---|
| tinyllama | 24 (0.047) | 24 | 24 | 63 |
| qwen2.5-3b | 0 (0.000) | 0 | 30 | 0 |
| Phi-3.5-mini | 66 (0.129) | 66 | 66 | 244 |
| mistral-7b | 68 (0.133) | 99 (0.193) | **512** | 68 |

Per-token prefill (`--prefill-batch 1`) runs the decode kernels, with the same Q8_1 activation rounding
decode uses for every token. Neither prefill route consistently tracks it better. The packed route
reproduces it for all 512 tokens on Mistral 7B, where the FP16 route parts at 68. On Qwen2.5-3B the FP16
route parts from it at the first token, between two equally plausible openings. `expected-general`: a
greedy sequence of 512 tokens crosses near-ties, and any change in rounding flips the first of them.
The threshold as written measures where that happens, not which route is closer to the model. The
kernel-level guard, packed error equal to the decode GEMV's within 0.1% (restated by the owner at step
3), is met.

**Raised with the owner (decision 3), options:**
- (a) Replace the greedy clause with what the data can score: the candidate is deterministic (10 of 10
  identical requests on TinyLlama, the study above, extended to one repeat on every sweep model in the
  closing smoke), and the first divergence from the FP16 route is recorded per model and schedule, with
  the outputs read for coherence. The kernel-level numerical-quality threshold stays the binding accuracy
  gate. Recommended.
- (b) Replace it with a teacher-forced top-1 agreement over the FP16 route's 512 tokens (`>= 99%` per
  position), which is insensitive to a single flip. This needs per-position logits that no API or test
  harness exposes today, so it means a new in-process GPU test before the tier can close.
- (c) Keep it: the tier cannot close on it, and the miss is reported with the numbers above.

**Out-of-tier changes** (rule 9). The attention barrier fix (`node/src/main/cuda/gqa_attention.cu`, its
PTX; `GqaAttentionReproducibilityTest`) is outside this tier's scope: the attention kernel is Tier 02's,
and the defect dates from Tier 01B's region (present at `f02bdae`). Made by owner decision 1 (a).
**It is a measurement boundary for output, and nominally for throughput.** Output: GPU-attention greedy
text can change wherever the race used to fire, so every earlier greedy-identity reading longer than about
24 tokens with GPU attention on is pre-fix. The step 4 and step 5 16-token identities and Tier 01B's
64-token region-greedy IT are below or near that. Throughput: two barriers per attention block, decode
included. The owner's closing gate gains a part A, a pinned A/B of the pre-fix HEAD jar against the fixed
jar on every sweep model, generation and prefill `>= 0.95x`. It invalidates no published reference until
then.

### 2026-10-04: step 6 regression runs on the fixed build; a step 5 regression found on tensor-parallel

**Unit suite, eleven modules** (fixed tree). Run 1: registry 93, lora 116, kvcache 78, health 23 pass; node
889 run, **1 failure**, 44 skipped; the six modules after it were skipped. The failure was
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: with the device full, the
free-memory query reported 71,172,096 bytes against the 64 MiB (67,108,864) allowance. It failed 3 of 3 re-run
alone (70.9 to 71.2 MiB), and **2 of 2 on a clean HEAD `59c53cc` tree** (71.3 and 73.7 MiB), so it is not the
attention fix. At step 5 the same reading was 44 to 54 MiB and passed 3 of 3. The driver now withholds about
70 MiB on this host (desktop session sharing the card). Not loosened: it is the constant the owner kept at
64 MiB pending Tier 10. The six skipped modules then ran: 860 run, 0 failures, 5 skipped. **Total 2,059 run,
1 failure (above), 49 skipped.** `GqaAttentionReproducibilityTest` is in node's 889.

**`ModelLiveRunnerIT`** (`-Pintegration`, the four sweep models, after `mvn install -DskipTests` of the fixed
tree): 35 of 36 checks pass. Check 9 (512-token prefill, both schedules, packed path active) passes on all
four (prompts of 573, 560, 583 and 561 tokens), with all 16 greedy tokens identical across per-token,
static-window and continuous prefill (first divergence -1). The failure: **Phi-3.5-mini, checks 7 and 8
(tensor-parallel)**, `cudaMalloc failed: rc=2` on one node.

**Cause, located** (`ModelLiveRunner`, Phi-3.5-mini, `JUNO_VERBOSE`; intermittent: 9 of 9 alone in tensor
mode; with all checks 1 pass and 1 fail, then 1 pass and 1 fail with a temporary stack-trace line in
`EmbeddedNodeServer`, since reverted). The failing allocation is `CudaMatVec.ensureFp32Scratch` in
`sgemvInto`, the decode GEMV's lazily grown FP32 scratch, from `Phi3TransformerHandler.projectFusedQ4`. A
tensor-parallel cluster on one card runs three node processes that each load the whole model (until Tier
09), concurrently. They uploaded 32, 32 and 31 layers. Each stops when device-wide free memory falls below
its own reserve, so the three share one reserve between them, not three. Since step 5 that reserve is a
64-row window plus the 64 MiB allowance, which the driver does not hand out (about 70 MiB now, above). The
KV mirror then falls back to the CPU (handled, warned), but the scratch has no fallback, and the forward
pass fails. Before step 5 the reserve also carried the widest FP16 weight matrix plus 40% (about 140 MiB
for Phi-3.5-mini), which absorbed it. **A step 5 regression for several processes on one device**:
checklist row 8, whose note asks that the reserve be computed per shard. It does not occur with one
process per device. `host-specific`: the margins; `expected-general`: lazily grown buffers outside the
reserve fail when the reserve is shared.

**Raised with the owner (decision 4), options:**
- (a) Allocate the decode GEMV's scratch at upload time, sized for the widest matrix the backend holds,
  so it is held before the stop rule runs and is part of each process's own footprint; and turn a
  device-memory failure in a forward pass into the existing fallback-and-warn path rather than a failed
  request. Fixes the failure for any number of processes per device. Then re-run `ModelLiveRunnerIT`.
  Recommended.
- (b) Add the scratch to the reserve's arithmetic only. Still one reserve shared by every process on the
  device, so it narrows the window but does not close it.
- (c) Declare several node processes on one device unsupported with `--gpu-layers auto` (fail closed at
  load with an explicit error), and run the tensor-parallel IT with an explicit `--gpu-layers`. Tier 09
  owns multi-GPU tensor parallelism.

**Not yet run** (all after decision 4, since (a) or (c) changes the candidate): `mvn verify -pl
juno-master`, the `-Pgpu` ITs, the vision gate, the unpinned LoRA readings, the earlier tiers' smoke
scripts, and a short unpublished `n_prompt=2048` check of the closing sweep on TinyLlama (2048-token
context).

*Owner decisions 2026-10-04: decision 3 (a), the greedy clause is replaced (threshold block restated);
decision 4 (a), the decode GEMV scratch is allocated at upload and a device-memory failure in a forward
pass falls back instead of failing the request.*

### 2026-10-04: decision 4 (a) implemented, the decode GEMV scratch held at upload

**Tests first.** `GemvScratchHeldAtUploadTest` (new, `gpu`): for packed K-quant, FP16 and FP32 uploads, the
single-row products and the shared-input `sgemvSameX` product leave `CudaMatVec.scratchDeviceBytes()`
exactly where the uploads left it. It failed 3 of 3 for the right reason (the uploads held no scratch).
`DeviceMatVecFallbackTest` (new, 2 cases: device allocation failures are absorbed, anything else is
rethrown unchanged) was written after the class it tests and passed on its first run. It is a
regression test, not one watched failing.

**What changed.**
- `CudaMatVec.holdGemvScratch`, called from `upload`, `uploadHalf` and `uploadKQuant`, grows the
  product scratch to that matrix's need: input `max(rows, cols)`, output three times that (the widest
  `sgemvSameX` group, separate Q, K and V with multi-head attention), and the Q8_1 input for packed
  weights. At most a few hundred KB per backend. It is held before the handler's stop rule reads the free
  memory. A failure frees the matrix and rethrows into the existing per-layer upload handling.
- `DeviceMatVecFallback` (new): Llama's `matVecProjection`, `matVecProjectionSameX` (falls back to the
  per-matrix path) and `matVecLayer`, Phi-3's `matVecFused`, `projectFusedQ4` (now passed the fused
  quantized tensor), `matVecProj` and output projection, and Qwen3's `matVecLayer` and output projection.
  Each catches a device out-of-memory failure, warns once per handler, and computes that product on the
  CPU from the quantized tensor. Nothing else is caught.
- ROCm is unchanged (`RocmMatVec` has its own scratch). `NEEDS-AMD-HARDWARE` to verify whether it has
  the same exposure.

**Result.** `GemvScratchHeldAtUploadTest` 3 of 3; `CudaMatVecScratchLifetimeTest` 5 of 5;
`CudaAttentionNormScratchLifetimeTest` 4 of 4. `ModelLiveRunner` on Phi-3.5-mini with every check, the
configuration that failed 2 of 4 before: **4 of 4 runs, 9 of 9 checks each**. The tensor-parallel nodes held
32, 32 and 32 layers, the KV mirror fell back to the CPU as before (handled), and the new CPU fallback never
fired. Holding the scratch is what fixed it; the fallback is the guard. Jar `5ab4c78c505a4cef`.
`expected-general`: a buffer grown lazily outside a reserve that several processes share fails on any
device. `host-specific`: the layer counts (32, 32, 32) and how often the old code failed (2 of 4).

**Out-of-tier change** (rule 9): a decode-path change in `CudaMatVec` and three handlers, outside the
tier's GEMM scope; made by owner decision 4 (a) to repair a step 5 regression. **Not a measurement
boundary by design:** decode allocates less, never more, and a `try` costs nothing until it throws.
It is covered anyway by part A of the owner's closing gate (the pre-fix HEAD jar against this build,
generation `>= 0.95x`).


### 2026-10-04: step 6, the closing matrix (agent part)

All runs below are on the closing build: HEAD `59c53cc` plus the attention barrier fix and the decode
GEMV scratch held at upload (jar `5ab4c78c505a4cef`, staged for the owner as
`dist/packed-kquant-close/candidate-shaded.jar`). The reference for vision and LoRA is `f02bdae`, the last
build before this tier (`cdd4de12314667d2`, `git archive`, package only). The greedy reference is `f02bdae`
plus only the attention fix (`463af3e52bc9801d`). One sequential chain, 07:29Z to 10:20Z.

| Check | Result |
|---|---|
| `check-plan-thresholds.sh` | ok |
| `mvn test`, eleven unit-test modules | **2,064 run, 0 failures, 49 skipped**, one green run. `PrefillReserveDeviceTest`'s allocator check passed this time (it failed earlier the same day at 71 MiB, on clean HEAD too: it moves with the desktop's use of the card) |
| `ModelLiveRunnerIT`, `-Pintegration`, four sweep models | **36 of 36**, Phi-3.5-mini tensor-parallel included. Check 9 (512-token prefill, both schedules, packed path): prompts of 573, 560, 583 and 561 tokens, 16 greedy tokens identical across per-token, static-window and continuous prefill on every model |
| `mvn verify -pl juno-master` (stub cluster ITs) | 20 of 20 |
| `mvn verify -pl juno-master -Pgpu` (TinyLlama) | 10 of 10 (`PrefillRegionGreedyIT`, `GpuAttentionDivergenceIT`, `GpuForwardPassIT`) |
| `smoke-packed-kquant-matmul.sh --baseline-jar` (this tier's script; four models, both schedules) | **48 of 48**: packed path active, no fallback, 512 tokens, streamed equal to unstreamed (determinism, 100% of cells); agreeing prefixes with the FP16 route as published (24, 0, 66, 68 static; 99 Mistral 7B continuous), unchanged by the decode change ([`20261004T050618Z-packed-kquant-smoke`](../perf-compare/20261004T050618Z-packed-kquant-smoke/INDEX.md)) |
| `smoke-long-prompt-prefill.sh --baseline-jar` (unmodified, against the fixed reference) | 48 of 48; greedy text (16 tokens) identical on all 12 cells |
| `smoke-consistency.sh`, `smoke-grammar.sh`, `smoke-tools.sh` (unmodified) | 54, 19 and 14 PASS, 0 failures each |
| `smoke-gpu-residency.sh --models tinyllama...,mistral...` (unmodified) | 0 failures: decode region active on 8 of 8 and 11 of 11 layers, no per-request memory growth, greedy identical on against off. Cluster pipeline and tensor answer and leave no node JVM; their text differs from local mode's from word 21 (reported by the script, not a failure). Expected from this tier: cluster prefill runs one token per pass (decode kernels), local prefill now runs the packed route, and the two part at token 24 on TinyLlama (determinism table above). At Tier 01B's close they matched, when local prefill ran the FP16 route. `expected-general` for the mechanism; `host-specific` for the word |
| `compare-vision.sh --gpu`, three runs per build, alternated | latency 1.002x, decode 0.995x; gate (`<= 1.25x`, `>= 0.80x`) **met**; six captions identical |
| `compare-lora.sh --gpu --reps 3`, per build | playback 1.023x; gate (`>= 0.80x`) **met**. Train 1.000x, every repetition 15 passes to loss 1.1797: reading only, since the train gate (`>= 0.95x`) is pinned and owed ([`20261004T091233Z-packed-kquant-close-vision-lora`](../perf-compare/20261004T091233Z-packed-kquant-close-vision-lora/INDEX.md)) |
| `compare-llama-cpp.sh --gpu --n-prompt 2048`, TinyLlama, unpinned, unpublished | the closing sweep's 2048 setting works on the 2048-context model: 2,048 of 2,048 prompt tokens, parity ok. Single reading pp 0.123x, for information only. It covered only TinyLlama, which is why the owner's first 2048 sweep met Phi-3.5-mini's heap limit (owner-gate record below) |

Vision and LoRA unchanged: `expected-general`, since neither reaches the changed code (the CLIP encoder and
Phi-2 run on the CPU; LoRA keeps FP32-resident frozen weights and per-row playback GEMVs). The ratios
themselves are `host-specific`.

**Closing cross-surface matrix.**

| # | Surface | Resolution | Evidence |
|---|---|---|---|
| 1 | CPU inference | N/A for the kernel (CUDA-only); correctness unchanged | unit suites; `smoke-consistency.sh` CPU legs |
| 2 | CUDA GPU inference | PASS | `ModelLiveRunnerIT` check 9 on four models; packed smoke 48 of 48; `-Pgpu` ITs; throughput by the step 4 pinned A/B and the owner's closing sweeps |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE | the kernel, the attention fix's PTX and the scratch hold are CUDA-only; `RocmMatVec` is unchanged and its non-hardware tests pass in the unit suite; whether ROCm's lazily grown GEMV scratch has the same multi-process exposure needs a device |
| 4 | Static schedule | PASS | check 9 (static window); packed and long-prompt smokes (`static` cells) |
| 5 | Continuous schedule | PASS | check 9 (32-token chunks); packed smoke (`continuous` cells, widths where the old dequant amortized worst); the step 4 width 16 and 64 A/B |
| 6 | Single-node local mode | PASS | every smoke above runs `./juno local` |
| 7 | Pipeline-parallel cluster | PASS (correct; per-token prefill, so the kernel is not reached until Tier 09 item 5) | `ModelLiveRunnerIT` checks 1 to 6 on four models; residency smoke's cluster leg; `ThreeNodeClusterIT`. The reserve is computed per node process (step 5 record) |
| 8 | Tensor-parallel cluster | PASS after the decision 4 fix | `ModelLiveRunnerIT` checks 7 and 8 on four models (Phi-3.5-mini: 4 of 4 runs alone, then in the IT); `TensorParallelClusterIT`; residency smoke's cluster leg |
| 9 | LoRA training | PASS (exempt, unchanged) | `compare-lora.sh` train 1.000x, same loss; `LoraMmqPolicyTest` in the unit suite |
| 10 | LoRA playback | PASS (unchanged; does not reach the tiled kernel, row 10 note; Tier 12 item 5) | `compare-lora.sh` playback 1.023x; `LoraQ4KPlaybackParityTest` |
| 11 | Vision | PASS (unchanged; CLIP on `CpuMatVec.INSTANCE`, Phi-2 text half on the CPU) | `compare-vision.sh` 1.002x / 0.995x, captions identical |
| 12 | OpenAI REST surface | PASS | packed and long-prompt smokes over `/v1/chat/completions`, streamed and not; grammar and tools smokes |
| 13 | Native REST surface | PASS | `smoke-consistency.sh` `/v1/inference` rounds |
| 14 | CLI | PASS | `--mmq` help says `off` multiplies FP16 weights in prefill and `auto` the packed ones; `docs/howto.md` |
| 15 | JVM embedding facade | N/A: no new embedder-invocable capability; the facade's prefill takes the same kernel through the same handlers | `juno-player` unit tests |

**Owed to the owner (pinned; needs prompt-free sudo):** `sudo -v && bash dist/packed-kquant-close/run-gate.sh`
(about 3 to 4 hours). Part A: pre-fix HEAD `59c53cc` (`425285a3daef839e`) against the closing build,
alternated three times, four sweep models at 512, generation and prefill `>= 0.95x`. Part L: LoRA train
speed `>= 0.95x` and playback `>= 0.80x` against `f02bdae`, governor and turbo pinned. Part B: closing
sweeps at `n_prompt` 128, 512 and 2048, published. The 512 sweep scores the milestone (`>= 0.20x` on every
sweep model), and the 2048 sweep fills the README's Tier 02 reference cell. The script refuses to run if the
tree's jar is not the staged candidate; re-copy it after any rebuild.


### 2026-10-04: the owner's closing gate (pinned), and two defects it exposed

Owner run of `dist/packed-kquant-close/run-gate.sh`, 11:05Z onward, clocks pinned throughout.

**Part A, the two out-of-tier fixes** (pre-fix HEAD `425285a3daef839e` against the closing build
`5ab4c78c505a4cef`, alternated three times, n_prompt 512; [`20261004T110558Z-packed-kquant-close-ab`](../perf-compare/20261004T110558Z-packed-kquant-close-ab/INDEX.md)):
prefill 0.996 / 0.985 / 1.008 / 0.972, generation 1.006 / **0.952** / 0.999 / 0.996 (TinyLlama, Qwen2.5-3B,
Phi-3.5-mini, Mistral 7B). Gate `>= 0.95x`: **met**. Qwen2.5-3B generation is lower in all three pairs, so
0.952 is not one outlier. No mechanism is known (the fixes add two barriers per attention block and take
work out of decode). Step 4 saw the same pattern on this model and its declared confirmation read 0.999.
Recorded, not explained. `host-specific`.

**Part L, LoRA** (same directory): train speed 1.000x, playback 1.030x against `f02bdae`. Gates met.

**Part B, closing sweeps** ([`20261004T113210Z`](../perf-compare/20261004T113210Z/INDEX.md) at 128,
[`20261004T114812Z`](../perf-compare/20261004T114812Z/INDEX.md) at 512; every row scorable, prompt-token
parity exact, the staged candidate jar):

| Model | pp 512 (Tier 01B close) | **pp 512 (closing)** | pp 128 | 512 over 128 | pp 2048 | 2048 over 512 | tg 128 |
|---|---|---|---|---|---|---|---|
| tinyllama | 0.254x | **0.317x** | 0.524x | 0.606 | 0.121x | 0.382 | 0.363x |
| qwen2.5-3b | 0.281x | **0.429x** | 0.616x | 0.697 | 0.170x | 0.395 | 0.432x |
| Phi-3.5-mini | 0.146x | **0.226x** | 0.303x | 0.743 | not measured | - | 0.545x |
| mistral-7b | 0.285x | **0.468x** | 0.729x | 0.642 | not reached | - | 0.627x |

Read against the program target (README):
- **This tier's milestone, GPU pp `>= 0.20x` at 512 on every sweep model: met**, binding on
  Phi-3.5-mini at 0.226x. Retired in the README's table.
- Tier 01B's milestone rows (retired) stay met.
- **End-of-plan GPU pp `>= 0.25x`: met on three models at 512, missed on Phi-3.5-mini (0.226x).**
- **Post-plan anchor `0.40x`: reached at 512 on Qwen2.5-3B (0.429x) and Mistral 7B (0.468x)**, not on
  TinyLlama or Phi-3.5-mini. Reported, not a gate.
- End-of-plan GPU tg `>= 0.70x`: missed (Phi-3.5-mini 0.545x, Mistral 7B 0.627x). No tg work in this tier.
- **Tier 02's rows moved the wrong way, as Tier 01B's decomposition predicted for this kind of change:**
  512 over 128 fell from 0.845 to 0.606 (binding TinyLlama), and 2048 over 512 reads 0.38 to 0.40 on the
  two models measured. The matmuls got 1.2x to 1.6x faster while attention did not, so attention's share
  of a long window grew. `expected-general`. Both rows are Tier 02's, and their reference cells are updated.

**The 2048 sweep did not finish.** TinyLlama and Qwen2.5-3B completed (published by hand as
[`20261004T120622Z-partial`](../perf-compare/20261004T120622Z-partial/INDEX.md)). On Phi-3.5-mini every
prefill request died with `java.lang.OutOfMemoryError: Java heap space` (6 GiB fixed heap) and returned no
response. Each one waited out the harness's 7,200-second `curl` timeout, three of them in the first
repetition (07:21 to 13:21 local). The owner stopped the run during the second repetition; clocks, governor
and turbo were restored. Mistral 7B was not reached. **The agent's 2048 pre-check had covered only
TinyLlama**, which is why this was not caught before the owner's run.

**Defect 1: an `Error` in a generation thread leaves the request unanswered.**
`RequestScheduler.dispatchSingle` and the batch path catch `Exception` only. An `OutOfMemoryError` escapes,
the request's future is never completed, and the HTTP client waits until its own timeout. A fail-closed
violation: the server should answer with an error. `expected-general`. Pre-existing, not introduced by
this tier.

**Defect 2: Phi-3.5-mini cannot prefill a 2048-token prompt in a 6 GiB heap** with the default prefill
window (one 2048-row window, since step 5's chunk sizing gives Phi-3.5-mini 3,659 rows). Phi-3's region
returns Q, K and V to the host for host-side RoPE and attention (32 heads, `kvHeads=32`, head dim 96), and
the host window, its attention scores and the host KV cache are all sized by the window. Not yet measured
term by term. First exercised by this sweep: no earlier run prefilled 2048 tokens on Phi-3.5-mini.

**Raised with the owner (decision 5), options:**
- (a) Fix defect 1 now (complete the future on any `Throwable`, map an `OutOfMemoryError` to an HTTP error,
  in every dispatch path including the continuous engine), measure Phi-3.5-mini's 2048-row host footprint
  term by term, then decide between a smaller footprint and a larger fixed sweep heap for that model. Then
  re-run part B at 2048 only. Recommended.
- (b) Fix defect 1 only; give Phi-3.5-mini a larger fixed heap in the sweep (`COMPARE_HEAP` per model) and
  re-run part B at 2048; record the footprint as a Tier 02 item.
- (c) Close the tier with the 2048 reading on two models, and move both defects to later tiers.

**Raised with the owner (decision 6):** whether the two closing sweeps become the new GPU reference column in
the README's program-target table, as Tier 01B's did. Recommended: yes, for 128 and 512, with the 2048
column added when complete.

*Owner decisions 2026-10-04: decision 5 (a), fix defect 1 in every dispatch path, then measure Phi-3.5-mini's
2048-row host footprint and choose between a smaller footprint and a larger sweep heap; decision 6, the 128
and 512 closing sweeps become the GPU reference column.*


### 2026-10-04: decision 5 (a), defect 1 fixed, Phi-3.5-mini's 2048-token footprint measured, and a third blocker

**Defect 1 fixed: an `Error` in generation now fails the request on every dispatch path.**
- Tests first: `RequestSchedulerErrorTest` (new, coordinator, 3 cases). A pipeline throws
  `OutOfMemoryError` from every forward pass; on single dispatch, batched dispatch and the continuous
  schedule, the request must end within 10 s with the error as its cause, and the next request must be
  served. Before the fix all three timed out at 10 s: the request never ended.
- Fix: `RequestScheduler`'s single path, batch path and batch collector loop, and `ContinuousBatchEngine`'s
  admit, slot finish and engine loop catch `Throwable`, complete the futures exceptionally, and keep
  serving. Before, an `Error` escaping the continuous engine's loop ended its thread, so every later
  request on that schedule hung. The HTTP handlers already turn a failed future into HTTP 500 (unstreamed)
  or a closed stream (streamed).
- Result: coordinator 372 run, 0 failures. End to end, Phi-3.5-mini at a 2056-token prompt with a 6 GiB
  heap now answers **HTTP 500 with `java.lang.OutOfMemoryError: Java heap space` in 4.6 s**, where the sweep
  waited 7,200 s per request.
- `expected-general`. Out of tier (coordinator), by owner decision 5 (a); not a measurement boundary
  (nothing changes unless a request throws an `Error`).

**Phi-3.5-mini's 2048-token footprint, measured** (one engine, `--gpu`, one greedy request of 2,056 prompt
tokens, `-Xlog:gc` and JFR; jar `bd3e162c95e530bc`):

| Fixed heap | Result | Live heap after GC, peak |
|---|---|---|
| 6 GiB (the sweep's) | HTTP 500, out of heap, in 4.6 s | 6,128 MB, 6 full collections |
| 8 GiB | HTTP 200, 35 s | 6,185 MB, no full collection |
| 10 GiB | HTTP 200, 33 s | 7,114 MB (garbage not yet collected) |

The live set is about 6.1 GB. Two terms make up most of it:
- **The host KV cache, about 3.2 GB.** `DenseKvTensor` doubles its capacity, and its `f16` element type
  stores 32-bit floats. A 2,048-token prompt plus one generated token needs position 2,048, one past
  2,048, so every layer's K and V double to 4,096 positions: 2 x 32 layers x 3,072 x 4 bytes x 4,096 =
  3.2 GB, where 1.6 GB would hold it. Phi-3.5-mini has 32 KV heads, so its per-token KV is three times
  Mistral 7B's. `expected-general`.
- **Host copies of the quantized weights, about 2.2 GB**, kept for the CPU fallbacks (including the new
  `DeviceMatVecFallback`).

Choice made under decision 5 (a): **a larger sweep heap for Phi-3.5-mini at 2048**, not a smaller
footprint in this tier. Exact-size KV growth and real FP16 KV storage are KV-cache work (Tier 03), and they
change memory behaviour on every model. The 2048 re-run passes `COMPARE_HEAP=8g` for Phi-3.5-mini only,
which the harness records as an `explicit` heap. The fixed table, and so the 128 and 512 references, are
unchanged. Mistral 7B at 2048 runs at its fixed 9 GiB (checked unpinned: pp 0.157x, no error).

**A third blocker: the harness withholds Phi-3.5-mini's 2048 reading, and the cause is the tokenizer.**
Unpinned, with 8 GiB, the harness ran Phi-3.5-mini at 2048 to completion but withheld the prefill
figure. 322 ms of the 30,195 ms request fell outside the forward-pass spans, against the integrity
check's fixed allowance of `300 + 3 x generated tokens` ms (303). That residual, per model and prompt
length, from the published sweeps:

| Model | 128 | 512 | 2048 |
|---|---|---|---|
| tinyllama | 5 ms | 17 ms | 277 to 284 ms |
| mistral-7b | 6 ms | 18 to 22 ms | 286 ms |
| Phi-3.5-mini | 8 ms | 25 to 26 ms | 322 ms |
| qwen2.5-3b | 4 ms | 5 to 7 ms | 12 to 14 ms |

It grows with the square of the prompt on the three SentencePiece models and not on Qwen2.5-3B (GPT-2 BPE
with a pre-tokenizer split). Timing `GgufTokenizer.encode` directly on the harness's prompt (`x x x ...`):

| Model | 128 words | 512 words | 2048 words |
|---|---|---|---|
| tinyllama | 4.3 ms | 23.7 ms | 262 ms |
| Phi-3.5-mini | 0.9 ms | 13.5 ms | 270 ms |
| mistral-7b | 0.8 ms | 13.7 ms | 253 ms |
| qwen2.5-3b | 0.8 ms | 2.6 ms | 5.0 ms |

The SentencePiece path merges over the whole unsplit text (`GgufTokenizer.mergeWholeRuns` /
`mergeInPlace`), so the work is quadratic in prompt length. It accounts for almost all of the residual.
Tokenization runs before the forward-pass span, so it does not enter the prefill figure. It does
enter time to first token, about 0.25 s at 2,048 tokens and, extrapolated at x16 per x4, several seconds at
8,192. `expected-general`. With turbo off (pinned) the tokenizer runs slower, so the owner's re-run would
withhold Phi-3.5-mini again, and TinyLlama and Mistral 7B (about 20 ms under the allowance unpinned)
could follow.

**Raised with the owner (decision 7), options:**
- (a) Make the SentencePiece merge sub-quadratic (a priority queue over adjacent pairs, the same
  highest-score, leftmost-first order), with a parity test that it produces identical tokens to the current
  implementation over a corpus that includes the sweep prompts. Out of tier: tokenizer work is Tier 04B's.
  It also cuts time to first token on every SentencePiece model.
- (b) Measure tokenization: a JFR span around prompt encoding, which the harness subtracts from the
  residual before applying the unchanged allowance. The integrity check still catches a misread clock. The
  quadratic tokenizer is recorded as a Tier 04B item with these numbers. Smaller change, no change to tokens.
  Recommended.
- (c) Close this tier with the 2048 reading on two models, and carry the Phi-3.5-mini and Mistral 7B 2048
  readings, the tokenizer and the KV footprint to Tiers 02, 03 and 04B.

*Owner decision 2026-10-04: decision 7 (b), a JFR span around prompt encoding that the harness subtracts
from the residual; the quadratic tokenizer is recorded as a Tier 04B item.*


### 2026-10-04: decision 7 (b), prompt encoding recorded and subtracted by the span check

**Tests first, each seen failing for the right reason:**
- `JfrMetricsExtractorPromptEncodeTest` (metrics, 2): `juno.PromptEncode.count` and `.total_ms` absent
  before. The duration case first asserted `>= 50.0` for sleeps of 30 and 20 ms, and read 49.89 in the full
  suite. That was a defect in the test, not the product (sleep granularity against the recorder's clock),
  so it now checks a 45-to-250 ms band;
- `PromptEncodeEventTest` (coordinator, 2): one `juno.PromptEncode` event per request carrying the
  prompt's token count, on the static and continuous schedules; 0 events before;
- `compare-llama-cpp.sh --selftest` (5 new checks, the exact Phi-3.5-mini 2048 shape: prefill 29,799 ms,
  request 30,195 ms). Without an encode figure the request is still withheld, and a 633 ms clock misread
  is still caught with one; with 270 ms recorded it passes, its prefill reading is published, and
  `prompt_encode_ms` is in the check. Three failed before the change; the two guards already held.

**What changed.** `PromptEncodeEvent` and `PromptEncoder` (new, coordinator): every generation path
(`GenerationLoop`'s single and batch paths, `ContinuousBatchEngine`) encodes through it.
`JfrMetricsExtractor` emits `juno.PromptEncode.count` / `.total_ms`. `juno-perf.jfc` enables the event.
`compare-llama-cpp.sh` subtracts the encode time from both residuals before the unchanged allowance
(`300 + 3 x generated tokens` ms, `-25` floor) and records `prompt_encode_ms`. A build without the event
contributes 0, which is the previous rule, so earlier published runs are unchanged.

**End to end** (unpinned, `--no-publish`, `COMPARE_HEAP=8g`, jar `4f851a1fb94abbd1`): Phi-3.5-mini at 2048
now publishes its reading. Encoding 306 ms, residual after it 25.7 ms (allowance 303), prompt 2,048 of 2,048
tokens; pp 0.074x for information (a third of its 0.226x at 512, the long-context gap Tier 02 owns).
`expected-general`: encoding outside every forward-pass span, and its quadratic growth on an unsplit
SentencePiece merge. `host-specific`: the 306 ms.

**Regression runs** (this build; the jar staged for the owner is `a90122b2855bb826`, the same source
rebuilt by `install`):
- `compare-llama-cpp.sh --selftest`: 97 ok, exit 0.
- Unit suite: **2,071 run, 1 failure**, across runs. Registry, lora, kvcache and health pass; node 889 with
  1 failure; tokenizer, sampler, coordinator (374) and vision pass; metrics 84 and juno-player 118 after the
  test fix above. The node failure is `GpuAttentionHandlerParityTest`'s device-wide free-memory check
  (106 MB short). Re-run alone: fail (22 MB short), pass, pass. node is unchanged since the green run of the
  same day. It is the same environment-sensitive class as `PrefillRegionHandlerParityTest`'s check (the
  card also drives the desktop), and it is not loosened.
- `mvn verify -pl juno-master` 20 of 20; `ModelLiveRunnerIT` four sweep models **36 of 36**.

**Out-of-tier change** (rule 9): coordinator, metrics and the harness, by owner decision 7 (b). **Not a
measurement boundary for throughput:** one JFR event per request. It changes which repetitions the
harness publishes on long SentencePiece prompts, and only by removing a known non-clock term from the
check. Docs: `docs/agent-arch.txt`, `docs/performance.md` (span check), Tier 04B item 5 (the quadratic
tokenizer), Tier 03 item 6 (the host KV footprint).

**Owed to the owner (pinned):** `sudo -v && bash dist/packed-kquant-close/run-2048.sh` (about 1 to 1.5
hours). It runs all four sweep models at 2048 on this build, in two published invocations
(TinyLlama, Qwen2.5-3B and Mistral 7B at their fixed heaps; Phi-3.5-mini with an explicit 8 GiB). The
script refuses to run unless the tree's jar is the staged candidate.


### 2026-10-04: the 2048 re-run (owner run, pinned), and the tier closed

Owner run of `dist/packed-kquant-close/run-2048.sh`, jar `a90122b2855bb826` (the staged candidate),
clocks pinned, both invocations published:
[`20261004T220015Z`](../perf-compare/20261004T220015Z/INDEX.md) (TinyLlama, Qwen2.5-3B, Mistral 7B at their
fixed heaps) and [`20261004T222758Z`](../perf-compare/20261004T222758Z/INDEX.md) (Phi-3.5-mini, explicit
8 GiB heap, recorded as `explicit` in every result file). Every row is scorable, every prefill is at 2,048 of
2,048 prompt tokens, and no repetition was withheld. Phi-3.5-mini's three repetitions recorded 266 to 269 ms
of prompt encoding, subtracted by the span check, leaving residuals of 26 to 27 ms.

| Model | pp 512 | **pp 2048** | 2048 over 512 | tg (2048 run) |
|---|---|---|---|---|
| tinyllama | 0.317x | **0.121x** | 0.381 | 0.323x |
| qwen2.5-3b | 0.429x | **0.180x** | 0.420 | 0.442x |
| Phi-3.5-mini | 0.226x | **0.099x** | 0.441 | 0.589x |
| mistral-7b | 0.468x | **0.154x** | 0.329 | 0.598x |

- **The first parity-corrected 2048 reading is now complete**, and it fills the README's Tier 02 milestone
  cell (2048 over 512 `>= 0.90`: reference 0.329, binding Mistral 7B). The two partial readings from the
  interrupted sweep agree to within 0.01 (TinyLlama 0.121x against 0.121x, Qwen2.5-3B 0.180x against
  0.170x). The 2048 figures are in the README's reference column (decision 6) and in `docs/performance.md`.
- Read against the program target: at 2048 no model reaches the end-of-plan `>= 0.25x`. Prefill falls to
  0.33 to 0.44 of its 512-token ratio, as attention grows with context. The tiled long-context attention
  kernel is Tier 02's, and this is its starting point. `expected-general` for the direction;
  `host-specific` for the factors.

**Marker audit** (the exit criterion). Every execution-record entry was checked for conclusions without a
`host-specific` or `expected-general` marker. The conclusions scope item 4 names are all marked: the
integer GEMM over `cublasGemmEx` (step 2, `host-specific`), the width at which dequantization stops
mattering (step 1, `expected-general`), and the packed path winning on compute (steps 3 and 4,
`host-specific`). Four entries had none. Markers were added to three: the scratch fix, the closing
matrix's cluster divergence and vision/LoRA lines, and the encoding figures. The fourth, the harness's
recorded window width, is tooling with no performance conclusion and now says so.


## Exit criteria

- [x] Item 1's per-width breakdown published, with the GEMM's share at 512 per sweep model and the
      throughput-threshold branch chosen in writing before the kernel was built.
      *Checked 2026-10-03: GEMM 32.9% / 49.3% / 37.8% / 53.3% of a 512-token window (TinyLlama,
      Qwen2.5-3B, Phi-3.5-mini, Mistral 7B), widths 9 to 512, unpinned attribution run
      [`docs/perf-compare/20261003T060301Z-packed-kquant-width-breakdown`](../perf-compare/20261003T060301Z-packed-kquant-width-breakdown/INDEX.md);
      throughput branch chosen (Mistral 7B >= 30%) in the step 2 record, no kernel code written.*
- [x] A tiled packed GEMM for Q4_K, Q5_K and Q6_K is the default for batch > `HALF_SGEMM_BATCH_MAX`,
      reading and writing through Tier 01B's prefill-window region; `Q4KDequantScratch` is off the
      default path for these formats; `--mmq off` still gives the FP16 baseline.
      *Checked 2026-10-04: `CudaMatVec.gemmPackedOnStream` serves both `sgemmInto(DeviceQ4KMatrix, ...)`
      and the region's `gemmOnStream`; dequantization only when the module fails to load
      (`KQuantGemmParityTest.batchedSgemmRunsThisKernel`, `DeviceStagingSpansTest`: no
      `juno.WeightDequant`); `--mmq off` uploads FP16 weights and never a `DeviceQ4KMatrix`. End to end the
      packed smoke asserts the kernel loaded with no fallback on every model and schedule, 48 of 48.*
- [x] The forward-pass reserve shrunk, proven by the fits-only-under-the-new-budget test.
      *Checked 2026-10-04: the reserve is the 64-row prefill window plus margin and a 64 MiB allocator
      allowance; the weight matrix is reserved only when the tiled kernel cannot load.
      `DeviceScratchBudgetTest.aLayerThatFitsOnlyUnderTheShrunkReserveIsUploaded` (unit) and
      `PrefillReserveDeviceTest` (GPU: a wide window runs in exactly what the reserve guarantees; the
      dequant route does not). On a real model, llama-1-30b on one node holds 23 GPU layers instead of 22
      (197 MiB kept free instead of 416), in
      [`docs/perf-compare/20261003T235146Z-packed-kquant-reserve`](../perf-compare/20261003T235146Z-packed-kquant-reserve/INDEX.md).*
- [x] Every threshold above met, or reported missed with its number.
      *2026-10-04: the VRAM threshold is met (prefill peak over decode peak 1.043 on Mistral 7B, 1.012 on
      llama-1-30b, against <= 1.05x;
      [`docs/perf-compare/20261003T235146Z-packed-kquant-reserve`](../perf-compare/20261003T235146Z-packed-kquant-reserve/INDEX.md)).
      Still owed at step 6, by the executor: the scored greedy-decode agreement over 512 tokens, and the
      vision and LoRA gates.*
      *2026-10-04, step 6: met. Throughput (step 4 pinned A/B, `20261003T195923Z-packed-kquant-ab`);
      numerical quality, kernel clause (`KQuantGemmQualityTest`) and the restated greedy clause (determinism
      on 100% of cells, divergence recorded:
      [`docs/perf-compare/20261004T050618Z-packed-kquant-smoke`](../perf-compare/20261004T050618Z-packed-kquant-smoke/INDEX.md));
      vision 1.002x / 0.995x and LoRA playback 1.023x
      ([`docs/perf-compare/20261004T091233Z-packed-kquant-close-vision-lora`](../perf-compare/20261004T091233Z-packed-kquant-close-vision-lora/INDEX.md)).
      **Still owed, by the owner (pinned):** LoRA train `>= 0.95x` (unpinned 1.000x) and the decode
      no-regression of the two out-of-tier fixes (gate part A), both in `dist/packed-kquant-close/run-gate.sh`.*
      *2026-10-04, owner gate: both met. Part A generation 0.952x to 1.006x and prefill 0.972x to 1.008x;
      LoRA train 1.000x, playback 1.030x
      ([`docs/perf-compare/20261004T110558Z-packed-kquant-close-ab`](../perf-compare/20261004T110558Z-packed-kquant-close-ab/INDEX.md)).
      Every threshold is met.*
- [x] The milestone (GPU pp >= 0.20x at `n_prompt=512` on every sweep model) read from the closing
      sweep and reported met or missed with the number; the first `n_prompt=2048` sweep published and
      the README's Tier 02 milestone reference cell filled from it in the same change.
      *2026-10-04: the milestone is met, 0.226x (Phi-3.5-mini, binding) to 0.468x
      ([`docs/perf-compare/20261004T114812Z`](../perf-compare/20261004T114812Z/INDEX.md)), and retired in
      the README. The 2048 half is incomplete: two of four models
      ([`docs/perf-compare/20261004T120622Z-partial`](../perf-compare/20261004T120622Z-partial/INDEX.md)),
      with the README cell filled from them. Phi-3.5-mini ran out of Java heap and Mistral 7B was not
      reached. Owed after decision 5.*
      *Checked 2026-10-04: the 2048 sweep is complete on all four models, 0.099x to 0.180x
      ([`docs/perf-compare/20261004T220015Z`](../perf-compare/20261004T220015Z/INDEX.md),
      [`docs/perf-compare/20261004T222758Z`](../perf-compare/20261004T222758Z/INDEX.md)), and the
      README's Tier 02 cell is filled from it (0.329, binding Mistral 7B).*
- [x] Every conclusion in the execution record carries a `host-specific` or `expected-general` marker.
      *2026-10-04: open. The step 6 entries mark their conclusions, but the record has not been audited
      end to end; done at close, after the owner's gate, so the sweep's conclusions are marked in the same pass.*
      *Checked 2026-10-04: audited end to end; markers added where missing (closing entry, "Marker audit").*
- [x] Cross-surface checklist fully resolved.
      *Checked 2026-10-04: the closing matrix in the step 6 record. 12 PASS, 2 N/A with reasons, ROCm
      NEEDS-AMD-HARDWARE. Throughput on row 2 is read by the owner's closing sweep, the other rows on
      correctness. Evidence (not published): the unit suite, ITs and earlier smokes listed there.*
- [x] Perf gate published; `compare-llama-cpp.sh` pp and tg ratios recorded in this file against the
      program target, Tier 01B's milestone rows and the post-plan anchor.
      *Checked 2026-10-04: the owner-gate record above, from
      [`docs/perf-compare/20261004T113210Z`](../perf-compare/20261004T113210Z/INDEX.md) and
      [`docs/perf-compare/20261004T114812Z`](../perf-compare/20261004T114812Z/INDEX.md) (pinned); compare-lora
      and compare-vision in `20261004T110558Z-packed-kquant-close-ab` and
      `20261004T091233Z-packed-kquant-close-vision-lora`.*
- [x] Docs (`docs/agent-arch.txt`, `docs/howto.md`, `docs/performance.md`) updated in Juno-native
      language; the `--mmq` help text says `off` and `auto` now differ in prefill.
      *2026-10-04: done except the closing sweep figures, which `docs/performance.md`'s new "Packed K-quant
      prefill matmul" section receives after the owner's gate part B. `agent-arch.txt`, `howto.md` and the
      help text are updated.*
      *2026-10-04, after the gate: the 128 and 512 figures are in `docs/performance.md`. Kept open for the
      2048 figures (decision 5).*
      *Checked 2026-10-04: the 2048 figures are in `docs/performance.md` too.*
- [x] `CHANGELOG.md` entry added.
      *Checked 2026-10-04: Sessions 107 (the packed prefill matmul), 108 (the reserve) and 109 (the
      attention race and the multi-process scratch fix).*
