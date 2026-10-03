# Tier 01C: Packed K-quant prefill matmul

Status: **in progress** (2026-10-03). Steps 1 to 3 done: the per-width breakdown is published, the
throughput branch is chosen (GEMM 53.3% of a 512-token Mistral 7B window, against the 30% line), and the
tiled kernel exists for Q4_K, Q5_K and Q6_K, validated at every test width but not yet routed. The
numerical-quality threshold was raised with the owner (the packed route's error is 14x the FP16 route's by
construction) and resolved the same day: restated against the fused decode GEMV (owner
decision). Step 4's switch is implemented (both prefill paths multiply packed weights; greedy answers
identical to the pre-switch build on 12 of 12 smoke requests); its pinned same-hour A/B is met
(prefill Mistral 7B 1.49x to 1.55x, TinyLlama 1.23x at 512; 1.8x to 5.1x at widths 16 and 64; generation
0.989x to 1.016x on the owner's pre-declared five-alternation confirmation, after a first run read 0.945x on
Qwen2.5-3B). Next: step 5, shrinking the forward-pass reserve. See "Execution record".
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

  **Threshold, numerical quality** (restated 2026-10-03, owner decision; see the step 3 record).
  Mean relative error against the FP32 oracle **<= 1.001x** that of the fused decode GEMV (the same
  Q8_1 activation rounding; the 0.1% admits float accumulation order only) and **<= 0.5%** absolute,
  on every tested shape of `KQuantGemmQualityTest`; greedy decode agrees with the FP16 path on
  **>= 99%** of the first 512 tokens at temperature 0. *Was: mean relative error <= 1.0x that of the
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
  the metadata gains fields. The step 2 run predates it. Its width is in its directory names, and its
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


## Exit criteria

- [x] Item 1's per-width breakdown published, with the GEMM's share at 512 per sweep model and the
      throughput-threshold branch chosen in writing before the kernel was built.
      *Checked 2026-10-03: GEMM 32.9% / 49.3% / 37.8% / 53.3% of a 512-token window (TinyLlama,
      Qwen2.5-3B, Phi-3.5-mini, Mistral 7B), widths 9 to 512, unpinned attribution run
      [`docs/perf-compare/20261003T060301Z-packed-kquant-width-breakdown`](../perf-compare/20261003T060301Z-packed-kquant-width-breakdown/INDEX.md);
      throughput branch chosen (Mistral 7B >= 30%) in the step 2 record, no kernel code written.*
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
