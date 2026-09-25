# Tier 04C: Packed-weight matmul — eliminating dequantize-to-FP16

Status: not started
Gap analysis refs: none directly — adjacent to §1.1, but the gap analysis treats quantization as a
*format coverage* question and never asks what Juno does with a format once it is supported. See
"Why this tier, why now".

All line numbers below are a snapshot taken at branch `67-gap-inference`, HEAD `1f90b68`. Re-verify
each one against current source before acting on it (README, opening paragraph).

## Objective

Make packed quantized weights the compute and residency representation on the GPU for every
quantization format and every backend, so that dequantizing a weight matrix to FP16 stops being
either the default prefill path or the default residency for anything. Where a packed path genuinely
cannot exist yet, make the FP16 fallback explicit, announced, and accounted for in the VRAM budget
instead of silent.

## Why this tier, why now

Juno does not dequantize to FP16 uniformly — it does so in three distinct regimes, and the reason it
looks like one behaviour from the outside is that nothing in this plan tree owns the boundary
between them. Stated precisely, on CUDA with `--mmq auto` (the real default — see item 5):

| Regime | When | What is FP16 | Bytes per weight |
|---|---|---|---|
| **A. Packed, no dequant** | Q4_K/Q5_K/Q6_K, CUDA, batch <= 8 | nothing | 0.5625 (Q4_K) / 0.8203 (Q6_K) |
| **B. Transient dequant** | Q4_K/Q5_K/Q6_K, CUDA, batch > 8 | a whole weight matrix, in device scratch, per matmul | 0.5625 resident **plus** a 2.0 B/weight scratch peak |
| **C. Resident dequant** | every other format, and **all** formats on ROCm | every weight, for the process lifetime | 2.0 (or 4.0 — see below) |

Regime A is the good one and this tier does not touch it. `Q4KMmqKernel.launch` (`:194`) quantizes
the activation to Q8_1 and integer-dots the packed bytes with `__dp4a`; no FP16 copy of the weights
exists anywhere on that path.

**Regime B is a missing kernel, not a design decision.** Above `HALF_SGEMM_BATCH_MAX = 8`
(`CudaMatVec.java:682`), `sgemm(DeviceQ4KMatrix, float[][])` (`:858`) routes to
`sgemmQ4KBatchedGemm` (`:870`), which calls `Q4KMmqKernel.launchDequant` (`:288`) to expand the
entire packed matrix into a device FP16 scratch buffer (`Q4KDequantScratch`) and then runs
`CudaFp16GemmOps.gemmHalf` — `cublasGemmEx` with `CUDA_R_16F` operands, `CUDA_R_32F` output and
`CUBLAS_COMPUTE_32F`. cuBLAS needs a dense operand, so if the compute goes through cuBLAS the
expansion is unavoidable; the alternative is a *tiled* integer GEMM over still-packed weights, which
is the one kernel shape Juno does not have. The fused kernels in `q4k_gemv.cu` are all GEMV
(`ncols=1`), and the three `*_dequant_to_fp16` entry points (`:357`, `:388`, `:422`) exist precisely
to hand the batched case back to cuBLAS.

llama.cpp's CUDA backend makes the same three-way split and picks differently: MMVQ for `ne11 == 1`
and small batches, **MMQ** — a tiled integer matmul over packed weights, int8 tensor cores on
Turing+ and `dp4a` on Pascal — for larger batches wherever the type supports it, and the
dequantize-to-`ggml_cuda_pool_alloc<half>` plus `cublasGemmEx` path only as the fallback
(`GGML_CUDA_FORCE_CUBLAS` forces it). Juno has the two ends and not the middle, so what is a
fallback there is the standard prefill route here.

**Regime C is a residency gap with a hard VRAM consequence.** `Q4KResidentUpload.preferPacked`
(`:33`) requires `DeviceQ4KMatrix.supportsType` (`:61`), which is exactly `{Q4_K, Q5_K, Q6_K}`.
Everything else takes `LlamaTransformerHandler.dequantize` (`:2439`) to a host `float[]`, converts
to halves and uploads FP16 for the process lifetime (`:412` logs it). On ROCm this is *every*
format: `RocmMatVec` has no `uploadKQuant` override, so `GpuMatVec`'s default (`:77`) throws and
`supportsQ4KMmq` (`:99`) stays false — and where `supportsHalfResident` is false too
(`RocmMatVec.java:115`, gfx1010/gfx1011) the fallback is FP32 resident at 4.0 B/weight.

The numbers make the consequence concrete. Q4_K resident is 0.5625 B/weight; FP16 is **3.56x** that.
`mistral-7b` Q4_K_M dequantized to FP16 is roughly 14.5 GB of weights against this host's 8 GB
GTX 1080 — it fully offloads today *only* because regime A exists, and `--mmq off` is therefore not
a usable escape hatch on that model, merely a slower one that does not fit. Two further costs ride
along: the host-side `float[rows * cols]` intermediate inside `dequantize` (a 4096 x 14336 FFN
matrix is a 235 MB transient heap allocation per projection at load), and `dequantize`'s switch not
covering Q4_0 (GGML type 2), Q4_1 (3), Q5_0 (6) or Q5_1 (7) at all — so those formats have neither a
packed path nor an FP16 path and simply cannot GPU-offload.

**Regime B has a second failure mode that is a live defect, not a performance note.** Weight upload
walks layers until the allocator refuses (`LlamaTransformerHandler.java:493` catches and warns), and
nothing reserves device memory for the forward pass. The result is a model that loads, decodes
single tokens correctly, and then dies on the first prompt wide enough to cross the batch-8
threshold — because that is when a weight-shaped FP16 scratch is allocated and nothing is left. For
`llama-1-30b.Q4_K_M.gguf` the widest matmul is the 6656 x 17920 FFN pair, 227 MiB of halves. At the
time this file was written a fix was in progress in the working tree (`DeviceScratchBudget.java`,
uncommitted); item 0 below states the exit condition either way.

### What this tier does not own

The boundary with two neighbouring tiers is deliberate and both directions are load-bearing:

- **[Tier 04](TIER-04-quantization-coverage.md) ships the per-format kernels; this tier ships the
  matmul and residency *policy* that consumes them.** Tier 04 item 3 extends `Q4KMmqKernel`-style
  fused kernels to Q2_K/Q3_K/Q8_0/Q4_0 and item 4 adds the ROCm fused K-quant kernels, both stated
  as GEMV work and both gated on "within 15% of the existing Q4_K MMQ kernel." Neither item says
  anything about the batched path, and Tier 04's exit criteria are satisfiable while every one of
  those formats still dequantizes to FP16 above batch 8. This tier is what closes that.
- **[Tier 01B](TIER-01B-prefill-throughput.md) owns the data movement *around* the GEMMs; this tier
  owns what the GEMM operand is.** Tier 01B item 2 is explicit that it is not touching dispatch
  ("`CudaMatVec` already overrides `sgemm` for all three... there is no dtype falling through to a
  serial GEMV loop on CUDA") and targets the host staging of activations instead. It also records,
  without owning, the fact that every other quant format is uploaded as FP16. The two changes meet
  at the same `MatVec.sgemm` contract, which is why this tier runs after it.

### Ordering

Running order: **after [Tier 04B](TIER-04B-tokenizer-fidelity.md), before
[Tier 05](TIER-05-sampling-grammar.md)** — no ordering exception is needed, the numbering and the
table agree. It sits there for three reasons:

- **After Tier 04**, so the tiled packed GEMM is written once against the complete set of packed
  types rather than against three and then extended to eight. This is the same argument the README
  uses for running Tier 08 before Tier 06, applied in the same direction.
- **After Tier 01B**, so it builds on the non-allocating batched `MatVec` contract that tier
  introduces instead of landing a second spelling of it.
- **Before Tier 05 and everything after it**, because every later tier that publishes a
  `compare-llama-cpp.sh` pp ratio is measuring a path that this tier changes.

The 04B adjacency carries no dependency — tokenizer fidelity and packed matmul are unrelated, and
04C could equally run immediately after 04 if 04B slips.

## Scope

### In scope

0. **Reserve device memory for the forward pass before the upload consumes it.** The upload loop
   must stop at a budget that leaves room for the widest matmul the model can take, rather than at
   the first allocation failure. The budget is `hidden x max(hidden, kvDim, ffnDim)` halves plus a
   margin for staging buffers and per-request activations. This runs first because it is a
   correctness defect on today's code and because item 2 is what eventually removes the term being
   reserved — the reserve must shrink when that lands, not be left as a permanent over-reservation.
   If this shipped during Tier 01 (a fix was in the working tree when this file was written), this
   tier verifies and tests it rather than re-implementing it, and says so in its execution record.
1. **Measure the batched dequant term before building anything.** Produce, per model and per batch
   width (9, 16, 32, 64, 128, 512), the split between `launchDequant` wall time, `gemmHalf` wall
   time, and the H2D/D2H staging around them.

   **Check the spans exist before planning around them.** An earlier draft said this split could be
   read "from the JFR spans Tier 01 widened." Tier 01 widened only the `jdk.*` bucket
   (`JdkEventBucket`); it added no sub-`MatVec` spans, and `juno.MatVec` wraps staging,
   dequantization and compute in one span, so this split cannot be read off it.
   [Tier 01B](TIER-01B-prefill-throughput.md) item 1a builds the two events this item needs —
   `juno.DeviceStaging` (direction, bytes, duration) and `juno.WeightDequant` (format, rows, cols,
   duration, wrapping `Q4KMmqKernel.launchDequant` among others) — and 01B runs well before this
   tier. Verify they are present and aggregated by `JfrMetricsExtractor` before starting; if 01B
   shipped them under different names or with a narrower payload, use what it shipped and say so
   here rather than adding a second spelling. **The expected
   answer is not uniform and the plan should not pretend otherwise**: the dequant pass is paid once
   per matmul and amortizes over the batch, so its share falls as the window widens. The certain win
   from item 2 is VRAM and the removal of a failure mode; the throughput win is batch-dependent and
   is what this measurement establishes. The continuous schedule's mixed prefill/decode steps live
   in the 9-to-64 range where the amortization is worst, which is the region to look at hardest.
2. **A tiled packed-weight GEMM, so batch > 8 stops materializing FP16 weights.** Extend the
   `q4k_gemv.cu` family with a `mul_mat_q`-shaped kernel: quantize the activation *batch* to Q8_1
   once, tile over K, integer-dot with `dp4a`, accumulate in FP32, write FP32. `launchPacked`
   (`Q4KMmqKernel.java:243`) already establishes the packed-pointer launch shape for a single
   column; this is its batched counterpart. Route `sgemm(DeviceQ4KMatrix, ...)` to it and retire
   `Q4KDequantScratch` from the default path (keep the dequant entry points — item 4 needs them).

   **The hardware argument, stated as a hypothesis for item 1 to confirm rather than as a promise.**
   On GP104 (this host's GTX 1080) native FP16 arithmetic runs at 1/64 of FP32 rate, which is why
   the current path uses `CUBLAS_COMPUTE_32F` and gets FP16 only as a storage format; `dp4a` runs at
   4x FP32 rate. A packed integer GEMM should therefore win on compute as well as on memory here —
   but Pascal is the favourable case for this argument, not the general one, and a Turing-or-later
   host with FP16 tensor cores could invert it. Record the measurement, not the expectation.

   **Mark which of this tier's conclusions are host-specific.** Every number this tier produces comes
   off a 2016 GP104 with no int8 tensor cores and 1/64-rate FP16, and several of its decisions turn
   directly on that: the choice to route batch > 8 through an integer GEMM rather than
   `cublasGemmEx`, the `HALF_SGEMM_BATCH_MAX` crossover width, and the claim that the packed path
   wins on compute as well as memory. The execution record states, for each conclusion it reaches,
   whether it is expected to hold on a host with FP16 or int8 tensor cores or should be re-derived
   there — a one-line `host-specific` / `expected-general` marker per finding, not an essay. A future
   reader on a Turing-or-later GPU needs to know which of these to re-measure and which to trust.
   [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md) carries the same obligation for its CPU
   findings on this host's AVX2-without-AVX-512, no-VNNI Xeon.
3. **Packed residency for every format that has a packed kernel, on every backend.** Widen
   `DeviceQ4KMatrix.supportsType` and `Q4KResidentUpload.preferPacked` to the formats Tier 04 added
   kernels for, and make `Q4KResidentUpload` the single decision point for packed-versus-FP16 across
   all handlers (Llama, Phi-2, Phi-3, Qwen3, Qwen3-MoE) rather than a helper three of them happen to
   call. Give `RocmMatVec` the `uploadKQuant` override and `supportsQ4KMmq` truthfulness that
   Tier 04 item 4's kernels make possible (`NEEDS-AMD-HARDWARE` per the README's ROCm rule).
4. **Make the remaining FP16 fallback explicit and budgeted.** A format or backend with no packed
   path still needs to run. When that is the case it must: log once, naming the format, the backend
   and the resulting bytes-per-weight; be reflected in the VRAM budget from item 0 *before* the
   upload starts, so the "loads then dies on the first wide prompt" failure cannot return through
   the fallback; and be reportable through the same capability mechanism Tier 01B item 0 builds for
   `--gpu-attention`, rather than a second one. Keep `--mmq off` as the FP16-resident parity
   baseline — the same role `--gpu-attention off` plays — and document it as a debugging baseline
   rather than a supported deployment mode for models that do not fit in FP16.
5. **Fix the `--mmq` default drift, repo-wide.** `MmqOptions.fromEnv()` defaults to `auto`, and
   `AUTO` resolves to on whenever CUDA is present; `ConsoleMain.java:1013` says `default: auto`.
   `docs/agent-arch.txt:213` says "Default off." Per execution rule 8 this is not a one-line fix:
   grep `src/main`, `docs/` and `CHANGELOG.md` for every claim about what `--mmq` defaults to and
   what it buys, and correct the class of claim rather than the cited line. The `--help` text's
   "measured CUDA decode win vs off" also needs re-reading once item 2 lands, because at that point
   `off` and `auto` differ in prefill as well as decode.

### Out of scope

- **New quantization formats.** Tier 04 owns format coverage. This tier assumes whatever Tier 04
  landed and makes the packed path reach it; if Tier 04 slipped a format, this tier reports that
  format as still-FP16 with a number attached rather than implementing it.
- **Activation quantization below Q8_1.** The Q8_1 activation format is the existing contract in
  `q4k_gemv.cu` and stays.
- **CPU packed matmul.** `CpuMatVec` already dots packed bytes directly
  (`matVecQ4KrawInto`/`matVecQ8_0rawInto`) and never builds an FP16 copy; CPU kernel work is
  Tier 10's.
- **LoRA training residency.** Training deliberately ignores MMQ (`LoraMmqPolicy.java:58`, `:84`
  gate it to playback) and keeps FP32-resident frozen weights so `ResidentWeightMatrix`'s batched
  `sgemmBatch` path stays available. That exemption is re-verified here, not changed; Tier 12 owns
  any decision to revisit it.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference (scalar) | N/A for the new kernel — no FP16 weight copy exists on the CPU path. Used as the FP32 correctness oracle for item 2, per the README's row-1 caveat |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | item 3 gives ROCm packed residency for the first time; `NEEDS-AMD-HARDWARE` until validated. Must not leave ROCm's current FP16/FP32 path worse than it found it |
| 4 | Static schedule | primary target — prefill windows always cross batch 8 |
| 5 | Continuous schedule | the 9-to-64 mixed prefill/decode batch widths are where the dequant amortizes worst; measure here specifically, do not extrapolate from a 512-wide static window |
| 6 | Single-node local mode | primary dev surface, and where the item 0 VRAM regression test runs |
| 7 | Pipeline-parallel cluster | each shard budgets its own device; item 0's reserve must be computed per shard, not per model |
| 8 | Tensor-parallel cluster | column/row-split shards change the widest-matmul dimension the reserve is computed from — verify the budget follows the shard's real shape |
| 9 | LoRA training | exempt by design (see Out of scope) — re-verify the exemption still holds and still produces its notice, and that item 0's reserve accounts for training's FP32 residency being larger, not smaller |
| 10 | LoRA playback | `LoraMmqPolicy` allows MMQ in playback; confirm the delta-add composes against the new batched packed path at every width, and that a LoRA-modified prefill is bit-comparable to the FP16 path within the item-2 tolerance |
| 11 | Vision | the CLIP encoder is pinned to `CpuMatVec.INSTANCE` at `LlavaHandlerFactory.java:194`, so the widest batches in the system (B around 741) never reach the CUDA dequant scratch today. **Verify that is still true rather than assuming it** — if any path hands the encoder a GPU backend, B=741 through a weight-shaped scratch is the worst case in the product. `compare-vision.sh` is a required gate here either way, because the text backbone's prefill changes |
| 12 | OpenAI REST surface | N/A directly; verified indirectly through TTFT and through a model that only fits packed |
| 13 | Native REST surface | same |
| 14 | CLI | `--mmq` help text and semantics change (item 5); any new fallback notice is user-visible output and must be documented in `docs/howto.md` |

## Implementation steps

1. Land item 0 (or verify it, if Tier 01 already shipped it) and add its regression test first — a
   model that loads and then dies on the first wide prompt is a worse failure than a slow one, and
   every measurement below is taken on a build that cannot hit it.
2. Produce item 1's per-batch-width breakdown on the post-Tier-04 build and publish it under
   `docs/perf-compare/`. Record the batch width at which `launchDequant` stops being material; that
   number is what item 2's threshold is read against, and it is also the honest answer to "is this
   worth doing at 512." If the breakdown says the dequant term is negligible at every width Juno
   actually runs, say so in this file's execution record and re-scope item 2 to the VRAM argument
   alone rather than proceeding on a throughput premise the data did not support.
3. Write the tiled packed GEMM against Q4_K first, validated against the CPU FP32 oracle at every
   width in the test matrix, then extend to the other packed types in one pass.
4. Switch `sgemm(DeviceQ4KMatrix, ...)` over, re-measure immediately, and only then widen residency
   (item 3) — so a residency change and a compute change never share one measurement.
5. Shrink item 0's reserve to what the new path actually needs, and prove the shrink with the same
   test that proved the reserve (a model that fits under the new budget and not the old one).
6. Item 4's fallback accounting, then item 5's doc-drift audit.
7. Full cross-surface smoke matrix including the vision gate.

## Tests to write/upgrade before implementation

- **`DeviceScratchBudgetTest`** (exists in the working tree at time of writing): reserve is positive
  for every architecture in `models/`, follows the widest matmul rather than the hidden dimension
  alone, and rejects non-positive dimensions instead of reserving nothing.
- **A device-OOM regression test**: with a deliberately constrained budget, a model that loads must
  survive a prefill window wider than `HALF_SGEMM_BATCH_MAX`. This is the test for the failure mode
  item 0 exists to fix, and it must fail on the pre-item-0 build.
- **Tiled packed GEMM correctness**: output against the `CpuMatVec` FP32 oracle for every packed
  format, at batch widths 1, 2, 8, 9, 16, 32, 64, 128, 512 and 741 (the vision width), on both
  square and FFN-shaped matrices, including a `cols` that is a single super-block and one that is
  not a multiple of the tile.
- **A numerical-quality comparison, not just a tolerance check**: mean and max relative error
  against the FP32 oracle for the packed path and for the current dequant-to-FP16 path, same
  matrices, published side by side. The packed path must be at least as accurate.
- **Greedy-decode parity**: same prompt and seed through the packed and FP16 paths, token-identical
  or characterised — this is the same class of evidence Tier 01B requires for the FP16 KV mirror.
- **Device-memory accounting**: no leak across repeated windows (reuse Tier 01's assertion), and
  peak device bytes during a wide prefill asserted against the decode-time figure.
- **`ModelLiveRunnerIT`**: a 512-token prefill check on a model whose FP16 residency would not fit
  in the available VRAM — the end-to-end form of the whole tier.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier04c-packed-matmul.sh` — drives
  `./juno local` on both schedules across every packed format, asserts correct output, asserts the
  fallback notice appears exactly when a format has no packed path, and records peak VRAM.
- **Perf gate (required)**: this is a MatVec and forward-pass change by definition.
  `compare-lora.sh`, `compare-vision.sh` (required, not optional), and `compare-llama-cpp.sh` at
  `n_prompt` 128 and 512; publish under `docs/perf-compare/<timestamp>-tier04c-packed-matmul/`.

  Every number below is a median of at least three runs with min/max published, per the README's
  noise-floor rule — this host resolves to about ±15% and two of these sit inside that.

  **Threshold, VRAM (item 2 + item 3).** Peak device memory during a 512-token prefill is
  **<= 1.05x** peak device memory during decode on the same model, on `mistral-7b` Q4_K_M and
  `llama-1-30b` Q4_K_M — that is, the weight-shaped scratch term is gone rather than reduced. Total
  resident weight bytes for fully-offloaded layers are **<= 1.20x** the corresponding on-disk tensor
  bytes for every format with a packed kernel, against roughly **3.56x** for Q4_K today.

  **Threshold, throughput (item 2).** The tiled packed GEMM's pp t/s is **>= 0.95x** the current
  dequant-plus-`cublasGemmEx` path at a 512-token window, and **>= 1.10x** at the 16-to-64 widths
  the continuous schedule runs, on `tinyllama-1.1b` and `mistral-7b` Q4_K_M. The 512 figure is a
  no-regression floor because the win there is VRAM; the narrow-batch figure is where the dequant
  amortizes worst and is the one this item is justified by. A result below either number is
  reported, not absorbed.

  **Threshold, numerical quality.** Mean relative error against the `CpuMatVec` FP32 oracle for the
  packed path is **<= 1.0x** that of the dequant-to-FP16 path on every tested shape, and greedy
  decode agrees with the FP16 path on **>= 99%** of the first 512 tokens at temperature 0.

  **Threshold, load (item 3).** Peak host heap during weight upload drops by **>= 60%** on
  `llama-1-30b.Q4_K_M.gguf` for formats that move to packed residency, since the `float[rows*cols]`
  intermediate is no longer built. If Tier 04 item 0's mapped loading already landed, measure the
  delta on top of it rather than against the pre-mmap baseline.

  **Milestone.** The README assigns Tier 04 GPU tg **>= 0.40x** on Phi-3.5-mini. This tier inherits
  that figure as a floor and additionally reports the pp ratio against Tier 01B's re-baselined
  prefill number, met or missed with the actual value.

## Models needed

Everything required is on disk per [`INVENTORY.md`](INVENTORY.md): `tinyllama-1.1b` and `mistral-7b`
Q4_K_M for the throughput thresholds, `llama-1-30b.Q4_K_M.gguf` for the scratch-reserve and
host-heap thresholds (it is the model whose 227 MiB FFN scratch motivates item 0), and Phi-3.5-mini
for the inherited milestone. Formats added by Tier 04 are exercised on whatever files that tier
ended with. No AMD hardware exists here, so item 3's ROCm half is unit-tested and marked
`NEEDS-AMD-HARDWARE` rather than passed.

## Exit criteria

- [ ] Forward-pass device memory is reserved before weight upload consumes it; the device-OOM
      regression test fails on the pre-fix build and passes after, on both single-node and
      per-shard cluster paths.
- [ ] A tiled packed-weight GEMM exists and is the default for batch > `HALF_SGEMM_BATCH_MAX` on
      every packed format; `Q4KDequantScratch` is off the default path.
- [ ] `Q4KResidentUpload` is the single packed-versus-FP16 decision point, used by every handler,
      and covers every format with a packed kernel.
- [ ] `RocmMatVec` reports `supportsQ4KMmq` truthfully and uploads packed where Tier 04's kernels
      allow; marked `NEEDS-AMD-HARDWARE`.
- [ ] Every remaining FP16-resident case logs once with format, backend and bytes-per-weight, and is
      included in the VRAM budget before upload begins.
- [ ] `--mmq off` still gives a working FP16-resident baseline, documented as a debugging baseline.
- [ ] The `--mmq` default claim is corrected everywhere it appears (`docs/agent-arch.txt`, `--help`,
      `CHANGELOG.md`), audited by grep rather than by the cited line.
- [ ] Item 1's per-batch-width breakdown published, and item 2's justification recorded against it —
      including the case where the breakdown did not support the throughput premise.
- [ ] All thresholds above met or explicitly reported as missed with the measured number.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, no unexplained regression; `compare-llama-cpp.sh` pp and tg ratios
      recorded in this file against the program target and the inherited Tier 04 milestone.
- [ ] Docs (`docs/agent-arch.txt`, `docs/howto.md`, `README.md`) updated in Juno-native language.
- [ ] `CHANGELOG.md` entry added.
