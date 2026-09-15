# Tier 17: GPU Batched Prefill GEMM

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS.
4. Prefer adding new Java classes over extending existing ones.
5. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
6. No emojis; be strict and precise.
7. Output: list changed files for preview; never zip files back.

Also read:

- `PLAN-Infra-ROADMAP.md` — Execution rules §1-§6 (one tier in flight, perf-compare gate, no tier
  labels shipped, all-architectures rule, interaction-matrix rule)
- `PLAN-Infra-Tier8.md` — owns `--prefill-batch` chunk-size semantics; this tier does not redefine
  chunk size, only what happens inside one chunk's GEMM call
- `PLAN-Infra-Tier13.md` — owns `--mmq` / Q4_K decode kernels; its own interaction matrix already
  flags the gap this tier closes (`--prefill-batch` cell: "wired (batched path uses Q4 `sgemm`
  serial GEMVs)")
- `PROMPT-P0-Gate.md` — currently-active decode-tg-focused P0 gate prompt; this tier is a sibling,
  not a replacement — do not fold decode-kernel work into this tier
- `node/src/main/java/cab/ml/juno/node/CudaMatVec.java`, `GpuBlasOps.java`, `MatVec.java`,
  `DeviceQ4KMatrix.java`, `Q4KMmqKernel.java`, `node/src/main/cuda/q4k_gemv.cu`

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P0 (sibling to the active P0-Gate decode work; this tier is prefill-side) |
| **Depends on** | Tier 8 (feature complete — owns chunk size), Tier 13 Phase B (feature complete — owns `DeviceQ4KMatrix`/`Q4KMmqKernel`) |
| **Blocks** | The still-open "GPU prefill re-run" item under Tier 8 in `PLAN-Infra-ROADMAP.md` — this tier is what finally makes that re-run meaningful |
| **Parallel with** | None — ROADMAP §1 allows only one Infra tier in flight; if `PROMPT-P0-Gate.md` work is active, sequence after it or explicitly swap in per §1 (do not run both at once) |

## Feature × surface interaction matrix

Per ROADMAP §6. This tier changes an internal kernel-selection path inside existing
`sgemm(DeviceHalfMatrix|DeviceQ4KMatrix, float[][])` methods — no new CLI/env flag — but §6 still
applies because it changes a MatVec/residency path combinable with `--prefill-batch`, `--mmq`,
`--gpu-layers`.

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|---|---|---|---|---|---|---|---|---|---|---|
| Batched prefill GEMM (`CudaMatVec.sgemm(DeviceHalfMatrix\|DeviceQ4KMatrix, ...)`, internal, no flag) | **wired** (Llama-family + Phi-3 + Qwen3, both FP16-resident default and `--mmq on` Q4_K-resident, via shared `sgemmLayerInto`/`backend.sgemm` — verified: `Phi3TransformerHandler.java:876,892,906,918`, `Qwen3TransformerHandler.java:683-696`) | **follow-up** — `LoraTrainableHandler` routes through `ResidentWeightMatrix`/`LoraResidentWeights`, not `CudaMatVec.sgemm` (verified: `ResidentWeightMatrix.java:68-75`, `LoraResidentWeights.java:173-182`); unaffected by this tier | **follow-up** — same as `--lora-play`; LoRA train's `matVecBatch` Q4 path stays serial GEVM per `LoraResidentWeights.java:178-181`'s existing documented behavior ("no batched MMQ GEMM") | **wired if** vision's forward pass routes through `LlamaTransformerHandler`/`Phi3TransformerHandler` batched prefill (confirm at implementation time — step 10 below; if not, **N/A — text MatVec only**, matching Tier 13's own precedent) | **wired** (decode-sized batches ≤ `HALF_SGEMM_BATCH_MAX` keep the existing `sgemmHalfBatched`/serial-Q4 path unchanged; this tier only changes the `> HALF_SGEMM_BATCH_MAX` branch, which prefill windows hit, not `--parallel` decode batches) | **wired** (independent — offloaded/resident layers use the fixed `sgemm` regardless of how many layers are on GPU) | **wired** (this is the change's primary target; does not redefine chunk size, only what happens inside one chunk) | **wired** | **follow-up** (`RocmMatVec` has zero `sgemm` overrides today, for any residency type, at any batch size — named follow-up, no ROCm hardware available to implement/validate this session; the existing `MatVec` default serial fallback still produces correct results, just slow — not fail-closed/broken, only unoptimized) | on (no flag; internal always-on kernel selection once implemented — matching `GpuBlasOps.forward`'s existing unconditional batch>1 behavior for FP32) |

`Qwen3MoeTransformerHandler`: **follow-up**, pre-existing — has no GPU-resident weight path at all
(batched or serial), confirmed via `Qwen3MoeTransformerHandler.java:363-366` always using host
`LlamaTransformerHandler.matVec`. Not a regression introduced by this tier; named here per ROADMAP
§5 so it isn't silently implied as covered.

## Cross-feature smoke (before feature complete)

- [ ] Base inference (CUDA, `--mmq off`): run `--prefill-batch 32` and `128` against TinyLlama /
      Qwen2.5-3B / Phi-3.5-mini with JFR on; confirm `juno.MatVec.backend.cuda_resident_fp16`
      count rises and per-call p95 reflects a batched call, not `batch_count` separate calls.
- [ ] Base inference (CUDA, `--mmq on`): same, confirm the new Q4_K batched path fires (new JFR
      backend label or log line naming the dequant-to-scratch + batched-GEMM path).
- [ ] `--lora-play` / LoRA train: confirm behavior is unchanged (still routes through
      `ResidentWeightMatrix`/`LoraResidentWeights`, serial for FP16/Q4); record as explicit no-op
      evidence, not silently skipped.
- [ ] Vision: determine (step 10) whether the shared path is exercised; record wired or N/A with
      evidence either way.
- [ ] ROCm: confirm no behavior change (still uses `MatVec` serial default); document as follow-up,
      not silently untested.
- [ ] §2 compares run as required by change surface (see Verification and exit gate).

## Exit checklist (compatibility)

- [ ] Interaction matrix complete (no empty cells)
- [ ] No silent flag ignore on any surface that accepts `--mmq`/`--prefill-batch`/`--gpu-layers`
- [ ] Launcher (`scripts/run.sh` / `run.bat`) unaffected (no new flags introduced by this tier)
- [ ] User-facing docs (`docs/performance.md`) state the GPU prefill improvement in Juno terms only
- [ ] ROADMAP §5 architectures covered (Llama-family, Phi-3, Qwen3) or named follow-up (Qwen3-MoE,
      ROCm)

## Overview

Live GPU bake-off this session (GTX 1080, 8 GB, HEAD `3f4a340`, `n_prompt=128`, `n_gen=64`) shows
Juno prefill (pp) throughput within roughly 1x of decode (tg) throughput on every model tested,
both with `--mmq off` and `--mmq on`, while the peer engine's pp is 15-20x its own tg on the same
hardware/GGUFs. This is the fingerprint of prefill never actually getting a batched GEMM:

1. `CudaMatVec.sgemm(DeviceHalfMatrix A, float[][] X)` (`CudaMatVec.java:679-688`) only takes the
   real batched kernel (`sgemmHalfBatched`, using `cublasHSSgemvStridedBatched`) when
   `X.length <= HALF_SGEMM_BATCH_MAX` (`= 8`, line 673). Any `--prefill-batch` window above 8
   (default 32; Tier 8's goal is 512-1024) falls through to `X.length` serial `sgemv(A, X[b])`
   calls.
2. `CudaMatVec` has **no** `sgemm(DeviceQ4KMatrix, float[][])` override at all, so every Q4_K-
   resident (`--mmq on`) prefill call silently uses `MatVec`'s default (`MatVec.java:190-194`) —
   unconditionally serial, regardless of batch size. Tier 13's own interaction matrix already names
   this: `--prefill-batch` cell = "wired (batched path uses Q4 `sgemm` serial GEMVs)".
3. `RocmMatVec.java` has zero `sgemm` overrides of any kind (confirmed:
   `grep -n "sgemm(" RocmMatVec.java` returns nothing) — every residency type is always serial on
   ROCm. No ROCm hardware is available to validate a fix this session; ROCm is scoped as a named
   follow-up only (see Non-goals).

The one place a real large-batch GEMM already exists is
`GpuBlasOps.forward(DeviceFloatMatrix, float[][], int)` (`GpuBlasOps.java:65-94`, one
`cublasSgemm_v2` call), used by `CudaMatVec.sgemm(DeviceFloatMatrix, float[][])`
(`CudaMatVec.java:661-670`) for `X.length > 1`. FP32-resident weights are not the
default/benchmarked path ("FP16 resident weights are not supported here — callers fall back to
sequential sgemv", `GpuBlasOps.java:37-38`) — this tier extends the *pattern*, not `GpuBlasOps`
itself, to `DeviceHalfMatrix` and `DeviceQ4KMatrix`.

**Confirmed shared primitive (ROADMAP §5):** `LlamaTransformerHandler.sgemmLayerInto`
(`LlamaTransformerHandler.java:1334-1349`), `Phi3TransformerHandler`
(`transformerLayerBatch`, `Phi3TransformerHandler.java:795`, calling
`backend.sgemm(attnQkvQ4Dev[li], X)` / `backend.sgemm(half, X)` at lines 876/892/906/918), and
`Qwen3TransformerHandler.sgemmLayerInto` (`Qwen3TransformerHandler.java:683-696`) all call
`backend.sgemm(DeviceQ4KMatrix|DeviceHalfMatrix, X)` for their batched-prefill path — i.e. they all
funnel through the exact same `CudaMatVec.sgemm` overrides this tier fixes. **A fix at that one
layer covers Llama-family, Phi-3, and Qwen3 automatically — no per-handler changes needed for
those three.**

**Confirmed gap, not covered by this tier:** `Qwen3MoeTransformerHandler.sgemmLayerInto`
(`Qwen3MoeTransformerHandler.java:363-366`) does **not** call `backend.sgemm` at all — it always
loops `LlamaTransformerHandler.matVec(quant, X[b], rows, cols)` (host quantized CPU matVec) per
batch element, with no GPU-resident weight arrays (`devQ4`/`devHalf`) referenced anywhere in the
file. Qwen3-MoE has no GPU residency path today, batched or serial — this predates this tier and
is out of scope; named as follow-up in the interaction matrix.

**Confirmed non-interaction with LoRA:** `LoraTrainableHandler` (`--lora-play` and LoRA train) does
**not** route through `CudaMatVec.sgemm(DeviceHalfMatrix|DeviceQ4KMatrix, ...)` at all. It uses its
own independent classes: `ResidentWeightMatrix.sgemmBatch` (`ResidentWeightMatrix.java:68-76`,
batched only when FP32-resident: `if (fp32 != null && ops != null && batch > 1) return
ops.forward(fp32, X, batch);` — FP16-resident always falls to serial `sgemv`) and
`LoraResidentWeights.matVecBatch` (`LoraResidentWeights.java:182-202`, doc comment: "Q4 residency
always uses sequential GEVM (no batched MMQ GEMM)"). Fixing `CudaMatVec.sgemm` therefore does
**not** speed up `--lora-play`/LoRA-train batched prefill; that would require a separate change to
`ResidentWeightMatrix`/`LoraResidentWeights` and is explicitly out of scope for this tier (see
Non-goals).

## Scope and compatibility

Goals:

1. `CudaMatVec.sgemm(DeviceHalfMatrix A, float[][] X)` uses a real batched/matrix-matrix GEMM for
   prefill-sized batches (not just `<= HALF_SGEMM_BATCH_MAX = 8`), for `X.length` up to at least
   512 (Tier 8's stated long-prefill goal), without changing the small-batch (`<= 8`, multi-request
   decode) code path or its numerics.
2. `CudaMatVec.sgemm(DeviceQ4KMatrix A, float[][] X)` gets a real override (today there is none —
   it silently uses the serial `MatVec` default) that is faster than serial `sgemv` for
   prefill-sized batches, for the same batch range.
3. Both fixes are internal kernel-selection swaps inside `CudaMatVec` — no new CLI flag, no change
   to `--prefill-batch`, `--mmq`, `--gpu-layers`, or any other externally observable flag/default.
   `MatVec`/`GpuMatVec` public method signatures are unchanged (same
   `sgemm(DeviceHalfMatrix, float[][])` / `sgemm(DeviceQ4KMatrix, float[][])` signatures, just real
   implementations for CUDA).
4. Llama-family (`LlamaTransformerHandler`), Phi-3 (`Phi3TransformerHandler`), and Qwen3
   (`Qwen3TransformerHandler`) all benefit automatically since they share `CudaMatVec.sgemm`
   (ROADMAP §5 "shared abstraction" clause) — verify this with parity/perf evidence per handler
   family, not just Llama.
5. Publish a GPU bake-off showing prefill throughput materially decoupled from decode throughput
   (pp >> tg), closing the "GPU prefill re-run still open" item under Tier 8 in the ROADMAP.
6. Correctness: batched output must match the existing serial `sgemv`-loop output within float
   tolerance for every batch size, including edge cases (`batch=1`, `batch=HALF_SGEMM_BATCH_MAX`,
   `batch=HALF_SGEMM_BATCH_MAX+1`, non-power-of-2 batches, `cols`/`rows` not evenly divisible by
   warp/tile sizes).

Non-goals:

- Decode-path kernels, the `q4k_gemv` tile-kernel P0 speed gate, or anything in
  `PROMPT-P0-Gate.md`'s scope (single-token `sgemv(DeviceQ4KMatrix, x)` /
  `sgemv(DeviceHalfMatrix, x)` are untouched).
- Any CLI/env flag addition or default change. `--prefill-batch`, `--mmq`, `--gpu-layers`,
  `--dtype` all keep current defaults and semantics.
- `--prefill-batch` chunk-size semantics (Tier 8's domain) — this tier only changes what happens
  inside one chunk's GEMM call, not how chunks are formed or sized.
- ROCm implementation (`RocmMatVec` gap) — named follow-up only, no code change, no hardware to
  validate this session.
- `--lora-play` / LoRA-train batched-prefill throughput (`ResidentWeightMatrix`/
  `LoraResidentWeights` are separate classes not touched by this tier — see Overview). A future
  LoRA-adjacency tier (pattern: `PLAN-Infra-LoRA-MMQ.md`) would wire
  `ResidentWeightMatrix.sgemmBatch`/`LoraResidentWeights.matVecBatch` to call the same fixed
  `CudaMatVec.sgemm` methods.
- FlashAttention / P5 track.
- `Qwen3MoeTransformerHandler` GPU residency (it has none today, batched or serial — pre-existing
  gap, named follow-up, not a regression introduced by this tier).
- Changing `GpuBlasOps`'s existing FP32 behavior/tests.

## Chosen design

### Gap 1 — `DeviceHalfMatrix` (FP16-resident, default GPU dtype)

**Chosen: extend the `cublasSgemm_v2`-style real-GEMM pattern with a new FP16-input path via
`cublasGemmEx`, not generalize `cublasHSSgemvStridedBatched` to large batches.**

Rationale:

- `cublasHSSgemvStridedBatched` is a *strided-batched GEMV* primitive — it launches `batch`
  independent GEMV kernels internally (one dot-product reduction per output row per batch
  element); it does not get matrix-matrix compute/bandwidth reuse across the batch dimension the
  way a true GEMM does. It is the right primitive for small decode batches (the existing use at
  `X.length <= 8`) where per-call overhead dominates, but it does **not** turn into a compute-bound
  GEMM just by raising the batch cap — you would still be paying to re-stream the full weight
  matrix from device memory once per batch column, the same problem serial `sgemv` has, just
  batched into fewer kernel launches. For W=128+ prefill windows, the point is precisely to *stop*
  re-streaming the weight matrix W times.
- `cublasGemmEx` on Pascal (GTX 1080, sm_61, no tensor cores) still executes as a real tiled GEMM
  in FP32 accumulate — `CUDA_R_16F` for A and X, `CUDA_R_32F` for the accumulator/output `Y`,
  `computeType = CUBLAS_COMPUTE_32F` (verify exact enum value against the installed CUDA 12
  headers, not assumed from this doc — `CUDA_R_16F=2`/`CUDA_R_32F=0` are stable since CUDA 8 and
  safe to hardcode; `CUBLAS_COMPUTE_32F` has changed shape across CUDA major versions), `algo =
  CUBLAS_GEMM_DEFAULT`. This gives real weight-stationary tiling and bandwidth reuse across the
  batch, matching the same rationale `GpuBlasOps.forward` already uses for FP32 (`cublasSgemm_v2`)
  — just with FP16 storage for `A`/`X` to match the FP16-resident default dtype.
- This mirrors the existing size-based branch already in
  `CudaMatVec.sgemm(DeviceFloatMatrix, float[][])` (`X.length <= 1` → `sgemv`, else →
  `GpuBlasOps`) and in `sgemm(DeviceHalfMatrix, float[][])` today (`X.length <= 1 ||
  X.length > HALF_SGEMM_BATCH_MAX` → serial) — this tier just replaces the "`>
  HALF_SGEMM_BATCH_MAX` → serial" branch with "`> HALF_SGEMM_BATCH_MAX` → `cublasGemmEx`".

Scratch buffer sizing: the existing `Fp16Scratch` (`CudaMatVec.java:96-101`, fields `dXh`/`dY`,
grown lazily via `ensureFp16Scratch`) already grows to `cols * batch * Short.BYTES` /
`rows * batch * Float.BYTES` for any batch — `sgemmHalfBatched` (`CudaMatVec.java:690-753`) already
calls `ensureFp16Scratch(scratch, bytesXh, bytesY)` with `batch`-scaled sizes today. **Reuse
`Fp16Scratch`/`ensureFp16Scratch` as-is** — no new scratch container needed; the existing
lazy-grow-and-keep-max logic already handles arbitrary batch up to whatever the caller passes
(512-1024 token windows just grow the ThreadLocal buffer once and keep it). Do not introduce a
separate "large-batch" scratch class — that would violate KISS for no benefit.

### Gap 2 — `DeviceQ4KMatrix` (Q4_K-packed, `--mmq on`)

**Chosen: (b) dequantize the packed Q4_K/Q5_K/Q6_K weights once into a device FP16 scratch buffer,
then hand off to the (now-fixed) Gap 1 FP16 batched-GEMM path — not (a) a new fused batched
dequant+GEMM CUDA kernel.**

Rationale:

- `q4k_gemv.cu` (340 lines) is already a nontrivial `mul_mat_vec`-class kernel: per-block
  affine-scale decode (`kq_affine_scales`, lines 59-79), `__dp4a` integer-dot against
  Q8_1-quantized activations, warp-shuffle + shared-memory reduction (`warp_sum`, lines 81-85). It
  is tuned for one output row per block with the activation vector held in registers/shared memory
  (`ROWS_PER_BLOCK = 1`, `Q4KMmqKernel.java:52`). Turning this into a genuine tiled
  weight-stationary GEMM (dequant once, multiply against all W columns while the dequantized block
  is hot — the CPU `LlamaTransformerHandler.sgemmQ4KWeightStationary` technique) means redesigning
  the reduction/shared-memory strategy for multiple simultaneous output columns per row, plus new
  batched Q8_1 quantization of W activation vectors. That is real new PTX authorship and tuning
  risk, with no CUDA hardware in this session's loop to iterate against beyond the one live
  bake-off already run.
- The dequant-to-scratch approach only needs a **new, much simpler** kernel: an
  embarrassingly-parallel elementwise dequant (packed Q4_K/Q5_K/Q6_K block → row-major FP16
  buffer), reusing the existing `kq_affine_scales`-style per-block decode logic but with no
  reduction, no batching complexity, and no interaction with the activation vector at all — dequant
  is independent of `X`, so it happens once per layer-projection, not once per token. It is
  naturally a new PTX entry point (`q4k_dequant_to_fp16` or similar) added to the existing
  `q4k_gemv.cu`/`q4k_gemv.ptx` module (same module load machinery in `Q4KMmqKernel`, no new
  module-loading class needed), consistent with KISS.
- Cost: extra VRAM for the dequantized FP16 scratch copy of whichever weight matrix is
  mid-prefill, and extra device time for the dequant pass. Both are bounded and
  one-time-per-chunk (not per-token): dequant cost is `O(rows * cols)` once, then the FP16 GEMM
  amortizes across the whole `W`-token batch — for `W >= 32` (default `--prefill-batch`) this is a
  clear net win over `W` serial Q4_K GEVM kernel launches, and the *fit* benefit of `--mmq on`
  (packed weights resident, not FP16) is preserved because the FP16 scratch buffer is transient
  (one weight matrix's worth, freed/reused per projection) rather than a second permanent
  FP16-resident copy of the whole model.
- This explicitly does **not** touch or replace the existing single-token
  `sgemv(DeviceQ4KMatrix, x)` decode path (`Q4KMmqKernel.launch`/`launchPacked`) — that stays
  exactly as-is, preserving `--mmq on`'s current fit/decode behavior and staying out of the
  P0-Gate's decode-kernel scope.
- Weighing against option (a): a genuine fused batched dequant+GEMM kernel would likely have a
  higher throughput ceiling (avoids materializing dequantized weights and the associated
  VRAM/bandwidth round-trip), and is the right thing to pursue *if* Q4_K prefill throughput after
  this tier's fix is still found wanting relative to the FP16 path. Flag it as a follow-up lever
  inside Tier 13/P0-Gate territory (decode-kernel authors own `q4k_gemv.ptx`), not something this
  tier should attempt given the authoring risk and no hardware to iterate against beyond one
  bake-off.

Scratch buffer sizing: new device buffer sized `rows * cols * Short.BYTES` (FP16) for the
dequantized weight matrix being processed — this is a *weight-shaped* scratch buffer, independent
of batch size `W` (unlike `Fp16Scratch.dXh`/`dY` which scale with `W`). Size it once per distinct
`(rows, cols)` shape actually seen (projections share shapes across layers of the same model),
grown lazily the same way `Fp32Scratch`/`Fp16Scratch` already do.

### Threshold design (both gaps)

Two thresholds, no new CLI flag — `sgemm()` picks the kernel purely by `X.length`, an internal
implementation detail:

| `X.length` (batch) | `DeviceHalfMatrix` path | `DeviceQ4KMatrix` path |
|---|---|---|
| `<= 1` | serial `sgemv` (unchanged) | serial `sgemv` (unchanged) |
| `2..HALF_SGEMM_BATCH_MAX` (`= 8`, unchanged constant, decode/multi-request) | `sgemmHalfBatched` (`cublasHSSgemvStridedBatched`, unchanged) | serial `sgemv` loop (new: still serial below the batched-worthwhile threshold — dequant has fixed overhead not worth paying for tiny batches; keep the same `HALF_SGEMM_BATCH_MAX = 8` cutover point for symmetry/simplicity rather than a second magic number) |
| `> HALF_SGEMM_BATCH_MAX` (prefill) | **new:** `sgemmHalfBatchedGemm` (`cublasGemmEx`) | **new:** dequant-to-FP16-scratch + `sgemmHalfBatchedGemm` |

Reuse the existing `HALF_SGEMM_BATCH_MAX` constant for both matrices' large-batch cutover rather
than adding a second threshold constant. If bake-off evidence later shows a different crossover
point is faster for Q4_K's dequant overhead specifically, that is a tuning follow-up, not a
blocker for this tier.

## New/modified classes

- **`node/src/main/java/cab/ml/juno/node/CudaBindings.java`** (modify): add `MethodHandle
  cublasGemmEx` binding (`cublasGemmEx` symbol in `libcublas.so.12`), `FunctionDescriptor` per the
  C signature `(handle, transa, transb, m, n, k, *alpha, *A, Atype, lda, *B, Btype, ldb, *beta, *C,
  Ctype, ldc, computeType, algo)` — enum params `JAVA_INT`, pointers `ADDRESS`. Add `static final
  int CUDA_R_16F`, `CUDA_R_32F`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT` constants alongside
  the existing `CUBLAS_OP_N`/`CUBLAS_OP_T`/`CUBLAS_POINTER_MODE_HOST` — **verify exact enum values
  against the installed CUDA 12.x headers before hardcoding**; only `CUDA_R_16F=2`/`CUDA_R_32F=0`
  are safe to assume from this doc alone.
- **New class `node/src/main/java/cab/ml/juno/node/CudaFp16GemmOps.java`** (CUDA-only — do not
  route through the vendor-neutral `GpuBindings`/`GpuBlasOps`, since the existing FP16 decode path
  in `CudaMatVec` already bypasses that abstraction and calls
  `CudaBindings.cublasHSSgemvStridedBatched` directly; `GpuBlasOps` is documented FP32/vendor-
  neutral and should stay that way). Mirrors `GpuBlasOps`'s shape
  (`forward(DeviceHalfMatrix W, float[][] X, int batch)` returning `float[][]`) but calls
  `cublasGemmEx` with `CUDA_R_16F` A/X and `CUDA_R_32F` C. Owns its own scratch reuse of
  `CudaMatVec`'s `Fp16Scratch` (pass it in, or expose a package-private accessor) rather than
  duplicating buffer-growth logic.
- **New class `node/src/main/java/cab/ml/juno/node/Q4KDequantScratch.java`**: owns the
  weight-shaped `(rows, cols)` FP16 device scratch buffer described above, lazily grown, one
  instance held per `CudaMatVec` (ThreadLocal, same pattern as `Fp16Scratch`/`Fp32Scratch`).
  Method `MemorySegment dequantToFp16(DeviceQ4KMatrix A, MemorySegment stream)` launches the new
  dequant kernel and returns the FP16 device pointer.
- **`node/src/main/cuda/q4k_gemv.cu`** (modify): add new `__global__` entry point (e.g.
  `q4k_dequant_to_fp16`, and `q5k_dequant_to_fp16`/`q6k_dequant_to_fp16` for full quant-type
  coverage matching `DeviceQ4KMatrix.supportsType`) that reuses `kq_affine_scales` and friends to
  write a row-major FP16 buffer, no reduction/no activation involved. Regenerate `q4k_gemv.ptx`
  (both `node/src/main/resources/cab/ml/juno/node/q4k_gemv.ptx` and the `target/classes` build
  artifact) per the existing `nvcc -ptx -arch=compute_61 -O3` build comment at the top of the file.
- **`node/src/main/java/cab/ml/juno/node/Q4KMmqKernel.java`** (modify): add `MemorySegment
  fnQ4Dequant`/`fnQ5Dequant`/`fnQ6Dequant` function handles (same `cuModuleGetFunction` pattern as
  existing `fnQ4K`/`fnQ5K`/`fnQ6K`), and a new method `void launchDequant(DeviceQ4KMatrix A,
  MemorySegment dOutFp16, MemorySegment stream)` mirroring `launchPacked`'s parameter-packing
  style.
- **`node/src/main/java/cab/ml/juno/node/CudaMatVec.java`** (modify):
  - `sgemm(DeviceHalfMatrix A, float[][] X)` (lines 679-688): change the
    `X.length > HALF_SGEMM_BATCH_MAX` branch from serial `sgemv` to a new private
    `sgemmHalfBatchedGemm(A, X)` method (uses `CudaFp16GemmOps`).
  - New override `public float[][] sgemm(DeviceQ4KMatrix A, float[][] X)` (does not exist today):
    `X.length <= HALF_SGEMM_BATCH_MAX` → serial `sgemv` loop (kept explicit for clarity rather than
    relying on the interface default); `X.length > HALF_SGEMM_BATCH_MAX` →
    `Q4KDequantScratch.dequantToFp16(A, stream)` + the Gap 1 GEMM call reinterpreting the dequant
    output as FP16-shaped device memory (a small internal overload taking a raw `MemorySegment`
    instead of a `DeviceHalfMatrix` wrapper, to avoid faking a `DeviceHalfMatrix` object around
    scratch memory it doesn't own).
- **`node/src/main/java/cab/ml/juno/node/MatVec.java`**: no change required — the default
  `sgemm(DeviceQ4KMatrix, float[][])` stays as the correctness-preserving fallback for backends
  without an override (documented behavior for `RocmMatVec` per the ROCm follow-up).
- **`node/src/main/java/cab/ml/juno/node/GpuBlasOps.java`**: no change — stays
  FP32-only/vendor-neutral as documented.
- **`node/src/main/java/cab/ml/juno/node/DeviceQ4KMatrix.java`**: no change to the class itself
  (already exposes `rows()`, `cols()`, `quantType()`, `devicePointer()` — sufficient for the new
  dequant kernel launch).

## Implementation

### 1. Parity test scaffolding — tests first

Before any kernel/binding code: add
`node/src/test/java/cab/ml/juno/node/CudaSgemmBatchedPrefillParityTest.java`, `@Tag("gpu")`,
modeled directly on `GpuBlasOpsTest.java`'s pattern (`assumeTrue(CudaAvailability.isAvailable(),
...)`, `GpuContext.init(0)`, float tolerance, `try (DeviceXMatrix dW = ...)`). Write it against the
**current** (serial-fallback) implementation first so it passes red→green trivially, establishing
the reference oracle: batched output must equal the existing serial `sgemv`-loop output. Cases:

- `DeviceHalfMatrix`, batch in {1, 8, 9, 16, 32, 128} — reference = current per-`b`
  `sgemv(A, X[b])` loop (host-side, computed in the test, not a second GPU call).
- `DeviceQ4KMatrix`, same batch set — reference = current per-`b` `sgemv(A, X[b])` loop via
  `Q4KMmqKernel`.
- Non-multiple-of-tile-size `rows`/`cols` (e.g. `rows=17, cols=13`, matching `GpuBlasOpsTest`'s
  existing convention) to catch tiling edge bugs early.
- `cols` must stay a multiple of the Q4_K block layout constant for the Q4_K case (existing
  constraint, keep test fixtures compliant).

### 2. `cublasGemmEx` binding + `CudaFp16GemmOps`

Implement the Gap 1 fix. Re-run step 1's `DeviceHalfMatrix` cases; they should now exercise the
new `cublasGemmEx` path for batch > 8 and stay green.

### 3. `CudaMatVec.sgemm(DeviceHalfMatrix, float[][])` threshold swap

Replace the `> HALF_SGEMM_BATCH_MAX` serial branch with the new GEMM call. Confirm
`sgemmHalfBatched` (<=8 path) is byte-for-byte unchanged (no regression to the existing
decode/multi-request path or its tests).

### 4. New Q4_K/Q5_K/Q6_K dequant-to-FP16 PTX kernel + `Q4KMmqKernel.launchDequant`

Add to `q4k_gemv.cu`, regenerate PTX, wire `Q4KMmqKernel`. Unit-test the dequant kernel in
isolation first (a small `Q4KDequantParityTest`, `@Tag("gpu")`: dequant a known Q4_K block,
compare against the existing host-side dequant reference values) before wiring it into the batched
`sgemm` path — this isolates dequant correctness bugs from GEMM correctness bugs.

### 5. `Q4KDequantScratch` + `CudaMatVec.sgemm(DeviceQ4KMatrix, float[][])` new override

Wire dequant + `CudaFp16GemmOps` together. Re-run step 1's `DeviceQ4KMatrix` cases; they should now
exercise the new path for batch > 8.

### 6. Per-handler perf/parity smoke (ROADMAP §5)

Confirm Llama-family, Phi-3, and Qwen3 all see the new batched path fire — add or extend a small
live/JFR-backed check per handler family (not just Llama) showing `MatVecEvent`/`juno.MatVec`
backend label for a batch > 8 forward call is the new batched backend, not per-token serial.

### 7. Concurrency check

Extend or add a test analogous to `CudaMatVecBackendTest`'s "Concurrent sgemv calls produce
correct results" test for the new batched path — the new `Q4KDequantScratch`/GEMM scratch is
`ThreadLocal` like existing scratch, but confirm no cross-thread corruption under concurrent
prefill on multiple sessions.

### 8. GPU bake-off

`scripts/performance-tests/compare-llama-cpp.sh --gpu` (default model set: TinyLlama, Qwen2.5-3B,
Phi-3.5-mini, Mistral-7B per ROADMAP §5) with both `--mmq off` and `--mmq on`, matching this
session's `n_prompt=128, n_gen=64` methodology so the before/after numbers in this doc's table are
directly comparable. Record pp/tg per model.

### 9. LoRA regression gate (ROADMAP §2)

`compare-lora.sh --gpu --baseline release-0.1.2` (or last published). Expect **no material
change** (LoRA doesn't route through the fixed methods — see Overview) — a flat/neutral result is
the correct outcome here, not evidence the tier did nothing; document that expectation before
running so a flat result isn't mistaken for a broken gate.

### 10. Vision gate check (ROADMAP §2)

Confirm whether `VisionAwareForwardPassHandler`/vision's batched prefill routes through
`LlamaTransformerHandler`/`Phi3TransformerHandler`'s `sgemmLayerInto`/`CudaMatVec.sgemm` path
(read `ForwardPassHandlerLoader`/`VisionAwareForwardPassHandler` decorator chain to confirm before
deciding). If yes, run `compare-vision.sh` per §2; if the vision path uses `--prefill single` by
default per the script's documented default, note whether the gate is even exercised by this
tier's change and say so explicitly rather than skipping silently.

### 11. Docs

Update `docs/performance.md`'s prefill-microbatching section (the existing CPU-only 2.30/5.39 t/s
table) with the new GPU row(s); update ROADMAP's Tier 8 status line to remove "GPU prefill re-run
still open"; publish under `docs/perf-compare/<timestamp>/`.

## Verification and exit gate

**Global rules** (`PLAN-Infra-ROADMAP.md` → Execution rules): only one Infra tier in flight at a
time; publish a `docs/perf-compare/` bake-off before marking this tier complete; run
`compare-lora.sh` (ROADMAP §2, this tier touches MatVec/GPU residency) and `compare-vision.sh` if
step 10 confirms vision shares the path.

**Primary qualitative gate (required):** the published bake-off must show a real batched-GEMM
signature — **pp materially greater than tg** (not `pp ~= tg`) — for every model in the default
bake-off set, under both `--mmq off` and `--mmq on`. Juno's absolute pp/tg ratio need not match the
peer engine's 15-20x (Juno carries broader per-launch/Panama-FFI overhead the peer's native C++
path doesn't), but "prefill and decode throughput are within ~1x of each other" must no longer be
true after this tier.

**Quantitative floor (judgment call, not a physics guarantee — stated honestly as such):**

| Model | `--mmq off` pp today (t/s) | `--mmq off` pp floor after fix | `--mmq on` pp today (t/s) | `--mmq on` pp floor after fix |
|---|---|---|---|---|
| tinyllama-1.1b | 95.2 | >= 3x today (>= ~285) | 39.2 | >= 3x today (>= ~118) |
| qwen2.5-3b | 15.7 | >= 3x today (>= ~47) | 22.4 | >= 3x today (>= ~67) |
| Phi-3.5-mini | 16.4 | >= 3x today (>= ~49) | 28.4 | >= 3x today (>= ~85) |
| mistral-7b | 0.92 (CPU-fallback, OOM) | N/A on `--mmq off` (unchanged VRAM-OOM behavior — this tier does not touch residency/fit) | 17.3 | >= 3x today (>= ~52) |

The `3x` floor is a conservative, round-number judgment call reasoning from "any real GEMM beats W
serial GEVM launches for W>=32," not a derived roofline number — do not treat it as a hard target
if honest measurement lands lower; report the real number and whether the qualitative gate
(pp >> tg) still holds, same spirit as `PROMPT-P0-Gate.md`'s "report fail with numbers"
instruction. mistral-7b's `--mmq off` row is explicitly N/A because that configuration OOMs into
CPU fallback on 8 GB regardless of this tier's fix (a `--gpu-layers`/fit concern, Tier 5's domain,
not this tier's).

**Full exit checklist:**

1. Published GPU compare (`compare-llama-cpp.sh --gpu`, default model set) shows pp >> tg
   (qualitative gate) for every model that fits GPU residency, under both `--mmq off` and
   `--mmq on`.
2. Quantitative floor table above met or a documented honest shortfall with numbers.
3. `CudaSgemmBatchedPrefillParityTest` (and the dequant-kernel parity test) green for
   `DeviceHalfMatrix` and `DeviceQ4KMatrix`, batches {1, 8, 9, 16, 32, 128}, non-tile-aligned
   shapes.
4. Per-handler smoke (step 6) confirms Llama-family, Phi-3, Qwen3 all exercise the new batched
   backend (JFR/log proof per ROADMAP §2 step 4 — not just process-exit-0).
5. `compare-lora.sh` run and flat/neutral result documented as expected (not a regression, per
   Overview's LoRA non-interaction finding); numeric gate (train <=1.25x, playback >=0.80x
   baseline) still passes since nothing on that path changed.
6. `compare-vision.sh` run if step 10 confirms the shared path; N/A documented otherwise.
7. Interaction matrix complete, no empty cells, ROCm/LoRA/MoE cells correctly marked
   **follow-up** (not silently implied wired).
8. `docs/performance.md`, ROADMAP Tier 8 status line, `docs/perf-compare/README.md` updated; no
   Infra tier numbers leaked into user-facing docs/CLI/JFR strings (ROADMAP §4); no competitor
   names outside `docs/infra-plan/`/`docs/perf-compare/` (ROADMAP §3).

## Risks

- **Does not touch decode-path kernels** — `sgemv(DeviceHalfMatrix, x)`,
  `sgemv(DeviceQ4KMatrix, x)`, `Q4KMmqKernel.launch`/`launchPacked` (single-token) are unchanged;
  `PROMPT-P0-Gate.md`'s decode-tg gate is neither helped nor blocked by this tier and must be
  pursued independently.
- **Does not change any CLI flag or default** — `--prefill-batch`, `--mmq`, `--gpu-layers`,
  `--dtype` semantics and defaults are identical before/after.
- **Does not implement ROCm** — named follow-up only; no code change, no hardware to validate. The
  existing serial `MatVec` default keeps ROCm correct (if slow) throughout.
- **Does not attempt FlashAttention or the P5 track.**
- **Does not change `--prefill-batch` chunk-size semantics** — Tier 8 continues to own chunk size.
- **Does not speed up `--lora-play`/LoRA-train batched prefill** — confirmed via code reading; a
  future LoRA-adjacency tier would need to wire those classes to the fixed methods, same pattern
  as `PLAN-Infra-LoRA-MMQ.md`.
- **New VRAM cost**: the Q4_K dequant-to-FP16-scratch approach introduces transient per-shape FP16
  scratch buffers (`rows * cols * 2` bytes, grown lazily, one live buffer at a time per distinct
  shape in flight) — on tight 8 GB configurations already running `--mmq on` specifically for VRAM
  fit, this could reduce headroom during prefill. Must be measured in the mistral-7b `--mmq on`
  bake-off row and documented if it changes fit behavior; if it does, consider bounding scratch
  reuse (evict/shrink after each projection) rather than keeping every shape's scratch resident
  simultaneously.
- **`cublasGemmEx` enum-value risk**: `CUBLAS_COMPUTE_32F`/`CUBLAS_GEMM_DEFAULT` values must be
  verified against the actual installed CUDA 12.x headers at implementation time, not assumed from
  this planning doc — getting these wrong fails loudly (non-zero `cublasStatus_t`) rather than
  silently, per existing `CudaBindings.check` pattern, so this is a build/smoke-test risk, not a
  silent-corruption risk.
- **New Q4_K dequant PTX kernel has no ROCm equivalent** — reinforces the ROCm follow-up scope; do
  not attempt to make the new `.cu` kernel portable to HIP as part of this tier.

## Implementation todos

1. Parity test scaffolding (`CudaSgemmBatchedPrefillParityTest`, red against current serial impl).
2. `cublasGemmEx` binding (`CudaBindings`) + `CudaFp16GemmOps`.
3. `CudaMatVec.sgemm(DeviceHalfMatrix, ...)` threshold swap; confirm <=8 path unchanged.
4. Q4_K/Q5_K/Q6_K dequant-to-FP16 PTX kernel + `Q4KMmqKernel.launchDequant`; `Q4KDequantParityTest`.
5. `Q4KDequantScratch` + `CudaMatVec.sgemm(DeviceQ4KMatrix, ...)` new override.
6. Per-handler (Llama/Phi-3/Qwen3) JFR/log smoke proof.
7. Concurrency test for new scratch buffers.
8. GPU bake-off (`compare-llama-cpp.sh --gpu`, `--mmq off` and `on`).
9. LoRA regression gate (`compare-lora.sh --gpu`), documented-flat-expected.
10. Vision gate check/run if shared path confirmed.
11. Docs: `docs/performance.md`, ROADMAP Tier 8 status, `docs/perf-compare/README.md`; preview
    files listed, no zip.

## Preview files (expected)

New: `CudaFp16GemmOps.java`, `Q4KDequantScratch.java`, `CudaSgemmBatchedPrefillParityTest.java`,
`Q4KDequantParityTest.java`

Modified: `CudaMatVec.java`, `CudaBindings.java`, `Q4KMmqKernel.java`, `q4k_gemv.cu`,
`q4k_gemv.ptx` (both copies), `docs/performance.md`, `docs/infra-plan/PLAN-Infra-ROADMAP.md`,
`docs/perf-compare/README.md`
