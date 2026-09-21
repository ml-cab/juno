# Tier 20: Prefill GPU-Residency Fixes — Pinned Staging Memory, Redundant Dequant, Adaptive Chunk Sizing

**Status: Phase A feature complete (2026-09-18).** Pinned host-staging memory (`GpuBindings.hostMalloc`/
`hostFree`, wired into `CudaMatVec`'s three batched-GEMM paths and `CudaRmsNorm.normalizeBatch`) and
adaptive whole-prompt chunk sizing for the `static` schedule (`PrefillBatchOptions.resolveAdaptive`,
querying live free VRAM via a new `GpuBindings.memGetInfo`/`GpuContext.freeVramBytes()`) are landed,
tested, and measured on real hardware: **-30% prefill time, -27% request wall time** on a real 488-token
mistral-7b prompt (16 fixed-32 chunks collapsed to 1). Phase B's checkpoint measurement is recorded:
**no-go** — GPU-resident Rope/SwiGlu still loses to CPU scalar (1.56-2.18x slower) even under pinned
memory, so Phase B is not pursued further. Phase C (tiled batched-quantized GEMM) remains a separately
gated stretch goal, not started. Full writeup: `docs/performance.md` → "Prefill GPU-residency fixes:
pinned staging memory + adaptive chunk sizing"; bake-off:
[`docs/perf-compare/20260918T153900Z-prefill-adaptive/`](../perf-compare/20260918T153900Z-prefill-adaptive/).
This session's `nsys` install could not reproduce Finding 3's exact timeline-collapse verification
(fails on every invocation, including argument-free — an environment defect, not a code issue); the
end-to-end wall-clock win and green parity/regression tests are relied on instead, honestly noted as an
open re-verification for a session with working `nsys`.

Grew out of a profiling pass on Juno's own prefill-vs-llama.cpp
gap (Juno gets **1-3%** of llama.cpp's pp throughput even on a token-count-matched bake-off — see
`docs/perf-compare/20260915T042421Z/INDEX.md`). This tier's evidence was gathered on the **prefill**
path, whereas [`PLAN-Infra-Tier19.md`](PLAN-Infra-Tier19.md) profiled **decode** (batch=1) and parked
further GPU-resident elementwise-op work pending an activation-residency redesign. One of this tier's
three findings (pinned memory) is a confound that also taints Tier 19's own round-trip measurements —
see "Relationship to Tier 19" below. Read this doc's "Measured evidence" section before implementing
anything; it is the tier's entire justification, mirroring Tier 19's own evidence-first structure.

## Relationship to Tier 19

Tier 19 measured a single ad-hoc GPU round trip (H2D + kernel + D2H + `cudaStreamSynchronize`) for
`rmsNorm` at decode's batch=1 and found it **~11x slower** than the scalar CPU path, concluding that
no per-op GPU port could pay off until activations stay device-resident across a **chain** of ops
(no intermediate host round trip) — a materially bigger redesign than porting one op at a time. That
conclusion is sound for what it measured, but this session's prefill-side profiling surfaces a
separate, cheaper-to-fix confound baked into the *same* round-trip measurement: the staging buffers
on both sides of every GPU call in the hot path (`CudaMatVec`'s batched-GEMM scratch, and Tier 19's
own `CudaRmsNorm.normalizeBatch`) use plain `Arena.ofConfined()` host memory, never registered as
CUDA pinned memory — which is independently known to inflate `cudaMemcpyAsync` host-side cost by an
order of magnitude regardless of batch size (see Finding 3). Tier 19's "11x regression" and this
session's "still 1.3x slower even at batch=136" (Finding 6) may both be measuring pinned-memory tax
as much as fundamental round-trip physics. Tier 20 fixes that confound first (Phase A), and only then
re-opens the GPU-resident-elementwise-op question as an explicit, re-measured checkpoint (Phase B) —
not a blind resumption of Tier 19's original plan, and not a reason to distrust Tier 19's own
discipline in stopping to ask rather than guessing.

## Measured evidence (2026-09-18)

**Methodology:** `nsys profile --trace=cuda,osrt` (CUDA kernel + host-API timeline, same workaround
for the broken `.qdstrm` importer as `PLAN-Infra-PERF-ANALYSIS.md`'s prior Nsight pass — invoke
`QdstrmImporter` directly, then `nsys export --type=sqlite` and query with `sqlite3`/Python) run
**simultaneously** with `--jfr 30m`, cross-validated against each other. Real GTX 1080,
`mistral-7b-instruct-v0.1-q4_k_m.gguf`, `--mmq on --gpu-layers auto`, a real 136-token chat prompt
(matches `compare-llama-cpp.sh --raw-prompt` token-count-matched convention), `max_tokens=4` to keep
the window prefill-dominated.

**Finding 1 — confirms the known gap, at real (not raw-prompt) token counts.** Request wall time
5.33s for 136 prompt tokens; prefill alone (JFR `juno.PrefillBatch`) is 4.94s of that — consistent
with the existing token-count-matched bake-off's 1.3-2.7% pp ratio
(`docs/perf-compare/20260915T042421Z/INDEX.md`), not a new number, just re-confirmed live with
kernel-level attribution.

**Finding 2 — redundant per-chunk weight dequant.** Default `--prefill-batch 32` splits the
136-token prompt into 5 chunks (32×4 + 8). `nsys` shows `cuda_resident_q4k_gemm` backend calls =
896 = **224 weight matrices × 4 full chunks** (the ragged 8-token 5th chunk falls to the serial
per-token path instead). Every one of those calls re-dequantizes its full Q4_K/Q6_K weight matrix to
FP16 from scratch via `q4k_dequant_to_fp16`/`q6k_dequant_to_fp16` — despite the weights being static
for the whole request. Kernel-level cost: dequant kernels total **835ms of the 4938ms prefill window
(16.9%)**, comparable to the **974ms (19.7%)** spent in the actual `maxwell_sgemm_fp16_*` GEMM
kernels doing the real matrix multiply.

**Finding 3 — host-side pageable-memory tax.** `cudaMemcpyAsync` host-API time totals **1994ms**
across the request (**38% of the 5.24s generation window**), while the *matching* GPU-side DMA
transfer (`CUPTI_ACTIVITY_KIND_MEMCPY`) is only **~50-130ms** — a >10x gap that is the textbook CUDA
signature of copying through non-pinned (pageable) host memory, which forces the driver to stage
through an internal pinned buffer synchronously on every call. Confirmed by code inspection: every
GPU staging buffer in the hot path — `CudaMatVec`'s `Fp16Scratch`/`Q4KDequantScratch` H2D/D2H
staging (`sgemmHalfBatchedGemm`, `sgemmQ4KBatchedGemm`) and Tier 19's own
`CudaRmsNorm.normalizeBatch` — allocates via `Arena.ofConfined()`, plain off-heap memory never
registered with the CUDA driver. `cudaMallocHost`/`GpuBindings.gpuMallocHost()` already exists as an
FFI binding (`CudaBindings.java`, mirrored in `RocmBindings.java`) but nothing in the hot path calls
it.

**Finding 4 — A/B confirms Finding 2 is real and fixable with zero new code.** Re-running the
*identical* request with `--prefill-batch 256` (one chunk covers the whole 136-token prompt):

| Metric | `--prefill-batch 32` (5 chunks) | `--prefill-batch 256` (1 chunk) | Δ |
|---|---|---|---|
| Request wall time | 5.33s | 4.05s | **-24%** |
| Prefill wall time (JFR) | 4938ms | 3650ms | **-26%** |
| Dequant kernel time | 835ms | 238ms | **-72%** |
| Actual GEMM kernel time | 974ms | 599ms | -39% (bonus: better cuBLAS tile selection at larger batch) |
| `cudaMemcpyAsync` host overhead | 1994ms | 897ms | -55% (fewer, bigger calls) |
| `cudaStreamSynchronize` calls | 2985 | 741 | -75% |

`cuda_resident_q4k_gemm` call count dropped from 896 to exactly **224** — one dequant per matrix per
request, matching the model's actual matrix count. The remaining 238ms is the *irreducible*
one-per-matrix-per-request dequant cost (only Phase C removes that; Phase A only removes the
redundant repeats).

**Finding 5 — after Finding 4's fix, scalar CPU elementwise ops dominate what's left.** Rope + SwiGlu
combined are **1505ms of the new 3650ms prefill window (41%)** — bigger than all of MatVec combined
(932ms, 25.5%). These are the same ops Tier 19 shelved for decode; at prefill's batch size the
launch/sync-amortization math is different in principle, which is why this reopens the question.

**Finding 6 — inconclusive microbenchmark, confounded by Finding 3.** A throwaway benchmark (not
checked in, mirrors Tier 19's own Phase A/B microbenchmark methodology) reused Tier 19's existing,
parity-tested `CudaRmsNorm.normalizeBatch` as a round-trip-shape proxy for Rope/SwiGlu (same
H2D-batch + kernel + D2H + sync shape; not a claim of identical per-element math cost), at
`batch=136` and two widths — `dim=4096` (Rope/RmsNorm/ResidualAdd hidden-size scale) and
`dim=14336` (SwiGlu FFN-intermediate scale, mistral-7b):

| Shape | CPU scalar (whole-batch, per-row loop) | GPU ad-hoc round trip (whole batch, one launch) | GPU/CPU |
|---|---|---|---|
| batch=136, dim=4096 | 1.733 ms/call | 2.290 ms/call | 1.32x slower |
| batch=136, dim=14336 | 5.926 ms/call | 7.699 ms/call | 1.30x slower |

The GPU round trip is *still* slower than CPU scalar even at prefill batch size — not the crossover
expected from batching alone. But `CudaRmsNorm.normalizeBatch` stages through the exact same
`Arena.ofConfined()` pattern Finding 3 flagged, so this result cannot be trusted as a verdict on
prefill-scale GPU residency. **It must be re-measured after Phase A's pinned-memory fix lands** before
any go/no-go decision on GPU-resident Rope/SwiGlu kernels (Phase B below).

## Overview

Three largely independent inefficiencies on the prefill path, found by profiling rather than assumed:
a fixable host-memory-allocation gap (Finding 3) that inflates every GPU round trip in the hot path
and confounds both this tier's and Tier 19's own measurements; a chunking default that redoes fixed
per-matrix work redundantly for no correctness reason (Finding 2); and an open, not-yet-decided
question about GPU-resident elementwise ops at prefill scale (Finding 5/6) that Phase A's fix must
resolve before Phase B can answer honestly. This tier fixes the first two as a real engineering
change — not a manual flag the user has to discover — and turns the third into an explicit,
re-measured checkpoint rather than either committing to new kernels blindly or dismissing the idea on
a confounded number.

## Scope and compatibility

### Phase A — pinned staging memory + adaptive prefill chunk sizing (the "fundamental fix")

1. **Pin host staging memory.** Replace `Arena.ofConfined()` staging allocations in
   `CudaMatVec`'s batched-GEMM paths (`Fp16Scratch`, `Q4KDequantScratch` — `sgemmHalfBatchedGemm`,
   `sgemmQ4KBatchedGemm`, and the small-batch `sgemmHalfBatched` for consistency) and in
   `CudaRmsNorm.normalizeBatch` with `cudaMallocHost`/`GpuBindings.gpuMallocHost()`-backed buffers,
   grown on demand and freed once — mirroring the grow-on-demand convention the *device*-side scratch
   (`dX`/`dWeight`/`dOut`, `dXh`/`dY`) already uses, just on the host side. Vendor-neutral per
   CLAUDE.md ("New GPU functionality should go through `GpuBindings`, not a vendor-specific class"):
   wire through `GpuBindings.gpuMallocHost()`, which already exists for both `CudaBindings` and
   `RocmBindings`. `RocmMatVec` has zero `sgemm` overrides today (named gap since Tier 13C/17/19) —
   this tier does not add one; it only ensures the binding-level fix isn't CUDA-only by construction.
2. **Adaptive prefill chunk sizing for `static` schedule.** Stop treating `--prefill-batch 32` as the
   only sizing signal for a single-request `static`-schedule prefill. When nothing else constrains
   it, size the chunk to cover the *whole* prompt, up to a VRAM-headroom ceiling (reuse whatever
   budget check `--gpu-layers auto`/Tier 5 already computes, rather than inventing a second one).
   `continuous` schedule keeps Tier 16's existing chunk sizing unchanged — that chunking exists for
   interleaving fairness with other requests' decode steps, a deliberate tradeoff Tier 16 already
   gates on, not a bug this tier should touch. Ship as the **default** behavior for `static` schedule
   (not a flag the user must find), while keeping `--prefill-batch` as an explicit override for
   memory-constrained setups that want the old fixed chunking.
3. Re-run this tier's own before/after evidence (Findings 2-4's exact nsys+JFR methodology) once 1-2
   land, plus §2's standard `compare-llama-cpp.sh`/`compare-lora.sh` gates.

### Phase B — re-measure GPU-resident Rope/SwiGlu at prefill scale (checkpoint, not pre-committed)

4. Re-run Finding 6's microbenchmark with Phase A's pinned memory in place. **Go** (build
   `RopeKernel`/`SwiGluKernel` scoped to the batched-prefill call sites, following Tier 19's own
   `RmsNormKernel`/PTX/parity-test pattern) only if the round trip now beats CPU scalar at prefill
   batch size. **No-go** (record honestly, close the avenue, do not build) if it still loses. Decode
   (batch=1) stays scalar regardless per Tier 19's own finding, unless a future session separately
   re-measures decode under pinned memory too — that is not this tier's exit gate.

**Phase B checkpoint result (2026-09-18): no-go.** Re-ran Finding 6's throwaway microbenchmark
(`CudaRmsNorm.normalizeBatch` as the round-trip-shape proxy, batch=136, real GTX 1080) against the
now-pinned-memory staging:

| Shape | CPU scalar (whole-batch, per-row loop) | GPU round trip (pinned memory) | GPU/CPU |
|---|---:|---:|---:|
| batch=136, dim=4096 | 0.697 ms/call | 1.519 ms/call | 2.18x slower |
| batch=136, dim=14336 | 2.948 ms/call | 4.614 ms/call | 1.56x slower |

Pinned memory did not close the gap — the round trip is still slower than CPU scalar, by a similar or
slightly wider margin than Finding 6's original confounded 1.30-1.32x. This confirms Tier 19's original
diagnosis: the bottleneck is the per-launch cost of one ad-hoc GPU round trip with no
activation-residency chain to amortize across, not memcpy speed. `RopeKernel`/`SwiGluKernel` are **not**
built on this evidence. Full numbers: `docs/perf-compare/20260918T153900Z-prefill-adaptive/INDEX.md`.

### Phase C — remove (not amortize) the remaining dequant cost (stretch, separately gated)

5. Extend the existing Q8_1/`dp4a` MMQ kernel (`Q4KMmqKernel`, Tier 13B — today decode-only, batch
   ≤ `HALF_SGEMM_BATCH_MAX`) into a real tiled batched quantized GEMM for large-batch prefill,
   removing the FP16-dequant-then-`cublasGemmEx` path entirely instead of amortizing it to
   once-per-matrix-per-request. This is the only lever that removes Finding 4's post-fix ~238ms
   rather than amortizing it, and is comparable in scope to authoring a new `mul_mat_q`-class kernel
   (new PTX, tiling, occupancy tuning) — **gate this on Phase A's measured results**; do not start
   speculatively, and do not let it block Phase A/B from shipping independently.

### Non-goals

- **Caching dequantized FP16 weights permanently in VRAM.** Ruled out this session by direct
  calculation: mistral-7b dense FP16 is ~14.5GB (7.24B params × 2 bytes) against a 4.37GB packed
  Q4_K_M file and an 8GB card — permanent caching does not fit and defeats the purpose of
  quantizing. Phase A's chunk-sizing fix achieves the same *within-request* dequant reduction (one
  dequant per matrix per request) without any extra residency, because it only needs the scratch
  buffer sized for one matrix at a time, reused across matrices in sequence — not all matrices
  resident simultaneously.
- Building Rope/SwiGlu GPU kernels before Phase B's re-measurement says they pay off.
- Changing `continuous` schedule's chunk sizing (Tier 16 owns that fairness tradeoff; untouched here).
- ROCm `sgemm` parity for the batched paths (named, pre-existing gap per Tier 13C/17/19 — not closed
  by this tier; Phase A's pinned-memory binding fix must not be CUDA-only by construction, but a
  working ROCm `sgemm` override is out of scope).
- Extending adaptive chunk sizing to Phi-2/Phi-3/Qwen3/Qwen3-MoE beyond whatever `--prefill-batch`
  already covers today for those handlers — no architecture-specific behavior change beyond the
  shared chunk-sizing call site, named follow-up if a gap is found.

## Chosen design

Phase A is two independent, additive changes reusing existing infrastructure end to end: pinned
memory reuses an FFI binding (`gpuMallocHost`) that already exists but is unused in the hot path, and
adaptive chunk sizing reuses whatever budget computation `--gpu-layers auto` already performs rather
than inventing a new one. Neither requires a new kernel, a new CLI flag as the primary interface, or
a new abstraction layer — this is deliberately the cheapest, lowest-risk fix that the session's own
measured evidence supports, consistent with CLAUDE.md's KISS guidance and Tier 19's own lesson about
not committing to bigger kernel work before measuring whether a cheaper fix already closes the gap.
Phase B stays a measurement-gated checkpoint, not a commitment, mirroring how Tier 19 itself refused
to proceed to `RopeKernel`/`ResidualAddKernel`/`SwiGluKernel` on an unverified assumption.

## New/modified classes

- **Modified**: `CudaMatVec` (`Fp16Scratch`/`Q4KDequantScratch` staging → pinned host memory),
  `CudaRmsNorm` (`normalizeBatch` staging → pinned host memory), whatever computes the `static`
  schedule's prefill chunk size today (chunk-size call site — leave exact location to the
  implementer; likely near `ContinuousPrefillState`/prefill-loop code shared with Tier 8/16), 
  `compare-llama-cpp.sh` (already has `--prefill-batch` pass-through per ROADMAP §2's "keep the
  regression gate able to parse what it gates" rule — no new flag needed unless adaptive sizing adds
  one).
- **New** (Phase B only, gated): `RopeKernel`, `SwiGluKernel` (PTX + Java binder, mirroring
  `RmsNormKernel`'s pattern) — only if Phase B's re-measurement says go.
- **New** (Phase C only, gated, stretch): a tiled batched-quantized-GEMM kernel extending
  `Q4KMmqKernel`'s dp4a approach to large batches.

## Feature × surface interaction matrix

| New feature / flag | Base inference | `--lora-play` | LoRA train | Vision | `--parallel` | `--gpu-layers` | `--prefill-batch` | CUDA | ROCm | Default |
|---|---|---|---|---|---|---|---|---|---|---|
| Pinned host-staging memory (`CudaMatVec`/`CudaRmsNorm`) | wired | wired (shares `CudaMatVec`) | wired (shares `CudaMatVec`) | follow-up (Phi-2 vision prefill never routes through `CudaMatVec`'s batched-GEMM paths) | wired (`forwardMultiDecode` shares `sgemmHalfBatched*`) | wired (orthogonal — pinning applies regardless of layer count) | wired (applies to every chunk size) | wired | binding exists (`hostMalloc`/`hostFree` in `RocmBindings`), no `sgemm` override to stage through it — follow-up | on |
| Adaptive whole-prompt chunk sizing (`resolveAdaptive`) | wired (`runLocalRepl`) | wired (same call site) | follow-up (LoRA train REPL keeps `resolve()`, unaffected) | wired (same call site/pipeline) | wired (shares `GenerationLoop.prefillBatchSize`) | wired (orthogonal) | explicit override always wins, unchanged | wired | wired (same `GpuBindings.memGetInfo` path) | on for `static` schedule + GPU; fixed 32 on CPU-only or `continuous` |

## Cross-feature smoke (before feature complete)

- Phase A correctness: existing `CudaSgemmBatchedPrefillParityTest`/`Q4KDequantParityTest`/
  `CudaRmsNormTest` must stay green against pinned-memory staging (same output, different host
  allocation) — extend rather than replace.
- Adaptive chunk sizing: prefill output (logits/tokens) must be identical to today's fixed
  `--prefill-batch 32` chunking for the same prompt — chunk boundaries must not change numerics
  (this is exactly what Tier 16's existing chunk-boundary parity tests already check for the
  `continuous` schedule; add the equivalent for `static`).
- `continuous` schedule must show **zero** change in chunk sizing/TTFT behavior (Tier 16's own gate
  stays the regression check — this tier does not touch that code path).
- `compare-lora.sh` (touches the forward pass / prefill path) — gate per ROADMAP §2.
- `compare-vision.sh` reasoned skip or run: check whether Phi-2's vision batched prefill shares any
  touched staging code first (Tier 17's own precedent found it doesn't — re-verify, don't assume).
- Phase B (if pursued): GPU-resident Rope/SwiGlu output within FP tolerance of scalar CPU, on the
  same covered-architecture list as Tier 19 (Llama-family/Mistral/Qwen2); Phi-3/Qwen3/Qwen3-MoE stay
  scalar, named follow-up, not silently implied covered.

## Exit checklist (compatibility)

- [x] Interaction matrix: which schedules/architectures get adaptive chunk sizing (static: yes —
      wired in `ConsoleMain.runLocalRepl()` local single-shard REPL, covering base inference,
      `--lora-play`, and vision, which all share that call site; `runClusterRepl()` and the LoRA
      train REPL keep the old fixed-32 `resolve()` path, named follow-up; continuous: unchanged, per
      design) and pinned memory (CUDA: yes; ROCm: binding exists (`hostMalloc`/`hostFree` implemented
      in `RocmBindings`), no `sgemm` override to apply it to yet — named, not silently implied
      covered).
- [x] No silent no-op: `--prefill-batch` explicit override still works exactly as today when passed
      (checked first in `resolveAdaptive`, ahead of the adaptive computation).
- [x] `docs/performance.md`, `docs/perf-compare/README.md`, and this tier's ROADMAP catalog row
      updated with the honest before/after numbers.
- [x] Phase B's go/no-go decision recorded with numbers either way, even if "no-go" — see "Phase B
      checkpoint" above: **no-go**.

## Verification and exit gate

Global rules (`PLAN-Infra-ROADMAP.md` → Execution rules): one Infra tier in flight; publish a
`docs/perf-compare/` bake-off (inference + LoRA; vision if applicable) before marking feature
complete.

**Feature complete** when Phase A lands: pinned memory verified via the same `nsys` methodology
(`cudaMemcpyAsync` host time should collapse toward the GPU-side DMA time it currently exceeds by
>10x), adaptive chunk sizing verified via the same before/after methodology as Finding 4 (on a prompt
long enough to exercise multiple old-default chunks), parity tests green, `continuous` schedule
regression-flat, `compare-llama-cpp.sh`/`compare-lora.sh` gates pass. Phase B's checkpoint measurement
is recorded (go or no-go) regardless of which way it lands — that recording, not a particular
outcome, is part of this tier's feature-complete bar. Phase C is explicitly out of this tier's
feature-complete bar (separately gated stretch goal).

**Gate met**: this tier does not target the P0 Phi-3.5/mistral decode-ratio gate directly (it is
prefill-side); its contribution is a `pp` throughput improvement on the existing
`docs/perf-compare/README.md` prefill bake-off rows, reported honestly against the
`20260915T042421Z` token-count-matched baseline.

## Implementation todos

1. Pinned host-staging memory for `CudaMatVec`'s batched-GEMM scratch and `CudaRmsNorm.normalizeBatch`
   via `GpuBindings.gpuMallocHost()`; parity tests extended; live `nsys` re-run confirming
   `cudaMemcpyAsync` host time collapses toward actual DMA time.
2. Adaptive `static`-schedule prefill chunk sizing (whole prompt up to a VRAM-headroom ceiling,
   reusing the `--gpu-layers auto` budget check); `--prefill-batch` stays a working explicit
   override; `continuous` schedule untouched; chunk-boundary numeric parity test for `static`.
3. Re-run this tier's own nsys+JFR before/after (mirrors Findings 2-4) as the tier's published
   evidence, plus `compare-llama-cpp.sh --gpu`/`compare-lora.sh`/vision-reasoned-skip-or-run.
4. Phase B checkpoint: re-run Finding 6's microbenchmark under pinned memory; record go/no-go with
   numbers; if go, build `RopeKernel`/`SwiGluKernel` following `RmsNormKernel`'s exact pattern,
   parity-test, wire behind the same auto-activate-on-CUDA-resident-path convention as
   `CudaRmsNorm`, re-measure the full bake-off.
5. Phase C (only if Phase A's numbers still leave meaningful dequant cost and a future session has
   budget for real kernel authorship): tiled batched quantized GEMM extending `Q4KMmqKernel`.
6. `docs/performance.md`, `docs/perf-compare/README.md`, ROADMAP catalog row and P0 step list
   updated with feature-complete status (Phase A/B) recorded separately from Phase C's stretch status.

## Preview files (expected)

Modified: `CudaMatVec.java`, `CudaRmsNorm.java`, the `static`-schedule prefill chunk-size call site,
`docs/performance.md`, `docs/perf-compare/README.md`, `docs/infra-plan/PLAN-Infra-ROADMAP.md`.

New (Phase B only, gated): `RopeKernel`/`SwiGluKernel` PTX + Java binder classes, parity tests,
`docs/perf-compare/<timestamp>/` (+ `-lora/`) bake-off artifacts.
