# Tier 10: GPU backend breadth & CPU SIMD hot path

Status: not started
Gap analysis refs: §1.7

## Objective

Bring ROCm to real parity with CUDA for the batched tiled-GEMM path (today ROCm only has
strided-batched-GEMV, so large-batch prefill is GEMV-looped on AMD), fix the CPU hot path so it
actually uses SIMD instead of thread-parallel scalar Java by default, and make an explicit,
scoped decision about whether to pursue additional backends (Metal, Vulkan, SYCL/CANN) at all.

## Why this tier, why now

This is sequenced late because it's the most hardware-constrained tier in the plan — there is no
ROCm hardware available in this environment (per [`INVENTORY.md`](INVENTORY.md)), so the ROCm work
here is necessarily code-plus-unit-tests-without-hardware, flagged `NEEDS-AMD-HARDWARE`, same as
every other ROCm-touching tier. Doing it after Tiers 01-09 means the CUDA-side patterns this tier
ports to ROCm (residency, tiled GEMM, block-table attention, tensor-parallel slicing) are already
proven out, so ROCm parity work is porting a known-good design rather than co-developing it blind.
The CPU SIMD fix is independent and could in principle move earlier, but is grouped here since it's
conceptually "backend breadth/parity," not a new feature.

## Scope

### In scope

1. **ROCm tiled-GEMM**: build a `CudaFp16GemmOps`-equivalent for ROCm (`rocblas_gemm_ex` or
   equivalent), closing the gap where large-batch prefill on ROCm currently falls back to a serial
   GEMV loop.
2. **ROCm fused K-quant MMQ kernels**: if not already fully done in Tier 04 (check status there —
   Tier 04 scoped Q4_K/Q5_K/Q6_K ROCm parity; this tier is where any remaining ROCm kernel work from
   Tier 04, plus the newer formats from Tier 04's IQ/legacy-format work, get ROCm coverage if not
   already complete).
3. **CPU SIMD hot path fix**: get `VectorQuantKernels.dot()` (or a redesigned equivalent) onto the
   actual hot path by default, fixing the root causes identified in the gap analysis — the
   `SPECIES_PREFERRED`-width-dependent regression at large batch widths, and the per-call dispatch
   overhead. This likely means: detecting vector width at runtime and choosing a fixed safe species
   width (mirroring the deliberate choice already made for the Q8_0 dequant path) rather than
   `SPECIES_PREFERRED`, and/or batching the dispatch so per-call overhead is amortized — both ideas
   should be measured, not assumed, before choosing one.
4. **Backend-breadth decision**: explicitly evaluate whether Metal/Vulkan/SYCL/CANN support is
   worth pursuing given Juno's target audience and the JVM/Panama-FFI architecture, and record the
   decision (pursue as a new, separately-scoped future tier; or explicitly decline with reasoning)
   rather than leaving it an open question indefinitely.

### Out of scope

- Actually implementing Metal/Vulkan/SYCL/CANN backends — this tier only makes the go/no-go
  decision; if "go," that becomes a new tier of its own, scoped and sequenced separately (likely a
  large, multi-tier effort in its own right given each is a distinct FFI/kernel-language surface).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | primary target for the SIMD fix; must remain correct (bit-identical to the existing scalar path within float tolerance) while getting faster |
| 2 | CUDA GPU inference | must remain unaffected — this tier's GPU work targets ROCm specifically |
| 3 | ROCm GPU inference | primary target for the tiled-GEMM/MMQ parity work, NEEDS-AMD-HARDWARE for final validation |
| 4 | Static schedule | SIMD fix and ROCm tiled-GEMM must both work under static micro-batching |
| 5 | Continuous schedule | same, under continuous's mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev/test surface for the CPU SIMD fix (doesn't need any GPU) |
| 7 | Pipeline-parallel cluster | ROCm nodes in a mixed CUDA/ROCm cluster (if that's ever a real deployment shape) must interoperate correctly — verify at least that a ROCm-only cluster works end to end |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm LoRA training's CPU fallback path (frozen weights kept FP16/FP32 host-resident) benefits from or is at least unaffected by the SIMD fix |
| 10 | LoRA playback | ROCm playback path must work through the new tiled-GEMM kernel same as CUDA already does |
| 11 | Vision | `VisionEncoder`'s CPU path currently uses `CpuMatVec.INSTANCE` directly — confirm the SIMD fix doesn't reintroduce the vision-scale-batch-width regression (`B≈741`) that got the general-purpose `dot()` excluded from the hot path in the first place; this is the specific regression case to guard against with a dedicated test |
| 12 | OpenAI REST surface | end-to-end correctness and measured latency improvement via chat completions on both CPU and ROCm |
| 13 | Native REST surface | same |
| 14 | CLI | no new flags expected; existing `--vector`/`--gpu-layers`/backend-selection flags should continue working unchanged |

## Implementation steps

1. Write the vision-scale-batch-width regression test *first*, using the exact shape
   (`B≈741`) that broke the general-purpose `dot()` before, so any new SIMD hot-path change is
   automatically checked against the specific failure this codebase already hit once.
2. Investigate and fix the CPU SIMD hot path (species-width detection and/or dispatch-batching),
   measuring both the TinyLlama/Mistral decode-path case and the vision-scale case before deciding
   the fix is safe to enable by default.
3. Build the ROCm tiled-GEMM kernel, porting the CUDA design.
4. Complete any remaining ROCm MMQ kernel coverage from Tier 04.
5. Make and record the Metal/Vulkan/SYCL/CANN decision.

## Tests to write/upgrade before implementation

- **New `VectorQuantKernelsTest` case**: the vision-scale (`B≈741`) regression guard, asserting the
  new hot-path SIMD change does not reproduce the old "tens to hundreds of times slower" finding.
- **New `VectorQuantKernelsTest` case**: species-width-dependent correctness — run on a host
  reporting a narrow `SPECIES_PREFERRED` (or simulate via a forced-width test hook, if the
  hardware in this environment doesn't naturally expose a narrow width) to confirm correctness
  holds regardless of actual hardware vector width.
- **New ROCm `CudaFp16GemmOps`-equivalent unit tests**: correctness against the existing
  strided-batched-GEMV oracle, at multiple batch sizes crossing the new tiled-GEMM threshold.
- **`ModelLiveRunnerIT`**: add a CPU-SIMD-hot-path-enabled check (correctness + basic timing sanity)
  and, when AMD hardware becomes available, a ROCm tiled-GEMM check.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier10-backend-breadth.sh` — runs
  the CPU decode path and (where available) the ROCm path, plus the vision smoke case specifically,
  asserting no regression in either correctness or the previously-fixed pathological slowdown.
- **Perf gate (required)**: both the CPU SIMD fix and ROCm tiled-GEMM are hot-path changes —
  `compare-lora.sh` and `compare-vision.sh` (vision is the specific regression risk here), plus a
  CPU-only microbenchmark before/after; publish under `docs/perf-compare/`.

## Models needed

Existing models suffice. `moondream2-q5_k.llamafile` is specifically needed to exercise the
vision-scale batch-width regression guard (already present).

## Exit criteria

- [ ] CPU SIMD hot-path fix lands, measured faster than the current thread-parallel-scalar default,
      with the vision-scale regression guard passing (no repeat of the old pathological slowdown).
- [ ] ROCm tiled-GEMM implemented, unit-tested, marked NEEDS-AMD-HARDWARE pending real validation.
- [ ] Any remaining ROCm MMQ kernel coverage from Tier 04 completed.
- [ ] Metal/Vulkan/SYCL/CANN decision made and recorded (pursue as a new tier, or explicitly
      declined with reasoning) — not left open.
- [ ] Cross-surface checklist fully resolved, vision regression guard explicitly passing.
- [ ] Perf gate published for both CPU and vision paths.
- [ ] `CLAUDE.md`'s "JDK Vector API CPU kernels" description (corrected in Tier 00 to describe the
      old, narrow reality) is updated again here to describe the new, actually-on-the-hot-path
      reality.
- [ ] `CHANGELOG.md` entry added.
