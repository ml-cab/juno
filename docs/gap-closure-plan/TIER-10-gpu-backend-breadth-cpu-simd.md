# Tier 10: GPU backend breadth

Status: not started
Gap analysis refs: §1.7 (the backend half)

**Split 2026-10-04 (plan review): read this first.** This tier was "GPU backend breadth & CPU hot path".
Its CPU items (3, 4, 5, 7 and 9) moved to [Tier 02C](TIER-02C-cpu-hot-path.md), which runs directly
after Tier 02B: CPU is the largest gap in the program (about 10x on both tg and pp) and nothing between
here and Tier 02 moves it. This tier keeps ROCm parity, the ROCm attention port, the allocator-holdback
re-measurement and the backend-breadth decision. The moved items keep a pointer at their old numbers so
references from the closed tiers' records still resolve. The file name keeps its old spelling for the
same reason.

## Objective

Bring ROCm to real parity with CUDA for the batched tiled-GEMM path (today ROCm only has
strided-batched-GEMV, so large-batch prefill is GEMV-looped on AMD) and for the GPU attention kernel, and
make an explicit, scoped decision about whether to pursue additional backends (Metal, Vulkan, SYCL/CANN)
at all.

## Why this tier, why now

This is sequenced late because it's the most hardware-constrained tier in the plan — there is no
ROCm hardware available in this environment (per [`INVENTORY.md`](INVENTORY.md)), so the ROCm work
here is necessarily code-plus-unit-tests-without-hardware, flagged `NEEDS-AMD-HARDWARE`, same as
every other ROCm-touching tier. Doing it after Tiers 01-09 means the CUDA-side patterns this tier
ports to ROCm (residency, tiled GEMM, block-table attention, tensor-parallel slicing) are already
proven out, so ROCm parity work is porting a known-good design rather than co-developing it blind.

## Scope

### In scope

1. **ROCm tiled-GEMM**: build a `CudaFp16GemmOps`-equivalent for ROCm (`rocblas_gemm_ex` or
   equivalent), closing the gap where large-batch prefill on ROCm currently falls back to a serial
   GEMV loop. `RocmMatVec` has no `sgemm` override at all today.
2. **ROCm fused K-quant MMQ kernels**: if not already fully done in Tier 04 (check status there —
   Tier 04 scoped Q4_K/Q5_K/Q6_K ROCm parity; this tier is where any remaining ROCm kernel work from
   Tier 04, plus the newer formats from Tier 04's IQ/legacy-format work, get ROCm coverage if not
   already complete).
3. *Moved 2026-10-04 to [Tier 02C](TIER-02C-cpu-hot-path.md) item 1 (integer CPU kernels, redefined
   from the CPU SIMD hot-path fix).*
4. *Moved 2026-10-04 to [Tier 02C](TIER-02C-cpu-hot-path.md) item 2 (hot-path allocation).*
5. *Moved 2026-10-04 to [Tier 02C](TIER-02C-cpu-hot-path.md) item 3 (threading and `--threads`).*
6. **Backend-breadth decision**: explicitly evaluate whether Metal/Vulkan/SYCL/CANN support is
   worth pursuing given Juno's target audience and the JVM/Panama-FFI architecture, and record the
   decision (pursue as a new, separately-scoped future tier; or explicitly decline with reasoning)
   rather than leaving it an open question indefinitely.
7. *Moved 2026-10-04 to [Tier 02C](TIER-02C-cpu-hot-path.md) item 4 (host-specific markers for CPU
   findings). This tier's ROCm conclusions carry the same `host-specific`/`expected-general` marker.*
8. **ROCm port of the GPU attention kernel** (handed over by Tier 01B, 2026-09-30, owner decision).
   `--gpu-attention` is CUDA-only. The kernel is PTX loaded through the CUDA driver API, and
   `CudaGqaAttention.tryCreate` returns nothing for any other backend, so on ROCm attention runs on the
   CPU and a launch that requests the kernel (`on` or `auto`) says so at startup
   (`GpuAttentionSupport`). Port the attention kernel **current when this tier runs** — Tier 02's tiled
   kernel, with the per-layer window parameter Tier 02B uses. The device KV mirror (`DeviceKvCache`)
   already goes through the vendor-neutral `GpuBindings`; the port needs the kernel in HIP, its loader,
   and a `GpuAttentionMirror`/handler path that accepts the ROCm backend. It is not a mechanical
   translation: the kernel's reductions assume 32-lane warps (`__shfl_xor_sync` with a 32-bit mask, warps
   counted as threads/32), AMD wavefronts are commonly 64 lanes, and HIP code objects are built per GPU
   architecture rather than JIT-compiled from PTX. It sits here because item 1 is what gives ROCm a
   batched prefill at all. Once ported, the ROCm row of the startup notice goes, and
   `GpuAttentionSupportTest` is updated to match.
9. *Moved 2026-10-04 to [Tier 02C](TIER-02C-cpu-hot-path.md) item 5 (CPU tg target restated against the
   bandwidth roofline).*
10. **Re-measure the allocator holdback** (added 2026-10-04, owner decision, Tier 01C step 5).
   `DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES` is 64 MiB because, with the device full, the free-memory
   query still reported 44 to 54 MiB that no allocation could obtain, on one GTX 1080 (driver 580.173.02,
   desktop session). The upload stop rule reads that query, so the allowance must cover the holdback on
   every device Juno runs on. Re-measure it with `PrefillReserveDeviceTest`'s
   `theAllocatorWithholdsNoMoreThanTheReservesAllowance` on any other GPU this tier brings up (ROCm
   included, through `RocmBindings.memGetInfo`), and either confirm 64 MiB with the readings or make it
   per-device. The test already runs on every CUDA GPU test pass.

### Out of scope

- Actually implementing Metal/Vulkan/SYCL/CANN backends — this tier only makes the go/no-go
  decision; if "go," that becomes a new tier of its own, scoped and sequenced separately.
- CPU kernels, allocation and threading — [Tier 02C](TIER-02C-cpu-hot-path.md).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | must remain unaffected; re-measured by the standing CPU gate |
| 2 | CUDA GPU inference | must remain unaffected — this tier's GPU work targets ROCm specifically |
| 3 | ROCm GPU inference | primary target for the tiled-GEMM, MMQ and attention work, NEEDS-AMD-HARDWARE for final validation |
| 4 | Static schedule | the ROCm tiled-GEMM must work under static micro-batching |
| 5 | Continuous schedule | same, under continuous's mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | ROCm nodes in a mixed CUDA/ROCm cluster (if that's ever a real deployment shape) must interoperate correctly — verify at least that a ROCm-only cluster works end to end |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | ROCm training keeps its current residency; confirm it is unaffected |
| 10 | LoRA playback | ROCm playback path must work through the new tiled-GEMM kernel same as CUDA already does |
| 11 | Vision | the CLIP encoder runs on `CpuMatVec.INSTANCE`; N/A for the ROCm kernels, verify it stays so |
| 12 | OpenAI REST surface | end-to-end correctness via chat completions on ROCm, NEEDS-AMD-HARDWARE |
| 13 | Native REST surface | same |
| 14 | CLI | no new flags expected; the ROCm row of the GPU-attention startup notice goes once item 8 lands |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Build the ROCm tiled-GEMM kernel, porting the CUDA design.
3. Complete any remaining ROCm MMQ kernel coverage from Tier 04.
4. Port the GPU attention kernel (item 8).
5. Re-measure the allocator holdback on any device brought up (item 10).
6. Make and record the Metal/Vulkan/SYCL/CANN decision.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **New ROCm `CudaFp16GemmOps`-equivalent unit tests**: correctness against the existing
  strided-batched-GEMV oracle, at multiple batch sizes crossing the new tiled-GEMM threshold.
- **ROCm attention**: parity against the scalar CPU attention within the bound the CUDA kernel is held
  to, as a hardware-gated test; a unit test that the 64-lane reduction path is selected for a 64-wide
  wavefront.
- **`ModelLiveRunnerIT`**: when AMD hardware becomes available, a ROCm tiled-GEMM check.
- **New bash smoke script**: `scripts/performance-tests/smoke-backend-breadth.sh` — runs the ROCm path
  where available and otherwise asserts the documented fail-closed notices.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar.
- **Perf gate (required)**: `MatVec` and the attention loader change — `compare-lora.sh`,
  `compare-llama-cpp.sh --gpu` (per README's llama.cpp-relative gate); publish under
  `docs/perf-compare/`.

  **Threshold.** CUDA and CPU are unchanged by this tier: Juno tg and pp t/s **>= 0.95x** the pre-tier
  build on every sweep model, GPU and CPU, from a same-hour interleaved A/B with pinned clocks (README,
  "No-regression gates tighter than the floor are Juno-against-Juno"). The ROCm kernels carry no
  throughput threshold on this host, which has no AMD device; on first AMD hardware, ROCm prefill at a
  512-token window must be **>= 5x** today's GEMV-looped ROCm prefill, recorded as `NEEDS-AMD-HARDWARE`
  until then.

## Models needed

Existing models suffice. ROCm validation needs an AMD GPU, which is not present.

## Exit criteria

- [ ] ROCm tiled-GEMM implemented, unit-tested, marked NEEDS-AMD-HARDWARE pending real validation.
- [ ] Any remaining ROCm MMQ kernel coverage from Tier 04 completed.
- [ ] The GPU attention kernel current at this tier ported to ROCm (scope item 8): parity against the
      scalar CPU attention within the bound the CUDA kernel is held to, the ROCm fallback notice
      removed, marked NEEDS-AMD-HARDWARE pending real validation. Until then the fallback stays
      announced, never silent.
- [ ] Allocator holdback re-measured on every device brought up, or recorded as not re-measurable here
      (item 10).
- [ ] Metal/Vulkan/SYCL/CANN decision made and recorded (pursue as a new tier, or explicitly
      declined with reasoning) — not left open.
- [ ] Every ROCm conclusion in the execution record carries a `host-specific` or `expected-general`
      marker.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published: CUDA and CPU >= 0.95x the pre-tier build, and the standing CPU and
      allocation gate met.
- [ ] `CHANGELOG.md` entry added.
