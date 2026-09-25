# Tier 10: GPU backend breadth & CPU hot path

Status: not started
Gap analysis refs: §1.7 (plus the allocation and threading items below, which the gap analysis does
not cover — see "Why this tier, why now")

## Objective

Bring ROCm to real parity with CUDA for the batched tiled-GEMM path (today ROCm only has
strided-batched-GEMV, so large-batch prefill is GEMV-looped on AMD); fix the CPU hot path — which
means SIMD, but also the allocation rate and the threading model, not SIMD alone; and make an
explicit, scoped decision about whether to pursue additional backends (Metal, Vulkan, SYCL/CANN) at
all.

## Why this tier, why now

This is sequenced late because it's the most hardware-constrained tier in the plan — there is no
ROCm hardware available in this environment (per [`INVENTORY.md`](INVENTORY.md)), so the ROCm work
here is necessarily code-plus-unit-tests-without-hardware, flagged `NEEDS-AMD-HARDWARE`, same as
every other ROCm-touching tier. Doing it after Tiers 01-09 means the CUDA-side patterns this tier
ports to ROCm (residency, tiled GEMM, block-table attention, tensor-parallel slicing) are already
proven out, so ROCm parity work is porting a known-good design rather than co-developing it blind.
The CPU work is independent and could in principle move earlier, but is grouped here since it's
conceptually "backend breadth/parity," not a new feature.

**This tier was originally scoped as a SIMD fix alone. That scope cannot reach its own stated goal,
and the repository's own measurements say so.** Back-to-back CPU sweeps with the Vector API off and
on show the difference is a wash: tinyllama tg is 3.271 t/s at `--vector 0`
(`docs/perf-compare/20260918T031702Z/`) and 3.300 t/s at `--vector 1`
(`docs/perf-compare/20260918T032455Z/`), and the Juno/llama.cpp CPU tg ratio sits at 0.106x to
0.147x under both. A dot-product vectorization on this host (Xeon E5-1650 v2: AVX2, no AVX-512, no
VNNI) does not close a roughly ninefold gap on its own. The other two costs are visible directly in
the code and are the kind a JVM engine loses to a C++ one on by default:

- **Allocation.** `MatVec.sgemv` is declared to return a **new** `float[]` per call, and every
  device-matrix overload honours that; `CpuMatVec.sgemm` allocates `float[B][rows]` per call; and
  `LlamaTransformerHandler.sgemmQ4KWeightStationary` allocates its 1 KB per-row dequant scratch
  **inside** the parallel row lambda, so a single FFN matmul produces one allocation per row. This is
  garbage generated per token, per layer, on the hottest path in the system.
- **Threading.** `SimdThreadPool.forEachRow` dispatches via `IntStream.parallel()` on
  `ForkJoinPool.commonPool()` — work-stealing task split per matmul, no persistent workers, no
  barrier, no affinity. Its own javadoc records that `-Djuno.simd.pool.size` builds a pool the hot
  path then ignores. There is no user-facing thread-count control anywhere in Juno, which is also why
  `compare-llama-cpp.sh` can pass `-t N` to llama-bench and nothing equivalent to Juno.

Fixing the kernel while leaving those two in place would produce another measured-and-shelved
result, which is the pattern this plan exists to stop repeating.

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
4. **Eliminate hot-path allocation.** Add an output-parameter form to the `MatVec` contract — a
   `void sgemvInto(..., float[] out)` alongside the existing allocating `sgemv`, with the allocating
   form kept as a default that delegates to it so no caller breaks. **Tier 01B already added the
   batched half of this contract** (`sgemm` writing into a caller-supplied buffer, so
   `sgemmLayerInto` stops allocating and copying on the GPU path); match that spelling rather than
   introducing a second one, and check what it left to do before designing.

   Then route the transformer handlers' CPU decode and prefill paths through the non-allocating form
   with a reusable per-request (or per-slot) buffer. Hoist the per-row dequant scratch in `sgemmQ4KWeightStationary`,
   `sgemmQ5KWeightStationary` and `sgemmQ8_0WeightStationary` out of the parallel lambda into a
   reusable per-worker buffer. Do the same for `CpuMatVec.sgemm`'s per-call `float[B][rows]`.
   Measure with `jdk.ThreadAllocationStatistics` (a rate), using `jdk.ObjectAllocationSample` only
   to attribute what remains to call sites — not by inspection, and not by treating the sampled
   event as a rate.
5. **Replace common-pool dispatch and expose a real thread-count control.** Measure
   `IntStream.parallel()` on `ForkJoinPool.commonPool()` against a persistent worker pool with a
   spin-then-park barrier, at both decode width (B=1, where per-dispatch cost dominates) and vision
   width (B around 741, the shape that caused the previous SIMD regression); adopt whichever wins on
   measurement rather than on argument. Ship a real `--threads N` flag (Juno-native naming, with the
   matching `JUNO_THREADS` env var, following the existing flag conventions) that actually governs
   the hot path, replacing the current situation where `-Djuno.simd.pool.size` sizes a pool the
   kernels do not use. This flag is a precondition of the benchmark-parity work in
   [`README.md`](README.md)'s "Benchmark parity preconditions" — until it exists, Juno and llama.cpp
   cannot be run at matched parallelism, so every published ratio carries an unquantified thread-count
   mismatch.
6. **Backend-breadth decision**: explicitly evaluate whether Metal/Vulkan/SYCL/CANN support is
   worth pursuing given Juno's target audience and the JVM/Panama-FFI architecture, and record the
   decision (pursue as a new, separately-scoped future tier; or explicitly decline with reasoning)
   rather than leaving it an open question indefinitely.
7. **Mark which of this tier's CPU conclusions are host-specific.** Every CPU number this tier
   produces comes off one machine: an Intel Xeon E5-1650 v2 with AVX2, **no AVX-512 and no VNNI**,
   12 threads (per [`INVENTORY.md`](INVENTORY.md)). Several of this tier's decisions turn directly on
   that — the fixed safe species width chosen in item 3 instead of `SPECIES_PREFERRED`, whether the
   SIMD kernel is worth putting on the hot path at all (the `--vector 0` versus `--vector 1` wash
   that rescoped this tier is a *this-host* result), the persistent-pool-versus-`commonPool` verdict
   in item 5, and the `--threads` default. For each conclusion the execution record reaches, state a
   one-line `host-specific` or `expected-general` marker and, where it is host-specific, what would
   have to be re-measured on an AVX-512/VNNI host. This is not an essay per finding; it is the line
   that tells a future reader on different silicon which results to trust and which to re-derive.
   [Tier 04C](TIER-04C-packed-weight-matmul.md) carries the same obligation for its GPU findings on
   this host's Pascal-generation GTX 1080.

### Out of scope

- Actually implementing Metal/Vulkan/SYCL/CANN backends — this tier only makes the go/no-go
  decision; if "go," that becomes a new tier of its own, scoped and sequenced separately (likely a
  large, multi-tier effort in its own right given each is a distinct FFI/kernel-language surface).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | primary target for the SIMD, allocation and threading work; must remain correct (bit-identical to the existing scalar path within float tolerance) while getting faster |
| 2 | CUDA GPU inference | must remain unaffected — this tier's GPU work targets ROCm specifically |
| 3 | ROCm GPU inference | primary target for the tiled-GEMM/MMQ parity work, NEEDS-AMD-HARDWARE for final validation |
| 4 | Static schedule | the CPU items and ROCm tiled-GEMM must all work under static micro-batching |
| 5 | Continuous schedule | same, under continuous's mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev/test surface for all three CPU items (needs no GPU) |
| 7 | Pipeline-parallel cluster | ROCm nodes in a mixed CUDA/ROCm cluster (if that's ever a real deployment shape) must interoperate correctly — verify at least that a ROCm-only cluster works end to end |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm LoRA training's CPU fallback path (frozen weights kept FP16/FP32 host-resident) benefits from or is at least unaffected by the SIMD, allocation and threading changes; training runs long enough that an allocation-rate reduction should show up in its GC behaviour, so measure it rather than assuming it is inherited |
| 10 | LoRA playback | ROCm playback path must work through the new tiled-GEMM kernel same as CUDA already does |
| 11 | Vision | `VisionEncoder`'s CPU path currently uses `CpuMatVec.INSTANCE` directly — confirm the SIMD fix doesn't reintroduce the vision-scale-batch-width regression (`B≈741`) that got the general-purpose `dot()` excluded from the hot path in the first place; this is the specific regression case to guard against with a dedicated test |
| 12 | OpenAI REST surface | end-to-end correctness and measured latency improvement via chat completions on both CPU and ROCm |
| 13 | Native REST surface | same |
| 14 | CLI | one new flag: `--threads N` (item 5). Any flag this plan ships is added to `compare-llama-cpp.sh`'s pass-through option set in the same change that ships it, so the standing regression gate can always exercise it — otherwise the gate silently stops covering the surface it is meant to gate. Existing `--vector`/`--gpu-layers`/backend-selection flags continue working unchanged; `-Djuno.simd.pool.size` is either wired to the new flag or removed, not left as a property that silently does nothing |

## Implementation steps

1. Write the vision-scale-batch-width regression test *first*, using the exact shape
   (`B≈741`) that broke the general-purpose `dot()` before, so any new SIMD hot-path change is
   automatically checked against the specific failure this codebase already hit once.
2. **Establish the CPU cost breakdown before choosing what to fix.** Capture a JFR recording of a
   steady-state CPU decode run on `mistral-7b` using the shared `juno-perf.jfc` configuration Tier 01
   added, and read five things off it via the `jdk.*` metrics Tier 01 taught `JfrMetricsExtractor` to
   emit: `jdk.ExecutionSample` hot methods, `jdk.ThreadAllocationStatistics` allocation *rate*,
   `jdk.ObjectAllocationSample` allocation *attribution* (the sampled event gives you the call sites,
   not the rate — do not compute a rate from it), `jdk.GCPhasePause` count and total, and
   `jdk.JavaMonitorEnter`/`jdk.ThreadPark` time. If Tier 01 did not land that extractor work, this
   step cannot run as written — escalate rather than eyeballing a recording in a GUI and calling it
   a published breakdown. That breakdown ranks items 3, 4 and 5 against
   each other instead of assuming the SIMD kernel is the dominant term — the `--vector 0` versus
   `--vector 1` data quoted above is direct evidence that it may not be. Publish the breakdown.
3. Fix the three CPU items in the order the breakdown ranks them, re-measuring after each rather
   than only at the end, so a negative result is attributable to one change:
   - the SIMD hot path (species-width detection and/or dispatch-batching), measuring both the
     TinyLlama/Mistral decode case and the vision-scale case before enabling it by default;
   - the allocation removal (item 4), verified by allocation rate rather than by inspection;
   - the threading replacement and `--threads` flag (item 5), measured at both batch widths.
4. Build the ROCm tiled-GEMM kernel, porting the CUDA design.
5. Complete any remaining ROCm MMQ kernel coverage from Tier 04.
6. Make and record the Metal/Vulkan/SYCL/CANN decision.
7. Re-verify, not assume, that every earlier tier's row-1 ("CPU inference") correctness result still
   holds under the new CPU defaults — vectorized float accumulation can legitimately reorder
   floating-point sums vs. the scalar path, and a different thread count or a different work-splitting
   strategy can reorder a row-parallel reduction the same way. Re-run the greedy-decode correctness
   checks Tiers 00-09 already established (at minimum `ModelLiveRunnerIT`'s CPU-path checks and each
   tier's own CPU-path unit tests) with the new defaults on, and document any
   within-tolerance-but-non-identical output rather than silently treating "still passes" as "output
   is unchanged."

## Tests to write/upgrade before implementation

- **New `VectorQuantKernelsTest` case**: the vision-scale (`B≈741`) regression guard, asserting the
  new hot-path SIMD change does not reproduce the old "tens to hundreds of times slower" finding.
- **New `VectorQuantKernelsTest` case**: species-width-dependent correctness — run on a host
  reporting a narrow `SPECIES_PREFERRED` (or simulate via a forced-width test hook, if the
  hardware in this environment doesn't naturally expose a narrow width) to confirm correctness
  holds regardless of actual hardware vector width.
- **New `MatVec` output-parameter contract tests**: `sgemvInto` produces bit-identical results to the
  allocating `sgemv` for every backend implementation, and the retained allocating form still works
  for any caller that was not converted.
- **New allocation-rate assertion**: a steady-state CPU decode run over N tokens allocates below a
  stated bytes-per-token ceiling, read from JFR `jdk.ObjectAllocationSample`. Set the ceiling from
  the step-2 baseline measurement, not from a guess.
- **New threading tests**: correctness under `--threads 1`, `--threads 2` and
  `--threads <availableProcessors>` (a row-parallel reduction must produce the same result at every
  thread count, within float tolerance); and a test that `--threads N` actually changes the observed
  parallelism of the hot path, so this flag cannot regress into another property the kernels ignore.
- **New ROCm `CudaFp16GemmOps`-equivalent unit tests**: correctness against the existing
  strided-batched-GEMV oracle, at multiple batch sizes crossing the new tiled-GEMM threshold.
- **`ModelLiveRunnerIT`**: add a CPU-hot-path-enabled check (correctness + basic timing sanity) and,
  when AMD hardware becomes available, a ROCm tiled-GEMM check.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier10-backend-breadth.sh` — runs
  the CPU decode path at more than one `--threads` value and (where available) the ROCm path, plus
  the vision smoke case specifically, asserting no regression in either correctness or the
  previously-fixed pathological slowdown.
- **Perf gate (required)**: the CPU changes and ROCm tiled-GEMM are hot-path changes —
  `compare-lora.sh` and `compare-vision.sh` (vision is the specific regression risk here), plus a
  CPU-only microbenchmark before/after, plus `compare-llama-cpp.sh --cpu` for a llama.cpp-relative
  CPU reading (per README's llama.cpp-relative gate); publish under `docs/perf-compare/`.

  **Threshold**: CPU tg ratio >= **0.20x** llama.cpp on every sweep model, up from 0.106x to 0.147x
  — this is the intermediate milestone the program target in [`README.md`](README.md) assigns to this
  tier. Allocation rate during steady-state CPU decode must drop by >= **50%** against the step-2
  baseline, measured from `jdk.ThreadAllocationStatistics` rather than from the sampled allocation
  event. Vision gate unchanged from the existing rule: `latency_ms` <= 1.25x baseline, decode tps
  >= 0.80x baseline. Every throughput number here is a median of at least three runs with min/max
  published, per the README's noise-floor rule — this host resolves to about +-15%, and the CPU tg
  milestone is a roughly 1.4x move, so a single run cannot establish it. Note for whoever executes this: the milestone demands roughly a doubling, and
  the `--vector 0` versus `--vector 1` evidence says the SIMD kernel alone will not deliver it — if
  the step-2 breakdown does not identify enough attributable cost across items 3, 4 and 5 to
  plausibly reach 0.20x, say so before implementing rather than after, and escalate.

## Models needed

Existing models suffice. `moondream2-q5_k.llamafile` is specifically needed to exercise the
vision-scale batch-width regression guard (already present).

## Exit criteria

- [ ] CPU cost breakdown (hot methods, allocation rate, GC pauses, lock/park time) published before
      any CPU fix was implemented, and the implementation order matches what it ranked.
- [ ] CPU SIMD hot-path fix lands, measured faster than the current thread-parallel-scalar default,
      with the vision-scale regression guard passing (no repeat of the old pathological slowdown).
- [ ] Hot-path allocation removed: `MatVec` has a non-allocating form, the transformer decode and
      prefill paths use it, the per-row dequant scratch is hoisted out of the parallel lambda, and the
      measured allocation-rate reduction meets this tier's threshold.
- [ ] Threading replaced or retained on measured evidence (not argument), and `--threads N` ships as
      a flag that genuinely governs the hot path, is passed through by `compare-llama-cpp.sh`, and
      leaves no property (`juno.simd.pool.size`) that silently does nothing.
- [ ] CPU tg ratio milestone met (>= 0.20x on every sweep model), or missed and reported plainly with
      the breakdown explaining which term remained dominant.
- [ ] Every earlier tier's CPU-inference ("row 1") correctness result re-verified under the new CPU
      defaults — SIMD, allocation and threading together — with any output changes (even
      within-tolerance ones) explicitly documented rather than assumed away.
- [ ] ROCm tiled-GEMM implemented, unit-tested, marked NEEDS-AMD-HARDWARE pending real validation.
- [ ] Any remaining ROCm MMQ kernel coverage from Tier 04 completed.
- [ ] Metal/Vulkan/SYCL/CANN decision made and recorded (pursue as a new tier, or explicitly
      declined with reasoning) — not left open.
- [ ] Every CPU conclusion in the execution record carries a `host-specific` or `expected-general`
      marker, and each host-specific one names what to re-measure on an AVX-512/VNNI host.
- [ ] Cross-surface checklist fully resolved, vision regression guard explicitly passing.
- [ ] Perf gate published for both CPU and vision paths.
- [ ] `CLAUDE.md`'s "JDK Vector API CPU kernels" description (corrected in Tier 00 to describe the
      old, narrow reality) is updated again here to describe the new, actually-on-the-hot-path
      reality.
- [ ] `CHANGELOG.md` entry added.
