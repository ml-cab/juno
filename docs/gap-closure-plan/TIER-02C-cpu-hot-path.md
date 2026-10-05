# Tier 02C: CPU hot path (integer kernels, allocation, threading)

Status: not started
Gap analysis refs: §1.7 (the CPU half), plus the allocation and threading items, which the gap analysis
does not cover — see "Why this tier, why now"

**Split out of [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md) and moved forward on 2026-10-04 (plan
review).** Item mapping, for references written before the split:

| Here | Was |
|---|---|
| item 1, integer CPU kernels (redefined, see below) | Tier 10 item 3 (CPU SIMD hot path) |
| item 2, hot-path allocation | Tier 10 item 4 |
| item 3, threading and `--threads` | Tier 10 item 5 |
| item 4, host-specific markers | Tier 10 item 7 |
| item 5, bandwidth roofline restatement | Tier 10 item 9 |

## Objective

Make Juno's CPU inference read its weights at a useful share of the host's memory bandwidth on decode,
and run prefill as a cache-blocked matrix multiply rather than a scalar row loop: integer dot products
over packed weights against a once-quantized activation, no garbage on the hot path, and a thread count
the user controls.

## Why this tier, why now

**CPU is the largest gap in the program and was scheduled last.** At the current reference, CPU tg reads
0.092x to 0.121x and CPU pp 0.049x to 0.103x of the reference tool (`docs/perf-compare/20261001T180241Z/`),
roughly a tenfold gap on both, against about 1.6x to 1.8x for GPU tg and 2x to 4x for GPU pp at 512
tokens. As part of Tier 10 this work ran after nine tiers, none of which moves a CPU number, and Tier 10's
own text said it was "independent and could in principle move earlier". The README's argument for pulling
Tier 01C forward (a gap nobody is working on, published tier after tier) applies here with more force. It
runs after Tier 02B and before Tier 03 so that Tier 04's mapped weight loading, which threads a
`MemorySegment` accessor through the same kernels, is written against the kernels this tier ships rather
than converting the old ones and then having them replaced.

**The lever was misidentified, and the code says so.** Tier 10 framed the fix as putting
`VectorQuantKernels.dot()` on the hot path and choosing a safe species width. The decode kernel
(`LlamaTransformerHandler.matVecQ4KrawInto`) dequantizes each nibble to a float and accumulates the whole
row into **one serial `float acc`**. That is a loop-carried floating-point dependency C2 may not reorder,
so it neither vectorizes nor overlaps: each row runs at the latency of one dependent add per element. That
explains the README roofline's uniform ~6% of attainable bandwidth on every model, a per-element cost
rather than a model-specific one. The prefill kernel (`sgemmQ4KWeightStationary`) has the same shape per
block. The back-to-back `--vector 0` / `--vector 1` sweeps (tinyllama tg 3.271 against 3.300 t/s,
`docs/perf-compare/20260918T031702Z/` and `20260918T032455Z/`) are consistent with this: the kernel being
switched was still a float dot. The reference tool's CPU path quantizes the activation once per matmul to
an 8-bit block format, integer-dots it against the packed weights with several independent accumulators,
and applies the block scales once per block. Integer accumulation is associative, so it can be split
across accumulators and vectorized without changing the result.

Two other costs remain, the kind a JVM engine loses to a C++ one on by default:

- **Allocation.** `MatVec.sgemv` is declared to return a **new** `float[]` per call, and every
  device-matrix overload honours that; `CpuMatVec.sgemm` allocates `float[B][rows]` per call; and
  `LlamaTransformerHandler.sgemmQ4KWeightStationary` allocates its 1 KB per-row dequant scratch
  **inside** the parallel row lambda, so a single FFN matmul produces one allocation per row.
- **Threading.** `SimdThreadPool.forEachRow` and the `matVecQ*raw` family dispatch via
  `IntStream.parallel()` on `ForkJoinPool.commonPool()` — work-stealing task split per matmul, no
  persistent workers, no barrier, no affinity. `-Djuno.simd.pool.size` builds a pool the hot path
  ignores, and there is no user-facing thread-count control. The harness matches the reference tool's
  `-t` through the common pool's parallelism property (Tier 01B item 7), which is a harness setting, not
  a product control.

## Scope

### In scope

1. **Integer CPU kernels over packed weights** (redefines Tier 10 item 3). Quantize the activation row
   once per matmul to an 8-bit block format (per-32 or per-256 block scale, chosen by measurement), and
   compute the Q4_K/Q5_K/Q6_K/Q8_0 dots as integer multiply-adds over the packed bytes with **at least 4
   independent accumulators per row**, applying each block's scale and minimum once per block. Vectorize
   with the Vector API over `ByteVector`/`ShortVector`/`IntVector` at a **fixed 256-bit species** (the
   width the Q8_0 dequant path already fixes deliberately), not `SPECIES_PREFERRED`. The Vector API has no
   single unsigned-by-signed byte multiply-add of the kind the reference tool's AVX2 path uses; widen bytes
   to shorts, multiply, and add pairs into ints, and measure what C2 makes of it on AVX2 rather than
   assuming. The prefill path is a **cache-blocked GEMM**: a weight tile of rows by one K-block is unpacked
   or integer-dotted once against all B activation rows while it sits in L2, replacing the per-row loop of
   `sgemmQ4KWeightStationary`.

   **Measure the variants separately, on one kernel, before choosing**: (a) today's single-accumulator
   float kernel; (b) the same float kernel with 4 to 8 split accumulators (scalar, no Vector API — this
   isolates the dependency chain from everything else); (c) integer dot over a quantized activation,
   scalar loops written so C2's auto-vectorizer can take them; (d) integer dot with explicit Vector API.
   Read each at decode width (B=1) and at prefill width (B=128 and B≈741, the vision width that excluded
   the general SIMD dot from the hot path before). Adopt the best on measurement, per width if they differ.
2. **Eliminate hot-path allocation** (was Tier 10 item 4). Add an output-parameter form to the `MatVec`
   contract — a `void sgemvInto(..., float[] out)` alongside the existing allocating `sgemv`, with the
   allocating form kept as a default that delegates to it so no caller breaks. **Tier 01B already added
   the batched half of this contract**: the spelling is `MatVec.sgemmInto(A, X, Y)`, one overload per
   weight type, writing `Y[b][0, rows)`, checking `Y` first (`SgemmOutput`), bit-identical to `sgemm`;
   interface defaults copy from `sgemm`, `CpuMatVec` and `CudaMatVec` implement it and their `sgemm`
   delegates. Match that spelling. Left for this item: `sgemvInto` in the contract (`CudaMatVec` has
   private ones), the handlers' CPU window paths (they call their own weight-stationary kernels, and
   `Qwen3`/`Qwen3Moe` still copy from the allocating `matVec`), `sgemvSameX`, `GpuBlasOps`' packed host
   copies, and the vision encoder's `sgemm` calls.

   Then route the transformer handlers' CPU decode and prefill paths through the non-allocating form
   with a reusable per-request (or per-slot) buffer, including the quantized-activation buffer item 1
   adds. Hoist the per-row dequant scratch in `sgemmQ4KWeightStationary`, `sgemmQ5KWeightStationary` and
   `sgemmQ8_0WeightStationary` (or their item 1 replacements) out of the parallel lambda into a reusable
   per-worker buffer. Do the same for `CpuMatVec.sgemm`'s per-call `float[B][rows]`. Measure with
   `jdk.ThreadAllocationStatistics` (a rate), using `jdk.ObjectAllocationSample` only to attribute what
   remains to call sites.
3. **Replace common-pool dispatch and expose a real thread-count control** (was Tier 10 item 5). Measure
   `IntStream.parallel()` on `ForkJoinPool.commonPool()` against a persistent worker pool with a
   spin-then-park barrier and static row partitioning, at decode width (B=1, where per-dispatch cost
   dominates) and vision width (B around 741); adopt whichever wins on measurement. Ship a real
   `--threads N` flag (with the matching `JUNO_THREADS` env var, following the existing flag conventions)
   that actually governs the hot path; `-Djuno.simd.pool.size` is either wired to it or removed. Switch
   `compare-llama-cpp.sh` from the common-pool property to the flag in the same change, and confirm on one
   CPU sweep that the hot-path thread count is unchanged, so the switch is not a second measurement
   boundary.
4. **Mark which CPU conclusions are host-specific** (was Tier 10 item 7). Every CPU number here comes off
   one machine: an Intel Xeon E5-1650 v2 with AVX2, **no AVX-512 and no VNNI**, 12 threads (per
   [`INVENTORY.md`](INVENTORY.md)). Several decisions turn on that — the fixed species width, the
   integer-variant choice in item 1 (VNNI would change it), the pool verdict in item 3, and the
   `--threads` default. For each conclusion the execution record reaches, state a one-line
   `host-specific` or `expected-general` marker and, where it is host-specific, what would have to be
   re-measured on an AVX-512/VNNI host.
5. **Restate the CPU tg end-of-plan target against the memory-bandwidth roofline** (was Tier 10 item 9;
   README, "CPU tg: the memory-bandwidth roofline"). Measure the host's attainable read bandwidth (a
   multi-threaded sequential-read microbenchmark over a buffer much larger than the last-level cache, at
   the thread count `--threads` defaults to, median of three; record the DIMM speed if it can be read) and
   publish it with the step-3 breakdown. From then on every CPU tg reading this tier publishes carries
   Juno's attained weight bandwidth (model weight bytes read per token x tg) and its share of the measured
   bandwidth, beside the reference tool's share. After the breakdown, restate the README's CPU tg
   end-of-plan row: the ratio Juno would read at the bandwidth share the breakdown says items 1 to 3 can
   plausibly reach, never below the 0.25x placeholder; if the restated number is below 0.40x, say which
   term holds it there.

### Out of scope

- GPU work of any kind; ROCm (Tier 10).
- Quantized formats Juno does not load yet (Tier 04); this tier covers the formats the CPU path runs
  today: Q4_K, Q5_K, Q6_K, Q8_0 and the float types.
- Mapped weight loading (Tier 04 item 0), which runs after this tier against these kernels.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | primary target; must remain correct against the float oracle within the item 1 quality threshold while getting faster |
| 2 | CUDA GPU inference | must remain unaffected; the GPU path's host-side work (LM head fallback, sampling, partial-offload CPU layers) uses these kernels and is re-measured by the standing gate |
| 3 | ROCm GPU inference | N/A for the kernels; ROCm attention runs on the CPU and so inherits the change — verified by unit tests, NEEDS-AMD-HARDWARE for end to end |
| 4 | Static schedule | the new kernels must work under static micro-batching at every `--parallel` width |
| 5 | Continuous schedule | same, under continuous's mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev/test surface (needs no GPU) |
| 7 | Pipeline-parallel cluster | `--threads` must reach forked nodes (`ClusterHarness` sets the common-pool parallelism per node today); verify the per-node value |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm LoRA training's CPU path benefits from or is at least unaffected by the kernel, allocation and threading changes; measure its GC behaviour rather than assuming the allocation reduction is inherited |
| 10 | LoRA playback | CPU playback composes its delta against the new kernels; parity against the float path within the item 1 threshold |
| 11 | Vision | `VisionEncoder` runs on `CpuMatVec.INSTANCE`; the B≈741 shape that got the general-purpose `dot()` excluded must not regress — dedicated test, and `compare-vision.sh` is required |
| 12 | OpenAI REST surface | end-to-end correctness and measured latency improvement via chat completions on CPU |
| 13 | Native REST surface | same |
| 14 | CLI | one new flag: `--threads N` (item 3), added to `compare-llama-cpp.sh`'s pass-through set in the same change; `--vector` keeps working or is retired with a notice |
| 15 | JVM embedding facade | `JunoPlayer`/`LoraTrainer` run in the embedder's JVM, so the thread count must be settable through the facade (or documented as read from `JUNO_THREADS` / a system property at first use); an embedder must not be left on the common pool's default silently |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first. Then write the vision-scale-batch-width
   regression test, using the exact shape (`B≈741`) that broke the general-purpose `dot()` before.
2. Measure the host's attainable read bandwidth (item 5).
3. **Establish the CPU cost breakdown before choosing what to fix.** Capture a JFR recording of a
   steady-state CPU decode run on `mistral-7b`, and a second of a 128-token CPU prefill on the same model
   and on Phi-3.5-mini (the binding model for the CPU pp milestone), with `juno-perf.jfc`, and read off
   the `jdk.*` metrics `JfrMetricsExtractor` emits: `jdk.ExecutionSample` hot methods,
   `jdk.ThreadAllocationStatistics` allocation rate, `jdk.ObjectAllocationSample` attribution,
   `jdk.GCPhasePause` count and total, and `jdk.JavaMonitorEnter`/`jdk.ThreadPark` time. Add item 1's
   variant (a)-against-(b) microbenchmark to it: the share of decode time the dependency chain alone
   accounts for. Publish the breakdown, with Juno's attained bandwidth share.
4. Restate the CPU tg end-of-plan row (item 5) from the breakdown, before implementing items 1 to 3.
5. Fix items 1, 2 and 3 in the order the breakdown ranks them, re-measuring after each, so a negative
   result is attributable to one change.
6. Re-verify, not assume, that every earlier tier's row-1 ("CPU inference") correctness result still
   holds under the new CPU defaults — integer accumulation over a quantized activation is a numerical
   change, not only a reordering, and a different thread count or work split can reorder a reduction.
   Re-run the greedy-decode checks Tiers 00 to 02B established (at minimum `ModelLiveRunnerIT`'s CPU-path
   checks and each tier's own CPU-path unit tests) and document any non-identical output.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **Vision-width regression guard**: a `VectorQuantKernelsTest` (or successor) case at B≈741 asserting
  the new kernel is no slower than today's default at that width.
- **Kernel correctness**: per format (Q4_K, Q5_K, Q6_K, Q8_0), the integer kernel against the float
  oracle on random and real-weight rows, at B = 1, 8, 128 and 741, including a `cols` that is a single
  super-block; activation-quantization error reported per format. Species-width independence under a
  forced-width test hook.
- **Model quality (item 1)**: a live test in the pattern of `RopePairingPerplexityLiveTest`: mean
  negative log-likelihood over a fixed 2,048-token text, teacher-forced, integer kernel against the
  float-activation kernel, on every sweep model.
- **`MatVec` output-parameter contract**: `sgemvInto` bit-identical to the allocating `sgemv` for every
  backend implementation, and the retained allocating form still works for unconverted callers.
- **Allocation-rate assertion**: a steady-state CPU decode run over N tokens allocates below a stated
  bytes-per-token ceiling read from `jdk.ThreadAllocationStatistics` (the `allocated_bytes_per_token`
  figure the harness publishes), the ceiling set from the step-3 baseline.
- **Threading**: correctness under `--threads 1`, `2` and `availableProcessors`, within float tolerance
  where a float reduction remains; and a test that `--threads N` changes the observed hot-path
  parallelism, so the flag cannot regress into another property the kernels ignore.
- **`ModelLiveRunnerIT`**: a CPU-hot-path check (correctness plus a basic timing sanity bound).
- **New bash smoke script**: `scripts/performance-tests/smoke-cpu-hot-path.sh` — runs CPU decode and a
  512-token CPU prefill at more than one `--threads` value on the four sweep models, plus the vision
  smoke case, asserting correct output and no repeat of the B≈741 slowdown.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar;
  for this tier its CPU half is the primary gate below, and its GPU-lane allocation half still applies.
- **Perf gate (required)**: `compare-lora.sh` and `compare-vision.sh`, a CPU microbenchmark per item
  before and after, plus `compare-llama-cpp.sh --cpu --pin-clocks` on the four sweep models at matched
  threads (per README's llama.cpp-relative gate); publish under `docs/perf-compare/`.

  **Threshold.**
  - CPU tg ratio **>= 0.20x** on every sweep model (milestone; reference 0.092x, mistral-7b binding, a
    2.2x move) and CPU pp ratio at `n_prompt=128` **>= 0.10x** on every sweep model (milestone;
    reference 0.049x, Phi-3.5-mini binding).
  - Attained weight bandwidth during mistral-7b CPU decode **>= 25%** of the host bandwidth measured in
    implementation step 2 (about 6% today, README roofline). On mistral-7b this is the stricter of the
    two decode gates: at about 40 GB/s it implies roughly 0.39x.
  - Allocation rate during steady-state CPU decode drops by **>= 50%** against the step-3 baseline,
    from `jdk.ThreadAllocationStatistics`.
  - Model quality: mean NLL with the integer kernel **<= 1.01x** the float-activation kernel's on every
    sweep model; greedy agreement over 512 tokens reported per model.
  - Vision gate unchanged from the existing rule: `latency_ms` **<= 1.25x** baseline, decode tps
    **>= 0.80x** baseline. LoRA: train **>= 0.95x** and wall-clock playback tps **>= 0.80x**.
  - GPU unchanged: Juno GPU tg and pp t/s **>= 0.95x** the pre-tier build on every sweep model, from a
    same-hour interleaved A/B with pinned clocks.
  - Every throughput number is a median of at least three runs with min/max published. If the step-3
    breakdown does not identify enough attributable cost across items 1 to 3 to plausibly reach the
    milestones, say so before implementing, and escalate.

## Models needed

The four sweep models (all Q4_K_M; Q5_K and Q6_K tensors occur inside them) and
`Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile` for Q8_0. `moondream2-q5_k.llamafile` for the vision-width
guard. All present.

## Exit criteria

- [ ] Host attainable read bandwidth measured and published; the CPU cost breakdown (hot methods,
      allocation rate, GC pauses, lock/park time, the dependency-chain share) published before any CPU
      fix was implemented, and the implementation order matches what it ranked.
- [ ] The README's CPU tg end-of-plan row restated against the roofline after the breakdown (item 5),
      never below 0.25x, with the holding term named if it is below 0.40x.
- [ ] Item 1's four kernel variants measured at B = 1, 128 and 741 and the adopted kernel chosen on that
      measurement; the integer kernel is the CPU default for Q4_K/Q5_K/Q6_K/Q8_0 at decode and prefill
      width, or the measurement that kept a float variant is recorded; prefill runs a cache-blocked GEMM.
- [ ] Hot-path allocation removed: `MatVec` has a non-allocating form, the transformer decode and
      prefill paths use it, the per-row dequant scratch is hoisted out of the parallel lambda, and the
      allocation-rate threshold is met.
- [ ] Threading replaced or retained on measured evidence, and `--threads N` ships as a flag that
      governs the hot path, reaches forked cluster nodes and the facade, is passed through by
      `compare-llama-cpp.sh`, and leaves no property (`juno.simd.pool.size`) that silently does nothing.
- [ ] CPU tg milestone (>= 0.20x), CPU pp milestone (>= 0.10x at `n_prompt=128`) and the attained
      bandwidth threshold (>= 25% on mistral-7b) each met, or missed and reported plainly with the term
      that remained dominant.
- [ ] Model-quality threshold met (NLL <= 1.01x the float-activation kernel on every sweep model).
- [ ] Every earlier tier's CPU-inference ("row 1") correctness result re-verified under the new CPU
      defaults, with any output changes documented.
- [ ] Every CPU conclusion in the execution record carries a `host-specific` or `expected-general`
      marker, and each host-specific one names what to re-measure on an AVX-512/VNNI host.
- [ ] Cross-surface checklist fully resolved, vision regression guard explicitly passing.
- [ ] Perf gate published for the CPU, vision and LoRA paths, GPU unchanged (>= 0.95x A/B).
- [ ] `CLAUDE.md`'s description of the CPU matmul path (`node` row and the Vector API paragraph)
      updated to describe what is now on the hot path.
- [ ] Docs (`docs/howto.md` `--threads`, `docs/agent-arch.txt`, `docs/performance.md`) updated,
      Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
