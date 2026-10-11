# Tier 02C: CPU hot path (integer kernels, allocation, threading)

Status: **complete** (2026-10-11). Every exit criterion is checked. The owner's pinned closing gate
(`docs/perf-compare/20261011T052951Z-cpu-hot-path-close-gate/`) met the CPU milestones on every sweep model (CPU tg 0.215x to
0.238x against 0.20x, pp at 128 0.224x to 0.256x against 0.10x; was 0.092x to 0.121x and 0.049x to 0.103x); GPU is unchanged
(tg 1.020x to 1.127x, pp 0.973x to 1.039x the pre-tier build, greedy equal); LoRA and vision gates met; decode allocation
fell from 20.8 MB to 0.87 MB per token; model quality 0.99935x to 1.00061x the float kernels. The attained-bandwidth gate is
**missed and reported** (Mistral 7B at 13.3% of host bandwidth against 25%; owner decision 4), and the end-of-plan CPU tg row
is restated to 0.25x with the holding term named. The CPU closing sweep `20261011T013059Z` is the new CPU reference column.
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

*Corrected 2026-10-10 (step 3): the serial accumulator does not explain the ~6%.* Splitting it into 4 or 8
independent float accumulators leaves the Q4_K decode kernel as fast as before (0.99x to 1.03x with 4, up to
1.15x slower with 8) and speeds Q6_K by 16% to 22%. That is about 3.5% of Mistral 7B decode time. The kernel is
bound by its instruction count per weight (about 4 to 5 core cycles per weight on one thread, 2.4 GB/s on 12
threads, 5.7% of the host's 42.4 GB/s), not by the add chain and not by memory
([`20261010T032108Z-cpu-cost-breakdown`](../perf-compare/20261010T032108Z-cpu-cost-breakdown/INDEX.md)). Item
1's remedy is unchanged and still the lever: integer dots over a once-quantized activation, vectorized,
cut the instructions per weight. The split accumulators are a means to that, not the fix in themselves.

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
   with the Vector API over `ByteVector`/`ShortVector`/`IntVector` at a species width **fixed in code per
   element type and chosen by measurement**, never wider than the host's integer SIMD width. The Vector
   API has no single unsigned-by-signed byte multiply-add of the kind the reference tool's AVX2 path uses;
   widen bytes to shorts, multiply, and add pairs into ints, and measure what C2 makes of it on this host
   rather than assuming.
   *Corrected 2026-10-09 (step 2): this item said "a fixed 256-bit species (the width the Q8_0 dequant path
   already fixes deliberately)" and "on AVX2". Both claims are false. The baseline host has **AVX but not
   AVX2, and no FMA** (`/proc/cpuinfo` flags; the JVM reports `UseAVX=1`, `UseFMA=false`). Without AVX2
   there are no 256-bit integer instructions, so a 256-bit `ByteVector`/`IntVector` cannot compile to
   hardware vector instructions here. The Vector API's preferred species is 128 bits for byte, int and
   float. The Q8_0 dequant path uses `FloatVector.SPECIES_PREFERRED` (128-bit here), not a fixed 256-bit
   species. Variant (d) is therefore measured at 128-bit integer species on this host. A float variant
   must not use `fma()`, which has no hardware instruction here. Item 4 marks every resulting choice
   host-specific.* The prefill path is a **cache-blocked GEMM**: a weight tile of rows by one K-block is unpacked
   or integer-dotted once against all B activation rows while it sits in L2, replacing the per-row loop of
   `sgemmQ4KWeightStationary`.
   *Widened 2026-10-10 (owner decision 3, after step 3):* the blocked GEMM covers **every** CPU prefill
   window on the sweep models, which means two gaps step 3 found are in this item. `Phi3TransformerHandler`'s
   CPU window path (`sgemmFusedInto`, today one `matVecInto` per batch row) is routed through the same
   batched kernels as the LLaMA handler, and Q6_K gets a batched kernel (today a per-row fallback that is
   38.7% of Mistral 7B CPU prefill). Without both, Phi-3.5-mini, the binding model, cannot reach the CPU pp
   milestone.

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
   one machine: an Intel Xeon E5-1650 v2 (Ivy Bridge-EP) with AVX, **no AVX2, no FMA, no AVX-512 and no
   VNNI** (corrected 2026-10-09; it said AVX2), 6 cores and 12 threads (per
   [`INVENTORY.md`](INVENTORY.md)). Several decisions turn on that — the fixed species width, the
   integer-variant choice in item 1 (VNNI would change it), the pool verdict in item 3, and the
   `--threads` default. For each conclusion the execution record reaches, state a one-line
   `host-specific` or `expected-general` marker and, where it is host-specific, what would have to be
   re-measured on an AVX2/FMA host and on an AVX-512/VNNI host.
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
- **Batched prefill coverage (owner decision 3)**: Phi-3's CPU window forward against its per-token forward on
  the same prompt (logits within the item 1 threshold), and a test that the window path takes the batched
  kernel for every weight format the sweep models carry, Q6_K included, so it cannot fall back to a
  per-row loop unnoticed.
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

- [x] Host attainable read bandwidth measured and published; the CPU cost breakdown (hot methods,
      allocation rate, GC pauses, lock/park time, the dependency-chain share) published before any CPU
      fix was implemented, and the implementation order matches what it ranked.
      *2026-10-10: `docs/perf-compare/20261009T220314Z-host-read-bandwidth/` and
      `docs/perf-compare/20261010T032108Z-cpu-cost-breakdown/`, published before any CPU fix. Step 5 followed the ranking:
      kernels (`20261010T085329Z-cpu-integer-kernels`), threading (`20261010T145223Z-cpu-threads`), allocation
      (`20261010T184826Z-cpu-decode-allocation`), re-measured after each.*
- [x] The README's CPU tg end-of-plan row restated against the roofline after the breakdown (item 5),
      never below 0.25x, with the holding term named if it is below 0.40x.
      *2026-10-10 (step 4): restated to 0.40x on every sweep model (README, "Restated 2026-10-10"), from
      `docs/perf-compare/20261009T220314Z-host-read-bandwidth/` and
      `docs/perf-compare/20261010T032108Z-cpu-cost-breakdown/`. It is not below 0.40x, so no holding term is
      owed. The term that would hold it lower is named there, with a re-restatement if item 1's kernel misses
      about 1.1 cycles per weight.*
- [x] Item 1's four kernel variants measured at B = 1, 128 and 741 and the adopted kernel chosen on that
      measurement; the integer kernel is the CPU default for Q4_K/Q5_K/Q6_K/Q8_0 at decode and prefill
      width, or the measurement that kept a float variant is recorded; prefill runs a cache-blocked GEMM,
      including Phi-3's CPU window path and Q6_K (owner decision 3, 2026-10-10).
      *2026-10-10: `docs/perf-compare/20261010T085329Z-cpu-integer-kernels/` section 4 (variants (a) to (d) at B=1 on
      Q4_K and Q6_K, at B=128 and 741 on Q4_K, (a) and (d) on Q5_K and Q6_K; Q8_0 measured at B=741 by the vision guard,
      0.42x). Integer kernels are the default at both widths for all four formats; prefill is `CpuIntegerGemm` on every
      handler, Phi-3 and Q6_K included (`CpuBatchedPrefillCoverageLiveTest`).*
- [x] Hot-path allocation removed: `MatVec` has a non-allocating form, the transformer decode and
      prefill paths use it, the per-row dequant scratch is hoisted out of the parallel lambda, and the
      allocation-rate threshold is met.
      *2026-10-10: `docs/perf-compare/20261010T184826Z-cpu-decode-allocation/`: Mistral 7B steady-state decode 0.87 MB
      per token against the 20.8 MB baseline (-95.8%, threshold a drop of at least 50%), no GC pause in any decode span.
      `MatVec.sgemvInto` (CPU in place, CUDA override, interface default); the LLaMA-family and Phi-3 decode paths use
      per-thread buffers (`CpuDecodeScratch`); prefill runs `CpuIntegerGemm` into the batch workspaces; the float
      weight-stationary kernels' dequant scratch is per thread. `CpuDecodeAllocationLiveTest` holds the ceiling at
      10.4 MB per token.*
- [x] Threading replaced or retained on measured evidence, and `--threads N` ships as a flag that
      governs the hot path, reaches forked cluster nodes and the facade, is passed through by
      `compare-llama-cpp.sh`, and leaves no property (`juno.simd.pool.size`) that silently does nothing.
      *2026-10-10: `docs/perf-compare/20261010T145223Z-cpu-threads/`. A persistent worker pool with chunked claiming
      measured at parity with the common pool (decode layer 0.97x to 1.01x, prefill 0.99x to 1.01x; static partitioning
      1.10x to 1.29x slower) and is retained for the control: `--threads`/`JUNO_THREADS` size it in every JVM. It reaches
      forked nodes (a live 3-node CPU cluster with `--threads 2`: every node JVM ran `-Djuno.threads=2` and served a
      request) and the facade (`JunoPlayer.Builder.threads`, `CpuThreads.setThreads`). `compare-llama-cpp.sh` passes it
      as `-Djuno.threads` (the property the flag sets; an older jar without the flag ignores it); thread count unchanged
      at 12 on one CPU sweep. `-Djuno.simd.pool.size` is no longer read, and the launchers refuse it with an error naming
      `--threads`.*
- [x] CPU tg milestone (>= 0.20x), CPU pp milestone (>= 0.10x at `n_prompt=128`) and the attained
      bandwidth threshold (>= 25% on mistral-7b) each met, or missed and reported plainly with the term
      that remained dominant.
      *2026-10-11: owed to the owner's pinned closing sweep (Part C of `dist/cpu-hot-path-close/run-gate.sh`). Unpinned
      (`docs/perf-compare/20261010T184826Z-cpu-decode-allocation/`): tg 0.244x to 0.362x and pp 0.221x to 0.254x on every
      model; Mistral 7B attained bandwidth 1.21 t/s x 4.37 GB = 5.3 GB/s, 12.5% of 42.4 GB/s, against the 25% gate. The
      term that remains dominant is per-weight instruction cost: the Q4_K vector kernel (about 57% of Mistral 7B decode)
      and the Q6_K scalar kernel (about 36%). Decision 4.*
      *2026-10-11, scored on the owner's pinned closing sweep (`docs/perf-compare/20261011T013059Z/`, gate summary
      `docs/perf-compare/20261011T052951Z-cpu-hot-path-close-gate/`): CPU tg milestone **met** (0.215x to 0.238x, Qwen2.5-3B
      binding); CPU pp milestone **met** (0.224x to 0.256x, TinyLlama binding); attained bandwidth **missed** (Mistral 7B
      13.3% of 42.4 GB/s, 14.0% of the pinned 40.34 GB/s, against 25%). The term that remained dominant: per-weight
      instruction cost in the Q4_K vector (about 57% of Mistral 7B decode) and Q6_K scalar (about 36%) decode kernels.
      Reported as missed by owner decision 4.*
- [x] Model-quality threshold met (NLL <= 1.01x the float-activation kernel on every sweep model).
      *2026-10-10: 0.99935x to 1.00061x over 2,048 tokens on all four sweep models, argmax agreement 98.8% to 99.8%
      (`docs/perf-compare/20261010T085329Z-cpu-integer-kernels/` section 5, `ModelQualityNllLiveTest`).*
- [x] Every earlier tier's CPU-inference ("row 1") correctness result re-verified under the new CPU
      defaults, with any output changes documented.
      *2026-10-10 (step 6 record): unit suites, RoPE/sliding-window/context-shift live checks, the NLL test, the CPU
      smokes (greedy text identical to the pre-tier build on TinyLlama and Mistral 7B at 128 and 512 tokens, both
      schedules), `ModelLiveRunnerIT` 50 of 50 on five models. **Evidence (not published):** test and smoke output, listed
      in the step 6 record.*
- [x] Every CPU conclusion in the execution record carries a `host-specific` or `expected-general`
      marker, and each host-specific one names what to re-measure on an AVX2/FMA host and on an
      AVX-512/VNNI host (AVX2/FMA added 2026-10-09: the baseline host has neither).
      *2026-10-10: consolidated in the step 6 record's marker table.*
- [x] Cross-surface checklist fully resolved, vision regression guard explicitly passing.
      *2026-10-11: every row resolved (cross-surface record) except the throughput halves of rows 2 (GPU unchanged A/B)
      and 9 (LoRA train and playback), owed to Parts A and L of the owner's gate. The vision guard passes, and so does
      the vision A/B (`docs/perf-compare/20261011T004601Z-cpu-hot-path-vision/`).*
      *2026-10-11: rows 2 and 9 resolved by the owner's gate: GPU unchanged (pinned A/B tg 1.020x to 1.127x, pp 0.973x to
      1.039x) and LoRA (training 1.000x on GPU and 1.022x on CPU, playback 1.006x and 1.032x),
      `docs/perf-compare/20261011T052951Z-cpu-hot-path-close-gate/`. Every row is now PASS or NEEDS-AMD-HARDWARE (row 3).*
- [x] Perf gate published for the CPU, vision and LoRA paths, GPU unchanged (>= 0.95x A/B).
      *2026-10-11: vision published and met (`docs/perf-compare/20261011T004601Z-cpu-hot-path-vision/`); the CPU microbench
      before and after each item is published with the item. Owed to the owner: the GPU A/B (Part A), the CPU closing
      sweep (Part C, published by the harness) and LoRA (Part L), all pinned.*
      *2026-10-11: all published: GPU unchanged and LoRA in `docs/perf-compare/20261011T052951Z-cpu-hot-path-close-gate/`
      (every gate met), the CPU closing sweep in `docs/perf-compare/20261011T013059Z/`, vision in
      `docs/perf-compare/20261011T004601Z-cpu-hot-path-vision/`.*
- [x] `CLAUDE.md`'s description of the CPU matmul path (`node` row and the Vector API paragraph)
      updated to describe what is now on the hot path.
      *2026-10-10: the Vector API paragraph names the integer kernels, `CpuKernelSelection` and the fallbacks; the
      `node` row names `CpuIntegerKernels`/`CpuIntegerGemm`/`CpuKernelSelection` and `CpuThreads`/`CpuWorkerPool`.*
- [x] Docs (`docs/howto.md` `--threads`, `docs/agent-arch.txt`, `docs/performance.md`) updated,
      Juno-native language only.
      *2026-10-10: `howto.md` (the `--threads` flag row and the integer-kernel paragraph), `agent-arch.txt` (the
      integer kernels, `CpuThreads`/`CpuWorkerPool`, `CpuDecodeScratch`), `performance.md` (a new section "CPU integer
      kernels and `--threads`", with the older SIMD section marked superseded), `README.md` (a What's new entry). The
      closing gate's pinned numbers are added to `performance.md` when the owner's run lands.*
- [x] `CHANGELOG.md` entry added.
      *2026-10-10: Session 125 (integer kernels, `--threads`, decode allocation).*

## Execution record

### 2026-10-09: step 1 (plan check, vision-width guard)

**Plan check.** `scripts/performance-tests/check-plan-thresholds.sh` passed before anything else in this
tier ("ok (22 tier files, milestone and end-of-plan tables checked)").

**Plan against code, re-verified at HEAD `aabc42cb`.** Every code claim the tier's "Why this tier" section
makes still holds:

- `LlamaTransformerHandler.matVecQ4KrawInto` accumulates a whole row into one serial `float acc` and
  dispatches with `IntStream.range(...).parallel()` on the common pool. `matVecQ6Kraw` has the same shape.
- `sgemmQ4KWeightStationary`, `sgemmQ5KWeightStationary` and `sgemmQ8_0WeightStationary` allocate their
  per-row `float[] dq` inside the `SimdThreadPool.forEachRow` lambda, and accumulate each (block, batch
  row) into one serial float.
- `SimdThreadPool.forEachRow` is `IntStream.parallel()` on the common pool. `-Djuno.simd.pool.size` still
  builds a `POOL` that only `diagnosticSummary()` reads.
- `CpuMatVec.sgemm` allocates `float[B][rows]` per call and delegates to `sgemmInto`, which is the same
  common-pool row loop with a single float accumulator per (row, batch row).
- Q6_K has no batched CPU kernel. Both handlers' batched dispatch fall back to one `matVecInto` per batch
  row for it.

No drift that changes scope or sequencing. Every model the tier names is on disk.

**Vision-width guard.** This is a regression test: no kernel has changed yet, so it passed on its first
run and was not seen failing. It is a new class rather than a `VectorQuantKernelsTest` case, because its
baseline is the whole pre-tier dispatch and not the `VectorQuantKernels` helpers.

- `node/src/test/java/cab/ml/juno/node/FrozenCpuKernels.java` holds verbatim copies of today's kernels
  (from `aabc42cb`): the three float weight-stationary kernels, the per-row Q6_K kernel, and
  `CpuMatVec.sgemmInto`, all on the common-pool parallel stream. They are the "today's default" baseline,
  and are never updated when the production kernels change.
- `VisionWidthCpuKernelGuardTest` times the production dispatch against them at B = 741 (moondream2's
  729 image tokens plus a prompt):
  - Q4_K, Q5_K, Q6_K and Q8_0 through `Phi2TransformerHandler.sgemmQuantBatch`, the exact path the
    moondream2 text prefill takes. It was made package-private for this, with no behaviour change.
    Shape: 256 rows by 2048 columns.
  - F32 through `CpuMatVec.sgemmInto`, the vision encoder's path. Shape: 256 by 1152.
- Method: three warm-ups each, then five alternating baseline/candidate timings. The candidate's median
  must be **<= 1.15x** the baseline's (the README noise floor). A relative-RMS bound of 0.02 against the
  baseline's output stops a kernel that is fast but wrong from passing.

Readings on this host, three runs (candidate median over baseline median, unpinned):

| Type | Run 1 | Run 2 | Run 3 | Baseline median (run 1) |
|---|---|---|---|---|
| Q4_K | 1.006 | 0.990 | 1.009 | 45.3 ms |
| Q5_K | 1.000 | 0.998 | 0.999 | 45.3 ms |
| Q6_K | 1.005 | 1.031 | 0.998 | 195.8 ms |
| Q8_0 | 1.030 | 1.010 | 1.059 | 76.3 ms |
| F32 | 0.971 | 0.994 | 1.010 | 23.2 ms |

All of them sit within 6% of 1.0, so the 1.15 bound leaves headroom against noise. One run 1 Q6_K
candidate timing read 475 ms against about 195 ms; the median absorbed it. **Evidence (not published):**
unit-test output only. This guard gates kernel changes; it is not a perf-gate reading.

**Related tests re-run, all passing:** `Q5KWeightStationaryBenchTest`, `VectorQuantKernelsTest` (one skip,
not caused by this change), `Phi2RopeTest`, `Phi2TransformerHandlerMultiDecodeTest`, `CpuMatVecSgemmTest`,
`MatVecSgemmIntoTest`, `SimdThreadPoolTest`.

**Out-of-tier changes:** none.

**Next:** step 2, measure the host's attainable read bandwidth (item 5): a multi-threaded sequential-read
microbenchmark over a buffer much larger than the last-level cache, median of three, with the DIMM speed
if it can be read. This is a CPU-heavy measurement and should run on an otherwise idle host.

### 2026-10-09: step 2 (host attainable read bandwidth)

**Plan correction found while measuring.** The host has AVX, not AVX2, and no FMA. Items 1 and 4 and the
host-marker exit criterion were corrected in place (dated notes there), as were [`INVENTORY.md`](INVENTORY.md)
and [Tier 04C](TIER-04C-packed-weight-matmul.md). Nothing measured so far depends on it. It changes how item 1's
integer variant (d) is built and read: the integer SIMD width on this host is 128 bits, and the Vector API's
preferred species reports that. Scope and sequencing are unchanged, because item 1 already chooses its kernel
by measurement. `host-specific`: an AVX2 host doubles the integer width and adds FMA, so all four item 1
variants are re-measured there.

**Tool.** `scripts/performance-tests/read-bandwidth-microbench.sh` runs `ReadBandwidth.java` (the JVM reading,
a single-file source program, no build needed) and `read-bandwidth.c` (a native cross-check, `-O3
-march=native -fopenmp`) over a 3 GiB buffer (250x the last-level cache). Each thread owns a static slice and
sums it with four accumulators; each timing is four full reads. The script also records CPU flags, governor,
turbo, load average and the DIMM population. Named for what it measures, with no tier number.

**Published:** [`docs/perf-compare/20261009T220314Z-host-read-bandwidth/`](../perf-compare/20261009T220314Z-host-read-bandwidth/INDEX.md),
unpinned, two back-to-back runs, median of three per thread count.

| Threads | JVM (run 1 / run 2) | Native (run 1 / run 2) |
|---|---|---|
| 1 | 14.09 / 13.93 GB/s | 16.89 / 16.56 GB/s |
| 6 | 42.15 / 41.53 GB/s | **42.39** / 42.20 GB/s |
| 12 | 39.81 / 41.00 GB/s | 40.23 / 40.88 GB/s |

- **Host attainable read bandwidth: 42.4 GB/s** (peak median, 6 threads). This is the denominator of the
  attained-bandwidth gate: 25% of it is 10.6 GB/s, about 2.43 t/s on mistral-7b, or 0.44x of the reference
  tool's pinned 5.57 t/s. `expected-general` as a method; the figure itself is `host-specific`.
- **The JVM is not the bandwidth limit.** It is within 2% of native at 6 and 12 threads, and about 16% below
  native on one thread. `expected-general`.
- **Read bandwidth saturates at one thread per physical core.** 8 and 12 threads read 1% to 6% less than 6.
  This is an input to item 3's `--threads` default, not a verdict on it. `host-specific`: re-measure on a host
  with more memory channels per core, or with no SMT.
- **Juno today reads weights at 5.0% to 5.4% of 42.4 GB/s** on all four sweep models, against the reference
  tool's 41% to 57% (pinned CPU reference `20261001T180241Z`, file size as bytes per token). This matches the
  README roofline's estimate of about 6% against an assumed 40 GB/s.
- **DIMM speed not read** (`dmidecode` needs root). Eight 8 GiB DIMMs on four channels, two per channel (EDAC).
  42.4 GB/s is 83% of DDR3-1600's theoretical 51.2 GB/s, or 71% of DDR3-1866's 59.7 GB/s.

**Evidence status.** These are unpinned, while the CPU tg readings they are compared with are pinned. All-core
read bandwidth here is DRAM-bound (6 to 12 threads all land at 40 to 42 GB/s), so pinning is expected to move
it little. **Owed to the owner, recommended before the closing gate and not blocking step 3:** a pinned
re-run plus the DIMM speed, with the exact commands in the INDEX's "Caveats". If the pinned peak differs by more
than 5%, the gate denominator moves to it.

**Owner decisions (2026-10-09, after step 2).**
1. The attained-bandwidth gate's denominator is **42.4 GB/s**, the peak median at 6 threads, and not the
   12-thread reading (about 40.6 GB/s). It is the stricter of the two.
2. Item 1's species wording is corrected as recorded in item 1: the width is fixed in code per element type
   and chosen by measurement, 128-bit integer on this host, and not a fixed 256-bit species.

**Out-of-tier changes:** none.

**Next:** step 3, the CPU cost breakdown. JFR (`juno-perf.jfc`) of steady-state CPU decode on mistral-7b and a
128-token CPU prefill on mistral-7b and Phi-3.5-mini: hot methods, allocation rate and attribution, GC
pauses, lock and park time, plus item 1's variant (a) against variant (b) microbenchmark (the dependency
chain's share of decode time), with Juno's attained bandwidth share. Published before any CPU fix.

### 2026-10-10: step 3 (CPU cost breakdown)

**Plan against code, re-verified.** The code claims step 3 depends on still hold at HEAD `aabc42cb` plus the
working tree: `matVecQ4KrawInto`, `matVecQ6Kraw` and `matVecQ5Kraw` each accumulate a row into one float on
the common pool. The weight-stationary prefill kernels allocate their scratch inside the parallel lambda.
Q6_K has no batched CPU kernel. Two findings change what item 1's prefill half has to cover (decision 3
below):

- **Phi-3's CPU prefill is not batched.** `Phi3TransformerHandler.sgemmFusedInto` falls back to one
  `LlamaTransformerHandler.matVecInto` per batch row when it has no device matrix. A Phi-3.5-mini prompt
  token therefore costs the same as a decode token (1,063 against 1,083 ms). This is why Phi-3.5-mini binds
  the CPU pp milestone.
- **Q6_K's per-row fallback is 38.7% of Mistral 7B CPU prefill.**

**Recordings.** The JFR sweep, run `20261009T230709Z` (`compare-llama-cpp.sh --cpu`, Mistral 7B and
Phi-3.5-mini, `n_prompt=128`, `n_gen=64`, three measured repetitions per lane, `juno-perf.jfc`, unpinned, jar
`1dde81d172ee5b41`), was run by an earlier session on 2026-10-09 and left unpublished. This step verified
that it measured the current tree's jar, analysed it, and published it. A new single-file tool,
`scripts/performance-tests/JfrSpanBreakdown.java`, splits each recording into prefill (`juno.PrefillBatch`
windows) and decode (`juno.ForwardPass` at decode width). Per span it reports hot methods, allocation per token
from `jdk.ThreadAllocationStatistics` deltas, allocation attribution, GC pauses, and park and monitor time. A
first version found no prefill span because it looked only at `juno.ForwardPass`; batched prefill emits
`juno.PrefillBatch`. It also dropped a busy-threads estimate, since `jdk.ExecutionSample` samples only a few
Java threads per period.

**Variant (a) against (b).** `CpuDecodeKernelVariantMicrobench` with
`scripts/performance-tests/cpu-decode-kernel-variants.sh` (written 2026-10-09 by the earlier session, first run
here), three runs, at 12 threads and at 1. Its single-thread label read `threads=2` (a common-pool parallelism
of 0 still reports 1). The work was verified to run on the caller alone, and the label was fixed before run 3.

**Published:** [`docs/perf-compare/20261010T032108Z-cpu-cost-breakdown/`](../perf-compare/20261010T032108Z-cpu-cost-breakdown/INDEX.md).
Unpinned, median of three. A breakdown, not a gate.

| Model, lane | ms per token | Quantized matmul kernels (share of samples) | Allocated per token | GC pauses in span |
|---|---|---|---|---|
| Mistral 7B, decode | 1,972 | 99.2% (Q4_K 75.7%, Q6_K 21.3%, `f16ToF32` 2.1%) | 20.8 MB | 0 |
| Mistral 7B, prefill | 1,130 | 99.1% (Q4_K weight-stationary 59.4%, Q6_K per row 38.7%) | 261.5 MB | 3, 11.1 ms over 143 s |
| Phi-3.5-mini, decode | 1,083 | 98.9% | 35.2 MB | 3, 11.3 ms over 69 s |
| Phi-3.5-mini, prefill | 1,063 | 98.8% (unbatched) | 66.5 MB | 0 |

| Kernel at 12 threads | b-split4 / a-single (runs 1/2/3) | b-split8 / a-single |
|---|---|---|
| Q4_K 14336x4096 | 1.003 / 0.992 / 1.000 | 1.139 / 1.145 / 1.109 |
| Q6_K 4096x14336 | 1.009 / 1.017 / 0.946 | 0.837 / 0.839 / 0.782 |

- **Hot methods.** The quantized matmul kernels take 98.8% to 99.2% of samples on every lane. Attention, norms,
  RoPE and the fork/join frames take about 1% together. `expected-general` for a scalar-kernel JVM engine.
- **Dependency chain's share of decode: about 3.5%** (Mistral 7B: Q6_K's 21.3% times its 16.3% median
  saving; Q4_K saves nothing). The tier's premise that the serial accumulator explains the ~6% is corrected
  in "Why this tier". The cost is instructions per weight, about 4 to 5 cycles each on one thread.
  `host-specific`: no FMA and no 256-bit integer here. Re-measure variants (a) and (b) on an AVX2/FMA host,
  where C2 may fuse the float multiply-add, and the integer variants on an AVX-512/VNNI host.
- **Threading.** The kernels run 5.3x to 5.9x faster on 12 threads than on 1 (6 physical cores), and
  fork/join frames are 0.1% to 0.6% of samples. Per-dispatch cost is not separable at today's kernel speed
  (about 225 matmul dispatches per token, each taking milliseconds). `host-specific` (core count, SMT).
- **Allocation and GC.** Decode allocates 20.8 MB per token on Mistral 7B (10.5 MB/s). Decode's largest sampled
  site is KV-cache doubling (`DenseKvTensor.allocate`, a one-off per growth step), then the allocating
  `matVecQ4Kraw`, `ffn`, `rmsNorm` and `add`. Prefill allocates 261.5 MB per token, 99% of it the
  per-row scratch in `sgemmQ4KWeightStationary`'s lambda. GC pauses are 0% to 0.016% of any span.
  `expected-general`.
- **Lock and park.** No monitor is contended for 10 ms or more. Park totals sum idle workers and do not
  measure lost time (README).
- **Attained bandwidth.** Mistral 7B 2.29 GB/s (5.4% of 42.4 GB/s), Phi-3.5-mini 2.24 GB/s (5.3%). The
  reference tool in the same session: 54% and 60%. Unpinned ratios: tg 0.099x and 0.088x, pp at 128 0.077x and
  0.048x.

**Ranking (the step 5 order).** 1. Item 1, kernels: about 99% of time. 2. Item 3, threading: not separable now;
its fixed per-dispatch cost grows as a share once item 1 lands, so re-measure then. 3. Item 2, allocation:
no measurable throughput cost, but still owed for its own threshold. **Baseline for that threshold: 20.8 MB
per token** (Mistral 7B steady-state decode).

**Is enough cost attributable to reach the milestones?** Yes for the milestones, and stretch for the
bandwidth gate. CPU tg 0.20x on Mistral 7B needs about 1.11 t/s against the pinned reference, a 2.1x faster
decode, all of it inside the kernels item 1 replaces. CPU pp 0.10x on Phi-3.5-mini needs about a 2.1x faster
prefill, and batching alone gave Mistral 7B 1.75x over decode. The 25% attained-bandwidth gate needs 4.6x,
about one core cycle per weight across six cores. Step 4 restates the end-of-plan row against these figures.
Nothing here asks for escalation before implementing.

**Owner decisions raised (2026-10-10, after step 3).**
3. **Item 1's prefill GEMM covers Phi-3's handler and Q6_K.** Item 1 says "prefill runs a cache-blocked
   GEMM", and the exit criterion repeats it. It does not say that Phi-3's CPU prefill runs no batched kernel
   at all, or that Q6_K has none. Recommendation: yes. Route `Phi3TransformerHandler`'s CPU window path
   through the same blocked GEMM as the LLaMA handler (it already shares `LlamaTransformerHandler`'s
   kernels), and give Q6_K a batched kernel. Phi-3.5-mini binds the pp milestone, and Q6_K is 38.7% of
   Mistral 7B prefill, so the milestone is not reachable otherwise. The other choice is to keep the
   handler change out of this tier and report the pp milestone as missed on Phi-3.5-mini. Either way it
   stays within item 1's formats and needs no new tier.
   **Decided 2026-10-10 (owner): yes.** Item 1's scope, its exit criterion and the test list were widened
   to match.

**Out-of-tier changes:** none. `CpuDecodeKernelVariantMicrobench` (label fix) and `JfrSpanBreakdown.java` are
measurement tools, not on any product path.

**Owed to the owner, not blocking step 4:** the pinned re-run of the bandwidth microbenchmark and the DIMM speed
(step 2). Nothing new.

**Next:** step 4, restate the README's CPU tg end-of-plan row from this breakdown (item 5), never below 0.25x,
naming the holding term if below 0.40x. Then step 5 in the order above, starting with item 1 at B = 1, 128 and
741.

### 2026-10-10: step 4 (CPU tg end-of-plan row restated)

**Restated: CPU tg >= 0.40x on every sweep model** (was the 0.25x placeholder). Both of the README's tables
and its roofline section were updated in the same change (README, "Restated 2026-10-10").

**Derivation.** A tg ratio is Juno's weight-bandwidth share over the reference tool's. At the pinned CPU
reference (`20261001T180241Z`), against the measured 42.4 GB/s, the reference tool reads 40.8% (TinyLlama),
54.3% (Qwen2.5-3B), 50.4% (Phi-3.5-mini) and 57.3% (Mistral 7B, binding). Juno reads 5.0% to 5.4%. The share the
breakdown supports for items 1 to 3 is 25%, the tier's own attained-bandwidth gate. It needs about 1.1 core
cycles per weight on six cores at 3.5 GHz, against about 4 to 5 today. A 128-bit integer dot kernel should
cost roughly 0.4 to 0.7 if C2 emits packed multiply-adds. Memory (the JVM reads at native bandwidth) and
threading (5.9x on 12 threads) do not cap it below that. At 25% the binding model reads 0.436x. Non-kernel
time and per-dispatch cost, which grow as a share once the kernels are faster, bring that to about 0.41x.
Hence 0.40x.

**What is and is not evidence.** The shares and the cycles per weight are measured. The 0.4 to 0.7 cycles for
an integer kernel is an estimate from instruction counts. C2's code generation for the Vector API's
byte-widening multiply-add is not yet measured; item 1 measures it. If item 1 shows the kernel cannot reach
about 1.1 cycles per weight, the row is restated again, never below 0.25x, naming that term. `host-specific`:
on an AVX2 host the integer width doubles, and on AVX-512 with VNNI a single-instruction byte dot exists. Both
move the budget, so the row is re-derived there.

**Unchanged.** The intermediate milestone (CPU tg >= 0.20x after this tier) and this tier's gates. The 25%
attained-bandwidth gate is the restated row's own premise, so meeting it on Mistral 7B is about what the row
asks of that model.

**Plan check:** `check-plan-thresholds.sh` passes after the change.

**Out-of-tier changes:** none.

**Next:** step 5, item 1 first: the four kernel variants at B = 1, 128 and 741 (variants (c) and (d) are new),
then the integer kernel and the blocked prefill GEMM, including Phi-3's CPU window path and a batched Q6_K
kernel (decision 3). Tests first: kernel correctness per format against the float oracle, the batched-prefill
coverage test, and the model-quality NLL live test.

### 2026-10-10: step 5, item 1 (integer CPU kernels, decode and prefill)

**What shipped (uncommitted).** Every quantized CPU matmul over Q4_K, Q5_K, Q6_K or Q8_0 weights now quantizes
its activation row once to 8 bits (one float scale per 32 values) and runs integer dot products:

- `CpuIntegerKernels`: decode width. Q4_K and Q5_K run on 128-bit Vector API lanes. The 16 packed bytes are
  reinterpreted as eight 16-bit lanes, and the even and odd bytes' nibbles are split with shifts and masks, as
  is the 8-bit activation, so the inner loop has no conversion and no shuffle. Each sub-block's lane sums are
  reduced as integers and scaled in scalar float. Q6_K and Q8_0 run on scalar integer loops.
- `CpuIntegerGemm`: prefill width, cache-blocked. Rows are split into tiles of 16. Each 256-column block of a
  tile is unpacked once to 16-bit lanes and dotted against every activation row while it sits in L1, and
  each output is written once. It serves all four formats, so Q6_K has a batched kernel for the first time.
- `CpuKernelSelection`: the switch every dispatch point asks. It references no Vector API type, so without
  `--add-modules jdk.incubator.vector`, or with `-Djuno.cpu.integerKernels=false`, every CPU matmul takes the
  float-activation kernels as before. This was found when `coordinator`'s tests, which run without the
  module, hit `NoClassDefFoundError`. Startup logs one `CPU quantized matmul: ...` line.
- Wired into `LlamaTransformerHandler` (`matVecInto`, `matVecQuantizedNoEvent`, `sgemmLayerInto`),
  `Phi2TransformerHandler.sgemmQuantBatch`, `Phi3TransformerHandler.sgemmFusedInto` (decision 3: its CPU
  window path was a per-row loop), and the Qwen3 and Qwen3-MoE `sgemmLayerInto`. Those two also looped the
  per-row matVec; they are not sweep models, but item 1 makes prefill a blocked matmul everywhere.
- LoRA training's frozen forward stays on the float activation (`LlamaTransformerHandler.matVecFloatActivation`,
  used by `LoraTrainableHandler` and `LoraResidentWeights`). Its backward pass differentiates that product with
  float transposes, so a quantized forward would make the gradient belong to a different function. With the
  integer forward, CPU-against-GPU gradient parity failed (worst element 8.5% of the largest gradient, against
  the test's 5%). With the float forward it passes, and training numerics are unchanged.

**Tests first.** `CpuIntegerKernelsTest` was written before `CpuIntegerKernels` existed and seen failing to
compile, for the right reason. It then caught a tolerance bug in itself, which was fixed by deriving the
bound from the quantization step, not by loosening it. `CpuIntegerGemmTest` was written after a first draft of
`CpuIntegerGemm` (recorded as such; it was not seen failing first). Its determinism case then caught two real
bugs, below. `ModelQualityNllLiveTest` and `CpuBatchedPrefillCoverageLiveTest` are new. Each kernel test
checks the kernel against an exact reference (dequantized weights against the quantized activation, in
double) to float rounding, which catches any lane or bit error.

**Two bugs found and fixed before measuring:**

1. **Nondeterminism.** `FloatVector.reduceLanes(ADD)` leaves the order unspecified, and the interpreted and
   C2 forms differ, so a row's result depended on which compilation tier ran it. The 8-bit rounding amplified
   that to a 0.5 to 0.66 logit difference, caught by `SlidingWindowLiveTest.qwen25`, which requires exact
   equality. Lane sums now use a fixed order, or are integer reductions, which are exact.
2. **Reentrancy.** A common-pool worker waiting on its own matmul may run another task's matmul on the same
   thread; with a per-thread activation buffer the second call overwrote the first. A reentrant call now
   takes a fresh buffer. Prefill tile scratch comes from a shared pool, taken once per task rather than once
   per tile: the common pool clears a worker's thread-locals after each task, so per-thread scratch was
   reallocated per tile.

A third problem was found by measuring, below. The first wiring's Q5_K decode was boxed by C2.

**Kernel variants (the item 1 measurement), [`20261010T085329Z-cpu-integer-kernels`](../perf-compare/20261010T085329Z-cpu-integer-kernels/INDEX.md) section 4.**
At decode width (B=1), the four variants on Q4_K and Q6_K at 12 threads and at 1. At prefill width (B=128 and
741), (a) to (d) on Q4_K, and (a) and (d) on Q5_K and Q6_K:

| Choice | Measured (ratio of time against today's float kernel, 12 threads) | `host-specific`? |
|---|---|---|
| Q4_K/Q5_K decode: Vector API integer lanes | Q4_K 0.328 shipped (0.300 for the float-accumulating form, which C2 boxed in the production loop); Q5_K 3.5 ms per call against 11.1 ms boxed | yes: AVX only, 16 vector registers; re-measure on AVX2/FMA and AVX-512/VNNI |
| Q6_K/Q8_0 decode: scalar integer | Q6_K 0.652 (vector form 1.51, register pressure) | yes |
| Prefill, every format: `CpuIntegerGemm` | Q4_K 0.235 to 0.329, Q5_K 0.219 to 0.292, Q6_K 0.141 to 0.146 (against per-row); float split 0.76 to 0.90, scalar integer 0.94 to 1.06 | the tile size is; the method is `expected-general` |
| Split float accumulators | rejected: Q4_K 1.00 to 1.15, Q6_K 0.78 to 0.95 | yes |

**C2 boxing, found by measuring.** The first sweep showed Phi-3.5-mini decode allocating 6.9 GB per token
(310 GC pauses), with `ShortVector.lanewiseTemplate`, `convert0` and `lanewiseShiftTemplate` at the top of
the allocation sites. The Q5_K row loop exceeded C2's `NodeCountInliningCutoff`, so the lane helpers were
not inlined and every vector was boxed. Splitting the group into a float-returning helper made it worse,
because C2 inlined the helper back. The fix that holds is fewer vector operations: integer lane reduction
per sub-block, scalar float scaling, no float vectors in the decode kernels. Each kernel now allocates 4.7 KB
per call (stream overhead) and is bitwise repeatable. `expected-general`: any large Vector API loop on C2
can hit the cutoff, so check allocation, not only time.

**Reading, unpinned, four sweep models (section 1 of the publication):**

| Model | pp ratio at 128 | tg ratio | Juno tg t/s (pre-fix pinned) | Attained bandwidth share (pre-fix) |
|---|---|---|---|---|
| TinyLlama | 0.236x | 0.308x | 7.75 (3.14) | 12.2% (5.0%) |
| Qwen2.5-3B | 0.232x | 0.236x | 2.58 (1.08) | 12.8% (5.4%) |
| Phi-3.5-mini | 0.255x | 0.255x | 2.22 (0.90) | 12.5% (5.1%) |
| Mistral 7B | 0.228x | 0.252x | 1.30 (0.51) | **13.4%** (5.3%) |

On this unpinned reading every row is above both CPU milestones (tg >= 0.20x, pp >= 0.10x at 128). The
attained-bandwidth gate (>= 25% on Mistral 7B) is not met. Mistral 7B decode is now 56.7% Q4_K vector kernel
and 36.4% Q6_K scalar kernel, so a faster Q6_K decode is the next term inside item 1. Prefill allocation fell
from 261.5 to 107.4 MB per token, because the per-row dequantization scratch is gone. Decode allocation is
unchanged at 20.2 MB per token; that is item 2's work.

**Model quality (exit criterion), NLL over 2,048 teacher-forced tokens, integer against float-activation kernels:**

| Model | NLL integer | NLL float | Ratio (<= 1.01) | Argmax agreement, first 512 |
|---|---|---|---|---|
| TinyLlama | 2.45603 | 2.45566 | 1.00015 | 98.8% |
| Qwen2.5-3B | 1.69713 | 1.69766 | 0.99969 | 99.8% |
| Phi-3.5-mini | 1.35909 | 1.35826 | 1.00061 | 99.8% |
| Mistral 7B | 1.52792 | 1.52891 | 0.99935 | 99.8% |

Met on every sweep model (`ModelQualityNllLiveTest`, opt-in, 3 h 38 min). The text is the RoPE test's public-domain
Declaration excerpt followed by a frozen excerpt of the how-to reference, with each model's ids checked in.

**Batched prefill coverage (decision 3).** `CpuBatchedPrefillCoverageLiveTest` counts `CpuIntegerGemm` calls per
format during one 40-token window: Phi-3.5-mini takes the batched matmul for Q4_K, Q5_K and Q6_K, and TinyLlama
for Q4_K and Q6_K. The window's last logits match token-by-token decode: same greedy token, worst difference 0.23 of
43.2 (Phi-3.5-mini) and 0.18 of 12.5 (TinyLlama), against a 2% bound. It was written after the wiring, so it passed
on its first run. With `-Djuno.cpu.integerKernels=false` it fails, with zero batched calls, so it discriminates.

**Tests run, all passing on the final code:** `CpuIntegerKernelsTest`, `CpuIntegerGemmTest` (including bitwise
repeatability and concurrent calls from inside fork/join tasks), `VisionWidthCpuKernelGuardTest` (B=741:
Q4_K 0.345, Q5_K 0.355, Q6_K 0.161, Q8_0 0.420 of the pre-tier kernels; F32 1.005), the RoPE perplexity live
checks (TinyLlama 2.913, Qwen2.5-3B 1.282 and Qwen3-1.7B 2.522, against 2.88, 1.29 and 2.50 measured on the
float kernels when that test was written), `SlidingWindowLiveTest` (exact zero again), `ContextShiftLiveTest`,
the LoRA suites, and `coordinator`'s live tests on the float fallback.

**Full unit suite on the final code:** `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`
passed in every module: 2,300 tests, 0 failures, 0 errors, 52 skipped (`node` 40 min 20 s, `juno-player` 15 min 29 s).
An earlier pass had failed in `coordinator` (missing Vector API module, fixed by `CpuKernelSelection` above), which
is why `vision`, `metrics` and `juno-player` first ran only on this pass.

**Tests changed, each for a stated reason, none loosened:**

- `PhiQuantizedMatVecTest`'s three weight-stationary parity cases now compare against the float per-row matVec,
  which is what the float weight-stationary kernels compute.
- `ContextShiftLiveTest` runs on the float kernels. Under the integer kernels the shifted and oracle requests'
  keys differ by float rounding, and 8-bit rounding turns that into steps of 0.08 to 0.25 logits (measured with
  the pin removed). The greedy token still agrees on every model. The test checks the re-rotation, which
  needs float precision.
- `LoraResidentWeightsTest` expects the float-activation forward that training uses.

**Out-of-tier changes:** none. Every change is item 1's, or a documentation update the tier lists.

**Next:** step 5 continues in the breakdown's order: item 3 (threading), measured now that the kernels are about
2.5x faster, then item 2 (allocation, baseline 20.8 MB per token). The faster Q6_K decode kernel stays open
inside item 1 against the bandwidth gate.

### 2026-10-10: step 5, item 3 (threading and `--threads`)

**Measured first** ([`20261010T145223Z-cpu-threads`](../perf-compare/20261010T145223Z-cpu-threads/INDEX.md)): the
integer kernels' row loops on common-pool parallel streams against a persistent worker pool, 12 threads each,
alternating on identical work (`scripts/performance-tests/cpu-dispatch-microbench.sh`, `CpuDispatchMicrobench`).

| Dispatch | Decode layer (TinyLlama / Mistral 7B shapes) | Prefill (B=741 / B=32) | `host-specific`? |
|---|---|---|---|
| Static contiguous partition, spin-then-park barrier (the tier's candidate) | 1.146 to 1.193 / 1.104 to 1.136 | 1.015 to 1.053 / 1.033 to 1.044 | yes: SMT, 12 threads on 6 cores |
| Chunks claimed from a shared counter, spin-then-park (shipped) | 0.974 to 1.007 / 0.981 to 1.004 | 1.000 to 1.007 / 0.992 to 1.001 | the parity is `expected-general` |

Neither wins on throughput. The worker pool is retained because `--threads` must govern the hot path in every
JVM: the common pool's size is fixed at JVM start, but a dedicated pool can be resized. The four-model CPU sweep on
the shipped build reads within 3.3% of the common-pool reading (average 1.7% lower, unpinned, below the noise
floor). The pinned closing gate decides whether that difference is real. `host-specific`: the dispatch verdict
should be re-read on a host without SMT, where static partitioning may not lose.

**What shipped (uncommitted):**

- `CpuWorkerPool`: persistent daemon workers plus the caller; chunks of about `n / (8 * threads)` claimed from an
  atomic counter; 50 microseconds of spinning, then park, between jobs. One job at a time: nested calls (from a
  worker or from the owning caller) run inline, and a concurrent caller waits, because the job already uses
  every thread. A worker's exception is rethrown to the caller.
- `CpuThreads`: the thread count (`-Djuno.threads`, else `JUNO_THREADS`, else available processors; `setThreads`
  rebuilds the pool) and `forEach`, which every CPU matmul loop now calls: integer and float-activation kernels,
  `CpuIntegerGemm` tiles and quantization, `CpuMatVec`'s F32 sgemm (the vision encoder), LoRA training's
  transposes, `SimdThreadPool.forEachRow`. `SimdThreadPool`'s unused pool and `-Djuno.simd.pool.size` are gone.
- `--threads N` for `local`, `cluster` and `lora` (`run.sh`, `run.bat`, `ConsoleMain`, with the `JUNO_THREADS`
  default). A value that is not a positive integer is refused at startup; so is `-Djuno.simd.pool.size`, with an
  error naming `--threads`. `ClusterHarness` passes `-Djuno.threads` to every forked node: the explicit value, else
  each node's even share of the cores, as the common pool already had. Facade: `JunoPlayer.Builder.threads(int)`;
  `LoraTrainer` documents the controls (`CpuThreads.setThreads` is public).
- `compare-llama-cpp.sh` passes `-Djuno.threads=<-t>`, plus the common-pool property for builds from before the
  worker pool; its selftest covers both, and its INDEX text names the flag. It passes the property, not the literal
  `--threads`, because the pre-tier `ConsoleMain` refuses unknown options, and the standing A/B runs that jar.

**Tests** (written after the pool, so regression tests that passed on their first run):

- `CpuThreadsTest`: precedence and refusal of bad values; every index runs exactly once at 1, 2 and 12 threads;
  the number of distinct threads running a loop equals the configured count; exceptions propagate; nested and
  concurrent callers are correct; both integer kernels' outputs are bitwise identical at 1, 2 and 12 threads.
- `SimdThreadPoolTest`: the property tests were replaced by a summary test.
- `CpuThreadsFlagTest` (`juno-player`): `--threads` parsing; forked-node count, explicit or even share; facade
  validation. The facade's apply-at-`build()` is not unit-tested, because building needs a model file.
- `scripts/performance-tests/smoke-cpu-hot-path.sh`, new: on all four sweep models at 4 and 12 threads, the
  server logs the requested count, decode and a 512-token prefill succeed (`prompt_tokens` 510 to 521), and greedy
  replies are identical across thread counts; moondream2 vision answers identically at both counts (173 s and
  256 s, bound 600 s). All checks passed.
- Live cluster: `./juno cluster --cpu --threads 2` on TinyLlama; all three node JVMs carried `-Djuno.threads=2`, and
  a chat request returned through the pipeline.

**Full unit suite after item 3:** all eleven modules pass, 2,312 tests, 0 failures, 52 skipped.

**Out-of-tier changes:** none.

**Next:** step 5, item 2: hot-path allocation (baseline 20.8 MB per token in Mistral 7B decode). Then the closing
work: step 6 (row-1 re-verification), the standing CPU and allocation gate, `compare-lora.sh` and
`compare-vision.sh`, and the pinned sweeps, which are the owner's runs.

### 2026-10-10: step 5, item 2 (hot-path allocation)

**Published:** [`20261010T184826Z-cpu-decode-allocation`](../perf-compare/20261010T184826Z-cpu-decode-allocation/INDEX.md),
unpinned, median of three.

| Model | Decode allocation per token, before | After | GC pauses in the decode span |
|---|---|---|---|
| Mistral 7B | 20.8 MB (step 3 baseline) | **0.87 MB** (-95.8%) | 0 |
| Phi-3.5-mini | 35.2 MB | 1.92 MB | 0 (was 3) |
| Qwen2.5-3B | 12.1 MB | 1.40 MB | 0 |
| TinyLlama | 5.6 MB | 0.38 MB | 0 |

Threshold (a drop of at least 50% on Mistral 7B steady-state decode): met. Throughput unchanged within noise.

**What shipped (uncommitted):**

- `MatVec.sgemvInto` for host F32, FP32-, FP16- and packed-Q4_K-resident weights. The interface default copies
  from `sgemv`; `CpuMatVec` writes in place and its `sgemv` delegates to it; `CudaMatVec`'s three existing private
  forms became public overrides.
- `CpuDecodeScratch`: per-thread, per-handler buffers for the single-token CPU decode path of
  `LlamaTransformerHandler` (LLaMA, Mistral, Qwen2) and `Phi3TransformerHandler`. The residual stream flows through
  one buffer with in-place adds, and the handler copies it once before returning. The non-allocating paths are
  `matVecProjectionInto` and `matVecEventInto`, which keep the `juno.MatVec` JFR event. Phi-3's old allocating
  helpers, now unused, were deleted.
- **A per-call KV allocation in every handler.** `kvCacheK.putIfAbsent(requestId, newKLayers(L))` built a full set
  of KV tensors (64 positions per layer) on every call, then discarded it whenever the request already had one:
  about 8.4 MB per token on Mistral 7B, at 14 sites in five handlers. It now allocates only for a new request
  (`!containsKey && putIfAbsent`, which keeps `putIfAbsent`'s race semantics). Found by measuring: the first
  scratch-only build still allocated 8.59 MB per token, and the arithmetic matched this exactly.
- The float weight-stationary kernels' block dequantization buffer is per thread (`dequantScratch`), not per row.

**Tests:**

- `MatVecSgemvIntoTest`, new: CPU in place and bit-identical, the interface default, and CUDA for all three device
  matrix types (`gpu` tag).
- `CpuDecodeAllocationLiveTest`, new: Mistral 7B decode at most 10.4 MB per token (it reads 0.20 MB). It was written
  after the change, not before; the pre-change rate of 20.8 MB would have failed it.
- Full unit suite: all eleven modules pass, 2,316 tests, 0 failures, 52 skipped. Sliding-window exact zero,
  batched-prefill coverage and the RoPE perplexities are unchanged.

**Not done in item 2, recorded:** `sgemvSameX` and `GpuBlasOps`' packed host copies (GPU-lane paths); the vision
encoder's `sgemm` calls (one allocation per image, not per token); prefill allocation (about 100 MB per token on
Mistral 7B, from per-window workspaces and the integer matmul's activation buffers, which are not covered by the
decode threshold); and the single-token decode of Phi-2, Qwen3 and Qwen3-MoE (not sweep models; they get the KV fix
but not the scratch).

**Out-of-tier changes:** none. The per-call KV allocation predates this tier, but removing hot-path allocation is
item 2's scope.

**Next:** step 6, re-verify every earlier tier's CPU correctness results under the new defaults. Then the closing
work: `compare-vision.sh`, the cross-surface table, and the owner's pinned gate (`dist/cpu-hot-path-close/`: GPU
unchanged A/B, the CPU closing sweep, pinned bandwidth, and LoRA).

### 2026-10-10: step 6 (row-1 correctness re-verified under the new CPU defaults)

Every earlier tier's CPU correctness checks were re-run on this tree (integer kernels, worker pool, decode scratch),
unmodified unless noted:

| Check | Result | Output changes |
|---|---|---|
| Full unit suite, eleven modules (after each of items 1, 3 and 2) | 2,300, 2,312, 2,316 tests; 0 failures | three tests re-pointed, each for a stated reason (item 1 record) |
| RoPE perplexity live checks (TinyLlama, Qwen2.5-3B, Qwen3-1.7B, 128 tokens) | pass: 2.902, 1.283, 2.559 | about 1% from the float kernels' 2.88, 1.29, 2.50 |
| `SlidingWindowLiveTest` (every handler) | exact zero, repeatable | none |
| `ContextShiftLiveTest` | pass on the float kernels (pinned, item 1 record) | with the integer kernels the oracle difference is 0.08 to 0.25 logits, the greedy token unchanged |
| `ModelQualityNllLiveTest` (2,048 tokens, four sweep models) | 0.99935x to 1.00061x the float kernels | argmax agreement 98.8% to 99.8% |
| `smoke-consistency.sh --no-gpu` | all checks pass | none |
| `smoke-grammar.sh`, `smoke-tools.sh` (CPU) | all checks pass | none |
| `smoke-long-prompt-prefill.sh --cpu --lengths "128 512" --baseline-jar <pre-tier>` | all checks pass | **greedy text identical to the pre-tier build** on TinyLlama and Mistral 7B, both schedules, 128 and 512 |
| `smoke-cpu-hot-path.sh` (new) | all checks pass | identical replies at 4 and 12 threads |
| `ModelLiveRunnerIT`, five models (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B, Qwen3-1.7B) | 5 of 5, **50 of 50 checks** (16 min) | check 9 first divergence none on every model, both schedules |
| GPU regression: `smoke-context-policy.sh`, `smoke-gpu-residency.sh`, `smoke-packed-kquant-matmul.sh` | all checks pass | none |
| Native REST `/v1/inference` on a CPU server, `--threads 6` | HTTP 200, answer correct | none |

`smoke-context-policy.sh --cpu` was started and stopped after 76 minutes. It uses Phi-3.5-mini at its 4,096-token
limit, and each turn re-prefills about 4,000 tokens on the CPU. It was re-run unmodified on its default GPU backend,
as Tier 02B ran it, where it passed. The behaviour it checks (fail at the limit without the opt-in, continue with it)
is independent of the matmul kernel, and CPU shifting is covered by `ContextShiftLiveTest`.

**Host-specific markers for every CPU conclusion in this tier** (exit criterion):

| Conclusion | Marker | Re-measure on AVX2/FMA | Re-measure on AVX-512/VNNI |
|---|---|---|---|
| Host bandwidth 42.4 GB/s; saturates at one thread per core | host-specific | the figure | the figure |
| JVM reads within 2% of native | expected-general | no | no |
| Kernels are about 99% of CPU time (step 3) | expected-general | no | no |
| The float add chain is about 3.5% of decode | host-specific (no FMA here) | variants (a) and (b): fused multiply-add changes the chain | as AVX2 |
| Q4_K/Q5_K decode: 128-bit integer lanes, integer reduction | host-specific | a 256-bit integer species (twice the lanes) | the VNNI byte dot-product instruction, if the Vector API maps it |
| Q6_K/Q8_0 decode: scalar integer beats vector | host-specific (16 vector registers, 128-bit integer) | the vector form at 256-bit integer width (still 16 registers) | the vector form with 32 registers |
| Prefill: `CpuIntegerGemm`, 16-row tiles | method expected-general, tile size host-specific | tile size | tile size and VNNI |
| Static partitioning loses to chunking | host-specific (SMT) | on a host without SMT | same |
| Chunked worker pool at parity with the common pool | expected-general | no | no |
| `--threads` default (all processors) | host-specific | the default against one thread per core | same |
| Decode allocation removal | expected-general | no | no |
| Activation quantization costs about 0.1% NLL | expected-general | no | no |

**Out-of-tier changes:** none.

**Next:** the closing work. `compare-vision.sh` A/B is running (mine; looser gates). The owner's pinned gate is
`dist/cpu-hot-path-close/`: the GPU-unchanged A/B, the CPU closing sweep that scores the milestones and the
bandwidth gate, pinned bandwidth, and the LoRA gates.

### 2026-10-10: cross-surface checklist (closing)

| # | Surface | Status | Evidence |
|---|---|---|---|
| 1 | CPU inference | PASS | integer kernels default; NLL 0.99935x to 1.00061x; step 6 re-verification; `smoke-cpu-hot-path.sh` |
| 2 | CUDA GPU inference | PASS | GPU smokes (context policy, residency, packed K-quant), `ModelLiveRunnerIT` checks 9 and 10 on CUDA, `MatVecSgemvIntoTest` on CUDA; the owner's pinned A/B: tg 1.020x to 1.127x, pp 0.973x to 1.039x, allocation 0.589x to 0.889x, greedy equal (`20261011T052951Z-cpu-hot-path-close-gate`) |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE | ROCm's host-side CPU matmuls and CPU attention run the same kernels and pool, unit-tested; no AMD device here |
| 4 | Static schedule | PASS | `smoke-long-prompt-prefill.sh --cpu` static (greedy text identical to the pre-tier build); `smoke-cpu-hot-path.sh` |
| 5 | Continuous schedule | PASS | the same smoke on continuous (identical to the pre-tier build); continuous prefill chunks run `CpuIntegerGemm` |
| 6 | Single-node local mode | PASS | every smoke above |
| 7 | Pipeline-parallel cluster | PASS | a live 3-node CPU cluster with `--threads 2` (every node at `-Djuno.threads=2`, request served); `ModelLiveRunnerIT` checks 1 to 6 |
| 8 | Tensor-parallel cluster | PASS | `ModelLiveRunnerIT` checks 7 and 8 on five models; `ClusterHarness` passes `-Djuno.threads` on the same path |
| 9 | LoRA training | PASS | the frozen forward stays on the float activation (CPU-against-GPU gradient parity passes); the transposes run on the worker pool; the owner's pinned gate: training 1.000x (GPU) and 1.022x (CPU), playback 1.006x and 1.032x |
| 10 | LoRA playback | PASS | inference through the integer kernels; `LoraQ4KPlaybackParityTest` and the LoRA suites pass |
| 11 | Vision | PASS | `VisionWidthCpuKernelGuardTest` at B=741 (0.16x to 0.42x the pre-tier kernels; F32 1.0x); the vision smoke at 4 and 12 threads; `compare-vision.sh` A/B ([`20261011T004601Z-cpu-hot-path-vision`](../perf-compare/20261011T004601Z-cpu-hot-path-vision/INDEX.md)): latency 0.565x (GPU lane) and 0.563x (CPU lane), decode 1.624x and 1.616x, gates met; the reply differs from the pre-tier build in one phrase |
| 12 | OpenAI REST surface | PASS | the smokes run `/v1/chat/completions`, streaming included |
| 13 | Native REST surface | PASS | `/v1/inference` on a CPU server at `--threads 6` |
| 14 | CLI | PASS | `--threads` on `local`, `cluster` and `lora` (`run.sh`, `run.bat`, `ConsoleMain`); a bad value and the removed `-Djuno.simd.pool.size` fail closed (`CpuThreadsFlagTest`); `compare-llama-cpp.sh` passes `-Djuno.threads` |
| 15 | JVM embedding facade | PASS | `JunoPlayer.Builder.threads`, public `CpuThreads.setThreads`, `LoraTrainer` documents both; without the Vector API module an embedder gets the float kernels and the startup log says so |

### 2026-10-11: closing state, decision 4, owed to the owner

**Ready for the owner:** `dist/cpu-hot-path-close/`, git-ignored. It holds the baseline jar `7ba4a3142c3ab0f4` (`aabc42cb`,
built from `git archive`), the candidate jar `6ae135b0e21839cf` (this tree, byte-identical to its own shaded jar), the
built baseline tree for Part L, `run-gate.sh` and `wait-and-run.sh`, and `READY`. Command, in a terminal left open:

```bash
bash dist/cpu-hot-path-close/wait-and-run.sh
```

It asks for sudo once and starts at once (READY exists, no Maven running). Default `PART=ACDL`, about four hours:

- **Part A:** GPU unchanged, pinned A/B on the four sweep models at 512 tokens: tg and pp at least 0.95x,
  allocation at most 1.10x, GC within bounds. Greedy equality is reported, not gated.
- **Part C:** the CPU closing sweep, pinned and published: tg ratio at least 0.20x and pp at least 0.10x on every
  model, and Mistral 7B attained bandwidth at least 25%.
- **Part D:** pinned host bandwidth plus the DIMM speed. It moves Part C's denominator if it differs from
  42.4 GB/s by more than 5%.
- **Part L:** LoRA on GPU and CPU, pinned by hand: training at least 0.95x, playback at least 0.80x.

After the run, the results are read into this file, the CPU reference column moves (README rule), and the tier's status
is set.

**Decision 4 (the owner's): the attained-bandwidth gate.** Unpinned, Mistral 7B decode reads its weights at about 12.5%
of the host's 42.4 GB/s, against the 25% gate (2.43 t/s). The milestones it sits beside (tg at least 0.20x, pp at
least 0.10x) are met on the unpinned reading, though Mistral 7B's and Qwen2.5-3B's tg margins against the pinned
reference are thin (an estimated 0.21x to 0.23x pinned).
- **(a)** Run the closing gate now and close the tier with the bandwidth gate reported as missed. The criterion allows
  that, provided the dominant term is named: per-weight instruction cost in the Q4_K vector and Q6_K scalar decode
  kernels. The restated end-of-plan row (0.40x, which assumed 25%) is then restated again from the pinned reading,
  per the README.
- **(b)** Keep Tier 02C open for one more kernel pass first: a vector Q6_K decode with fewer operations (36% of
  Mistral 7B decode) and a Q4_K pass toward about 1.1 cycles per weight. The closing gate is then re-run.

Recommendation: **(a)**, after looking at Part C's numbers. If the pinned tg milestone holds on every model, the tier
has done what its milestones ask, and the bandwidth miss is reported with its cause. If Part C misses the 0.20x tg
milestone on any model, take (b).

**Decided 2026-10-11 (owner): (a).** The closing gate runs as prepared. If Part C's pinned tg milestone holds on every
model, the tier closes with the bandwidth gate reported as missed, its cause named (per-weight instruction cost in the
Q4_K vector and Q6_K scalar decode kernels), and the end-of-plan CPU tg row restated again from the pinned reading. A
pinned tg miss on any model reopens the question.

**Not run, by design:** the `--cpu` context-policy smoke (stopped at 76 minutes; GPU run passed; step 6 record).

**Uncommitted:** all of this tier's work since `aabc42cb` (git is read-only in this repository).

### 2026-10-11: closing gate (owner run, pinned) and tier close

**Published:** [`20261011T052951Z-cpu-hot-path-close-gate`](../perf-compare/20261011T052951Z-cpu-hot-path-close-gate/INDEX.md)
(Parts A, D, L and the gate log) and [`20261011T013059Z`](../perf-compare/20261011T013059Z/INDEX.md) (Part C, the CPU closing
sweep, published by the harness). Baseline jar `7ba4a3142c3ab0f4` (`aabc42cb`), candidate `6ae135b0e21839cf`; clocks pinned for
every part.

| Part | Gate | Result |
|---|---|---|
| A, GPU unchanged (four sweep models, 512 tokens, A/B x3) | tg and pp >= 0.95x, allocation <= 1.10x, GC within bounds | **met**: tg 1.020x to 1.127x, pp 0.973x to 1.039x, allocation 0.589x to 0.889x, no GC pause in span on the candidate; greedy output equal in 3 of 3 on every model |
| C, CPU closing sweep | tg ratio >= 0.20x, pp >= 0.10x on every model; Mistral 7B bandwidth >= 25% | tg **met** (0.238x, 0.215x, 0.222x, 0.224x); pp **met** (0.224x, 0.230x, 0.256x, 0.226x); bandwidth **missed** (13.3%) |
| D, pinned host bandwidth | denominator moves only if beyond 5% | 40.34 GB/s at 6 threads (4.9% below 42.4: denominator kept); DIMMs DDR3-1600, 51.2 GB/s theoretical |
| L, LoRA (each tree's own script, A/B x3) | training >= 0.95x, playback >= 0.80x | **met**: training 1.000x (GPU) and 1.022x (CPU); playback 1.006x and 1.032x; recall correct on every run |

The script's final "GATES MISSED" line comes from the bandwidth gate alone, which is reported as missed by owner decision 4.

**Done in this close:**

- The README's CPU reference column moves to `20261011T013059Z`; `20261001T180241Z` is marked superseded, with banners in
  both INDEX files and `docs/perf-compare/README.md` (the README's reference rule).
- Both 02C milestone rows are retired as met (tg 0.215x, pp 0.224x).
- The end-of-plan CPU tg row is restated from 0.40x to its 0.25x floor (README, "Restated again 2026-10-11"). The holding
  term is named: per-weight instruction cost in the Q4_K vector and Q6_K scalar decode kernels, at about 1.5 and 2.7
  cycles per weight against the about 1.1 the 0.40x assumed.
- `docs/performance.md`, `INVENTORY.md` (pinned bandwidth, DIMM speed) and `CHANGELOG.md` carry the pinned numbers.

**Carried forward, unowned:** a faster Q6_K decode kernel and a Q4_K decode kernel nearer one cycle per weight (the
levers for the restated 0.25x row and the 25% bandwidth share); prefill allocation (about 100 MB per token); the
single-token decode scratch for Phi-2, Qwen3 and Qwen3-MoE. Tier 14 reports the CPU row's distance.

**Out-of-tier changes:** none.

**Next tier:** Tier 02D (GPU host overhead and decode-region coverage), the next row of the README's running order.
