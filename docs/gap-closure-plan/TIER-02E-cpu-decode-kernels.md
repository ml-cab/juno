# Tier 02E: CPU decode and prefill kernels (Q6_K vector kernel, Q4_K instruction cost, SIMD width)

Status: **complete** (2026-10-27). Every exit criterion is checked. The owner's pinned closing gate run 2
(`docs/perf-compare/20261027T154748Z-cpu-kernels-close-gate-2/`) met every gate: CPU tg 0.259x to 0.337x against 0.25x and
pp at 512 0.422x to 0.512x against 0.30x on every sweep model (both end-of-plan CPU rows retired as met, owner decision
9), pp at 2048 3.27x to 4.38x the pre-tier build, same-hour pp 1.92x to 2.68x and tg 1.31x to 1.65x, allocation 0.98x to
1.01x; Q6_K 2.13 and Q4_K 0.93 cycles per weight on one thread; NLL within 1.0021x the float kernels; GPU, LoRA and vision
unchanged or faster; context shift re-verified on the CPU. 256-bit throughput is `NEEDS-AVX2-HARDWARE`. The history of
the tier follows. As of 2026-10-26: Steps 1 to 3 closed. Items 1 (vector Q6_K), 2 (Q4_K candidate 3 adopted;
`<= 1.1` not yet met), 6 (pooled prefill attention) and 7 (gemm tile G1 to G3) implemented and unit-tested; unpinned
CPU pp at 512 about 0.27x to 0.28x against `>= 0.30x`. Open: item 7 and item 2 candidates, item 3, the smoke script,
quality and live tests, the owner's pinned closing gate, docs. *2026-10-12, later: item 7's weight unpack vectorised
(adopted, exact); `mvn test -pl node` re-run green after G3. 2026-10-13: item 2 candidate 5 (activation split once
per call) adopted, exact; Q4_K about 0.91 cycles per weight on one thread and 1.06 per core at 12, unpinned, under
`<= 1.1` pending the pinned reading. 2026-10-13, later: step 5 (item 3, width-generic species) done; 128 bits on
this host, 256-bit throughput `NEEDS-AVX2-HARDWARE`; owner decision 4: `auto` is 128 on every host until 256 is measured.
2026-10-13, later: item 6's vectorization candidate run (the record had dropped it): `GqaVectorHead` adopted (value sums
in registers, A1, and the dot in vector lanes, A2); unpinned CPU pp at 512 about 0.34x to 0.40x on the three models
read. `smoke-cpu-kernels.sh` written; the closing gate drafted under `dist/cpu-kernels-close/`, not yet READY.
2026-10-14: gate run 1 (owner, pinned) met every throughput, kernel and LoRA bar on the final numbers' terms but missed
allocation; the miss traced to boxed vectors in the prefill tile's leftover rows (fixed, same bits) and to a metric that
charged warmup to the measured tokens. 2026-10-26: owner decisions 6 to 8 (allocation counted inside the recording; Q4_K
1.102 met within resolution; re-run Parts S and C); the extractor and harness carry the new field, the gate's run 2 is
prepared; pre-gate checks in progress, then READY.*
Gap analysis refs: §1.7 (the CPU half, continued from [Tier 02C](TIER-02C-cpu-hot-path.md))

**Added 2026-10-11 (plan review).** The only active end-of-plan row is CPU tg (>= 0.25x on every sweep
model, 0.215x today, Qwen2.5-3B binding), and the README named its levers, from [Tier 02C](TIER-02C-cpu-hot-path.md)'s closing record, while saying no tier owned them:
"A faster Q6_K decode kernel and a Q4_K kernel nearer one cycle per weight are the levers; no tier owns them
yet." Every tier after Tier 02D in the running order is feature breadth that cannot move a CPU number, so
without this tier the plan reaches Tier 14 at about 0.215x with no work aimed at its own largest gap. This
tier owns that work.

## Objective

Cut the instruction cost per weight of the two kernels that are about 93% of CPU decode, make the CPU
integer kernels use the host's integer SIMD width instead of a fixed 128 bits, and give CPU prefill a
measured length profile and an end-of-plan target of its own.

## Why this tier, why now

**The binding term is named and measured.** Tier 02C's closing sweep (`docs/perf-compare/20261011T013059Z/`,
pinned) put Juno at 12.2% to 13.3% of the host's attainable read bandwidth on CPU decode. The shipped Q4_K
vector kernel (`CpuIntegerKernels.q4kRow`) runs at about 1.5 core cycles per weight on one thread and is about
57% of Mistral 7B decode; the Q6_K kernel (`CpuIntegerKernels.q6kRow`) is a **scalar** integer loop at about
2.7 cycles per weight and about 36% of decode. *Corrected 2026-10-12 (step 3 preparation): no published reading
supports 2.7. The single-thread readings of the shipped Q6_K kernel at Mistral 7B's FFN shape are 84.8 ms per call
(`docs/perf-compare/20261010T085329Z-cpu-integer-kernels/kernel-variants/decode-final-threads1.md`) and 86.4 ms
(unpinned re-check, 2026-10-12), about 5.1 cycles per weight at 3.5 GHz; the Q4_K figure (25.1 ms, about 1.5) is
right. Step 3's pinned reading replaces both.* The README's restatement says the row returns toward 0.40x only
if these two costs fall; nothing else in the CPU decode profile is larger than a few percent.

**The SIMD width is fixed in code.** `CpuIntegerKernels` (`B`, `S`, `F` at lines 52 to 54) and
`CpuIntegerGemm` (lines 50 to 51) declare `ByteVector.SPECIES_128`, `ShortVector.SPECIES_128` and
`FloatVector.SPECIES_128`. That was the right call on this host, which has AVX but no AVX2 (Tier 02C item 1's
correction), but it means an AVX2 or AVX-512 host runs Juno at 128 bits while the reference tool doubles or
quadruples its integer width. Every CPU ratio this plan publishes is then specific to this one host, and
moving to a newer one widens the gap instead of closing it.

**CPU prefill has no target and no length profile.** The CPU pp milestone (`>= 0.10x` at `n_prompt=128`) was
retired at 0.224x, and no end-of-plan row replaced it. Both pinned CPU sweeps on record
(`20261001T180241Z`, `20261011T013059Z`) are at `n_prompt=128`. GPU prefill had exactly this blind spot until
Tier 01B found it collapsing with length (0.011x at 512 on Phi-3.5-mini against 0.040x at 128).

**Sequenced here** because CPU is the largest gap in the program, it is ready now, and Tier 04's mapped
weight loading (item 0 there) threads a `MemorySegment` accessor through these same kernels: it should
convert the kernels this tier leaves, not the ones it replaces. The same argument put Tier 02C before
Tier 04.

## Scope

### In scope

1. **Q6_K vector integer decode kernel.** Replace the scalar loop in `CpuIntegerKernels.q6kRow` with a
   Vector API kernel in the shape of `q4kRow`: unpack the 4-bit low and 2-bit high planes into signed 6-bit
   values in vector registers, widen to shorts, multiply against the Q8 activation and add pairs into ints,
   with at least 4 independent accumulators, applying each 16-weight sub-block's scale once. Integer
   accumulation is associative, so the result must equal the scalar kernel's bit for bit as long as the
   float scaling order is kept. Give `CpuIntegerGemm`'s Q6_K prefill path the same vector inner loop.
   *Corrected 2026-10-12 (item 1): `CpuIntegerGemm` already runs Q6_K through its shared vector integer `tile`;
   only its per-type weight `unpack` is scalar (about 5.5% of 2048-token prefill samples for every type
   together). The prefill half of this item is therefore that unpack, a measured candidate, not a new kernel.*
2. **Q4_K decode kernel toward one cycle per weight.** Measure where `q4kRow`'s 1.5 cycles per weight go
   (nibble unpack, widening, the pairwise add, the per-sub-block scale and minimum, the horizontal sum per
   row) with a single-thread microbenchmark at Mistral 7B's FFN shape, and remove the largest. Candidates,
   each measured before adoption: processing two rows per pass so each activation vector load is reused;
   deferring the horizontal sum to once per row instead of once per block; folding the minimum term into a
   precomputed per-block activation sum (the reference tool's `bsums`). Adopt on measurement only.
3. **Width-generic integer species.** Choose the byte, short, int and float species once at class
   initialisation from the host's preferred integer width (128 or 256 bits; 512 only if measured faster than
   256 on such a host), keep a forced-width hook so tests run the 128-bit path anywhere, and keep the 128-bit
   path as the parity oracle. On this host the selection must resolve to 128 bits and change nothing.
   Record the selected width at startup next to the existing `SIMD available` line.
4. **CPU prefill length profile and target.** Take a pinned CPU sweep at `n_prompt=512` and `2048` on the
   four sweep models before any change in this tier, publish it, and decompose where 2048-token CPU prefill
   time goes (`CpuIntegerGemm` by weight type, attention, everything else) with `juno-perf.jfc` and
   `JfrSpanBreakdown`. *Corrected 2026-10-12 (step 2): "by weight type" cannot be read from method sampling,
   because `CpuIntegerGemm.tile` is one loop for every type and only `unpack` (about 5.5% of samples) branches
   on it; the per-type cost is measured directly by step 3's kernel baseline instead.* If CPU attention is the term that grows with length, name it in this tier's record
   and say which tier owns it; do not leave it implicit.
5. **Restate the CPU rows from what this tier measures.** After items 1 and 2, restate the README's CPU tg
   row from the measured cycles per weight, as Tier 02C item 5 did, never below 0.25x, and the CPU pp row
   from item 4's sweep, never below 0.25x.
6. **Host prefill attention on the worker pool** *(added 2026-10-12 by owner decision 1, from step 2's
   decomposition)*. Every handler's CPU prefill window attends row by row on the calling thread
   (`gqaInto` to `GqaMath.attend`, scalar float, every head in turn) while the other `--threads` pool threads
   park; at 2048 tokens this is at least 48% to 65% of CPU prefill time on the sweep models, and 14% to 25%
   at 512 ([`20261012T205149Z-cpu-prefill-length`](../perf-compare/20261012T205149Z-cpu-prefill-length/INDEX.md)).
   Add one window-level entry point beside `GqaMath.attend` that spreads the window's (row, query head) pairs
   over `CpuThreads` with per-worker score scratch, and route every handler's host window loop through it
   (every `GqaMath.attend` caller that loops over a window: the LLaMA-family, Phi-2, Phi-3, Qwen3 and
   Qwen3-MoE handlers, and the LoRA trainable handlers' forward). The split alone keeps each
   (row, head) computed exactly as today, so it is bit-identical; vectorizing the dot and accumulate loops is a
   second, separately measured candidate held to the `GqaMath` oracle bound. Decode attention at depth may take
   the same entry point if measured faster; it is not required.
7. **Prefill matmul tile** *(added 2026-10-12 by owner decision 3)*. After items 1, 2 and 6 the CPU pp ratio at
   512 reads about 0.225x (unpinned) against `>= 0.30x`; the remaining cost is `CpuIntegerGemm`'s shared tile.
   Decompose where the tile's time goes at the prefill window width (32), then measure candidates one at a time
   (float-lane accumulation as adopted in `q4kRow`, tile shape, unpack cost), adopting on measurement only, each
   bit-identical to the current gemm or within the stated float-rounding bound.

### Out of scope

- Q8_0, Q5_K and the float types' kernels, beyond keeping them correct: none is more than a few percent of
  sweep-model decode. Q8_0's scalar kernel is recorded as the next candidate if `Meta-Llama-3.2-1B` Q8_0
  matters to a later tier.
- Mapped weight loading (Tier 04 item 0), which runs after this tier against these kernels.
- GPU kernels of any kind (Tier 02D decision 4 leaves the GPU K-quant GEMV with the owner).
- NUMA placement and thread affinity: this host is a single socket.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | primary target; correct against the float-activation oracle within the quality threshold, and bit-identical to the current scalar Q6_K integer kernel |
| 2 | CUDA GPU inference | must remain unaffected; partial-offload CPU layers and the CPU LM-head fallback use these kernels, held by the standing gate's GPU lane |
| 3 | ROCm GPU inference | N/A for the kernels; ROCm attention runs on the CPU and is unaffected. NEEDS-AMD-HARDWARE for end to end, as before |
| 4 | Static schedule | the kernels must work at every `--parallel` width |
| 5 | Continuous schedule | same, under mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | each node JVM selects its own species width; verify a 3-node CPU cluster serves a request |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | the frozen forward's matmuls use the float-activation kernels (`CLAUDE.md`), which items 1 to 3 do not touch; verify it stays so. Item 6 (2026-10-12): the trainable handlers attend through `GqaMath` too, so their forward takes the pooled window attention, bit-identical, and the LoRA train gate below applies |
| 10 | LoRA playback | CPU playback composes its delta against the new kernels; parity against the float path within the quality threshold |
| 11 | Vision | `VisionEncoder` runs on `CpuMatVec.INSTANCE`; the B=741 width must not regress, `compare-vision.sh` required |
| 12 | OpenAI REST surface | correctness via chat completions on CPU |
| 13 | Native REST surface | same |
| 14 | CLI | no new flag; the selected species width is printed at startup |
| 15 | JVM embedding facade | N/A: no new capability; the width is selected per JVM at class initialisation, the same in an embedder |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Item 4's pinned CPU sweeps at 512 and 2048, published before any kernel change, and the README's CPU pp
   row's reference cell filled from them.
3. Single-thread cycles-per-weight microbenchmark of the shipped `q4kRow` and `q6kRow` at Mistral 7B's FFN
   shape, pinned, median of three, published as this tier's kernel baseline.
4. Item 1 (Q6_K), re-measured; then item 2 (Q4_K), one candidate at a time, re-measured after each; then
   item 6 (prefill attention on the pool), the thread split first and vectorization second, each re-measured
   at 512 and 2048; then item 7 (prefill matmul tile), decomposition first, one candidate at a time.
5. Item 3 (species width), with the forced-width tests.
6. Item 5's restatement, then the closing sweeps at 128, 512 and 2048, pinned.
7. Re-verify every earlier tier's row-1 correctness result under the new kernels (greedy text on the CPU
   smokes, `ModelLiveRunnerIT`'s CPU checks), and document any change.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **Q6_K kernel parity**: the vector `q6kRow` bit-identical to the scalar integer kernel it replaces on
  random and real-weight rows, at B = 1, 8, 128 and 741, including a single super-block row; the same for
  `CpuIntegerGemm`'s Q6_K path.
- **Q4_K kernel parity**: each adopted item 2 candidate bit-identical to the current `q4kRow`, or, where it
  reorders a float step, within the model-quality threshold and stated as such.
- **Prefill attention parity** (item 6): the pooled window attention bit-identical to the row-by-row
  `GqaMath.attend` loop at window widths 1, 7, 32 and 128, start positions 0, 500 and 2016, every sweep
  model's head shape, with and without a sliding window, at `--threads` 1, 2 and 12; a vectorized variant
  within `GqaMathOracleTest`'s bound instead, stated as such.
- **Species width**: under the forced-width hook, the 128-bit and 256-bit paths produce identical integer
  sums and outputs within float tolerance on every format; on this host the default selection is 128 bits.
- **Model quality**: `ModelQualityNllLiveTest` (Tier 02C) re-run on every sweep model.
- **`ModelLiveRunnerIT`**: the existing CPU-hot-path check, unchanged, must pass.
- **New bash smoke script**: `scripts/performance-tests/smoke-cpu-kernels.sh` — CPU decode and a 512-token
  CPU prefill on the four sweep models at two `--threads` values, greedy text compared against the pre-tier
  build, plus the vision smoke case at B=741.
- **Standing throughput and allocation gate** (README, "Test infrastructure"): run against the pre-tier
  jar; for this tier its CPU lane is superseded by the stricter primary gate below, and its GPU lane and
  allocation half apply as written.
- **Perf gate (required)**: the kernel microbenchmark before and after, `compare-vision.sh`,
  `compare-lora.sh`, and `compare-llama-cpp.sh --cpu --pin-clocks` on the four sweep models at 128, 512
  and 2048 at matched threads; publish under `docs/perf-compare/` with descriptive directory names.

  **Threshold.**
  - CPU tg ratio **>= 0.25x** on every sweep model (the README's end-of-plan row; reference 0.215x,
    Qwen2.5-3B binding).
  - CPU pp ratio at `n_prompt=512` **>= 0.30x** on every sweep model (the README's new end-of-plan row,
    reference set by step 2; restated by item 5 if step 2 shows it is already met or out of reach, never
    below 0.25x).
  - Kernel cost at Mistral 7B's FFN shape, pinned, on one thread **and** per core at 12 threads: Q6_K **<= 2.7**
    cycles per weight, Q4_K **<= 1.1** cycles per weight. Baseline 5.41 and 1.52 on one thread, 5.24 and 1.52 per
    core at 12 ([`20261012T225531Z-cpu-kernel-baseline`](../perf-compare/20261012T225531Z-cpu-kernel-baseline/INDEX.md)).
    *Restated 2026-10-12 by owner decision 2: the Q6_K bound was `<= 1.5`, written against a believed 2.7 today
    (about 1.8x); against the measured 5.41 it asked 3.6x. 2.7 keeps the 2.0x the tier intended. Both widths are
    scored because the earlier Q6_K vector form is faster on one thread and 42% slower at 12 on this host.*
  - Prefill attention (item 6): Juno CPU pp t/s at `n_prompt=2048` **>= 1.50x** the pre-tier build on every
    sweep model (same-hour A/B, pinned, pre-tier jar `b9a0440b3a1b597a`), and at `n_prompt=512` **>= 1.10x**.
    *Restated 2026-10-13 by owner decision: the 2048 bar is read from the pinned closing sweeps (median of three)
    against the step 2 sweeps of the same pre-tier jar (`20261012T071213Z`, `20261012T162344Z`, pinned, median of
    three); a 1.50x bar sits far outside the 15% drift floor the same-hour rule guards. The 512 bar stays a same-hour
    pinned A/B.*
  - Model quality: mean NLL **<= 1.01x** the float-activation kernel's on every sweep model.
  - Juno CPU tg and pp t/s **>= 1.10x** the pre-tier build on Mistral 7B and Qwen2.5-3B (same-hour A/B,
    pinned) and **>= 0.95x** on the other two, so a kernel that wins on one shape and loses on another
    does not close the tier.
  - Vision: `latency_ms` **<= 1.25x** baseline, decode tps **>= 0.80x** baseline. LoRA: train **>= 0.95x**,
    wall-clock playback tps **>= 0.80x**.
  - On an AVX2 host, if one is obtained: the 256-bit Q4_K kernel **>= 1.5x** the 128-bit kernel's
    single-thread throughput. Without one, the row is marked `NEEDS-AVX2-HARDWARE` and the forced-width
    parity tests are the evidence; flag this to the user when step 5 is reached (see
    [`INVENTORY.md`](INVENTORY.md)). *2026-10-13 (step 5): `NEEDS-AVX2-HARDWARE`; flagged to the owner; the
    forced-width parity tests (`CpuIntegerWidthParityTest`) pass.*
  - Every throughput number is a median of at least three runs with min/max published.

## Models needed

The four sweep models (Q4_K_M; Q6_K tensors occur in every one) and `moondream2-q5_k.llamafile` for the
vision guard. All present. An AVX2 or AVX-512 host for the width threshold is not present.

## Exit criteria

- [x] Pinned CPU sweeps at `n_prompt=512` and `2048` published before any kernel change, with the 2048
      decomposition, and the README's CPU pp row's reference cell filled from them.
      *2026-10-12:* [`20261012T045931Z`](../perf-compare/20261012T045931Z/INDEX.md) (512),
      [`20261012T071213Z`](../perf-compare/20261012T071213Z/INDEX.md) and
      [`20261012T162344Z`](../perf-compare/20261012T162344Z/INDEX.md) (2048), decomposition
      [`20261012T205149Z-cpu-prefill-length`](../perf-compare/20261012T205149Z-cpu-prefill-length/INDEX.md);
      README CPU pp row reference 0.165x (TinyLlama binding).
- [x] Kernel baseline published (cycles per weight, Q4_K and Q6_K, single thread, pinned) before item 1.
      *2026-10-12:* [`20261012T225531Z-cpu-kernel-baseline`](../perf-compare/20261012T225531Z-cpu-kernel-baseline/INDEX.md):
      Q4_K 1.52, Q6_K 5.41 cycles per weight on one thread (1.52 and 5.24 per core at 12).
- [x] Q6_K runs a vector integer kernel at decode and prefill width, bit-identical to the scalar kernel it
      replaced.
      *2026-10-27:* decode: `CpuIntegerKernels.q6kRow` in 128-bit lanes, raw float bits equal to a frozen copy of the
      scalar loop (`CpuQ6kKernelParityTest`, 11 tests, B = 1 to 741, real TinyLlama rows). Prefill: `CpuIntegerGemm`'s
      shared vector tile with the Q6_K unpack in vector lanes, exact against the scalar unpack
      (`CpuIntegerGemmUnpackParityTest`); the tile's G2 candidate reorders the float offset term for every format by
      design, held to `CpuIntegerGemmTest`'s exact-reference bound and to the NLL threshold (item 1's correction: the
      prefill half is the unpack). Pinned: 2.13 cycles per weight on one thread, 2.49 per core at 12 (was 5.41 and 5.24).
- [x] Item 2's candidates measured one at a time and adopted on measurement, each recorded.
      *2026-10-13:* candidates 1 to 5, each measured alone and recorded in the execution record (3 and 5 adopted,
      1, 2 and 4 rejected). This box records the process only; the Q4_K kernel bound is scored by the pinned kernel
      microbenchmark in the closing gate, under the threshold criterion below, owed to the owner.
- [x] Item 7's tile decomposition published and its candidates measured one at a time, each recorded; the CPU pp
      row at 512 scored on the pinned closing sweep.
      *2026-10-27:* the decomposition is in this record (2026-10-12, item 7: unpack about 8% of gemm samples in the
      published 2048-token recordings, the multiply loop about 92%, the scalar offset term 17% of Q4_K and 29% of Q6_K
      gemm time; an unpinned probe). Candidates G1 to G4 measured one at a time and recorded (all adopted; G2 within float
      rounding, stated), then the padded tile after gate run 1 (same output bits). CPU pp at 512 on the pinned closing
      sweep: 0.422x to 0.512x against 0.30x ([`20261027T161927Z`](../perf-compare/20261027T161927Z/INDEX.md)).
- [x] Host prefill attention runs on the worker pool in every handler with a host window loop, bit-identical
      to the row-by-row loop (or within the oracle bound where vectorized, stated), and its threshold above met.
      *2026-10-27:* `GqaMath.attendPooled` in the LLaMA-family, Phi-3, Qwen3 and Qwen3-MoE window loops and in `gqa` of the
      Phi-2 and LoRA trainable handlers; the split bit-identical (`GqaMathPooledTest`), the vector head `GqaVectorHead`
      (candidates A1 and A2) within `GqaMathOracleTest`'s bound (`GqaVectorHeadTest`), stated. Threshold: pp at 2048 3.27x
      to 4.38x the pre-tier build (Part S against the step 2 sweeps, >= 1.50x;
      [`20261027T172646Z`](../perf-compare/20261027T172646Z/INDEX.md), [`20261027T202604Z`](../perf-compare/20261027T202604Z/INDEX.md))
      and at 512 2.252x to 2.684x same hour (Part C, >= 1.10x;
      [`20261027T154748Z-cpu-kernels-close-gate-2`](../perf-compare/20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md)): met.
- [x] Integer species chosen from the host's preferred width, 128 bits on this host, forced-width parity
      tests passing; the selected width is printed at startup.
      *2026-10-13:* `CpuIntegerWidth`, `CpuIntegerKernels256`, `CpuIntegerGemm256`; `CpuIntegerWidthTest` and
      `CpuIntegerWidthParityTest` pass; startup line `CPU integer kernels: 128-bit lanes (host preferred 128-bit)`.
      The 256-bit throughput row is `NEEDS-AVX2-HARDWARE`. Owner decision 4 (2026-10-13): `auto` resolves to 128
      on every host until the 256-bit kernels are measured on an AVX2 host; 256 is an explicit opt-in.
- [x] The README's CPU tg and CPU pp rows restated from this tier's measurements, never below 0.25x.
      *2026-10-27:* **Owner decision 9**: both rows kept at the values written (0.25x and 0.30x) and retired as met by
      this tier's closing sweeps (tg 0.259x, Phi-3.5-mini binding; pp at 512 0.422x, Qwen2.5-3B binding), not raised;
      the reference column moved to those sweeps; `check-plan-thresholds.sh` passes.
- [x] Every threshold above met on the pinned closing sweeps. A miss keeps the tier open; closing on a miss
      takes an explicit **Owner decision** recorded here with the measured number and the term that held it.
      *2026-10-26:* **Owner decision 7**: Q4_K at 12 threads, 1.102 cycles per weight per core against `<= 1.1` (Part K of
      run 1, [`20261013T230239Z-cpu-kernels-close-gate`](../perf-compare/20261013T230239Z-cpu-kernels-close-gate/INDEX.md)),
      is met within measurement resolution; the other Part K bounds were met. Owed: the owner's gate run 2 (Parts S and C,
      owner decision 8) on the final build, then the scoring here.
      *2026-10-27:* gate run 2 met every bar; every threshold is scored in the 2026-10-27 record (the width row
      `NEEDS-AVX2-HARDWARE`, as recorded at step 5),
      [`20261027T154748Z-cpu-kernels-close-gate-2`](../perf-compare/20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md).
- [x] Standing throughput and allocation gate met (README, "Test infrastructure"): GPU tg and pp >= 0.95x
      the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
      *2026-10-26:* allocation is read inside the recording (**Owner decision 6**). GPU lane met in run 1 (tg and pp
      0.969x to 1.013x; allocation 0.930x to 0.999x on the new figure,
      [`20261026T204605Z-cpu-kernels-allocation-reread`](../perf-compare/20261026T204605Z-cpu-kernels-allocation-reread/INDEX.md)).
      Owed: the CPU lane's allocation half on gate run 2.
      *2026-10-27:* CPU lane on gate run 2: allocation per token in the recording 0.981x to 1.013x, in-span GC 0 ms on
      every run of both builds: met.
- [x] Every earlier tier's CPU row-1 correctness result re-verified under the new kernels.
      *2026-10-26, final build `4e480ab9131887eb`, all but one re-verified:* `ModelLiveRunnerIT` checks 1 to 8 on the CPU backend, five models, all
      PASS; `smoke-cpu-kernels.sh` 18 of 18 greedy replies identical to the pre-tier build (four sweep models at 4 and 12
      threads, short and 512-token prompts, CPU vision at the 741 width); `smoke-consistency.sh` CPU half,
      `smoke-grammar.sh`, `smoke-tools.sh` and `smoke-long-prompt-prefill.sh --cpu` all pass; NLL 0.99935x to 1.00209x
      ([`20261027T025856Z-cpu-kernels-quality`](../perf-compare/20261027T025856Z-cpu-kernels-quality/INDEX.md)).
      Not re-verified on the CPU backend: context shifting. `smoke-context-policy.sh --cpu` was not run (Tier 02C stopped
      its 4,096-token CPU run after 76 minutes), and `ModelLiveRunnerIT`'s context-shift check 10 took CUDA on this host.
      Owed (executor, after the gate): a CPU context-shift check on TinyLlama, then tick.
      *2026-10-27:* check 10 on the CPU backend, TinyLlama at a 192-position limit, both schedules: PASS (200 of 200
      tokens, 3 shifts each, highest position 191), from the candidate build's `juno-master.jar` with CUDA hidden and the
      integer kernels on
      ([`20261027T025856Z-cpu-kernels-quality`](../perf-compare/20261027T025856Z-cpu-kernels-quality/INDEX.md),
      "Context shift on the CPU"). Every CPU row-1 result is re-verified.
- [x] Every conclusion carries a `host-specific` or `expected-general` marker (Tier 02C item 4's rule).
      *2026-10-27:* checked over the whole record; the one unmarked conclusion (item 2 candidate 3's speedup) is marked.
- [x] Cross-surface checklist fully resolved.
      *2026-10-27:* every row PASS, row 3 `NEEDS-AMD-HARDWARE`, row 15 N/A (cross-surface record, 2026-10-27).
- [x] Docs (`docs/howto.md` startup line, `docs/agent-arch.txt`, `docs/performance.md`) updated,
      Juno-native language only.
      *2026-10-27:* `docs/performance.md` "CPU decode and prefill kernels"; `docs/agent-arch.txt` (width, attention head,
      four-row padded tile, the Vector API signature rule); `docs/howto.md` (width property and startup line, since
      2026-10-13; the in-recording allocation field, 2026-10-26).
- [x] `CHANGELOG.md` entry added.
      *2026-10-27:* Session 127.

## Execution record

### 2026-10-11: steps 1 and 2 (plan check; CPU prefill length sweep prepared)

**Step 1.** `scripts/performance-tests/check-plan-thresholds.sh` passes (23 tier files, milestone and end-of-plan
tables checked).

**Plan against code, re-verified at HEAD `01027ee9`.** `CpuIntegerKernels` declares `ByteVector.SPECIES_128`,
`ShortVector.SPECIES_128` and `FloatVector.SPECIES_128` at lines 52 to 54 and `CpuIntegerGemm` the short and float
128-bit species at lines 50 to 51, as stated. `CpuIntegerKernels.q6kRow` (line 320) is a scalar integer loop with
four `int` sub-sums per 16-weight sub-block; `q4kRow` is at line 206. The startup `SIMD available` line is built in
`VectorQuantKernels` (line 244). The four sweep models and `moondream2-q5_k.llamafile` are present. No drift.

**Step 2, prepared; the pinned run is the owner's.** One build is measured (HEAD `01027ee9`, no engine change in
this tier yet), jar `b9a0440b3a1b597a`, in `dist/cpu-prefill-length/` with `run-gate.sh` and `wait-and-run.sh`.
Method: `compare-llama-cpp.sh --cpu --pin-clocks --n-gen 64 --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2
--jfr 120m --publish`, the Tier 02C closing sweep's flags at the new lengths: 512 on all four sweep models at
their fixed heaps; 2048 on TinyLlama, Qwen2.5-3B and Mistral 7B at their fixed heaps, and on Phi-3.5-mini in its
own sweep at `COMPARE_HEAP=8g` (its 32 KV heads put float KV at about 3.2 GB at 2048 + 1 positions; the GPU 2048
references were taken the same way, Tier 01C decision 5 (a)). JFR runs for up to 120 minutes per recording
because a 2048-token CPU prefill on Mistral 7B is expected to take 15 to 25 minutes, too close to the harness's
30-minute default. Part P then runs `JfrSpanBreakdown` over every measured 2048 prefill recording
(`juno-perf.jfc`, the harness default) for the decomposition. Arguments checked with a harness dry run; the
breakdown tool checked against an existing recording.

**Step 2, run and published (2026-10-12).** The owner's pinned run completed (all three sweeps
`clock_pinned: true`, every row scorable, every prefill exactly 512 or 2048 tokens, Juno spread under 2%).
Published: [`20261012T045931Z`](../perf-compare/20261012T045931Z/INDEX.md) (512),
[`20261012T071213Z`](../perf-compare/20261012T071213Z/INDEX.md) and
[`20261012T162344Z`](../perf-compare/20261012T162344Z/INDEX.md) (2048), local paths scrubbed; the length
profile and decomposition in
[`20261012T205149Z-cpu-prefill-length`](../perf-compare/20261012T205149Z-cpu-prefill-length/INDEX.md), from
`JfrSpanBreakdown` over the measured recordings (`juno-perf.jfc`). The README's CPU pp row's reference cell is
filled (0.165x, TinyLlama binding); the row stays `active` against `>= 0.30x`.

| Model | pp @128 | pp @512 | pp @2048 | ms/token @128 / @512 / @2048 | 2048 cost above the 128 cost | Attention samples @512 / @2048 |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 0.224x | 0.165x | 0.086x | 60.5 / 80.8 / 164.0 | 63% | 12.3% / 37.3% |
| Qwen2.5-3B | 0.230x | 0.186x | 0.122x | 161.9 / 194.6 / 314.2 | 48% | 7.1% / 23.4% |
| Phi-3.5-mini | 0.256x | 0.198x | 0.092x | 206.1 / 262.7 / 595.8 | 65% | 10.3% / 39.8% |
| Mistral 7B | 0.226x | 0.192x | 0.117x | 381.7 / 445.4 / 772.8 | 51% | 6.2% / 26.6% |

Findings:

- **CPU attention is the term that grows with length** (`expected-general`). Every host prefill window attends
  row by row on the calling thread (`gqaInto` to `GqaMath.attend`, scalar float) while the other pool threads
  park: average parked pool threads rise from about 4 to 4.7 at 512 to 7.6 to 9.9 at 2048. Both lengths run
  32-token windows, so matmul work per token is fixed and the cost above the 128-token cost is a lower bound on
  attention's share of wall time: at least 48% to 65% at 2048, 14% to 25% at 512 (magnitudes `host-specific`).
  Sample shares understate it, because a one-thread phase yields fewer execution samples per second.
- **The matmul kernels are still most of the cost at 512** (81% to 89% of samples), so items 1 to 3 still
  apply; but the CPU pp row at 512 (0.165x against 0.30x) needs about 1.8x on TinyLlama, which kernel work
  alone is unlikely to reach while a quarter of its 512-token prefill is one-thread attention.
- The reference tool loses 6% or less of its prompt throughput from 512 to 2048; Juno loses 42% to 56%.
- The generation columns of these sweeps are not a reference (generation does not depend on `n_prompt`; the
  TinyLlama 512 sweep's reference-tool generation reading came out low, tg ratio 0.379x against Juno's unchanged
  7.81 t/s). The CPU tg reference stays `20261011T013059Z`.
- GC is not a factor: 0 to 9 ms of pauses in prefill spans of 42 s to 26 minutes.

**Owner decision 1 (2026-10-12): host prefill attention joins this tier as item 6.** Raised with the owner with
three choices (this tier; a new tier after it; defer to a later tier and restate the pp row down). The owner chose
this tier, the recommendation: the pp row at 512 is this tier's threshold and is unlikely to be met without it.
Item 6, its parity test, its threshold (CPU pp t/s at 2048 >= 1.50x and at 512 >= 1.10x the pre-tier build on
every sweep model) and its exit criterion are added above; cross-surface row 9 now notes that the LoRA trainable
handlers attend through `GqaMath` and take the change.

**Step 2's weight-type split, corrected.** `CpuIntegerGemm` is one figure in the sampling (`tile` is shared by
every type; `unpack`, the only type-specific method, is about 5.5% of samples), so the split by weight type is
not attributable from these recordings. Item 4's text is corrected (dated) to take the per-type cost from step 3's
kernel baseline, which measures it directly. No tooling change was made.

**Next: step 3**, the single-thread cycles-per-weight baseline of `q4kRow` and `q6kRow` at Mistral 7B's FFN
shape, pinned (the owner's run), before item 1.

### 2026-10-12: step 3 prepared (kernel baseline)

**Method.** `scripts/performance-tests/cpu-decode-kernel-variants.sh` already times the shipped kernels (its
`production` variant calls `LlamaTransformerHandler.matVec`, which takes `CpuIntegerKernels` when the Vector API
module is present, as the script passes it) at Mistral 7B's FFN shapes, Q4_K 14336x4096 and Q6_K 4096x14336,
single-threaded and at 12 threads, with weights cycled through 96 MB or more so every call reads DRAM. No new
harness was written. Cycles per weight = median ms per call x 3.5 GHz / 58,720,256 weights; pinning (governor
`performance`, turbo off) holds every core at the E5-1650 v2's 3.50 GHz base clock, which the script records.

**Unpinned re-check (2026-10-12, scratch, not published).** Q4_K 25.06 ms per call single-threaded, about 1.5
cycles per weight; Q6_K 86.40 ms, about **5.1** (12 threads: 4.32 and 15.71 ms). The tier's "about 2.7" for
Q6_K had no source; the 2026-10-10 published reading (84.8 ms) agrees with about 5.1. Corrected above and in the
README (dated). Consequence: the Q6_K bound of `<= 1.5` cycles per weight asks for about 3.4x, where the tier
was written expecting about 1.8x. It is kept until the pinned baseline exists and will then be raised with the
owner, with the option of anchoring it to measured cycles instead.

**Owed to the owner:** `bash dist/cpu-kernel-baseline/wait-and-run.sh` (about 15 minutes, pinned; three runs,
READY carries the class-file hash `1d8e89c50bfa406d`). Then: publish the baseline under `docs/perf-compare/`,
tick the kernel-baseline exit criterion, and start item 1 (vector Q6_K kernel) tests first.

### 2026-10-12: step 3 done (kernel baseline, pinned)

Owner run, governor `performance`, turbo off, mean core clock 3493 to 3495 MHz; class-file hash `1d8e89c50bfa406d`
(HEAD `01027ee9`). Published:
[`20261012T225531Z-cpu-kernel-baseline`](../perf-compare/20261012T225531Z-cpu-kernel-baseline/INDEX.md).

| Kernel | 1 thread ms (cycles/weight) | 12 threads ms (cycles/weight per core) |
|---|---|---|
| Q4_K shipped vector | 25.42 (**1.52**) | 4.25 (1.52) |
| Q4_K earlier vector form `d2` | 22.85 (1.36) | 4.21 (1.51) |
| Q6_K shipped scalar | 90.75 (**5.41**) | 14.65 (5.24) |
| Q6_K earlier vector form `d2` | 78.03 (4.65) | 20.75 (7.42) |

- The Q4_K figure (about 1.5) stands. Q6_K is 5.41, not about 2.7, as the unpinned re-check said. `host-specific`.
- **A single-thread win can be a 12-thread loss here** (`host-specific`: two hyperthreads share each core's vector
  execution units and caches). The earlier Q6_K vector form is 14% faster on one thread and 42% slower at 12. Item 1
  is read at both. *Corrected the same day: this entry first said the hyperthreads share the vector registers; each
  keeps its own. Item 1 below found that form's likely cause (vectors boxed at helper calls).*

**Owner decision 2 (2026-10-12): the kernel bounds are scored at both widths, and Q6_K's is restated to `<= 2.7`
cycles per weight** (Q4_K stays `<= 1.1`). Raised with three choices: `<= 2.7` at both widths (recommended), keep
`<= 1.5` (3.6x), or relative speedups (Q6_K >= 2.0x, Q4_K >= 1.35x). By arithmetic, halving Q6_K alone (about 36%
of Mistral 7B decode) is about 1.2x on decode, which lifts the binding CPU tg ratio (0.215x) to about the 0.25x row.

**Next: item 1**, the vector Q6_K kernel: parity tests first (bit-identical to the scalar `q6kRow` at B = 1, 8, 128
and 741, and for `CpuIntegerGemm`'s Q6_K path), then candidates read at 1 and 12 threads.

### 2026-10-12: item 1, vector Q6_K decode kernel

**Tests first.** New `node/src/test/java/cab/ml/juno/node/CpuQ6kKernelParityTest.java` (11 tests): `q6kRow`
against a verbatim frozen copy of the scalar loop, compared as raw float bits, on random rows at 256 (one
super-block), 4096 and 14336 columns and a one-row matrix; extreme blocks (every weight 63 and every weight 0,
activations at +127, -127 and alternating, the 16-bit lane bound); the multi-threaded `matVecInto` dispatch at
741 rows; real TinyLlama Q6_K rows (four tensors, first 256 rows each, runs when the model is present); and
`CpuIntegerGemm`'s Q6_K output at B = 1, 8, 128 and 741 against golden hashes of its output bits captured from
the pre-change build. On the pre-change build all 11 passed (the oracle is that build's own loop), so they
are regression tests; a mutation check (one `qh` shift changed from 6 to 7) failed 6 of 11 with wrong row values,
as it must (the golden gemm hashes and the all-ones block do not exercise that shift), and the file was restored.

**Implementation.** `CpuIntegerKernels.q6kRow` now calls `q6kPass` per 64-weight pass: the pass's `ql`, `ql + 32`
and `qh` bytes are loaded once as eight 16-bit lanes each, and its four sub-block sums are formed one at a time
(even and odd unsigned 6-bit weights times the even and odd activation bytes, the `q4kRow` byte-pair method),
widened to int and reduced; the offset of 32 comes from the activation's existing `sum16`. The integer sums equal
the scalar loop's and the float expression is unchanged (`acc += d * (...)`), so the output is bit-identical.

**Two dead ends, measured (scratch probe, `q6kRow` alone, `ThreadMXBean` allocated bytes).** Every lane
operation must be written in place. With the widening reduction in a helper method, and then with the even and
odd byte extraction in helpers (`evenBytes`, `oddBytes`), C2 boxed the vectors: 352 MB allocated per Mistral 7B
FFN matrix call, 92 ms single-threaded and 28.3 ms at 12 threads (1.9x slower than the scalar kernel). Moving
the pass into its own method did not change it; writing the extraction in place did: 0 bytes per call. The frozen
Q6_K vector variant Tier 02C rejected (`CpuDecodeIntegerVariants`, "register pressure") calls the same kind of
helpers, so boxing is the likely cause of that verdict too; not verified. `expected-general` (C2 behaviour).

**Result, unpinned (same harness as the baseline; the pinned reading is owed with the closing work).**

| Q6_K 4096x14336 | 1 thread ms | cycles/weight at 3.5 GHz | 12 threads ms | cycles/weight per core | allocated per call |
|---|---|---|---|---|---|
| Baseline, scalar (pinned, `20261012T225531Z-cpu-kernel-baseline`) | 90.75 | 5.41 | 14.65 | 5.24 | 0 |
| Vector `q6kRow` (unpinned probe) | 37.89 | about 2.26 | 6.65 | about 2.38 | 0 |

About 2.4x on one thread and 2.2x at 12; both under the restated `<= 2.7` bound on this unpinned reading.
`host-specific` magnitudes. Tests run after the final change: `CpuQ6kKernelParityTest`, `CpuIntegerKernelsTest`,
`CpuIntegerGemmTest`, all passing.


**End to end, unpinned A/B (2026-10-12, not published; same session, A B A B).** `compare-llama-cpp.sh --cpu
--n-prompt 128 --n-gen 64 --threads 12 --juno-warmup 2 --juno-reps 1 --reps 1 --no-publish`, pre-tier jar
`b9a0440b3a1b597a` against the item 1 jar `6d9782c3249af31a`:

| Model | CPU tg t/s, pre-tier (runs) | CPU tg t/s, item 1 (runs) | CPU pp t/s, pre-tier / item 1 (means) |
|---|---|---|---|
| TinyLlama 1.1B | 8.05 / 7.88 | 10.19 / 9.93 (+26%) | 16.97 / 17.21 |
| Mistral 7B | withheld / 1.29 | 1.74 / 1.69 (+33%) | 2.68 / 2.65 |

The kernel gain reaches decode; prefill is unchanged, as expected (it runs `CpuIntegerGemm`). One pre-tier Mistral
generation reading was withheld by the harness's span check. Against the pinned reference tool speeds of
`20261011T013059Z` these read about 0.31x (TinyLlama) and 0.30x (Mistral 7B) CPU tg; Qwen2.5-3B, which binds the
row, was not in this A/B. Every figure here is indicative; the pinned closing sweep scores the row.

### 2026-10-12: item 2, Q4_K candidates (one at a time; unpinned scratch probe)

Method: a scratch probe (not committed) calls the row kernel alone at Mistral 7B's FFN Q4_K shape (14336x4096,
weights cycled through eight copies), one thread and twelve, and reads time per call, `ThreadMXBean` allocated
bytes and bit differences against the shipped kernel. Cycles per weight at 3.5 GHz (unpinned, so turbo may run
the core faster and these read low by up to about 10%; the pinned baseline is 1.52).

**Where the shipped kernel's cycles go** (diagnostic variants that drop a term and compute a wrong result; one
thread): the vector multiply-accumulate core 0.80 cycles per weight (56%), the per-sub-block scalar scale and
minimum decode and float tail about 0.34 (23%), the per-sub-block horizontal reductions about 0.17 (12%), of
1.44. The README's third candidate, folding the minimum into a precomputed per-block activation sum, was already
in the shipped kernel (`dmin * min * sum32`), so it was not a candidate.

| Candidate | Bit-identical | 1 thread cycles/weight (shipped 1.43 to 1.49) | 12 threads, per core (shipped 1.46 to 1.54) | Allocated | Decision |
|---|---|---|---|---|---|
| 1. Two rows per pass (shared activation loads and lane shifts) | yes | 2.06 | 2.21 | 0 | rejected: 41% slower (twice the live vectors) |
| 2. Scales decoded once per block, passes in a helper method | yes | 1.57 to 1.60 | 1.54 to 1.61 | 0 | rejected: the 12-argument helper was not inlined |
| 2b. Scales decoded once per block, branch-free, into two packed longs, shipped loop shape | yes | 1.39 to 1.41 | 1.37 to 1.49 | 0 | within noise (about 3%); folded into 3 |
| 3. 2b plus sub-block sums scaled in float lanes, one horizontal sum per row | no (float rounding) | **1.25 to 1.28** | 1.23 to 1.58 (median 1.29) | 0 | **adopted** |
| 4. 3 with the 16-to-32-bit widening by two sign-extending conversions and one add instead of shifts | no (float rounding) | 1.26 to 1.30 (adopted 3: 1.27 to 1.30) | 1.28 to 1.40 (adopted 3: 1.30 to 1.32) | 0 | rejected: no difference |

Candidate 3 first allocated 48 bytes per row: the final lane sum went through `sumLanes(FloatVector)` and the
vector was boxed. Written in place, 0 bytes.

**Adopted: candidate 3 in `CpuIntegerKernels.q4kRow`.** Tests first: new
`node/src/test/java/cab/ml/juno/node/CpuQ4kKernelParityTest.java` (7 tests) holds `q4kRow` to a verbatim frozen
copy of the replaced kernel on random rows (256 to 14336 columns), extreme blocks, real TinyLlama Q4_K rows (four
tensors), and checks the same bits on 20,000 successive calls. It passed on the replaced kernel (regression
test). The bound is stated, not bit-identity: **1e-6 of the row's term magnitude** (the sum of absolute sub-block
scale and minimum terms, in double). The first form normalised by the row's result and failed on a one-block row
whose result is a small difference of large terms (3.6e-5 of the result); that is rounding of the terms' size,
so the bound was restated to the terms, not loosened to what passed. Measured worst difference: 1.4e-7 of term
magnitude. A mutation check (the scale decode's top-bits shift changed from 6 to 5) failed 6 of 7 tests,
and the file was restored. `CpuIntegerKernelsTest`'s exact-reference check (1e-5) passes unchanged. Model quality
(`ModelQualityNllLiveTest`) is owed with the closing work.

Production `q4kRow` after adoption (probe): **1.29 to 1.30** cycles per weight on one thread, 1.22 to 1.42
(median 1.31) per core at 12; about 12% faster (magnitude `host-specific`; the direction, one horizontal sum per row
and one scale decode per block, `expected-general`), 0 bytes allocated. **Still above the `<= 1.1` bound**; the
remaining cost is the vector core (about 0.8) plus the per-sub-block widening, conversion and scalar factor
work. Further candidates are owed before item 2 closes. *Assessment, not a measurement: the vector core alone is
about 0.8 cycles per weight, and 128-bit Vector API code on this host has no instruction for a pairwise 16-bit
multiply-add into 32 bits, so the widening stays explicit; `<= 1.1` may not be reachable here, in which case the
miss goes to the owner at the close with these numbers (`host-specific`).*

### 2026-10-12: item 6, host prefill attention on the worker pool (thread split)

**Rows or heads.** Each window row builds its K/V view with `viewForAttention(seqLen, scratch)`, which may write a
dequantised copy into a scratch buffer shared by the window's rows, so rows cannot safely run in parallel without
per-row scratch. The split is therefore over **query heads within each row**: rows stay in order, and each row's
heads (16 to 32 on the sweep models) spread over `CpuThreads`. Each head is the serial loop's body verbatim, with
a per-thread score scratch (grown once, reused), writing only its own slice of the output, so the result is
bit-identical at any thread count. A call from inside a pool task runs inline (the pool's existing rule), and
concurrent callers queue on the pool's lock as matmuls already do.

**Tests first.** New `node/src/test/java/cab/ml/juno/node/GqaMathPooledTest.java` (13 tests): `attendPooled`
bit-identical to `GqaMath.attend` for every sweep model's head shape (TinyLlama 32/4/64, Mistral 7B 32/8/128,
Qwen2.5-3B 16/2/128, Phi-3.5-mini 32/32/96), context lengths 1, 7, 33, 128, 500, 517, 600, 2016 and 2048, sliding
windows 0, 7, 8 and 300, at `--threads` 1, 2 and 12, and from inside a pool task; the output is pre-filled with NaN
so every element must be written. Seen failing against a stub that only zeroes the output (13 of 13, wrong values),
then passing. `GqaMathOracleTest` and `GqaMathWindowTest` pass unchanged.

**Wiring.** Every handler's host window-attention helper calls `GqaMath.attendPooled`: `gqaInto` in the
LLaMA-family, Phi-3, Qwen3 and Qwen3-MoE handlers, and `gqa` in the Phi-2 handler and the LoRA trainable handlers
(`LoraTrainableHandler`, `Phi3LoraTrainableHandler`; their training forward takes it, bit-identical, cross-surface
row 9). Phi-2's and the LoRA handlers' `gqa` also serve their decode step and no longer allocate a score array per
call. The decode-only sites (`LlamaTransformerHandler` decode step, `Phi3TransformerHandler` decode step,
`Qwen3TransformerHandler.gqa`) stay serial: optional under item 6, to be measured. The plan's handler-level test
list (window widths 1, 7, 32, 128 at start positions 0, 500, 2016) is covered at the `GqaMath` level, since each
handler's change replaces one call with a bit-identical one on the same arguments; the end-to-end check is the
smoke script's greedy text against the pre-tier build.

**Module tests after items 1, 2 (candidate 3) and 6 (thread split):** `mvn test -pl node`: 1,129 tests, 0
failures, 0 errors, 47 skipped (GPU-tagged and model-gated groups), BUILD SUCCESS. The other modules are owed with
the closing work.

**End to end after items 1, 2 (candidate 3) and 6, unpinned A/B (2026-10-12, not published, one run per side).**
`compare-llama-cpp.sh --cpu --threads 12 --juno-warmup 1 --juno-reps 1 --reps 1 --n-gen 16 --no-publish`, pre-tier
jar `b9a0440b3a1b597a` against `8736f409971dfeb4`:

| Model, `n_prompt` | CPU pp t/s pre-tier | CPU pp t/s now | pp | CPU tg t/s pre-tier | now | tg |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B, 2048 | 6.40 | 13.07 | **2.04x** | 8.42 | 11.56 | 1.37x |
| TinyLlama 1.1B, 512 | 12.91 | 17.02 | 1.32x | 8.61 | 11.75 | 1.36x |
| Qwen2.5-3B, 512 | 5.36 | 6.22 | 1.16x | 2.82 | 4.16 | 1.47x |

Indicative against the item 6 threshold (`>= 1.50x` at 2048, `>= 1.10x` at 512) and the CPU tg row (Qwen2.5-3B
about 0.35x against the pinned reference tool's 12.05 t/s). **The CPU pp row at 512 is not on track:** about
0.227x (TinyLlama) and 0.225x (Qwen2.5-3B) against `>= 0.30x`. At 512 the remaining prefill cost is the batched
matmul (`CpuIntegerGemm`'s shared tile, 81% to 89% of samples before this tier), which none of items 1 to 6 changes
(item 1's prefill half is only the Q6_K unpack). Raised with the owner, 2026-10-12.

**Owner decision 3 (2026-10-12): the prefill matmul tile joins this tier as item 7.** Raised with three choices:
add an item for `CpuIntegerGemm`'s tile (recommended), restate the CPU pp row to 0.25x now, or defer and take the
miss to an owner decision at close. The owner chose the item; the `>= 0.30x` row stays the threshold.


### 2026-10-12: item 7, prefill matmul tile (unpinned probe, `CpuIntegerGemm.sgemmInto` at B = 32, 12 threads)

Probe (scratch, not committed): the whole `sgemmInto` at Mistral 7B's FFN shapes, B = 32 (the prefill window),
time per call and bytes allocated across all threads. Candidates were tried in place in `CpuIntegerGemm` with the
file backed up, and kept only on measurement.

**Decomposition.** No allocation (0.0 MB per call: unlike the Q6_K decode kernel, the tile's vector helpers are
inlined). From the published 2048-token recordings, the weight `unpack` is about 8% of gemm samples and the tile's
multiply loop about 92%. Inside the tile, a diagnostic that drops the scalar minimum/offset term (wrong results)
showed it at **17% of Q4_K and 29% of Q6_K gemm time**: one scalar multiply-add chain per sub-block per (row,
activation row), and Q6_K has twice the sub-blocks.

| Candidate | Bit-identical | Q4_K 14336x4096 ms/call (before: 65.1) | Q6_K 4096x14336 ms/call (before: 97.9) | Decision |
|---|---|---|---|---|
| G1. Two activation rows per pass, each loaded weight vector used twice | yes (golden hashes held) | 53.6 (-18%) | 97.4 (none) | adopted |
| G2. G1 plus the offset term as four-lane float products against a per-call `scale x sum` per activation sub-block, subtracted from the lane sums | no (float rounding) | **44.6** (-31% cumulative) | **73.1** (-25%) | **adopted** |
| G3. G2 with four activation rows per pass (remainder one at a time; the two-row loop removed) | yes, against G2 (each pair keeps its order) | 42.1 / 44.3 / 45.9 against G2's 44.6 / 50.7 / 47.7, interleaved (median -7%) | 70.5 / 70.4 / 72.0 against 73.1 / 72.0 / 73.0 (median -3%) | **adopted** |

**Correctness.** G1 kept `CpuQ6kKernelParityTest`'s golden gemm hashes. G2 changes float rounding by design, so
those four golden tests failed as they should; their purpose (pinning item 1's unchanged unpack) was served, and
they were removed, since `CpuIntegerGemmTest` already holds the gemm to an exact double reference within 1e-5 at
B = 1, 8, 128 and 741 for Q4_K, Q5_K, Q6_K and Q8_0, and to the decode kernel within 1e-5. That test passes; a
mutation of G2's offset indexing (row b reading row b+1's sums) failed 10 of its 23 cases, and the file was
restored. `VisionWidthCpuKernelGuardTest` and `KQuantGemmQualityTest` pass.

**End to end after G2, unpinned A/B (2026-10-12, not published, one run per side, one warmup).** `n_prompt=512`,
pre-tier jar `b9a0440b3a1b597a` against `bc08e4fb45a6f43a` (items 1, 2, 6 and G2): TinyLlama CPU pp 12.89 to
20.72 t/s, ratio **0.173x to 0.267x**; Qwen2.5-3B 5.32 to 7.81 t/s, ratio **0.187x to 0.276x**; CPU tg 8.29 to 11.84
and 2.81 to 4.12 t/s. G3 adds about 3% to 7% to the gemm on top. The `>= 0.30x` row is closer but not reached on
this unpinned reading; further item 7 candidates are owed (the weight unpack, about 8% of gemm time, is the next
named term).

**Tests after G3:** `CpuIntegerGemmTest` (B = 1, 8, 128 and 741 cover the four-row loop and the one-row
remainder), `CpuQ6kKernelParityTest`, `CpuQ4kKernelParityTest`, `CpuIntegerKernelsTest`,
`VisionWidthCpuKernelGuardTest`, `KQuantGemmQualityTest`: all pass. The full `node` module run (1,129 tests) predates
item 7 and is owed again.

### 2026-10-12: item 7, vectorised weight unpack (unpinned probe); node module re-run

**Module tests after G3.** `mvn test -pl node`: 1,125 tests, 0 failures, 0 errors, 47 skipped, BUILD SUCCESS. The
count is 4 below the earlier 1,129 because the four golden gemm hash tests were removed at G2 (recorded above). The
other modules are still owed with the closing work.

**Candidate G4: the per-block weight `unpack` in vector lanes.** Each format's extraction (Q4_K nibbles, Q5_K nibbles
plus the fifth bit, Q6_K low nibbles plus the two-bit high plane, Q8_0 signed bytes) now loads 16 packed bytes as one
128-bit byte vector, masks and shifts in byte lanes (Q6_K's `((qh >> k) & 3) << 4` as a byte shift masked with
`0x30`), and widens to 16 bits with two `B2S` conversions into the tile scratch. The scale and offset factors are
untouched. This is integer work only, so the 16-bit weights, and with them every gemm output, are unchanged.

Tests first: new `node/src/test/java/cab/ml/juno/node/CpuIntegerGemmUnpackParityTest.java` (8 tests) holds `unpack`
(made package-private for the test, no other change) to a verbatim frozen copy of the scalar unpack, exact on the
16-bit weights and on the raw bits of every scale and offset, for Q8_0, Q4_K, Q5_K and Q6_K, on random blocks and on
every weight byte set to 0x00, 0xFF, 0x80, 0x7F, 0x0F, 0xF0, 0x55 and 0xAA, writing into non-zero tile slots with a
sentinel-filled scratch. It passed 8 of 8 on the scalar unpack (a regression test, not seen failing first). After
adoption, a mutation check (Q6_K's second plane shifted by 3 instead of 2) failed it 2 of 8 and `CpuIntegerGemmTest`
4 of 23, and the file was restored.

Probe: the earlier `GemmProbe` (scratch, not committed), `CpuIntegerGemm.sgemmInto` at Mistral 7B's FFN shapes, B =
32, 12 threads. The candidate class was compiled separately and put ahead of the tree's on the classpath. Six runs per
side, interleaved in two rounds (baseline first, then candidate first). Unpinned:

| Shape | Baseline ms/call (6 runs) | G4 ms/call (6 runs) | Median change | Allocated |
|---|---|---|---|---|
| Q4_K 14336x4096 | 42.79 44.66 41.85 41.91 44.34 55.64 (median about 43.6) | 44.45 39.70 39.93 42.43 39.84 39.86 (median about 39.9) | **about -8%** (5 of 6 pairs faster) | 0 |
| Q6_K 4096x14336 | 71.74 70.71 71.93 71.69 72.39 70.72 (median about 71.7) | 65.94 66.73 68.36 66.24 67.87 66.26 (median about 66.5) | **about -7%** (6 of 6 pairs faster) | 0 |

**Adopted** (`expected-general` direction; magnitudes `host-specific`). That is about the unpack's whole recorded
share (about 8% of gemm time), so the unpack is no longer a named term. Tests after adoption: `CpuIntegerGemmUnpackParityTest`,
`CpuIntegerGemmTest`, `CpuQ6kKernelParityTest`, `CpuQ4kKernelParityTest`, `CpuIntegerKernelsTest`,
`VisionWidthCpuKernelGuardTest`, `KQuantGemmQualityTest`: all pass. The end-to-end effect on the CPU pp row at 512 is
not measured yet. Cumulative item 7 at the probe, from the pre-G1 65.1 and 97.9 ms: Q4_K about -39%, Q6_K about -32%.

**Full unit suite after G4** (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`):
BUILD SUCCESS, 2,372 tests, 0 failures, 0 errors, 52 skipped (node 1,133 including the 8 new; juno-player 399;
coordinator 109), 1 h 05 min. The SEVERE log lines in the run are the loader's negative tests (unsupported
architecture, heap exhaustion, missing file), which pass.

**Next:** item 2's remaining Q4_K candidates or a recorded expected miss against `<= 1.1`, then item 3 (species
width; this host is 128-bit only, so `NEEDS-AVX2-HARDWARE` for the width threshold).

### 2026-10-13: item 2, Q4_K candidate 5 (activation split once per call; unpinned probe)

**Term.** In the adopted candidate 3 kernel, every 16 weight bytes of every row also split 16 activation bytes into
sign-extended even and odd 16-bit lanes (shift left 8 and arithmetic shift right 8 for the even bytes, arithmetic
shift right 8 for the odd, on two activation vectors: 6 of about 20 lane operations per 16 weight bytes). The
activation is the same for every row of the call (14,336 rows at Mistral 7B's FFN up and gate shape), so this work
is repeated once per row for nothing.

**Candidate 5.** `CpuIntegerKernels.Q8Activation` splits its quantized bytes once per call into two `short[]`
arrays (`xe[i] = q[2i]`, `xo[i] = q[2i + 1]`, 8 KB each at 4096 columns, grown and reused with the other buffers),
and `q4kRow` loads its even and odd activation lanes from them directly (signature now takes `xe`, `xo` in place of
the byte array). Every product, integer sum and float step is unchanged, so the result is bit-identical to candidate 3.

Tests first: new `node/src/test/java/cab/ml/juno/node/CpuQ4kActivationSplitParityTest.java` (6 tests) holds `q4kRow`
bit for bit to a verbatim frozen copy of the candidate 3 kernel on random rows (256 to 14336 columns, with outliers),
extreme blocks (every weight 15 or 0, every scale and minimum 63 or mixed, activations at +127, -127 and alternating),
and checks the split arrays, including a reused activation of a narrower row. It was written against the new
signature and failed to compile before the change (missing `xe`, `xo` and the new `q4kRow` signature), then passed
6 of 6. `CpuQ4kKernelParityTest`'s calls were updated to the new signature; its bound and oracle are unchanged and
it passes. A mutation check (the last term reading the even array instead of the odd one) failed this test 5 of 6,
`CpuQ4kKernelParityTest` 6 of 7 and `CpuIntegerKernelsTest` 3 of 13, and the file was restored.

Probe: as for candidates 1 to 4 (scratch, not committed), the row kernel alone at 14336x4096, weights cycled through
eight copies, 0 bytes allocated, 0 rows differing in bits from candidate 3. Unpinned, so cycles per weight may read
low by up to about 10%.

| Q4_K 14336x4096 | 1 thread cycles/weight (3 runs) | 12 threads, per core (3 runs) |
|---|---|---|
| Candidate 3 (shipped before this change), interleaved | 1.30 / 1.38 / 1.31 | 1.23 / 1.36 / 1.36 |
| Candidate 5 in the probe, interleaved | 0.91 / 0.93 / 0.92 | 1.07 / 1.08 / 1.14 |
| Production `q4kRow` after adoption | **0.89 / 0.94 / 0.91** | **1.02 / 1.06 / 1.11** |

**Adopted.** About 30% faster on one thread and about 21% per core at 12 (`host-specific` magnitudes; the direction is
`expected-general`, since it removes work repeated per row). Both widths read under the `<= 1.1` bound on this
unpinned reading; the 12-thread reading sits close to it (1.02 to 1.11), so the pinned kernel microbenchmark in the
closing gate decides the criterion. The Q6_K decode kernel (`q6kPass`) and the Q5_K kernel split their activation the
same way per row; applying the split there is a named, unmeasured candidate (Q6_K is already under its bound; Q5_K is
out of scope).

**Module tests after candidate 5:** `mvn test -pl node`: 1,139 tests (the 1,133 before plus the 6 new), 0 errors,
47 skipped, **1 failure**: `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (free bytes
reported with the device full 110,755,840 against `<= 67,108,864`). It failed again on two runs of the class alone.
It is a device-wide free-memory check on the GPU: it fills the device and reads the driver's free-memory query. No
class changed by this tier's work today is on its path (`PrefillWindowRegion`, `DeviceScratchBudget`, `CudaMatVec`,
`CudaBindings` reference none of `CpuIntegerKernels`, `Q8Activation`, `CpuIntegerGemm`), and the device was shared
with the desktop (Xorg, the desktop shell and two browser processes, about 437 MiB). It passed in the 2026-10-12 full
suite. Recorded as environment-sensitive, in the same class as `PrefillRegionHandlerParityTest`'s free-VRAM check, and
not loosened; it is re-run in the closing work on a quiet device.

**Next:** item 3 (width-generic integer species, forced-width tests; `NEEDS-AVX2-HARDWARE` for the 256-bit
throughput threshold on this host). Optional before it: the activation split for `q6kPass`, measured.

### 2026-10-13: step 5, item 3 (width-generic integer species)

**Plan against code, re-verified.** `scripts/performance-tests/check-plan-thresholds.sh` passes. The 128-bit species
are now at `CpuIntegerKernels` lines 58 to 60 and `CpuIntegerGemm` lines 57 to 59 (line drift only); the `SIMD
available` line is still built in `VectorQuantKernels` and printed by `ConsoleMain`. On this host the JVM reports a
128-bit preferred species for byte, short, int and float alike (`VectorShape.preferredShape()` is `S_128_BIT`,
`/proc/cpuinfo` AVX without AVX2, as `INVENTORY.md` records). No plan drift.

**Design.** The 128-bit classes are kept unchanged as the parity oracle, and the 256-bit forms are new classes
beside them (`CLAUDE.md`: prefer a new class), so no kernel line on this host's path changed:

- `CpuIntegerWidth` chooses once per JVM: `ByteVector.SPECIES_PREFERRED` at 256 bits or more selects 256, else 128;
  512 is not used (the plan allows it only if measured faster than 256). The forced-width hook is
  `-Djuno.cpu.integerWidth=128|256|auto`; any other value fails closed with the property named (at startup, through
  the `SIMD available` line, and at first kernel use in an embedder). `WIDE` is a `static final` constant, so the
  per-call dispatch folds away.
- `CpuIntegerKernels256`: Q4_K, Q5_K and Q6_K decode rows with one 32-byte load where the 128-bit form takes two.
  Q5_K and Q6_K reduce every sub-block to an exact int and keep the 128-bit float expression, so they return the
  same bits; Q6_K's two 16-weight sub-blocks per load are split by reducing the low four int lanes (masked with a
  constant vector, no mask operations) and taking the high half as the total minus the low, exactly. Q4_K scales
  its sub-block sums in eight float lanes and sums them in a fixed order, so it differs by float rounding only.
  Q8_0 stays on the scalar kernel at either width.
- `CpuIntegerGemm256`: the tile (a 16-weight sub-block as one sixteen-lane vector, eight float lanes, the offset
  term eight sub-blocks at a time) and the unpack (32 packed bytes per vector, same weights and factors).
  `CpuIntegerGemm`'s activations, scratch and task split are shared.
- `CpuIntegerKernels.matVecInto` and `CpuIntegerGemm.sgemmInto` take the selected width; overloads with an explicit
  `wide` flag let tests run both widths in one JVM.
- The selected width ends the startup `SIMD available` line: `... ; CPU integer kernels: 128-bit lanes (host
  preferred 128-bit)` here, with `, forced by -Djuno.cpu.integerWidth` when forced. Checked from the compiled
  classes for unset, `auto`, `128`, `256` and `512` (the last fails with `-Djuno.cpu.integerWidth must be auto, 128
  or 256, got '512'`).

**Tests first.** New `CpuIntegerWidthTest` (5 tests: auto selection at 64, 128, 256 and 512 preferred bits, forced
widths overriding the host, unsupported values failing with the property and value named, the selected width on
this host is 128, the summary in the startup line) and `CpuIntegerWidthParityTest` (25 tests: decode parity per
format at 256 and 4096 columns and on extreme blocks (every byte 0xFF, 0x00, 0x80, 0x7F, 0x55, 0xAA; activations
all +, all - and alternating); the unpack identical on weights and on the raw bits of every scale and offset; the
matmul at B = 1, 4, 7 and 32 (four-row loop and remainder) per format, rows not a multiple of the tile). Both were
written against the new API and failed to compile before it existed (missing `CpuIntegerWidth`, `CpuIntegerGemm256`
and the `wide` overloads), then passed. On this host the 256-bit kernels run in the Vector API's portable form,
which is correct and slow, so the shapes are small.

Bounds, stated: unpack, Q5_K, Q6_K and Q8_0 bit-identical; Q4_K within 1e-6 of the row's term magnitude
(`CpuQ4kKernelParityTest`'s bound, its helper reused); the matmul within 1e-6 of the output's term magnitude (the
sum of the absolute dot and offset terms the tile scales, from the unpacked weights, in double). The matmul bound
was first written against the call's largest output and passed (worst 7.9e-7 on Q6_K at 1024 columns), but it
tightens as columns grow because Q6_K's offset terms are large, so it was restated to the terms before any failure,
not loosened. Measured worst: 2.1e-7 of term magnitude, flat from 256 to 4096 columns.

Mutation checks, each from a backup and restored byte for byte: Q6_K's low-half mask dropped (2 of 25 failed: the
random rows; uniform extreme blocks have equal scales, which hides a misassigned half); Q4_K's high activation half
reading the low half (2 failed); the matmul offset reading the next sub-block's sum (8 failed); the Q6_K unpack's
second plane shifted by 3 (5 failed).

Existing kernel tests after the change: `CpuIntegerKernelsTest`, `CpuIntegerGemmTest`, `CpuQ4kKernelParityTest`,
`CpuQ6kKernelParityTest`, `CpuQ4kActivationSplitParityTest`, `CpuIntegerGemmUnpackParityTest`,
`VisionWidthCpuKernelGuardTest`, `KQuantGemmQualityTest`: 111 tests with the new ones, 0 failures, 1 skipped.
`mvn test -pl node` (48 min): 1,169 tests, 47 skipped, **2 failures, both device-wide free-VRAM checks on the GPU**:
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (78,118,912 free bytes with the device
full against `<= 67,108,864`, as on 2026-10-13 above) and `GpuAttentionHandlerParityTest.kernel_matches_scalar_attention`
at its closing memory check (free memory 64.8 MB below where the test started; its parity assertions passed). Neither
path reaches the CPU integer kernels. Both classes re-run alone straight after: 5 of 5 pass (desktop holding 397 MiB of
the device). Recorded as environment-sensitive, not loosened. The other 1,167 pass.

**Not measured, and why.** The 256-bit throughput threshold (`>= 1.5x` the 128-bit Q4_K kernel on one thread)
needs a host with 256-bit integer instructions: **NEEDS-AVX2-HARDWARE** (flagged to the owner, as the threshold
asks). This host's 128-bit path is unchanged by construction (no kernel line changed; one branch on a constant per
decode call and per prefill tile of 16 rows); the pinned closing gate's same-hour A/B scores it. `host-specific`: the 128-bit selection here.
`expected-general`: the selection rule and the parity results (integer sums are exact at any width).

**Raised with the owner (2026-10-13): the 256-bit default on AVX2 hosts is unmeasured.** As the plan specifies,
`auto` selects 256 bits on any host whose preferred integer width is 256 or more, so every AVX2 host would run
kernels no one has timed. This tier adopts kernels on measurement only, and on this host a vector form that was
faster on one thread was 42% slower at 12 (`20261012T225531Z-cpu-kernel-baseline`). Choices: (a) keep `auto` as
specified; (b) make `auto` resolve to 128 until a 256-bit measurement exists, keeping 256 as an explicit opt-in
(a one-line change in `CpuIntegerWidth.select`, plus its test); (c) obtain an AVX2 host and measure before the
tier closes. Recommendation: (b) now, then (a) once (c) is done.

**Owner decision 4 (2026-10-13): choice (b).** `auto` (and an unset property) now resolves to 128 bits on every host;
`-Djuno.cpu.integerWidth=256` is the explicit opt-in. Test first: `CpuIntegerWidthTest`'s auto case was rewritten to
expect 128 at preferred widths 64, 128, 256 and 512, failed on the old rule (`unset, preferred 256`: expected 128),
then passed after the one-line change in `CpuIntegerWidth.select`; `CpuIntegerWidthTest` and
`CpuIntegerWidthParityTest` 30 of 30. `docs/howto.md` and `docs/agent-arch.txt` updated. The width row stays
`NEEDS-AVX2-HARDWARE`; the owner will have it measured on borrowed hardware later (choice (c)), after which `auto` is
revisited. The design bullet above that says `auto` selects 256 on a 256-bit host is superseded by this decision.

**Next:** (optional) the activation split for `q6kPass` and the
decode-only attention sites, each measured; then the smoke script `smoke-cpu-kernels.sh`, `ModelQualityNllLiveTest`
on the four sweep models, `ModelLiveRunnerIT`, the earlier smokes, and the owner's pinned closing gate (step 6).

### 2026-10-13: plan change, two tiers added after this one (owner decisions)

[Tier 02F](TIER-02F-hardware-validation-kit.md) (hardware validation kit) and
[Tier 02G](TIER-02G-cuda-on-windows.md) (CUDA on Windows) now run after this tier and before Tier 03. No
change to this tier's scope: its width row still closes as `NEEDS-AVX2-HARDWARE`, with the forced-width
parity tests as the evidence. The kit's first run on a 256-bit integer host gives the reading on which
owner decision 4 (`auto` resolves to 128) is revisited, and that is recorded in Tier 02F.

**CI re-examined, owner decision 2026-10-13** (the README asks for the outcome to be recorded in the active
tier's file): a workflow for [Tier 02F](TIER-02F-hardware-validation-kit.md)'s kit, on free hosted runners,
is approved as part of that tier. The 2026-10-01 decision stands for `ci.yml`, and no tier's exit criteria
wait on CI. Recorded in the README's "Test infrastructure" section.

### 2026-10-13: item 6, second candidate (vector attention head; unpinned probe and A/B)

**Plan against record, corrected.** Step 4 orders item 6 as "the thread split first and vectorization second, each
re-measured at 512 and 2048". The split landed on 2026-10-12; the vectorization candidate was never run, and this
record's later "Next" lines (and the resume prompt) had dropped it. It is run now. No scope change.

**Decomposition (scratch probe, not committed).** One row's pooled attention (`CpuThreads.forEach` over the heads, as
`attendPooled` runs it) at 12 threads, per sweep model head shape, keys at 512 and 2048, min of five timings of 60
rows each; accuracy against a double-precision reference; allocation summed over every thread. Variants of the head
body, one term changed at a time:

| ms per row, 12 threads | TinyLlama 512 / 2048 | Mistral 7B 512 / 2048 | Qwen2.5-3B 512 / 2048 | Phi-3.5-mini 512 / 2048 |
|---|---|---|---|---|
| Shipped scalar head | 0.59 / 2.25 | 0.89 / 4.08 | 0.47 / 1.88 | 0.98 / 4.07 |
| Dot in 128-bit vectors only | 0.53 / 1.95 | 0.72 / 3.80 | 0.41 / 1.51 | 0.82 / 3.70 |
| Diagnostic: dot removed (wrong result) | 0.47 / 1.62 | 0.56 / 2.28 | 0.35 / 1.28 | 0.54 / 2.26 |
| Exact dot, value sum in registers, 32 floats per pass (128-bit) | **0.23 / 0.94** | **0.43 / 2.91** | **0.25 / 0.98** | **0.67 / 2.95** |
| Same, 64 floats per pass (256-bit) | 0.27 / 0.91 | 0.43 / 2.63 | 0.25 / 0.92 | 0.58 / 2.79 |
| Vector dot and value sum in registers (16 floats per pass) | 0.21 / 0.66 | 0.36 / 2.86 | 0.19 / 0.64 | 0.61 / 2.97 |

The dot is not the dominant term: removing it leaves 55% to 85% of the time. The serial loop's weighted value sum
(`out[d] += w * v[d]` for every key) loads, updates and stores every output element once per key; holding the sums
in vector registers across all keys and storing them once is the large term (`expected-general`; magnitudes
`host-specific`). Every variant allocated nothing beyond the probe's own bookkeeping. Mistral 7B and Phi-3.5-mini gain
least at 2048: their keys and values at 2048 positions (16 MB and 48 MB) exceed this host's 12 MB L3, so streaming
them is the floor (`host-specific`). The 256-bit form is a few percent faster on the wide heads here (AVX has 256-bit
float lanes), but would run in the Vector API's portable form, far slower, on a 128-bit-only CPU, so it is not taken.

**Candidate A1, adopted: `GqaVectorHead`.** New class: one query head with the scores computed by the serial loop's
own code and the weighted value sum held in 128-bit float vectors in blocks of 32 elements (eight vectors), then 16,
then a scalar tail, stored once. Each output element receives the same products in the same order from zero, so the
output is **bit-identical** to `GqaMath.attend`. `GqaMath.attendPooled` runs it when `VECTOR_HEAD`
(`VectorQuantKernels.AVAILABLE`) holds and keeps the serial head otherwise; an overload with an explicit flag lets
tests run both. `GqaMath.attend`, the oracle for the GPU attention parity tests, is unchanged. Callers: every
handler's host window loop (item 6's wiring), Phi-2's and the LoRA handlers' `gqa` (their prefill, decode and LoRA
playback); the LoRA training step that keeps attention weights for its backward pass has its own loop and is untouched.

**Tests first.** New `node/src/test/java/cab/ml/juno/node/GqaVectorHeadTest.java` (17 tests): bit-identical to
`GqaMath.attend` at every sweep model's head shape and the other handlers' (64, 80, 96, 128 wide), widths that leave
a remainder after the 32-wide and 16-wide blocks (48, 72, 40, 20, 8, 256), windows, 1 to 2048 keys, logits in the
thousands with signed zeros in the values, the serial head still available and identical, and the vector head
selected when the Vector API module is present. Written against the new API, it failed to compile first (missing
overload, `VECTOR_HEAD`, `GqaVectorHead`), then passed 17 of 17. `GqaMathPooledTest` (13), `GqaMathOracleTest` (7) and
`GqaMathWindowTest` (3) pass unchanged; the pooled test now exercises the vector head by default. Mutation checks, each
restored byte for byte: a 32-wide block lane offset (14 of 17 failed), a 16-wide block lane offset (3 failed), the
scalar tail skipping the first key (4 failed).

**End to end, unpinned A/B (2026-10-13, not published).** `compare-llama-cpp.sh --cpu --threads 12 --juno-warmup 1
--juno-reps 1 --reps 1 --n-gen 16 --no-publish`, alternating A B A B: jar `b8b3e1d50c9fdeb5` (this tree before A1)
against `d1c139c0d1f25754` (with A1).

| Model, `n_prompt` | CPU pp t/s before (runs) | CPU pp t/s with A1 (runs) | Change of means | CPU tg t/s before / with A1 |
|---|---|---|---|---|
| TinyLlama 1.1B, 512 | 21.11 / 21.40 | 28.13 / 22.84 | +20% | 11.56, 12.32 / 11.40, 12.42 |
| Qwen2.5-3B, 512 | 10.43 / 8.28 | 11.37 / 10.99 | +19% | 4.58, 4.54 / 4.65, 4.58 |
| Phi-3.5-mini, 512 | 6.25 / 6.36 | 6.47 / 6.36 | +1.5% | 2.96, 2.99 / 2.98, 2.90 |
| TinyLlama 1.1B, 2048 | 14.86 / 15.13 | 19.54 / 19.90 | **+31%** | 12.27, 12.30 / 12.27, 11.83 |

Every A1 run is at or above its paired baseline run; the 512-token rows are within this unpinned host's run-to-run
spread (Juno's own readings of one jar ranged 21.1 to 27.3 t/s on TinyLlama 512 in one session, while the reference
tool held within 6%), so their size is indicative; the 2048 row is consistent. Generation is unchanged, as expected
(the LLaMA-family decode site still calls the serial `attend`). Against the reference tool's readings in the same
runs, A1 reads about 0.36x (TinyLlama), 0.40x (Qwen2.5-3B) and 0.34x (Phi-3.5-mini) CPU pp at 512, and 0.30x at 2048
on TinyLlama: above the `>= 0.30x` row at 512 on the three models read, unpinned; Mistral 7B was not in this A/B. The
pinned closing sweep scores the row.

**Next: candidate A2**, the dot in vector lanes (four 128-bit accumulators, fixed-order lane sum), held to
`GqaMathOracleTest`'s bound instead of bit-identity; the probe puts it at a further 10% to 34% of attention time on the
64- and 128-wide grouped heads (TinyLlama, Qwen2.5-3B) and none on the memory-bound ones. Measured on its own against
the A1 jar.

### 2026-10-13: item 6, candidate A2 (the dot in vector lanes; adopted)

**Change.** `GqaVectorHead` computes each key's dot product in four 128-bit float accumulators (16 elements per step,
then 4, then a scalar tail) and sums the lanes in a fixed order; the value sum stays A1's. The output is no longer
bit-identical to `GqaMath.attend`: it is held to `GqaMathOracleTest`'s bound (2e-6 + 2e-6 sqrt(keys), absolute, on
value rows in [-1, 1]) against a double-precision reference, and to the same bits on every call and at every thread
count. `GqaMath.attend` is unchanged; the serial pooled head (taken without the Vector API module) stays
bit-identical to it.

**Tests.** `GqaVectorHeadTest` rewritten for the bound (18 tests: the A1 shapes and widths plus a 6-wide head, query
scales 1 and 6, windows; same bits at 1, 2, 5 and 12 threads over 400 calls each; logits in the thousands; the vector
head selected with the module present), with its own double reference. `GqaMathPooledTest`'s bit-identity checks now
call the serial pooled head explicitly (`attendPooled(..., false)`), the form whose bit-identity still holds. Both
passed on A1 first (A1 is exact, so within the bound): regression tests, not seen failing. Worst error with A2, 2.3e-6
against a bound of 4.7e-5 (A1 and the serial loop: 3.0e-6 on the same rows). Mutation checks, restored byte for byte:
one dot accumulator left out of the final sum (13 of 18 failed); a dot loop's key index off by one (11 failed, 4
errors). `GqaMathOracleTest` and `GqaMathWindowTest` pass unchanged.

**Probe (production form, A1 against A2, two interleaved repetitions, ms per row at 12 threads).**

| Shape | 512: A1 / A2 | 2048: A1 / A2 |
|---|---|---|
| TinyLlama (32/4/64) | 0.24, 0.28 / 0.18, 0.15 | 0.87, 0.87 / 0.54, 0.58 |
| Mistral 7B (32/8/128) | 0.40, 0.52 / 0.28, 0.27 | 2.80, 2.79 / 2.46, 2.42 |
| Qwen2.5-3B (16/2/128) | 0.27, 0.23 / 0.15, 0.14 | 0.94, 0.96 / 0.54, 0.55 |
| Phi-3.5-mini (32/32/96) | 0.54, 0.57 / 0.67, 0.49 | 2.88, 2.89 / 2.52, 2.54 |

**End to end, unpinned A/B (2026-10-13, not published; A1 jar `d1c139c0d1f25754` against A2 jar `3e2c4fc17be7ffbc`,
alternating).** CPU pp t/s: TinyLlama 512, A1 23.06 / 24.34, A2 24.60 / 31.65; Qwen2.5-3B 512, A1 9.14 / 10.11, A2
8.54 / 8.87; TinyLlama 2048, A1 21.10 / 21.57, A2 20.43 / 20.18. Inconclusive at this resolution: the reference
tool's own prompt speed moved by up to 22% within the session (TinyLlama 2048: 71.3 and 69.5 t/s during the A1 runs,
60.6 and 66.5 during the A2 runs). The JFR method samples of the same measured prefills, which drift does not
distort, put the attention head at 10.3% and 11.0% of TinyLlama 2048 samples with A1 and 6.0% and 5.6% with A2, and at
1% to 2% of every 512-token prefill with either. **Adopted** on the probe and the in-situ share: about 5% of a
2048-token TinyLlama prefill, and nothing measurable at 512, where the attention head is too small a share to move the
CPU pp row (`host-specific` magnitudes; the direction is `expected-general`). The pinned closing gate reads the final
build.

**Finding: the matmul tile sometimes runs its 16-bit lane operations through the Vector API's generic Java path.** In 3
of 28 measured prefills across both A/Bs (Phi-3.5-mini 512 with the A1 jar; both Qwen2.5-3B 512 runs with the A2 jar),
about 74% of execution samples sit in `jdk.incubator.vector.ShortVector.lanewiseTemplate` instead of
`CpuIntegerGemm.tile` (6% there, against 69% to 79% in the other runs). It is not caused by A2 (it hit an A1 run too)
and costs little on these readings (Phi-3.5-mini 6.36 t/s in that run against 6.25 to 6.47 in the others; Qwen2.5-3B
8.5 to 8.9 against 9.1 to 10.1). The likely mechanism, not verified, is C2 declining to inline the lane operations into
the very large `tile` method in some JVM runs (the order in which methods reach C2 varies). It adds run-to-run variance
to every CPU prefill reading. Recorded for the closing work: the pinned gate's three repetitions per side expose it,
and if it shows there, splitting the tile's 16-wide and 32-wide sub-block loops into separate methods (each written in
place, so no vector crosses a call) is the candidate (`expected-general`: C2 inlining behaviour).

**Module tests after A1 and A2:** `mvn test -pl node`: 1,187 tests (the 1,169 before plus `GqaVectorHeadTest`'s 18), 0
failures, 0 errors, 47 skipped, BUILD SUCCESS, 49 min. The two device-wide free-memory checks that failed earlier today
(`PrefillReserveDeviceTest`, `GpuAttentionHandlerParityTest`'s closing memory check) passed in this run. The other
modules are owed with the closing work.

**Also prepared, not run:** `scripts/performance-tests/smoke-cpu-kernels.sh` (runs `smoke-cpu-hot-path.sh` on the
pre-tier jar and on the candidate, and compares every greedy reply between them; a different reply is reported with
both texts, not failed, since this tier changes float rounding by design and the NLL test scores quality), and the
closing gate under `dist/cpu-kernels-close/` (`wait-and-run.sh`, `run-gate.sh`: K kernel microbenchmark, S closing
sweeps at 128, 512 and 2048, C same-hour CPU A/B at 128 and 512, W same-hour CPU A/B at 2048, A GPU A/B, L LoRA). No
jars, baseline tree or READY yet.

**Raised with the owner, 2026-10-13: the 2048-token host-attention gate.** As written it is a same-hour pinned A/B
against the pre-tier jar on all four sweep models (Part W). The harness's two warmups use the measured prompt, and the
pre-tier jar's 2048-token prefill takes about 26 minutes on Mistral 7B and 20 on Phi-3.5-mini, so Part W alone runs
about 13 to 14 hours. Choices: (a) keep Part W; (b) read the 2048 gate (`>= 1.50x`) from the closing sweeps (Part S,
pinned, median of three) against the step 2 sweeps of the same pre-tier jar (`20261012T071213Z`, `20261012T162344Z`,
pinned, median of three). A 1.50x bar sits far outside the README's 15% drift floor, which is what the same-hour rule
guards. The 512 gate (`>= 1.10x`) is inside the floor and stays a same-hour A/B (Part C) either way. Recommendation:
(b).

**Next:** the owner's decision above; then `ModelQualityNllLiveTest` on the four sweep models,
`smoke-cpu-kernels.sh` against the pre-tier jar, `ModelLiveRunnerIT`, the earlier smokes, the vision and LoRA
playback gates (unpinned, looser than 0.90x), the full unit suite, docs and `CHANGELOG.md`; then the baseline tree,
both jars and READY for the owner's pinned closing gate.

**Owner decision 5 (2026-10-13): choice (b).** The 2048-token host-attention gate (`>= 1.50x`) is read from the
closing sweeps against the step 2 sweeps of the pre-tier jar (Threshold block above); the gate's Part S scores it and
Part W stays out of the run. The 512-token gate stays in Part C. The owner started
`bash dist/cpu-kernels-close/wait-and-run.sh` the same evening; READY is written after the pre-gate checks.

### 2026-10-13: closing gate prepared and started (owner run)

**Pre-gate checks (this tree, after A1 and A2).**
- Unit tests, the ten other modules (`node` excluded, its 1,187 having passed an hour earlier): 1,239 tests, 0
  failures, 0 errors, 5 skipped, BUILD SUCCESS (the same count as the 2026-10-12 full suite's non-`node` modules).
- `ModelLiveRunnerIT` on TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B and Qwen3-1.7B (`mvn install -DskipTests`,
  then `mvn verify -pl juno-master -Pintegration`): 5 of 5 models, every check PASS, 17 min 35 s. Checks 1 to 8 run in
  forked cluster JVMs on the CPU backend, so they exercise this tier's kernels and the vector attention head in both
  cluster modes; checks 9 and 10 took CUDA on this host (the IT uses the GPU when one is present), so their CPU form is
  left to the earlier smokes.

**Gate.** `dist/cpu-kernels-close/` (git-ignored): baseline jar `b9a0440b3a1b597a` (HEAD `01027ee9`, the jar of step 2's
sweeps), candidate jar `e37e16c004f4c213` (this tree; equal to the tree's shaded jar), and `baseline-tree/` (`git
archive 01027ee9`, built with `mvn package -DskipTests`, jar `d3c4f655130ad917`, used only by Part L's
`compare-lora.sh`). READY written 18:02; the owner's `wait-and-run.sh` started `run-gate.sh` with `PART=KSCAL` at
18:02:39 (Part W left out by owner decision 5). Expected about eight to ten hours.

**Deferred to after the gate** (they need the machine, and none is expected to change the jar): `ModelQualityNllLiveTest`
on the four sweep models, `smoke-cpu-kernels.sh` against the pre-tier jar, the earlier smokes (CPU long prompt and
context policy among them), and the vision and LoRA playback gates (unpinned; looser than 0.90x). If one of them
forces a code change, the affected gate parts are re-run on the new jar.

### 2026-10-14: closing gate run 1 (owner, pinned): results, the allocation miss, and the fix

**Run.** `PART=KSCAL`, 2026-10-13 18:02 to 2026-10-14 08:37 local, jars baseline `b9a0440b3a1b597a`, candidate
`e37e16c004f4c213`. Published: [`20261013T230239Z-cpu-kernels-close-gate`](../perf-compare/20261013T230239Z-cpu-kernels-close-gate/INDEX.md)
(Parts K, C, A, L and the log) and the four closing sweeps [`20261013T231340Z`](../perf-compare/20261013T231340Z/INDEX.md)
(128), [`20261013T235235Z`](../perf-compare/20261013T235235Z/INDEX.md) (512), [`20261014T011910Z`](../perf-compare/20261014T011910Z/INDEX.md)
and [`20261014T050210Z`](../perf-compare/20261014T050210Z/INDEX.md) (2048), local paths scrubbed.

| Part | Gate | Result |
|---|---|---|
| K | Q6_K <= 2.7, Q4_K <= 1.1 cycles per weight, 1 thread and per core at 12 | Q6_K 2.13 and 2.49, met; Q4_K 0.931 met on one thread, **1.102** per core at 12 (median of 1.083, 1.102, 1.111): 0.2% above the bound. The gate printed 1.10 and passed it because it compared the rounded value; raised with the owner below |
| S | CPU tg >= 0.25x; CPU pp at 512 >= 0.30x; 2048 >= 1.50x the step 2 sweeps | met: tg 0.308x to 0.391x (Phi-3.5-mini binding); pp at 512 0.312x to 0.369x (Qwen2.5-3B binding); 2048 2.56x to 3.63x |
| C | CPU tg and pp >= 1.10x (Mistral 7B, Qwen2.5-3B) and >= 0.95x (TinyLlama, Phi-3.5-mini) at 128; pp >= 1.10x at 512; allocation <= 1.10x; GC <= 1.25x | throughput met on every model (128: pp 1.34x to 1.46x, tg 1.31x to 1.66x; 512: pp 1.56x to 2.02x); GC 0 ms; **allocation missed on every model**: 7.04x, 1.24x, 5.66x, 1.53x |
| A | GPU tg and pp >= 0.95x, allocation <= 1.10x, GC <= 1.25x | met: 0.969x to 1.013x, allocation 1.000x, GC 0 ms; greedy output equal in 3 of 3 runs per model |
| L | training >= 0.95x, playback >= 0.80x | met: training 1.024x (GPU), 0.999x (CPU); playback 0.977x, 0.995x; recall in every run |

**The allocation miss, traced.** The candidate's JFR allocation samples put the excess in the Vector API's generic
path: `AbstractVector.convert0` and `ShortVector.lanewiseTemplate`, about 23 GB and 20 GB in one 128-token TinyLlama
prefill and 16 GB and 14 GB over 64 decode tokens, against none in the baseline. The stacks lead to `CpuIntegerGemm`:
the one-row remainder loop of the tile (rows left over after the four-row loop) passed a vector to
`widenPairs(ShortVector)`. After G3 the tile method was large enough that C2 stopped inlining the helper in a server
JVM, so the vector was boxed and each operation on it allocated. Every scratch probe had read 0 bytes, because a
probe's small, single-purpose JVM compiled the helper inline. A short prompt (the generation lane's is about 21 tokens)
or any window not a multiple of four reaches the remainder.

**Fix, in two steps (tests first).**
1. New `node/src/test/java/cab/ml/juno/node/CpuKernelVectorEscapeTest.java`: (a) no production kernel class
   (`CpuIntegerKernels`, `CpuIntegerGemm`, their 256-bit forms, `GqaVectorHead`) declares a method that takes or returns
   a Vector API type, which is the rule these classes' comments already stated, now checked; (b) `sgemmInto` with rows
   left over (B = 1, 3, 5, 6, 7) allocates under 16 KB per call across all threads after warmup. On the gate's code (a)
   failed listing `CpuIntegerGemm.widenPairs` and four dead helpers in `CpuIntegerKernels` (`evenBytes`, `oddBytes`,
   `widenPairs`, `sumLanes`, no callers), and (b) failed 7 of 8 cases at 0.8 to 6.3 MB per call. Writing the
   remainder's widening in place and deleting the five helpers left Q6_K at B = 5 and 7 still allocating (3.1 MB per
   call): once both sub-block widths and both row forms had run in one JVM, the one tile method was recompiled too
   large to keep its lane operations in registers. The tile was then split into one method per row form and
   sub-block width (`fourRows16`, `fourRows32` and their one-row counterparts), each taking only arrays and ints; all
   nine cases passed, twice.
2. In a server the split's one-row forms still allocated on two models (19.3 GB in one Qwen2.5-3B 128-token prefill,
   0.10 GB on Mistral 7B, all in `oneRow32`): leftover rows are rare, so those methods run too seldom to be compiled
   reliably. The one-row forms were removed: `Activations` pads the batch with zero rows to a multiple of four
   (reset on every call, since the buffers are reused), the tile runs only the four-row forms, and only real rows are
   written out. A real row's arithmetic is the same in any position of the four, so the output is unchanged. New
   `CpuIntegerGemmTest.aBatchNotAMultipleOfFourIsPaddedWithoutChangingItsRows` (rows of a batch of 5 equal to the same
   rows inside a batch of 8, and again after a larger call reused the buffers); a mutation reading the accumulator at
   the unpadded stride failed 11 of 33. The same split and padding went into `CpuIntegerGemm256`.

**Bits unchanged.** A scratch probe hashing `sgemmInto`'s output bits over Q8_0, Q4_K, Q5_K and Q6_K at B = 1, 3, 4, 5,
7, 8, 32 and 33 reads `da2138d0e9513815` for the gate's tile, the split tile and the padded tile alike. Mutation
checks on the split methods: a row reading another row's offset sums (3 of 23 failed), a one-row activation scale index
(5 failed).

**The allocation metric counts more than the measured request.** `allocated_bytes_per_token` divides the sum, over the
threads seen in the recording, of each thread's largest `jdk.ThreadAllocationStatistics` running total by the lane's
tokens. The running total counts from the thread's start, so the long-lived worker and server threads bring model
load and both warmup requests into the figure. Measured inside the recording instead (each thread's last running
total minus its first), the step-1 fix's jar allocated what the baseline did on every model and lane except the two
remainder cases just named (TinyLlama prefill 0.013 GB against 0.013 GB, Phi-3.5-mini prefill 0.024 against 0.025,
every generation lane within 0.002 GB), while its harness figure still read 1.3x to 1.7x the baseline's: this tier's
kernels are more vector code, and vector code allocates while it runs interpreted or under C1, before C2 compiles it,
which happens during the warmups. Raised with the owner below.

**Verification of the padded build in a server (2026-10-14, unpinned, two runs; allocation does not depend on
clocks).** `compare-llama-cpp.sh --cpu --threads 12 --n-prompt 128 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane
--no-publish --juno-jar <jar b40e1ce4154d1af5>`, four sweep models, run twice:

| Model, lane | Allocated inside the measured recording: pre-tier (gate run 1) / padded run 1 / run 2 | Vector API allocation sites, run 1 / run 2 |
|---|---|---|
| TinyLlama, prefill / generate | 0.013 / 0.013 / 0.014 GB; 0.019 / 0.019 / 0.018 GB | 0 / 0 MB |
| Qwen2.5-3B, prefill / generate | 0.021 / 0.023 / 0.023 GB; 0.077 / 0.074 / 0.078 GB | 0 / 0 MB |
| Phi-3.5-mini, prefill / generate | 0.025 / 0.001 / 0.025 GB; 0.021 / 0.021 / 0.024 GB | 0 / 0 MB |
| Mistral 7B, prefill / generate | 0.029 / 0.031 / 0.029 GB; 0.021 / 0.021 / 0.021 GB | 0 / 0 MB |

The measured requests now allocate what the pre-tier build's do (Phi-3.5-mini's 0.001 GB is a recording too short to
hold two samples per thread). The harness's `allocated_bytes_per_token` still reads 1.16x to 1.67x the gate's pre-tier
medians (TinyLlama 1.67x / 1.48x, Qwen2.5-3B 1.65x / 1.46x, Phi-3.5-mini 1.40x / 1.46x, Mistral 7B 1.16x / 1.27x), all
of it outside the measured request. Direct check (scratch probe, fresh JVM, `sgemmInto` Q4_K 2048x2048 at B = 32): the
first call allocates 2.2 GB, while the method still runs the Vector API's generic path before C2 compiles it; every
call after it allocates 1.2 to 1.8 KB. Each vector kernel pays that once per JVM, during the warmups, and the harness's
figure charges it to the measured tokens; this tier added vector kernels (Q6_K decode, matmul unpack, attention head),
so the charge grew. `expected-general` (C2 and the incubating Vector API's interpreted form).

**Raised with the owner (2026-10-14).**
1. *The allocation metric.* (a) Count allocation inside the recording (each thread's last running total minus its
   first), keep the current figure as a separate field so history stays comparable, and re-read both sides of the
   gate's A/B with the same extractor; (b) keep the metric and close the allocation half by owner decision, the excess
   recorded as one-time warmup allocation; (c) keep it and try to cut warmup allocation. Recommendation: (a).
2. *Q4_K at 12 threads, 1.102 against `<= 1.1`.* (a) Met within measurement resolution, by owner decision recorded with
   the number; (b) open, pending a further Q4_K candidate (none in hand). Recommendation: (a). The gate script compares
   unrounded values from now on.
3. *What to re-run.* The build changed (prefill matmul: same output bits, no allocation). Part C failed and must be
   re-run; Part S's ratios are for the previous jar. Part K (decode kernels), Part A (GPU) and Part L's training are
   untouched by the change. (a) Parts C and S, about eleven hours; (b) Part C only, keeping run 1's sweeps.
   Recommendation: (a), so the README's CPU rows and the exit criteria read the final build.

**Module tests on the padded build:** `mvn test -pl node`: 1,197 tests (1,187 before, plus `CpuKernelVectorEscapeTest`'s
9 and the padding test), 47 skipped, 1 failure: `ResidentQkvPathTest.noDeviceAllocationPerCall`, a device-wide
free-memory equality on the GPU, which read 4 MiB *more* free memory after the calls than after the first (another
process on the device released memory mid-test; nothing on this tier's CPU path reaches it). The class alone straight
after: 19 of 19 pass. Recorded as environment-sensitive with the other device-wide checks, not loosened.

### 2026-10-26: owner decisions 6 to 8; allocation counted inside the recording; closing gate run 2 prepared

**Plan against code, re-verified at HEAD `c2633a68`** (the owner's commit of this tier's work through 2026-10-14).
`scripts/performance-tests/check-plan-thresholds.sh` passes (25 tier files). The tree is the padded build of 2026-10-14:
`CpuIntegerGemm` and `CpuIntegerGemm256` run only `fourRows16` and `fourRows32` over a batch padded to a multiple of four
(`Activations.padded`), the five helpers that took a vector are gone, `CpuKernelVectorEscapeTest` and
`CpuIntegerGemmTest.aBatchNotAMultipleOfFourIsPaddedWithoutChangingItsRows` exist, and no source file is newer than the
tree jar `b40e1ce4154d1af5`. The gate script compares the kernel cycles unrounded. No drift. The three questions raised on
2026-10-14 had no recorded answer and were put to the owner.

**Owner decision 6 (2026-10-26): allocation is counted inside the recording (choice (a)).** The allocation half of the
standing gate reads each thread's growth inside the measured recording; the old whole-thread figure stays published
beside it, so history stays comparable, and both sides of an A/B are read by the same extractor.

**Owner decision 7 (2026-10-26): Q4_K at 12 threads is met within measurement resolution (choice (a)).** Part K of run 1
read 1.102 cycles per weight per core against `<= 1.1` (median of 1.083, 1.102 and 1.111; 0.2% above the bound), 0.931 on
one thread. The bound is not restated; the threshold criterion records this decision with the number.

**Owner decision 8 (2026-10-26): the re-run covers Parts S and C (choice (a)), about eleven hours,** so the README's CPU
rows and the exit criteria read the final build. Parts K (decode kernels), A (GPU) and L (LoRA) stand from run 1: what
changed after run 1 is the prefill matmul tile's row handling (same output bits) and the metrics extractor.

**Implementation, tests first.**
- `metrics`: `JdkEventBucket` keeps each thread's smallest `jdk.ThreadAllocationStatistics` running total beside its
  largest and emits `jdk.ThreadAllocationStatistics.bytes_in_recording`, the sum of each thread's largest minus smallest;
  `bytes_total` is unchanged. A thread sampled once adds nothing. Test:
  `JfrMetricsExtractorJdkEventsTest.threadAllocationStatistics_inRecordingCountsOnlyEachThreadsGrowthInsideTheRecording`
  (256 MB allocated by the test thread before the recording starts must be in `bytes_total` and not in the new figure;
  the value must equal the per-thread growth recomputed from the same recording; the test fails as vacuous if the test
  thread was sampled fewer than twice), and the no-events test now requires the new key at zero. Seen failing first (2
  of 10, the key missing), then passing; `metrics` module 86 tests, 0 failures. Mutation check (each thread's whole
  running total summed into the new figure): the new test failed; the file was restored.
- Harness, `compare-llama-cpp.sh`: `allocated_bytes_in_recording` and `allocated_bytes_per_token_in_recording` in every
  per-rep result, the rep aggregate (worst rep, every reading kept, as for the old figure), the lane merge (generation
  lane, as for the old figure) and the compare record; an `Alloc B/tok in request` column and a note in `INDEX.md`.
  Recordings are now extracted by the harness tree's own jar whichever jar `--juno-jar` launched (`find_extractor_jar`;
  `host.json` records `jfr_extractor_jar` and its hash), so a pre-tier jar that predates the field still gets it and both
  sides share one definition. Self-test cases written first: 8 failed against the old script (6 for the fields, 2 for
  the extractor jar), then `--selftest` passed, 128 checks.
- The README's standing gate names `jfr.allocated_bytes_per_token_in_recording`, with a dated note; `docs/howto.md` and
  `docs/agent-arch.txt` describe the new field. Tree jar after the change: `5d68f809c10e2292`.

**Run 1 re-read with the new extractor** (offline, all 216 measured recordings, both sides, extractor jar
`5d68f809c10e2292`): [`20261026T204605Z-cpu-kernels-allocation-reread`](../perf-compare/20261026T204605Z-cpu-kernels-allocation-reread/INDEX.md).
Generation lane, bytes per token, median of three per side, candidate over baseline:

| Part | TinyLlama | Qwen2.5-3B | Phi-3.5-mini | Mistral 7B |
|---|---|---|---|---|
| cpu128, whole-thread (run 1's gate figure) | 7.04x | 1.24x | 5.66x | 1.53x |
| cpu128, in the recording | **332x** | **1.000x** | **468x** | **1.003x** |
| cpu512, in the recording (not gated at 512) | 325x | 0.991x | 482x | 1.001x |
| gpu512, in the recording (Part A) | 0.930x | 0.995x | 0.991x | 0.999x |

- Run 1's candidate fails on the new figure too: its leftover-row defect allocated inside the measured request, and the
  new figure reads it 50 to 80 times more sharply than the old one did, whose warmup bytes diluted it. Qwen2.5-3B and
  Mistral 7B, whose generation lanes did not reach that path, read 1.00x; the old figure's 1.24x and 1.53x there was
  warmup. So the change makes the gate stricter where the request itself allocates, not looser (`expected-general`).
- Part A's allocation half re-reads at 0.930x to 0.999x on the new figure: met, as on the old one.
- The re-read `bytes_total` equals run 1's published figure on all 216 recordings.
- Known limit (`expected-general`): threads are sampled every 500 ms under `juno-perf.jfc`, so a request shorter than about a
  second undercounts (the GPU prefill lanes read 0 to 3 MB). Both sides of an A/B are read the same way; the gate reads
  the generation lane, which runs for seconds on the CPU.

**Closing gate run 2, prepared** (`dist/cpu-kernels-close/`, git-ignored). Run 1's outputs, candidate jar, READY and log
are in `run1/`. `run-gate.sh`: default `PART=SC`; allocation scored on `jfr.allocated_bytes_per_token_in_recording`, the
whole-thread figure printed beside it and not gated; a side without a reading, or a baseline reading of 0, fails as not
scorable (before, a missing reading was read as 0 and passed); every ratio gate compares the unrounded B/A (the table
still prints three places). Checked offline on run 1's Part C results with the re-read values put in: TinyLlama and
Phi-3.5-mini fail allocation (332x, 468x), Qwen2.5-3B and Mistral 7B pass, throughput as in run 1; with every baseline
reading of one model removed, that model fails as not scorable. `wait-and-run.sh` defaults to `PART=SC`. The baseline jar
is unchanged (`b9a0440b3a1b597a`); the candidate jar and READY are written after the pre-gate checks below pass.

**Pre-gate checks on this tree (padded tile plus the in-recording field).**
- Full unit suite (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`,
  `-Dmaven.test.failure.ignore=true` so that no module is skipped, results read from every module's surefire reports):
  **2,437 tests, 0 failures, 0 errors, 52 skipped**, BUILD SUCCESS, 1 h 01 min (`node` 1,197 with 47 skipped, `metrics` 86).
  The device-wide free-memory checks passed in this run. The SEVERE lines in the log are the loader's negative tests.
- `ModelLiveRunnerIT` (`mvn install -DskipTests`, then `mvn verify -pl juno-master -Pintegration`) on TinyLlama,
  Qwen2.5-3B, Phi-3.5-mini, Mistral 7B and Qwen3-1.7B: **5 of 5 models, every check PASS**, 0 FAIL, 14 min in the IT.
  Checks 1 to 8 run in forked cluster JVMs on the CPU backend (pipeline and tensor parallel), so they cover the padded
  tile; checks 9 and 10 took CUDA on this host.
- `smoke-cpu-kernels.sh --baseline-jar <pre-tier b9a0440b3a1b597a>` (candidate: the tree jar `4e480ab9131887eb`, the same
  source rebuilt by the IT's `mvn install`): both builds pass every check of `smoke-cpu-hot-path.sh` (four sweep models
  at `--threads` 4 and 12, short decode and a prompt of about 512 tokens, and the moondream2 CPU vision case at the
  741 width), and **all 18 greedy replies are identical between the builds** (0 different), 78 min.
- `ModelQualityNllLiveTest` (`mvn test -pl node -Dtest=ModelQualityNllLiveTest -Djuno.test.modelQuality=true`), 2,048
  tokens per model, the integer kernels against the float-activation kernels: 4 tests, 0 failures, 148 min. Mean NLL
  ratio **1.00017** (TinyLlama), **0.99935** (Qwen2.5-3B), **1.00209** (Phi-3.5-mini), **1.00080** (Mistral 7B), against
  `<= 1.01`; greedy argmax agreement over the first 512 positions 507, 512, 509 and 510. Phi-3.5-mini moved from Tier 02C's
  reading (0.99935 to 1.00061 across the four) as expected from this tier's float-lane scaling and vector attention dot
  (`host-specific` magnitudes). Published with the smoke comparison above:
  [`20261027T025856Z-cpu-kernels-quality`](../perf-compare/20261027T025856Z-cpu-kernels-quality/INDEX.md).
- Earlier tiers' smokes, unmodified, one at a time: `smoke-consistency.sh` (CPU and GPU) **54 PASS, 0 FAIL**;
  `smoke-grammar.sh` **19 PASS, 0 failures**; `smoke-tools.sh` **14 PASS, 0 failures**;
  `smoke-long-prompt-prefill.sh --cpu --lengths "128 512"` (TinyLlama and Mistral 7B, both schedules) **all checks
  pass**; GPU regression: `smoke-packed-kquant-matmul.sh` (four sweep models, both schedules) **all checks pass**,
  `smoke-context-policy.sh` (Phi-3.5-mini, both schedules) **all checks pass**.
- `smoke-gpu-residency.sh` (default models: TinyLlama with its cluster runs, Mistral 7B, llama-1-30b) **13 PASS, 0 FAIL**,
  15 min. llama-1-30b runs at partial offload (20 layers on the device, the rest on the CPU, so on this tier's
  kernels): greedy output identical with the region on and off over 32 tokens, no device-memory growth (cross-surface
  row 2).
- Vision gate (unpinned, median of three, each tree's own `compare-vision.sh --skip-build`, A B alternating; baseline
  tree `01027ee9`, jar `d3c4f655130ad917`): GPU lane latency **0.556x**, decode **2.106x**; CPU lane latency **0.555x**,
  decode **2.104x** (gates `<= 1.25x`, `>= 0.80x`): **met**. Each build's reply is identical across its six runs; the
  builds part after about 28 tokens (float rounding, by design).
  [`20261027T035713Z-cpu-kernels-vision`](../perf-compare/20261027T035713Z-cpu-kernels-vision/INDEX.md).
- Harness end to end (scratch, unpinned, not a reading): `compare-llama-cpp.sh --cpu --juno-jar <pre-tier jar>` on
  TinyLlama, 128 + 16 tokens. The pre-tier jar ran (`juno_jar_sha256_16` `b9a0440b3a1b597a`), the tree jar extracted
  (`jfr_extractor_jar_sha256_16` `4e480ab9131887eb`), and the new figure is present in the per-rep, aggregate, compare and
  `INDEX.md` outputs (0.34 MB per token in the request against 338 MB on the whole-thread figure). `--selftest` and
  `check-plan-thresholds.sh` pass.

**Gate run 2 released (2026-10-26).** Jars: baseline `b9a0440b3a1b597a` (unchanged), candidate `4e480ab9131887eb` (equal
to the tree's shaded jar). `dist/cpu-kernels-close/READY` written last. Owner command:
`bash dist/cpu-kernels-close/wait-and-run.sh` (default `PART=SC`, about eleven hours; Part S publishes its four sweeps to
`docs/perf-compare/` itself). Nothing that builds, tests or loads the CPU or GPU runs on this host until the gate ends.

**Next:** after the owner's run: publish Part C (and the gate log) under `docs/perf-compare/<UTC>-cpu-kernels-close-gate-2/`
and Part S's sweeps; score every threshold on the final build; restate the README's CPU tg and CPU pp rows from the
sweeps (item 5) and move the reference column (README rule) if they supersede the current CPU reference; add
`docs/performance.md`'s section and the `CHANGELOG.md` entry; resolve the cross-surface checklist; tick the exit criteria
with evidence.

### 2026-10-27: closing gate run 2 (owner, pinned): results; owner decisions 9 and 10

**Run.** `bash dist/cpu-kernels-close/wait-and-run.sh` (`PART=SC`), 2026-10-27 10:47 to 21:30 local, jars baseline
`b9a0440b3a1b597a`, candidate `4e480ab9131887eb` (equal to the tree's shaded jar). A first start at 00:38 local was cut
off at about 05:56 by a host reset during Part S's last sweep (Phi-3.5-mini at 2048, Juno repetition 2 of 3): the
system journal stops mid-stream with no panic, oops, out-of-memory kill, lockup, machine-check, thermal or GPU message,
and the kernel on this host does not reboot on panic (`kernel.panic = 0`), so the reset came from below the operating
system (hardware, firmware or power). Not diagnosed further; the gate was re-run whole and completed. Published:
[`20261027T154748Z-cpu-kernels-close-gate-2`](../perf-compare/20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md) (Part C,
both gate logs) and the four closing sweeps [`20261027T154748Z`](../perf-compare/20261027T154748Z/INDEX.md) (128),
[`20261027T161927Z`](../perf-compare/20261027T161927Z/INDEX.md) (512), [`20261027T172646Z`](../perf-compare/20261027T172646Z/INDEX.md)
and [`20261027T202604Z`](../perf-compare/20261027T202604Z/INDEX.md) (2048); the interrupted attempt's three finished sweeps
(`20261027T053823Z`, `20261027T060952Z`, `20261027T071820Z`) are published marked not scored (owner decision 10). Local
paths scrubbed in every one. The gate printed `GATES MET`; every run reports `clock_pinned: true`.

| Part | Gate | Result |
|---|---|---|
| S | CPU tg >= 0.25x; CPU pp at 512 >= 0.30x; pp at 2048 >= 1.50x the step 2 sweeps | met: tg 0.259x (Phi-3.5-mini, binding) to 0.337x; pp at 512 0.422x (Qwen2.5-3B, binding) to 0.512x; 2048 3.27x to 4.38x |
| C | CPU tg and pp >= 1.10x (Mistral 7B, Qwen2.5-3B) and >= 0.95x (TinyLlama, Phi-3.5-mini) at 128; pp >= 1.10x at 512; allocation in the recording <= 1.10x; GC <= 1.25x | met: 128 pp 1.916x to 2.052x, tg 1.314x to 1.646x; 512 pp 2.252x to 2.684x; allocation 0.981x to 1.013x; GC 0 ms on every run |

**Two readings to note.** (1) Phi-3.5-mini's 2048 sweep is flagged `NOISY` on its generation lane only (2.52 to 2.96
t/s, 15.9% of the median against 15%); the 2048 bar reads its prefill lane, three readings 6.34 to 6.54 t/s (3.1%),
scorable. (2) The reference tool's CPU decode reading moved 13% to 34% between the interrupted attempt and the complete
run on the same jar (TinyLlama 27.75 and 37.08 t/s), Juno's 2% to 7%; the reference tool's own within-sweep deviation is
1.7% to 4.8% on three models and 16.5% on TinyLlama. The scored tg ratios are the complete run's, the lower pair
(Phi-3.5-mini 0.259x; 0.301x in the interrupted attempt). `host-specific`. Against run 1, prefill rose on every model
(Part C pp at 128 1.34x to 1.46x, now 1.92x to 2.05x): run 1's candidate boxed vectors in the tile's leftover rows, and
the padded tile removes that allocation and its time (`expected-general` mechanism, `host-specific` magnitudes).

**Every threshold, scored on the final build** (Parts K, A and L from run 1, which the tile fix and the extractor
change do not reach; owner decision 8):

| Threshold | Reading | Source | Result |
|---|---|---|---|
| CPU tg ratio >= 0.25x, every sweep model | 0.259x to 0.337x | Part S, `20261027T154748Z` | met |
| CPU pp ratio at 512 >= 0.30x, every sweep model | 0.422x to 0.512x | Part S, `20261027T161927Z` | met |
| Q6_K <= 2.7 cycles per weight, 1 thread and per core at 12 | 2.13 and 2.49 | Part K, run 1 | met |
| Q4_K <= 1.1 cycles per weight, 1 thread and per core at 12 | 0.931 and 1.102 | Part K, run 1 | met (12 threads within measurement resolution, owner decision 7) |
| Prefill attention: pp at 2048 >= 1.50x the pre-tier build | 3.27x to 4.38x | Part S against the step 2 sweeps | met |
| Prefill attention: pp at 512 >= 1.10x, same hour | 2.252x to 2.684x | Part C | met |
| Mean NLL <= 1.01x the float-activation kernels | 0.99935x to 1.00209x | `20261027T025856Z-cpu-kernels-quality` | met |
| CPU tg and pp >= 1.10x (Mistral 7B, Qwen2.5-3B), >= 0.95x (others), same hour | pp 1.916x to 2.052x, tg 1.314x to 1.646x | Part C | met |
| Vision latency <= 1.25x, decode >= 0.80x | GPU lane 0.556x and 2.106x, CPU lane 0.555x and 2.104x | `20261027T035713Z-cpu-kernels-vision` | met |
| LoRA training >= 0.95x, playback >= 0.80x | 1.024x (GPU) and 0.999x (CPU); 0.977x and 0.995x | Part L, run 1 | met |
| 256-bit Q4_K >= 1.5x the 128-bit kernel on an AVX2 host | not measurable here | | `NEEDS-AVX2-HARDWARE`; forced-width parity tests pass |
| Every throughput number a median of at least three, min/max published | sweeps publish min/max; Part C's per-run files and ranges are in the gate folder | | met |

Standing gate: GPU tg and pp 0.969x to 1.013x (Part A, run 1), allocation in the recording 0.930x to 0.999x (GPU, run 1
re-read) and 0.981x to 1.013x (CPU, run 2), in-span GC 0 ms on both lanes: met.

**Owner decision 9 (2026-10-27): both CPU end-of-plan rows retired as met, not raised (choice (a)).** CPU tg (>= 0.25x)
reads 0.259x, CPU pp at 512 (>= 0.30x) reads 0.422x. Item 5's restatement keeps both at the values written and retires
them, as Tier 02D's close did for the GPU rows: no later tier owns CPU kernel work, no further Q4_K candidate is in hand,
and the term that would hold a higher tg row is still the decode kernels' instruction cost at 128-bit integer width.
Tier 14 reports the distances. The README's current-reference table, its end-of-plan rows and the "Restated again
2026-10-11" paragraph are updated, the reference column moves to these sweeps (README rule: the new column in
`TARGET-HISTORY.md`, the 2026-10-11 column's provenance moved there, superseded banners on `20261011T013059Z`,
`20261012T045931Z`, `20261012T071213Z` and `20261012T162344Z`), and `check-plan-thresholds.sh` passes.

**Owner decision 10 (2026-10-27): the interrupted attempt's sweeps are kept, marked not scored (choice (a)),** as the
record of the reference tool's movement between sessions.

**CPU context shift (after the gate).** The one CPU row-1 result left unverified on 2026-10-26: check 10 of
`ModelLiveChecks` on TinyLlama with the handler's limit lowered to 192 positions, run on the CPU backend (CUDA hidden;
the log names the integer-dot kernels, 12 threads) from `juno-master.jar`, built in the same package run as the candidate
jar, its kernel classes byte-identical. Static and continuous both return 200 of 200 tokens over 3 shifts, highest
position 191: **PASS**, about 65 s. Published in
[`20261027T025856Z-cpu-kernels-quality`](../perf-compare/20261027T025856Z-cpu-kernels-quality/INDEX.md). A first try ran
`smoke-context-policy.sh --cpu` on TinyLlama with the limit set to 2,048 through a temporary environment override; it is
not evidence and the override was removed: TinyLlama's handler limit is the 32,768-position KV cap (only Phi-3.5-mini's
is lower), so no request reached a limit and nothing shifted.

**Cross-surface record (closing).**

| # | Surface | Status | Evidence |
|---|---|---|---|
| 1 | CPU inference | PASS | kernels parity-tested (`CpuQ6kKernelParityTest` bit-identical, `CpuQ4kKernelParityTest`, `CpuIntegerGemmUnpackParityTest`, `CpuIntegerGemmTest`, `GqaVectorHeadTest`, `GqaMathPooledTest`); NLL 0.99935x to 1.00209x; `smoke-cpu-kernels.sh` 18 of 18 greedy replies identical to the pre-tier build; context-shift check 10 on the CPU, both schedules; closing gate |
| 2 | CUDA GPU inference | PASS | Part A (run 1): tg and pp 0.969x to 1.013x, greedy equal in 3 of 3 runs per model; `smoke-gpu-residency.sh` 13 PASS with llama-1-30b at partial offload (its CPU layers on these kernels, output identical with the region on and off); `smoke-packed-kquant-matmul.sh` and `smoke-context-policy.sh` on the GPU pass |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE | ROCm's host-side CPU work runs the same kernels and pool, unit-tested; no AMD device here |
| 4 | Static schedule | PASS | `smoke-consistency.sh` (static at `--parallel 2`, CPU and GPU, 54 PASS); `smoke-long-prompt-prefill.sh --cpu` static |
| 5 | Continuous schedule | PASS | `smoke-consistency.sh` continuous; `smoke-long-prompt-prefill.sh --cpu` continuous; continuous prefill chunks run `CpuIntegerGemm` and the pooled attention |
| 6 | Single-node local mode | PASS | every smoke above and the closing gate |
| 7 | Pipeline-parallel cluster | PASS | `ModelLiveRunnerIT` checks 1 to 6 in forked node JVMs on the CPU backend, five models; each node JVM selects its own width at class initialisation |
| 8 | Tensor-parallel cluster | PASS | `ModelLiveRunnerIT` checks 7 and 8 on the CPU backend, five models |
| 9 | LoRA training | PASS | the frozen forward stays on the float-activation kernels (items 1 to 3 do not reach it); the trainable handlers' attention takes the pooled window entry point, bit-identical; Part L training 1.024x (GPU) and 0.999x (CPU) |
| 10 | LoRA playback | PASS | playback composes against the integer kernels; `LoraQ4KPlaybackParityTest` and the LoRA suites pass in the full unit suite on the final build (2,437 tests, 0 failures); Part L playback 0.977x and 0.995x, recall in every run |
| 11 | Vision | PASS | `VisionWidthCpuKernelGuardTest` at B = 741; `smoke-cpu-kernels.sh`'s CPU vision case identical to the pre-tier build; `compare-vision.sh` A/B: CPU lane latency 0.555x, decode 2.104x |
| 12 | OpenAI REST surface | PASS | the smokes run `/v1/chat/completions` (consistency, grammar 19 PASS, tools 14 PASS, long prompt) |
| 13 | Native REST surface | PASS | `smoke-consistency.sh` checks `/v1/inference` against the single-request response |
| 14 | CLI | PASS | no new flag; the width ends the startup `SIMD available` line (`CPU integer kernels: 128-bit lanes (host preferred 128-bit)`); `-Djuno.cpu.integerWidth` other than `128`, `256` or `auto` fails closed (`CpuIntegerWidthTest`) |
| 15 | JVM embedding facade | N/A | no new capability; the width is chosen per JVM at class initialisation, the same in an embedder |

**Docs.** `docs/performance.md`: new section "CPU decode and prefill kernels" (changes, kernel cost, the same-hour
A/B, the pinned closing reading, quality, allocation, what bounds it now), with a dated pointer on the 2026-10-11
section. `CHANGELOG.md`: Session 127. `docs/agent-arch.txt`: the four-row tile, the zero-row padding and the rule that no
production kernel method takes or returns a Vector API type (`CpuKernelVectorEscapeTest`). `docs/howto.md` has carried
the width property and the startup line since 2026-10-13. Juno-native language throughout.

**Markers.** Every adopted or rejected candidate in this record carries `host-specific` or `expected-general`; the one
without (item 2 candidate 3's speedup, 2026-10-12) is marked in place.
