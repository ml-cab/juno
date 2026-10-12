# Tier 02E: CPU decode and prefill kernels (Q6_K vector kernel, Q4_K instruction cost, SIMD width)

Status: not started
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
2.7 cycles per weight and about 36% of decode. The README's restatement says the row returns toward 0.40x only
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
   `JfrSpanBreakdown`. If CPU attention is the term that grows with length, name it in this tier's record
   and say which tier owns it; do not leave it implicit.
5. **Restate the CPU rows from what this tier measures.** After items 1 and 2, restate the README's CPU tg
   row from the measured cycles per weight, as Tier 02C item 5 did, never below 0.25x, and the CPU pp row
   from item 4's sweep, never below 0.25x.

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
| 9 | LoRA training | N/A: the frozen forward uses the float-activation kernels (`CLAUDE.md`), which this tier does not touch; verify it stays so |
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
4. Item 1 (Q6_K), re-measured; then item 2 (Q4_K), one candidate at a time, re-measured after each.
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
  - Single-thread kernel cost at Mistral 7B's FFN shape, pinned: Q6_K **<= 1.5** cycles per weight (about
    2.7 today), Q4_K **<= 1.1** cycles per weight (about 1.5 today; the figure the README's 0.40x derivation needed).
  - Model quality: mean NLL **<= 1.01x** the float-activation kernel's on every sweep model.
  - Juno CPU tg and pp t/s **>= 1.10x** the pre-tier build on Mistral 7B and Qwen2.5-3B (same-hour A/B,
    pinned) and **>= 0.95x** on the other two, so a kernel that wins on one shape and loses on another
    does not close the tier.
  - Vision: `latency_ms` **<= 1.25x** baseline, decode tps **>= 0.80x** baseline. LoRA: train **>= 0.95x**,
    wall-clock playback tps **>= 0.80x**.
  - On an AVX2 host, if one is obtained: the 256-bit Q4_K kernel **>= 1.5x** the 128-bit kernel's
    single-thread throughput. Without one, the row is marked `NEEDS-AVX2-HARDWARE` and the forced-width
    parity tests are the evidence; flag this to the user when step 5 is reached (see
    [`INVENTORY.md`](INVENTORY.md)).
  - Every throughput number is a median of at least three runs with min/max published.

## Models needed

The four sweep models (Q4_K_M; Q6_K tensors occur in every one) and `moondream2-q5_k.llamafile` for the
vision guard. All present. An AVX2 or AVX-512 host for the width threshold is not present.

## Exit criteria

- [ ] Pinned CPU sweeps at `n_prompt=512` and `2048` published before any kernel change, with the 2048
      decomposition, and the README's CPU pp row's reference cell filled from them.
- [ ] Kernel baseline published (cycles per weight, Q4_K and Q6_K, single thread, pinned) before item 1.
- [ ] Q6_K runs a vector integer kernel at decode and prefill width, bit-identical to the scalar kernel it
      replaced.
- [ ] Item 2's candidates measured one at a time and adopted on measurement, each recorded.
- [ ] Integer species chosen from the host's preferred width, 128 bits on this host, forced-width parity
      tests passing; the selected width is printed at startup.
- [ ] The README's CPU tg and CPU pp rows restated from this tier's measurements, never below 0.25x.
- [ ] Every threshold above met on the pinned closing sweeps. A miss keeps the tier open; closing on a miss
      takes an explicit **Owner decision** recorded here with the measured number and the term that held it.
- [ ] Standing throughput and allocation gate met (README, "Test infrastructure"): GPU tg and pp >= 0.95x
      the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
- [ ] Every earlier tier's CPU row-1 correctness result re-verified under the new kernels.
- [ ] Every conclusion carries a `host-specific` or `expected-general` marker (Tier 02C item 4's rule).
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` startup line, `docs/agent-arch.txt`, `docs/performance.md`) updated,
      Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
