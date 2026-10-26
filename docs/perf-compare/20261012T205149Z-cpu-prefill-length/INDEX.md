# CPU prefill length profile and its decomposition (2026-10-12)

**Purpose.** How Juno's CPU prompt processing scales with prompt length, measured before any change to the CPU
decode and prefill kernels, and where a 2048-token CPU prefill spends its time. Until this run every CPU
reference was taken at `n_prompt=128`, so a cost that grows with prompt length could not be seen.

**Pinned: yes.** Owner run, CPU governor `performance`, turbo off (`clock_pinned: true` in all three sweeps).
Three Juno repetitions per row (each after two discarded warmup requests), three reference-tool repetitions,
`--threads 12` against `-t 12`, temperature 0.

**Build.** One build is measured (there is no A/B): Juno `01027ee9`, jar sha256 `b9a0440b3a1b597a`, OpenJDK
25.0.3. Reference tool build `379ac6673`.

## Commands

```bash
# owner run: dist/cpu-prefill-length/run-gate.sh (git-ignored), three sweeps, each:
scripts/performance-tests/compare-llama-cpp.sh --cpu --pin-clocks --n-gen 64 --threads 12 --reps 3 \
    --juno-reps 3 --juno-warmup 2 --jfr 120m --publish --models <models> --n-prompt <512|2048>
#   512:  the four sweep models at their fixed heaps                       -> ../20261012T045931Z/
#   2048: TinyLlama, Qwen2.5-3B, Mistral 7B at their fixed heaps            -> ../20261012T071213Z/
#   2048: Phi-3.5-mini at COMPARE_HEAP=8g (float KV of 32 heads at 2049     -> ../20261012T162344Z/
#         positions doubles to 4096 and needs about 3.2 GB)
# decomposition, from the measured recordings of those runs (juno-perf.jfc):
java scripts/performance-tests/JfrSpanBreakdown.java --top 15 <run>/*-prefill-rep*-measured.jfr
```

`jfr-512-prefill.jsonl` and `jfr-2048-prefill.jsonl` hold the full per-repetition output (prefill and decode
span of each recording; the prefill span is the one read here). The `.jfr` files are not copied.

## 1. Length profile (Juno over the reference tool, prompt processing)

Median of three; every prefill processed exactly the requested token count; every row scorable.

| Model | pp ratio @128 (reference `../20261011T013059Z`) | pp ratio @512 | pp ratio @2048 | Juno pp t/s @512 (min/max) | Juno pp t/s @2048 (min/max) | Ref. tool pp t/s @512 / @2048 |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 0.224x | **0.165x** | **0.086x** | 12.38 (12.26 / 12.51) | 6.10 (6.09 / 6.14) | 75.0 / 70.5 |
| Qwen2.5-3B | 0.230x | 0.186x | 0.122x | 5.14 (5.13 / 5.15) | 3.18 (3.16 / 3.18) | 27.7 / 26.1 |
| Phi-3.5-mini | 0.256x | 0.198x | 0.092x | 3.81 (3.78 / 3.85) | 1.68 (1.65 / 1.68) | 19.2 / 18.2 |
| Mistral 7B | 0.226x | 0.192x | 0.117x | 2.25 (2.23 / 2.26) | 1.29 (1.27 / 1.30) | 11.7 / 11.0 |

The reference tool loses 6% or less of its prompt throughput from 512 to 2048; Juno loses 42% to 56%. The
ratio at 2048 is 36% (Phi-3.5-mini) to 53% (Qwen2.5-3B) of the ratio at 128.

The generation columns of these sweeps are not a reference: generation does not depend on `n_prompt`, and the
TinyLlama 512 sweep's reference-tool generation reading came out low (tg ratio 0.379x against Juno's unchanged
7.81 t/s). The CPU generation reference stays `../20261011T013059Z`.

## 2. Where the time goes

Both lengths run 32-token prefill windows (16 windows at 512, 64 at 2048), so the matmul work per token is the
same at every length. Anything that makes a token dearer at 2048 than at 128 is a term that grows with length.

| Model | ms per prompt token @128 / @512 / @2048 | Share of the 2048 cost above the 128 cost | Attention, share of samples @512 / @2048 | `CpuIntegerGemm`, share of samples @512 / @2048 | Average parked pool threads @512 / @2048 |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 60.5 / 80.8 / 164.0 | 63% | 12.3% / 37.3% | 81.1% / 58.2% | 4.7 / 9.2 |
| Qwen2.5-3B | 161.9 / 194.6 / 314.2 | 48% | 7.1% / 23.4% | 86.3% / 71.3% | 4.1 / 7.6 |
| Phi-3.5-mini | 206.1 / 262.7 / 595.8 | 65% | 10.3% / 39.8% | 86.3% / 57.9% | 4.7 / 9.9 |
| Mistral 7B | 381.7 / 445.4 / 772.8 | 51% | 6.2% / 26.6% | 88.9% / 69.6% | 4.1 / 8.5 |

Attention is `GqaMath.attend` plus the softmax it calls; `CpuIntegerGemm` is its `tile` and `unpack`. Shares are
medians of three repetitions of `jdk.ExecutionSample`. Parked threads are `jdk.ThreadPark` time in the span over
the span's length.

How to read it:

- **The term that grows with length is CPU attention, and it runs on one thread.** In the sweep models' prefill
  window (the LLaMA-family handler, which runs TinyLlama, Qwen2.5-3B and Mistral 7B, and the Phi-3 handler)
  the host attention is a loop over the window's rows on the calling thread
  (`gqaInto` to `GqaMath.attend`, scalar float, every head in turn), while the other eleven pool threads park.
  The parked-thread column doubles from 512 to 2048 for that reason. `expected-general`: a property of the code,
  not of this host.
- **The sample shares understate attention's share of wall time.** `jdk.ExecutionSample` samples running
  threads, so a phase that runs on one thread yields a fraction of the samples a twelve-thread matmul phase
  yields in the same time. The "share above the 128 cost" column is the better reading, and it is a lower bound
  for attention, since attention already costs something at 128: **attention is at least about half of a
  2048-token CPU prefill on every sweep model** (48% to 65%), and 14% to 25% of a 512-token one. The magnitudes
  are `host-specific` (12 threads, 128-bit SIMD).
- **The matmul kernels are still the larger cost at 512.** They are what limits the ratio at 128 and most of
  it at 512, so the kernel work this measurement precedes still applies; at 2048 attention is of the same order.
- GC is not a factor: pause totals in the prefill span are 0 to 9 ms against spans of 42 s to 26 minutes.

## Owner of the attention term

No tier of the plan owned CPU prefill attention when this was measured. Its owner is recorded in the plan's
CPU kernel tier.
