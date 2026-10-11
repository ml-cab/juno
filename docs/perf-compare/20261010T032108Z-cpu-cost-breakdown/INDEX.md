# CPU cost breakdown before any CPU fix (2026-10-10)

**Purpose.** Where Juno's CPU inference spends its time, measured before any CPU kernel, allocation or
threading change, so the fixes can be ranked by attributable cost. Three inputs:

1. JFR recordings (`scripts/performance-tests/juno-perf.jfc`) of steady-state CPU decode (64 tokens) and a
   128-token CPU prefill on Mistral 7B and Phi-3.5-mini (both Q4_K_M), three measured repetitions each, split
   into their prefill and decode spans by `scripts/performance-tests/JfrSpanBreakdown.java`.
2. A decode-width kernel microbenchmark (`scripts/performance-tests/cpu-decode-kernel-variants.sh`): the
   shipped single-accumulator Q4_K and Q6_K kernels against the same arithmetic with 4 and 8 independent
   float accumulators, at Mistral 7B's FFN shapes. This measures how much of a decode matmul is the
   loop-carried dependency of accumulating a whole row into one float.
3. Juno's attained weight bandwidth against the host's measured 42.4 GB/s
   (`../20261009T220314Z-host-read-bandwidth/`).

**Pinned: no.** CPU governor `schedutil`, turbo on. This is a breakdown, not a gate. Shares and
per-token ratios are what it is read for; absolute t/s are within the 15% noise floor of a pinned run at
best.

**Build.** Juno `aabc42cb` plus uncommitted changes (a test-only visibility change in
`Phi2TransformerHandler.sgemmQuantBatch`, no behaviour change), jar sha256 `1dde81d172ee5b41`, OpenJDK
25.0.3. One build is measured, so there is no second jar. The microbenchmark ran from `node/target/classes`
of the same tree. Reference tool build `379ac6673`, CPU, `-t 12`.

## Commands

```bash
# 1. JFR sweep (run 20261009T230709Z; its own INDEX is sweep/INDEX-sweep.md)
scripts/performance-tests/compare-llama-cpp.sh --cpu --models mistral-7b,Phi-3.5-mini --n-prompt 128 \
    --n-gen 64 --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2 --jfr 30m --no-publish --out <run>
java scripts/performance-tests/JfrSpanBreakdown.java --top 10 <run>/*-measured.jfr > jfr-span-breakdown.jsonl

# 2. Kernel variants, three runs (each: 12 threads, then 1 thread)
scripts/performance-tests/cpu-decode-kernel-variants.sh --out <dir>
```

The `.jfr` files themselves (about 1.8 MB each) are not copied; every figure below is in
`jfr-span-breakdown.jsonl` or the sweep's JSON.

## 1. Where the time goes (JFR, `jdk.ExecutionSample`, share of samples in the span)

Median of three repetitions; per-repetition values and the top 10 frames are in `jfr-span-breakdown.jsonl`.

| Model, lane | ms per token | Q4_K kernel | Q6_K kernel (per row) | Q5_K kernel | `f16ToF32` | All quantized matmul kernels | Attention | Fork/join frames |
|---|---|---|---|---|---|---|---|---|
| Mistral 7B, decode | 1,972 (1,912 to 2,177) | 75.7% (decode `matVecQ4KrawInto`) | 21.3% | - | 2.1% | **99.2%** | 0.2% | 0.4% |
| Mistral 7B, prefill (windows of 32) | 1,130 (1,129 to 1,175) | 59.4% (`sgemmQ4KWeightStationary`) | 38.7% | - | 1.0% | **99.1%** | 0.5% | 0.1% |
| Phi-3.5-mini, decode | 1,083 (1,082 to 1,084) | 47.8% | 18.9% | 29.4% | 1.3% (42.9% in one rep) | **98.9%** | 0.3% | 0.5% |
| Phi-3.5-mini, prefill | 1,063 (1,062 to 1,066) | 25.8% (decode kernel) | 17.9% | 11.8% | 43.3% | **98.8%** | 0.3% | 0.6% |

How to read it:

- **CPU inference is the quantized matmul kernels and nothing else.** Attention, norms, RoPE, sampling and
  the fork/join machinery together are about 1%.
- **`f16ToF32`'s share is JIT attribution, not cost.** Phi-3.5-mini decode repetition 1 puts 42.9% of
  samples in it and repetitions 2 and 3 put 1.3%, at the same 1,083 ms per token. Read the kernel and
  its scale conversion as one cost.
- **Mistral 7B prefill is batched, but only 1.75x cheaper per token than decode** (1,130 against 1,972 ms).
  Its Q6_K matrices have no batched kernel and run one row-pass per prompt token, which costs 38.7% of
  prefill.
- **Phi-3.5-mini prefill is not batched on the CPU at all.** It costs the same per token as decode (1,063
  against 1,083 ms). `Phi3TransformerHandler.sgemmFusedInto` falls back to one `matVecInto` per batch row
  when no device matrix is present, so every prompt token reads every weight. This is why Phi-3.5-mini
  binds the CPU pp milestone.

## 2. The dependency chain (kernel variants, decode width, synthetic weights)

Median ms per call over five repetitions. The ratio is against `a-single`, a verbatim copy of the shipped
kernel. Three runs; `production` (the shipped entry point) agrees with `a-single` to within 2.4% in every
run, so the copy is faithful. Weights are cycled through 96 MB or more of copies so that every call reads
from DRAM, as a real decode does.

| Kernel, shape | Threads | a-single ms (runs 1/2/3) | b-split4 / a-single | b-split8 / a-single |
|---|---|---|---|---|
| Q4_K 14336x4096 | 12 | 13.59 / 13.75 / 13.61 | 1.003 / 0.992 / 1.000 | 1.139 / 1.145 / 1.109 |
| Q6_K 4096x14336 | 12 | 23.48 / 23.02 / 25.04 | 1.009 / 1.017 / 0.946 | **0.837 / 0.839 / 0.782** |
| Q4_K 14336x4096 | 1 | 80.33 / 80.55 / 79.63 | 1.025 / 1.019 / 1.028 | 1.034 / 1.027 / 1.032 |
| Q6_K 4096x14336 | 1 | 128.08 / 126.04 / 133.13 | 1.000 / 1.010 / 0.990 | **0.812 / 0.817 / 0.823** |

How to read it:

- **Splitting the float accumulator does not speed up the Q4_K kernel** at either thread count (0.99x to
  1.03x with 4 accumulators, 1.03x to 1.15x slower with 8). It speeds up Q6_K by 16% to 22% with 8.
- **Share of decode time the dependency chain accounts for: about 3.5% on Mistral 7B** (Q6_K's 21.3% of
  decode samples, times its 16.3% median saving; Q4_K's 75.7% saves nothing).
- **The kernel is bound by per-weight instruction count, not by the add chain or by memory.** One thread
  reads 0.41 GB/s of Q4_K weights, about 0.82 billion weights a second, or roughly 4 to 5 core cycles per
  weight. Twelve threads reach 2.4 GB/s, 5.9x one thread on 6 physical cores, which is 5.7% of the host's
  42.4 GB/s. The host is nowhere near its memory limit, and the threads scale.
- `host-specific`: no FMA and no 256-bit integer instructions here (AVX only). On an AVX2/FMA host, re-measure
  all variants; a float kernel that C2 compiles with FMA may change the split-accumulator result.

Note: the single-thread headers of runs 1 and 2 originally printed `threads=2`. With a common-pool
parallelism of 0, every task runs on the calling thread, but `getCommonPoolParallelism()` still reports 1.
The work was single-threaded (verified separately), and the label was fixed in
`CpuDecodeKernelVariantMicrobench` before run 3.

## 3. Allocation, GC pauses, lock and park time (JFR, inside each span)

Allocation is from the `jdk.ThreadAllocationStatistics` cumulative counters (period 500 ms), summed over
threads. Median and range of three repetitions.

| Model, lane | Allocated per token | Allocation rate | GC pauses in span | Pause total | Monitor enters (>= 10 ms) | Park events (>= 10 ms) | Park total |
|---|---|---|---|---|---|---|---|
| Mistral 7B, decode | **20.8 MB** (20.3 to 20.9) | 10.5 MB/s | 0 | 0 ms | 0 | 282 (246 to 1,209) | 342 s over 126 s |
| Mistral 7B, prefill | 261.5 MB | 231.5 MB/s | 3 | 11.1 ms (10.7 to 12.7) over 143 s | 0 | 4,086 | 479 s over 143 s |
| Phi-3.5-mini, decode | 35.2 MB | 32.5 MB/s | 3 | 11.3 ms over 69 s | 0 | 233 | 168 s over 69 s |
| Phi-3.5-mini, prefill | 66.5 MB | 62.6 MB/s | 0 | 0 ms | 0 | 1,487 | 379 s over 135 s |

How to read it:

- **Allocation costs no measurable time today.** GC pauses are 0% to 0.016% of a span, and no monitor is
  ever contended for 10 ms or more.
- **Mistral 7B prefill allocates 12.6x more per token than decode,** and 99% of the sampled weight is the
  per-row dequantization scratch inside `sgemmQ4KWeightStationary`'s parallel lambda.
- **Decode's largest sampled site is `DenseKvTensor.allocate`, which is KV-cache doubling** (a one-off per
  growth step, not per-token garbage). The rest is the allocating `matVecQ4Kraw`, `ffn`, `rmsNorm` and
  `add`.
- **Park totals sum every thread, so idle workers park longer than the run takes** (the README's
  withdrawn rule says why this does not measure lost time). Monitor time is zero.
- Note: `jdk.ExecutionSample` records only a few Java threads per sampling period, so the number of sampled
  threads is not a utilisation figure. The thread scaling above comes from the microbenchmark.

**The baseline for the tier's allocation threshold** (>= 50% drop in steady-state CPU decode allocation)
is **20.8 MB per token** on Mistral 7B (10.5 MB/s).

## 4. Attained weight bandwidth (file size as bytes per token, as in the step-2 publication)

| Model | Juno tg (this sweep) | Juno attained | Share of 42.4 GB/s | Reference tool tg (this sweep) | Its share | tg ratio | pp ratio at 128 |
|---|---|---|---|---|---|---|---|
| Mistral 7B (4.368 GB) | 0.524 t/s | 2.29 GB/s | **5.4%** | 5.28 t/s | 54% | 0.099x | 0.077x |
| Phi-3.5-mini (2.393 GB) | 0.937 t/s | 2.24 GB/s | **5.3%** | 10.65 t/s | 60% | 0.088x | 0.048x |

The tier's attained-bandwidth gate (>= 25% on Mistral 7B) is 10.6 GB/s, about 2.43 t/s, which is 4.6x
today's decode.

## Ranking for implementation (step 5 order)

1. **Integer and vectorized CPU kernels (item 1): about 99% of decode and prefill time.** Splitting the
   float accumulator recovers about 3.5%, so the gain has to come from fewer instructions per weight:
   integer dots over a once-quantized activation, vectorized. The prefill half has two structural gaps
   the kernel alone does not close: Q6_K has no batched kernel (38.7% of Mistral 7B prefill), and Phi-3's
   CPU prefill is not batched at all.
2. **Threading (item 3): not separable at today's kernel speed.** Fork/join frames are 0.1% to 0.6% of
   samples, and the kernels scale 5.3x to 5.9x from 1 to 12 threads. Per-dispatch cost is fixed, about
   225 matmul dispatches per token, so its share rises as item 1 shortens each matmul. Re-measure after
   item 1.
3. **Allocation (item 2): no measurable throughput cost** (GC 0% to 0.016% of a span). It is still owed
   for its own threshold (50% less decode allocation), and the prefill scratch is the largest single site.
