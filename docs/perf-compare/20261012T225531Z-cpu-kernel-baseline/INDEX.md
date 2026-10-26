# CPU decode kernel baseline: cycles per weight (2026-10-12)

**Purpose.** The instruction cost per weight of the shipped Q4_K and Q6_K CPU decode kernels, measured before
any change to them, so later kernel work is read against a pinned number. These two kernels are about 93% of
Mistral 7B CPU decode (`../20261010T085329Z-cpu-integer-kernels/`).

**Pinned: yes.** Owner run, CPU governor `performance`, turbo off: every core at the Xeon E5-1650 v2's 3.50 GHz
base clock (`scaling_max_freq` 3500000 kHz; mean core clock after each run 3493 to 3495 MHz, `clock-state.txt`).

**Build.** HEAD `01027ee9`, no engine change, `node/target/classes` (class-file hash `1d8e89c50bfa406d` over
`CpuIntegerKernels` and the benchmark class), OpenJDK 25.0.3, `--add-modules jdk.incubator.vector`.

## Command

```bash
# three times, pinned (owner run: dist/cpu-kernel-baseline/run-gate.sh, git-ignored)
scripts/performance-tests/cpu-decode-kernel-variants.sh --skip-build --out <runN>
```

Each run times every variant at Mistral 7B's FFN shapes, Q4_K 14336x4096 and Q6_K 4096x14336 (58,720,256 weights
each), at 12 threads and then at 1, five timed repetitions per variant, weights cycled through 96 MB or more of
copies so every call reads DRAM. `production` is the shipped path (`LlamaTransformerHandler.matVec`, which takes
`CpuIntegerKernels.q4kRow` / `q6kRow`); the other variants are the frozen candidates from the integer-kernel
work. The per-run reports are `runN-threads1.md` and `runN-threads12.md`.

**Cycles per weight** = median ms per call x 3.5e6 cycles per ms / 58,720,256 weights; at 12 threads, times the
6 physical cores (cycles per weight per core).

## Results (median of three runs; ms per call per run in brackets)

| Kernel | Variant | 1 thread: ms | 1 thread: cycles/weight | 12 threads: ms | 12 threads: cycles/weight per core |
|---|---|---|---|---|---|
| Q4_K | **production** (shipped vector) | 25.42 (25.42 / 25.10 / 25.59) | **1.52** | 4.25 (4.25 / 4.25 / 4.16) | 1.52 |
| Q4_K | d2-int-vector-lanes | 22.85 (22.57 / 22.85 / 22.93) | 1.36 | 4.21 (4.21 / 4.08 / 4.21) | 1.51 |
| Q4_K | c-int-scalar | 57.82 | 3.45 | 9.66 | 3.46 |
| Q6_K | **production** (shipped scalar) | 90.75 (90.75 / 90.16 / 90.98) | **5.41** | 14.65 (14.78 / 14.63 / 14.65) | 5.24 |
| Q6_K | d2-int-vector-lanes | 78.03 (78.27 / 77.51 / 78.03) | 4.65 | 20.75 (20.68 / 20.75 / 20.87) | 7.42 |
| Q6_K | c-int-scalar | 91.75 | 5.47 | 14.90 | 5.33 |

How to read it:

- **Q4_K costs about 1.5 cycles per weight and Q6_K about 5.4**, on one thread and per core at 12 threads alike.
  Q6_K is 3.6 times the cost per weight of Q4_K. At Mistral 7B's shapes a Q6_K matrix call takes 3.5 times as
  long as a Q4_K one of the same size.
- **A faster single-thread kernel is not a faster decode kernel on this host.** The earlier Q6_K vector form is
  14% faster than the shipped scalar kernel on one thread and 42% slower at 12 threads, where two hyperthreads
  share each core's vector execution units and caches (each keeps its own registers). The Q4_K vector form gains
  10% on one thread and nothing at 12. Any candidate is therefore read at both widths, and decode runs at 12.
  *Added the same day: that Q6_K vector form passes vectors to helper methods, which made C2 box them in the
  vector kernel written after this baseline; its allocation, not the hyperthread sharing, is the likely cause of
  its 12-thread loss (not verified).* `host-specific`
  (AVX, no AVX2, 16 vector registers, two threads per core).
- Run-to-run spread is under 2% for every row shown.
