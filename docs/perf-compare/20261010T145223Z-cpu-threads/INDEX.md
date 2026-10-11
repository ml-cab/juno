# CPU matmul threading: common-pool streams against a persistent worker pool (2026-10-10)

**Purpose.** Whether replacing the fork-join common pool's parallel streams with a persistent worker pool speeds
the CPU matmuls now that the integer kernels are about 2.5x faster, and a CPU sweep of the shipped choice (the
worker pool, sized by the new `--threads` flag) against the previous reading on the common pool. The sweep also
confirms that the harness's switch from the common-pool property to `-Djuno.threads` leaves Juno's hot-path thread
count unchanged.

**Pinned: no.** CPU governor `schedutil`, turbo on. Read against the 15% noise floor; the CPU closing gate is the
owner's pinned run.

**Build.** Juno `aabc42cb` plus the uncommitted CPU work (integer kernels, then `CpuThreads`/`CpuWorkerPool` and
`--threads`), jar sha256 `9134cb6b4a71588a`, OpenJDK 25.0.3. The previous reading
(`../20261010T085329Z-cpu-integer-kernels/`, jar `91ca04a9caa4f403`) is the same code with the integer kernels on
the common pool. Reference tool build `379ac6673`, CPU, `-t 12`.

## Commands

```bash
scripts/performance-tests/cpu-dispatch-microbench.sh --skip-build --out <dir>   # dispatch/run*.md
scripts/performance-tests/compare-llama-cpp.sh --cpu \
    --models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b,Phi-3.5-mini,mistral-7b --n-prompt 128 --n-gen 64 \
    --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2 --jfr 30m --no-publish --out <run>
```

## 1. Dispatch microbenchmark (`dispatch/`), 12 threads each side, median of five

Common pool: parallelism 11 plus the calling thread. Worker pool: 11 workers plus the calling thread. Integer
kernels on synthetic weights; "decode layer" is one layer's seven projections back to back, as a decode step
issues them.

| Case | Static partition / common (run 1, run 2) | Chunked (shipped) / common (run 2, run 3) |
|---|---|---|
| decode Q4_K 256x2048 | 1.023, 1.141 | 1.329, 1.393 |
| decode Q4_K 2048x2048 | 1.288, 1.225 | 1.032, 1.031 |
| decode Q6_K 2048x5632 | 1.170, 1.098 | 0.962, 0.969 |
| decode Q4_K 1024x4096 | 1.270, 1.239 | 1.017, 1.057 |
| decode Q4_K 14336x4096 | 1.112, 1.113 | 0.982, 0.988 |
| decode Q6_K 4096x14336 | 1.118, 1.097 | 0.982, 0.984 |
| decode layer, TinyLlama shapes | 1.193, 1.146 | 1.007, 0.974 |
| decode layer, Mistral 7B shapes | 1.136, 1.104 | 0.981, 1.004 |
| prefill B=741, Q4_K 2048x2048 | 1.015, 1.053 | 1.007, 1.000 |
| prefill B=32, Q4_K 4096x4096 | 1.033, 1.044 | 1.001, 0.992 |

How to read it:

- **Static partitioning loses** (1.10x to 1.29x at decode). With 12 threads on 6 cores, two threads share each
  core, so some ranges finish later, and a barrier waits for the slowest. `host-specific` in degree (SMT, core count).
- **Chunked claiming from a shared counter is at parity** with the common pool on whole decode layers (0.97x to
  1.01x) and prefill (0.99x to 1.01x). A single 256-row matmul costs about 30 microseconds more, which does not
  show at layer level. `expected-general`: once the kernels are this fast, dispatch is still not a measurable
  share of a decode step.

## 2. Sweep of the shipped choice against the previous reading (Juno t/s, median of three)

| Model | pp, common pool (previous) | pp, worker pool | tg, common pool (previous) | tg, worker pool | pp ratio now | tg ratio now |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 16.81 | 16.26 (-3.3%) | 7.75 | 7.59 (-2.1%) | 0.263x | 0.411x |
| Qwen2.5-3B | 6.24 | 6.17 (-1.2%) | 2.58 | 2.54 (-1.5%) | 0.238x | 0.280x |
| Phi-3.5-mini | 4.86 | 4.87 (+0.1%) | 2.22 | 2.24 (+0.7%) | 0.263x | 0.303x |
| Mistral 7B | 2.65 | 2.59 (-2.0%) | 1.30 | 1.26 (-2.8%) | 0.226x | 0.414x |

How to read it:

- Every Juno difference is inside 3.3%, below this host's noise floor for unpinned sweeps in different sessions;
  the average is about 1.7% lower. The pinned closing gate decides whether it is real.
- The tg ratios here are higher than in the previous reading because the reference tool's own generation read
  lower this session (for example Mistral 7B 3.04 t/s against 5.15 before). That is the reference tool's
  run-to-run variation, not a Juno change; compare Juno t/s across the two readings, not ratios.
- **Thread count unchanged by the harness switch:** `host.json` records `juno_threads` 12 and the JVM flags
  `-Djuno.threads=12`; every one of the 24 engine runs logged `CPU matmul threads: 12`. Before the switch, Juno's
  kernels also ran on 12 threads (common pool parallelism 11 plus the calling thread).

## Choice

The worker pool is retained on parity. Neither dispatch wins on throughput, so the choice falls to the flag's
requirement. `--threads` must govern the hot path in every JVM, including an embedder's and a cluster node's.
The common pool's size is fixed when the JVM starts and cannot be changed later; a dedicated pool can be resized at
any time. `-Djuno.simd.pool.size`, which sized a pool the hot path never used, is removed and refused.
