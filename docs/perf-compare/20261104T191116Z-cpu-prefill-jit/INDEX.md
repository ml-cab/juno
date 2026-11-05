# CPU live check's slow prefill windows: the JIT sometimes compiles the batched integer matmul with boxed vectors

**Purpose.** Find why `cpu.live.tinyllama`'s batched prefill windows (check 9's 573-token window, check 10's chunk-64
windows) read 2x to 5.5x their usual time in most kit runs since 2026-11-03, while the same window outside the kit read
its usual time (`20261103T205950Z-kit-quick-estimate/`).

**Pinned:** no. This is a diagnosis; nothing here is a throughput gate.

**Build.** Juno `80ec949e` (clean tree), kit jar SHA-256 `bba50dfb33c23ca5`, the jar the reference run
(`20261104T173942Z-kit-reference/`) built. Host as there: Xeon E5-1650 v2, 12 threads, 128-bit integer vectors, kernel
clock source `hpet`. JDK 25.0.4.1 (`/usr/lib/jvm/java-25-openjdk-amd64`). Transparent huge pages: `madvise` (Juno does
not request them).

**Commands.**

```bash
./juno-kit.sh --probes cpu.live.tinyllama --yes --models-dir models --skip-build   # each kit run below
java <the probe's command line, copied from its log> >> exact.log 2>&1 < /dev/null  # the shell runs below
jcmd <probe pid> Compiler.codelist; jcmd <probe pid> Thread.print                  # 8 s into the window
jcmd <probe pid> JFR.start name=w settings=profile duration=30s filename=run.jfr   # as the window opens
```

## Runs

| Run | Setup | Check 9 window, 573 tokens | Check 10 chunk-64 window | Probe |
|---|---|---|---|---|
| 2026-10-28 | kit | 16.5 s | 5.1 s | 243 s |
| 2026-11-03 `quick` | kit | **92.7 s** | **28.6 s** | 451 s |
| 2026-11-03 re-run | kit | **89.7 s** | **28.2 s** | |
| `20261104T043255Z` (reference run) | kit | **34.1 s** | **10.5 s** | 294 s |
| `20261104T183707Z` | kit, processes sampled every 2 s | **33.7 s** | **10.3 s** | 286 s |
| `20261104T184954Z` | kit, `jcmd` 10 s in | 16.6 s | 5.0 s | 242 s |
| `20261104T185917Z` | kit, `jcmd` 8 s in | 16.7 s | 5.0 s | 242 s |
| `20261104T190339Z` | kit, `jcmd` 8 s in | **33.2 s** | **10.4 s** | 289 s |
| `20261104T191116Z` | kit, JFR from the window's start | **90.8 s** | **28.3 s** | 435 s |
| `20261104T191844Z` | kit, JFR from the window's start | 17.1 s | 5.0 s | 244 s |
| 2026-11-03, five runs | outside the kit, a driver calling the same method | 18.0 to 18.8 s | | |
| 2026-11-04, two runs | shell, the probe's exact command line | 17.0 s, 17.4 s | 5.0 s, 5.0 s | |

Slow in 6 of 10 kit runs, in two modes (about 34 s and about 91 s); fast in all 7 runs outside the kit. The
one-token-at-a-time leg of check 9 (572 decode steps) reads 59.1 to 60.5 s in every 2026-11-04 run, slow or fast: the
decode kernels are not affected, only the batched (prefill-width) matmul.

## What differs between a slow and a fast run

- **By layer** (`layers.txt`): in the 34 s mode ten layers take about 2.3 s and the other twelve about 0.63 s, the same
  ten in every such run. They are exactly the layers whose `ffn_down` and `attn_v` are Q6_K in TinyLlama Q4_K_M (1, 2,
  5, 8, 9, 10, 13, 16, 19 and 21, from the file's tensor table). Fast runs take 0.6 to 0.8 s on every layer. In the 91 s mode every
  layer takes 3.5 to 4.5 s. So the 34 s mode is one kernel slow (`fourRows16`, Q6_K and Q8_0) and the 91 s mode is both
  (`fourRows32`, Q4_K and Q5_K, as well).
- **No other process.** No forked node JVM survives check 8; during the window the probe JVM is the only busy process
  (about 11 of 12 threads), other processes under 5%. Nothing attaches to the probe JVM (the editor's `jcmd` polls
  target its own language server only).
- **Same code, flags, threads and lane width.** The shell runs use the probe's exact command line, and both read
  `CPU matmul threads: 12` and the 128-bit kernels.
- **JFR** (`jfr-slow-vs-fast.txt`):
  - slow: about 506 GB allocated in 30 s (JFR's sampled weight), almost all `float[]` and `Float128Vector`, about 290
    collections, and 90% of CPU samples inside `FloatVector.lanewiseTemplate` inlined into the compiled `fourRows32`,
    at its four accumulator updates (`f0 = f0.add(((FloatVector) ... .convertShape(I2F, F, 0)).mul(...))`, lines 407
    to 431);
  - fast: 13.6 GB, under 1/30 of that, the largest share from `fourRows16` while still interpreted, before its
    compile; 56% of samples in compiled `fourRows32` itself;
  - the C2 code of `fourRows16`: **17,912 bytes slow, 6,640 bytes fast**, with about 4,900 bytes inlined in both;
    `fourRows32` is 17,512 bytes in the slow run (the fast recording has no `fourRows32` compile: it finished before the
    recording started or took under the 100 ms the settings record);
  - only the slow run has `class_check` traps in `ByteVector.lanewiseShiftTemplate` (22) and the Vector API's Java
    fallback `ByteVector.uOpTemplate` running (10 traps).

**How to read it.** In a slow JVM, C2 compiles the prefill kernels so that the float vector accumulators are boxed:
every vector add, subtract and multiply allocates a `Float128Vector` and its `float[]`. The matmul is then 3x to 6x
slower and the collector runs about ten times a second. The engine's code is the same in both runs; which compile a JVM
gets varies from run to run. The checks still pass (the results are the same; only the time differs). This is a
fragility of `CpuIntegerGemm`'s kernels under the JIT, not a defect in the kit, and nothing in it is specific to the
probe: any JVM that runs CPU prefill at window width can get the slow compile. It has been seen only in the kit's probe
JVM so far (6 of 10 runs there, 0 of 7 outside). Why the kit's runs get it more often is not found; the probe's
command line, environment, class path and flags are the same, so the remaining difference is timing. The server-side
CPU prefill readings in the reference run (throughput at 128 to 2048, `long-prompt`, `prefill-batch`) are in line with
earlier runs and their repetitions' spreads are small, so none of those JVMs got it there.

**Not tried.** A deterministic reproducer (for example warming the shared Vector API methods with a second vector shape
before the kernels compile), and any change to the kernels. Both are engine work on the CPU prefill hot path.

*2026-11-04, later: both done (owner decision (a)), in `20261104T234612Z-cpu-prefill-jit-fix/`. The reproducer is
exactly that: the kernels' lane operations at 64 and 256 bits after 128 makes the old kernels allocate about 8.5 GB per
call in every fresh JVM. The cause is the kernels' calls on loop-carried vectors, bound through that shared type
profile; the changed kernels make none and allocate nothing.*

## Files

- `layers.txt`: per-layer times of check 9's window in the slow modes, a fast kit run and the runs outside the kit.
- `jfr-slow-vs-fast.txt`: the two JFR recordings summarized (event counts, allocation by class and by Juno frame,
  execution samples, the kernels' C2 compiles, deoptimizations). The recordings themselves (9.6 and 1.1 MB) are not
  published.
