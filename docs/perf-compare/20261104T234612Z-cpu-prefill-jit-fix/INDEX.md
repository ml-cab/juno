# CPU batched matmul without loop-carried receivers: the slow prefill compile reproduced, and gone

**Purpose.** Show that the change to `CpuIntegerGemm`'s four-row kernels removes the slow compile diagnosed in
`20261104T191116Z-cpu-prefill-jit/`, that the outputs are unchanged, and how much the fast path costs. The throughput
gate is the owner's pinned A/B (`dist/cpu-prefill-jit-gate/`); nothing here is pinned.

**The change.** In `fourRows16` and `fourRows32` no Vector API method is called on a vector carried from one loop
iteration to the next. `f0 = f0.add(term)` became `f0 = term.add(f0)`. `f0 = f0.sub(o.mul(x))` became
`f0 = o.mul(x).add(f0)`, with `o` negated once per sub-block. The lanes are read through `zero(F).add(f0)` into a
scratch array instead of `f0.lane(i)`. While C2 parses a loop it types a carried vector only as `FloatVector`, so a call
on one is bound to `Float128Vector` through the JDK's type profile, which every caller shares. A freshly computed vector
is typed exactly. Addition commutes, `a - o*x` is `(-o)*x + a` exactly, and adding zero changes at most the sign of a
zero, which the output's sum absorbs: every output is bit for bit the earlier kernels'.

**Builds.** Baseline Juno `80ec949e` (shaded jar `15083b9c320c1c36`). Candidate: the same tree with the change (jar
`ea700f227c83b911`; kit jar `448867f1d5a028c3`). An earlier form that negated per row instead of per sub-block (jar
`4713925ad8ed6067`, kit jar `dfa50f93a60aebd0`) is reported where it was measured. Host: Xeon E5-1650 v2, 12 threads,
GTX 1080, JDK 25.0.4.1, unpinned.

## The slow compile, on demand (`reproducer.txt`)

In fresh JVMs the old kernels never got the slow compile, under nine JIT regimes. They do get it every time once the
kernels' own lane operations have run hot at 128 bits and then a few thousand times at 64 and 256 bits:

| Fresh JVM, prefill-width windows | Old kernels | Changed kernels |
|---|---|---|
| 22 layers, after the other shapes | 58.4 to 61.7 s, about 8.5 GB per call | 9.9 s, 1,261 bytes per call |
| 4 layers (`CpuGemmProfilePollutionTest`'s child) | 11.0 to 17.8 s, 8.5 to 14.9 GB per call | 1.7 to 1.8 s, 1,288 bytes per call |

`CpuGemmProfilePollutionTest` now runs that check, in a child JVM: under 64 KB per call. It passes in 8.8 s; its child
reads 8,494,463,032 bytes per call on the old kernels. `CpuKernelVectorEscapeTest` checks the rule itself from
`CpuIntegerGemm`'s bytecode. On the old class it finds 48 calls (`add`, `sub` and `lane` on `f0` to `f3` in both
kernels); on the new class it finds none.

Not removed: when the 256-bit kernels themselves run first, they take the Vector API's portable form on this 128-bit
host. The kernels then still allocate 0.8 to 2.1 MB per call (old: 1.0 to 12.9), in the JDK's Java fallbacks, which no
kernel code avoids. The two widths never run in one production JVM.

## The CPU live probe in the kit (`kit-runs.txt`)

| Build | Runs | Check 9, 573-token window | Check 10, chunk 64 |
|---|---|---|---|
| Old (`20261104T191116Z-cpu-prefill-jit/`) | 10 | slow in 6: about 34 s or 91 s; fast 16.5 to 17.1 s | fast 5.0 to 5.1 s; slow 10.3 to 10.5 or 28.2 to 28.6 s |
| First form | 10 | 17.9 to 19.8 s, none slow | 5.2 to 6.0 s |
| Final form | 10 | 17.8 to 21.6 s, none slow | 5.2 to 5.8 s |

If the change did nothing, 20 fast runs out of 20 would have about a 1 in 10^8 chance at the old rate (0.4^20). All
30 runs pass 10 of 10 checks.

## Fast-path cost and outputs (`ab-unpinned.txt`, `reproducer.txt` item 5)

- **Server path, same hour, A B A B, `n_prompt` 512, CPU:**
  - first form: TinyLlama 0.955x, Qwen2.5-3B 1.01x;
  - final form: TinyLlama about 1.03x (candidate 31.73 and 32.67 t/s against 32.68 and 29.71), Qwen2.5-3B about 1.03x
    (10.51 and 12.00 against 11.25 and 10.68).
  - Unpinned spreads are about 5%.
- **Greedy replies:** identical across all four runs of each model, in both sessions.
- **The matmul alone, interleaved:** the final form read 10.46 s against 10.55 s for the old code (median window).
- **The kit runs' layers** read about 7% above the old build's fast runs. Those were taken hours earlier, not
  interleaved, on a host warmed by continuous load, so they do not measure the change.
- **Dropped:** a form keeping the accumulators in a `float[]` round trip read 1.13x the time and was not used.

**How to read it.** The slow compile is the old kernels' calls on loop-carried vectors meeting a type profile that
another vector shape has touched. The change removes those calls, so the reproducer and the kit both stop finding it,
and the outputs are unchanged. The unpinned readings show no fast-path cost beyond their noise. The pinned A/B
(`dist/cpu-prefill-jit-gate/`: pp at 512 and 128, tg, allocation, greedy equality, LoRA) is the gate.

*2026-11-05: the gate is met, owner run, in
[`20261105T040822Z-cpu-prefill-jit-gate`](../20261105T040822Z-cpu-prefill-jit-gate/INDEX.md): pinned CPU pp B/A 0.965
to 1.040 at 512 and 1.004 to 1.037 at 128, tg 1.003 to 1.015, allocation 0.984x to 0.995x, greedy output equal in
every run, LoRA unchanged.*

**Not changed.** The decode kernels, the attention head and the 256-bit forms carry vectors the same way. In the
reproducer the unchanged decode matvec allocated 0.67 to 3.7 MB per call in some JVMs after the same kind of
operations, and nothing in others. No real run has shown it.

## Files

- `kit-runs.txt`: the 20 kit runs on the changed build.
- `ab-unpinned.txt`: the two server-path A/Bs and the greedy check.
- `reproducer.txt`: every fresh-JVM reading behind the tables above.
