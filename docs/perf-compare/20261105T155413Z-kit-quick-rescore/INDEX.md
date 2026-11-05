# Kit `quick` suite on the reference host, re-scored: first estimate against actual time, on the corrected catalog

**Purpose.** Re-score the kit's first time estimate for the `quick` suite on the reference host. The first reading,
[`20261103T205950Z-kit-quick-estimate`](../20261103T205950Z-kit-quick-estimate/INDEX.md) (0.741x), was taken before
three catalog costs were corrected and while `cpu.live.tinyllama` ran about 200 s long. That probe's extra time was
the JIT's slow compile of the CPU batched matmul
([`20261104T191116Z-cpu-prefill-jit`](../20261104T191116Z-cpu-prefill-jit/INDEX.md)), since removed
([`20261104T234612Z-cpu-prefill-jit-fix`](../20261104T234612Z-cpu-prefill-jit-fix/INDEX.md), pinned gate met in
[`20261105T040822Z-cpu-prefill-jit-gate`](../20261105T040822Z-cpu-prefill-jit-gate/INDEX.md)).

**Pinned:** no. Nothing here is a throughput gate; the reading is the kit's estimate of its own time.

**Build.** Juno `80ec949e` plus the uncommitted `CpuIntegerGemm` change (the kit reports "modified tree"). Kit jar
SHA-256 `448867f1d5a028c3`, built from that tree, newer than every `src/main` file; the catalog is `80ec949e`'s (the
corrected costs). Host: Xeon E5-1650 v2 (6 cores, 12 threads, 128-bit integer vectors), GTX 1080, OpenJDK 25.0.4.1,
kernel clock source `hpet`.

**Command.** `./juno-kit.sh --suite quick --yes --models-dir models --skip-build`, 2026-11-05 15:54:13Z to 16:36:39Z.
`SUMMARY.txt` is the kit's summary, `estimate.json` its first estimate, `probe-times.json` each probe's wall time,
attempts and the other processes' CPU share (from the zip's `checkpoint.json`).

## Estimate against actual

| | Seconds |
|---|---|
| Calibration scale | 1.0068 (28 s, 3 attempts, NOISY: other processes used 5% to 8% of the CPU) |
| First estimate, low / point / high (shown as "32 min to 45 min") | 1904 / **2240** / 2688 |
| Actual, from the estimate to the zip (start plus the 28 s calibration, to the finish) | **2518** |
| Point over actual | **0.890** |

The exit criterion's band is 0.65x to 1.35x: **met**, and the actual time is inside the estimate's range. The estimate
was refined during the run: after the re-run probe below it read "about 41 min to 58 min left" at 12:05 elapsed, and
it closed on the actual from there.

## Per probe

All 18 probes PASS or DONE, three of them NOISY (re-run by the noise rule); `cpu.kernel-width` not runnable (128-bit
JVM). Against the 2026-11-03 run, every probe took the same time within 8% (calibration aside, before the
estimate), except these:

| Probe | Catalog cost (s) | 2026-11-03 (s) | This run (s) | Why |
|---|---|---|---|---|
| `cpu.live.tinyllama` | 250 | 451 | **249** | The slow compile is gone: check 9's 573-token batched window read 17.6 s (old: about 34 s or 91 s in 6 of 10 kit runs) |
| `cpu.throughput.tinyllama.512` | 130 | 127 | **415** | NOISY, 3 attempts: other processes used 8% of the CPU in each |
| `gpu.throughput.tinyllama.2048` | 20 | 16 | **50** | NOISY, 3 attempts: other processes used 6% to 9% |

The two re-run probes account for about 310 s of the actual; without them it would have been about 2208 s, 1.01x the
point. The other processes' CPU share was higher than on 2026-11-03 throughout (median 5.8% per probe, against about
2% then); nothing else was started on the host, and the source was not identified (the editor and its extensions
were open). The noise rule's re-runs are part of the kit's design, so they count in the actual.

Two catalog costs disagree with both runs and offset each other: `gpu.residency.tinyllama` 20 s against 46 and 48 s,
and `gpu.long-prompt.tinyllama` 90 s against 44 and 44 s (net -18 s). Not changed; recorded.

**How to read it.** On the corrected catalog and the changed kernels, the first estimate reads 0.890x the actual on this
host, inside the 0.65x to 1.35x band, and about 1.0x once the noise rule's re-runs are taken out. `host-specific`.

## Files

- `SUMMARY.txt`: the kit's summary of the run (scrubbed by the kit; no paths).
- `estimate.json`: the first estimate and the calibration scale.
- `probe-times.json`: per probe, status, wall time, attempts, other processes' CPU share.
- The results zip itself is not published.
