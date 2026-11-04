# Kit `quick` suite on the reference host: first estimate against actual time; CPU live-check prefill isolated

**Purpose.** Score the kit's time estimate for the `quick` suite on the reference host, the first full `quick` run
since every quick probe exists. Then explain the two probes that ran far from their reference cost.

**Pinned:** no. Nothing here is a throughput gate. The kit compares inside one run, and this reading is about its own
time estimate.

**Build.** Juno `a1fe71668a5e` with an uncommitted `.gitignore` line, so the kit reports "modified tree". Kit jar
SHA-256 `8ef5098cf3034632` (from the manifest), built 2026-10-31 and newer than every `src/main` file. Host:
Xeon E5-1650 v2 (6 cores, 12 threads, 128-bit integer vectors), GTX 1080, kernel clock source `hpet` (TSC rejected
at boot).

**Commands.**

```bash
./juno-kit.sh --suite quick --yes --models-dir models --skip-build          # 20261103T205950Z, SUMMARY-quick.txt
./juno-kit.sh --probes cpu.live.tinyllama --yes --models-dir models --skip-build   # 20261103T214211Z, SUMMARY-live-rerun.txt
java <the probe's JVM flags> -cp juno-kit-0.1.2-shaded.jar PrefillCheck.java <tinyllama.gguf>   # isolation runs below
```

## Estimate against actual

| | Seconds |
|---|---|
| Calibration scale | 1.0255 (18 s, 2 attempts; attempt 1 re-run: other processes used 6% of the CPU) |
| First estimate, low / point / high | 1478 / **1738** / 2086 |
| Actual, from the estimate to the zip (15:00:08 to 15:39:13 local) | **2345** |
| Point over actual | **0.741** |

The exit criterion's band is 0.65x to 1.35x. The reading falls inside it, but the actual time is above the range's
top. Most of the miss is in two places below: three catalog costs that disagree with every measurement, and one
probe that is slower inside the kit than outside it.

## Per probe: actual against reference cost x scale

All 18 probes PASS or DONE; `cpu.kernel-width` not runnable (128-bit JVM).

| Probe | Reference cost (s) | Actual (s) | Other processes' CPU | Earlier measurement |
|---|---|---|---|---|
| `cpu.live.tinyllama` | 250 | **451** | 9.1% | 243 s, 2026-10-28 (`20261028T173815Z-quick`, 2.6%) |
| `long-prompt.tinyllama` | 240 | **626** | 2.1% | 641 s, 2026-10-28 (step 5 part 1) |
| `quality.tinyllama-1.1b` | 120 | **458** | 2.1% | 468 s, 2026-10-28 (step 5 part 1) |
| `grammar.tinyllama` | 300 | **97** | 2.1% | 100 s, 2026-10-28 (step 5 part 1) |
| every other quick probe (14) | 1 to 130 each | within about 25% of its cost | 2.2% | |

**Catalog costs corrected after this run:** `long-prompt.tinyllama` 630 s, `quality.tinyllama-1.1b` 460 s and
`grammar.tinyllama` 100 s. These are the measured times, twice each, five days apart; the old values matched no
measurement. Rebuilt kit jar `df00680b5bbc0f29`; `juno-kit` tests 116, 0 failures. `--plan` now reads `quick` 32 to 45 min (was 24 to 34),
`cpu` 636 to 898, `gpu` 57 to 81, `all` 693 to 979. With the corrected costs this run's point estimate would have been
about 2270 s, 0.97x the actual. The criterion is re-scored on a fresh run with the corrected catalog, after the
live-check finding below is explained.

## `cpu.live.tinyllama`: two prefills about 5.5x slower inside the kit only

The probe passed 10 of 10. Its log times each prefill window. Checks 1 to 8 (cluster) and the first leg of check 9
(one token at a time) take what they took on 2026-10-28. The batched windows of checks 9 and 10 do not:

| Window | 2026-10-28, kit | 2026-11-03, kit quick run | 2026-11-03, kit re-run, idle |
|---|---|---|---|
| check 9, 573 tokens, one window (`PrefillMode.BATCHED`, chunk 4096) | 16.5 s | **92.7 s** | **89.7 s** (other processes 3.9%) |
| check 10, chunk 64 | 5.1 s | **28.6 s** | **28.2 s** |

Per layer, at W=572, `qkvProj` reads about 70 ms on 2026-10-28 and about 440 ms now; `woProj+ffn+residuals` about 550 ms
and about 3,600 ms; attention is unchanged (45 to 110 ms both days).

Isolation, check 9's 573-token window outside the kit (`PrefillCheck.java`: `ModelLiveChecks.run` with the CPU
backend, from the same kit jar, one JVM):

| Run | Window | Check 9 total (s) |
|---|---|---|
| `PREFILL` only, `-Xmx4g`, GC log: 18 pauses | 18.6 s | 116 |
| `PREFILL` only, `-Xmx16g`, GC log: 3 pauses | 18.4 s | 118 |
| checks 1 to 9 in one JVM, `-Xmx4g` | 18.2 s | 198 (whole run) |
| checks 1 to 9, the probe's exact JVM flags (`-XX:-UseFastUnorderedTimeStamps` included) | 18.0 s | 199 |
| the same, with `nvidia-smi` compute-process and utilization queries every 2 s beside it | 18.8 s | 202 |

**How to read it.** The engine's CPU prefill is not slower. Outside the kit, the same window reads 18.0 to 18.8 s
under the probe's flags, heap and GPU polling. Inside the kit it reads about 90 s, with other processes under 4%. The same
day's server-side CPU prefill (`long-prompt.tinyllama`: 512 tokens 15.2 s static, 16.1 s continuous, against 15.3 and
16.2 on 2026-10-29) and the throughput probe's (33.6 t/s at 512, against 33.1 on 2026-10-28) are unchanged. What makes
the probe JVM's wide windows slow under the kit's runner is not found. Ruled out: heap size, GC, JVM flags, the `hpet`
clock source, `nvidia-smi` polling, the probe's entry class (it calls the same method), and other load. The finding is
open in the tier record; the probe's checks are unaffected (all pass).

## Files

- `SUMMARY-quick.txt`, `SUMMARY-live-rerun.txt`: the kit's summaries, from the scrubbed results zips.
- `PrefillCheck.java`: the isolation driver. The suites line was edited between runs (`PREFILL` alone, then
  `PIPELINE, TENSOR, PREFILL`).
