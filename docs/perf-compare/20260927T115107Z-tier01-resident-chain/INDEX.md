# Resident RMS-norm + RoPE chain, re-read on the cheaper CPU RoPE — 20260927T115107Z

Host: **medion-Precision-T3610** · Intel Xeon E5-1650 v2 (12 threads) · NVIDIA GeForce GTX 1080
(SM 1860 MHz and memory 5005 MHz read at idle right after the run) · CPU governor `schedutil`.

Harness: `scripts/performance-tests/resident-chain-microbench.sh`, unchanged since
[`20260927T025430Z-tier01-resident-chain`](../20260927T025430Z-tier01-resident-chain/INDEX.md) (dim
2048 as 32 heads of 64, RoPE base 10000, decode row at position 512, prefill window at positions 0 to
511, 3 repetitions, 3000 ms warm-up per lane, 800 ms measurement window per repetition). Commit
`cc94c53` plus uncommitted work; the change that matters here is that the scalar CPU rotation now reads
its cosines and sines from a per-position table (`RopeTable`) instead of recomputing a power, a cosine
and a sine for every rotated pair on every call. The table is bit-identical to the old computation.
The GPU lanes are the same code as in the earlier run. Device otherwise idle.

## Why re-read it

The earlier run found that about 95% of the CPU chain was the rotation's redundant angle computation,
so its "resident chain vs CPU" ratio mostly measured that waste. With the waste removed, this run shows
what the chain is worth against a CPU path that does only the necessary work.

| width | batch | dim | lane | median ms | min ms | max ms | spread % | speedup vs CPU scalar | cost vs op-at-a-time | scorable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| decode | 1 | 2048 | cpu-scalar | 0.0049 | 0.0049 | 0.0050 | 2.9 | 1.00x | 0.15 | yes |
| decode | 1 | 2048 | gpu-op-at-a-time | 0.0320 | 0.0313 | 0.0327 | 4.3 | 0.15x | 1.00 | yes |
| decode | 1 | 2048 | gpu-resident-chain | 0.0187 | 0.0187 | 0.0190 | 1.4 | **0.26x** | **0.58** | yes |
| decode | 1 | 2048 | gpu-device-only | 0.0111 | 0.0109 | 0.0113 | 3.0 | 0.45x | 0.35 | yes |
| prefill | 512 | 2048 | cpu-scalar | 2.1306 | 2.1185 | 2.1572 | 1.8 | 1.00x | 0.55 | yes |
| prefill | 512 | 2048 | gpu-op-at-a-time | 3.8676 | 3.8624 | 3.9339 | 1.8 | 0.55x | 1.00 | yes |
| prefill | 512 | 2048 | gpu-resident-chain | 2.1264 | 2.1184 | 2.1340 | 0.7 | **1.00x** | **0.55** | yes |
| prefill | 512 | 2048 | gpu-device-only | 0.1570 | 0.1561 | 0.1575 | 0.9 | 13.57x | 0.04 | yes |

Every row is scorable: repetitions agree to within 4.3%, against the 15% dispersion rule.

## Reading

- **The CPU chain got 13.7 times cheaper at decode (0.0670 to 0.0049 ms) and 16 times at prefill
  (34.06 to 2.13 ms).** That is the rotation's redundant angle computation removed; the GPU lanes did
  not move (resident chain 0.0183 to 0.0187 ms at decode, 2.148 to 2.126 ms at prefill).
- **Against that CPU path the resident chain is 0.26x at decode width and 1.00x at prefill width.**
  At decode a two-operation region that pays its own entry and exit loses to two elementwise
  operations that cost five microseconds on the CPU. At prefill it is at parity: its median is 1.002x
  the CPU's and the two lanes' min-max ranges overlap, so the difference is inside the noise.
- **The chaining result, which isolates residency, holds: 0.58 of op-at-a-time at decode, 0.55 at
  prefill** (0.51 and 0.55 before). Removing the round trip between two operations still roughly
  halves their cost.
- **The device-only lane is where the case for residency now rests**: the two operations on an
  activation already on the device cost 0.157 ms at prefill against 2.13 ms on the CPU (13.6x), and
  0.0111 ms at decode against 0.0049 ms (0.45x). A region only pays when it spans enough operations
  that its single entry and exit are amortised; norm and RoPE alone are not enough at decode width.
- Execution samples now put the CPU norm ahead of the rotation (`rmsNormInto` 612, `rope` 300 of
  1562), where the earlier run had the rotation at 1004 of 1663.

## Correctness

Every GPU lane's output is checked against the scalar CPU chain on every run; the harness refuses to
report timings if they diverge beyond `1e-4`:

- decode (batch 1): largest absolute divergence `5.960e-07`
- prefill (batch 512): largest absolute divergence `2.384e-06`

## Noise and device memory

- `jdk.GCPhasePause`: **1 pause, 3.70 ms**, over the whole run.
- `jdk.ThreadAllocationStatistics.bytes_total`: 56.4 MB for the whole process, setup included.
- Free device memory before 8020295680 bytes, after 8020688896 bytes: **nothing left allocated**.

Files: `report.md` (the harness's own report), `jfr.json` (metrics extracted from the recording),
`host.txt`.
