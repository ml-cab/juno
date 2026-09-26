# RMS-norm host-round-trip baseline — 20260926T060301Z

Host: **medion-Precision-T3610** · Intel Xeon E5-1650 v2 (12 threads) · NVIDIA GeForce GTX 1080
(SM 1911 MHz, mem 5005 MHz) · CPU governor `schedutil`.

Harness: `scripts/performance-tests/rmsnorm-roundtrip-microbench.sh` (dim 2048, batches 1 and 512,
3 repetitions, 3000 ms warm-up per lane, 800 ms measurement window per repetition). Commit `a906a8d`
plus the uncommitted harness itself; no `src/main` inference path was changed to take this reading.
Device otherwise idle: no other compute process held the GPU.

## What this measures and why

`CudaRmsNorm` is built, tested and deliberately left unconstructed by `LlamaTransformerHandler`,
because a live decode comparison found it slower than the scalar CPU path it replaces. That finding
is the premise of the activation-residency work, so it is re-established here as a repeatable
measurement instead of being carried forward as a remembered number. This is the **before** side:
the cost of one GPU norm when every call stages its own activation to the device and reads the
result back.

`vs CPU scalar` is the scalar CPU median over the lane median: at or above `1.00x` the lane is at
least as fast as the path it would replace.

| width | batch | dim | lane | median ms | min ms | max ms | spread % | vs CPU scalar | scorable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| decode | 1 | 2048 | cpu-scalar | 0.0035 | 0.0034 | 0.0035 | 0.6 | 1.00x | yes |
| decode | 1 | 2048 | gpu-round-trip | 0.0371 | 0.0365 | 0.0373 | 2.3 | **0.09x** | yes |
| prefill | 512 | 2048 | cpu-scalar | 1.3004 | 1.2991 | 1.3132 | 1.1 | 1.00x | yes |
| prefill | 512 | 2048 | gpu-round-trip | 2.0859 | 2.0825 | 2.1129 | 1.5 | **0.62x** | yes |

Every row is scorable: repetitions agree to within 2.3%, against the 15% dispersion rule.

## Reading

- **Decode width: 10.6x slower than scalar CPU.** This reproduces the finding recorded in
  `CudaRmsNorm`'s class javadoc, which measured roughly 11x on a live decode. The absolute
  milliseconds differ — this times the operation in isolation at steady state, not a p95 inside a
  full forward pass — but the ratio lands in the same place.
- **Prefill width: 1.60x slower than scalar CPU.** `docs/performance.md`'s Phase B checkpoint
  recorded 1.56x to 2.18x slower at prefill batch scale under pinned-memory staging; this sits at
  the fast end of that range.
- **The gap narrows with width but does not close.** At batch 1 the fixed per-call cost dominates
  and the GPU lane loses by an order of magnitude; at batch 512 that same fixed cost is amortised
  across 512 rows and the loss falls to 1.6x, while about 4 MB is staged each way. The two widths
  therefore fail for different reasons — launch overhead against staging bandwidth — so a result at
  one says nothing about the other, and neither is at parity. There is no width at which the current
  path is already good enough to leave alone.

## Correctness

The GPU lane's output is checked against the scalar CPU lane's on every run, and the harness refuses
to report timings if they diverge beyond `1e-4`:

- decode (batch 1): largest absolute divergence `5.960e-07`
- prefill (batch 512): largest absolute divergence `1.907e-06`

## Noise and device memory

- `jdk.GCPhasePause`: **1 pause, 2.71 ms**, over the whole run. Far below anything that could move a
  reading; no repetition needed re-running on collection-pause grounds.
- `jdk.ThreadAllocationStatistics.bytes_total`: 248.2 MB for the whole process, dominated by
  warm-up-time method-handle and class-building allocation rather than by either measured lane.
- `jdk.ExecutionSample` confirms what was measured: `rmsNormInto` 856 samples,
  `CudaRmsNorm.normalizeBatch` 284, out of 1245.
- Free VRAM: 7785021440 bytes before, 7784955904 bytes after, of 8497594368 total; **65536 bytes not
  returned**. That is `CudaRmsNorm`'s per-thread device scratch, allocated on first use and grown in
  place rather than freed per call — a fixed retention, not a per-call leak.

**The retention figure is device-wide, not per-process.** During this harness's bring-up a stray
test JVM held 7.2 GB of the GPU alongside a run, and the same reading reported 4.1 GB "not
returned" — memory that belonged to the other process. The harness now carries device total
alongside free bytes and states plainly when retention exceeds what its own scratch could be, so
that contamination is visible in the report rather than published as a leak. Take this reading only
on an otherwise idle device.

## Scope note

No perf gate against a published baseline applies to this run. The harness adds a standalone
measurement class that no handler references, so the forward pass, MatVec, KV and batching paths are
unchanged, and `compare-lora.sh` / `compare-llama-cpp.sh` have nothing to detect. Those gates become
required when the residency path itself lands.

Artifacts: [`report.md`](report.md) (harness output verbatim), [`jfr.json`](jfr.json) (extracted JFR
metrics), [`host.txt`](host.txt) (GPU clocks, CPU governor, thread count).
