# Resident RMS-norm + RoPE chain — 20260927T025430Z

Host: **medion-Precision-T3610** · Intel Xeon E5-1650 v2 (12 threads) · NVIDIA GeForce GTX 1080
(SM 1607 MHz and memory 5005 MHz read at idle just before the run) · CPU governor `schedutil`.

Harness: `scripts/performance-tests/resident-chain-microbench.sh` (dim 2048 as 32 heads of 64, RoPE
base 10000, decode row at position 512, prefill window at positions 0 to 511, 3 repetitions, 3000 ms
warm-up per lane, 800 ms measurement window per repetition). Commit `cc94c53` plus the uncommitted
residency primitive and harness themselves; no handler references either, so no inference path was
changed to take this reading. Device otherwise idle: no other compute process held the GPU.

## What this measures and why

[`20260926T060301Z-tier01-rmsnorm-roundtrip`](../20260926T060301Z-tier01-rmsnorm-roundtrip/INDEX.md)
showed that one GPU RMS norm which stages its own activation both ways is slower than the scalar CPU
path at both widths. The claim under test here is that the cost is the host round trip, not the
device work: keep the activation on the device between two operations and the chain should pay.

The chain is RMS norm, then RoPE on the normalized rows. Four lanes, all GPU lanes dispatching
through the same `ResidentActivation` path with device-resident weights, so the ratio between them
isolates the round trip:

- `cpu-scalar` — `rmsNormInto`, then `rope` in place, per row: what the handler runs today.
- `gpu-op-at-a-time` — upload, norm, download; upload, RoPE, download.
- `gpu-resident-chain` — upload once, norm, RoPE, download once.
- `gpu-device-only` — norm and RoPE on an activation already on the device, then wait: the marginal
  cost of the two operations inside a longer region whose entry and exit are paid elsewhere.

| width | batch | dim | lane | median ms | min ms | max ms | spread % | speedup vs CPU scalar | cost vs op-at-a-time | scorable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| decode | 1 | 2048 | cpu-scalar | 0.0670 | 0.0669 | 0.0677 | 1.2 | 1.00x | 1.86 | yes |
| decode | 1 | 2048 | gpu-op-at-a-time | 0.0359 | 0.0332 | 0.0364 | 9.0 | 1.86x | 1.00 | yes |
| decode | 1 | 2048 | gpu-resident-chain | 0.0183 | 0.0182 | 0.0188 | 3.7 | **3.65x** | **0.51** | yes |
| decode | 1 | 2048 | gpu-device-only | 0.0112 | 0.0112 | 0.0114 | 2.2 | 5.98x | 0.31 | yes |
| prefill | 512 | 2048 | cpu-scalar | 34.0589 | 33.9104 | 34.4949 | 1.7 | 1.00x | 8.77 | yes |
| prefill | 512 | 2048 | gpu-op-at-a-time | 3.8835 | 3.8555 | 3.8993 | 1.1 | 8.77x | 1.00 | yes |
| prefill | 512 | 2048 | gpu-resident-chain | 2.1479 | 2.1047 | 2.1586 | 2.5 | **15.86x** | **0.55** | yes |
| prefill | 512 | 2048 | gpu-device-only | 0.1581 | 0.1563 | 0.1592 | 1.8 | 215.47x | 0.04 | yes |

Every row is scorable: repetitions agree to within 9.0%, against the 15% dispersion rule.

## Reading

- **Removing the round trip between two operations roughly halves their cost, at both widths.** The
  resident chain costs 0.51 of op-at-a-time at decode and 0.55 at prefill. This is the mechanism the
  residency work rests on, measured with everything else held equal.
- **Against today's scalar CPU chain the resident chain is 3.65x faster at decode and 15.86x at
  prefill — but that ratio mostly measures the CPU RoPE, not residency.** The CPU norm alone costs
  0.0035 ms at decode and 1.3004 ms at prefill on this host (the run linked above), so about 95% of
  the CPU chain is `rope`, which recomputes a power, a cosine and a sine for every rotated pair on
  every call. Execution samples agree: `LlamaTransformerHandler.rope` holds 1004 of the run's 1663.
  Against the CPU norm alone, the resident chain at decode is still about five times slower. A
  two-operation region that pays its own entry and exit does not beat an elementwise operation that
  is cheap on the CPU at decode width; it pays here because RoPE is not cheap on the CPU today.
- **At prefill the resident chain is almost all boundary.** The two operations themselves take
  0.158 ms (`gpu-device-only`); the other 1.99 ms of the resident chain is 4 MB staged each way plus
  the host copies into and out of the pinned buffer. That boundary is what a region spanning more
  operations — the matrix-vector projections between norm and RoPE — would stop paying per pair of
  operations.
- **At decode the device-only lane is 0.0112 ms for two operations and one wait**, so a device
  operation inside a region costs a few microseconds of launch and synchronization before any work
  is done. Whether a region pays at decode width depends on how many operations share each wait.

## Correctness

Every GPU lane's output is checked against the scalar CPU chain on every run, and the harness refuses
to report timings if they diverge beyond `1e-4`:

- decode (batch 1): largest absolute divergence `5.960e-07`
- prefill (batch 512): largest absolute divergence `2.384e-06`

The whole divergence is the RMS norm's summation order: the RoPE kernel is bit-identical to the CPU
rotation in unit tests at positions up to 30000.

## Noise and device memory

- `jdk.GCPhasePause`: **1 pause, 3.72 ms**, over the whole run. No repetition needed re-running on
  collection-pause grounds.
- `jdk.ThreadAllocationStatistics.bytes_total`: 56.2 MB for the whole process, all of it setup: the
  input and output arrays (the largest site, `ResidentChainMicrobench.measure`, 26.2 MB) and class
  generation during warm-up. The measured lanes allocate nothing per call.
- Free VRAM: 8022720512 bytes before, 8028880896 bytes after, of 8497594368 total; **0 bytes not
  returned**. Free memory rose by 6.2 MB over the run. The figure is device-wide, so it cannot say
  whose memory that was, only that nothing this run allocated is still held: the kernel modules and
  the RoPE table were loaded before the first reading, and every activation buffer is freed when its
  chain closes. The unit tests make the same assertion exactly, per allocate-and-close cycle.

## Scope note

No perf gate against a published baseline applies to this run. The residency primitive and the RoPE
kernel are standalone: no handler references them, so the forward pass, MatVec, KV and batching paths
are unchanged, and `compare-lora.sh` / `compare-llama-cpp.sh` have nothing to detect. Those gates
become required when the resident path is wired into a handler.

See [`../../performance.md`](../../performance.md) for how this sits next to the round-trip baseline
and for the CPU RoPE's measured share of forward-pass time in the standing sweep.
