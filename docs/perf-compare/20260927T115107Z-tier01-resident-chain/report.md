# Resident RMS-norm + RoPE chain microbench

Host: NVIDIA GeForce GTX 1080

The chain is RMS norm, then RoPE on the normalized rows, as heads of head size 64 (base 10000): the decode row at position 512, the prefill window at positions 0 onwards.

`speedup vs CPU scalar` is the scalar CPU chain's median over the lane median: at or above
`1.00x` the lane is at least as fast as the path it would replace. `cost vs op-at-a-time` is the
lane median over the op-at-a-time median: the share of that cost left once the host round trip
between the two operations is removed. A row whose repetitions disagree by more than 15% of their
median is not scorable.

| width | batch | dim | lane | median ms | min ms | max ms | spread % | speedup vs CPU scalar | cost vs op-at-a-time | scorable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| decode | 1 | 2048 | cpu-scalar | 0.0049 | 0.0049 | 0.0050 | 2.9 | 1.00x | 0.15 | yes |
| decode | 1 | 2048 | gpu-op-at-a-time | 0.0320 | 0.0313 | 0.0327 | 4.3 | 0.15x | 1.00 | yes |
| decode | 1 | 2048 | gpu-resident-chain | 0.0187 | 0.0187 | 0.0190 | 1.4 | 0.26x | 0.58 | yes |
| decode | 1 | 2048 | gpu-device-only | 0.0111 | 0.0109 | 0.0113 | 3.0 | 0.45x | 0.35 | yes |
| prefill | 512 | 2048 | cpu-scalar | 2.1306 | 2.1185 | 2.1572 | 1.8 | 1.00x | 0.55 | yes |
| prefill | 512 | 2048 | gpu-op-at-a-time | 3.8676 | 3.8624 | 3.9339 | 1.8 | 0.55x | 1.00 | yes |
| prefill | 512 | 2048 | gpu-resident-chain | 2.1264 | 2.1184 | 2.1340 | 0.7 | 1.00x | 0.55 | yes |
| prefill | 512 | 2048 | gpu-device-only | 0.1570 | 0.1561 | 0.1575 | 0.9 | 13.57x | 0.04 | yes |

Reading per width:

- decode (batch 1): resident chain 0.26x the CPU scalar chain, 0.58 of op-at-a-time; device-only 0.45x the CPU scalar chain, 0.35 of op-at-a-time
- prefill (batch 512): resident chain 1.00x the CPU scalar chain, 0.55 of op-at-a-time; device-only 13.57x the CPU scalar chain, 0.04 of op-at-a-time

Free VRAM before: 8020295680 bytes, after: 8020688896 bytes, of 8497594368 total; not returned: 0 bytes.

Largest divergence of any GPU lane from the scalar CPU chain, per width (tolerance 1.0E-4):

- decode (batch 1): 5.960e-07
- prefill (batch 512): 2.384e-06
