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
| decode | 1 | 2048 | cpu-scalar | 0.0670 | 0.0669 | 0.0677 | 1.2 | 1.00x | 1.86 | yes |
| decode | 1 | 2048 | gpu-op-at-a-time | 0.0359 | 0.0332 | 0.0364 | 9.0 | 1.86x | 1.00 | yes |
| decode | 1 | 2048 | gpu-resident-chain | 0.0183 | 0.0182 | 0.0188 | 3.7 | 3.65x | 0.51 | yes |
| decode | 1 | 2048 | gpu-device-only | 0.0112 | 0.0112 | 0.0114 | 2.2 | 5.98x | 0.31 | yes |
| prefill | 512 | 2048 | cpu-scalar | 34.0589 | 33.9104 | 34.4949 | 1.7 | 1.00x | 8.77 | yes |
| prefill | 512 | 2048 | gpu-op-at-a-time | 3.8835 | 3.8555 | 3.8993 | 1.1 | 8.77x | 1.00 | yes |
| prefill | 512 | 2048 | gpu-resident-chain | 2.1479 | 2.1047 | 2.1586 | 2.5 | 15.86x | 0.55 | yes |
| prefill | 512 | 2048 | gpu-device-only | 0.1581 | 0.1563 | 0.1592 | 1.8 | 215.47x | 0.04 | yes |

Reading per width:

- decode (batch 1): resident chain 3.65x the CPU scalar chain, 0.51 of op-at-a-time; device-only 5.98x the CPU scalar chain, 0.31 of op-at-a-time
- prefill (batch 512): resident chain 15.86x the CPU scalar chain, 0.55 of op-at-a-time; device-only 215.47x the CPU scalar chain, 0.04 of op-at-a-time

Free VRAM before: 8022720512 bytes, after: 8028880896 bytes, of 8497594368 total; not returned: 0 bytes.

Largest divergence of any GPU lane from the scalar CPU chain, per width (tolerance 1.0E-4):

- decode (batch 1): 5.960e-07
- prefill (batch 512): 2.384e-06
