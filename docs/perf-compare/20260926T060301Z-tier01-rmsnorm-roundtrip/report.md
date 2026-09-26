# RMS-norm host-round-trip microbench

Host: NVIDIA GeForce GTX 1080

`vs CPU scalar` is the scalar CPU median over the lane median: at or above `1.00x`
the lane is at least as fast as the path it would replace. A row whose repetitions
disagree by more than 15% of their median is not scorable.

| width | batch | dim | lane | median ms | min ms | max ms | spread % | vs CPU scalar | scorable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| decode | 1 | 2048 | cpu-scalar | 0.0035 | 0.0034 | 0.0035 | 0.6 | 1.00x | yes |
| decode | 1 | 2048 | gpu-round-trip | 0.0371 | 0.0365 | 0.0373 | 2.3 | 0.09x | yes |
| prefill | 512 | 2048 | cpu-scalar | 1.3004 | 1.2991 | 1.3132 | 1.1 | 1.00x | yes |
| prefill | 512 | 2048 | gpu-round-trip | 2.0859 | 2.0825 | 2.1129 | 1.5 | 0.62x | yes |

Free VRAM before: 7785021440 bytes, after: 7784955904 bytes, of 8497594368 total; not returned: 65536 bytes.

Largest divergence from the scalar CPU path, per width (tolerance 1.0E-4):

- decode (batch 1): 5.960e-07
- prefill (batch 512): 1.907e-06
