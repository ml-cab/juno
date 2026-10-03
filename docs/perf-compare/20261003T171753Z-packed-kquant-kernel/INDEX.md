# Packed K-quant tiled GEMM: kernel speed and numerical quality - 20261003T171753Z (gpu)

## Purpose

First readings of the tiled matmul over packed Q4_K / Q5_K / Q6_K weights (`KQuantGemmKernel`,
`kquant_gemm.cu`) before it is routed into any prefill path. Two questions:

1. **Speed, per matmul.** On Mistral 7B's Q4_K shapes, how does Q8_1 quantize + tiled integer GEMM
   compare with today's prefill route, dequantize to FP16 + cuBLAS FP16 GEMM at FP32 compute, at
   window widths 16 to 512?
2. **Numerical quality.** Mean and max relative error of both routes against a CPU FP32 oracle, on the
   same matrices and activations.

**Not pinned, not a gate, not end to end.** Clocks were recorded, not fixed (CPU governor
`schedutil`, turbo on, GPU clock unlocked). These are device timings of isolated matmuls from a unit
test (host wall clock around ten back-to-back launches after three warm-up launches, then one stream
synchronize). The tier's throughput thresholds are read later, from a pinned same-hour end-to-end
A/B against the pre-switch build. Use this run to judge whether the kernel is worth routing, not to
score a threshold.

## Command

```bash
for i in 1 2 3; do
  mvn -q test -pl node -Dtest=KQuantGemmMicrobenchTest -Dsurefire.failIfNoSpecifiedTests=false
done
mvn -q test -pl node -Dtest=KQuantGemmQualityTest -Dsurefire.failIfNoSpecifiedTests=false
```

- Build: Juno `f02bdae` plus the uncommitted kernel change (tree dirty); `kquant_gemm.ptx` sha256
  `83db837f7e0ce82f`, compiled `nvcc -ptx -arch=compute_61 -O3` (CUDA 12.0). No jar: the tests run
  from the module's classes.
- Host: GTX 1080 (GP104, 8 GiB), driver 580.173.02.
- Raw output: `microbench-run1.txt` to `microbench-run3.txt`, `quality.txt`.

## Speed: ms per matmul, median of three runs

"Packed" is Q8_1 quantization of the activation rows plus the tiled kernel. "Dequant + FP16" is the
FP16 cast of the activation rows, the dequantization of the whole weight matrix and the FP16 GEMM,
which is what one matmul of the prefill-window region costs today. Speedup is the median of the three
per-run ratios.

| Shape (rows x cols) | Width | Packed ms | Dequant + FP16 ms | Speedup |
|---|---|---|---|---|
| 4096 x 4096 (Q, O) | 16 | 0.197 | 1.000 | 5.24x |
| | 32 | 0.175 | 0.979 | 5.63x |
| | 64 | 0.284 | 1.118 | 3.75x |
| | 128 | 0.479 | 1.227 | 2.56x |
| | 512 | 1.573 | 3.642 | 2.31x |
| 14336 x 4096 (gate, up) | 16 | 0.362 | 3.506 | 9.29x |
| | 32 | 0.552 | 3.512 | 6.41x |
| | 64 | 0.874 | 3.479 | 3.98x |
| | 128 | 1.509 | 4.644 | 3.10x |
| | 512 | 5.191 | 12.640 | 2.42x |
| 4096 x 14336 (down) | 16 | 0.545 | 4.649 | 8.32x |
| | 32 | 0.615 | 4.716 | 7.83x |
| | 64 | 0.867 | 4.945 | 5.69x |
| | 128 | 1.523 | 5.968 | 3.87x |
| | 512 | 5.386 | 15.511 | 2.96x |

Several rows' three readings spread by more than 15% (4096 x 4096 at width 16: 0.158 to 0.199 ms), so
read the table to the first digit of the speedup only. At width 512 the kernel sustains about 10 to 11
TOPS-equivalent on the 4096 x 4096 shape (17.2 G multiply-adds in 1.57 ms), against the about 7.4 the
step 2 decomposition says the 1.30x Mistral 7B threshold needs. `host-specific`: GP104 has no tensor
cores and runs FP16 arithmetic at 1/64 rate, so cuBLAS is held to FP32 compute here; on a device with
FP16 or int8 tensor cores the comparison must be redone.

## Numerical quality: relative error against the CPU FP32 oracle

Synthetic weights and activations uniform in [-1, 1], weights encoded with Juno's K-quant encoder.
Mean = sum |err| / sum |ref|, max = max |err| / max |ref|, over every output of the case.

| Type | rows x cols, width | Packed mean | Packed max | Dequant + FP16 mean | Dequant + FP16 max | Mean ratio |
|---|---|---|---|---|---|---|
| Q4_K | 512 x 512, 16 | 3.71e-3 | 4.62e-3 | 2.58e-4 | 2.12e-4 | 14.4 |
| Q4_K | 1376 x 512, 64 | 3.76e-3 | 3.81e-3 | 2.61e-4 | 2.80e-4 | 14.4 |
| Q4_K | 512 x 1536, 128 | 3.73e-3 | 3.54e-3 | 2.60e-4 | 2.52e-4 | 14.4 |
| Q4_K | 512 x 4096, 512 | 3.76e-3 | 4.05e-3 | 2.62e-4 | 2.62e-4 | 14.4 |
| Q5_K | 512 x 512, 16 | 3.70e-3 | 3.42e-3 | 2.58e-4 | 2.73e-4 | 14.4 |
| Q5_K | 1376 x 512, 64 | 3.75e-3 | 3.63e-3 | 2.61e-4 | 2.55e-4 | 14.4 |
| Q5_K | 512 x 1536, 128 | 3.75e-3 | 3.13e-3 | 2.61e-4 | 2.07e-4 | 14.4 |
| Q5_K | 512 x 4096, 512 | 3.76e-3 | 3.95e-3 | 2.62e-4 | 2.58e-4 | 14.4 |
| Q6_K | 512 x 512, 16 | 3.76e-3 | 3.62e-3 | 2.56e-4 | 2.34e-4 | 14.7 |
| Q6_K | 1376 x 512, 64 | 3.75e-3 | 3.37e-3 | 2.61e-4 | 2.48e-4 | 14.4 |
| Q6_K | 512 x 1536, 128 | 3.73e-3 | 3.86e-3 | 2.59e-4 | 2.42e-4 | 14.4 |
| Q6_K | 512 x 4096, 512 | 3.76e-3 | 3.52e-3 | 2.61e-4 | 2.37e-4 | 14.4 |

How to read it: the packed route's error is about 0.37% of the output magnitude on every type and
shape, 14x the FP16 route's. It is the rounding of the activations to 8 bits (Q8_1, one scale per 32
values), the same rounding the fused decode GEMV already applies to every generated token: the tiled
kernel agrees with the GEMV to within 2e-5 relative (`KQuantGemmParityTest`). The FP16 route rounds
activations to an 11-bit mantissa. `expected-general`: any integer-dot kernel over 8-bit activations
carries this error; it does not depend on the device. Uniform activations are a mild case: real
activations with outliers inside a 32-value block lose more precision under a shared 8-bit scale.
The end-to-end check of whether it matters is the greedy-decode agreement threshold, read once the
kernel is routed.
