/*
 * Context shift of the device KV mirror for Juno.
 *
 * A shift moves the kept keys from position p to p - d. Every rotary variant
 * Juno runs rotates pair i of a head by p * freq[i] (times a scale that does not
 * depend on the position), so moving a key is a pure rotation by -d * freq[i]:
 * the same rotation for every moved row. The host builds that per-pair table
 * (RopeShift.back) and this kernel applies it to FP16 K rows in place, after the
 * rows were moved down with device-to-device copies. Each product and sum is
 * rounded separately (no fused multiply-add), as the host rotation computes it,
 * so a row rotated here equals the host rotation of the same FP16 values,
 * rounded to FP16.
 *
 * Kernel:
 *   kv_rotate_half(k, rows, kvDim, headDim, pairs, splitHalf, cosSin)
 *     k:       FP16 rows, row-major, stride kvDim (rows to rotate only)
 *     pairs:   rotated pairs per head (headDim / 2, or fewer for a partial rotation)
 *     splitHalf: pairs (i, i + pairs) when non-zero, else (2i, 2i + 1)
 *     cosSin:  float[2 * pairs], cos then sin of pair i at 2i and 2i + 1
 *   one thread per (row, head, pair).
 *
 * Build (PTX checked into resources; the JVM loads it with cuModuleLoadData):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/kv_shift.ptx kv_shift.cu
 */
#include <cuda_fp16.h>

#define KS_THREADS 256

extern "C" __global__ void __launch_bounds__(KS_THREADS)
kv_rotate_half(__half* __restrict__ k, long long rows, int kvDim, int headDim, int pairs, int splitHalf,
               const float* __restrict__ cosSin) {
    const int heads = kvDim / headDim;
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * heads * pairs)
        return;
    const int i = (int)(t % pairs);
    const long long rowHead = t / pairs;
    const int h = (int)(rowHead % heads);
    const long long r = rowHead / heads;
    const long long base = r * kvDim + (long long)h * headDim;
    const long long i0 = splitHalf ? base + i : base + 2 * i;
    const long long i1 = splitHalf ? base + i + pairs : base + 2 * i + 1;
    const float c = cosSin[2 * i];
    const float s = cosSin[2 * i + 1];
    const float x0 = __half2float(k[i0]);
    const float x1 = __half2float(k[i1]);
    k[i0] = __float2half_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
    k[i1] = __float2half_rn(__fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c)));
}
