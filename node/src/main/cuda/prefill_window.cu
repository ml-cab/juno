/*
 * Elementwise kernels of the prefill-window device region for Juno.
 *
 * The region keeps a prefill window's activations on the device across a
 * transformer layer: the RMS norms run through rms_norm.cu, the matmuls through
 * cuBLAS on FP16 operands, attention through gqa_attention.cu, and the work
 * between them through the kernels here. Each matches the CPU window path step
 * for step, so the device and host paths differ only where the CPU path's own
 * rounding cannot be reproduced exactly:
 *
 *   to_half     out[i] = fp16(in[i]), round to nearest even - the same rounding
 *               as Float.floatToFloat16, which packs every GEMM input on the host.
 *   swiglu_half out[r][i] = fp16(silu(gu[r][i]) * gu[r][I + i]), the gate and up
 *               projections side by side in one [rows][2I] buffer. silu follows
 *               LlamaTransformerHandler.silu: the exponential in double, rounded
 *               to float, then x / (1 + e) in float. The double exponential costs
 *               a few hundred microseconds per layer on a consumer card and keeps
 *               the result within one FP16 ulp of the host loop.
 *   swiglu      out[r][i] = silu(gu[r][i]) * gu[r][I + i] in float: swiglu_half
 *               without the FP16 cast, for the decode region, whose down
 *               projection quantizes an FP32 row.
 *   add_inplace x[i] += y[i], the residual add.
 *   residual_add_both
 *               s = x[i] + y[i]; x[i] = s; y[i] = s - the decode region's second
 *               residual add, which leaves the layer output both in the residual
 *               row (the next layer's input) and in the row it downloads.
 *   decode_attention_table
 *               writes one decode row's attention table (K pointer, V pointer,
 *               sequence length) from launch arguments, so the table needs no
 *               host-to-device copy.
 *   add_bias    x[r][j] += bias[j], the Q/K/V bias of Qwen2-family models.
 *   rms_norm_host_order
 *               out[r][i] = (w[i] * x[r][i]) * scale, scale = 1 / sqrt(ss / n + eps),
 *               with ss the squares summed one after another in float - the order
 *               and roundings of LlamaTransformerHandler.rmsNormInto, so the result
 *               is bit-identical to it. rms_norm.cu reduces in a tree and uses
 *               rsqrtf, which is faster at decode width but lands a few ulps away,
 *               and the FP16 cast of the GEMM input that follows turns a few ulps
 *               into a flipped half-precision rounding now and then. At prefill
 *               width the sequential sum costs tens of microseconds per norm.
 *
 * Every multiply, add and divide is rounded separately (__fmul_rn and friends)
 * so the compiler cannot contract them into fused multiply-adds the CPU path
 * does not use.
 *
 * Compile (Pascal+ reference SKU - sm_61, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/prefill_window.ptx prefill_window.cu
 */

#include <cuda_fp16.h>

#define PW_THREADS 256
#define NORM_THREADS 128

extern "C" __global__ void __launch_bounds__(PW_THREADS)
to_half(const float* __restrict__ in, __half* __restrict__ out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half_rn(in[i]);
}

extern "C" __global__ void __launch_bounds__(PW_THREADS)
swiglu_half(const float* __restrict__ gateUp, __half* __restrict__ out, int rows, int inter) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * inter;
    if (i >= total)
        return;
    const long long r = i / inter;
    const int c = (int)(i - r * inter);
    const float* row = gateUp + r * 2LL * inter;
    const float g = row[c];
    const float u = row[inter + c];
    const float e = (float)exp(-(double)g);
    const float silu = __fdiv_rn(g, __fadd_rn(1.0f, e));
    out[i] = __float2half_rn(__fmul_rn(silu, u));
}

extern "C" __global__ void __launch_bounds__(PW_THREADS)
swiglu(const float* __restrict__ gateUp, float* __restrict__ out, int rows, int inter) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * inter;
    if (i >= total)
        return;
    const long long r = i / inter;
    const int c = (int)(i - r * inter);
    const float* row = gateUp + r * 2LL * inter;
    const float g = row[c];
    const float u = row[inter + c];
    const float e = (float)exp(-(double)g);
    const float silu = __fdiv_rn(g, __fadd_rn(1.0f, e));
    out[i] = __fmul_rn(silu, u);
}

extern "C" __global__ void __launch_bounds__(PW_THREADS)
residual_add_both(float* __restrict__ x, float* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = __fadd_rn(x[i], y[i]);
        x[i] = s;
        y[i] = s;
    }
}

extern "C" __global__ void
decode_attention_table(void** table, void* k, void* v, int seqLen) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        table[0] = k;
        table[1] = v;
        *(int*)(table + 2) = seqLen;
    }
}

extern "C" __global__ void __launch_bounds__(PW_THREADS)
add_inplace(float* __restrict__ x, const float* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __fadd_rn(x[i], y[i]);
}

extern "C" __global__ void __launch_bounds__(PW_THREADS)
add_bias(float* __restrict__ x, const float* __restrict__ bias, int rows, int dim) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * dim;
    if (i >= total)
        return;
    const int c = (int)(i % dim);
    x[i] = __fadd_rn(x[i], bias[c]);
}

extern "C" __global__ void __launch_bounds__(NORM_THREADS)
rms_norm_host_order(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out,
        int dim, float eps) {
    const float* row = x + (long long)blockIdx.x * dim;
    float* o = out + (long long)blockIdx.x * dim;
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int i = 0; i < dim; i++)
            ss = __fadd_rn(ss, __fmul_rn(row[i], row[i]));
        const float t = __fadd_rn(__fdiv_rn(ss, (float)dim), eps);
        scale = __fdiv_rn(1.0f, (float)sqrt((double)t));
    }
    __syncthreads();
    const float s = scale;
    for (int i = (int)threadIdx.x; i < dim; i += (int)blockDim.x)
        o[i] = __fmul_rn(__fmul_rn(weight[i], row[i]), s);
}
