/*
 * GPU-resident RMS-normalisation kernel for Juno (Tier 19 Phase A, step 2 —
 * see docs/infra-plan/PLAN-Infra-Tier19.md).
 *
 * One block per row of a [B][dim] activation batch: a block-wide sum-of-
 * squares reduction, then a second elementwise pass writing
 * x[i] * rsqrt(meanSq + eps) * weight[i] — the exact same math as
 * LlamaTransformerHandler.rmsNorm/rmsNormInto's CPU scalar loop, just
 * parallelized across GPU threads. weight[] is shared across every row in
 * the batch (the same layer's attn_norm/ffn_norm tensor), unlike
 * gqa_attention.cu's per-row K/V pointers.
 *
 * Batched design: B = 1 for single-token decode, B = window size for
 * batched prefill, B = N for --parallel multi-decode — one launch covers
 * all three of LlamaTransformerHandler's call sites, matching
 * gqa_attention.cu's batching convention.
 *
 * Compile (Pascal+ reference SKU — sm_61, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/rms_norm.ptx rms_norm.cu
 */

#define RMSNORM_THREADS 128
#define RMSNORM_WARPS (RMSNORM_THREADS / 32)

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* Block-wide sum reduction, result broadcast to every thread (every thread
 * needs meanSq for the normalizing pass that follows). */
static __device__ __forceinline__ float block_reduce_sum(float v) {
    v = warp_sum(v);
    __shared__ float warpSums[RMSNORM_WARPS];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0)
        warpSums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = (lane < RMSNORM_WARPS) ? warpSums[lane] : 0.f;
        w = warp_sum(w);
        if (lane == 0)
            warpSums[0] = w;
    }
    __syncthreads();
    return warpSums[0];
}

extern "C" __global__ void __launch_bounds__(RMSNORM_THREADS)
rms_norm(
        const float* __restrict__ xBatch,   // [B][dim]
        const float* __restrict__ weight,   // [dim], shared across all rows
        float* __restrict__ outBatch,       // [B][dim]
        int dim,
        float eps) {
    const int b = (int)blockIdx.x;
    const float* x = xBatch + (size_t)b * dim;
    float* out = outBatch + (size_t)b * dim;

    float localSumSq = 0.f;
    for (int i = (int)threadIdx.x; i < dim; i += (int)blockDim.x) {
        float v = x[i];
        localSumSq += v * v;
    }
    const float sumSq = block_reduce_sum(localSumSq);
    const float invRms = rsqrtf(sumSq / (float)dim + eps);

    for (int i = (int)threadIdx.x; i < dim; i += (int)blockDim.x)
        out[i] = x[i] * invRms * weight[i];
}
