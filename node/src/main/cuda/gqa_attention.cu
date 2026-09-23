/*
 * GPU-resident grouped-query attention kernel for Juno (a straightforward
 * parallel kernel, not a tiled/online-softmax FlashAttention design).
 *
 * One block per (batch row b, query head h): QK^T dot products, a max-
 * subtracted softmax, and the weighted-V-sum — the exact same math as
 * LlamaTransformerHandler's GqaMath.attend() CPU oracle (grouped-query
 * mapping kvHead = h / gqaRatio, causal masking implicit via seqLens[b]),
 * just parallelized across GPU threads instead of one CPU thread's nested
 * loops.
 *
 * K/V are read as real IEEE FP16 (see DeviceKvCache — not to be confused with
 * kvcache's KvElementType.F16, which is actually float32) and converted to
 * float per element; Q, the softmax, and the accumulation stay FP32 for
 * numerical stability, matching the project's existing FP16-weight /
 * FP32-activation convention (e.g. DeviceHalfMatrix).
 *
 * Batched-pointer design: kPtrs[b]/vPtrs[b] let one launch serve all three
 * attention call sites in LlamaTransformerHandler — a growing prefill window
 * passes the SAME device pointer for every b (one DeviceKvCache being
 * appended to across the window, B = window size), single-token decode is
 * the B=1 case, and --parallel multi-decode passes N different pointers (one
 * DeviceKvCache per stream).
 *
 * Compile (Pascal+ reference SKU — sm_61, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/gqa_attention.ptx gqa_attention.cu
 */
#include <cuda_fp16.h>

#define GQA_THREADS 128
#define GQA_WARPS (GQA_THREADS / 32)

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/*
 * Block-wide reduction, result broadcast to every thread — unlike
 * q4k_gemv.cu's reduce_and_store (single writer to global y[]), both the
 * softmax max and the exp-sum computed here are needed by every thread in
 * the pass that follows.
 */
static __device__ __forceinline__ float block_reduce(float v, bool isMax) {
    v = isMax ? warp_max(v) : warp_sum(v);
    __shared__ float warpVals[GQA_WARPS];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0)
        warpVals[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = (lane < GQA_WARPS) ? warpVals[lane] : (isMax ? -INFINITY : 0.f);
        w = isMax ? warp_max(w) : warp_sum(w);
        if (lane == 0)
            warpVals[0] = w;
    }
    __syncthreads();
    return warpVals[0];
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention(
        const float* __restrict__ qBatch,      // [B][numHeads*headDim]
        const half* const* __restrict__ kPtrs, // [B] device ptrs -> [rows][kvDim]
        const half* const* __restrict__ vPtrs, // [B] device ptrs -> [rows][kvDim]
        const int* __restrict__ seqLens,       // [B]
        float* __restrict__ scoresScratch,     // [B*numHeads*rowStride]
        float* __restrict__ outBatch,          // [B][numHeads*headDim]
        int numHeads,
        int gqaRatio,
        int headDim,
        int kvDim,
        int rowStride) {
    const int bh = (int)blockIdx.x;   // 0 .. B*numHeads-1
    const int b = bh / numHeads;
    const int h = bh % numHeads;
    const int seqLen = seqLens[b];
    const int kBase = (h / gqaRatio) * headDim;

    const float* q = qBatch + (size_t)bh * headDim;
    const half* K = kPtrs[b];
    const half* V = vPtrs[b];
    float* scores = scoresScratch + (size_t)bh * rowStride;
    float* out = outBatch + (size_t)bh * headDim;

    const float scale = rsqrtf((float)headDim);

    // Pass 1: scores[t] = scale * dot(q, K[t, kBase:kBase+headDim)), track max.
    float localMax = -INFINITY;
    for (int t = (int)threadIdx.x; t < seqLen; t += (int)blockDim.x) {
        const half* krow = K + (size_t)t * kvDim + kBase;
        float dot = 0.f;
        for (int d = 0; d < headDim; d++)
            dot += q[d] * __half2float(krow[d]);
        const float s = dot * scale;
        scores[t] = s;
        localMax = fmaxf(localMax, s);
    }
    const float blockMax = block_reduce(localMax, true);

    // Pass 2: exponentiate in place (max-subtracted for stability), track sum.
    float localSum = 0.f;
    for (int t = (int)threadIdx.x; t < seqLen; t += (int)blockDim.x) {
        const float e = expf(scores[t] - blockMax);
        scores[t] = e;
        localSum += e;
    }
    const float invSum = 1.f / block_reduce(localSum, false);

    __syncthreads(); // all scores[] writes visible before the cross-thread read below

    // Pass 3: weighted V sum, one thread per output dim (normalizing by invSum
    // at the end is equivalent to normalizing scores first — same math, one
    // fewer pass over scores[]).
    for (int d = (int)threadIdx.x; d < headDim; d += (int)blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < seqLen; t++)
            acc += scores[t] * __half2float(V[(size_t)t * kvDim + kBase + d]);
        out[d] = acc * invSum;
    }
}
