/*
 * GPU-resident grouped-query attention over a device KV stored in any of
 * kv_format.cuh's formats (FP16, Q8_0 or Q4_0 blocks), chosen per launch for the
 * keys (kType) and for the values (vType). The gqa_attention_kv entries are
 * gqa_attention.cu's kernel, instantiated from gqa_attention.cuh with QUANT true:
 * the same geometry, the same online softmax and the same order of every
 * accumulation; only the reads of K and V go through kv_format.cuh, which
 * dequantizes each value in registers (decode rows) or while staging the FP32
 * shared tiles (prefill windows, eight values of a key per thread and read, the
 * same floats an FP16 cache of those values stages). The gqa_decode_kv entries
 * below run decode rows with a wider block, so gqa_attention_kv_d64 and _d128
 * leave the decode loop out (GqaAttentionKvKernel sends no decode row to them; a
 * row of one would still be correct, through the tiles); _d256 keeps it. No
 * scratch buffer holds dequantized keys or values.
 *
 * A separate module from gqa_attention.ptx, loaded only when a request's device
 * KV is quantized, so an FP16 launch loads and compiles exactly what it did.
 *
 * Entries gqa_attention_kv_d64, _d128 and _d256 take gqa_attention.cu's
 * arguments followed by kType and vType (KV_F16, KV_Q8_0 or KV_Q4_0). A
 * quantized format needs kvDim to be a multiple of 32.
 *
 * Entries gqa_decode_kv_d64 and _d128 run decode rows (one row a block, each
 * over its own cache) with twice the slots: 256 threads, 64 slots of 4 lanes,
 * the same per-slot loop at a stride of 64 keys and the same merge over 64
 * slots. A decode launch has one block per head and row, so with few heads
 * (16 on Qwen2.5-3B) a 128-thread block leaves each warp scheduler one warp and
 * pays every dependent latency in full; the dequantizing adds instructions to
 * that chain. Twice the warps per block halves each slot's keys. Arguments as
 * the entries above without rowsPerBlock (always 1). A row's result depends on
 * its own cache and length only, so a row decoded in a batch is bit for bit the
 * row decoded alone.
 *
 * Compile (Pascal+ reference SKU - sm_61, matching gqa_attention.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/gqa_attention_kv.ptx gqa_attention_kv.cu
 */
#include "gqa_attention.cuh"

#define GQA_KV_ARGS \
        const float* __restrict__ qBatch,      /* [B][numHeads*headDim] */                    \
        const half* const* __restrict__ kPtrs, /* [B] device ptrs -> rows of kType */          \
        const half* const* __restrict__ vPtrs, /* [B] device ptrs -> rows of vType */          \
        const int* __restrict__ seqLens,       /* [B] */                                      \
        float* __restrict__ outBatch,          /* [B][numHeads*headDim] */                    \
        int batch, int numHeads, int gqaRatio, int headDim, int kvDim,                        \
        int rowsPerBlock, int window, int kType, int vType

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_kv_d64(GQA_KV_ARGS) {
    gqa_attention_tiled<64, 32, true, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, kType, vType);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_kv_d128(GQA_KV_ARGS) {
    gqa_attention_tiled<128, 32, true, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, kType, vType);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_kv_d256(GQA_KV_ARGS) {
    gqa_attention_tiled<256, 16, true>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, kType, vType);
}

#define GQD_THREADS 256
#define GQD_SLOTS (GQD_THREADS / GQA_LANES)

template <int DMAX>
static __device__ __forceinline__ void gqa_decode_wide(
        const float* __restrict__ qBatch, const half* const* __restrict__ kPtrs, const half* const* __restrict__ vPtrs,
        const int* __restrict__ seqLens, float* __restrict__ outBatch, int batch, int numHeads, int gqaRatio,
        int headDim, int kvDim, int window, int kType, int vType) {
    constexpr int QUADS = DMAX / 4 / GQA_LANES;
    __shared__ __align__(16) float mergeAcc[GQD_SLOTS * DMAX];
    __shared__ float slotMax[GQD_SLOTS];
    __shared__ float slotSum[GQD_SLOTS];

    const int tid = (int)threadIdx.x;
    const int lane = tid & (GQA_LANES - 1);
    const int slot = tid / GQA_LANES;
    const int h = (int)blockIdx.y;
    const int b = (int)blockIdx.x;
    if (b >= batch)
        return; // the whole block: one row a block
    const int nQuads = headDim >> 2;
    const int seqLen = seqLens[b];
    const int lo = (window > 0 && seqLen > window) ? seqLen - window : 0;

    // Query quarter, pre-scaled by 1/sqrt(headDim) * log2(e); block-major quads, as the decode loop reads them.
    const float qScale = rsqrtf((float)headDim) * GQA_LOG2E;
    float4 qv[QUADS];
    float4 acc[QUADS];
    const float4* q4 = reinterpret_cast<const float4*>(qBatch + ((size_t)b * numHeads + h) * headDim);
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_quad_block_major<QUADS>(lane, i);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c < nQuads) {
            v = q4[c];
            v.x *= qScale;
            v.y *= qScale;
            v.z *= qScale;
            v.w *= qScale;
        }
        qv[i] = v;
        acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float runMax = -INFINITY;
    float runSum = 0.f;
    const unsigned char* Kb = reinterpret_cast<const unsigned char*>(kPtrs[b]);
    const unsigned char* Vb = reinterpret_cast<const unsigned char*>(vPtrs[b]);
    gqa_decode_kv_dispatch<DMAX, GQD_SLOTS>(kType, vType, qv, acc, runMax, runSum, Kb, Vb, kvDim,
            (h / gqaRatio) * headDim, lane, slot, lo, seqLen, lo, seqLen, nQuads);

    // Merge the slots: weight slot s by exp2(max_s - max) and normalize (gqa_attention_tiled's merge).
    if (lane == 0) {
        slotMax[slot] = runMax;
        slotSum[slot] = runSum;
    }
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_quad_block_major<QUADS>(lane, i);
        if (c < nQuads)
            *reinterpret_cast<float4*>(mergeAcc + slot * DMAX + 4 * c) = acc[i];
    }
    __syncthreads();
    for (int d = tid; d < headDim; d += GQD_THREADS) {
        float m = -INFINITY;
        for (int s = 0; s < GQD_SLOTS; s++)
            m = fmaxf(m, slotMax[s]);
        float den = 0.f, num = 0.f;
        for (int s = 0; s < GQD_SLOTS; s++) {
            const float sm = slotMax[s];
            if (sm == -INFINITY)
                continue;
            const float w = exp2f(sm - m);
            den = fmaf(slotSum[s], w, den);
            num = fmaf(mergeAcc[s * DMAX + d], w, num);
        }
        outBatch[((size_t)b * numHeads + h) * headDim + d] = num / den;
    }
}

#define GQD_ARGS \
        const float* __restrict__ qBatch, const half* const* __restrict__ kPtrs,                \
        const half* const* __restrict__ vPtrs, const int* __restrict__ seqLens,                 \
        float* __restrict__ outBatch, int batch, int numHeads, int gqaRatio, int headDim,       \
        int kvDim, int window, int kType, int vType

extern "C" __global__ void __launch_bounds__(GQD_THREADS, 1)
gqa_decode_kv_d64(GQD_ARGS) {
    gqa_decode_wide<64>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim, kvDim, window,
            kType, vType);
}

extern "C" __global__ void __launch_bounds__(GQD_THREADS, 1)
gqa_decode_kv_d128(GQD_ARGS) {
    gqa_decode_wide<128>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim, kvDim, window,
            kType, vType);
}
