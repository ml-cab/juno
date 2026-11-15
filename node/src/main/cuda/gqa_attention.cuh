/*
 * The tiled, online-softmax grouped-query attention kernel body shared by
 * gqa_attention.cu (FP16 device KV) and gqa_attention_kv.cu (device KV in any of
 * kv_format.cuh's formats, chosen per launch for K and for V). The geometry, the
 * arithmetic and the order of every accumulation are described in
 * gqa_attention.cu. QUANT changes only how a key or value is read: through
 * kv_format.cuh's kv_row and kv_load2/kv_load4 instead of FP16 pairs, so a
 * quantized cache is dequantized in registers (decode) or into the FP32 shared
 * tiles it is staged through (prefill), and nowhere else. With QUANT false the
 * kernel is the FP16 kernel unchanged; kType and vType are then unused.
 */
#ifndef JUNO_GQA_ATTENTION_CUH
#define JUNO_GQA_ATTENTION_CUH

#include <cuda_fp16.h>
#include "kv_format.cuh"

#define GQA_THREADS 128
#define GQA_LANES 4
#define GQA_SLOTS (GQA_THREADS / GQA_LANES)
#define GQA_FULL_MASK 0xffffffffu
#define GQA_LOG2E 1.4426950408889634f

/*
 * Shared tile row stride in floats: the head plus 4. With slots of one warp
 * reading consecutive keys at once (the split layout), the extra 16 bytes move
 * each key's row to other banks.
 */
template <int DMAX, int TILE>
struct GqaTile {
    static constexpr int LD = DMAX + 4;
    static constexpr int FLOATS = TILE * LD;
};

/*
 * Decode keys over a cache whose K rows are format KT and V rows format VT: the
 * FP16 decode loop of gqa_attention_tiled (same keys per slot, same online softmax,
 * same merge), with each lane holding a contiguous chunk of the head instead of
 * interleaved quads, so that it reads whole blocks: one scale and a few aligned
 * words per block (kv_dot_chunk, kv_axpy_chunk). The formats are template
 * arguments, so the loop has no format branch. Needs a head width that is a
 * multiple of 32 (DeviceKvFormat refuses quantized formats otherwise).
 */
template <int QUADS>
static __device__ __forceinline__ int gqa_quad_block_major(int lane, int i) {
    return lane * QUADS + i;
}
template <int DMAX, int KT, int VT, int SLOTS>
static __device__ __forceinline__ void gqa_decode_kv(const float4* qv, float4* acc, float& runMax, float& runSum,
        const unsigned char* Kb, const unsigned char* Vb, int kvDim, int kElemBase, int lane, int split, int lo,
        int seqLen, int rangeLo, int hi, int nQuads) {
    constexpr int QUADS = DMAX / 4 / GQA_LANES;
    // Block-major quads (gqa_quad): this lane holds quads [lane * QUADS, lane * QUADS + QUADS) of the head.
    const int e0 = kElemBase + 4 * lane * QUADS;
    const int nValid = nQuads - lane * QUADS;
    for (int base = (rangeLo / SLOTS) * SLOTS; base < hi; base += SLOTS) {
        const int pos = base + split;
        const bool valid = pos >= lo && pos < seqLen;
        float part = valid ? kv_dot_chunk<KT, QUADS>(kv_row_t<KT>(Kb, kvDim, pos), e0, qv, nValid) : 0.f;
        part += __shfl_xor_sync(GQA_FULL_MASK, part, 1);
        part += __shfl_xor_sync(GQA_FULL_MASK, part, 2);
        if (valid) {
            const float newMax = fmaxf(runMax, part);
            const float corr = exp2f(runMax - newMax);
            runSum *= corr;
            const float w = exp2f(part - newMax);
            runSum += w;
#pragma unroll
            for (int i = 0; i < QUADS; i++) {
                acc[i].x *= corr;
                acc[i].y *= corr;
                acc[i].z *= corr;
                acc[i].w *= corr;
            }
            kv_axpy_chunk<VT, QUADS>(kv_row_t<VT>(Vb, kvDim, pos), e0, w, acc, nValid);
            runMax = newMax;
        }
    }
}

/* Stages one tile of keys and values (FP32, shared) from a cache of formats KT and VT. */
template <int KT, int VT>
static __device__ __forceinline__ void gqa_stage_kv(float* Ks, float* Vs, int LD, const unsigned char* Kb,
        const unsigned char* Vb, int kvDim, int kElemBase, int t0, int hi, int nPairs, int tileElems, int tid) {
    for (int e = tid; e < tileElems; e += GQA_THREADS) {
        const int t = e / nPairs;
        const int p = e - t * nPairs;
        const int pos = t0 + t;
        float2 kk = make_float2(0.f, 0.f);
        float2 vv = kk;
        if (pos < hi) {
            kk = kv_load2_t<KT>(kv_row_t<KT>(Kb, kvDim, pos), kElemBase + 2 * p);
            vv = kv_load2_t<VT>(kv_row_t<VT>(Vb, kvDim, pos), kElemBase + 2 * p);
        }
        *reinterpret_cast<float2*>(Ks + t * LD + 2 * p) = kk;
        *reinterpret_cast<float2*>(Vs + t * LD + 2 * p) = vv;
    }
}

#define GQA_KV_PAIRS(X) \
    X(KV_F16, KV_F16) X(KV_F16, KV_Q8_0) X(KV_F16, KV_Q4_0) \
    X(KV_Q8_0, KV_F16) X(KV_Q8_0, KV_Q8_0) X(KV_Q8_0, KV_Q4_0) \
    X(KV_Q4_0, KV_F16) X(KV_Q4_0, KV_Q8_0) X(KV_Q4_0, KV_Q4_0)

/* One switch per launch, uniform across the grid, selects the decode loop for the pair (SLOTS slots a block). */
template <int DMAX, int SLOTS>
static __device__ __forceinline__ void gqa_decode_kv_dispatch(int kType, int vType, const float4* qv, float4* acc,
        float& runMax, float& runSum, const unsigned char* Kb, const unsigned char* Vb, int kvDim, int kElemBase,
        int lane, int split, int lo, int seqLen, int rangeLo, int hi, int nQuads) {
#define GQA_DECODE_CASE(K, V) \
    case K * 3 + V: \
        gqa_decode_kv<DMAX, K, V, SLOTS>(qv, acc, runMax, runSum, Kb, Vb, kvDim, kElemBase, lane, split, lo, seqLen, \
                rangeLo, hi, nQuads); \
        break;
    switch (kType * 3 + vType) { GQA_KV_PAIRS(GQA_DECODE_CASE) }
#undef GQA_DECODE_CASE
}

static __device__ __forceinline__ void gqa_stage_kv_dispatch(int kType, int vType, float* Ks, float* Vs, int LD,
        const unsigned char* Kb, const unsigned char* Vb, int kvDim, int kElemBase, int t0, int hi, int nPairs,
        int tileElems, int tid) {
#define GQA_STAGE_CASE(K, V) \
    case K * 3 + V: \
        gqa_stage_kv<K, V>(Ks, Vs, LD, Kb, Vb, kvDim, kElemBase, t0, hi, nPairs, tileElems, tid); \
        break;
    switch (kType * 3 + vType) { GQA_KV_PAIRS(GQA_STAGE_CASE) }
#undef GQA_STAGE_CASE
}

/*
 * Which quad of the head register i of a lane holds. Interleaved (quads lane,
 * lane + 4, ...) everywhere, except a decode row over a quantized cache, which is
 * block-major so that a lane reads whole blocks (gqa_decode_kv).
 */
template <bool QUANT, int QUADS>
static __device__ __forceinline__ int gqa_quad(int lane, int i, int rowsPerBlock) {
    if constexpr (QUANT) {
        if (rowsPerBlock == 1)
            return gqa_quad_block_major<QUADS>(lane, i);
    }
    return lane + GQA_LANES * i;
}

template <int DMAX, int TILE, bool QUANT>
static __device__ __forceinline__ void gqa_attention_tiled(
        const float* __restrict__ qBatch,
        const half* const* __restrict__ kPtrs,
        const half* const* __restrict__ vPtrs,
        const int* __restrict__ seqLens,
        float* __restrict__ outBatch,
        int batch, int numHeads, int gqaRatio, int headDim, int kvDim,
        int rowsPerBlock, int window, int kType, int vType) {
    constexpr int QUADS = DMAX / 4 / GQA_LANES;   // groups of four values per lane, at most
    constexpr int LD = GqaTile<DMAX, TILE>::LD;
    static_assert(2 * TILE * LD >= GQA_SLOTS * DMAX, "tiles double as the merge buffer");

    // K tile, V tile (FP32); reused as the merge buffer ([slot][DMAX] floats) at the end.
    __shared__ __align__(16) float tiles[2 * GqaTile<DMAX, TILE>::FLOATS];
    __shared__ float slotMax[GQA_SLOTS];
    __shared__ float slotSum[GQA_SLOTS];
    __shared__ int rangeLo, rangeHi;

    float* Ks = tiles;
    float* Vs = tiles + GqaTile<DMAX, TILE>::FLOATS;

    const int tid = (int)threadIdx.x;
    const int lane = tid & (GQA_LANES - 1);
    const int slot = tid / GQA_LANES;
    const int splits = GQA_SLOTS / rowsPerBlock;
    const int r = slot / splits;
    const int split = slot - r * splits;
    const int h = (int)blockIdx.y;
    const int b0 = (int)blockIdx.x * rowsPerBlock;
    const int b = b0 + r;
    const bool rowValid = b < batch;
    const int nQuads = headDim >> 2;
    const int nPairs = headDim >> 1;
    const int kPairBase = ((h / gqaRatio) * headDim) >> 1;
    const int kvPairs = kvDim >> 1;

    const int seqLen = rowValid ? seqLens[b] : 0;
    const int lo = (window > 0 && seqLen > window) ? seqLen - window : 0;

    if (tid == 0) {
        int bl = 0x7fffffff, bh = 0;
        for (int i = 0; i < rowsPerBlock && b0 + i < batch; i++) {
            const int L = seqLens[b0 + i];
            const int l0 = (window > 0 && L > window) ? L - window : 0;
            bl = min(bl, l0);
            bh = max(bh, L);
        }
        rangeLo = bl;
        rangeHi = bh;
    }

    // Query quarter, pre-scaled by 1/sqrt(headDim) * log2(e).
    const float qScale = rsqrtf((float)headDim) * GQA_LOG2E;
    float4 qv[QUADS];
    float4 acc[QUADS];
    const float4* q4 = reinterpret_cast<const float4*>(qBatch + ((size_t)b * numHeads + h) * headDim);
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_quad<QUANT, QUADS>(lane, i, rowsPerBlock);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (rowValid && c < nQuads) {
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

    __syncthreads();
    const int hi = rangeHi;
    const int keysPerSlot = TILE / splits;
    const half2* Kg = reinterpret_cast<const half2*>(kPtrs[b0]);
    const half2* Vg = reinterpret_cast<const half2*>(vPtrs[b0]);
    // QUANT: rows and elements of a quantized cache are addressed and read by kv_format.cuh.
    const unsigned char* Kb = reinterpret_cast<const unsigned char*>(kPtrs[b0]);
    const unsigned char* Vb = reinterpret_cast<const unsigned char*>(vPtrs[b0]);
    const int kElemBase = (h / gqaRatio) * headDim;
    const int tileElems = TILE * nPairs;

    if constexpr (QUANT) {
        // Decode over a quantized cache: the loop below, with the formats fixed at compile time.
        if (rowsPerBlock == 1)
            gqa_decode_kv_dispatch<DMAX, GQA_SLOTS>(kType, vType, qv, acc, runMax, runSum, Kb, Vb, kvDim, kElemBase, lane, split,
                    lo, seqLen, rangeLo, hi, nQuads);
    } else if (rowsPerBlock == 1) {
        // Decode: one row, so no other row shares a staged key. Each slot reads its keys
        // straight from global memory: slot s takes keys base + s, base + s + 32, ... with
        // the same arithmetic, in the same order, as the tile loop below at splits = 32 and
        // TILE = 32, so the result is the same to the bit; staging those keys through
        // shared memory between two barriers serialises the loads and buys nothing here.
        // The loop is uniform across the block (every lane reaches the shuffles); a slot
        // whose key is outside [lo, seqLen) changes nothing.
        for (int base = (rangeLo / GQA_SLOTS) * GQA_SLOTS; base < hi; base += GQA_SLOTS) {
            const int pos = base + split;
            const bool valid = pos >= lo && pos < seqLen;
            const size_t row = (size_t)(valid ? pos : lo) * kvPairs + kPairBase;
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < QUADS; i++) {
                const int c = lane + GQA_LANES * i;
                if (valid && c < nQuads) {
                    const float2 k0 = __half22float2(Kg[row + 2 * c]);
                    const float2 k1 = __half22float2(Kg[row + 2 * c + 1]);
                    part = fmaf(qv[i].x, k0.x, part);
                    part = fmaf(qv[i].y, k0.y, part);
                    part = fmaf(qv[i].z, k1.x, part);
                    part = fmaf(qv[i].w, k1.y, part);
                }
            }
            part += __shfl_xor_sync(GQA_FULL_MASK, part, 1);
            part += __shfl_xor_sync(GQA_FULL_MASK, part, 2);
            if (valid) {
                const float newMax = fmaxf(runMax, part);
                const float corr = exp2f(runMax - newMax);
                runSum *= corr;
                const float w = exp2f(part - newMax);
                runSum += w;
#pragma unroll
                for (int i = 0; i < QUADS; i++) {
                    const int c = lane + GQA_LANES * i;
                    acc[i].x *= corr;
                    acc[i].y *= corr;
                    acc[i].z *= corr;
                    acc[i].w *= corr;
                    if (c < nQuads) {
                        const float2 v0 = __half22float2(Vg[row + 2 * c]);
                        const float2 v1 = __half22float2(Vg[row + 2 * c + 1]);
                        acc[i].x = fmaf(w, v0.x, acc[i].x);
                        acc[i].y = fmaf(w, v0.y, acc[i].y);
                        acc[i].z = fmaf(w, v1.x, acc[i].z);
                        acc[i].w = fmaf(w, v1.y, acc[i].w);
                    }
                }
                runMax = newMax;
            }
        }
    }

    for (int t0 = (rangeLo / TILE) * TILE; rowsPerBlock > 1 && t0 < hi; t0 += TILE) {
        __syncthreads(); // the previous tile is no longer read
        if constexpr (QUANT) {
            gqa_stage_kv_dispatch(kType, vType, Ks, Vs, LD, Kb, Vb, kvDim, kElemBase, t0, hi, nPairs, tileElems, tid);
        } else {
            for (int e = tid; e < tileElems; e += GQA_THREADS) {
                const int t = e / nPairs;
                const int p = e - t * nPairs;
                const int pos = t0 + t;
                float2 kk = make_float2(0.f, 0.f);
                float2 vv = kk;
                if (pos < hi) {
                    const size_t g = (size_t)pos * kvPairs + kPairBase + p;
                    kk = __half22float2(Kg[g]);
                    vv = __half22float2(Vg[g]);
                }
                *reinterpret_cast<float2*>(Ks + t * LD + 2 * p) = kk;
                *reinterpret_cast<float2*>(Vs + t * LD + 2 * p) = vv;
            }
        }
        __syncthreads();

        // Scores of this slot's keys (log2 domain); keys outside [lo, seqLen) are -inf.
        float sc[TILE];
        float tileMax = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < TILE; jj++) {
            float s = -INFINITY;
            if (jj < keysPerSlot) { // uniform across the block
                const int j = jj * splits + split;
                const float4* krow = reinterpret_cast<const float4*>(Ks + j * LD);
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < QUADS; i++) {
                    const int c = lane + GQA_LANES * i;
                    if (c < nQuads) {
                        const float4 kf = krow[c];
                        part = fmaf(qv[i].x, kf.x, part);
                        part = fmaf(qv[i].y, kf.y, part);
                        part = fmaf(qv[i].z, kf.z, part);
                        part = fmaf(qv[i].w, kf.w, part);
                    }
                }
                part += __shfl_xor_sync(GQA_FULL_MASK, part, 1);
                part += __shfl_xor_sync(GQA_FULL_MASK, part, 2);
                const int pos = t0 + j;
                if (pos >= lo && pos < seqLen)
                    s = part;
            }
            sc[jj] = s;
            tileMax = fmaxf(tileMax, s);
        }

        // Online softmax update; a slot with no key in range this tile changes nothing.
        if (tileMax != -INFINITY) {
            const float newMax = fmaxf(runMax, tileMax);
            const float corr = exp2f(runMax - newMax); // 0 on the first contributing tile
            runSum *= corr;
#pragma unroll
            for (int i = 0; i < QUADS; i++) {
                acc[i].x *= corr;
                acc[i].y *= corr;
                acc[i].z *= corr;
                acc[i].w *= corr;
            }
#pragma unroll
            for (int jj = 0; jj < TILE; jj++) {
                if (jj < keysPerSlot && sc[jj] != -INFINITY) {
                    const float w = exp2f(sc[jj] - newMax);
                    runSum += w;
                    const float4* vrow = reinterpret_cast<const float4*>(Vs + (jj * splits + split) * LD);
#pragma unroll
                    for (int i = 0; i < QUADS; i++) {
                        const int c = lane + GQA_LANES * i;
                        if (c < nQuads) {
                            const float4 vf = vrow[c];
                            acc[i].x = fmaf(w, vf.x, acc[i].x);
                            acc[i].y = fmaf(w, vf.y, acc[i].y);
                            acc[i].z = fmaf(w, vf.z, acc[i].z);
                            acc[i].w = fmaf(w, vf.w, acc[i].w);
                        }
                    }
                }
            }
            runMax = newMax;
        }
    }

    // Merge the splits of each row: weight split s by exp2(max_s - max) and normalize.
    __syncthreads(); // tiles are reused as the merge buffer
    float* mergeAcc = tiles;
    if (lane == 0) {
        slotMax[slot] = runMax;
        slotSum[slot] = runSum;
    }
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_quad<QUANT, QUADS>(lane, i, rowsPerBlock);
        if (c < nQuads)
            *reinterpret_cast<float4*>(mergeAcc + slot * DMAX + 4 * c) = acc[i];
    }
    __syncthreads();

    const int items = rowsPerBlock * headDim;
    for (int it = tid; it < items; it += GQA_THREADS) {
        const int row = it / headDim;
        const int d = it - row * headDim;
        const int ob = b0 + row;
        if (ob >= batch)
            continue;
        const int first = row * splits;
        float m = -INFINITY;
        for (int s = 0; s < splits; s++)
            m = fmaxf(m, slotMax[first + s]);
        float den = 0.f, num = 0.f;
        for (int s = 0; s < splits; s++) {
            const float sm = slotMax[first + s];
            if (sm == -INFINITY)
                continue;
            const float w = exp2f(sm - m);
            den = fmaf(slotSum[first + s], w, den);
            num = fmaf(mergeAcc[(first + s) * DMAX + d], w, num);
        }
        outBatch[((size_t)ob * numHeads + h) * headDim + d] = num / den;
    }
}

#endif /* JUNO_GQA_ATTENTION_CUH */
