/*
 * The wide decode block shared by gqa_attention.cu (FP16 device KV) and
 * gqa_attention_kv.cu (device KV in any of kv_format.cuh's formats): decode
 * rows, one row a block, each over its own cache, with 256 threads, 64 slots of
 * 4 lanes. Slot s takes keys lo + s, lo + s + 64, ... (from the 64-key stride
 * at or below lo), each with gqa_attention_tiled's per-key arithmetic and online
 * softmax, and the 64 slots are merged as the tiled kernel merges its 32. A
 * decode launch has one block per head and row, so with few heads (16 on
 * Qwen2.5-3B) a 128-thread block leaves each warp scheduler one warp and pays
 * every dependent latency in full; twice the warps per block halves each slot's
 * keys. A row's result depends on its own cache and length only, so a row
 * decoded in a batch is bit for bit the row decoded alone.
 *
 * F16 (an FP16 cache, gqa_attention.cu): each lane holds interleaved quads of
 * the head (quads lane, lane + 4, ...) and reads FP16 pairs, as the tiled
 * kernel's decode loop does, so every lane of a 96- or 80-wide head has work.
 * Otherwise (gqa_attention_kv.cu): each lane holds a contiguous chunk of the
 * head and reads whole blocks (gqa_decode_kv_wide), formats chosen per launch.
 * Both loops load a key's K and V up front, so their loads are in flight
 * together.
 */
#ifndef JUNO_GQA_DECODE_WIDE_CUH
#define JUNO_GQA_DECODE_WIDE_CUH

#include "gqa_attention.cuh"

#define GQD_THREADS 256
#define GQD_SLOTS (GQD_THREADS / GQA_LANES)

/* The two FP16 pairs of a quad loaded as one 8-byte word, as floats. */
static __device__ __forceinline__ float2 gqa_lo_pair(uint2 w) {
    return __half22float2(*reinterpret_cast<const half2*>(&w.x));
}
static __device__ __forceinline__ float2 gqa_hi_pair(uint2 w) {
    return __half22float2(*reinterpret_cast<const half2*>(&w.y));
}

/*
 * Decode keys [lo, seqLen) of an FP16 cache over SLOTS slots: gqa_attention_tiled's FP16 decode loop with the stride
 * SLOTS, the same arithmetic in the same order. Each lane loads its key's and value's quads up front, one 8-byte
 * read-only load a quad and no branch between them (a slot outside [lo, seqLen) reads row lo and discards it; a quad
 * past the head reads quad 0 and is never used), so the loads of a key are in flight together. A quad is 8-byte
 * aligned: kvDim and the head offset are multiples of 4 values. The loop is uniform across the block (every lane
 * reaches the shuffles); a slot whose key is outside [lo, seqLen) changes nothing.
 */
template <int DMAX, int SLOTS>
static __device__ __forceinline__ void gqa_decode_f16(const float4* qv, float4* acc, float& runMax, float& runSum,
        const half2* __restrict__ Kg, const half2* __restrict__ Vg, int kvPairs, int kPairBase, int lane, int split,
        int lo, int seqLen, int nQuads) {
    constexpr int QUADS = DMAX / 4 / GQA_LANES;
    for (int base = (lo / SLOTS) * SLOTS; base < seqLen; base += SLOTS) {
        const int pos = base + split;
        const bool valid = pos >= lo && pos < seqLen;
        const size_t row = (size_t)(valid ? pos : lo) * kvPairs + kPairBase;
        const uint2* K4 = reinterpret_cast<const uint2*>(Kg + row);
        const uint2* V4 = reinterpret_cast<const uint2*>(Vg + row);
        uint2 kw[QUADS];
        uint2 vw[QUADS];
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            const int c = lane + GQA_LANES * i;
            kw[i] = __ldg(K4 + (c < nQuads ? c : 0));
            vw[i] = __ldg(V4 + (c < nQuads ? c : 0));
        }
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            const int c = lane + GQA_LANES * i;
            if (valid && c < nQuads) {
                const float2 k0 = gqa_lo_pair(kw[i]);
                const float2 k1 = gqa_hi_pair(kw[i]);
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
                    const float2 v0 = gqa_lo_pair(vw[i]);
                    const float2 v1 = gqa_hi_pair(vw[i]);
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

/*
 * The four values of one quad from its quant word, the floats kv_quad_ints gives, bit for bit, without a conversion
 * instruction: each byte (q8_0: offset to unsigned; q4_0: the nibble) is placed under the exponent of 2^23 by one
 * byte permute, and 2^23 plus the offset is subtracted, which is exact for integers this small.
 */
template <int T>
static __device__ __forceinline__ float4 gqa_quad_floats(const unsigned int* words, int k, bool hi) {
    constexpr float BIAS = T == KV_Q8_0 ? 8388736.f : 8388616.f; // 2^23 + 128, 2^23 + 8
    const unsigned int u = T == KV_Q8_0 ? words[k] ^ 0x80808080u : (hi ? words[k] >> 4 : words[k]) & 0x0f0f0f0fu;
    return make_float4(__int_as_float(__byte_perm(u, 0x4b000000u, 0x7440u)) - BIAS,
            __int_as_float(__byte_perm(u, 0x4b000000u, 0x7441u)) - BIAS,
            __int_as_float(__byte_perm(u, 0x4b000000u, 0x7442u)) - BIAS,
            __int_as_float(__byte_perm(u, 0x4b000000u, 0x7443u)) - BIAS);
}

/*
 * A decode lane's chunk of one K or V row of format T: QUADS consecutive quads from element e0, whole blocks (QUADS 8,
 * the 128-wide entry) or the first or second half of one (QUADS 4, the 64-wide entry). With a head width that is a
 * multiple of 32 a lane's chunk lies wholly inside the head or wholly past it (a lane of a 96-wide head at the 128-wide
 * entry, two lanes of a 32-wide head at the 64-wide one). load() issues every load of the chunk, with no branch between
 * them; a chunk past the head loads the head's first block (eSafe) instead, so no load leaves the row, and its sums are
 * discarded by the caller. dot() and axpy() are kv_dot_chunk's and kv_axpy_chunk's arithmetic, in the same order, over
 * the loaded words, without their per-quad conditions.
 */
template <int T, int QUADS>
struct GqaChunk {
    static_assert(QUADS == 4 || QUADS == 8, "a decode chunk is half a block or one block");
    static constexpr int QW = T == KV_Q8_0 ? QUADS : 4; // quant words the chunk reads
    unsigned int w[QW];
    float d;

    __device__ __forceinline__ void load(const unsigned char* row, int e, int q0) {
        const unsigned char* blk = row + (size_t)(e / KV_BLOCK) * kv_block_bytes_t<T>();
        d = kv_scale(blk);
        kv_words<QW>(blk + 2 + (T == KV_Q8_0 ? 4 * q0 : 0), w);
    }

    __device__ __forceinline__ float dot(const float4* qv, int q0) const {
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < QUADS; q++) {
            const float4 k = gqa_quad_floats<T>(w, T == KV_Q8_0 ? q : (q & 3), q0 + q >= 4);
            s = fmaf(qv[q].x, k.x, s);
            s = fmaf(qv[q].y, k.y, s);
            s = fmaf(qv[q].z, k.z, s);
            s = fmaf(qv[q].w, k.w, s);
        }
        return fmaf(d, s, 0.f);
    }

    __device__ __forceinline__ void axpy(float wt, float4* acc, int q0) const {
        const float wd = wt * d;
#pragma unroll
        for (int q = 0; q < QUADS; q++) {
            const float4 v = gqa_quad_floats<T>(w, T == KV_Q8_0 ? q : (q & 3), q0 + q >= 4);
            acc[q].x = fmaf(wd, v.x, acc[q].x);
            acc[q].y = fmaf(wd, v.y, acc[q].y);
            acc[q].z = fmaf(wd, v.z, acc[q].z);
            acc[q].w = fmaf(wd, v.w, acc[q].w);
        }
    }
};

/* An FP16 side of a mixed pair: one 8-byte load a quad (the two FP16 pairs kv_load4_t reads). */
template <int QUADS>
struct GqaChunk<KV_F16, QUADS> {
    uint2 w[QUADS];

    __device__ __forceinline__ void load(const unsigned char* row, int e, int q0) {
#pragma unroll
        for (int i = 0; i < QUADS; i++)
            w[i] = __ldg(reinterpret_cast<const uint2*>(row + 2 * (size_t)(e + 4 * i)));
    }

    __device__ __forceinline__ float dot(const float4* qv, int q0) const {
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            const float2 a = gqa_lo_pair(w[i]);
            const float2 b = gqa_hi_pair(w[i]);
            part = fmaf(qv[i].x, a.x, part);
            part = fmaf(qv[i].y, a.y, part);
            part = fmaf(qv[i].z, b.x, part);
            part = fmaf(qv[i].w, b.y, part);
        }
        return part;
    }

    __device__ __forceinline__ void axpy(float wt, float4* acc, int q0) const {
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            const float2 a = gqa_lo_pair(w[i]);
            const float2 b = gqa_hi_pair(w[i]);
            acc[i].x = fmaf(wt, a.x, acc[i].x);
            acc[i].y = fmaf(wt, a.y, acc[i].y);
            acc[i].z = fmaf(wt, b.x, acc[i].z);
            acc[i].w = fmaf(wt, b.y, acc[i].w);
        }
    }
};

/*
 * Decode keys [lo, seqLen) of a cache whose K rows are format KT and V rows format VT over SLOTS slots: gqa_decode_kv's
 * loop (block-major lanes, the same keys per slot, the same arithmetic in the same order), with each key's K and V
 * chunks loaded up front, before the dot product, instead of V after the softmax update. A slot outside [lo, seqLen)
 * loads row lo and discards it. Needs a head width that is a multiple of 32, as gqa_decode_kv does; a chunk past the
 * head adds nothing to the key's score and its accumulator is never stored (the merge skips quads past the head).
 */
template <int DMAX, int KT, int VT, int SLOTS>
static __device__ __forceinline__ void gqa_decode_kv_wide(const float4* qv, float4* acc, float& runMax, float& runSum,
        const unsigned char* Kb, const unsigned char* Vb, int kvDim, int kElemBase, int lane, int split, int lo,
        int seqLen, int nQuads) {
    constexpr int QUADS = DMAX / 4 / GQA_LANES;
    const bool inHead = lane * QUADS < nQuads;
    const int e0 = inHead ? kElemBase + 4 * lane * QUADS : kElemBase;
    const int q0 = ((kElemBase + 4 * lane * QUADS) & (KV_BLOCK - 1)) >> 2; // 0, or 4 for the second half of a block
    for (int base = (lo / SLOTS) * SLOTS; base < seqLen; base += SLOTS) {
        const int pos = base + split;
        const bool valid = pos >= lo && pos < seqLen;
        const int row = valid ? pos : lo;
        GqaChunk<KT, QUADS> k;
        GqaChunk<VT, QUADS> v;
        k.load(kv_row_t<KT>(Kb, kvDim, row), e0, q0);
        v.load(kv_row_t<VT>(Vb, kvDim, row), e0, q0);
        const float dot = k.dot(qv, q0);
        float part = valid && inHead ? dot : 0.f;
        part += __shfl_xor_sync(GQA_FULL_MASK, part, 1);
        part += __shfl_xor_sync(GQA_FULL_MASK, part, 2);
        if (valid) {
            const float newMax = fmaxf(runMax, part);
            const float corr = exp2f(runMax - newMax);
            const float w = exp2f(part - newMax);
            // One fused multiply-add, written out: gqa_decode_kv's loop leaves the fusing to the compiler, which fused
            // it for four format pairs and not for K q4_0 with V f16, so this sum rounds as the old loop's did for
            // those four, and for every pair the same way whatever the compiler would choose.
            runSum = fmaf(runSum, corr, w);
#pragma unroll
            for (int i = 0; i < QUADS; i++) {
                acc[i].x *= corr;
                acc[i].y *= corr;
                acc[i].z *= corr;
                acc[i].w *= corr;
            }
            v.axpy(w, acc, q0);
            runMax = newMax;
        }
    }
}

/* One switch per launch, uniform across the grid, selects gqa_decode_kv_wide for the pair. */
template <int DMAX, int SLOTS>
static __device__ __forceinline__ void gqa_decode_kv_wide_dispatch(int kType, int vType, const float4* qv,
        float4* acc, float& runMax, float& runSum, const unsigned char* Kb, const unsigned char* Vb, int kvDim,
        int kElemBase, int lane, int split, int lo, int seqLen, int nQuads) {
#define GQD_DECODE_CASE(K, V) \
    case K * 3 + V: \
        gqa_decode_kv_wide<DMAX, K, V, SLOTS>(qv, acc, runMax, runSum, Kb, Vb, kvDim, kElemBase, lane, split, lo, \
                seqLen, nQuads); \
        break;
    switch (kType * 3 + vType) { GQA_KV_PAIRS(GQD_DECODE_CASE) }
#undef GQD_DECODE_CASE
}

/* Which quad of the head register i of a lane holds: interleaved over an FP16 cache, block-major otherwise. */
template <bool F16, int QUADS>
static __device__ __forceinline__ int gqa_wide_quad(int lane, int i) {
    if constexpr (F16)
        return lane + GQA_LANES * i;
    else
        return gqa_quad_block_major<QUADS>(lane, i);
}

template <int DMAX, bool F16>
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

    // Query quarter, pre-scaled by 1/sqrt(headDim) * log2(e), in the quads the decode loop reads.
    const float qScale = rsqrtf((float)headDim) * GQA_LOG2E;
    float4 qv[QUADS];
    float4 acc[QUADS];
    const float4* q4 = reinterpret_cast<const float4*>(qBatch + ((size_t)b * numHeads + h) * headDim);
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_wide_quad<F16, QUADS>(lane, i);
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
    if constexpr (F16) {
        gqa_decode_f16<DMAX, GQD_SLOTS>(qv, acc, runMax, runSum, reinterpret_cast<const half2*>(kPtrs[b]),
                reinterpret_cast<const half2*>(vPtrs[b]), kvDim >> 1, ((h / gqaRatio) * headDim) >> 1, lane, slot, lo,
                seqLen, nQuads);
    } else {
        const unsigned char* Kb = reinterpret_cast<const unsigned char*>(kPtrs[b]);
        const unsigned char* Vb = reinterpret_cast<const unsigned char*>(vPtrs[b]);
        gqa_decode_kv_wide_dispatch<DMAX, GQD_SLOTS>(kType, vType, qv, acc, runMax, runSum, Kb, Vb, kvDim,
                (h / gqaRatio) * headDim, lane, slot, lo, seqLen, nQuads);
    }

    // Merge the slots: weight slot s by exp2(max_s - max) and normalize (gqa_attention_tiled's merge).
    if (lane == 0) {
        slotMax[slot] = runMax;
        slotSum[slot] = runSum;
    }
#pragma unroll
    for (int i = 0; i < QUADS; i++) {
        const int c = gqa_wide_quad<F16, QUADS>(lane, i);
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

#endif /* JUNO_GQA_DECODE_WIDE_CUH */
