/*
 * GPU-resident grouped-query attention for Juno: a tiled, online-softmax
 * kernel that streams the keys and never materializes a score row.
 *
 * Same math as the CPU oracle (GqaMath.attend): scores = scale * q . K[t] over
 * the head's KV head (kvHead = h / gqaRatio), a max-subtracted softmax, and the
 * softmax-weighted sum of V[t]; causal masking is implicit in seqLens[b]. The
 * softmax is computed online (running maximum, running sum, rescaled
 * accumulator), so the device memory this kernel needs is independent of the
 * context length: no scratch beyond the query and output rows.
 *
 * Geometry. A block of 128 threads is 32 slots of 4 lanes. The block owns one
 * query head (blockIdx.y) and a tile of rowsPerBlock consecutive query rows
 * (blockIdx.x); the 32 slots are rowsPerBlock rows times splits = 32 /
 * rowsPerBlock key splits. Keys are staged through shared memory in tiles of
 * GQA_TILE rows, FP16 as stored; within a tile, split s of a row takes keys s,
 * s + splits, ... . The 4 lanes of a slot each hold an interleaved quarter of
 * the head (pairs l, l + 4, ...) of the query and of the output accumulator in
 * registers and combine their partial dot products with two shuffles.
 *
 *   rowsPerBlock = 32, splits = 1:  a prefill window, every row of the tile
 *                                   reading each staged key once.
 *   rowsPerBlock = 1,  splits = 32: a decode row (or --parallel streams, one
 *                                   block per row), the keys spread over the
 *                                   32 slots and merged at the end. No other
 *                                   row shares a key, so this case reads its
 *                                   keys straight from global memory instead
 *                                   of staging tiles between barriers (same
 *                                   keys, order and arithmetic as a 32-key
 *                                   tile, so the same result to the bit
 *                                   where the tile is 32 keys; the 16-key
 *                                   tiles of 256-wide heads gave a slot no
 *                                   key here, so this path is what makes
 *                                   them correct at decode).
 *
 * A tile of rows shares one K/V pointer (kPtrs[first row of the tile]): the
 * caller passes rowsPerBlock > 1 only when every row attends over the same
 * cache (a growing prefill window), and 1 otherwise.
 *
 * window: when > 0, row b attends over its last `window` keys only,
 * [max(0, seqLen - window), seqLen); 0 means no window. A window no shorter
 * than every row's context reads exactly the keys of window 0, in the same
 * order, so it is bit-identical to it.
 *
 * K/V are real IEEE FP16 (see DeviceKvCache); the staged tiles, Q, the softmax
 * and every accumulation are FP32. The scale folds log2(e) in so the softmax runs on
 * exp2f.
 *
 * Three instantiations bound the head width held in registers: gqa_attention_d64
 * (headDim <= 64; fewer registers, so more blocks per SM), gqa_attention_d128
 * (headDim <= 128, 32-key tiles) and gqa_attention_d256 (headDim <= 256, 16-key
 * tiles, so the tiles fit the 48 KB of static shared memory). headDim must be a
 * multiple of 4.
 *
 * Compile (Pascal+ reference SKU - sm_61, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/gqa_attention.ptx gqa_attention.cu
 */
#include <cuda_fp16.h>

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

template <int DMAX, int TILE>
static __device__ __forceinline__ void gqa_attention_tiled(
        const float* __restrict__ qBatch,
        const half* const* __restrict__ kPtrs,
        const half* const* __restrict__ vPtrs,
        const int* __restrict__ seqLens,
        float* __restrict__ outBatch,
        int batch, int numHeads, int gqaRatio, int headDim, int kvDim,
        int rowsPerBlock, int window) {
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
        const int c = lane + GQA_LANES * i;
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
    const int tileElems = TILE * nPairs;

    if (rowsPerBlock == 1) {
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
        const int c = lane + GQA_LANES * i;
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

#define GQA_ARGS \
        const float* __restrict__ qBatch,      /* [B][numHeads*headDim] */            \
        const half* const* __restrict__ kPtrs, /* [B] device ptrs -> [rows][kvDim] */ \
        const half* const* __restrict__ vPtrs, /* [B] device ptrs -> [rows][kvDim] */ \
        const int* __restrict__ seqLens,       /* [B] */                              \
        float* __restrict__ outBatch,          /* [B][numHeads*headDim] */            \
        int batch, int numHeads, int gqaRatio, int headDim, int kvDim,                \
        int rowsPerBlock, int window

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_d64(GQA_ARGS) {
    gqa_attention_tiled<64, 32>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_d128(GQA_ARGS) {
    gqa_attention_tiled<128, 32>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_d256(GQA_ARGS) {
    gqa_attention_tiled<256, 16>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window);
}
