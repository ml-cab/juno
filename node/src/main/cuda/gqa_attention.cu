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
#include "gqa_attention.cuh"

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
    gqa_attention_tiled<64, 32, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, KV_F16, KV_F16);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_d128(GQA_ARGS) {
    gqa_attention_tiled<128, 32, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, KV_F16, KV_F16);
}

extern "C" __global__ void __launch_bounds__(GQA_THREADS)
gqa_attention_d256(GQA_ARGS) {
    gqa_attention_tiled<256, 16, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim,
            kvDim, rowsPerBlock, window, KV_F16, KV_F16);
}
