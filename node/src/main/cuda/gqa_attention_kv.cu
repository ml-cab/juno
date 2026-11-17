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
 * over its own cache) through gqa_decode_wide.cuh's wide block: 256 threads,
 * 64 slots of 4 lanes, the same per-slot loop at a stride of 64 keys and the
 * same merge over 64 slots (the dequantizing adds instructions to each slot's
 * dependent chain, which the extra warps hide). Arguments as the entries above
 * without rowsPerBlock (always 1). A row's result depends on its own cache and
 * length only, so a row decoded in a batch is bit for bit the row decoded
 * alone.
 *
 * Compile (Pascal+ reference SKU - sm_61, matching gqa_attention.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/gqa_attention_kv.ptx gqa_attention_kv.cu
 */
#include "gqa_attention.cuh"
#include "gqa_decode_wide.cuh"

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

#define GQD_ARGS \
        const float* __restrict__ qBatch, const half* const* __restrict__ kPtrs,                \
        const half* const* __restrict__ vPtrs, const int* __restrict__ seqLens,                 \
        float* __restrict__ outBatch, int batch, int numHeads, int gqaRatio, int headDim,       \
        int kvDim, int window, int kType, int vType

/*
 * Two blocks a multiprocessor: a decode launch has one block per head and row, and with 32 heads on 20 multiprocessors
 * a block that takes a multiprocessor alone runs the launch in two waves. The up-front K and V loads asked for 144
 * registers at the 128-wide entry, one block's worth; capped at 128 they spill nothing.
 */
extern "C" __global__ void __launch_bounds__(GQD_THREADS, 2)
gqa_decode_kv_d64(GQD_ARGS) {
    gqa_decode_wide<64, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim, kvDim, window,
            kType, vType);
}

extern "C" __global__ void __launch_bounds__(GQD_THREADS, 2)
gqa_decode_kv_d128(GQD_ARGS) {
    gqa_decode_wide<128, false>(qBatch, kPtrs, vPtrs, seqLens, outBatch, batch, numHeads, gqaRatio, headDim, kvDim, window,
            kType, vType);
}
