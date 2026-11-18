/*
 * The FFN gate and up projections of one decode row in one launch: two packed Q4_K matrices of the same shape
 * against one Q8_1 input vector.
 *
 * Each output row is computed exactly as q4k_gemv computes it: the same lane mapping (8 lanes a super-block, 16
 * super-block slots over 4 warps), the same per-lane accumulation and the same warp and 4-way reductions, so the two
 * outputs equal two q4k_gemv launches bit for bit. What changes is how often the input is read: a lane loads its
 * Q8_1 slice, block sums and scales once and dots them with the gate row and with the up row.
 *
 * In q4k_gemv a row of at most PAIR_SPLIT_MAX_BLOCKS super-blocks (width up to 2,048) keeps only warps 0 and 1 busy.
 * Here warps 2 and 3 then take the next row index, so a block computes two gate and two up rows and reads the input
 * once per row index instead of once per output. Each row index reduces in its own slots, its warps' sums where
 * q4k_gemv's warps 0 and 1 put theirs and zeros where its idle warps add zeros.
 *
 * Compile (Pascal+ reference SKU - sm_61 for hardware dp4a, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/q4k_gemv_pair.ptx q4k_gemv_pair.cu
 */
#include "kquant_gemv.cuh"

#define PAIR_SPLIT_MAX_BLOCKS 8

/*
 * One super-block's contribution to a row: q4k_gemv's expression, d * (sc0*d8l*dl + sc1*d8h*dh) -
 * dmin * (mn0*d8l*sl + mn1*d8h*sh), with each rounding written out as q4k_gemv's machine code has it: the first
 * product of each sum fused into its add, and d * (...) fused into the subtraction. Left to the compiler, this kernel's
 * contraction fused other products and differed by an ulp.
 */
static __device__ __forceinline__ float q4k_block_term(const uint8_t* blk, int g, int sub,
        const int* uLo, const int* uHi, int sl, int sh, float d8l, float d8h) {
    uint4 hdr = *reinterpret_cast<const uint4*>(blk);
    uint4 q = *reinterpret_cast<const uint4*>(blk + 16 + g * 32 + sub);
    const int dl = dp4a4(q, uLo, 0x0F0F0F0F);
    const int dh = dp4a4(make_uint4(q.x >> 4, q.y >> 4, q.z >> 4, q.w >> 4), uHi, 0x0F0F0F0F);
    float d, dmin, sc0, mn0, sc1, mn1;
    kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
    const float s = __fmaf_rn(__fmul_rn(sc0, d8l), (float)dl, __fmul_rn(__fmul_rn(sc1, d8h), (float)dh));
    const float m = __fmaf_rn(__fmul_rn(mn0, d8l), (float)sl, __fmul_rn(__fmul_rn(mn1, d8h), (float)sh));
    return __fmaf_rn(d, s, -__fmul_rn(dmin, m));
}

extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32)
q4k_gemv_pair(
        const uint8_t* __restrict__ A0,
        const uint8_t* __restrict__ A1,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ y0,
        float* __restrict__ y1,
        int rows,
        int cols) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nb = cols >> 8;
    const bool split = nb <= PAIR_SPLIT_MAX_BLOCKS;
    const int group = split ? (warp >> 1) : 0;   // the block's row index this warp works on
    const int gw = split ? (warp & 1) : warp;     // the warp q4k_gemv would run this work on
    const int row = (int)blockIdx.x * (split ? 2 : 1) + group;
    const bool live = row < rows;

    const size_t rowBytes = (size_t)nb * Q4K_BLOCK_BYTES;
    const uint8_t* rp0 = A0 + (size_t)row * rowBytes;
    const uint8_t* rp1 = A1 + (size_t)row * rowBytes;
    const int quarter = lane >> 3;
    const int l = lane & 7;
    const int g = l >> 1;
    const int sub = (l & 1) * 16;

    float acc0 = 0.f;
    float acc1 = 0.f;
    if (live) {
        for (int b = gw * 4 + quarter; b < nb; b += 16) {
            const size_t off = (size_t)b * Q4K_BLOCK_BYTES;
            const uint8_t* blo = q8_block(xq8, b * 8 + 2 * g);
            const uint8_t* bhi = q8_block(xq8, b * 8 + 2 * g + 1);
            const int* uLo = reinterpret_cast<const int*>(blo + 4 + sub);
            const int* uHi = reinterpret_cast<const int*>(bhi + 4 + sub);
            const int lo[4] = { uLo[0], uLo[1], uLo[2], uLo[3] };
            const int hi[4] = { uHi[0], uHi[1], uHi[2], uHi[3] };
            const int sl = sum4_u8(lo);
            const int sh = sum4_u8(hi);
            const float d8l = q8_scale(blo);
            const float d8h = q8_scale(bhi);
            acc0 += q4k_block_term(rp0 + off, g, sub, lo, hi, sl, sh, d8l, d8h);
            acc1 += q4k_block_term(rp1 + off, g, sub, lo, hi, sl, sh, d8l, d8h);
        }
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    __shared__ float s0[2][KQ_WARPS];
    __shared__ float s1[2][KQ_WARPS];
    if (lane == 0) {
        s0[group][gw] = acc0;
        s1[group][gw] = acc1;
        if (split) {
            s0[group][gw + 2] = 0.f;
            s1[group][gw + 2] = 0.f;
        }
    }
    __syncthreads();
    if (gw == 0 && live) {
        float v0 = (lane < KQ_WARPS) ? s0[group][lane] : 0.f;
        float v1 = (lane < KQ_WARPS) ? s1[group][lane] : 0.f;
        v0 = warp_sum(v0);
        v1 = warp_sum(v1);
        if (lane == 0) {
            y0[row] = v0;
            y1[row] = v1;
        }
    }
}
