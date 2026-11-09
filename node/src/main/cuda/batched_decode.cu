/*
 * Kernels of Juno's decode batch device region: N independent decode requests on
 * the device at once, each at its own position, each attending over its own KV
 * cache. The batch's matmuls, RoPE and KV append live here; every other operation
 * of the batched step is an existing kernel fed N rows.
 *
 * A request's reply must not depend on what is batched with it, so each kernel here
 * computes, for every row, what one request's decode computes, bit for bit:
 *
 *   q?k_gemv_rows_NC
 *               Y[c][r] = sum_k A[r][k] * x[c][k] for up to NC activation rows x
 *               packed as Q8_1 (row c's blocks at c * cols/32), by the same
 *               per-lane accumulation and the same reduction tree as q4k_gemv.cu's
 *               GEMVs for one row. Only the weights are shared: each lane reads its
 *               16 packed bytes once and integer-dots them against all NC
 *               activation rows, so a batch reads each weight once per layer
 *               instead of once per row. The float accumulation is copied
 *               expression for expression from q4k_gemv.cu, so that every row
 *               takes exactly the float operations a single request's GEMV takes
 *               (see KQuantGemvRowsTest, which holds the two to bit equality).
 *   rope_rows, rope_split_half_rows, rope_ext_split_half_rows
 *               rope.cu's three rotations with a per-row position, seqLens[r] - 1
 *               (the batch's rows are different requests, not consecutive positions).
 *   kv_append_rows
 *               the K and V rows cast to FP16 into each request's own mirror at its
 *               own position: prefill_window.cu's to_half for N caches in one launch.
 *
 * Mapping of the GEMV kernels (Pascal decode, as q4k_gemv.cu): 128 threads = 4 warps
 * per block, one weight row per block; lane (quarter, l) of warp w owns the same
 * packed bytes of the same super-blocks as in q4k_gemv.cu, and holds one accumulator
 * per activation row.
 *
 * Compile (Pascal+ reference SKU - sm_61 for hardware dp4a, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/batched_decode.ptx batched_decode.cu
 */
#include <stdint.h>
#include <cuda_fp16.h>

#define KQ_WARPS 4
#define Q4K_BLOCK_BYTES 144
#define Q5K_BLOCK_BYTES 176
#define Q6K_SLOT_BYTES 224
#define Q8_1_BLOCK_BYTES 36
#define BD_THREADS 256

#define KQ_Q4K 0
#define KQ_Q5K 1
#define KQ_Q6K 2

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    return c
        + ((a << 24) >> 24) * ((b << 24) >> 24)
        + ((a << 16) >> 24) * ((b << 16) >> 24)
        + ((a <<  8) >> 24) * ((b <<  8) >> 24)
        + ( a        >> 24) * ( b        >> 24);
#endif
}

static __device__ __forceinline__ uint32_t byte_of(uint32_t w, int k) {
    return (w >> (8 * k)) & 0xFFu;
}

/* As q4k_gemv.cu: d, dmin and the (scale, min) pairs of sub-blocks 2g and 2g+1. */
static __device__ __forceinline__ void kq_affine_scales(uint4 hdr, int g,
        float& d, float& dmin, float& sc0, float& mn0, float& sc1, float& mn1) {
    half2 dd = *reinterpret_cast<half2*>(&hdr.x);
    d = __low2float(dd);
    dmin = __high2float(dd);
    const int k = (2 * g) & 3;
    const bool hi = g >= 2;
    uint32_t b0 = byte_of(hdr.y, k), b1 = byte_of(hdr.z, k), b2 = byte_of(hdr.w, k);
    uint32_t b0n = byte_of(hdr.y, k + 1), b1n = byte_of(hdr.z, k + 1), b2n = byte_of(hdr.w, k + 1);
    sc0 = (float)(hi ? ((b2 & 0xF) | ((b0 >> 6) << 4)) : (b0 & 63));
    mn0 = (float)(hi ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63));
    sc1 = (float)(hi ? ((b2n & 0xF) | ((b0n >> 6) << 4)) : (b0n & 63));
    mn1 = (float)(hi ? ((b2n >> 4) | ((b1n >> 6) << 4)) : (b1n & 63));
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ int dp4a4(uint4 q, const int* u, int mask) {
    return dp4a(q.x & mask, u[0],
           dp4a(q.y & mask, u[1],
           dp4a(q.z & mask, u[2],
           dp4a(q.w & mask, u[3], 0))));
}

static __device__ __forceinline__ int sum4_u8(const int* u) {
    const int ones = 0x01010101;
    return dp4a(ones, u[0], dp4a(ones, u[1], dp4a(ones, u[2], dp4a(ones, u[3], 0))));
}

static __device__ __forceinline__ const uint8_t* q8_block(const uint8_t* xq8, int blk) {
    return xq8 + (size_t)blk * Q8_1_BLOCK_BYTES;
}

static __device__ __forceinline__ float q8_scale(const uint8_t* blk) {
    return __half2float(*reinterpret_cast<const half*>(blk));
}

static __device__ __forceinline__ uint32_t q5_bits(uint32_t qh, int bit) {
    return ((qh >> bit) & 0x01010101u) << 4;
}

static __device__ __forceinline__ uint32_t q6_assemble(uint32_t nib, uint32_t qh, int shift) {
    return nib | (((qh >> shift) & 0x03030303u) << 4);
}

static __device__ __forceinline__ int q6_signed(uint32_t v) {
    return __vsubss4((int)v, 0x20202020);
}

static __device__ __forceinline__ int dp4a2_signed(uint32_t a0, uint32_t a1, const int* u) {
    return dp4a(q6_signed(a0), u[0], dp4a(q6_signed(a1), u[1], 0));
}

/*
 * q4k_gemv.cu's reduce_and_store for each of NC accumulators: a warp sum, then the
 * four warps' sums reduced by warp 0. Row c of the result goes to Y[c * ldc + row].
 */
template <int NC>
static __device__ __forceinline__ void reduce_and_store_rows(const float* acc, int lane, int warp, int row,
        float* Y, int batch, int ldc) {
    __shared__ float wsum[NC * KQ_WARPS];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float v = warp_sum(acc[c]);
        if (lane == 0)
            wsum[c * KQ_WARPS + warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = (lane < KQ_WARPS) ? wsum[c * KQ_WARPS + lane] : 0.f;
            v = warp_sum(v);
            if (lane == 0 && c < batch)
                Y[(size_t)c * ldc + row] = v;
        }
    }
}

/*
 * One weight row against up to NC activation rows. The weight bytes of a lane are
 * loaded once per super-block; everything downstream of them is q4k_gemv.cu's
 * single-row arithmetic, once per activation row.
 */
template <int TYPE, int NC>
static __device__ __forceinline__ void kq_gemv_rows(
        const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ Y,
        int rows, int cols, int batch, int ldc) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    const int nb = cols >> 8;
    const size_t blockBytes = TYPE == KQ_Q4K ? Q4K_BLOCK_BYTES : TYPE == KQ_Q5K ? Q5K_BLOCK_BYTES : Q6K_SLOT_BYTES;
    const uint8_t* rp = A + (size_t)row * ((size_t)nb * blockBytes);
    const size_t q8RowBytes = (size_t)(cols >> 5) * Q8_1_BLOCK_BYTES;
    const int quarter = lane >> 3;
    const int l = lane & 7;

    /* Q4_K / Q5_K lane layout */
    const int g = l >> 1;
    const int sub = (l & 1) * 16;
    const int bitLo = 2 * g;
    const int bitHi = 2 * g + 1;
    /* Q6_K lane layout */
    const int n = l >> 2;
    const int l0 = (l & 3) * 8;
    const int is = l0 >> 4;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; c++)
        acc[c] = 0.f;

    for (int b = warp * 4 + quarter; b < nb; b += 16) {
        const size_t off = (size_t)b * blockBytes;
        if (TYPE == KQ_Q6K) {
            const uint8_t* blk = rp + off;
            uint2 qla = *reinterpret_cast<const uint2*>(blk + n * 64 + l0);
            uint2 qlb = *reinterpret_cast<const uint2*>(blk + n * 64 + 32 + l0);
            uint2 qh = *reinterpret_cast<const uint2*>(blk + 128 + n * 32 + l0);
            uint2 sc = *reinterpret_cast<const uint2*>(blk + 192 + n * 8);
            uint32_t dbits = *reinterpret_cast<const uint16_t*>(blk + 208);
            const uint32_t p1a = q6_assemble(qla.x & 0x0F0F0F0Fu, qh.x, 0);
            const uint32_t p1b = q6_assemble(qla.y & 0x0F0F0F0Fu, qh.y, 0);
            const uint32_t p2a = q6_assemble(qlb.x & 0x0F0F0F0Fu, qh.x, 2);
            const uint32_t p2b = q6_assemble(qlb.y & 0x0F0F0F0Fu, qh.y, 2);
            const uint32_t p3a = q6_assemble((qla.x >> 4) & 0x0F0F0F0Fu, qh.x, 4);
            const uint32_t p3b = q6_assemble((qla.y >> 4) & 0x0F0F0F0Fu, qh.y, 4);
            const uint32_t p4a = q6_assemble((qlb.x >> 4) & 0x0F0F0F0Fu, qh.x, 6);
            const uint32_t p4b = q6_assemble((qlb.y >> 4) & 0x0F0F0F0Fu, qh.y, 6);
            const float s1 = (float)(int)(int8_t)byte_of(sc.x, is);
            const float s2 = (float)(int)(int8_t)byte_of(sc.x, is + 2);
            const float s3 = (float)(int)(int8_t)byte_of(sc.y, is);
            const float s4 = (float)(int)(int8_t)byte_of(sc.y, is + 2);
            const float d = __half2float(__ushort_as_half((unsigned short)dbits));
#pragma unroll
            for (int c = 0; c < NC; c++) {
                if (c >= batch)
                    continue;
                const uint8_t* xrow = xq8 + (size_t)c * q8RowBytes;
                const int q8base = b * 8 + n * 4;
                const uint8_t* b1 = q8_block(xrow, q8base);
                const uint8_t* b2 = q8_block(xrow, q8base + 1);
                const uint8_t* b3 = q8_block(xrow, q8base + 2);
                const uint8_t* b4 = q8_block(xrow, q8base + 3);
                const int* u1 = reinterpret_cast<const int*>(b1 + 4 + l0);
                const int* u2 = reinterpret_cast<const int*>(b2 + 4 + l0);
                const int* u3 = reinterpret_cast<const int*>(b3 + 4 + l0);
                const int* u4 = reinterpret_cast<const int*>(b4 + 4 + l0);
                int dot1 = dp4a2_signed(p1a, p1b, u1);
                int dot2 = dp4a2_signed(p2a, p2b, u2);
                int dot3 = dp4a2_signed(p3a, p3b, u3);
                int dot4 = dp4a2_signed(p4a, p4b, u4);
                acc[c] += d * (s1 * q8_scale(b1) * (float)dot1
                             + s2 * q8_scale(b2) * (float)dot2
                             + s3 * q8_scale(b3) * (float)dot3
                             + s4 * q8_scale(b4) * (float)dot4);
            }
        } else {
            uint4 hdr = *reinterpret_cast<const uint4*>(rp + off);
            uint4 ql, qh5;
            int mask;
            if (TYPE == KQ_Q4K) {
                uint4 q = *reinterpret_cast<const uint4*>(rp + off + 16 + g * 32 + sub);
                ql = q;
                qh5 = make_uint4(q.x >> 4, q.y >> 4, q.z >> 4, q.w >> 4);
                mask = 0x0F0F0F0F;
            } else {
                uint4 qh = *reinterpret_cast<const uint4*>(rp + off + 16 + sub);
                uint4 q = *reinterpret_cast<const uint4*>(rp + off + 48 + g * 32 + sub);
                ql = make_uint4(
                        (q.x & 0x0F0F0F0Fu) | q5_bits(qh.x, bitLo),
                        (q.y & 0x0F0F0F0Fu) | q5_bits(qh.y, bitLo),
                        (q.z & 0x0F0F0F0Fu) | q5_bits(qh.z, bitLo),
                        (q.w & 0x0F0F0F0Fu) | q5_bits(qh.w, bitLo));
                qh5 = make_uint4(
                        ((q.x >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.x, bitHi),
                        ((q.y >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.y, bitHi),
                        ((q.z >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.z, bitHi),
                        ((q.w >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.w, bitHi));
                mask = 0xFFFFFFFF;
            }
            float d, dmin, sc0, mn0, sc1, mn1;
            kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                if (c >= batch)
                    continue;
                const uint8_t* xrow = xq8 + (size_t)c * q8RowBytes;
                const uint8_t* blo = q8_block(xrow, b * 8 + 2 * g);
                const uint8_t* bhi = q8_block(xrow, b * 8 + 2 * g + 1);
                const int* uLo = reinterpret_cast<const int*>(blo + 4 + sub);
                const int* uHi = reinterpret_cast<const int*>(bhi + 4 + sub);
                const int dl = dp4a4(ql, uLo, mask);
                const int dh = dp4a4(qh5, uHi, mask);
                const int sl = sum4_u8(uLo);
                const int sh = sum4_u8(uHi);
                const float d8l = q8_scale(blo);
                const float d8h = q8_scale(bhi);
                acc[c] += d * (sc0 * d8l * (float)dl + sc1 * d8h * (float)dh)
                        - dmin * (mn0 * d8l * (float)sl + mn1 * d8h * (float)sh);
            }
        }
    }
    reduce_and_store_rows<NC>(acc, lane, warp, row, Y, batch, ldc);
}

#define GEMV_ROWS_ENTRY(name, TYPE, NC) \
    extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32) \
    name(const uint8_t* __restrict__ A, const uint8_t* __restrict__ xq8, float* __restrict__ Y, \
         int rows, int cols, int batch, int ldc) { \
        kq_gemv_rows<TYPE, NC>(A, xq8, Y, rows, cols, batch, ldc); \
    }

GEMV_ROWS_ENTRY(q4k_gemv_rows_2, KQ_Q4K, 2)
GEMV_ROWS_ENTRY(q4k_gemv_rows_4, KQ_Q4K, 4)
GEMV_ROWS_ENTRY(q4k_gemv_rows_8, KQ_Q4K, 8)
GEMV_ROWS_ENTRY(q5k_gemv_rows_2, KQ_Q5K, 2)
GEMV_ROWS_ENTRY(q5k_gemv_rows_4, KQ_Q5K, 4)
GEMV_ROWS_ENTRY(q5k_gemv_rows_8, KQ_Q5K, 8)
GEMV_ROWS_ENTRY(q6k_gemv_rows_2, KQ_Q6K, 2)
GEMV_ROWS_ENTRY(q6k_gemv_rows_4, KQ_Q6K, 4)
GEMV_ROWS_ENTRY(q6k_gemv_rows_8, KQ_Q6K, 8)

/* rope.cu's adjacent-pair rotation, row r at position seqLens[r] - 1. */
extern "C" __global__ void __launch_bounds__(BD_THREADS)
rope_rows(
        float* __restrict__ x,              // [rows][nHeads * headDim], rotated in place
        const double* __restrict__ invFreq, // [headDim / 2]
        const int* __restrict__ seqLens,    // [rows]
        int rows,
        int nHeads,
        int headDim) {
    const int half = headDim >> 1;
    const long long pairsPerRow = (long long)nHeads * half;
    const long long total = (long long)rows * pairsPerRow;
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total)
        return;

    const int row = (int)(p / pairsPerRow);
    const int rem = (int)(p - (long long)row * pairsPerRow);
    const int h = rem / half;
    const int i = rem - h * half;

    double s;
    double c;
    sincos((double)(seqLens[row] - 1) * invFreq[i], &s, &c);
    const float sinA = (float)s;
    const float cosA = (float)c;

    float* pair = x + (size_t)row * nHeads * headDim + (size_t)h * headDim + 2 * i;
    const float x0 = pair[0];
    const float x1 = pair[1];
    pair[0] = __fsub_rn(__fmul_rn(x0, cosA), __fmul_rn(x1, sinA));
    pair[1] = __fadd_rn(__fmul_rn(x0, sinA), __fmul_rn(x1, cosA));
}

/* rope.cu's split-half rotation (Qwen2 family), row r at position seqLens[r] - 1. */
extern "C" __global__ void __launch_bounds__(BD_THREADS)
rope_split_half_rows(
        float* __restrict__ x,              // [rows][nHeads * headDim], rotated in place
        const double* __restrict__ invFreq, // [headDim / 2]
        const int* __restrict__ seqLens,    // [rows]
        int rows,
        int nHeads,
        int headDim) {
    const int half = headDim >> 1;
    const long long pairsPerRow = (long long)nHeads * half;
    const long long total = (long long)rows * pairsPerRow;
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total)
        return;

    const int row = (int)(p / pairsPerRow);
    const int rem = (int)(p - (long long)row * pairsPerRow);
    const int h = rem / half;
    const int i = rem - h * half;

    double s;
    double c;
    sincos((double)(seqLens[row] - 1) * invFreq[i], &s, &c);
    const float sinA = (float)s;
    const float cosA = (float)c;

    float* head = x + (size_t)row * nHeads * headDim + (size_t)h * headDim;
    const float x0 = head[i];
    const float x1 = head[i + half];
    head[i] = __fsub_rn(__fmul_rn(x0, cosA), __fmul_rn(x1, sinA));
    head[i + half] = __fadd_rn(__fmul_rn(x0, sinA), __fmul_rn(x1, cosA));
}

/* rope.cu's extended rotation (Phi-3 family), row r at position seqLens[r] - 1. */
extern "C" __global__ void __launch_bounds__(BD_THREADS)
rope_ext_split_half_rows(
        float* __restrict__ x,              // [rows][nHeads * headDim], rotated in place
        const float* __restrict__ factors,  // [headDim / 2], or null
        const int* __restrict__ seqLens,    // [rows]
        int rows,
        int nHeads,
        int headDim,
        float thetaScale,
        float freqScale,
        float mscale) {
    const int half = headDim >> 1;
    const long long pairsPerRow = (long long)nHeads * half;
    const long long total = (long long)rows * pairsPerRow;
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total)
        return;

    const int row = (int)(p / pairsPerRow);
    const int rem = (int)(p - (long long)row * pairsPerRow);
    const int h = rem / half;
    const int i = rem - h * half;

    float theta = (float)(seqLens[row] - 1);
    for (int j = 0; j < i; j++)
        theta = __fmul_rn(theta, thetaScale);
    const float ff = factors != nullptr ? factors[i] : 1.0f;
    const float angle = __fmul_rn(freqScale, __fdiv_rn(theta, ff));

    double s;
    double c;
    sincos((double)angle, &s, &c);
    const float cosA = __fmul_rn((float)c, mscale);
    const float sinA = __fmul_rn((float)s, mscale);

    float* head = x + (size_t)row * nHeads * headDim + (size_t)h * headDim;
    const float x0 = head[i];
    const float x1 = head[i + half];
    head[i] = __fsub_rn(__fmul_rn(x0, cosA), __fmul_rn(x1, sinA));
    head[i + half] = __fadd_rn(__fmul_rn(x0, sinA), __fmul_rn(x1, cosA));
}

/*
 * The batch's K and V rows, cast to FP16 (the rounding of prefill_window.cu's
 * to_half and Float.floatToFloat16) into each request's own mirror at its position
 * seqLens[r] - 1. kPtrs and vPtrs are the attention kernel's tables.
 */
extern "C" __global__ void __launch_bounds__(BD_THREADS)
kv_append_rows(
        const float* __restrict__ k,        // [rows][kvDim]
        const float* __restrict__ v,        // [rows][kvDim]
        half* const* __restrict__ kPtrs,    // [rows]
        half* const* __restrict__ vPtrs,    // [rows]
        const int* __restrict__ seqLens,    // [rows]
        int rows,
        int kvDim) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long perSide = (long long)rows * kvDim;
    if (i >= 2 * perSide)
        return;
    const bool isV = i >= perSide;
    const long long e = isV ? i - perSide : i;
    const int r = (int)(e / kvDim);
    const int c = (int)(e - (long long)r * kvDim);
    half* mirror = (isV ? vPtrs : kPtrs)[r] + (size_t)(seqLens[r] - 1) * kvDim + c;
    *mirror = __float2half_rn((isV ? v : k)[e]);
}
