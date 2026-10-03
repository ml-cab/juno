/*
 * Tiled K-quant matrix multiply for Juno (device-resident packed weights,
 * prefill-width batches).
 *
 * Y[c][r] = sum_k A[r][k] * x[c][k] for a batch of activation rows x already
 * packed as Q8_1 (row c's blocks at c * cols/32), by the GEMV module's
 * quantize_q8_1 from FP32 or by quantize_q8_1_half below from FP16. The packed Q4_K / Q5_K / Q6_K weights are never expanded to
 * floating point in device memory: each block unpacks one 256-element
 * super-block of its weight rows into int8 in shared memory, integer-dots it
 * against the Q8_1 activations with __dp4a, and folds the per-sub-block scales
 * into FP32 accumulators. Super-block layouts are those of q4k_gemv.cu
 * (Q6_K in 224-byte slots, see DeviceQ4KMatrix).
 *
 * Mapping (KG_THREADS = 256 = 8 warps per block):
 *   - a block owns KG_TILE_ROWS = 64 weight rows and NC * 8 activation rows
 *     (the column tile: 16, 32 or 64). blockIdx.x walks column tiles, so the
 *     blocks resident at one time share a weight tile through L2.
 *   - lane l of every warp owns weight rows l and l + 32; warp w owns
 *     activation rows w*NC .. w*NC + NC-1. Every lane of a warp reads the same
 *     activation ints from shared memory (a broadcast) and its own weight row
 *     (row stride 68 ints, so a warp's 128-bit loads do not conflict).
 *   - per super-block: the 256 threads unpack 64 weight rows (4 threads per
 *     row, 64 elements each) and the column tile's 8 Q8_1 blocks per row, then
 *     every thread runs 8 sub-blocks x 2 rows x NC columns of 8-int dp4a dots.
 *   - Q4_K / Q5_K sub-blocks are 32 elements with a scale and a min:
 *       d*sc*d8*dot(q, q8) - dmin*mn*d8*sum(q8)
 *     Q6_K sub-blocks are 16 elements with a signed scale and no min, so each
 *     32-element Q8_1 block takes two half dots.
 *
 * Compile (Pascal+ reference SKU - sm_61 for hardware dp4a):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/kquant_gemm.ptx kquant_gemm.cu
 */
#include <stdint.h>
#include <cuda_fp16.h>

#define KG_THREADS 256
#define KG_WARPS 8
#define KG_TILE_ROWS 64
#define KG_W_STRIDE 68 /* ints per unpacked weight row: 64 + 4 pad */
#define KG_S_STRIDE 9  /* float2 scale pairs per weight row: 8 + 1 pad */
#define Q4K_BLOCK_BYTES 144
#define Q5K_BLOCK_BYTES 176
#define Q6K_SLOT_BYTES 224
#define Q8_1_BLOCK_BYTES 36

#define KG_Q4K 0
#define KG_Q5K 1
#define KG_Q6K 2

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

static __device__ __forceinline__ uint32_t q5_bits(uint32_t qh, int bit) {
    return ((qh >> bit) & 0x01010101u) << 4;
}

static __device__ __forceinline__ uint32_t q6_bits(uint32_t qh, int shift) {
    return ((qh >> shift) & 0x03030303u) << 4;
}

/* Stored Q6 is 0..63; the signed element is value - 32, per byte. */
static __device__ __forceinline__ int q6_signed(uint32_t v) {
    return __vsubss4((int)v, 0x20202020);
}

static __device__ __forceinline__ int dot4(int4 a, int4 b, int acc) {
    return dp4a(a.x, b.x, dp4a(a.y, b.y, dp4a(a.z, b.z, dp4a(a.w, b.w, acc))));
}

static __device__ __forceinline__ void store8(int* dst, const uint32_t* v) {
    reinterpret_cast<int4*>(dst)[0] = make_int4((int)v[0], (int)v[1], (int)v[2], (int)v[3]);
    reinterpret_cast<int4*>(dst)[1] = make_int4((int)v[4], (int)v[5], (int)v[6], (int)v[7]);
}

static __device__ __forceinline__ void load8(const uint8_t* src, uint32_t* v) {
    const uint4 a = reinterpret_cast<const uint4*>(src)[0];
    const uint4 b = reinterpret_cast<const uint4*>(src)[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

/*
 * Unpacks this thread's quarter of weight row r (super-block sb) into wq[r]
 * and its scale pairs into ws[r]. Thread part t = 0..3.
 *   Q4_K / Q5_K: group g = t, elements 64g .. 64g+63 = ints 16g .. 16g+15;
 *                ws[r][j] = (d*sc_j, dmin*mn_j) for sub-blocks j = 2g, 2g+1.
 *   Q6_K:        half n = t>>1, part p = t&1, elements 128n+32p+[0,32) and
 *                128n+64+32p+[0,32); ws[r][j] = (d*sc_2j, d*sc_2j+1) for
 *                j = 2t, 2t+1 (16-element scales 4t .. 4t+3).
 */
template <int TYPE>
static __device__ __forceinline__ void unpack_weights(const uint8_t* __restrict__ blk, int t,
        int* wq, float2* ws) {
    uint32_t lo[8], hi[8];
    if (TYPE == KG_Q6K) {
        const int n = t >> 1, p = t & 1;
        uint32_t ql[8], qh[8];
        load8(blk + 64 * n + 32 * p, ql);
        load8(blk + 128 + 32 * n, qh);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            lo[i] = (uint32_t)q6_signed((ql[i] & 0x0F0F0F0Fu) | q6_bits(qh[i], 2 * p));
            hi[i] = (uint32_t)q6_signed(((ql[i] >> 4) & 0x0F0F0F0Fu) | q6_bits(qh[i], 4 + 2 * p));
        }
        store8(wq + 32 * n + 8 * p, lo);
        store8(wq + 32 * n + 16 + 8 * p, hi);
        const float d = __half2float(*reinterpret_cast<const half*>(blk + 208));
        const uint32_t sc = *reinterpret_cast<const uint32_t*>(blk + 192 + 4 * t);
        ws[2 * t]     = make_float2(d * (float)(int8_t)byte_of(sc, 0), d * (float)(int8_t)byte_of(sc, 1));
        ws[2 * t + 1] = make_float2(d * (float)(int8_t)byte_of(sc, 2), d * (float)(int8_t)byte_of(sc, 3));
    } else {
        const int g = t;
        const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
        uint32_t q[8];
        if (TYPE == KG_Q4K) {
            load8(blk + 16 + 32 * g, q);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                lo[i] = q[i] & 0x0F0F0F0Fu;
                hi[i] = (q[i] >> 4) & 0x0F0F0F0Fu;
            }
        } else {
            uint32_t qh[8];
            load8(blk + 16, qh);
            load8(blk + 48 + 32 * g, q);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                lo[i] = (q[i] & 0x0F0F0F0Fu) | q5_bits(qh[i], 2 * g);
                hi[i] = ((q[i] >> 4) & 0x0F0F0F0Fu) | q5_bits(qh[i], 2 * g + 1);
            }
        }
        store8(wq + 16 * g, lo);
        store8(wq + 16 * g + 8, hi);
        float d, dmin, sc0, mn0, sc1, mn1;
        kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
        ws[2 * g]     = make_float2(d * sc0, dmin * mn0);
        ws[2 * g + 1] = make_float2(d * sc1, dmin * mn1);
    }
}

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
 * Q8_1 packing of FP16 activations: q4k_gemv.cu's quantize_q8_1 with a half
 * input, the same operations in the same order, so the same values give the
 * same bytes. The prefill paths hold their GEMM input as FP16 (the window the
 * host stages, the region's cast and SwiGLU output); quantizing it directly
 * avoids widening it back to FP32 first. One warp per 32-element block.
 */
extern "C" __global__ void __launch_bounds__(128)
quantize_q8_1_half(const half* __restrict__ x, uint8_t* __restrict__ y, int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = (int)blockIdx.x * 4 + warp;
    const int nblocks = n >> 5;
    if (b >= nblocks)
        return;
    const float xi = __half2float(x[(size_t)b * 32 + lane]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = amax / 127.f;
    int q = (amax == 0.f) ? 0 : (int)rintf(xi / d);
    q = max(-127, min(127, q));
    uint8_t* blk = y + (size_t)b * Q8_1_BLOCK_BYTES;
    blk[4 + lane] = (uint8_t)(int8_t)q;
    if (lane == 0)
        *reinterpret_cast<half2*>(blk) = __floats2half2_rn(d, sum);
}

template <int TYPE, int NC>
static __device__ __forceinline__ void kq_gemm(
        const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ Y,
        int rows, int cols, int batch, int ldc) {
    constexpr int TN = NC * KG_WARPS;
    constexpr int BLOCK_BYTES = TYPE == KG_Q4K ? Q4K_BLOCK_BYTES
                              : TYPE == KG_Q5K ? Q5K_BLOCK_BYTES : Q6K_SLOT_BYTES;

    __shared__ __align__(16) int wq[KG_TILE_ROWS * KG_W_STRIDE];
    __shared__ __align__(16) float2 ws[KG_TILE_ROWS * KG_S_STRIDE];
    __shared__ __align__(16) int xq[TN * 64];
    __shared__ __align__(16) float2 xds[TN * 8];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int colBase = (int)blockIdx.x * TN;
    const int rowBase = (int)blockIdx.y * KG_TILE_ROWS;
    const int nb = cols >> 8;
    const size_t rowBytes = (size_t)nb * BLOCK_BYTES;
    const int q8PerRow = cols >> 5;

    /* Weight-unpack assignment: 4 threads per row, consecutive threads read
     * consecutive 32-byte runs of one row. */
    const int ur = tid >> 2;
    const int ut = tid & 3;
    const int urow = rowBase + ur;
    const uint8_t* wrow = A + (size_t)min(urow, rows - 1) * rowBytes;

    float acc[2][NC];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int k = 0; k < NC; k++)
            acc[i][k] = 0.f;

    for (int sb = 0; sb < nb; sb++) {
        /* ── load: weights ── */
        int* wdst = wq + ur * KG_W_STRIDE;
        float2* sdst = ws + ur * KG_S_STRIDE;
        if (urow < rows) {
            unpack_weights<TYPE>(wrow + (size_t)sb * BLOCK_BYTES, ut, wdst, sdst);
        } else {
            const uint32_t z[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            store8(wdst + 16 * ut, z);
            store8(wdst + 16 * ut + 8, z);
            sdst[2 * ut] = make_float2(0.f, 0.f);
            sdst[2 * ut + 1] = make_float2(0.f, 0.f);
        }

        /* ── load: activations, TN rows x 8 Q8_1 blocks ── */
#pragma unroll
        for (int item = tid; item < TN * 8; item += KG_THREADS) {
            const int c = item >> 3;
            const int j = item & 7;
            const int col = colBase + c;
            int* xdst = xq + c * 64 + 8 * j;
            if (col < batch) {
                const uint8_t* blk = xq8 + ((size_t)col * q8PerRow + (size_t)sb * 8 + j) * Q8_1_BLOCK_BYTES;
                const int* qs = reinterpret_cast<const int*>(blk + 4);
                int v[8];
                int isum = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    v[i] = qs[i];
                    isum = dp4a(0x01010101, v[i], isum);
                }
                reinterpret_cast<int4*>(xdst)[0] = make_int4(v[0], v[1], v[2], v[3]);
                reinterpret_cast<int4*>(xdst)[1] = make_int4(v[4], v[5], v[6], v[7]);
                const float d8 = __half2float(*reinterpret_cast<const half*>(blk));
                xds[c * 8 + j] = make_float2(d8, d8 * (float)isum);
            } else {
                reinterpret_cast<int4*>(xdst)[0] = make_int4(0, 0, 0, 0);
                reinterpret_cast<int4*>(xdst)[1] = make_int4(0, 0, 0, 0);
                xds[c * 8 + j] = make_float2(0.f, 0.f);
            }
        }
        __syncthreads();

        /* ── compute: 8 sub-blocks x 2 rows x NC columns ── */
        const int* w0 = wq + lane * KG_W_STRIDE;
        const int* w1 = wq + (lane + 32) * KG_W_STRIDE;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int4 a0 = reinterpret_cast<const int4*>(w0 + 8 * j)[0];
            const int4 a1 = reinterpret_cast<const int4*>(w0 + 8 * j)[1];
            const int4 b0 = reinterpret_cast<const int4*>(w1 + 8 * j)[0];
            const int4 b1 = reinterpret_cast<const int4*>(w1 + 8 * j)[1];
            const float2 sa = ws[lane * KG_S_STRIDE + j];
            const float2 sb2 = ws[(lane + 32) * KG_S_STRIDE + j];
#pragma unroll
            for (int k = 0; k < NC; k++) {
                const int c = warp * NC + k;
                const int4 x0 = reinterpret_cast<const int4*>(xq + c * 64 + 8 * j)[0];
                const int4 x1 = reinterpret_cast<const int4*>(xq + c * 64 + 8 * j)[1];
                const float2 xd = xds[c * 8 + j];
                if (TYPE == KG_Q6K) {
                    const float da = sa.x * (float)dot4(a0, x0, 0) + sa.y * (float)dot4(a1, x1, 0);
                    const float db = sb2.x * (float)dot4(b0, x0, 0) + sb2.y * (float)dot4(b1, x1, 0);
                    acc[0][k] += xd.x * da;
                    acc[1][k] += xd.x * db;
                } else {
                    const int da = dot4(a0, x0, dot4(a1, x1, 0));
                    const int db = dot4(b0, x0, dot4(b1, x1, 0));
                    acc[0][k] += sa.x * xd.x * (float)da - sa.y * xd.y;
                    acc[1][k] += sb2.x * xd.x * (float)db - sb2.y * xd.y;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int row = rowBase + lane + 32 * i;
        if (row >= rows)
            continue;
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int col = colBase + warp * NC + k;
            if (col < batch)
                Y[(size_t)col * ldc + row] = acc[i][k];
        }
    }
}

#define KG_ENTRY(name, TYPE, NC) \
    extern "C" __global__ void __launch_bounds__(KG_THREADS) \
    name(const uint8_t* __restrict__ A, const uint8_t* __restrict__ xq8, float* __restrict__ Y, \
         int rows, int cols, int batch, int ldc) { \
        kq_gemm<TYPE, NC>(A, xq8, Y, rows, cols, batch, ldc); \
    }

KG_ENTRY(q4k_gemm_16, KG_Q4K, 2)
KG_ENTRY(q4k_gemm_32, KG_Q4K, 4)
KG_ENTRY(q4k_gemm_64, KG_Q4K, 8)
KG_ENTRY(q5k_gemm_16, KG_Q5K, 2)
KG_ENTRY(q5k_gemm_32, KG_Q5K, 4)
KG_ENTRY(q5k_gemm_64, KG_Q5K, 8)
KG_ENTRY(q6k_gemm_16, KG_Q6K, 2)
KG_ENTRY(q6k_gemm_32, KG_Q6K, 4)
KG_ENTRY(q6k_gemm_64, KG_Q6K, 8)
