/*
 * Fused K-quant matrix-vector kernels for Juno (device-resident packed weights).
 *
 * mul_mat_vec class: quantize the activation to Q8_1, then integer-dot packed
 * Q4_K / Q5_K / Q6_K weights with __dp4a (hardware on sm_61+ / GTX 1080).
 *
 * GGUF / ggml super-block layouts (256 elements), matching
 * LlamaTransformerHandler.matVecQ4Kraw / matVecQ5Kraw / matVecQ6Kraw:
 *   Q4_K  [d:f16][dmin:f16][scales:12][qs:128]                     = 144 bytes
 *   Q5_K  [d:f16][dmin:f16][scales:12][qh:32][qs:128]              = 176 bytes
 *   Q6_K  [ql:128][qh:64][sc:16 int8][d:f16]                       = 210 bytes,
 *         uploaded as 224-byte slots (14 zero pad bytes) so every load below
 *         stays 16-byte aligned; see DeviceQ4KMatrix.
 *   Q8_1  [d:f16][sum:f16][qs:32 int8]                             = 36 bytes
 *   Q4_K / Q5_K: qs[32g .. 32g+32) holds the 64 elements of group g (0..3):
 *   low nibbles -> elements 64g + i, high nibbles -> 64g + 32 + i, with
 *   (scale, min) sub-block 2g for the low and 2g+1 for the high half.
 *   Q6_K: two 128-element halves n; ql[64n + l] low/high nibbles + 2 qh bits
 *   give elements 128n + l, +32, +64, +96 with int8 scales sc[8n + l/16 + {0,2,4,6}].
 *
 * Mapping (Pascal decode, ncols=1):
 *   - 128 threads = 4 warps per block; warps cooperate on one output row
 *     (split the K super-blocks). 16 super-blocks in flight per iteration.
 *   - 8 lanes per super-block. Lane (q, l): q = lane >> 3 selects super-block
 *     b = warp*4+q, q+16, ...; l = lane & 7 selects 32 elements of that block.
 *   - Affine types: sum((d*sc*q - dmin*mn) * x) with x ≈ d8 * q8, so
 *     d*sc*d8*dp4a(q,q8) - dmin*mn*d8*sum(q8).
 *   - Warp shuffle + 4-way shared reduction; thread 0 writes y[row].
 *
 * Compile (Pascal+ reference SKU — sm_61 for hardware dp4a):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/q4k_gemv.ptx q4k_gemv.cu
 */
#include <stdint.h>
#include <cuda_fp16.h>

#define KQ_WARPS 4
#define Q4K_BLOCK_BYTES 144
#define Q5K_BLOCK_BYTES 176
#define Q6K_SLOT_BYTES 224
#define Q8_1_BLOCK_BYTES 36

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

/*
 * Decode d, dmin and the (scale, min) pairs for sub-blocks j = 2g (low nibbles)
 * and j + 1 (high nibbles) from the 16-byte Q4_K / Q5_K header. The 12 scale
 * bytes are hdr.y (sc[0..3]), hdr.z (sc[4..7]), hdr.w (sc[8..11]); ggml's
 * get_scale_min_k4 layout: j < 4 -> 6-bit fields in sc[j] / sc[j+4];
 * j >= 4 -> low 4 bits in sc[j+4] nibbles, high 2 bits in sc[j-4] / sc[j] top bits.
 */
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

static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

static __device__ __forceinline__ void reduce_and_store(float acc, int lane, int warp,
        int row, int rows, float* y) {
    acc = warp_sum(acc);
    __shared__ float wsum[KQ_WARPS];
    if (lane == 0)
        wsum[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < KQ_WARPS) ? wsum[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0 && row < rows)
            y[row] = v;
    }
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

// ───────────────────────────── Q8_1 quantize ─────────────────────────────

/* One warp per 32-element block. qs sit at byte 4 so int loads stay aligned. */
extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32)
quantize_q8_1(const float* __restrict__ x, uint8_t* __restrict__ y, int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = (int)blockIdx.x * KQ_WARPS + warp;
    const int nblocks = n >> 5;
    if (b >= nblocks)
        return;
    const float xi = x[(size_t)b * 32 + lane];
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

// ───────────────────────────── Q4_K ─────────────────────────────

extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32)
q4k_gemv(
        const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ y,
        int rows,
        int cols) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    const int nb = cols >> 8;
    const size_t rowBytes = (size_t)nb * Q4K_BLOCK_BYTES;
    const uint8_t* rp = A + (size_t)row * rowBytes;
    const int quarter = lane >> 3;
    const int l = lane & 7;
    const int g = l >> 1;
    const int sub = (l & 1) * 16;

    float acc = 0.f;
    for (int b = warp * 4 + quarter; b < nb; b += 16) {
        const size_t off = (size_t)b * Q4K_BLOCK_BYTES;
        uint4 hdr = *reinterpret_cast<const uint4*>(rp + off);
        uint4 q = *reinterpret_cast<const uint4*>(rp + off + 16 + g * 32 + sub);
        const uint8_t* blo = q8_block(xq8, b * 8 + 2 * g);
        const uint8_t* bhi = q8_block(xq8, b * 8 + 2 * g + 1);
        const int* uLo = reinterpret_cast<const int*>(blo + 4 + sub);
        const int* uHi = reinterpret_cast<const int*>(bhi + 4 + sub);
        const int dl = dp4a4(q, uLo, 0x0F0F0F0F);
        const int dh = dp4a4(make_uint4(q.x >> 4, q.y >> 4, q.z >> 4, q.w >> 4), uHi, 0x0F0F0F0F);
        const int sl = sum4_u8(uLo);
        const int sh = sum4_u8(uHi);
        float d, dmin, sc0, mn0, sc1, mn1;
        kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
        const float d8l = q8_scale(blo);
        const float d8h = q8_scale(bhi);
        acc += d * (sc0 * d8l * (float)dl + sc1 * d8h * (float)dh)
             - dmin * (mn0 * d8l * (float)sl + mn1 * d8h * (float)sh);
    }
    reduce_and_store(acc, lane, warp, row, rows, y);
}

// ───────────────────────────── Q5_K ─────────────────────────────

static __device__ __forceinline__ uint32_t q5_bits(uint32_t qh, int bit) {
    return ((qh >> bit) & 0x01010101u) << 4;
}

extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32)
q5k_gemv(
        const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ y,
        int rows,
        int cols) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    const int nb = cols >> 8;
    const size_t rowBytes = (size_t)nb * Q5K_BLOCK_BYTES;
    const uint8_t* rp = A + (size_t)row * rowBytes;
    const int quarter = lane >> 3;
    const int l = lane & 7;
    const int g = l >> 1;
    const int sub = (l & 1) * 16;
    const int bitLo = 2 * g;
    const int bitHi = 2 * g + 1;

    float acc = 0.f;
    for (int b = warp * 4 + quarter; b < nb; b += 16) {
        const size_t off = (size_t)b * Q5K_BLOCK_BYTES;
        uint4 hdr = *reinterpret_cast<const uint4*>(rp + off);
        uint4 qh = *reinterpret_cast<const uint4*>(rp + off + 16 + sub);
        uint4 q = *reinterpret_cast<const uint4*>(rp + off + 48 + g * 32 + sub);
        const uint8_t* blo = q8_block(xq8, b * 8 + 2 * g);
        const uint8_t* bhi = q8_block(xq8, b * 8 + 2 * g + 1);
        const int* uLo = reinterpret_cast<const int*>(blo + 4 + sub);
        const int* uHi = reinterpret_cast<const int*>(bhi + 4 + sub);
        uint4 ql = make_uint4(
                (q.x & 0x0F0F0F0Fu) | q5_bits(qh.x, bitLo),
                (q.y & 0x0F0F0F0Fu) | q5_bits(qh.y, bitLo),
                (q.z & 0x0F0F0F0Fu) | q5_bits(qh.z, bitLo),
                (q.w & 0x0F0F0F0Fu) | q5_bits(qh.w, bitLo));
        uint4 qh5 = make_uint4(
                ((q.x >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.x, bitHi),
                ((q.y >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.y, bitHi),
                ((q.z >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.z, bitHi),
                ((q.w >> 4) & 0x0F0F0F0Fu) | q5_bits(qh.w, bitHi));
        const int dl = dp4a4(ql, uLo, 0xFFFFFFFF);
        const int dh = dp4a4(qh5, uHi, 0xFFFFFFFF);
        const int sl = sum4_u8(uLo);
        const int sh = sum4_u8(uHi);
        float d, dmin, sc0, mn0, sc1, mn1;
        kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
        const float d8l = q8_scale(blo);
        const float d8h = q8_scale(bhi);
        acc += d * (sc0 * d8l * (float)dl + sc1 * d8h * (float)dh)
             - dmin * (mn0 * d8l * (float)sl + mn1 * d8h * (float)sh);
    }
    reduce_and_store(acc, lane, warp, row, rows, y);
}

// ───────────────────────────── Q6_K ─────────────────────────────

static __device__ __forceinline__ uint32_t q6_assemble(uint32_t nib, uint32_t qh, int shift) {
    return nib | (((qh >> shift) & 0x03030303u) << 4);
}

/* Stored Q6 is 0..63; signed element is value-32. Per-byte subtract — a 32-bit
 * add of 0xE0E0E0E0 carries when a byte is 63 (63+224=287). */
static __device__ __forceinline__ int q6_signed(uint32_t v) {
    return __vsubss4((int)v, 0x20202020);
}

static __device__ __forceinline__ int dp4a2_signed(uint32_t a0, uint32_t a1, const int* u) {
    return dp4a(q6_signed(a0), u[0], dp4a(q6_signed(a1), u[1], 0));
}

extern "C" __global__ void __launch_bounds__(KQ_WARPS * 32)
q6k_gemv(
        const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ xq8,
        float* __restrict__ y,
        int rows,
        int cols) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    const int nb = cols >> 8;
    const size_t rowBytes = (size_t)nb * Q6K_SLOT_BYTES;
    const uint8_t* rp = A + (size_t)row * rowBytes;
    const int quarter = lane >> 3;
    const int l = lane & 7;
    const int n = l >> 2;
    const int l0 = (l & 3) * 8;
    const int is = l0 >> 4;

    float acc = 0.f;
    for (int b = warp * 4 + quarter; b < nb; b += 16) {
        const uint8_t* blk = rp + (size_t)b * Q6K_SLOT_BYTES;
        uint2 qla = *reinterpret_cast<const uint2*>(blk + n * 64 + l0);
        uint2 qlb = *reinterpret_cast<const uint2*>(blk + n * 64 + 32 + l0);
        uint2 qh = *reinterpret_cast<const uint2*>(blk + 128 + n * 32 + l0);
        uint2 sc = *reinterpret_cast<const uint2*>(blk + 192 + n * 8);
        uint32_t dbits = *reinterpret_cast<const uint16_t*>(blk + 208);

        const int q8base = b * 8 + n * 4;
        const uint8_t* b1 = q8_block(xq8, q8base);
        const uint8_t* b2 = q8_block(xq8, q8base + 1);
        const uint8_t* b3 = q8_block(xq8, q8base + 2);
        const uint8_t* b4 = q8_block(xq8, q8base + 3);
        const int* u1 = reinterpret_cast<const int*>(b1 + 4 + l0);
        const int* u2 = reinterpret_cast<const int*>(b2 + 4 + l0);
        const int* u3 = reinterpret_cast<const int*>(b3 + 4 + l0);
        const int* u4 = reinterpret_cast<const int*>(b4 + 4 + l0);

        int dot1 = dp4a2_signed(q6_assemble(qla.x & 0x0F0F0F0Fu, qh.x, 0),
                                q6_assemble(qla.y & 0x0F0F0F0Fu, qh.y, 0), u1);
        int dot2 = dp4a2_signed(q6_assemble(qlb.x & 0x0F0F0F0Fu, qh.x, 2),
                                q6_assemble(qlb.y & 0x0F0F0F0Fu, qh.y, 2), u2);
        int dot3 = dp4a2_signed(q6_assemble((qla.x >> 4) & 0x0F0F0F0Fu, qh.x, 4),
                                q6_assemble((qla.y >> 4) & 0x0F0F0F0Fu, qh.y, 4), u3);
        int dot4 = dp4a2_signed(q6_assemble((qlb.x >> 4) & 0x0F0F0F0Fu, qh.x, 6),
                                q6_assemble((qlb.y >> 4) & 0x0F0F0F0Fu, qh.y, 6), u4);

        float s1 = (float)(int)(int8_t)byte_of(sc.x, is);
        float s2 = (float)(int)(int8_t)byte_of(sc.x, is + 2);
        float s3 = (float)(int)(int8_t)byte_of(sc.y, is);
        float s4 = (float)(int)(int8_t)byte_of(sc.y, is + 2);
        float d = __half2float(__ushort_as_half((unsigned short)dbits));
        acc += d * (s1 * q8_scale(b1) * (float)dot1
                  + s2 * q8_scale(b2) * (float)dot2
                  + s3 * q8_scale(b3) * (float)dot3
                  + s4 * q8_scale(b4) * (float)dot4);
    }
    reduce_and_store(acc, lane, warp, row, rows, y);
}

/*
 * ───────────────────── Dequant-to-FP16 (batched-prefill GEMM path) ─────────────────────
 *
 * Elementwise dequant of packed K-quant weights into a row-major FP16 buffer, no
 * activation/reduction involved (dequant is independent of x, done once per
 * layer-projection rather than once per token). Feeds the tiled cublasGemmEx
 * path (see CudaFp16GemmOps) for prefill-sized batches, replacing W serial
 * mul_mat_vec launches with one dequant pass + one weight-stationary GEMM.
 *
 * One thread per output element; one block per (row, super-block). Reuses
 * kq_affine_scales (Q4_K/Q5_K) and the Q6_K scale/qh/ql bit-unpacking already
 * proven correct by the mul_mat_vec kernels above — this is the same decode
 * math, just writing every element instead of dot-reducing against x.
 */

extern "C" __global__ void
q4k_dequant_to_fp16(
        const uint8_t* __restrict__ A,
        half* __restrict__ out,
        int rows,
        int cols) {
    const int row = (int)blockIdx.x;
    const int b = (int)blockIdx.y;
    const int nb = cols >> 8;
    if (row >= rows || b >= nb)
        return;

    const int e = (int)threadIdx.x; // 0..255
    const size_t rowBytes = (size_t)nb * Q4K_BLOCK_BYTES;
    const uint8_t* rp = A + (size_t)row * rowBytes + (size_t)b * Q4K_BLOCK_BYTES;
    uint4 hdr = *reinterpret_cast<const uint4*>(rp);

    const int g = e >> 6;        // 0..3 (group of 64 elements)
    const int r = e & 63;        // 0..63 within group
    const bool hi = r >= 32;
    const int i = hi ? (r - 32) : r; // 0..31
    const uint8_t qbyte = rp[16 + g * 32 + i];
    const int q = hi ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);

    float d, dmin, sc0, mn0, sc1, mn1;
    kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
    float val = hi ? (d * sc1 * (float)q - dmin * mn1) : (d * sc0 * (float)q - dmin * mn0);

    out[(size_t)row * cols + (size_t)b * 256 + e] = __float2half(val);
}

extern "C" __global__ void
q5k_dequant_to_fp16(
        const uint8_t* __restrict__ A,
        half* __restrict__ out,
        int rows,
        int cols) {
    const int row = (int)blockIdx.x;
    const int b = (int)blockIdx.y;
    const int nb = cols >> 8;
    if (row >= rows || b >= nb)
        return;

    const int e = (int)threadIdx.x; // 0..255
    const size_t rowBytes = (size_t)nb * Q5K_BLOCK_BYTES;
    const uint8_t* rp = A + (size_t)row * rowBytes + (size_t)b * Q5K_BLOCK_BYTES;
    uint4 hdr = *reinterpret_cast<const uint4*>(rp);

    const int g = e >> 6;        // 0..3
    const int r = e & 63;        // 0..63
    const bool hi = r >= 32;
    const int i = hi ? (r - 32) : r; // 0..31
    const uint8_t qbyte = rp[48 + g * 32 + i];
    const int nib = hi ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
    const int bit = hi ? (2 * g + 1) : (2 * g);
    const uint8_t hbyte = rp[16 + i];
    const int q = nib | (((hbyte >> bit) & 1) << 4);

    float d, dmin, sc0, mn0, sc1, mn1;
    kq_affine_scales(hdr, g, d, dmin, sc0, mn0, sc1, mn1);
    float val = hi ? (d * sc1 * (float)q - dmin * mn1) : (d * sc0 * (float)q - dmin * mn0);

    out[(size_t)row * cols + (size_t)b * 256 + e] = __float2half(val);
}

extern "C" __global__ void
q6k_dequant_to_fp16(
        const uint8_t* __restrict__ A,
        half* __restrict__ out,
        int rows,
        int cols) {
    const int row = (int)blockIdx.x;
    const int b = (int)blockIdx.y;
    const int nb = cols >> 8;
    if (row >= rows || b >= nb)
        return;

    const int e = (int)threadIdx.x; // 0..255
    const size_t rowBytes = (size_t)nb * Q6K_SLOT_BYTES;
    const uint8_t* blk = A + (size_t)row * rowBytes + (size_t)b * Q6K_SLOT_BYTES;
    const float d = __half2float(*reinterpret_cast<const half*>(blk + 208));

    const int half_ = e >> 7;    // 0 or 1 (of the two 128-element halves)
    const int k = e & 127;       // 0..127 within half
    const int group = k >> 5;    // 0..3 (q1/q2/q3/q4)
    const int l = k & 31;        // 0..31
    const int is = l >> 4;       // 0 or 1

    const int qlOff = half_ * 64;
    const int qhOff = 128 + half_ * 32;
    const int scOff = 192 + half_ * 8;
    const uint8_t qhByte = blk[qhOff + l];

    int nib, hibits, scIdx;
    switch (group) {
        case 0: nib = blk[qlOff + l] & 0x0F;              hibits = (qhByte >> 0) & 3; scIdx = scOff + is + 0; break;
        case 1: nib = blk[qlOff + l + 32] & 0x0F;         hibits = (qhByte >> 2) & 3; scIdx = scOff + is + 2; break;
        case 2: nib = (blk[qlOff + l] >> 4) & 0x0F;       hibits = (qhByte >> 4) & 3; scIdx = scOff + is + 4; break;
        default: nib = (blk[qlOff + l + 32] >> 4) & 0x0F; hibits = (qhByte >> 6) & 3; scIdx = scOff + is + 6; break;
    }
    int q = (nib | (hibits << 4)) - 32;
    float sc = (float)(int8_t)blk[scIdx];
    float val = d * sc * (float)q;

    out[(size_t)row * cols + (size_t)b * 256 + e] = __float2half(val);
}
