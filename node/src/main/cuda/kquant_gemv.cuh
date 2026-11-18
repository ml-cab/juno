/*
 * Shared helpers of the fused K-quant GEMV kernels (q4k_gemv.cu, q4k_gemv_pair.cu): the dp4a integer dot, the
 * Q4_K / Q5_K affine scale decode, warp reductions and the Q8_1 block accessors. Layouts are described in q4k_gemv.cu.
 */
#pragma once
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
