/*
 * Fused Q4_K matrix-vector kernel for Juno (device-resident packed weights).
 *
 * Block layout matches GGUF / ggml Q4_K and LlamaTransformerHandler.matVecQ4Kraw:
 *   [d:f16(2)][dmin:f16(2)][scales:12][qs:128] = 144 bytes per 256 elements.
 *
 * Compile (Pascal+ reference SKU):
 *   nvcc -ptx -arch=compute_60 -code=sm_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/q4k_gemv.ptx q4k_gemv.cu
 */
#include <stdint.h>

static __device__ __forceinline__ float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t fBits;
    if (e == 0) {
        if (m == 0) {
            fBits = s << 31;
        } else {
            int exp = -14;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                exp--;
            }
            m &= 0x3FFu;
            fBits = (s << 31) | (uint32_t)((exp + 127) << 23) | (m << 13);
        }
    } else if (e == 31) {
        fBits = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        fBits = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    return __int_as_float(fBits);
}

static __device__ __forceinline__ uint16_t load_u16_le(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static __device__ __forceinline__ float q4k_scale(const uint8_t* sc, int j) {
    int v;
    if (j < 4)
        v = sc[j] & 0x3F;
    else
        v = ((sc[j + 4] & 0x0F) | ((sc[j - 4] & 0xC0) >> 2)) & 0x3F;
    return (float)v;
}

static __device__ __forceinline__ float q4k_min(const uint8_t* sc, int j) {
    int v;
    if (j < 4)
        v = sc[j + 4] & 0x3F;
    else
        v = (((sc[j + 4] & 0xFF) >> 4) | ((sc[j] & 0xC0) >> 2)) & 0x3F;
    return (float)v;
}

extern "C" __global__ void q4k_gemv(
        const uint8_t* __restrict__ A,
        const float* __restrict__ x,
        float* __restrict__ y,
        int rows,
        int cols) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
        return;

    const int BLOCK_SIZE = 256;
    const int BLOCK_BYTES = 144;
    int blocksPerRow = cols / BLOCK_SIZE;
    int bytesPerRow = blocksPerRow * BLOCK_BYTES;

    const uint8_t* rowPtr = A + (size_t)row * (size_t)bytesPerRow;
    float acc = 0.f;
    int xBase = 0;

    for (int b = 0; b < blocksPerRow; b++) {
        const uint8_t* blk = rowPtr + b * BLOCK_BYTES;
        const uint8_t* sc = blk + 4;
        const uint8_t* qs = blk + 16;
        float d = f16_to_f32(load_u16_le(blk));
        float dmin = f16_to_f32(load_u16_le(blk + 2));

        int qi = 0;
        for (int g = 0; g < BLOCK_SIZE; g += 64) {
            int s0 = g / 32;
            int s1 = s0 + 1;
            float scale0 = d * q4k_scale(sc, s0);
            float min0 = dmin * q4k_min(sc, s0);
            float scale1 = d * q4k_scale(sc, s1);
            float min1 = dmin * q4k_min(sc, s1);

#pragma unroll
            for (int i = 0; i < 32; i++)
                acc += (scale0 * (float)(qs[qi + i] & 0x0F) - min0) * x[xBase + g + i];
#pragma unroll
            for (int i = 0; i < 32; i++)
                acc += (scale1 * (float)((qs[qi + i] >> 4) & 0x0F) - min1)
                        * x[xBase + g + 32 + i];
            qi += 32;
        }
        xBase += BLOCK_SIZE;
    }
    y[row] = acc;
}
