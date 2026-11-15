/*
 * Device KV row formats for Juno: where a key or value element of a request's
 * device KV (DeviceKvCache) lives, and how it is read and written. Every kernel
 * that reads or writes a quantized device KV row goes through these functions,
 * so the address of row `pos` and of element `e` within it is computed here and
 * nowhere else: a request's base pointer, then the row, then the element.
 *
 *   KV_F16  (0)  IEEE FP16, 2 bytes per value.
 *   KV_Q8_0 (1)  blocks of 32 values, 34 bytes each: the FP16 scale d, then 32
 *                int8 q; value = d * q. Q8_0KvCodec's block, byte for byte.
 *   KV_Q4_0 (2)  blocks of 32 values, 18 bytes each: the FP16 scale d, then 16
 *                bytes, element j in the low nibble of byte j and element j + 16
 *                in its high nibble; value = d * (nibble - 8). Q4_0KvCodec's
 *                block, byte for byte.
 *
 * A row is kvDim values; the quantized formats need kvDim to be a multiple of 32
 * (DeviceKvFormat refuses them otherwise). Rows are contiguous, so every block
 * starts at an even byte offset and every two-byte read below is aligned.
 *
 * Reads are templates on the format (kv_row_t, kv_load2_t, kv_load4_t), so a kernel
 * dispatches the format once per launch and its loops carry no format branch.
 * They return the values the host codecs decode, bit for bit: d * q is exact in
 * FP32 (an 11-bit scale times an integer of at most 8 bits), and is written with
 * __fmul_rn so that no contraction can fold it into a neighbouring add.
 *
 * Writes (kv_encode_block_warp) are the host encoders' arithmetic, one warp per
 * block and one value per lane, with every rounding spelled out (__fdiv_rn,
 * __fmul_rn, __fadd_rn: no FMA contraction, IEEE division), so a block the device
 * writes equals the block Q8_0KvCodec or Q4_0KvCodec writes for the same 32 floats.
 */
#ifndef JUNO_KV_FORMAT_CUH
#define JUNO_KV_FORMAT_CUH

#include <cuda_fp16.h>

#define KV_F16 0
#define KV_Q8_0 1
#define KV_Q4_0 2
#define KV_BLOCK 32

/* Bytes of one block of a quantized format. */
template <int T>
static __device__ __forceinline__ constexpr int kv_block_bytes_t() {
    return T == KV_Q8_0 ? 34 : 18;
}

/* Bytes of one row of kvDim values. */
template <int T>
static __device__ __forceinline__ size_t kv_row_bytes_t(int kvDim) {
    if constexpr (T == KV_F16)
        return (size_t)kvDim * 2;
    else
        return (size_t)(kvDim / KV_BLOCK) * kv_block_bytes_t<T>();
}

static __device__ __forceinline__ size_t kv_row_bytes(int type, int kvDim) {
    return type == KV_F16 ? kv_row_bytes_t<KV_F16>(kvDim)
            : type == KV_Q8_0 ? kv_row_bytes_t<KV_Q8_0>(kvDim) : kv_row_bytes_t<KV_Q4_0>(kvDim);
}

/* Row `pos` of the cache at `base`: the one place a row's address is computed. */
template <int T>
static __device__ __forceinline__ const unsigned char* kv_row_t(const void* base, int kvDim, int pos) {
    return reinterpret_cast<const unsigned char*>(base) + (size_t)pos * kv_row_bytes_t<T>(kvDim);
}

static __device__ __forceinline__ unsigned char* kv_row_w(void* base, int type, int kvDim, int pos) {
    return reinterpret_cast<unsigned char*>(base) + (size_t)pos * kv_row_bytes(type, kvDim);
}

/*
 * Every read of a device KV row goes through the read-only data cache (__ldg): the
 * rows do not change while an attention kernel runs, and a decode lane reads whole
 * blocks, so the four lanes of a slot touch neighbouring sectors that the cache
 * then serves instead of L2.
 */
static __device__ __forceinline__ float kv_scale(const unsigned char* blk) {
    return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(blk))));
}

static __device__ __forceinline__ unsigned int kv_u16(const unsigned char* p) {
    return __ldg(reinterpret_cast<const unsigned short*>(p));
}

static __device__ __forceinline__ float kv_q8(float d, unsigned int byte) {
    return __fmul_rn(d, (float)(int)(signed char)(byte & 0xffu));
}

static __device__ __forceinline__ float kv_q4(float d, unsigned int nibble) {
    return __fmul_rn(d, (float)((int)(nibble & 15u) - 8));
}

/* Elements e and e + 1 of a row of format T (e even); the element's address within the row is computed here. */
template <int T>
static __device__ __forceinline__ float2 kv_load2_t(const unsigned char* row, int e) {
    if constexpr (T == KV_F16) {
        return __half22float2(__ldg(reinterpret_cast<const half2*>(row + 2 * (size_t)e)));
    } else {
        const unsigned char* blk = row + (size_t)(e / KV_BLOCK) * kv_block_bytes_t<T>();
        const float d = kv_scale(blk);
        const int j = e & (KV_BLOCK - 1);
        if constexpr (T == KV_Q8_0) {
            const unsigned int w = kv_u16(blk + 2 + j);
            return make_float2(kv_q8(d, w), kv_q8(d, w >> 8));
        } else {
            const unsigned int w = kv_u16(blk + 2 + (j & 15)) >> (j < 16 ? 0 : 4);
            return make_float2(kv_q4(d, w), kv_q4(d, w >> 8));
        }
    }
}

/* Elements e .. e + 3 of a row of format T (e a multiple of 4): all four lie in one block. */
template <int T>
static __device__ __forceinline__ float4 kv_load4_t(const unsigned char* row, int e) {
    if constexpr (T == KV_F16) {
        const half2* p = reinterpret_cast<const half2*>(row + 2 * (size_t)e);
        const float2 a = __half22float2(__ldg(p));
        const float2 b = __half22float2(__ldg(p + 1));
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const unsigned char* blk = row + (size_t)(e / KV_BLOCK) * kv_block_bytes_t<T>();
        const float d = kv_scale(blk);
        const int j = e & (KV_BLOCK - 1);
        if constexpr (T == KV_Q8_0) {
            const unsigned int w0 = kv_u16(blk + 2 + j);
            const unsigned int w1 = kv_u16(blk + 4 + j);
            return make_float4(kv_q8(d, w0), kv_q8(d, w0 >> 8), kv_q8(d, w1), kv_q8(d, w1 >> 8));
        } else {
            const int sh = j < 16 ? 0 : 4;
            const unsigned int w0 = kv_u16(blk + 2 + (j & 15)) >> sh;
            const unsigned int w1 = kv_u16(blk + 4 + (j & 15)) >> sh;
            return make_float4(kv_q4(d, w0), kv_q4(d, w0 >> 8), kv_q4(d, w1), kv_q4(d, w1 >> 8));
        }
    }
}

/*
 * The FP16 scale of a block, as one aligned 16-bit store (every block starts at an
 * even byte). Two byte stores do not work: with nvcc 12.0 for sm_61, storing the
 * low byte straight from the converted FP16 value wrote that value's integer part
 * (1.0 gave byte 1), not its low 8 bits; DeviceKvQuantParityTest holds the bytes.
 */
static __device__ __forceinline__ void kv_store_scale(unsigned char* blk, float d) {
    *reinterpret_cast<unsigned short*>(blk) = __half_as_ushort(__float2half_rn(d));
}

/*
 * The 4-byte words of `quads` quads starting at a, which is two-byte aligned (every
 * block starts at an even byte): aligned 4-byte loads realigned with a byte permute,
 * reading no byte past the last one asked for.
 */
template <int QUADS>
static __device__ __forceinline__ void kv_words(const unsigned char* a, unsigned int* out) {
    const unsigned int* wp = reinterpret_cast<const unsigned int*>(reinterpret_cast<size_t>(a) & ~(size_t)3);
    const unsigned int odd = (unsigned int)(reinterpret_cast<size_t>(a) & 2);
    unsigned int w[QUADS + 1];
#pragma unroll
    for (int k = 0; k < QUADS; k++)
        w[k] = __ldg(wp + k);
    w[QUADS] = odd ? (unsigned int)__ldg(reinterpret_cast<const unsigned short*>(wp + QUADS)) : 0u;
    const unsigned int sel = odd ? 0x5432u : 0x3210u;
#pragma unroll
    for (int k = 0; k < QUADS; k++)
        out[k] = __byte_perm(w[k], w[k + 1], sel);
}

/*
 * The four values of one quad of a block from its quant words: word k as int8 (Q8_0),
 * or the low (hi false) or high nibbles of word k, less 8 (Q4_0: quads 0-3 are the
 * low nibbles of words 0-3, quads 4-7 their high nibbles). k must be a compile-time
 * index, so the words stay in registers.
 */
template <int T>
static __device__ __forceinline__ float4 kv_quad_ints(const unsigned int* words, int k, bool hi) {
    if constexpr (T == KV_Q8_0) {
        const unsigned int w = words[k];
        return make_float4((float)(int)(signed char)(w & 0xffu), (float)(int)(signed char)((w >> 8) & 0xffu),
                (float)(int)(signed char)((w >> 16) & 0xffu), (float)(int)(signed char)(w >> 24));
    } else {
        const unsigned int w = (hi ? words[k] >> 4 : words[k]) & 0x0f0f0f0fu;
        return make_float4((float)((int)(w & 0xffu) - 8), (float)((int)((w >> 8) & 0xffu) - 8),
                (float)((int)((w >> 16) & 0xffu) - 8), (float)((int)(w >> 24) - 8));
    }
}

/*
 * A decode lane's chunk: QUADS consecutive quads of one head, starting at element e0
 * of a row of format T, of which the first nValid exist (the head may end inside the
 * chunk). A quantized chunk is whole blocks, or the first or second half of one
 * (QUADS = 4), which holds when the head width is a multiple of 32. Each block's
 * scale is read once and its quants as realigned 4-byte words.
 *
 * kv_dot_chunk: the chunk's dot product with q (the scale applied once per block).
 * kv_axpy_chunk: acc += w * value over the chunk (w times the scale, once per block).
 * Both round differently from a value-by-value d * q product; the attention they
 * serve is held to a tolerance, not to bits.
 */
template <int T, int QUADS>
static __device__ __forceinline__ float kv_dot_chunk(const unsigned char* row, int e0, const float4* qv, int nValid) {
    float part = 0.f;
    if constexpr (T == KV_F16) {
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            if (i < nValid) {
                const float4 k = kv_load4_t<KV_F16>(row, e0 + 4 * i);
                part = fmaf(qv[i].x, k.x, part);
                part = fmaf(qv[i].y, k.y, part);
                part = fmaf(qv[i].z, k.z, part);
                part = fmaf(qv[i].w, k.w, part);
            }
        }
    } else {
        constexpr int QPB = QUADS < 8 ? QUADS : 8;
        constexpr int NB = QUADS / QPB;
        constexpr int QW = T == KV_Q8_0 ? QPB : 4; // quant words a block's chunk reads
        const int q0 = (e0 & (KV_BLOCK - 1)) >> 2; // 0, or 4 for the second half of a block
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b * QPB < nValid) {
                const unsigned char* blk = row + (size_t)(e0 / KV_BLOCK + b) * kv_block_bytes_t<T>();
                unsigned int words[QW];
                kv_words<QW>(blk + 2 + (T == KV_Q8_0 ? 4 * q0 : 0), words);
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < QPB; q++) {
                    const int i = b * QPB + q;
                    if (i < nValid) {
                        const float4 k = kv_quad_ints<T>(words, T == KV_Q8_0 ? q : (q & 3), q0 + q >= 4);
                        s = fmaf(qv[i].x, k.x, s);
                        s = fmaf(qv[i].y, k.y, s);
                        s = fmaf(qv[i].z, k.z, s);
                        s = fmaf(qv[i].w, k.w, s);
                    }
                }
                part = fmaf(kv_scale(blk), s, part);
            }
        }
    }
    return part;
}

template <int T, int QUADS>
static __device__ __forceinline__ void kv_axpy_chunk(const unsigned char* row, int e0, float w, float4* acc,
        int nValid) {
    if constexpr (T == KV_F16) {
#pragma unroll
        for (int i = 0; i < QUADS; i++) {
            if (i < nValid) {
                const float4 v = kv_load4_t<KV_F16>(row, e0 + 4 * i);
                acc[i].x = fmaf(w, v.x, acc[i].x);
                acc[i].y = fmaf(w, v.y, acc[i].y);
                acc[i].z = fmaf(w, v.z, acc[i].z);
                acc[i].w = fmaf(w, v.w, acc[i].w);
            }
        }
    } else {
        constexpr int QPB = QUADS < 8 ? QUADS : 8;
        constexpr int NB = QUADS / QPB;
        constexpr int QW = T == KV_Q8_0 ? QPB : 4;
        const int q0 = (e0 & (KV_BLOCK - 1)) >> 2;
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b * QPB < nValid) {
                const unsigned char* blk = row + (size_t)(e0 / KV_BLOCK + b) * kv_block_bytes_t<T>();
                unsigned int words[QW];
                kv_words<QW>(blk + 2 + (T == KV_Q8_0 ? 4 * q0 : 0), words);
                const float wd = w * kv_scale(blk);
#pragma unroll
                for (int q = 0; q < QPB; q++) {
                    const int i = b * QPB + q;
                    if (i < nValid) {
                        const float4 v = kv_quad_ints<T>(words, T == KV_Q8_0 ? q : (q & 3), q0 + q >= 4);
                        acc[i].x = fmaf(wd, v.x, acc[i].x);
                        acc[i].y = fmaf(wd, v.y, acc[i].y);
                        acc[i].z = fmaf(wd, v.z, acc[i].z);
                        acc[i].w = fmaf(wd, v.w, acc[i].w);
                    }
                }
            }
        }
    }
}

/* Math.round(float): the closest int, ties toward positive infinity. */
static __device__ __forceinline__ int kv_round_half_up(float s) {
    const float f = floorf(s);
    return (int)f + (__fsub_rn(s, f) >= 0.5f ? 1 : 0);
}

#define KV_FULL_MASK 0xffffffffu

/*
 * Block b (32 values) of a row, encoded by one warp: lane i holds value i of the block
 * in x. FP16 (prefill_window.cu's to_half rounding), or the host codec's block
 * (Q8_0KvCodec / Q4_0KvCodec.encodeBlock), byte for byte: every lane applies the
 * scalar encoder's operations to its own value, and the block's largest magnitude is
 * reduced across the warp with the scalar scan's comparisons (a NaN or a zero never
 * becomes it; for q4_0 the first index among equal magnitudes wins, as the scan keeps
 * the first). The whole warp must call it.
 */
static __device__ __forceinline__ void kv_encode_block_warp(int type, float x, unsigned char* row, int b, int lane) {
    if (type == KV_F16) {
        reinterpret_cast<half*>(row)[(size_t)b * KV_BLOCK + lane] = __float2half_rn(x);
        return;
    }
    const float ax = fabsf(x);
    const bool counts = ax > 0.f;
    if (type == KV_Q8_0) {
        float amax = counts ? ax : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float other = __shfl_xor_sync(KV_FULL_MASK, amax, o);
            amax = other > amax ? other : amax;
        }
        const float d = __fdiv_rn(amax, 127.f);
        const float id = d > 0.f ? __fdiv_rn(1.f, d) : 0.f;
        unsigned char* blk = row + (size_t)b * 34;
        if (lane == 0)
            kv_store_scale(blk, d);
        int q = kv_round_half_up(__fmul_rn(x, id));
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        blk[2 + lane] = (unsigned char)(signed char)q;
        return;
    }
    float key = counts ? ax : 0.f;
    int idx = counts ? lane : KV_BLOCK;
    float mx = counts ? x : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float k2 = __shfl_xor_sync(KV_FULL_MASK, key, o);
        const int i2 = __shfl_xor_sync(KV_FULL_MASK, idx, o);
        const float m2 = __shfl_xor_sync(KV_FULL_MASK, mx, o);
        if (k2 > key || (k2 == key && i2 < idx)) {
            key = k2;
            idx = i2;
            mx = m2;
        }
    }
    const float d = __fdiv_rn(mx, -8.f);
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    unsigned char* blk = row + (size_t)b * 18;
    const float xh = __shfl_down_sync(KV_FULL_MASK, x, KV_BLOCK / 2);
    if (lane == 0)
        kv_store_scale(blk, d);
    if (lane < KV_BLOCK / 2) {
        const int lo = min(15, (int)__fadd_rn(__fmul_rn(x, id), 8.5f));
        const int hi = min(15, (int)__fadd_rn(__fmul_rn(xh, id), 8.5f));
        blk[2 + lane] = (unsigned char)(lo | (hi << 4));
    }
}

#endif /* JUNO_KV_FORMAT_CUH */
