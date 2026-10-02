/*
 * GPU rotary position embedding (RoPE) for Juno, applied in place to a
 * [rows][nHeads * headDim] activation batch that is already device-resident.
 *
 * Same math as LlamaTransformerHandler.rope: adjacent-pair rotation
 * (x[2i], x[2i+1]) within each head, by angle pos * invFreq[i] with
 * invFreq[i] = 1 / theta^(2i / headDim), where row r sits at position
 * startPos + r. One launch covers a single decode row (rows = 1) and a
 * prefill window of consecutive positions (rows = window size). A second
 * entry, rope_split_half below, rotates the split-half pairs (x[i],
 * x[i + headDim/2]) with the same arithmetic.
 *
 * Precision follows the CPU path step by step so the two agree to within one
 * float rounding of a sine or cosine:
 *   - invFreq is computed on the host in double by the same expression the CPU
 *     path uses, and uploaded once per model;
 *   - the angle and its sine and cosine are computed here in double, then
 *     rounded to float, as the CPU path does;
 *   - each multiply and add of the rotation is rounded separately
 *     (__fmul_rn / __fadd_rn / __fsub_rn), so the compiler cannot contract
 *     them into fused multiply-adds, which the CPU path does not use.
 * A single-precision angle would not do: at position 30000 its rounding error
 * alone is about 2e-3 radians.
 *
 * One thread per rotated pair; grid covers rows * nHeads * headDim / 2 pairs.
 *
 * Compile (Pascal+ reference SKU - sm_61, matching q4k_gemv.cu):
 *   nvcc -ptx -arch=compute_61 -O3 \
 *     -o ../resources/cab/ml/juno/node/rope.ptx rope.cu
 */

#define ROPE_THREADS 256

extern "C" __global__ void __launch_bounds__(ROPE_THREADS)
rope(
        float* __restrict__ x,              // [rows][nHeads * headDim], rotated in place
        const double* __restrict__ invFreq, // [headDim / 2]
        int rows,
        int nHeads,
        int headDim,
        int startPos) {
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
    sincos((double)(startPos + row) * invFreq[i], &s, &c);
    const float sinA = (float)s;
    const float cosA = (float)c;

    float* pair = x + (size_t)row * nHeads * headDim + (size_t)h * headDim + 2 * i;
    const float x0 = pair[0];
    const float x1 = pair[1];
    pair[0] = __fsub_rn(__fmul_rn(x0, cosA), __fmul_rn(x1, sinA));
    pair[1] = __fadd_rn(__fmul_rn(x0, sinA), __fmul_rn(x1, cosA));
}

/*
 * Split-half (rotate-half) pairing: pair i of a head is (x[i], x[i + headDim/2])
 * rather than (x[2i], x[2i+1]). Same frequencies, same double-precision angle and
 * the same separately rounded rotation as rope above; only the pair members
 * differ. Same math as LlamaTransformerHandler.rope with RopePairing.SPLIT_HALF,
 * which the Qwen2 family uses.
 */
extern "C" __global__ void __launch_bounds__(ROPE_THREADS)
rope_split_half(
        float* __restrict__ x,              // [rows][nHeads * headDim], rotated in place
        const double* __restrict__ invFreq, // [headDim / 2]
        int rows,
        int nHeads,
        int headDim,
        int startPos) {
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
    sincos((double)(startPos + row) * invFreq[i], &s, &c);
    const float sinA = (float)s;
    const float cosA = (float)c;

    float* head = x + (size_t)row * nHeads * headDim + (size_t)h * headDim;
    const float x0 = head[i];
    const float x1 = head[i + half];
    head[i] = __fsub_rn(__fmul_rn(x0, cosA), __fmul_rn(x1, sinA));
    head[i + half] = __fadd_rn(__fmul_rn(x0, sinA), __fmul_rn(x1, cosA));
}
