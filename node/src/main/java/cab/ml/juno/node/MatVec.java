/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

/**
 * Hardware backend for matrix-vector multiply: y[rows] = A[rows, cols] ×
 * x[cols].
 *
 * <p>
 * Separates the <em>compute substrate</em> (CPU threads, CUDA, Vulkan, …) from
 * the <em>transformer architecture</em> ({@link LlamaTransformerHandler},
 * {@link Phi3TransformerHandler}, …). Any transformer handler accepts a
 * {@code MatVecBackend} at construction time — swapping backends changes where
 * the arithmetic runs without touching the model logic.
 *
 * <p>
 * A is stored row-major: {@code A[r, c] = weights[r * cols + c]}.
 *
 * <p>
 * Implementations:
 * <ul>
 * <li>{@link CpuMatVec} — parallel {@code IntStream} across
 * {@code ForkJoinPool.commonPool()}; used on CPU-only nodes and as the
 * reference implementation in tests.
 * <li>{@link CudaMatVec} — {@code cublasSgemv_v2} via JCublas2; used on
 * Nvidia GPU nodes (CUDA 12.x).
 * </ul>
 *
 * <p>
 * Contract:
 * <ul>
 * <li>A and x are not mutated.
 * <li>Returns a new {@code float[]} of length {@code rows}.
 * <li>Throws {@link IllegalArgumentException} if dimensions are inconsistent.
 * <li>Thread-safe: implementations may be called concurrently for different
 * requests. {@link CudaMatVec} uses per-thread device scratch and CUDA streams while
 * serializing cuBLAS work on a shared {@link GpuContext} lock.
 * </ul>
 */
public interface MatVec {

	/**
	 * Compute y = A * x.
	 *
	 * @param A    weight matrix, row-major, length rows * cols
	 * @param x    input vector, length cols
	 * @param rows number of output elements
	 * @param cols number of input elements (inner dimension)
	 * @return new float[rows] — the result vector
	 */
	float[] sgemv(float[] A, float[] x, int rows, int cols);

	/**
     * Compute y = A * x with {@code A} already on the device (see
     * {@link DeviceFloatMatrix}).
     *
     * @throws UnsupportedOperationException for backends that only support host
     *                                       weights (e.g. {@link CpuMatVec})
     */
    default float[] sgemv(DeviceFloatMatrix A, float[] x) {
        throw new UnsupportedOperationException(
            "device-resident weights are not supported by this GpuMatVec implementation");
    }

    /**
     * Compute y = A * x with {@code A} in FP16 on the device ({@link DeviceHalfMatrix}).
     *
     * @throws UnsupportedOperationException for backends that only support host weights
     */
    default float[] sgemv(DeviceHalfMatrix A, float[] x) {
        throw new UnsupportedOperationException(
                "FP16 device-resident weights are not supported by this MatVec implementation");
    }

    /**
     * Compute Y = A * X for a batch of B input columns in one call.
     * A: [rows, cols] row-major (unchanged from sgemv). X: [B][cols].
     * Returns Y: [B][rows].
     *
     * <p>Weight-stationary default: implementations that override this should load
     * each weight row once and multiply it against all B columns before advancing
     * to the next row, maximising weight reuse — the actual performance win over
     * calling {@link #sgemv} B times, which re-streams the full weight matrix from
     * memory B times.
     *
     * <p>Default (correctness-preserving): B calls to {@link #sgemv}. Every existing
     * {@link MatVec} implementation is correct by construction; only
     * {@link CpuMatVec} overrides for speed.
     *
     * @param A    weight matrix, row-major, length rows * cols
     * @param X    batch of input vectors, X[b] has length cols
     * @param rows number of output elements per vector
     * @param cols number of input elements (inner dimension)
     * @return Y[b] = A * X[b] for each b in 0..X.length
     */
    default float[][] sgemm(float[] A, float[][] X, int rows, int cols) {
        float[][] Y = new float[X.length][];
        for (int b = 0; b < X.length; b++) Y[b] = sgemv(A, X[b], rows, cols);
        return Y;
    }

    /**
     * Compute Y = A * X with {@code A} already on the device as FP32
     * ({@link DeviceFloatMatrix}).
     *
     * <p>Default: B calls to {@link #sgemv(DeviceFloatMatrix, float[])}. GPU backends
     * may override with a single batched BLAS SGEMM call.
     */
    default float[][] sgemm(DeviceFloatMatrix A, float[][] X) {
        float[][] Y = new float[X.length][];
        for (int b = 0; b < X.length; b++) Y[b] = sgemv(A, X[b]);
        return Y;
    }

    /**
     * Compute Y = A * X with {@code A} in FP16 on the device
     * ({@link DeviceHalfMatrix}).
     *
     * <p>Default: B calls to {@link #sgemv(DeviceHalfMatrix, float[])}. GPU backends
     * may override with a single batched BLAS SGEMM call.
     */
    default float[][] sgemm(DeviceHalfMatrix A, float[][] X) {
        float[][] Y = new float[X.length][];
        for (int b = 0; b < X.length; b++) Y[b] = sgemv(A, X[b]);
        return Y;
    }
}