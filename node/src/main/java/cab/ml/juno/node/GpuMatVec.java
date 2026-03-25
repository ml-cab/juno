/*
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
 * Matrix-vector multiply contract: y[rows] = A[rows, cols] × x[cols].
 *
 * A is stored row-major: A[r, c] = weights[r * cols + c].
 *
 * Implementations:
 *   CublasMatVec   — real cuBLAS cublasSgemv via org.bytedeco cuda (Nvidia GPU)
 *   CpuMatVec      — thin wrapper around CpuForwardPassHandler.matVec(),
 *                    used as the CPU reference in tests and CPU-only nodes
 *
 * Contract: - A and x are not mutated. - Returns a new float[] of length rows.
 * - Thread-safe: implementations may be called concurrently for different
 * requests (each call is self-contained with its own device memory).
 */
public interface GpuMatVec {

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
}