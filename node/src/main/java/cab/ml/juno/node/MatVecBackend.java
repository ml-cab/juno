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
 * <li>{@link CpuMatVecBackend} — parallel {@code IntStream} across
 * {@code ForkJoinPool.commonPool()}; used on CPU-only nodes and as the
 * reference implementation in tests.
 * <li>{@link CudaMatVecBackend} — {@code cublasSgemv_v2} via JCublas2; used on
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
 * requests (each call is self-contained with its own device memory).
 * </ul>
 */
public interface MatVecBackend {

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
}