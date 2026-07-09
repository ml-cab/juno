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
 * {@link MatVec} backed by the CPU parallel matVec from
 * {@link LlamaTransformerHandler}.
 *
 * <p>
 * Two uses:
 * <ol>
 * <li>CPU-only nodes — default backend when CUDA is unavailable.
 * <li>Tests — {@link MatVecBackendContractTest} runs the full contract suite
 * against this implementation without needing a GPU, ensuring correctness of
 * the contract itself before testing {@link CudaMatVec} on AWS.
 * </ol>
 */
public final class CpuMatVec implements MatVec {

	/** Singleton — stateless, no resources to manage. */
	public static final CpuMatVec INSTANCE = new CpuMatVec();

	CpuMatVec() {
	}

	@Override
	public float[] sgemv(float[] A, float[] x, int rows, int cols) {
		if (A.length != (long) rows * cols)
			throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
		if (x.length != cols)
			throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);
		MatVecEvent evt = new MatVecEvent();
		evt.begin();
		float[] result = LlamaTransformerHandler.matVec(A, x, rows, cols);
		evt.backend(MatVecBackend.CPU);
		evt.rows = rows;
		evt.cols = cols;
		evt.commit();
		return result;
	}

	/**
	 * Batched Y = A * X, weight-stationary: each row of {@code A} is loaded once
	 * and dot-producted against all {@code B} input columns before advancing to the
	 * next row. Compared to calling {@link #sgemv} B times (which re-streams the
	 * full weight matrix from memory B times), this halves memory bandwidth for
	 * {@code B &ge; 2} and scales near-linearly with B on DRAM-bound workloads.
	 *
	 * <p>Row parallelism: {@code IntStream.parallel()} over rows via
	 * {@code ForkJoinPool.commonPool()} — same pool as {@link LlamaTransformerHandler#matVec}.
	 * Each parallel task now does {@code cols * B} multiply-adds instead of {@code cols},
	 * improving the compute-per-task-dispatch ratio.
	 */
	@Override
	public float[][] sgemm(float[] A, float[][] X, int rows, int cols) {
		if (A.length != (long) rows * cols)
			throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
		int B = X.length;
		if (B == 0) return new float[0][];
		for (int b = 0; b < B; b++) {
			if (X[b].length != cols)
				throw new IllegalArgumentException("X[" + b + "].length=" + X[b].length + " != cols=" + cols);
		}
		float[][] Y = new float[B][rows];
		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			int base = r * cols;
			for (int b = 0; b < B; b++) {
				float acc = 0f;
				float[] xb = X[b];
				for (int c = 0; c < cols; c++) {
					acc += A[base + c] * xb[c];
				}
				Y[b][r] = acc;
			}
		});
		return Y;
	}
}