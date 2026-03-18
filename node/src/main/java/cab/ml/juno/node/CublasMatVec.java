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

import java.util.logging.Logger;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;

/**
 * GpuMatVec backed by cublasSgemv on an Nvidia GPU.
 *
 * Computes y[rows] = A[rows, cols] × x[cols] where A is row-major.
 *
 * Row-major to cuBLAS mapping: cuBLAS is column-major. A row-major matrix
 * A[rows x cols] is identical in memory to the transpose of a column-major
 * matrix A^T[cols x rows]. To compute y = A * x using cuBLAS we therefore call
 * cublasSgemv with CUBLAS_OP_T (transpose), m=rows, n=cols, lda=cols. This asks
 * cuBLAS to compute y = (A^T)^T * x = A * x. No data is copied or reordered.
 *
 * Memory management: Each sgemv call allocates device buffers, copies data,
 * executes, copies result back, and frees. This is correct and simple (KISS). A
 * future GpuMemoryPool optimisation can replace the per-call
 * cudaMalloc/cudaFree with slab allocation — the interface contract is
 * unchanged.
 *
 * Thread safety: cuBLAS handles are safe for concurrent calls from different
 * Java threads. Each call has its own device buffers so there is no shared
 * mutable state.
 */
public final class CublasMatVec implements GpuMatVec {

	@SuppressWarnings("unused")
	private static final Logger log = Logger.getLogger(CublasMatVec.class.getName());

	private static final Pointer ALPHA = Pointer.to(new float[] { 1.0f });
	private static final Pointer BETA = Pointer.to(new float[] { 0.0f });

	private final GpuContext ctx;

	/**
	 * @param ctx an open GpuContext — must outlive all sgemv calls on this instance
	 */
	public CublasMatVec(GpuContext ctx) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		this.ctx = ctx;
	}

	/**
	 * Compute y = A * x on the GPU via cublasSgemv.
	 *
	 * <p>
	 * Steps:
	 * <ol>
	 * <li>cudaMalloc device buffers for A, x, y</li>
	 * <li>cudaMemcpy host → device for A and x</li>
	 * <li>cublasSgemv_v2 with CUBLAS_OP_T</li>
	 * <li>cudaMemcpy device → host for y</li>
	 * <li>cudaFree all device buffers</li>
	 * </ol>
	 */
	@Override
	public float[] sgemv(float[] A, float[] x, int rows, int cols) {
		if (A.length != (long) rows * cols)
			throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
		if (x.length != cols)
			throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

		float[] y = new float[rows];

		// ── Allocate device memory ────────────────────────────────────────────
		Pointer d_A = new Pointer();
		Pointer d_x = new Pointer();
		Pointer d_y = new Pointer();

		long bytesA = (long) rows * cols * Sizeof.FLOAT;
		long bytesX = (long) cols * Sizeof.FLOAT;
		long bytesY = (long) rows * Sizeof.FLOAT;

		JCuda.cudaMalloc(d_A, bytesA);
		JCuda.cudaMalloc(d_x, bytesX);
		JCuda.cudaMalloc(d_y, bytesY);

		try {
			// ── Copy host → device ────────────────────────────────────────────
			JCuda.cudaMemcpy(d_A, Pointer.to(A), bytesA, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaMemcpy(d_x, Pointer.to(x), bytesX, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice);

			// ── cublasSgemv ───────────────────────────────────────────────────
			// Row-major A[rows x cols] is treated as column-major A^T[cols x rows].
			// CUBLAS_OP_T transposes it back → op(A^T) = A, giving y = A * x.
			// m = rows (output size), n = cols (input size), lda = cols.
			JCublas2.cublasSgemv(ctx.handle(), cublasOperation.CUBLAS_OP_T, cols, // m — rows of the stored column-major
																					// matrix
					rows, // n — cols of the stored column-major matrix
					ALPHA, d_A, cols, // lda = cols (leading dimension of row-major A)
					d_x, 1, BETA, d_y, 1);

			// ── Copy device → host ────────────────────────────────────────────
			JCuda.cudaMemcpy(Pointer.to(y), d_y, bytesY, jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost);

		} finally {
			// ── Free device memory (always, even on exception) ────────────────
			JCuda.cudaFree(d_A);
			JCuda.cudaFree(d_x);
			JCuda.cudaFree(d_y);
		}

		return y;
	}
}