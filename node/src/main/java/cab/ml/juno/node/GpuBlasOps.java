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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Microbatched FP32 frozen linears via {@code cublasSgemm_v2} / {@code rocblas_sgemm}.
 *
 * <p>Row-major {@code W[rows×cols]} uses the same column-major reinterpretation as
 * {@link CudaMatVec} GEMV: {@code lda = cols}. Batch columns are packed
 * column-major by {@link DeviceActivationBatch}.
 *
 * <ul>
 *   <li>Forward {@code Y = W X}: {@code op(W)=T}, {@code op(X)=N},
 *       {@code m=rows}, {@code n=batch}, {@code k=cols}</li>
 *   <li>Transpose {@code dX = W^T G}: {@code op(W)=N}, {@code op(G)=N},
 *       {@code m=cols}, {@code n=batch}, {@code k=rows}</li>
 * </ul>
 *
 * <p>FP16 resident weights are not supported here — callers fall back to sequential
 * {@code sgemv} / {@code sgemvTranspose}.
 */
final class GpuBlasOps implements AutoCloseable {

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private final DeviceActivationBatch scratch;
	private boolean closed;

	GpuBlasOps(GpuContext ctx) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.scratch = new DeviceActivationBatch(ctx);
	}

	static GpuBlasOps of(GpuMatVec gpu) {
		return new GpuBlasOps(gpu.gpuContext());
	}

	/**
	 * {@code Y[b] = W * X[b]} for {@code b in [0, batch)}.
	 *
	 * @param X length ≥ batch; each column length {@code W.cols()}
	 * @return length {@code batch}; each row length {@code W.rows()}
	 */
	float[][] forward(DeviceFloatMatrix W, float[][] X, int batch) {
		ensureOpen();
		if (W == null || W.isClosed())
			throw new IllegalStateException("W must be an open DeviceFloatMatrix");
		if (batch <= 0)
			throw new IllegalArgumentException("batch must be > 0");
		if (X.length < batch)
			throw new IllegalArgumentException("X.length < batch");
		int rows = W.rows();
		int cols = W.cols();
		float[] packedX = DeviceActivationBatch.packColumns(X, batch, cols);
		long bytesX = (long) packedX.length * Float.BYTES;
		long bytesY = (long) rows * batch * Float.BYTES;

		synchronized (ctx.cublasSerializationLock()) {
			MemorySegment dX = scratch.ensureInput(bytesX);
			MemorySegment dY = scratch.ensureOutput(bytesY);
			scratch.copyH2D(dX, packedX);
			callSgemm(
					gpu.opTranspose(), gpu.opNoTranspose(),
					rows, batch, cols,
					W.devicePointer(), cols,
					dX, cols,
					dY, rows);
			float[] packedY = scratch.copyD2H(dY, rows * batch);
			float[][] Y = new float[batch][];
			DeviceActivationBatch.unpackColumns(packedY, Y, batch, rows);
			return Y;
		}
	}

	/**
	 * {@code dX[b] = W^T * G[b]} for {@code b in [0, batch)}.
	 *
	 * @param G length ≥ batch; each column length {@code W.rows()}
	 * @return length {@code batch}; each row length {@code W.cols()}
	 */
	float[][] transpose(DeviceFloatMatrix W, float[][] G, int batch) {
		ensureOpen();
		if (W == null || W.isClosed())
			throw new IllegalStateException("W must be an open DeviceFloatMatrix");
		if (batch <= 0)
			throw new IllegalArgumentException("batch must be > 0");
		if (G.length < batch)
			throw new IllegalArgumentException("G.length < batch");
		int rows = W.rows();
		int cols = W.cols();
		float[] packedG = DeviceActivationBatch.packColumns(G, batch, rows);
		long bytesG = (long) packedG.length * Float.BYTES;
		long bytesZ = (long) cols * batch * Float.BYTES;

		synchronized (ctx.cublasSerializationLock()) {
			MemorySegment dG = scratch.ensureInput(bytesG);
			MemorySegment dZ = scratch.ensureOutput(bytesZ);
			scratch.copyH2D(dG, packedG);
			callSgemm(
					gpu.opNoTranspose(), gpu.opNoTranspose(),
					cols, batch, rows,
					W.devicePointer(), cols,
					dG, rows,
					dZ, cols);
			float[] packedZ = scratch.copyD2H(dZ, cols * batch);
			float[][] dX = new float[batch][];
			DeviceActivationBatch.unpackColumns(packedZ, dX, batch, cols);
			return dX;
		}
	}

	private void callSgemm(int transA, int transB,
			int m, int n, int k,
			MemorySegment A, int lda,
			MemorySegment B, int ldb,
			MemorySegment C, int ldc) {
		try (Arena scalars = Arena.ofConfined()) {
			MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
			MemorySegment beta = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
			GpuBindings.check(
					GpuBindings.callInt(gpu.blasSetPointerMode(), ctx.handle(), gpu.pointerModeHost()),
					"blasSetPointerMode");
			GpuBindings.check(
					GpuBindings.callInt(gpu.blasSgemm(),
							ctx.handle(), transA, transB,
							m, n, k,
							alpha, A, lda,
							B, ldb,
							beta, C, ldc),
					"blasSgemm");
		}
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		scratch.close();
	}

	private void ensureOpen() {
		if (closed)
			throw new IllegalStateException("GpuBlasOps is closed");
	}
}
