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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.cuda.global.cudart;

/**
 * Row-major FP16 (IEEE binary16) weight matrix on the GPU — half the VRAM of
 * {@link DeviceFloatMatrix} for the same logical shape.
 *
 * <p>Used with {@link CudaMatVec#sgemv(DeviceHalfMatrix, float[])} (mixed
 * FP16/FP32 cuBLAS path) so activations stay float32 while weights stay compact.
 */
public final class DeviceHalfMatrix implements AutoCloseable {

	private final GpuContext ctx;
	private final Pointer dA;
	private final int rows;
	private final int cols;
	private volatile boolean closed;

	private DeviceHalfMatrix(GpuContext ctx, Pointer dA, int rows, int cols) {
		this.ctx = ctx;
		this.dA = dA;
		this.rows = rows;
		this.cols = cols;
	}

	/**
	 * Converts {@code host} float32 weights to FP16 and uploads row-major
	 * {@code [rows × cols]} to the device.
	 */
	public static DeviceHalfMatrix uploadFromFloat32(GpuContext ctx, float[] host, int rows, int cols) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		if (host.length != (long) rows * cols)
			throw new IllegalArgumentException(
					"host.length=" + host.length + " != rows*cols=" + ((long) rows * cols));
		int n = rows * cols;
		byte[] bytes = new byte[n * 2];
		ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++)
			bb.putShort(i * 2, Float.floatToFloat16(host[i]));
		long bytesTotal = (long) n * 2;
		PointerPointer pp = new PointerPointer(1);
		try {
			checkCuda(cudart.cudaSetDevice(ctx.deviceIndex()), "cudaSetDevice");
			int rc = cudart.cudaMalloc(pp, bytesTotal);
			if (rc != 0)
				throw new IllegalStateException("cudaMalloc failed: " + rc);
			Pointer d = pp.get(0);
			org.bytedeco.javacpp.BytePointer hp = new org.bytedeco.javacpp.BytePointer(bytes);
			try {
				checkCuda(cudart.cudaMemcpy(d, hp, bytesTotal, cudart.cudaMemcpyHostToDevice), "cudaMemcpy(A H2D)");
			} finally {
				hp.close();
			}
			return new DeviceHalfMatrix(ctx, d, rows, cols);
		} finally {
			pp.close();
		}
	}

	public int rows() {
		return rows;
	}

	public int cols() {
		return cols;
	}

	Pointer devicePointer() {
		if (closed)
			throw new IllegalStateException("DeviceHalfMatrix already closed");
		return dA;
	}

	public boolean isClosed() {
		return closed;
	}

	@Override
	public void close() {
		if (!closed) {
			closed = true;
			if (dA != null) {
				cudart.cudaSetDevice(ctx.deviceIndex());
				cudart.cudaFree(dA);
			}
		}
	}

	private static void checkCuda(int rc, String op) {
		if (rc != 0)
			throw new IllegalStateException(op + " failed: " + rc);
	}
}
