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
 * One frozen projection matrix resident on a {@link GpuMatVec} backend.
 *
 * <p>Holds either FP16 or FP32 device storage (exactly one). Used by
 * {@link LoraTrainableHandler} for both forward {@code W*x} and transpose
 * backward {@code W^T*g} without duplicating half/float call sites.
 */
final class ResidentWeightMatrix implements AutoCloseable {

	private final GpuMatVec gpu;
	private final DeviceHalfMatrix half;
	private final DeviceFloatMatrix fp32;
	private boolean closed;

	private ResidentWeightMatrix(GpuMatVec gpu, DeviceHalfMatrix half, DeviceFloatMatrix fp32) {
		this.gpu = gpu;
		this.half = half;
		this.fp32 = fp32;
	}

	static ResidentWeightMatrix uploadHalf(GpuMatVec gpu, float[] host, int rows, int cols) {
		return new ResidentWeightMatrix(gpu, gpu.uploadHalf(host, rows, cols), null);
	}

	static ResidentWeightMatrix uploadFp32(GpuMatVec gpu, float[] host, int rows, int cols) {
		return new ResidentWeightMatrix(gpu, null, gpu.upload(host, rows, cols));
	}

	float[] sgemv(float[] x) {
		ensureOpen();
		return half != null ? gpu.sgemv(half, x) : gpu.sgemv(fp32, x);
	}

	float[] sgemvTranspose(float[] g) {
		ensureOpen();
		return half != null ? gpu.sgemvTranspose(half, g) : gpu.sgemvTranspose(fp32, g);
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		if (half != null && !half.isClosed())
			half.close();
		if (fp32 != null && !fp32.isClosed())
			fp32.close();
	}

	boolean isClosed() {
		return closed;
	}

	private void ensureOpen() {
		if (closed)
			throw new IllegalStateException("ResidentWeightMatrix is closed");
	}
}
