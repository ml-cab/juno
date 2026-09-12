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
 * Forward-only Q4_K packed frozen projection for LoRA playback MMQ.
 *
 * <p>Wraps {@link DeviceQ4KMatrix} + {@link GpuMatVec}; no transpose API (training
 * stays on {@link ResidentWeightMatrix}).
 */
final class ResidentQ4KWeight implements AutoCloseable {

	private final GpuMatVec gpu;
	private final DeviceQ4KMatrix q4;
	private boolean closed;

	private ResidentQ4KWeight(GpuMatVec gpu, DeviceQ4KMatrix q4) {
		this.gpu = gpu;
		this.q4 = q4;
	}

	/** Upload packed Q4_K bytes; {@code cols} must be divisible by {@code QK_K}. */
	static ResidentQ4KWeight upload(GpuMatVec gpu, byte[] raw, int rows, int cols) {
		return upload(gpu, raw, rows, cols, QuantizationLayout.TYPE_Q4_K);
	}

	/** Upload packed K-quant bytes ({@code typeId} in Q4_K / Q5_K / Q6_K). */
	static ResidentQ4KWeight upload(GpuMatVec gpu, byte[] raw, int rows, int cols, int typeId) {
		return new ResidentQ4KWeight(gpu, gpu.uploadKQuant(raw, rows, cols, typeId));
	}

	float[] sgemv(float[] x) {
		ensureOpen();
		return gpu.sgemv(q4, x);
	}

	int rows() {
		return q4.rows();
	}

	int cols() {
		return q4.cols();
	}

	boolean isClosed() {
		return closed;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		if (q4 != null && !q4.isClosed())
			q4.close();
	}

	private void ensureOpen() {
		if (closed)
			throw new IllegalStateException("ResidentQ4KWeight is closed");
	}
}
