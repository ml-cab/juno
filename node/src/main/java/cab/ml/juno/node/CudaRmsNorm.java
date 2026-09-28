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

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Handler-facing entry point for the GPU RMS-norm kernel ({@code rms_norm.cu} /
 * {@link RmsNormKernel}), which computes RMS normalisation - the same math as
 * {@code LlamaTransformerHandler.rmsNorm}/{@code rmsNormInto} - in parallel
 * on-device. Two paths, differing only in where the activation lives:
 * <ul>
 *   <li>{@link #normalizeBatch}: host rows in, host rows out. Every call uploads
 *       the activation and the weight, launches, and downloads the result.</li>
 *   <li>{@link #normalizeResident}: a {@link ResidentActivation} in, another one
 *       out, with a weight already on the device. Nothing crosses to the host and
 *       nothing waits; the next operation on the same {@link ResidentChain} reads
 *       the result where it is.</li>
 * </ul>
 * Both return {@code false} when the kernel is unavailable, and callers fall back
 * to the scalar CPU path.
 *
 * <p><b>Not constructed by default.</b> {@code LlamaTransformerHandler} leaves
 * its reference null: a live decode A/B measured the round-trip path as slower
 * than the scalar CPU path, because every call pays its own host-to-device
 * upload, kernel launch and device-to-host download while the surrounding
 * matrix-vector calls each do their own host round trip, so nothing keeps the
 * activation device-resident. The resident path removes that round trip; it is
 * measured by {@code ResidentChainMicrobench} and is not yet wired into a
 * handler.
 */
final class CudaRmsNorm {

	private final GpuContext ctx;
	private final GpuBindings gpu;

	/**
	 * Round-trip scratch, pooled by concurrent callers rather than kept per
	 * thread, so threads that come and go do not each leave device and pinned
	 * host buffers behind. The resident path uses none of it.
	 */
	private final DeviceScratchPool<RmsNormScratch> scratch;

	private static final class RmsNormScratch {
		MemorySegment dX, dWeight, dOut;
		long xBytes, weightBytes, outBytes;
		MemorySegment hX, hWeight, hOut;      // pinned host staging, grown as needed
		long hXBytes, hWeightBytes, hOutBytes;

		long deviceBytes() {
			return xBytes + weightBytes + outBytes;
		}
	}

	private CudaRmsNorm(GpuContext ctx) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.scratch = new DeviceScratchPool<>(RmsNormScratch::new, this::free);
	}

	private void free(RmsNormScratch s) {
		gpu.deviceFree(s.dX);
		gpu.deviceFree(s.dWeight);
		gpu.deviceFree(s.dOut);
		gpu.hostFree(s.hX);
		gpu.hostFree(s.hWeight);
		gpu.hostFree(s.hOut);
		s.dX = s.dWeight = s.dOut = s.hX = s.hWeight = s.hOut = null;
		s.xBytes = s.weightBytes = s.outBytes = s.hXBytes = s.hWeightBytes = s.hOutBytes = 0L;
	}

	/** Frees the pooled round-trip scratch. A call after this still works and frees its own. */
	void close() {
		scratch.close();
	}

	/** Device bytes held by idle pooled round-trip scratch entries. */
	long scratchDeviceBytes() {
		return scratch.idleBytes(RmsNormScratch::deviceBytes);
	}

	/** Returns a usable instance for {@code ctx}, or {@code null} when the backend isn't CUDA. */
	static CudaRmsNorm tryCreate(GpuContext ctx) {
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return null;
		return new CudaRmsNorm(ctx);
	}

	/**
	 * Real GPU-parallel RMS norm for {@code B = x.length} rows sharing one
	 * {@code weight} vector (the same layer's attn_norm/ffn_norm tensor), in
	 * one kernel launch. Writes into {@code out[b]} (allocated if needed or
	 * mismatched in length — reused otherwise, matching the zero-alloc
	 * workspace convention of the batched CPU path).
	 *
	 * @return {@code false} (writing nothing) if the kernel failed to load —
	 *         caller must fall back to the scalar {@code rmsNorm}/{@code rmsNormInto} path
	 */
	boolean normalizeBatch(float[][] x, float[] weight, float eps, float[][] out) {
		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		if (kernel == null)
			return false;

		int batch = x.length;
		int dim = weight.length;

		long xBytes = (long) batch * dim * Float.BYTES;
		long weightBytes = (long) dim * Float.BYTES;
		long outBytes = xBytes;

		RmsNormScratch s = scratch.acquire();
		try {
			normalizeWith(s, kernel, x, weight, eps, out, batch, dim, xBytes, weightBytes, outBytes);
		} finally {
			scratch.release(s);
		}
		return true;
	}

	private void normalizeWith(RmsNormScratch s, RmsNormKernel kernel, float[][] x, float[] weight, float eps,
			float[][] out, int batch, int dim, long xBytes, long weightBytes, long outBytes) {
		int dev = ctx.deviceIndex();
		if (s.xBytes < xBytes) {
			gpu.deviceFree(s.dX);
			s.dX = gpu.deviceMalloc(dev, xBytes);
			s.xBytes = xBytes;
		}
		if (s.weightBytes < weightBytes) {
			gpu.deviceFree(s.dWeight);
			s.dWeight = gpu.deviceMalloc(dev, weightBytes);
			s.weightBytes = weightBytes;
		}
		if (s.outBytes < outBytes) {
			gpu.deviceFree(s.dOut);
			s.dOut = gpu.deviceMalloc(dev, outBytes);
			s.outBytes = outBytes;
		}
		if (s.hXBytes < xBytes) {
			gpu.hostFree(s.hX);
			s.hX = gpu.hostMalloc(dev, xBytes);
			s.hXBytes = xBytes;
		}
		if (s.hWeightBytes < weightBytes) {
			gpu.hostFree(s.hWeight);
			s.hWeight = gpu.hostMalloc(dev, weightBytes);
			s.hWeightBytes = weightBytes;
		}
		if (s.hOutBytes < outBytes) {
			gpu.hostFree(s.hOut);
			s.hOut = gpu.hostMalloc(dev, outBytes);
			s.hOutBytes = outBytes;
		}

		MemorySegment hostX = s.hX;
		for (int b = 0; b < batch; b++)
			MemorySegment.copy(x[b], 0, hostX, JAVA_FLOAT, (long) b * dim * Float.BYTES, dim);
		GpuBindings.check(
				GpuBindings.callInt(gpu.gpuMemcpy(), s.dX, hostX, xBytes, GpuBindings.H2D),
				"memcpy(rmsNorm xBatch H2D)");

		MemorySegment hostWeight = s.hWeight;
		MemorySegment.copy(weight, 0, hostWeight, JAVA_FLOAT, 0, dim);
		GpuBindings.check(
				GpuBindings.callInt(gpu.gpuMemcpy(), s.dWeight, hostWeight, weightBytes, GpuBindings.H2D),
				"memcpy(rmsNorm weight H2D)");

		kernel.launch(s.dX, s.dWeight, s.dOut, batch, dim, eps, null);

		MemorySegment hostOut = s.hOut;
		GpuBindings.check(
				GpuBindings.callInt(gpu.gpuMemcpy(), hostOut, s.dOut, outBytes, GpuBindings.D2H),
				"memcpy(rmsNorm outBatch D2H)");
		for (int b = 0; b < batch; b++) {
			if (out[b] == null || out[b].length != dim)
				out[b] = new float[dim];
			MemorySegment.copy(hostOut, JAVA_FLOAT, (long) b * dim * Float.BYTES, out[b], 0, dim);
		}
	}

	/**
	 * Normalizes every valid row of {@code in} into {@code out} on the device, one
	 * kernel launch, asynchronous on their chain's stream. {@code weight} is the
	 * layer's norm vector as a {@code 1 x dim} device matrix, uploaded once rather
	 * than per call. Never in place: the input of a pre-norm block is the residual
	 * stream, which the block adds back afterwards.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load - caller
	 *         must fall back to the scalar {@code rmsNorm}/{@code rmsNormInto} path
	 */
	boolean normalizeResident(ResidentActivation in, DeviceFloatMatrix weight, float eps, ResidentActivation out) {
		in.requireOpen();
		out.requireOpen();
		if (weight.isClosed())
			throw new IllegalStateException("norm weight matrix is closed");
		if (in == out)
			throw new IllegalArgumentException(
					"RMS norm must not run in place: its input is the residual stream the block adds back");
		if (in.chain() != out.chain())
			throw new IllegalArgumentException(
					"input and output are on different chains; operations are ordered by one chain's stream");
		if (weight.rows() != 1 || weight.cols() != in.dim())
			throw new IllegalArgumentException("norm weight is " + weight.rows() + " x " + weight.cols()
					+ ", expected 1 x " + in.dim());
		if (out.dim() != in.dim())
			throw new IllegalArgumentException("output is " + out.dim() + " wide, input " + in.dim());
		if (in.rows() > out.capacityRows())
			throw new IllegalArgumentException(
					in.rows() + " input rows do not fit an output of capacity " + out.capacityRows());
		if (in.rows() == 0)
			throw new IllegalStateException("input activation holds no rows to normalize");

		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		if (kernel == null)
			return false;
		kernel.launchResident(in.devicePointer(), weight.devicePointer(), out.devicePointer(), in.rows(), in.dim(),
				eps, in.chain().stream());
		out.markWritten(in.rows());
		return true;
	}
}
