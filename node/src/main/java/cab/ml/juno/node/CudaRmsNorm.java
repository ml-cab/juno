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
 * Handler-facing entry point for the GPU-resident RMS-norm path.
 *
 * <p>{@link #normalizeBatch} dispatches to the real CUDA kernel
 * ({@code rms_norm.cu} / {@link RmsNormKernel}) that computes RMS
 * normalisation (the same math as {@code LlamaTransformerHandler.rmsNorm}/
 * {@code rmsNormInto}) in parallel on-device. Callers fall back to the scalar
 * {@code rmsNorm}/{@code rmsNormInto} CPU path when this returns {@code false}
 * (kernel unavailable).
 *
 * <p><b>Not constructed by default.</b> {@code LlamaTransformerHandler} leaves
 * its reference null: a live decode A/B measured this path as slower than the
 * scalar CPU path, because every call pays its own host-to-device upload, kernel
 * launch and device-to-host download while the surrounding matrix-vector calls
 * each do their own host round trip, so nothing keeps the activation
 * device-resident. The class, its kernel and its unit tests are kept until that
 * residency gap is closed.
 */
final class CudaRmsNorm {

	private final GpuContext ctx;
	private final GpuBindings gpu;

	private static final ThreadLocal<RmsNormScratch> SCRATCH = ThreadLocal.withInitial(RmsNormScratch::new);

	private static final class RmsNormScratch {
		MemorySegment dX, dWeight, dOut;
		long xBytes, weightBytes, outBytes;
		MemorySegment hX, hWeight, hOut;      // pinned host staging, grown as needed
		long hXBytes, hWeightBytes, hOutBytes;
	}

	private CudaRmsNorm(GpuContext ctx) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
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

		RmsNormScratch s = SCRATCH.get();
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
		return true;
	}
}
