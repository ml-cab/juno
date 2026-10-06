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

/**
 * The rest of a decode layer after attention, issued on the decode region's
 * stream ({@link ResidentQkvPath}): output projection, first residual add, FFN
 * RMS norm, gate and up projections, SwiGLU, down projection and second residual
 * add. The residual row stays on the device throughout; the layer output lands
 * both in the residual activation (the next layer's input) and in the packed
 * download row.
 *
 * <pre>
 *   attn -> W_o -> x += . -> norm -> W_gate, W_up -> silu(gate) * up -> W_down -> x += . (also into the download row)
 * </pre>
 *
 * <p>Each step is the kernel the op-at-a-time path runs, fed the same bits: the
 * projections are the K-quant MMQ GEMVs of {@link CudaMatVec#sgemv(DeviceQ4KMatrix, float[])}
 * (gate and up share one Q8_1 quantization, as {@link CudaMatVec#sgemvSameX} does),
 * the norm is the RMS-norm kernel of {@link CudaRmsNorm}, and SwiGLU and the adds
 * round each operation separately exactly as the host loops do. So the layer
 * output matches that path bit for bit.
 *
 * <p>Holds no per-call state: the buffers it writes belong to the caller's region
 * ({@link #scratchBytes}). Device work is issued under the caller's lock.
 */
final class ResidentLayerTail implements AutoCloseable {

	/** Per-layer device weights the tail runs from; a null entry makes that layer run only the attention half. */
	record Weights(float[][] ffnNorm, DeviceQ4KMatrix[] wo, DeviceQ4KMatrix[] wGate, DeviceQ4KMatrix[] wUp,
			DeviceQ4KMatrix[] wDown) {
	}

	private final Q4KMmqKernel mmq;
	private final CudaRmsNorm norm;
	private final PrefillWindowKernels kernels;
	private final int hidden;
	private final int inter;
	private final float eps;
	private final DeviceQ4KMatrix[] wo;
	private final DeviceQ4KMatrix[] wGate;
	private final DeviceQ4KMatrix[] wUp;
	private final DeviceQ4KMatrix[] wDown;
	/** Per layer: the FFN norm weight as a 1 x hidden device matrix, null where the tail does not run. */
	private final DeviceFloatMatrix[] ffnNormWeight;

	private ResidentLayerTail(Q4KMmqKernel mmq, CudaRmsNorm norm, PrefillWindowKernels kernels, LlamaConfig cfg,
			Weights w, DeviceFloatMatrix[] ffnNormWeight) {
		this.mmq = mmq;
		this.norm = norm;
		this.kernels = kernels;
		this.hidden = cfg.hiddenDim();
		this.inter = cfg.intermediateSize();
		this.eps = cfg.rmsNormEps();
		this.wo = w.wo();
		this.wGate = w.wGate();
		this.wUp = w.wUp();
		this.wDown = w.wDown();
		this.ffnNormWeight = ffnNormWeight;
	}

	/**
	 * Uploads the FFN norm weight of every layer whose head runs on the device
	 * ({@code headNorm[li]} not null) and whose four tail matrices are all on the
	 * device with the model's shape.
	 */
	static ResidentLayerTail create(GpuContext ctx, LlamaConfig cfg, Q4KMmqKernel mmq, CudaRmsNorm norm,
			PrefillWindowKernels kernels, Weights w, DeviceFloatMatrix[] headNorm) {
		int layers = headNorm.length;
		DeviceFloatMatrix[] weights = new DeviceFloatMatrix[layers];
		try {
			for (int li = 0; li < layers; li++)
				if (headNorm[li] != null && hasTail(w, li, cfg))
					weights[li] = DeviceFloatMatrix.upload(ctx, w.ffnNorm()[li], 1, cfg.hiddenDim());
		} catch (RuntimeException e) {
			for (DeviceFloatMatrix m : weights)
				if (m != null)
					m.close();
			throw e;
		}
		return new ResidentLayerTail(mmq, norm, kernels, cfg, w, weights);
	}

	private static boolean hasTail(Weights w, int li, LlamaConfig cfg) {
		int h = cfg.hiddenDim();
		int i = cfg.intermediateSize();
		return shaped(w.wo(), li, h, h) && shaped(w.wGate(), li, i, h) && shaped(w.wUp(), li, i, h)
				&& shaped(w.wDown(), li, h, i) && w.ffnNorm() != null && li < w.ffnNorm().length
				&& w.ffnNorm()[li] != null;
	}

	private static boolean shaped(DeviceQ4KMatrix[] m, int li, int rows, int cols) {
		return m != null && li < m.length && m[li] != null && m[li].rows() == rows && m[li].cols() == cols;
	}

	/** Whether layer {@code li} runs its tail on the device. */
	boolean eligible(int li) {
		return li >= 0 && li < ffnNormWeight.length && ffnNormWeight[li] != null;
	}

	/** Q8_1 scratch the tail quantizes into: wide enough for the down projection's input. */
	long q8Bytes() {
		return Q4KMmqKernel.q8Bytes(Math.max(hidden, inter));
	}

	/** Region scratch the tail writes: the output projection, gate and up side by side, and the SwiGLU row. */
	long scratchBytes() {
		return (long) (hidden + 3 * inter) * Float.BYTES;
	}

	/**
	 * Issues layer {@code li}'s tail on {@code stream}. {@code resid} holds the
	 * layer's input row and ends holding its output; {@code attn} is the attention
	 * output; {@code xn} is a free {@code 1 x hidden} activation on the same chain;
	 * {@code q8} is at least {@link #q8Bytes} and {@code scratch} at least
	 * {@link #scratchBytes}. The layer output is also written to {@code out}.
	 */
	void issue(int li, ResidentActivation resid, ResidentActivation xn, MemorySegment attn, MemorySegment out,
			MemorySegment q8, MemorySegment scratch, MemorySegment stream) {
		long hBytes = (long) hidden * Float.BYTES;
		long iBytes = (long) inter * Float.BYTES;
		MemorySegment attnProj = scratch.asSlice(0, hBytes);
		MemorySegment gateUp = scratch.asSlice(hBytes, 2 * iBytes);
		MemorySegment swiglu = scratch.asSlice(hBytes + 2 * iBytes, iBytes);

		project(wo[li], attn, attnProj, q8, stream);
		kernels.addInPlace(resid.devicePointer(), attnProj, hidden, stream);
		DeviceSpanTally.compute(DeviceComputeEvent.RESIDUAL_ADD, 1, -1L);
		if (!norm.normalizeResident(resid, ffnNormWeight[li], eps, xn))
			throw new IllegalStateException("RMS-norm kernel failed to load after the path was built");
		DeviceSpanTally.compute(DeviceComputeEvent.RMS_NORM, 1, -1L);
		mmq.quantizeX(xn.devicePointer(), q8, hidden, stream);
		packed(wGate[li], q8, gateUp, stream);
		packed(wUp[li], q8, gateUp.asSlice(iBytes), stream);
		kernels.swiglu(gateUp, swiglu, 1, inter, stream);
		DeviceSpanTally.compute(DeviceComputeEvent.SWIGLU, 1, -1L);
		project(wDown[li], swiglu, out, q8, stream);
		kernels.residualAddBoth(resid.devicePointer(), out, hidden, stream);
		DeviceSpanTally.compute(DeviceComputeEvent.RESIDUAL_ADD, 1, -1L);
	}

	private void project(DeviceQ4KMatrix w, MemorySegment x, MemorySegment y, MemorySegment q8,
			MemorySegment stream) {
		mmq.quantizeX(x, q8, w.cols(), stream);
		packed(w, q8, y, stream);
	}

	private void packed(DeviceQ4KMatrix w, MemorySegment q8, MemorySegment y, MemorySegment stream) {
		mmq.launchPacked(w.devicePointer(), q8, y, w.rows(), w.cols(), w.quantType(), stream);
		DeviceSpanTally.compute(DeviceComputeEvent.MMQ_PACKED, 1, -1L);
	}

	/** Device bytes held by the FFN norm weights. */
	long deviceBytes() {
		long total = 0;
		for (DeviceFloatMatrix w : ffnNormWeight)
			if (w != null)
				total += (long) w.rows() * w.cols() * Float.BYTES;
		return total;
	}

	@Override
	public void close() {
		for (DeviceFloatMatrix w : ffnNormWeight)
			if (w != null)
				w.close();
	}
}
