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

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;

/**
 * Handler-facing entry point for GPU rotary position embeddings on a
 * device-resident activation ({@code rope.cu} / {@link RopeKernel}).
 *
 * <p>Holds the model's inverse-frequency table on the device - uploaded once,
 * here, rather than per call - and rotates a {@link ResidentActivation} in place
 * on its chain's stream: no host transfer, no synchronization. Row {@code r} of
 * the activation is rotated as position {@code startPos + r}, which covers one
 * decode row and a prefill window of consecutive positions.
 *
 * <p>Same math as {@link LlamaTransformerHandler#rope} with the model's
 * {@link RopePairing}: adjacent pairs, or the split-half pairs of the Qwen2 family.
 * There is deliberately no host-array entry point: moving RoPE to the device only
 * pays when the activation is already there.
 *
 * <p>Used by the decode residency region ({@link ResidentQkvPath}, adjacent pairs
 * only) and the prefill-window device region ({@link PrefillWindowRegion}).
 */
final class CudaRope implements AutoCloseable {

	private final GpuContext ctx;
	private final int headDim;
	private final float ropeTheta;
	private final RopePairing pairing;
	private final MemorySegment invFreq; // device double[headDim / 2]
	private boolean closed;

	private CudaRope(GpuContext ctx, int headDim, float ropeTheta, RopePairing pairing, MemorySegment invFreq) {
		this.ctx = ctx;
		this.headDim = headDim;
		this.ropeTheta = ropeTheta;
		this.pairing = pairing;
		this.invFreq = invFreq;
	}

	/**
	 * Uploads the inverse-frequency table for {@code headDim} and
	 * {@code ropeTheta}. Returns {@code null} when the backend is not CUDA, the
	 * same contract as {@link CudaRmsNorm#tryCreate}.
	 */
	static CudaRope tryCreate(GpuContext ctx, int headDim, float ropeTheta) {
		return tryCreate(ctx, headDim, ropeTheta, RopePairing.ADJACENT);
	}

	/** As {@link #tryCreate(GpuContext, int, float)}, rotating the pairs {@code pairing} names. */
	static CudaRope tryCreate(GpuContext ctx, int headDim, float ropeTheta, RopePairing pairing) {
		java.util.Objects.requireNonNull(pairing, "pairing");
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return null;
		double[] table = RopeKernel.inverseFrequencies(headDim, ropeTheta);
		GpuBindings gpu = ctx.bindings();
		long tableBytes = (long) table.length * Double.BYTES;
		MemorySegment device = gpu.deviceMalloc(ctx.deviceIndex(), tableBytes);
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment host = staging.allocate(tableBytes, Double.BYTES);
			MemorySegment.copy(table, 0, host, JAVA_DOUBLE, 0, table.length);
			DeviceStaging.copy(gpu, device, host, tableBytes, GpuBindings.H2D, 0, "memcpy(rope inverse frequencies H2D)");
		} catch (RuntimeException e) {
			gpu.deviceFree(device);
			throw e;
		}
		return new CudaRope(ctx, headDim, ropeTheta, pairing, device);
	}

	/**
	 * Rotates every valid row of {@code x} in place, row {@code r} at position
	 * {@code startPos + r}, treating each row as {@code x.dim() / headDim} heads.
	 * Asynchronous on {@code x}'s chain.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load
	 */
	boolean applyResident(ResidentActivation x, int startPos) {
		requireOpen();
		x.requireOpen();
		if (x.dim() % headDim != 0)
			throw new IllegalArgumentException(
					"activation width " + x.dim() + " is not a whole number of " + headDim + "-wide heads");
		if (startPos < 0)
			throw new IllegalArgumentException("startPos must not be negative: " + startPos);
		if (x.chain().context().deviceIndex() != ctx.deviceIndex())
			throw new IllegalArgumentException("activation is on device " + x.chain().context().deviceIndex()
					+ ", the RoPE table on device " + ctx.deviceIndex());
		if (x.rows() == 0)
			throw new IllegalStateException("activation holds no rows to rotate");
		RopeKernel kernel = RopeKernel.tryLoad();
		if (kernel == null)
			return false;
		if (pairing == RopePairing.SPLIT_HALF)
			kernel.launchSplitHalf(x.devicePointer(), invFreq, x.rows(), x.dim() / headDim, headDim, startPos,
					x.chain().stream());
		else
			kernel.launch(x.devicePointer(), invFreq, x.rows(), x.dim() / headDim, headDim, startPos,
					x.chain().stream());
		return true;
	}

	/**
	 * Rotates columns {@code [from, from + width)} of the one valid row of {@code x}
	 * in place at {@code pos}, treating them as {@code width / headDim} heads: for an
	 * activation that packs several results into one row, so they leave the device
	 * in one copy. Asynchronous on {@code x}'s chain.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load
	 */
	boolean applyResidentColumns(ResidentActivation x, int from, int width, int pos) {
		requireOpen();
		x.requireOpen();
		if (x.rows() != 1)
			throw new IllegalStateException("column rotation needs one row, the activation holds " + x.rows());
		if (from < 0 || width <= 0 || from + width > x.dim() || width % headDim != 0)
			throw new IllegalArgumentException("columns [" + from + ", " + (from + width) + ") of a " + x.dim()
					+ "-wide row are not whole " + headDim + "-wide heads inside it");
		if (pos < 0)
			throw new IllegalArgumentException("pos must not be negative: " + pos);
		RopeKernel kernel = RopeKernel.tryLoad();
		if (kernel == null)
			return false;
		MemorySegment cols = x.devicePointer().asSlice((long) from * Float.BYTES, (long) width * Float.BYTES);
		if (pairing == RopePairing.SPLIT_HALF)
			kernel.launchSplitHalf(cols, invFreq, 1, width / headDim, headDim, pos, x.chain().stream());
		else
			kernel.launch(cols, invFreq, 1, width / headDim, headDim, pos, x.chain().stream());
		return true;
	}

	RopePairing pairing() {
		return pairing;
	}

	int headDim() {
		return headDim;
	}

	float ropeTheta() {
		return ropeTheta;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		ctx.bindings().deviceFree(invFreq);
	}

	private void requireOpen() {
		if (closed)
			throw new IllegalStateException("CudaRope is closed");
	}
}
