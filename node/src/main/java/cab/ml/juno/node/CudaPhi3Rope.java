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
 * The Phi-3 family's rotary position embedding ({@link Phi3Rope#ropeExt}) on a
 * device-resident activation: split-half pairs, per-pair frequency factors and a
 * magnitude scale ({@code rope.scaling.attn_factor}, about 1.19 on Phi-3.5-mini).
 *
 * <p>The scale and the CPU path's float-built angle are why this is not
 * {@link CudaRope} with a folded inverse-frequency table: a table changes the
 * angle's bits and cannot scale cos and sin. The kernel ({@code rope_ext_split_half}
 * in {@code rope.cu}) repeats the CPU arithmetic step for step instead, so the two
 * agree bit for bit. The factor set is the one {@link Phi3RopeConfig#selectFactors}
 * picks, uploaded once; positions that would need the held-back long factors are
 * refused here as on the CPU ({@link Phi3RopeConfig#requirePosition}).
 */
final class CudaPhi3Rope implements ResidentRope {

	private final GpuContext ctx;
	private final Phi3RopeConfig cfg;
	private final int headDim;
	private final float thetaScale;
	/** Device float[headDim / 2], or null for a model without factors. */
	private final MemorySegment factors;
	private boolean closed;

	private CudaPhi3Rope(GpuContext ctx, Phi3RopeConfig cfg, int headDim, MemorySegment factors) {
		this.ctx = ctx;
		this.cfg = cfg;
		this.headDim = headDim;
		this.thetaScale = (float) Math.pow(cfg.freqBase(), -2.0 / headDim);
		this.factors = factors;
	}

	/**
	 * Uploads the selected frequency factors for {@code headDim}. Returns
	 * {@code null} when the backend is not CUDA.
	 */
	static CudaPhi3Rope tryCreate(GpuContext ctx, int headDim, Phi3RopeConfig cfg) {
		java.util.Objects.requireNonNull(cfg, "cfg");
		if (headDim <= 0 || (headDim & 1) != 0)
			throw new IllegalArgumentException("headDim must be positive and even: " + headDim);
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return null;
		float[] selected = cfg.selectFactors();
		if (selected == null)
			return new CudaPhi3Rope(ctx, cfg, headDim, null);
		// The CPU path reads factor i for pair i and 1 past the table's end.
		float[] table = new float[headDim / 2];
		for (int i = 0; i < table.length; i++)
			table[i] = i < selected.length ? selected[i] : 1.0f;
		GpuBindings gpu = ctx.bindings();
		long bytes = (long) table.length * Float.BYTES;
		MemorySegment device = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment host = staging.allocate(bytes, Float.BYTES);
			MemorySegment.copy(table, 0, host, JAVA_FLOAT, 0, table.length);
			DeviceStaging.copy(gpu, device, host, bytes, GpuBindings.H2D, 0, "memcpy(rope frequency factors H2D)");
		} catch (RuntimeException e) {
			gpu.deviceFree(device);
			throw e;
		}
		return new CudaPhi3Rope(ctx, cfg, headDim, device);
	}

	@Override
	public boolean applyResident(ResidentActivation x, int startPos) {
		requireOpen();
		x.requireOpen();
		if (x.dim() % headDim != 0)
			throw new IllegalArgumentException(
					"activation width " + x.dim() + " is not a whole number of " + headDim + "-wide heads");
		if (x.rows() == 0)
			throw new IllegalStateException("activation holds no rows to rotate");
		cfg.requirePosition(startPos + x.rows() - 1);
		RopeKernel kernel = RopeKernel.tryLoad();
		if (kernel == null)
			return false;
		kernel.launchExtSplitHalf(x.devicePointer(), factors, x.rows(), x.dim() / headDim, headDim, startPos,
				thetaScale, cfg.freqScale(), cfg.attnFactor(), x.chain().stream());
		return true;
	}

	@Override
	public boolean applyResidentColumns(ResidentActivation x, int from, int width, int pos) {
		requireOpen();
		x.requireOpen();
		if (x.rows() != 1)
			throw new IllegalStateException("column rotation needs one row, the activation holds " + x.rows());
		if (from < 0 || width <= 0 || from + width > x.dim() || width % headDim != 0)
			throw new IllegalArgumentException("columns [" + from + ", " + (from + width) + ") of a " + x.dim()
					+ "-wide row are not whole " + headDim + "-wide heads inside it");
		cfg.requirePosition(pos);
		RopeKernel kernel = RopeKernel.tryLoad();
		if (kernel == null)
			return false;
		MemorySegment cols = x.devicePointer().asSlice((long) from * Float.BYTES, (long) width * Float.BYTES);
		kernel.launchExtSplitHalf(cols, factors, 1, width / headDim, headDim, pos, thetaScale, cfg.freqScale(),
				cfg.attnFactor(), x.chain().stream());
		return true;
	}

	@Override
	public int headDim() {
		return headDim;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		if (factors != null)
			ctx.bindings().deviceFree(factors);
	}

	private void requireOpen() {
		if (closed)
			throw new IllegalStateException("CudaPhi3Rope is closed");
	}
}
