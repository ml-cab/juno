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
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * The decode-time attention entry as one residency region: the residual row is
 * uploaded once, RMS-normalized, projected to Q, K and V by the K-quant MMQ
 * kernels, Q and K are rotated, and q, k and v are downloaded together - one
 * wait, where the op-at-a-time path pays a round trip for the norm and another
 * for the projection and runs RoPE on the host between them.
 *
 * <pre>
 *   upload x -> norm -> quantize to Q8_1 -> W_q, W_k, W_v -> RoPE(q), RoPE(k) -> download q, k, v
 * </pre>
 *
 * <p>Every kernel is the one the op-at-a-time path already runs, fed the same
 * bits, so the result is bit-identical to it. What changes is where the data
 * waits: all of it is issued on one {@link ResidentChain} stream, and the host
 * waits once.
 *
 * <p>Scope. Single-sequence decode (one row) on CUDA, for a layer whose Q, K and
 * V projections are all K-quant MMQ matrices on the device, with adjacent-pair
 * RoPE and no Q/K/V bias. {@link #unsupportedReason} names what the device path
 * cannot run for a whole model, and {@link #eligible} for one layer; the caller
 * keeps its existing path there.
 *
 * <p>Concurrency. A call takes a device region (chain, activations and Q8_1
 * scratch) from a pool and returns it when done, so there are only as many
 * regions as calls ever ran at once - not one per thread that ever called, which
 * a server running each request on a new thread would grow without bound. The
 * results are copied into host arrays owned by the calling thread, so a region
 * can go back to the pool before the caller has read them. A call allocates no
 * device memory once the pool holds a free region. Device work is issued under
 * the context's serialization lock, like the matrix-vector path's.
 * {@link #close} frees every region and the norm weights.
 */
final class ResidentQkvPath implements AutoCloseable {

	private final GpuContext ctx;
	private final Q4KMmqKernel mmq;
	private final CudaRmsNorm norm;
	private final CudaRope rope;
	private final int hidden;
	private final int kvDim;
	private final float eps;
	private final DeviceQ4KMatrix[] wq;
	private final DeviceQ4KMatrix[] wk;
	private final DeviceQ4KMatrix[] wv;
	/** Per layer: the norm weight as a 1 x hidden device matrix, null where the layer is not eligible. */
	private final DeviceFloatMatrix[] normWeight;

	/** Every region ever opened, for {@link #close} and {@link #deviceBytes}. */
	private final List<Region> regions = new CopyOnWriteArrayList<>();
	/** Regions not in use by a call. */
	private final ConcurrentLinkedQueue<Region> free = new ConcurrentLinkedQueue<>();
	/** Per calling thread: {q, k, v} host arrays the results are copied into. */
	private final ThreadLocal<float[][]> results;
	private volatile boolean closed;

	private ResidentQkvPath(GpuContext ctx, Q4KMmqKernel mmq, CudaRmsNorm norm, CudaRope rope, LlamaConfig cfg,
			DeviceQ4KMatrix[] wq, DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, DeviceFloatMatrix[] normWeight) {
		this.ctx = ctx;
		this.mmq = mmq;
		this.norm = norm;
		this.rope = rope;
		this.hidden = cfg.hiddenDim();
		this.kvDim = cfg.kvDim();
		this.eps = cfg.rmsNormEps();
		this.wq = wq;
		this.wk = wk;
		this.wv = wv;
		this.normWeight = normWeight;
		int h = hidden;
		int kv = kvDim;
		this.results = ThreadLocal.withInitial(() -> new float[][] { new float[h], new float[kv], new float[kv] });
	}

	/**
	 * Why the device region cannot run this model at all, or {@code null} when it
	 * can: not a CUDA context, a split-half RoPE layout (the device RoPE kernel
	 * rotates adjacent pairs only), or Q/K/V biases (no bias-add inside the region).
	 */
	static String unsupportedReason(GpuContext ctx, RopePairing pairing, boolean qkvBias) {
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return "needs a CUDA backend (the device RoPE and MMQ kernels are CUDA-only)";
		if (pairing != RopePairing.ADJACENT)
			return "uses the split-half RoPE layout, which the device RoPE kernel does not implement";
		if (qkvBias)
			return "has Q/K/V projection biases, which the device region does not add";
		return null;
	}

	/** As {@link #unsupportedReason(GpuContext, RopePairing, boolean)}, plus the model's shape. */
	static String unsupportedReason(GpuContext ctx, RopePairing pairing, boolean qkvBias, LlamaConfig cfg) {
		String reason = unsupportedReason(ctx, pairing, qkvBias);
		if (reason != null)
			return reason;
		if (cfg.numHeads() * cfg.headDim() != cfg.hiddenDim())
			return "has a query width different from its hidden size";
		return null;
	}

	/**
	 * Builds the region for a model. Uploads each eligible layer's attention norm
	 * weight once. The caller has checked {@link #unsupportedReason}; this throws
	 * if a kernel is missing rather than returning a half-working path.
	 *
	 * @param attnNorm per-layer norm weights, indexed like {@code wq}
	 * @param wq       per-layer device Q projections; a null entry makes that layer ineligible
	 */
	static ResidentQkvPath create(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv) {
		Q4KMmqKernel mmq = Q4KMmqKernel.tryLoad();
		if (mmq == null)
			throw new IllegalStateException("Q4_K MMQ kernel is not loaded");
		if (RmsNormKernel.tryLoad() == null || RopeKernel.tryLoad() == null)
			throw new IllegalStateException("RMS-norm or RoPE kernel is not loaded");
		CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
		CudaRope rope = CudaRope.tryCreate(ctx, cfg.headDim(), cfg.ropeTheta());
		if (cfg.numHeads() * cfg.headDim() != cfg.hiddenDim())
			throw new IllegalArgumentException("query width " + cfg.numHeads() * cfg.headDim()
					+ " differs from hidden size " + cfg.hiddenDim());
		if (norm == null || rope == null) {
			if (rope != null)
				rope.close();
			throw new IllegalStateException("resident norm and RoPE need a CUDA context");
		}
		int layers = wq.length;
		DeviceFloatMatrix[] weights = new DeviceFloatMatrix[layers];
		try {
			for (int li = 0; li < layers; li++) {
				if (layerHasWeights(wq, wk, wv, li))
					weights[li] = DeviceFloatMatrix.upload(ctx, attnNorm[li], 1, cfg.hiddenDim());
			}
		} catch (RuntimeException e) {
			for (DeviceFloatMatrix w : weights)
				if (w != null)
					w.close();
			rope.close();
			throw e;
		}
		return new ResidentQkvPath(ctx, mmq, norm, rope, cfg, wq, wk, wv, weights);
	}

	private static boolean layerHasWeights(DeviceQ4KMatrix[] wq, DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv,
			int li) {
		return wq != null && wk != null && wv != null && li < wq.length && wq[li] != null && wk[li] != null
				&& wv[li] != null;
	}

	/** Whether layer {@code li} (0-based within this shard) runs on the device region. */
	boolean eligible(int li) {
		return li >= 0 && li < normWeight.length && normWeight[li] != null;
	}

	/**
	 * Runs the region for layer {@code li} on the residual row {@code x} at
	 * {@code pos}. Returns {q, k, v}, RoPE already applied to q and k, in arrays
	 * owned by the calling thread and overwritten by its next call; or
	 * {@code null} when the layer is not {@link #eligible} and the caller must use
	 * its existing path. {@code x} is not modified.
	 */
	float[][] run(int li, float[] x, int pos) {
		if (!eligible(li))
			return null;
		if (closed)
			throw new IllegalStateException("resident QKV path is closed");
		Region r = free.poll();
		if (r == null) {
			r = openRegion();
			regions.add(r);
		}
		float[][] out = results.get();
		try {
			runOn(r, li, x, pos, out);
		} finally {
			free.offer(r);
		}
		return out;
	}

	private void runOn(Region r, int li, float[] x, int pos, float[][] out) {
		MemorySegment stream = r.chain.stream();
		synchronized (ctx.cublasSerializationLock()) {
			r.in[0] = x;
			r.x.upload(r.in, 1);
			r.in[0] = null;
			if (!norm.normalizeResident(r.x, normWeight[li], eps, r.xn))
				throw new IllegalStateException("RMS-norm kernel failed to load after the path was built");
			mmq.quantizeX(r.xn.devicePointer(), r.q8, hidden, stream);
			project(wq[li], r.q, r.q8, stream);
			project(wk[li], r.k, r.q8, stream);
			project(wv[li], r.v, r.q8, stream);
			if (!rope.applyResident(r.q, pos) || !rope.applyResident(r.k, pos))
				throw new IllegalStateException("RoPE kernel failed to load after the path was built");
			ResidentActivation.materializeRows(r.acts, out);
		}
	}

	private void project(DeviceQ4KMatrix w, ResidentActivation y, MemorySegment q8, MemorySegment stream) {
		mmq.launchPacked(w.devicePointer(), q8, y.devicePointer(), w.rows(), w.cols(), w.quantType(), stream);
		y.markWritten(1);
	}

	/** Device bytes held by every thread's region plus the norm weights. */
	long deviceBytes() {
		long total = 0;
		for (Region r : regions)
			total += r.chain.deviceBytes();
		for (DeviceFloatMatrix w : normWeight)
			if (w != null)
				total += (long) w.rows() * w.cols() * Float.BYTES;
		return total;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		synchronized (ctx.cublasSerializationLock()) {
			for (Region r : regions)
				r.chain.close();
			regions.clear();
			free.clear();
			for (DeviceFloatMatrix w : normWeight)
				if (w != null)
					w.close();
			rope.close();
		}
	}

	private Region openRegion() {
		ResidentChain chain = ResidentChain.open(ctx);
		try {
			return new Region(chain);
		} catch (RuntimeException e) {
			chain.close();
			throw e;
		}
	}

	/** One call's device buffers: allocated once, reused by every later call that takes it from the pool. */
	private final class Region {
		final ResidentChain chain;
		final ResidentActivation x;
		final ResidentActivation xn;
		final ResidentActivation q;
		final ResidentActivation k;
		final ResidentActivation v;
		final MemorySegment q8;
		final ResidentActivation[] acts;
		final float[][] in = new float[1][];

		Region(ResidentChain chain) {
			this.chain = chain;
			this.x = chain.allocate(1, hidden);
			this.xn = chain.allocate(1, hidden);
			this.q = chain.allocate(1, hidden);
			this.k = chain.allocate(1, kvDim);
			this.v = chain.allocate(1, kvDim);
			this.q8 = chain.allocateScratch(Q4KMmqKernel.q8Bytes(hidden));
			this.acts = new ResidentActivation[] { q, k, v };
		}
	}
}
