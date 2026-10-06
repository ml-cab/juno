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

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * The decode-time attention block as one residency region: the residual row is
 * uploaded once, RMS-normalized, projected to Q, K and V by the K-quant MMQ
 * kernels, and Q and K are rotated. With a device KV mirror
 * ({@link DeviceKvCache}) the region then goes on: K and V are cast to FP16 into
 * the mirror at the token's position and the attention kernel reads Q and the
 * mirror there, so one download brings back K, V and the attention output.
 * Given the rest of the layer's weights ({@link ResidentLayerTail}), it goes on
 * through the output projection and the FFN, so one download brings back K, V
 * and the layer output, and the layer output stays on the device: inside a
 * {@link Lease}, the next layer reads it there and uploads nothing.
 *
 * <pre>
 *   upload x (unless the previous layer left it) -> norm -> quantize to Q8_1 -> W_q, W_k, W_v -> RoPE(q), RoPE(k)
 *       with a mirror:    -> fp16(k, v) into the mirror -> attention(q, mirror)
 *           and a tail:   -> W_o, residual, norm, SwiGLU FFN, residual -> download k, v, attention, layer output
 *           without one:  -> download k, v, attention
 *       without one:      -> download q, k, v
 * </pre>
 *
 * <p>Every kernel is the one the op-at-a-time path already runs, fed the same
 * bits, so the result is bit-identical to it: the projections to the path's
 * MMQ GEMVs, the FP16 cast to the host's round-to-nearest-even pack, and the
 * attention launch to {@link CudaGqaAttention#attendBatched} at one row. What
 * changes is where the data waits: all of it is issued on one
 * {@link ResidentChain} stream, and the host waits once.
 *
 * <p>The mirror contract is the handler's: the host KV tensors are the source of
 * truth and are written first. The region writes the row into the mirror but
 * does not move its written-prefix watermark; the caller writes the returned K
 * and V into the host tensors and then marks the row
 * ({@link DeviceKvCache#markWritten}). The region attends only through a mirror
 * that holds every earlier position ({@link DeviceKvCache#readableThrough}) and
 * already has room for this one, so it never grows a mirror, and so never runs
 * out of device memory, inside the region; the caller grows it first.
 *
 * <p>Scope. Single-sequence decode (one row) on CUDA, for a layer whose Q, K and
 * V projections are all K-quant MMQ matrices on the device, with adjacent-pair
 * RoPE and no Q/K/V bias. {@link #unsupportedReason} names what the device path
 * cannot run for a whole model, and {@link #eligible} for one layer; the caller
 * keeps its existing path there. Attention moves in when the attention kernel
 * runs this head width ({@link #attendsOnDevice}).
 *
 * <p>Concurrency. A call takes a device region (chain, activations, Q8_1 scratch
 * and the attention table) from a pool and returns it when done, so there are
 * only as many regions as calls ever ran at once - not one per thread that ever
 * called, which a server running each request on a new thread would grow without
 * bound. The results are copied into host arrays owned by the calling thread, so
 * a region can go back to the pool before the caller has read them. A call
 * allocates no device memory once the pool holds a free region. Device work is
 * issued under the context's serialization lock, like the matrix-vector path's.
 * {@link #close} frees every region and the norm weights.
 */
final class ResidentQkvPath implements AutoCloseable {

	/** The attention table a decode row's launch reads: K pointer, V pointer, sequence length. */
	private static final long TABLE_BYTES = 2 * ADDRESS.byteSize() + Integer.BYTES;

	/**
	 * One call's results, owned by the calling thread and overwritten by its next
	 * call. With {@link #attended} the region ran attention and {@link #attn} is its
	 * output ({@link #q} is not current); without it, {@link #q} is the rotated
	 * query and the caller runs attention. {@link #k} and {@link #v} are current
	 * either way.
	 */
	static final class Output {
		final float[] q;
		final float[] k;
		final float[] v;
		final float[] attn;
		/** The layer's output row (the residual stream after the FFN) when {@link #layerDone}. */
		final float[] layer;
		boolean attended;
		/**
		 * The region ran the whole layer: output projection, both residual adds, the
		 * FFN norm and the SwiGLU FFN as well. {@link #layer} is the layer's output.
		 */
		boolean layerDone;
		/** {q, k, v}, the arrays above, for the call without a mirror. */
		final float[][] qkv;
		/** The packed row the region downloads: k, v, then the attention output. */
		final float[][] packed;
		final float[][] qAndPacked;

		private Output(int hidden, int kvDim, int packedWidth) {
			q = new float[hidden];
			k = new float[kvDim];
			v = new float[kvDim];
			attn = new float[hidden];
			layer = new float[hidden];
			qkv = new float[][] { q, k, v };
			packed = new float[][] { new float[packedWidth] };
			qAndPacked = new float[][] { q, packed[0] };
		}
	}

	private final GpuContext ctx;
	private final Q4KMmqKernel mmq;
	private final CudaRmsNorm norm;
	private final CudaRope rope;
	/** Null when attention stays outside the region (kernel missing or head width it does not run). */
	private final GqaAttentionKernel attention;
	private final PrefillWindowKernels kernels;
	private final int hidden;
	private final int kvDim;
	private final int numHeads;
	private final int headDim;
	private final int gqaRatio;
	private final float eps;
	private final DeviceQ4KMatrix[] wq;
	private final DeviceQ4KMatrix[] wk;
	private final DeviceQ4KMatrix[] wv;
	/** Per layer: the norm weight as a 1 x hidden device matrix, null where the layer is not eligible. */
	private final DeviceFloatMatrix[] normWeight;
	/** The rest of the layer after attention, or null when the region stops at attention. */
	private final ResidentLayerTail tail;

	/** Every region ever opened, for {@link #close} and {@link #deviceBytes}. */
	private final List<Region> regions = new CopyOnWriteArrayList<>();
	/** Regions not in use by a call. */
	private final ConcurrentLinkedQueue<Region> free = new ConcurrentLinkedQueue<>();
	/** Per calling thread: the host arrays the results are copied into. */
	private final ThreadLocal<Output> results;
	private volatile boolean closed;

	private ResidentQkvPath(GpuContext ctx, Q4KMmqKernel mmq, CudaRmsNorm norm, CudaRope rope,
			GqaAttentionKernel attention, PrefillWindowKernels kernels, LlamaConfig cfg, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, DeviceFloatMatrix[] normWeight, ResidentLayerTail tail) {
		this.ctx = ctx;
		this.mmq = mmq;
		this.norm = norm;
		this.rope = rope;
		this.attention = attention;
		this.kernels = kernels;
		this.hidden = cfg.hiddenDim();
		this.kvDim = cfg.kvDim();
		this.numHeads = cfg.numHeads();
		this.headDim = cfg.headDim();
		this.gqaRatio = cfg.gqaRatio();
		this.eps = cfg.rmsNormEps();
		this.wq = wq;
		this.wk = wk;
		this.wv = wv;
		this.normWeight = normWeight;
		this.tail = tail;
		int h = hidden;
		int kv = kvDim;
		int width = packedWidth();
		this.results = ThreadLocal.withInitial(() -> new Output(h, kv, width));
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
	 * if a kernel the region cannot run without is missing rather than returning a
	 * half-working path. The attention kernel is optional: without it (or for a
	 * head width it does not run) attention stays outside the region.
	 *
	 * @param attnNorm per-layer norm weights, indexed like {@code wq}
	 * @param wq       per-layer device Q projections; a null entry makes that layer ineligible
	 */
	static ResidentQkvPath create(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv) {
		return create(ctx, cfg, attnNorm, wq, wk, wv, null);
	}

	/**
	 * As {@link #create(GpuContext, LlamaConfig, float[][], DeviceQ4KMatrix[], DeviceQ4KMatrix[], DeviceQ4KMatrix[])},
	 * and when {@code tail} is given and the region attends on the device, the
	 * region runs the rest of each layer whose tail weights are all on the device
	 * ({@link ResidentLayerTail.Weights}): one decode layer is then one upload of the
	 * residual row (none when the previous layer ran in the same {@link Lease}) and
	 * one download of k, v and the layer output.
	 */
	static ResidentQkvPath create(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, ResidentLayerTail.Weights tailWeights) {
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
		GqaAttentionKernel attention = GqaAttentionKernel.supportsHeadDim(cfg.headDim())
				? GqaAttentionKernel.tryLoad()
				: null;
		PrefillWindowKernels kernels = attention != null ? PrefillWindowKernels.tryLoad() : null;
		if (kernels == null)
			attention = null;
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
		ResidentLayerTail tail = null;
		if (tailWeights != null && attention != null) {
			try {
				tail = ResidentLayerTail.create(ctx, cfg, mmq, norm, kernels, tailWeights, weights);
			} catch (RuntimeException e) {
				for (DeviceFloatMatrix w : weights)
					if (w != null)
						w.close();
				rope.close();
				throw e;
			}
		}
		return new ResidentQkvPath(ctx, mmq, norm, rope, attention, kernels, cfg, wq, wk, wv, weights, tail);
	}

	/** The packed download row: k, v and the attention output, then the layer output when the tail runs. */
	private int packedWidth() {
		return 2 * kvDim + (tail != null ? 2 : 1) * hidden;
	}

	/**
	 * Whether the region runs the whole of layer {@code li} when it attends: the
	 * layer is {@link #eligible} and its output projection, FFN norm and FFN
	 * weights are on the device.
	 */
	boolean runsWholeLayer(int li) {
		return tail != null && eligible(li) && tail.eligible(li);
	}

	/**
	 * Holds one device region across a token's layer loop, so that a layer the
	 * region runs whole leaves its output on the device for the next layer and the
	 * next layer skips its upload. Used by one thread at a time; closing it returns
	 * the region to the pool.
	 */
	final class Lease implements AutoCloseable {
		private Region region;
		/** The layer and position whose input is already on the device, or -1. */
		private int nextLayer = -1;
		private int nextPos = -1;

		private Lease() {
		}

		@Override
		public void close() {
			if (region != null) {
				free.offer(region);
				region = null;
			}
			nextLayer = -1;
		}
	}

	/**
	 * Whether {@code row} is the calling thread's {@link Output#layer}, which its
	 * next call overwrites; a caller keeping a layer output past that copies it.
	 */
	boolean ownsResult(float[] row) {
		return row == results.get().layer;
	}

	/** A lease over one device region for a token's layer loop; see {@link Lease}. */
	Lease lease() {
		return new Lease();
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

	/** Whether the region runs the KV append and attention when it is given a mirror. */
	boolean attendsOnDevice() {
		return attention != null;
	}

	/**
	 * Runs the region for layer {@code li} on the residual row {@code x} at
	 * {@code pos} without attention. Returns {q, k, v}, RoPE already applied to q
	 * and k, in arrays owned by the calling thread and overwritten by its next
	 * call; or {@code null} when the layer is not {@link #eligible} and the caller
	 * must use its existing path. {@code x} is not modified.
	 */
	float[][] run(int li, float[] x, int pos) {
		Output out = run(li, x, pos, null);
		return out == null ? null : out.qkv;
	}

	/**
	 * Runs the region for layer {@code li} on the residual row {@code x} at
	 * {@code pos}, attending inside it when {@code mirror} allows: the region
	 * {@link #attendsOnDevice}, and the mirror is live, holds positions
	 * {@code [0, pos)} and has room for {@code pos}. Otherwise (including a null
	 * mirror) it returns q, k and v as {@link #run(int, float[], int)} does and
	 * leaves the mirror untouched. After an attended call the caller writes
	 * {@link Output#k} and {@link Output#v} to its host KV tensors and then marks
	 * the row in the mirror. Returns {@code null} when the layer is not
	 * {@link #eligible}. {@code x} is not modified.
	 */
	Output run(int li, float[] x, int pos, DeviceKvCache mirror) {
		return run(li, x, pos, mirror, null);
	}

	/**
	 * As {@link #run(int, float[], int, DeviceKvCache)}, inside {@code lease} when it
	 * is not null. When the call attends and {@link #runsWholeLayer} holds, the region
	 * runs the whole layer and {@link Output#layer} is its output; the caller passes
	 * that row as {@code x} to the next layer's call in the same lease, which then
	 * reads it from the device instead of uploading it.
	 */
	Output run(int li, float[] x, int pos, DeviceKvCache mirror, Lease lease) {
		if (!eligible(li))
			return null;
		if (closed)
			throw new IllegalStateException("resident QKV path is closed");
		boolean attend = attention != null && mirror != null && mirror.readableThrough(pos)
				&& mirror.capacityTokens() > pos;
		Region r = lease != null ? lease.region : null;
		if (r == null)
			r = takeRegion();
		Output out = results.get();
		if (lease != null) {
			lease.region = r;
			runOn(r, li, x, pos, attend ? mirror : null, out, lease);
			return out;
		}
		try {
			runOn(r, li, x, pos, attend ? mirror : null, out, null);
		} finally {
			free.offer(r);
		}
		return out;
	}

	private Region takeRegion() {
		Region r = free.poll();
		if (r == null) {
			r = openRegion();
			regions.add(r);
		}
		return r;
	}

	private void runOn(Region r, int li, float[] x, int pos, DeviceKvCache mirror, Output out, Lease lease) {
		boolean whole = mirror != null && tail != null && tail.eligible(li);
		boolean inputOnDevice = lease != null && lease.nextLayer == li && lease.nextPos == pos;
		if (lease != null)
			lease.nextLayer = -1;
		MemorySegment stream = r.chain.stream();
		MemorySegment packed = r.packed.devicePointer();
		long kvBytes = (long) kvDim * Float.BYTES;
		synchronized (ctx.cublasSerializationLock()) {
			// The previous layer in this lease left its output, which is x, in r.x.
			if (!inputOnDevice) {
				r.in[0] = x;
				r.x.upload(r.in, 1);
				r.in[0] = null;
			}
			if (!norm.normalizeResident(r.x, normWeight[li], eps, r.xn))
				throw new IllegalStateException("RMS-norm kernel failed to load after the path was built");
			mmq.quantizeX(r.xn.devicePointer(), r.q8, hidden, stream);
			project(wq[li], r.q.devicePointer(), r.q8, stream);
			project(wk[li], packed, r.q8, stream);
			project(wv[li], packed.asSlice(kvBytes), r.q8, stream);
			r.q.markWritten(1);
			r.packed.markWritten(1);
			if (!rope.applyResident(r.q, pos) || !rope.applyResidentColumns(r.packed, 0, kvDim, pos))
				throw new IllegalStateException("RoPE kernel failed to load after the path was built");
			if (mirror != null) {
				attend(r, mirror, pos, packed, kvBytes, stream);
				if (whole)
					tail.issue(li, r.x, r.xn, packed.asSlice(2 * kvBytes), packed.asSlice(2 * kvBytes + hiddenBytes()),
							r.q8, r.tailScratch, stream);
				r.packed.materialize(out.packed);
			} else {
				ResidentActivation.materializeRows(r.qAndPacked, out.qAndPacked);
			}
		}
		float[] row = out.packed[0];
		System.arraycopy(row, 0, out.k, 0, kvDim);
		System.arraycopy(row, kvDim, out.v, 0, kvDim);
		if (mirror != null)
			System.arraycopy(row, 2 * kvDim, out.attn, 0, hidden);
		if (whole)
			System.arraycopy(row, 2 * kvDim + hidden, out.layer, 0, hidden);
		out.attended = mirror != null;
		out.layerDone = whole;
		if (whole && lease != null) {
			lease.nextLayer = li + 1;
			lease.nextPos = pos;
		}
	}

	private long hiddenBytes() {
		return (long) hidden * Float.BYTES;
	}

	/**
	 * fp16(k, v) into the mirror at {@code pos}, then attention over positions
	 * {@code [0, pos]} into the packed row's attention columns. The attention table
	 * (K pointer, V pointer, length) is written by a kernel from its launch
	 * arguments, so it costs no host-to-device copy.
	 */
	private void attend(Region r, DeviceKvCache mirror, int pos, MemorySegment packed, long kvBytes,
			MemorySegment stream) {
		DeviceSpanTimer spans = r.chain.spans();
		int mark = spans.begin(stream, 1);
		mirror.writeWindowOnDevice(pos, 1, packed, packed.asSlice(kvBytes), kernels, stream);
		spans.compute(DeviceComputeEvent.KV_APPEND, 1, mark, stream);
		kernels.decodeAttentionTable(r.table, mirror.kPointer(), mirror.vPointer(), pos + 1, stream);
		mark = spans.begin(stream, 1);
		attention.launch(r.q.devicePointer(), r.table, r.table.asSlice(ADDRESS.byteSize()),
				r.table.asSlice(2 * ADDRESS.byteSize()), packed.asSlice(2 * kvBytes), 1, numHeads, gqaRatio,
				headDim, kvDim, GqaAttentionKernel.rowsPerBlock(true, 1), 0, stream);
		spans.compute(DeviceComputeEvent.GQA_ATTENTION_REGION, 1, mark, stream);
	}

	private void project(DeviceQ4KMatrix w, MemorySegment y, MemorySegment q8, MemorySegment stream) {
		mmq.launchPacked(w.devicePointer(), q8, y, w.rows(), w.cols(), w.quantType(), stream);
	}

	/** Device bytes held by every thread's region plus the norm weights. */
	long deviceBytes() {
		long total = 0;
		for (Region r : regions)
			total += r.chain.deviceBytes();
		for (DeviceFloatMatrix w : normWeight)
			if (w != null)
				total += (long) w.rows() * w.cols() * Float.BYTES;
		if (tail != null)
			total += tail.deviceBytes();
		return total;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		synchronized (ctx.cublasSerializationLock()) {
			for (Region r : regions)
				r.close();
			regions.clear();
			free.clear();
			for (DeviceFloatMatrix w : normWeight)
				if (w != null)
					w.close();
			if (tail != null)
				tail.close();
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
		/** One row of k, v and the attention output, so an attended call leaves the device in one copy. */
		final ResidentActivation packed;
		final MemorySegment q8;
		/** The tail's scratch ({@link ResidentLayerTail#scratchBytes}), null without a tail. */
		final MemorySegment tailScratch;
		final MemorySegment table;
		final ResidentActivation[] qAndPacked;
		final float[][] in = new float[1][];

		Region(ResidentChain chain) {
			this.chain = chain;
			this.x = chain.allocate(1, hidden, "decode region input");
			this.xn = chain.allocate(1, hidden);
			this.q = chain.allocate(1, hidden, "decode region q");
			this.packed = chain.allocate(1, packedWidth(),
					tail != null ? "decode region k, v, attention, layer output" : "decode region k, v, attention");
			this.q8 = chain.allocateScratch(tail != null ? tail.q8Bytes() : Q4KMmqKernel.q8Bytes(hidden));
			this.tailScratch = tail != null ? chain.allocateScratch(tail.scratchBytes()) : null;
			this.table = chain.allocateScratch(TABLE_BYTES);
			this.qAndPacked = new ResidentActivation[] { q, packed };
		}

		void close() {
			chain.close();
		}
	}
}
