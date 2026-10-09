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
 * V projections are all K-quant MMQ matrices on the device (row views of one fused
 * tensor will do, {@link DeviceQ4KMatrix#rowSlice}), with no Q/K/V bias. The
 * rotation is the caller's {@link ResidentRope}: adjacent pairs by default, or the
 * Phi-3 family's extended rotation. {@link #unsupportedReason} names what the
 * device path cannot run for a whole model, and {@link #eligible} for one layer;
 * the caller keeps its existing path there. Attention moves in when the attention
 * kernel runs this head width ({@link #attendsOnDevice}).
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
	private final ResidentRope rope;
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
	/** Per layer: the per-head q and k norm weights as 1 x headDim device matrices, or null without head norms. */
	private final DeviceFloatMatrix[] qNormWeight;
	private final DeviceFloatMatrix[] kNormWeight;
	/** The rest of the layer after attention, or null when the region stops at attention. */
	private final ResidentLayerTail tail;
	/**
	 * Per layer: the attention window ({@link SlidingWindow}), 0 for none. Set once
	 * by {@link #withWindows} before the handler that opened the path is used.
	 */
	private int[] attentionWindows;

	/** Every region ever opened, for {@link #close} and {@link #deviceBytes}. */
	private final List<Region> regions = new CopyOnWriteArrayList<>();
	/** Regions not in use by a call. */
	private final ConcurrentLinkedQueue<Region> free = new ConcurrentLinkedQueue<>();
	/** Per calling thread: the host arrays the results are copied into. */
	private final ThreadLocal<Output> results;
	private volatile boolean closed;
	/**
	 * Set once opening a region ran out of device memory: no new region is opened
	 * after that (the pool's regions are still used), so a full card costs one failed
	 * allocation, not one per layer and token.
	 */
	private volatile boolean deviceFull;

	private ResidentQkvPath(GpuContext ctx, Q4KMmqKernel mmq, CudaRmsNorm norm, ResidentRope rope,
			GqaAttentionKernel attention, PrefillWindowKernels kernels, LlamaConfig cfg, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, DeviceFloatMatrix[] normWeight, DeviceFloatMatrix[] qNormWeight,
			DeviceFloatMatrix[] kNormWeight, ResidentLayerTail tail) {
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
		this.qNormWeight = qNormWeight;
		this.kNormWeight = kNormWeight;
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
		return reason != null ? reason : unsupportedReason(ctx, cfg);
	}

	/**
	 * Why the device region cannot run a model of this shape, for a handler that
	 * gives the region its own rotation ({@link ResidentRope}) and has no Q/K/V
	 * biases: not a CUDA context, or a query width different from the hidden size.
	 */
	static String unsupportedReason(GpuContext ctx, LlamaConfig cfg) {
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return "needs a CUDA backend (the device RoPE and MMQ kernels are CUDA-only)";
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
		CudaRope rope = CudaRope.tryCreate(ctx, cfg.headDim(), cfg.ropeTheta());
		if (rope == null)
			throw new IllegalStateException("resident norm and RoPE need a CUDA context");
		return create(ctx, cfg, attnNorm, wq, wk, wv, tailWeights, rope);
	}

	/**
	 * As {@link #create(GpuContext, LlamaConfig, float[][], DeviceQ4KMatrix[], DeviceQ4KMatrix[], DeviceQ4KMatrix[], ResidentLayerTail.Weights)},
	 * rotating q and k with {@code rope} (adjacent or split-half pairs from an
	 * inverse-frequency table, or the Phi-3 family's extended rotation). The path
	 * owns {@code rope} from here on and closes it, including when this throws.
	 */
	static ResidentQkvPath create(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, ResidentLayerTail.Weights tailWeights, ResidentRope rope) {
		return create(ctx, cfg, attnNorm, wq, wk, wv, tailWeights, rope, null);
	}

	/**
	 * Per-layer RMS-norm weights applied to each head of q and of k after the
	 * projection and before RoPE ({@code headDim} floats each; Qwen3). A layer
	 * without both is not eligible.
	 */
	record HeadNorms(float[][] q, float[][] k) {
		boolean covers(int li, int headDim) {
			return q != null && k != null && li < q.length && li < k.length && q[li] != null && k[li] != null
					&& q[li].length == headDim && k[li].length == headDim;
		}
	}

	/**
	 * As {@link #create(GpuContext, LlamaConfig, float[][], DeviceQ4KMatrix[], DeviceQ4KMatrix[], DeviceQ4KMatrix[], ResidentLayerTail.Weights, ResidentRope)},
	 * normalizing each head of q and k with {@code headNorms} before the rotation.
	 */
	static ResidentQkvPath create(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, ResidentLayerTail.Weights tailWeights, ResidentRope rope,
			HeadNorms headNorms) {
		java.util.Objects.requireNonNull(rope, "rope");
		try {
			return build(ctx, cfg, attnNorm, wq, wk, wv, tailWeights, rope, headNorms);
		} catch (RuntimeException e) {
			rope.close();
			throw e;
		}
	}

	private static ResidentQkvPath build(GpuContext ctx, LlamaConfig cfg, float[][] attnNorm, DeviceQ4KMatrix[] wq,
			DeviceQ4KMatrix[] wk, DeviceQ4KMatrix[] wv, ResidentLayerTail.Weights tailWeights, ResidentRope rope,
			HeadNorms headNorms) {
		Q4KMmqKernel mmq = Q4KMmqKernel.tryLoad();
		if (mmq == null)
			throw new IllegalStateException("Q4_K MMQ kernel is not loaded");
		if (RmsNormKernel.tryLoad() == null || RopeKernel.tryLoad() == null)
			throw new IllegalStateException("RMS-norm or RoPE kernel is not loaded");
		if (rope.headDim() != cfg.headDim())
			throw new IllegalArgumentException("RoPE head width " + rope.headDim() + " differs from the model's "
					+ cfg.headDim());
		CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
		if (cfg.numHeads() * cfg.headDim() != cfg.hiddenDim())
			throw new IllegalArgumentException("query width " + cfg.numHeads() * cfg.headDim()
					+ " differs from hidden size " + cfg.hiddenDim());
		if (norm == null)
			throw new IllegalStateException("resident norm and RoPE need a CUDA context");
		GqaAttentionKernel attention = GqaAttentionKernel.supportsHeadDim(cfg.headDim())
				? GqaAttentionKernel.tryLoad()
				: null;
		PrefillWindowKernels kernels = attention != null ? PrefillWindowKernels.tryLoad() : null;
		if (kernels == null)
			attention = null;
		int layers = wq.length;
		DeviceFloatMatrix[] weights = new DeviceFloatMatrix[layers];
		DeviceFloatMatrix[] qNormW = headNorms != null ? new DeviceFloatMatrix[layers] : null;
		DeviceFloatMatrix[] kNormW = headNorms != null ? new DeviceFloatMatrix[layers] : null;
		try {
			for (int li = 0; li < layers; li++) {
				if (!layerHasWeights(wq, wk, wv, li) || (headNorms != null && !headNorms.covers(li, cfg.headDim())))
					continue;
				weights[li] = DeviceFloatMatrix.upload(ctx, attnNorm[li], 1, cfg.hiddenDim());
				if (headNorms != null) {
					qNormW[li] = DeviceFloatMatrix.upload(ctx, headNorms.q()[li], 1, cfg.headDim());
					kNormW[li] = DeviceFloatMatrix.upload(ctx, headNorms.k()[li], 1, cfg.headDim());
				}
			}
		} catch (RuntimeException e) {
			closeAll(weights);
			closeAll(qNormW);
			closeAll(kNormW);
			throw e;
		}
		ResidentLayerTail tail = null;
		if (tailWeights != null && attention != null) {
			try {
				tail = ResidentLayerTail.create(ctx, cfg, mmq, norm, kernels, tailWeights, weights);
			} catch (RuntimeException e) {
				closeAll(weights);
				closeAll(qNormW);
				closeAll(kNormW);
				throw e;
			}
		}
		return new ResidentQkvPath(ctx, mmq, norm, rope, attention, kernels, cfg, wq, wk, wv, weights, qNormW, kNormW,
				tail);
	}

	private static void closeAll(DeviceFloatMatrix[] matrices) {
		if (matrices != null)
			for (DeviceFloatMatrix m : matrices)
				if (m != null)
					m.close();
	}

	/**
	 * Logs what {@code path} runs on a shard of {@code layers} layers and announces,
	 * once, what it leaves outside: layers that leave the region after attention,
	 * and attention itself when {@code gpuAttention} (the handler's GPU attention) is
	 * off or the kernel does not run this head width. Returns {@code path}, or closes
	 * it and returns {@code null} when no layer is eligible. Shared by every handler
	 * that builds a region, so the notices read the same whichever runs.
	 */
	static ResidentQkvPath activate(java.util.logging.Logger log, ResidentQkvPath path, int layers,
			boolean gpuAttention) {
		int eligible = 0;
		for (int li = 0; li < layers; li++)
			if (path.eligible(li))
				eligible++;
		if (eligible == 0) {
			path.close();
			GpuResidencyOptions.announceUnsupported(log, "this shard",
					"has no layer whose Q/K/V projections are all K-quant MMQ matrices on the device");
			return null;
		}
		boolean attends = gpuAttention && path.attendsOnDevice();
		int whole = 0;
		if (attends)
			for (int li = 0; li < layers; li++)
				if (path.runsWholeLayer(li))
					whole++;
		if (attends && whole < eligible)
			GpuResidencyOptions.announceUnsupported(log, (eligible - whole) + " of " + eligible + " region layers",
					"leave the region after attention: their output projection, FFN norm or FFN weights are not"
							+ " K-quant MMQ matrices on the device");
		if (!gpuAttention)
			GpuResidencyOptions.announceUnsupported(log, "the KV append and attention",
					"stay outside the region: --gpu-attention is off, so attention runs on the CPU");
		else if (!attends)
			GpuResidencyOptions.announceUnsupported(log, "the KV append and attention",
					"stay outside the region: the attention kernel does not run " + path.headDim + "-wide heads");
		log.info("GPU-resident decode region active (gpu-residency=" + GpuResidencyOptions.fromEnv().policyLabel()
				+ ") on " + eligible + " of " + layers + " layers: "
				+ (whole > 0
						? "the whole layer (norm, Q/K/V projection, RoPE, the KV append, attention, output projection,"
								+ " residual adds, FFN norm and SwiGLU FFN) on " + whole + " of them, the residual row"
								+ " staying on the device between such layers and k, v and the layer output downloaded"
								+ " in one copy"
						: attends
								? "norm, Q/K/V projection, RoPE, the KV append and attention, with the residual row"
										+ " uploaded and k, v and the attention output downloaded in one copy"
								: "norm, Q/K/V projection and RoPE with one upload and one download")
				+ " per layer. Single-sequence decode only; prefill windows and batched decode"
				+ " (--parallel above 1, continuous schedule) keep the existing path.");
		return path;
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

	/**
	 * Gives each layer, indexed like {@code wq}, its attention window
	 * ({@link SlidingWindow#forShard}); returns this. Call once, right after
	 * building the path and before it runs.
	 */
	ResidentQkvPath withWindows(int[] layerWindows) {
		if (layerWindows.length != wq.length)
			throw new IllegalArgumentException("windows for " + layerWindows.length + " layers, path has " + wq.length);
		this.attentionWindows = layerWindows.clone();
		return this;
	}

	/** Whether the region runs the KV append and attention when it is given a mirror. */
	boolean attendsOnDevice() {
		return attention != null;
	}

	/**
	 * Runs the region for layer {@code li} on the residual row {@code x} at
	 * {@code pos} without attention. Returns {q, k, v}, RoPE already applied to q
	 * and k, in arrays owned by the calling thread and overwritten by its next
	 * call; or {@code null} when the layer is not {@link #eligible}, or no device
	 * region could be allocated, and the caller must use its existing path. {@code x}
	 * is not modified.
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
	 * {@link #eligible}, or when no device region could be allocated (the card is
	 * full; the mirror is then untouched). {@code x} is not modified.
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
		if (r == null)
			return null; // no region could be opened: the caller keeps its existing path
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

	/**
	 * A region from the pool, or a new one; {@code null} when the pool is empty and
	 * a new one cannot be allocated because the device is out of memory (a card that
	 * filled after the path was built), which is logged once.
	 */
	private Region takeRegion() {
		Region r = free.poll();
		if (r != null || deviceFull)
			return r;
		try {
			r = openRegion();
		} catch (IllegalStateException e) {
			if (!GpuLayerOffload.isVramOom(e))
				throw e;
			deviceFull = true;
			java.util.logging.Logger.getLogger(ResidentQkvPath.class.getName())
					.warning("out of device memory opening a device-resident decode region (" + e.getMessage()
							+ "); decode calls that find no free region use the existing path from now on");
			return null;
		}
		regions.add(r);
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
			if (qNormWeight == null) {
				project(wq[li], r.q.devicePointer(), r.q8, stream);
				project(wk[li], packed, r.q8, stream);
			} else {
				// q and k land in scratch first: each head is normalized from there into place.
				project(wq[li], r.headNormIn, r.q8, stream);
				project(wk[li], r.headNormIn.asSlice(hiddenBytes()), r.q8, stream);
				headNorm(r.headNormIn, qNormWeight[li], r.q.devicePointer(), numHeads, stream);
				headNorm(r.headNormIn.asSlice(hiddenBytes()), kNormWeight[li], packed, kvDim / headDim, stream);
			}
			project(wv[li], packed.asSlice(kvBytes), r.q8, stream);
			r.q.markWritten(1);
			r.packed.markWritten(1);
			if (!rope.applyResident(r.q, pos) || !rope.applyResidentColumns(r.packed, 0, kvDim, pos))
				throw new IllegalStateException("RoPE kernel failed to load after the path was built");
			if (mirror != null) {
				attend(r, mirror, pos, packed, kvBytes, attentionWindows == null ? 0 : attentionWindows[li], stream);
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
	 * {@code [0, pos]} (its last {@code window} of them; 0: all) into the packed
	 * row's attention columns. The attention table
	 * (K pointer, V pointer, length) is written by a kernel from its launch
	 * arguments, so it costs no host-to-device copy.
	 */
	private void attend(Region r, DeviceKvCache mirror, int pos, MemorySegment packed, long kvBytes, int window,
			MemorySegment stream) {
		DeviceSpanTimer spans = r.chain.spans();
		int mark = spans.begin(stream, 1);
		mirror.writeWindowOnDevice(pos, 1, packed, packed.asSlice(kvBytes), kernels, stream);
		spans.compute(DeviceComputeEvent.KV_APPEND, 1, mark, stream);
		kernels.decodeAttentionTable(r.table, mirror.kPointer(), mirror.vPointer(), pos + 1, stream);
		mark = spans.begin(stream, 1);
		attention.launch(r.q.devicePointer(), r.table, r.table.asSlice(ADDRESS.byteSize()),
				r.table.asSlice(2 * ADDRESS.byteSize()), packed.asSlice(2 * kvBytes), 1, numHeads, gqaRatio,
				headDim, kvDim, GqaAttentionKernel.rowsPerBlock(true, 1), window, stream);
		spans.compute(DeviceComputeEvent.GQA_ATTENTION_REGION, 1, mark, stream);
	}

	/** RMS norm over each {@code headDim}-wide head of {@code in} into {@code out}: one row per head, one launch. */
	private void headNorm(MemorySegment in, DeviceFloatMatrix weight, MemorySegment out, int heads,
			MemorySegment stream) {
		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		if (kernel == null)
			throw new IllegalStateException("RMS-norm kernel failed to load after the path was built");
		kernel.launchResident(in, weight.devicePointer(), out, heads, headDim, eps, stream);
		DeviceSpanTally.compute(DeviceComputeEvent.RMS_NORM, 1, -1L);
	}

	private void project(DeviceQ4KMatrix w, MemorySegment y, MemorySegment q8, MemorySegment stream) {
		mmq.launchPacked(w.devicePointer(), q8, y, w.rows(), w.cols(), w.quantType(), stream);
	}

	/** Device bytes held by every thread's region plus the norm weights. */
	long deviceBytes() {
		long total = 0;
		for (Region r : regions)
			total += r.chain.deviceBytes();
		for (DeviceFloatMatrix[] ws : new DeviceFloatMatrix[][] { normWeight, qNormWeight, kNormWeight })
			if (ws != null)
				for (DeviceFloatMatrix w : ws)
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
			closeAll(normWeight);
			closeAll(qNormWeight);
			closeAll(kNormWeight);
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
		/** Projected q then k before their per-head norms, null without head norms. */
		final MemorySegment headNormIn;
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
			this.headNormIn = qNormWeight != null ? chain.allocateScratch((long) (hidden + kvDim) * Float.BYTES) : null;
			this.qAndPacked = new ResidentActivation[] { q, packed };
		}

		void close() {
			chain.close();
		}
	}
}
