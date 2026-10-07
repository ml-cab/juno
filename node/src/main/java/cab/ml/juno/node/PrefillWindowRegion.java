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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * The prefill-window device region: a transformer layer over a prefill window with
 * the activations kept on the device between its operations.
 *
 * <pre>
 *   [upload x] -> norm -> fp16 -> Q, K, V -> [bias | head norms] -> [RoPE] -> | KV mirror append -> attention | -> fp16 -> O
 *              -> x += O -> norm -> fp16 -> gate, up -> SwiGLU (fp16) -> down -> x += down -> download k, v
 *   (x is uploaded at the window's first layer and downloaded once, at the end of the window)
 * </pre>
 *
 * <p>On the host path every matmul uploads its input and downloads its result, and
 * the norms, SwiGLU and residual adds run on the host between them, single-threaded
 * for SwiGLU. Here the work between the matmuls runs on the device, and the window's
 * residual stream stays there from layer to layer: it is uploaded at the window's
 * first layer and downloaded once, when the caller takes it
 * ({@link Window#materializeResidual}) at the end of the window or before a layer
 * that runs on the host. Per layer the host sees only the K and V rows (the host KV
 * tensors stay the source of truth). When attention cannot run inside the region
 * (RoPE on the host, the attention kernel off, or no usable KV mirror), the layer
 * splits in two: {@link Window#runLayer} returns Q, K and V, the caller runs RoPE
 * and attention, and {@link Window#finishLayer} runs the rest.
 *
 * <p>Every operation matches the host window path step for step: the matmuls are the
 * same cuBLAS FP16 GEMMs fed the same FP16 bits ({@link CudaMatVec#gemmOnStream}), the
 * casts round as {@code Float.floatToFloat16} does, the norms sum in the host loop's
 * order, SwiGLU and the adds reproduce the host arithmetic ({@code prefill_window.cu}),
 * RoPE is the device kernel already held to the CPU rotation, and attention is the
 * same kernel the host path calls, reading the same mirror. So the region computes
 * what the host path computes, bit for bit wherever each kernel's parity test holds
 * it to bit identity.
 *
 * <p>A fused {@code [q; k; v]} projection (Phi-3) runs as one GEMM, as on the host
 * path. With device RoPE it is split into Q, K and V on the device ({@code split_qkv},
 * exact copies) and the layer rotates and attends here like any other; without it the
 * fused rows are split on their way to the host, which rotates and attends. The region
 * applies no Q/K/V biases to a fused projection (no fused model has them).
 *
 * <p>Scope: CUDA only (the kernels are PTX). A layer runs here when all of its
 * projections are device matrices (packed K-quant or FP16); any other layer, and any
 * window of {@value #MAX_HOST_WINDOW} rows or fewer (where the host path's packed GEMV
 * needs no dequantized weights), keeps the host path. On by default;
 * {@code -DJUNO_PREFILL_REGION=off} (or the environment variable) keeps every window
 * on the host path, for comparison.
 *
 * <p>Concurrency and memory. A call takes a {@link Window} (its own stream and
 * buffers) from a pool and returns it when the prefill window is done, so there are
 * as many windows as prefills ever ran at once. Each operation on a window holds the
 * context's serialization lock and waits for its stream before releasing it, because
 * the K-quant dequantization scratch is the matmul backend's own. A window's buffers
 * grow to the widest prefill window it has served and are freed by {@link #close}.
 * Running out of device memory anywhere in a call surfaces as the allocator's own
 * {@link IllegalStateException}; the caller takes the layer's input back
 * ({@link Window#recoverLayerInput}; a layer whose input is only on the device first
 * copies it aside, device to device, before updating the residual in place) and runs
 * that layer on the host path instead.
 */
final class PrefillWindowRegion implements AutoCloseable {

	private static final Logger log = Logger.getLogger(PrefillWindowRegion.class.getName());

	/** The switch's name; see {@link PrefillRegionOptions}. */
	static final String ENV_PROPERTY = PrefillRegionOptions.ENV_PROPERTY;

	/** Windows of at most this many rows keep the host path (the backend's packed GEMV threshold). */
	static final int MAX_HOST_WINDOW = 8;

	/** The residual stream's staging-site name: its copies are {@code upload(prefill residual)} and {@code materialize(prefill residual)}. */
	static final String RESIDUAL = "prefill residual";

	/** Window capacities are rounded up to this, so windows of slightly different widths share buffers. */
	static final int CAPACITY_STEP = 64;

	/** One device weight matrix: packed K-quant or FP16. */
	record Matrix(DeviceQ4KMatrix q4, DeviceHalfMatrix half) {

		/** Layer {@code li}'s device matrix from whichever array holds it, or {@code null} when neither does. */
		static Matrix of(DeviceQ4KMatrix[] q4, DeviceHalfMatrix[] half, int li) {
			if (q4 != null && li < q4.length && q4[li] != null)
				return new Matrix(q4[li], null);
			if (half != null && li < half.length && half[li] != null)
				return new Matrix(null, half[li]);
			return null;
		}

		int rows() {
			return q4 != null ? q4.rows() : half.rows();
		}

		int cols() {
			return q4 != null ? q4.cols() : half.cols();
		}
	}

	/**
	 * One layer's weights. Q, K and V are three matrices or one fused {@code [q; k; v]}
	 * matrix ({@code qkv}); gate and up are two matrices or one fused {@code [gate; up]}
	 * matrix ({@code gateUp}). The biases are null for a model without them, and so are
	 * the per-head Q and K norm weights ({@code headDim} floats each; Qwen3).
	 */
	record Layer(Matrix q, Matrix k, Matrix v, Matrix qkv, Matrix o, Matrix gate, Matrix up, Matrix gateUp,
			Matrix down, float[] attnNorm, float[] ffnNorm, float[] bq, float[] bk, float[] bv, float[] qNorm,
			float[] kNorm) {

		/** A layer with separate projections, or {@code null} when any of them is not on the device. */
		static Layer separate(Matrix q, Matrix k, Matrix v, Matrix o, Matrix gate, Matrix up, Matrix down,
				float[] attnNorm, float[] ffnNorm, float[] bq, float[] bk, float[] bv) {
			if (q == null || k == null || v == null || o == null || gate == null || up == null || down == null)
				return null;
			return new Layer(q, k, v, null, o, gate, up, null, down, attnNorm, ffnNorm, bq, bk, bv, null, null);
		}

		/**
		 * This layer with each Q and K head RMS-normalized by {@code qNorm} and {@code kNorm}
		 * between the projections and RoPE, as Qwen3 does; {@code null} stays {@code null}.
		 */
		static Layer withHeadNorms(Layer l, float[] qNorm, float[] kNorm) {
			if (l == null)
				return null;
			return new Layer(l.q(), l.k(), l.v(), l.qkv(), l.o(), l.gate(), l.up(), l.gateUp(), l.down(), l.attnNorm(),
					l.ffnNorm(), l.bq(), l.bk(), l.bv(), java.util.Objects.requireNonNull(qNorm, "qNorm"),
					java.util.Objects.requireNonNull(kNorm, "kNorm"));
		}

		/**
		 * A layer whose Q/K/V and gate/up projections are each given fused or separate:
		 * pass {@code qkv} or {@code q, k, v}, and {@code gateUp} or {@code gate, up}.
		 * {@code null} when a projection is not on the device.
		 */
		static Layer mixed(Matrix qkv, Matrix q, Matrix k, Matrix v, Matrix o, Matrix gateUp, Matrix gate,
				Matrix up, Matrix down, float[] attnNorm, float[] ffnNorm) {
			boolean attnOk = qkv != null || (q != null && k != null && v != null);
			boolean ffnOk = gateUp != null || (gate != null && up != null);
			if (!attnOk || !ffnOk || o == null || down == null)
				return null;
			return new Layer(qkv != null ? null : q, qkv != null ? null : k, qkv != null ? null : v, qkv, o,
					gateUp != null ? null : gate, gateUp != null ? null : up, gateUp, down, attnNorm, ffnNorm, null,
					null, null, null, null);
		}
	}

	/** The model dimensions the region's buffers and kernels are sized by. */
	record Shape(int hidden, int qDim, int kvDim, int inter, int numHeads, int numKvHeads, int headDim, int gqaRatio,
			float eps) {
	}

	private final String handler;
	private final GpuContext ctx;
	private final CudaMatVec mv;
	private final Shape shape;
	private final Layer[] layers;
	private final DeviceFloatMatrix[] attnNormDev;
	private final DeviceFloatMatrix[] ffnNormDev;
	private final DeviceFloatMatrix[] bqDev;
	private final DeviceFloatMatrix[] bkDev;
	private final DeviceFloatMatrix[] bvDev;
	/** Per-head Q and K norm weights on the device; null entries where a layer has none. */
	private final DeviceFloatMatrix[] qNormDev;
	private final DeviceFloatMatrix[] kNormDev;
	/** Device RoPE, or {@code null} when the caller rotates on the host. */
	private final ResidentRope rope;
	/** Whether attention runs inside the region (device RoPE and the attention kernel both available). */
	private final boolean attentionOnDevice;
	private final PrefillWindowKernels kernels;
	private final GqaAttentionKernel attention;

	/** Whether any layer has a fused Q/K/V projection, which needs the fused activation. */
	private final boolean fusedQkv;

	private final List<Window> windows = new CopyOnWriteArrayList<>();
	private final ConcurrentLinkedQueue<Window> idle = new ConcurrentLinkedQueue<>();
	private final AtomicBoolean mirrorWarned = new AtomicBoolean();
	private volatile boolean closed;

	private PrefillWindowRegion(String handler, CudaMatVec mv, Shape shape, Layer[] layers,
			DeviceFloatMatrix[] attnNormDev, DeviceFloatMatrix[] ffnNormDev, DeviceFloatMatrix[] bqDev,
			DeviceFloatMatrix[] bkDev, DeviceFloatMatrix[] bvDev, DeviceFloatMatrix[] qNormDev,
			DeviceFloatMatrix[] kNormDev, ResidentRope rope, boolean attentionOnDevice, PrefillWindowKernels kernels,
			GqaAttentionKernel attention) {
		this.handler = handler;
		this.ctx = mv.gpuContext();
		this.mv = mv;
		this.shape = shape;
		this.layers = layers;
		this.attnNormDev = attnNormDev;
		this.ffnNormDev = ffnNormDev;
		this.bqDev = bqDev;
		this.bkDev = bkDev;
		this.bvDev = bvDev;
		this.qNormDev = qNormDev;
		this.kNormDev = kNormDev;
		this.rope = rope;
		this.attentionOnDevice = attentionOnDevice;
		this.kernels = kernels;
		this.attention = attention;
		boolean fused = false;
		for (Layer l : layers)
			fused |= l != null && l.qkv() != null;
		this.fusedQkv = fused;
	}

	/**
	 * Builds the region for a model, or returns {@code null} (logging why) when it is
	 * off, the backend is not CUDA, a kernel does not load, no layer has all of its
	 * projections on the device, or the norm weights do not fit on the device.
	 *
	 * @param layers        per layer, {@code null} where the layer is not on the device
	 * @param ropePairing   the pairing to rotate on the device, or {@code null} to leave RoPE to the caller
	 * @param attentionKernel whether the caller runs the attention kernel; attention moves into the region
	 *                      only when it does and RoPE runs on the device
	 */
	static PrefillWindowRegion create(String handler, MatVec backend, Shape shape, Layer[] layers,
			RopePairing ropePairing, float ropeTheta, boolean attentionKernel) {
		java.util.function.Function<GpuContext, ResidentRope> rope = ropePairing == null ? null
				: ctx -> RopeKernel.tryLoad() != null ? CudaRope.tryCreate(ctx, shape.headDim(), ropeTheta, ropePairing)
						: null;
		return create(handler, backend, shape, layers, rope, attentionKernel);
	}

	/**
	 * {@link #create(String, MatVec, Shape, Layer[], RopePairing, float, boolean)} for the
	 * Phi-3 family: RoPE on the device is its extended rotation ({@link CudaPhi3Rope}).
	 */
	static PrefillWindowRegion create(String handler, MatVec backend, Shape shape, Layer[] layers,
			Phi3RopeConfig phi3Rope, boolean attentionKernel) {
		java.util.Objects.requireNonNull(phi3Rope, "phi3Rope");
		return create(handler, backend, shape, layers,
				ctx -> RopeKernel.tryLoad() != null ? CudaPhi3Rope.tryCreate(ctx, shape.headDim(), phi3Rope) : null,
				attentionKernel);
	}

	/**
	 * @param ropeFactory the device rotation for the region's context ({@code null} result: the
	 *                    caller rotates), or {@code null} to leave RoPE to the caller
	 */
	private static PrefillWindowRegion create(String handler, MatVec backend, Shape shape, Layer[] layers,
			java.util.function.Function<GpuContext, ResidentRope> ropeFactory, boolean attentionKernel) {
		if (!PrefillRegionOptions.requested()) {
			log.info(handler + ": prefill-window device region off (" + ENV_PROPERTY + "=off)");
			return null;
		}
		if (!(backend instanceof CudaMatVec mv))
			return null;
		int eligible = 0;
		for (Layer l : layers) {
			if (l == null)
				continue;
			eligible++;
			if (l.qkv() != null && l.bq() != null)
				throw new IllegalArgumentException("the region applies no Q/K/V biases to a fused projection");
			if (l.qNorm() != null) {
				// Q and K are projected into the attention-output and projection scratch and
				// normalized from there into q and k; see Window#attentionInputs.
				if (l.qkv() != null || l.bq() != null)
					throw new IllegalArgumentException("the region's per-head Q/K norm takes separate projections without biases");
				if (shape.kvDim() > shape.hidden())
					throw new IllegalArgumentException("the region's per-head Q/K norm needs kvDim <= hidden");
				if (l.qNorm().length != shape.headDim() || l.kNorm().length != shape.headDim())
					throw new IllegalArgumentException("per-head norm weights must be headDim (" + shape.headDim() + ") wide");
			}
		}
		if (eligible == 0)
			return null;
		PrefillWindowKernels kernels = PrefillWindowKernels.tryLoad();
		GqaAttentionKernel attention = attentionKernel && GqaAttentionKernel.supportsHeadDim(shape.headDim())
				? GqaAttentionKernel.tryLoad()
				: null;
		if (kernels == null) {
			log.warning(handler + ": prefill-window device region unavailable (its kernels did not load);"
					+ " prefill windows stay on the host path");
			return null;
		}
		GpuContext ctx = mv.gpuContext();
		int n = layers.length;
		DeviceFloatMatrix[] attnNormDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] ffnNormDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] bqDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] bkDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] bvDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] qNormDev = new DeviceFloatMatrix[n];
		DeviceFloatMatrix[] kNormDev = new DeviceFloatMatrix[n];
		boolean headNorms = false;
		ResidentRope rope = null;
		try {
			if (ropeFactory != null)
				rope = ropeFactory.apply(ctx);
			for (int li = 0; li < n; li++) {
				Layer l = layers[li];
				if (l == null)
					continue;
				attnNormDev[li] = DeviceFloatMatrix.upload(ctx, l.attnNorm(), 1, shape.hidden());
				ffnNormDev[li] = DeviceFloatMatrix.upload(ctx, l.ffnNorm(), 1, shape.hidden());
				if (l.bq() != null) {
					bqDev[li] = DeviceFloatMatrix.upload(ctx, l.bq(), 1, shape.qDim());
					bkDev[li] = DeviceFloatMatrix.upload(ctx, l.bk(), 1, shape.kvDim());
					bvDev[li] = DeviceFloatMatrix.upload(ctx, l.bv(), 1, shape.kvDim());
				}
				if (l.qNorm() != null) {
					qNormDev[li] = DeviceFloatMatrix.upload(ctx, l.qNorm(), 1, shape.headDim());
					kNormDev[li] = DeviceFloatMatrix.upload(ctx, l.kNorm(), 1, shape.headDim());
					headNorms = true;
				}
			}
		} catch (IllegalStateException ex) {
			closeAll(attnNormDev, ffnNormDev, bqDev, bkDev, bvDev, qNormDev, kNormDev);
			if (rope != null)
				rope.close();
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			log.warning(handler + ": out of device memory for the prefill-window device region's norm weights;"
					+ " prefill windows stay on the host path");
			return null;
		}
		boolean attentionOnDevice = rope != null && attention != null;
		log.info(handler + ": prefill windows of more than " + MAX_HOST_WINDOW + " rows run on the device ("
				+ eligible + " of " + n + " layers): norms, matmuls, SwiGLU and residual adds"
				+ (headNorms ? ", per-head Q/K norms" : "") + (rope != null ? ", RoPE" : "") + (attentionOnDevice ? ", attention" : "")
				+ (rope == null ? "; RoPE and attention stay on the host" : "")
				+ (rope != null && !attentionOnDevice ? "; attention stays on the host" : ""));
		return new PrefillWindowRegion(handler, mv, shape, layers, attnNormDev, ffnNormDev, bqDev, bkDev, bvDev,
				qNormDev, kNormDev, rope, attentionOnDevice, kernels, attention);
	}

	/** Device bytes a window of {@code rows} rows holds on this region ({@link PrefillWindowFootprint}). */
	long windowDeviceBytes(int rows) {
		return PrefillWindowFootprint.bytes(shape, fusedQkv, rows);
	}

	/** Whether layer {@code li} (0-based within the shard) runs on the region. */
	boolean eligible(int li) {
		return li >= 0 && li < layers.length && layers[li] != null;
	}

	/** Whether {@link Window#runLayer} applies RoPE to Q and K; otherwise the caller rotates them. */
	boolean ropeOnDevice() {
		return rope != null;
	}

	/**
	 * Takes a window of at least {@code rows} rows from the pool, allocating one when
	 * none is idle and wide enough. Close it when the prefill window is done.
	 *
	 * @throws IllegalStateException the allocator's out-of-memory error when its buffers do not fit
	 */
	Window open(int rows) {
		if (closed)
			throw new IllegalStateException("prefill-window region is closed");
		if (rows <= MAX_HOST_WINDOW)
			throw new IllegalArgumentException("a window of " + rows + " rows belongs on the host path");
		Window w = idle.poll();
		while (w != null && w.capacity < rows) {
			retire(w);
			w = idle.poll();
		}
		if (w == null) {
			int capacity = PrefillWindowFootprint.capacity(rows);
			synchronized (ctx.cublasSerializationLock()) {
				w = new Window(capacity);
			}
			windows.add(w);
		}
		return w;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		synchronized (ctx.cublasSerializationLock()) {
			for (Window w : windows)
				w.free();
			windows.clear();
			idle.clear();
			closeAll(attnNormDev, ffnNormDev, bqDev, bkDev, bvDev, qNormDev, kNormDev);
			if (rope != null)
				rope.close();
		}
	}

	private void retire(Window w) {
		windows.remove(w);
		synchronized (ctx.cublasSerializationLock()) {
			w.free();
		}
	}

	private static void closeAll(DeviceFloatMatrix[]... arrays) {
		for (DeviceFloatMatrix[] a : arrays)
			for (DeviceFloatMatrix m : a)
				if (m != null)
					m.close();
	}

	/**
	 * One prefill window's device buffers and stream. Owned by one thread from
	 * {@link #open} to {@link #close}; layer after layer runs through it.
	 */
	final class Window implements AutoCloseable {

		private final int capacity;
		private final ResidentChain chain;
		private final MemorySegment stream;
		private final DeviceSpanTimer spans;
		private final ResidentActivation x;
		private final ResidentActivation q;
		private final ResidentActivation k;
		private final ResidentActivation v;
		private final ResidentActivation attn;
		/** The fused Q/K/V projection, {@code [capacity][qDim + 2 kvDim]}; null when no layer is fused. */
		private final ResidentActivation qkv;
		/** Host rows {@link #qkv} is downloaded into before it is split; null when no layer is fused. */
		private final float[][] qkvHost;
		/** Normalized residual, FP32 {@code [capacity][hidden]}. */
		private final MemorySegment xn;
		/** FP16 GEMM input, {@code [capacity][max(hidden, qDim, inter)]}. */
		private final MemorySegment xh;
		/** O and down projection output, FP32 {@code [capacity][hidden]}. */
		private final MemorySegment proj;
		/** Gate and up side by side, FP32 {@code [capacity][2 * inter]}. */
		private final MemorySegment gateUp;
		/** Attention pointer and length tables on the device: K pointers, V pointers, sequence lengths. */
		private final MemorySegment tables;
		/** Pinned host copy of {@link #tables}, written before each attention launch. */
		private final MemorySegment tablesHost;
		private final long tablesBytes;
		/** The layer's input, copied here before the layer updates the residual in place; see {@link #recoverLayerInput}. */
		private final MemorySegment xIn;
		private final ResidentActivation[] qkvActs;
		private final ResidentActivation[] kvActs;
		private final float[][][] outs = new float[3][][];
		private final float[][][] kvOuts = new float[2][][];
		/** The device residual {@link #x} is newer than the caller's host rows. */
		private boolean hostStale;
		/** The current layer's input is on the host (it was uploaded, not carried over from the last layer). */
		private boolean inputOnHost;
		/** {@link #xIn} holds the current layer's input. */
		private boolean inputSaved;
		private boolean freed;

		private Window(int capacity) {
			this.capacity = capacity;
			ResidentChain c = ResidentChain.open(ctx);
			MemorySegment host = null;
			try {
				int widestIn = Math.max(shape.hidden(), Math.max(shape.qDim(), shape.inter()));
				this.x = c.allocate(capacity, shape.hidden(), RESIDUAL);
				this.q = c.allocate(capacity, shape.qDim(), "prefill q");
				this.k = c.allocate(capacity, shape.kvDim(), "prefill k");
				this.v = c.allocate(capacity, shape.kvDim(), "prefill v");
				this.attn = c.allocate(capacity, shape.qDim(), "prefill attention");
				this.qkv = fusedQkv ? c.allocate(capacity, shape.qDim() + 2 * shape.kvDim(), "prefill qkv") : null;
				this.xn = c.allocateScratch((long) capacity * shape.hidden() * Float.BYTES);
				this.xIn = c.allocateScratch((long) capacity * shape.hidden() * Float.BYTES);
				this.xh = c.allocateScratch((long) capacity * widestIn * Short.BYTES);
				this.proj = c.allocateScratch((long) capacity * shape.hidden() * Float.BYTES);
				this.gateUp = c.allocateScratch((long) capacity * 2 * shape.inter() * Float.BYTES);
				this.tablesBytes = (long) capacity * (2 * ADDRESS.byteSize() + Integer.BYTES);
				this.tables = c.allocateScratch(tablesBytes);
				host = ctx.bindings().hostMalloc(ctx.deviceIndex(), tablesBytes);
			} catch (RuntimeException e) {
				c.close();
				throw e;
			}
			this.chain = c;
			this.tablesHost = host.reinterpret(tablesBytes);
			this.stream = c.stream();
			this.spans = c.spans();
			this.qkvHost = fusedQkv ? new float[capacity][shape.qDim() + 2 * shape.kvDim()] : null;
			this.qkvActs = new ResidentActivation[] { q, k, v };
			this.kvActs = new ResidentActivation[] { k, v };
		}

		/** Device bytes this window holds: its buffers (attention keeps no scratch of its own). */
		long deviceBytes() {
			return chain.deviceBytes();
		}

		/**
		 * Runs layer {@code li} over the window {@code xHost} (rows at positions
		 * {@code startPos, startPos + 1, ...}).
		 *
		 * <p>The residual stream stays on the device from one layer to the next: it is
		 * uploaded from {@code xHost} only when the device does not already hold it (the
		 * window's first layer, or the first after the caller ran a layer on the host),
		 * and the layer's output is left on the device. {@code xHost} is not written;
		 * the caller takes the residual with {@link #materializeResidual} before
		 * anything on the host reads it, and at the end of the window.
		 *
		 * <p>When attention runs inside the region (device RoPE, the attention kernel,
		 * and a live {@code mirror} holding every position before {@code startPos}),
		 * the whole layer runs: the window's K and V rows are cast into the mirror,
		 * attention reads it, and the method returns {@code true} with the window's K
		 * and V rows (rotated) in {@code kOut}/{@code vOut}. The mirror's watermark is
		 * not moved: the caller writes the host KV tensors from {@code kOut}/{@code vOut}
		 * first and then marks the window ({@link DeviceKvCache#markWritten}).
		 *
		 * <p>Otherwise it returns {@code false} with Q, K and V in {@code qOut},
		 * {@code kOut}, {@code vOut} (rotated if {@link #ropeOnDevice}); the caller runs
		 * attention and then {@link #finishLayer}. A mirror that runs out of device
		 * memory here is retired (closed), as on the host path, and the layer takes this
		 * second form.
		 *
		 * <p>Any other device out-of-memory error surfaces as the allocator's
		 * {@link IllegalStateException}; the caller then takes the layer's input with
		 * {@link #recoverLayerInput} and runs the layer on the host path.
		 */
		boolean runLayer(int li, float[][] xHost, int startPos, DeviceKvCache mirror, float[][] qOut,
				float[][] kOut, float[][] vOut) {
			int w = requireWindow(li, xHost.length);
			boolean inside = attentionOnDevice && mirror != null && mirror.live() && mirror.readableThrough(startPos);
			synchronized (ctx.cublasSerializationLock()) {
				boolean done = false;
				try {
					enterLayer(xHost, w);
					attentionInputs(li, w, startPos);
					if (inside)
						inside = attendInside(mirror, startPos, w);
					if (!inside) {
						if (layers[li].qkv() != null && rope == null)
							downloadFused(w, qOut, kOut, vOut);
						else
							download(qkvActs, qOut, kOut, vOut);
						done = true;
						return false;
					}
					feedForward(li, w);
					kvOuts[0] = kOut;
					kvOuts[1] = vOut;
					try {
						ResidentActivation.materializeAll(kvActs, kvOuts);
					} finally {
						kvOuts[0] = kvOuts[1] = null;
					}
					done = true;
					return true;
				} finally {
					if (!done)
						drain();
				}
			}
		}

		/**
		 * Finishes layer {@code li} after {@link #runLayer} returned {@code false}: uploads
		 * the attention output, runs the output projection, both residual adds, the
		 * second norm and the feed-forward block, and leaves the layer's output on the
		 * device, as {@link #runLayer} does.
		 */
		void finishLayer(int li, float[][] attnHost, float[][] xHost) {
			int w = requireWindow(li, xHost.length);
			synchronized (ctx.cublasSerializationLock()) {
				boolean done = false;
				try {
					attn.upload(attnHost, w);
					feedForward(li, w);
					done = true;
				} finally {
					if (!done)
						drain();
				}
			}
		}

		/**
		 * The materialization boundary of the residual stream: writes it into
		 * {@code xHost} when the device holds a newer one than the host, after which the
		 * host rows are current and the next {@link #runLayer} uploads them. Call it
		 * before the host reads the residual (a layer that does not run on the region,
		 * the LM head, the hand-off to the next node) and at the end of the window. A
		 * no-op when the host is already current.
		 */
		void materializeResidual(float[][] xHost) {
			if (!hostStale)
				return;
			synchronized (ctx.cublasSerializationLock()) {
				x.materialize(xHost);
			}
			hostStale = false;
		}

		/**
		 * After {@link #runLayer} or {@link #finishLayer} failed with a device
		 * out-of-memory error, writes the failed layer's input into {@code xHost}, so the
		 * caller can run that layer on the host path: from the host rows themselves when
		 * the layer uploaded them, from the device residual when the layer had not yet
		 * updated it, and otherwise from the copy taken before it did. Afterwards the
		 * host rows are current, as after {@link #materializeResidual}.
		 */
		void recoverLayerInput(float[][] xHost) {
			if (inputOnHost) {
				hostStale = false;
				return;
			}
			int w = x.rows();
			synchronized (ctx.cublasSerializationLock()) {
				if (inputSaved) {
					copyDeviceToDevice(x.devicePointer(), xIn, (long) w * shape.hidden() * Float.BYTES, w,
							"memcpy(prefill residual layer input restore)");
					x.markWritten(w);
				}
				x.materialize(xHost);
			}
			hostStale = false;
			inputOnHost = true;
			inputSaved = false;
		}

		/** Returns the window to the pool, or frees it when the region has been closed. */
		@Override
		public void close() {
			hostStale = false;
			inputOnHost = false;
			inputSaved = false;
			if (closed) {
				synchronized (ctx.cublasSerializationLock()) {
					free();
				}
				return;
			}
			idle.offer(this);
		}

		// ── the layer, issued on this window's stream ────────────────────────────

		/**
		 * norm -> fp16 -> Q, K, V -> bias -> RoPE; for a fused projection, norm -> fp16 -> QKV,
		 * then, with device RoPE, split -> RoPE. With per-head norms, Q and K are projected
		 * into scratch ({@link #attn} and {@link #proj}, both written again later in the
		 * layer) and each head is normalized from there into {@link #q} and {@link #k}
		 * before RoPE: out of place, so the kernel never reads a row it is writing.
		 */
		private void attentionInputs(int li, int w, int startPos) {
			Layer l = layers[li];
			int h = shape.hidden();
			normalize(x.devicePointer(), attnNormDev[li], w);
			toHalf(xn, (long) w * h, w);
			if (l.qkv() != null) {
				// One GEMM over the fused rows, as the host path runs it: a GEMM per row
				// range could take another cuBLAS algorithm and round differently.
				gemm(l.qkv(), qkv.devicePointer(), qkv.dim(), w);
				qkv.markWritten(w);
				if (rope == null)
					return;
				int mark = spans.begin(stream, w);
				kernels.splitQkv(qkv.devicePointer(), q.devicePointer(), k.devicePointer(), v.devicePointer(), w,
						shape.qDim(), shape.kvDim(), stream);
				spans.compute(DeviceComputeEvent.SPLIT_QKV, w, mark, stream);
			} else if (qNormDev[li] != null) {
				gemm(l.q(), attn.devicePointer(), shape.qDim(), w);
				gemm(l.k(), proj, shape.kvDim(), w);
				gemm(l.v(), v.devicePointer(), shape.kvDim(), w);
				normalizeHeads(attn.devicePointer(), qNormDev[li], q.devicePointer(), w, shape.numHeads());
				normalizeHeads(proj, kNormDev[li], k.devicePointer(), w, shape.numKvHeads());
			} else {
				gemm(l.q(), q.devicePointer(), shape.qDim(), w);
				gemm(l.k(), k.devicePointer(), shape.kvDim(), w);
				gemm(l.v(), v.devicePointer(), shape.kvDim(), w);
			}
			q.markWritten(w);
			k.markWritten(w);
			v.markWritten(w);
			if (bqDev[li] != null) {
				addBias(q, bqDev[li], w);
				addBias(k, bkDev[li], w);
				addBias(v, bvDev[li], w);
			}
			if (rope != null) {
				int mark = spans.begin(stream, w);
				if (!rope.applyResident(q, startPos) || !rope.applyResident(k, startPos))
					throw new IllegalStateException("RoPE kernel failed to load after the region was built");
				spans.compute(DeviceComputeEvent.ROPE, w, mark, stream);
			}
		}

		/**
		 * Starts a layer: uploads the residual from {@code xHost} unless the device
		 * already holds it from the previous layer.
		 */
		private void enterLayer(float[][] xHost, int w) {
			if (hostStale) {
				if (x.rows() != w)
					throw new IllegalStateException(
							"the device residual holds " + x.rows() + " rows, the layer was given " + w);
				inputOnHost = false;
			} else {
				x.upload(xHost, w);
				inputOnHost = true;
			}
			inputSaved = false;
		}

		/**
		 * fp16(attn) -> O -> x += O -> norm -> fp16 -> gate, up -> SwiGLU -> down -> x += down.
		 * When the layer's input exists only on the device, it is copied aside first
		 * (device to device), so a failure after the residual has been updated in place
		 * can still hand the caller the input ({@link #recoverLayerInput}).
		 */
		private void feedForward(int li, int w) {
			Layer l = layers[li];
			int h = shape.hidden();
			int inter = shape.inter();
			if (!inputOnHost && !inputSaved) {
				copyDeviceToDevice(xIn, x.devicePointer(), (long) w * h * Float.BYTES, w,
						"memcpy(prefill residual layer input D2D)");
				inputSaved = true;
			}
			toHalf(attn.devicePointer(), (long) w * shape.qDim(), w);
			gemm(l.o(), proj, h, w);
			addInPlace(x.devicePointer(), proj, (long) w * h, w);
			normalize(x.devicePointer(), ffnNormDev[li], w);
			toHalf(xn, (long) w * h, w);
			if (l.gateUp() != null) {
				gemm(l.gateUp(), gateUp, 2 * inter, w);
			} else {
				gemm(l.gate(), gateUp, 2 * inter, w);
				gemm(l.up(), gateUp.asSlice((long) inter * Float.BYTES), 2 * inter, w);
			}
			int mark = spans.begin(stream, w);
			kernels.swigluToHalf(gateUp, xh, w, inter, stream);
			spans.compute(DeviceComputeEvent.SWIGLU, w, mark, stream);
			gemm(l.down(), proj, h, w);
			addInPlace(x.devicePointer(), proj, (long) w * h, w);
			x.markWritten(w);
			hostStale = true;
		}

		/**
		 * Casts the window's K and V rows into the mirror and runs attention over the
		 * mirror into {@link #attn}. Returns {@code false}, retiring the mirror, when
		 * the device runs out of memory for the mirror's growth.
		 */
		private boolean attendInside(DeviceKvCache mirror, int startPos, int w) {
			try {
				int mark = spans.begin(stream, w);
				mirror.writeWindowOnDevice(startPos, w, k.devicePointer(), v.devicePointer(), kernels, stream);
				spans.compute(DeviceComputeEvent.KV_APPEND, w, mark, stream);
				long ptrBytes = (long) w * ADDRESS.byteSize();
				MemorySegment kPtr = mirror.kPointer();
				MemorySegment vPtr = mirror.vPointer();
				for (int b = 0; b < w; b++) {
					tablesHost.setAtIndex(ADDRESS, b, kPtr);
					tablesHost.setAtIndex(ADDRESS, w + b, vPtr);
					tablesHost.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, startPos + b + 1);
				}
				long used = 2 * ptrBytes + (long) w * Integer.BYTES;
				copyTables(used, w);
				mark = spans.begin(stream, w);
				attention.launch(q.devicePointer(), tables, tables.asSlice(ptrBytes), tables.asSlice(2 * ptrBytes),
						attn.devicePointer(), w, shape.numHeads(), shape.gqaRatio(), shape.headDim(), shape.kvDim(),
						GqaAttentionKernel.rowsPerBlock(true, w), 0, stream);
				spans.compute(DeviceComputeEvent.GQA_ATTENTION_REGION, w, mark, stream);
				attn.markWritten(w);
				return true;
			} catch (IllegalStateException ex) {
				if (!GpuLayerOffload.isVramOom(ex))
					throw ex;
				if (mirrorWarned.compareAndSet(false, true))
					log.warning(handler + ": out of device memory for prefill attention inside the device region"
							+ " - attention continues on the CPU, which holds the same history. Lower --gpu-layers,"
							+ " or pass --gpu-attention off.");
				chain.sync();
				mirror.close();
				return false;
			}
		}

		// ── operations, each timed as its own compute site ───────────────────────

		private void normalize(MemorySegment in, DeviceFloatMatrix weight, int w) {
			int mark = spans.begin(stream, w);
			kernels.rmsNormHostOrder(in, weight.devicePointer(), xn, w, shape.hidden(), shape.eps(), stream);
			spans.compute(DeviceComputeEvent.RMS_NORM, w, mark, stream);
		}

		/**
		 * Each of the window's {@code w x heads} head rows of {@code in}, normalized by
		 * {@code weight} into {@code out}: the host-order norm, so bit for bit what
		 * {@code Qwen3TransformerHandler.rmsNormPerHead} computes on the host.
		 */
		private void normalizeHeads(MemorySegment in, DeviceFloatMatrix weight, MemorySegment out, int w, int heads) {
			int mark = spans.begin(stream, w);
			kernels.rmsNormHostOrder(in, weight.devicePointer(), out, w * heads, shape.headDim(), shape.eps(), stream);
			spans.compute(DeviceComputeEvent.RMS_NORM, w, mark, stream);
		}

		private void toHalf(MemorySegment in, long n, int w) {
			int mark = spans.begin(stream, w);
			kernels.toHalf(in, xh, n, stream);
			spans.compute(DeviceComputeEvent.CONVERT_FP16, w, mark, stream);
		}

		private void addInPlace(MemorySegment target, MemorySegment addend, long n, int w) {
			int mark = spans.begin(stream, w);
			kernels.addInPlace(target, addend, n, stream);
			spans.compute(DeviceComputeEvent.RESIDUAL_ADD, w, mark, stream);
		}

		private void addBias(ResidentActivation a, DeviceFloatMatrix bias, int w) {
			int mark = spans.begin(stream, w);
			kernels.addBias(a.devicePointer(), bias.devicePointer(), w, a.dim(), stream);
			spans.compute(DeviceComputeEvent.BIAS_ADD, w, mark, stream);
		}

		/** {@code out = A xh}, row {@code b} at {@code out + b * ldc} floats. */
		private void gemm(Matrix a, MemorySegment out, int ldc, int w) {
			if (a.q4() != null)
				mv.gemmOnStream(a.q4(), xh, out, ldc, w, stream, spans);
			else
				mv.gemmOnStream(a.half(), xh, out, ldc, w, stream, spans);
		}

		private void copyDeviceToDevice(MemorySegment dst, MemorySegment src, long bytes, int w, String site) {
			int mark = spans.begin(stream, w);
			int rc;
			try {
				rc = (int) ctx.bindings().gpuMemcpyAsync().invokeExact(dst, src, bytes, GpuBindings.D2D, stream);
			} catch (Throwable t) {
				throw new IllegalStateException(site + ": native call failed", t);
			}
			GpuBindings.check(rc, site);
			spans.staging(GpuBindings.D2D, bytes, w, site, mark, stream);
		}

		private void copyTables(long bytes, int w) {
			int mark = spans.begin(stream, w);
			int rc;
			try {
				rc = (int) ctx.bindings().gpuMemcpyAsync().invokeExact(tables, tablesHost, bytes, GpuBindings.H2D,
						stream);
			} catch (Throwable t) {
				throw new IllegalStateException("memcpy(region attention tables H2D): native call failed", t);
			}
			GpuBindings.check(rc, "memcpy(region attention tables H2D)");
			spans.staging(GpuBindings.H2D, bytes, w, "memcpy(region attention tables H2D)", mark, stream);
		}

		/** Downloads the fused Q/K/V rows and splits them into the three host outputs. */
		private void downloadFused(int w, float[][] qOut, float[][] kOut, float[][] vOut) {
			qkv.materialize(qkvHost);
			int qd = shape.qDim();
			int kv = shape.kvDim();
			for (int b = 0; b < w; b++) {
				System.arraycopy(qkvHost[b], 0, qOut[b], 0, qd);
				System.arraycopy(qkvHost[b], qd, kOut[b], 0, kv);
				System.arraycopy(qkvHost[b], qd + kv, vOut[b], 0, kv);
			}
		}

		private void download(ResidentActivation[] acts, float[][] a, float[][] b, float[][] c) {
			outs[0] = a;
			outs[1] = b;
			outs[2] = c;
			try {
				ResidentActivation.materializeAll(acts, outs);
			} finally {
				outs[0] = outs[1] = outs[2] = null;
			}
		}

		/** Waits for whatever was issued before a failure, so nothing queued outlives the lock. */
		private void drain() {
			try {
				chain.sync();
			} catch (RuntimeException ignored) {
				// the failure that brought us here is the one the caller sees
			}
		}

		private int requireWindow(int li, int rows) {
			if (freed)
				throw new IllegalStateException("prefill-window region window is closed");
			if (!eligible(li))
				throw new IllegalArgumentException("layer " + li + " does not run on the prefill-window region");
			if (rows <= 0 || rows > capacity)
				throw new IllegalArgumentException("window of " + rows + " rows in a region of capacity " + capacity);
			return rows;
		}

		/** Frees the buffers. Called under the serialization lock. */
		private void free() {
			if (freed)
				return;
			freed = true;
			chain.close();
			ctx.bindings().hostFree(tablesHost);
		}
	}
}
