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

import cab.ml.juno.kvcache.SessionKvTensor;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * The decode batch device region: N independent decode requests, each at its own
 * position and over its own KV mirror, through a transformer layer in one batched
 * step, on the device between the layer's one upload and one download.
 *
 * <pre>
 *   upload x (unless the previous layer left it) -> norm -> quantize -> W_q, W_k, W_v -> [bias | head norms]
 *       -> RoPE(q), RoPE(k) at each row's own position -> fp16(k, v) into each request's mirror
 *       -> attention -> W_o -> x += . -> norm -> W_gate, W_up -> SwiGLU -> W_down -> x += .
 *       -> download k, v and the layer output, one wait
 * </pre>
 *
 * <p>Every operation is what one request's decode region ({@link ResidentQkvPath}) runs,
 * fed the same bits, row by row: the projections are {@link BatchedDecodeKernels#gemvRows},
 * which reproduces the single-row GEMV's arithmetic and reads each weight once for the
 * whole batch; the norms, the quantization, SwiGLU and the residual adds are per-row and
 * per-element by construction; RoPE is the same rotation at each row's position; the
 * attention kernel takes one block per row, as for a single row, over that row's own
 * mirror. So every row of the result is bit for bit that request's single decode: a
 * request's reply does not depend on what it is batched with.
 *
 * <p>The mirror contract is the handler's, as in {@link ResidentQkvPath}: the caller
 * grows each mirror first ({@link DeviceKvCache#ensureCapacity}), so the region never
 * allocates; it writes each K and V row into the mirror but does not move the
 * watermark. The caller writes the returned K and V rows into its host KV tensors and
 * then marks the rows ({@link DeviceKvCache#markWritten}), so the mirror stays a copy
 * of the host KV.
 *
 * <p>Concurrency and memory. A call takes a region from a pool and returns it when the
 * step is done; regions hold the widest batch they have served. Device work is issued
 * under the context's serialization lock, like the single-request path's.
 */
final class ResidentBatchDecode implements AutoCloseable {

	private static final Logger log = Logger.getLogger(ResidentBatchDecode.class.getName());

	/** The attention table a batched launch reads per row: K pointer, V pointer, sequence length. */
	private static final long TABLE_BYTES_PER_ROW = 2 * ADDRESS.byteSize() + Integer.BYTES;

	/** Region capacities are rounded up to a multiple of this, so batches of similar widths share buffers. */
	private static final int CAPACITY_STEP = 8;

	private final ResidentQkvPath single;
	private final ResidentQkvPath.Parts p;
	private final ResidentLayerTail.Parts t;
	private final BatchedDecodeKernels batched;

	private final List<Region> regions = new CopyOnWriteArrayList<>();
	private final ConcurrentLinkedQueue<Region> idle = new ConcurrentLinkedQueue<>();
	private final AtomicBoolean mirrorGrowthWarned = new AtomicBoolean();
	private volatile boolean deviceFull;
	private volatile boolean closed;

	private ResidentBatchDecode(ResidentQkvPath single, ResidentQkvPath.Parts p, BatchedDecodeKernels batched) {
		this.single = single;
		this.p = p;
		this.t = p.tail().parts();
		this.batched = batched;
	}

	/**
	 * Builds the batch region over {@code single}'s weights, or returns {@code null}
	 * when it cannot run: no attention or tail inside the single region, or the batch
	 * kernels do not load.
	 */
	static ResidentBatchDecode create(ResidentQkvPath single) {
		ResidentQkvPath.Parts p = single.parts();
		if (p.tail() == null || p.attention() == null)
			return null;
		BatchedDecodeKernels batched = BatchedDecodeKernels.tryLoad();
		if (batched == null) {
			log.warning("the decode batch device region is unavailable (its kernels did not load);"
					+ " batched decode stays on the host-driven path");
			return null;
		}
		return new ResidentBatchDecode(single, p, batched);
	}

	/** Whether layer {@code li} (0-based within this shard) can run here at all. */
	boolean runsLayer(int li) {
		return single.runsWholeLayer(li);
	}

	/**
	 * Whether layer {@code li} can run here for these requests: the layer runs whole
	 * on the device, and every request's mirror is live, holds every position before
	 * its own and has room for it.
	 */
	boolean canRun(int li, int[] positions, DeviceKvCache[] mirrors) {
		if (!runsLayer(li))
			return false;
		for (int b = 0; b < positions.length; b++) {
			DeviceKvCache m = mirrors[b];
			if (m == null || !m.live() || !m.readableThrough(positions[b]) || m.capacityTokens() <= positions[b])
				return false;
		}
		return true;
	}

	/**
	 * Takes a region for {@code rows} rows, or returns {@code null} when none can be
	 * opened because the device is out of memory (logged once; the caller keeps the
	 * host-driven batch).
	 */
	Step open(int rows) {
		if (closed)
			throw new IllegalStateException("decode batch region is closed");
		if (rows <= 0)
			throw new IllegalArgumentException("rows must be positive: " + rows);
		Region r = idle.poll();
		while (r != null && r.capacity < rows) {
			retire(r);
			r = idle.poll();
		}
		if (r == null) {
			if (deviceFull)
				return null;
			int capacity = Math.max(CAPACITY_STEP, (rows + CAPACITY_STEP - 1) / CAPACITY_STEP * CAPACITY_STEP);
			try {
				synchronized (p.ctx().cublasSerializationLock()) {
					r = new Region(capacity);
				}
			} catch (IllegalStateException e) {
				if (!GpuLayerOffload.isVramOom(e))
					throw e;
				deviceFull = true;
				log.warning("out of device memory opening a decode batch region (" + e.getMessage()
						+ "); batched decode stays on the host-driven path from now on");
				return null;
			}
			regions.add(r);
		}
		return new Step(r);
	}

	/**
	 * Opens the batch region over {@code single} when it is requested and can run, and
	 * says once what was decided. Returns {@code null} (the host-driven batch) otherwise.
	 */
	static ResidentBatchDecode open(String handler, ResidentQkvPath single) {
		if (single == null)
			return null;
		if (!BatchDecodeRegionOptions.requested()) {
			log.info(handler + ": decode batch device region off (" + BatchDecodeRegionOptions.ENV_PROPERTY + "=off)");
			return null;
		}
		ResidentBatchDecode region = create(single);
		if (region == null) {
			log.info(handler + ": decode batches keep the host-driven path (the batch region needs the whole layer and"
					+ " attention on the device, and its kernels)");
			return null;
		}
		log.info(handler + ": a decode step of several requests (--parallel, continuous schedule) runs on the device,"
				+ " on the layers the single-request region runs whole; each row is bit for bit a single decode");
		return region;
	}

	/**
	 * Runs layer {@code li} for the batch when it can run here: grows each request's
	 * mirror first (a request whose mirror runs out of device memory has it retired,
	 * and the layer takes the host-driven path), runs the layer, and writes K and V to
	 * the host KV tensors before marking the mirrors. Returns whether it ran; when it
	 * returns {@code false} nothing has changed, {@code x} included, and the caller
	 * runs its host-driven layer.
	 */
	boolean runLayerIfEligible(Step step, int li, float[][] x, int[] positions, SessionKvTensor[] kLayers,
			SessionKvTensor[] vLayers, DeviceKvCache[] mirrors, float[][] kOut, float[][] vOut) {
		if (step == null || !runsLayer(li))
			return false;
		int n = x.length;
		for (int b = 0; b < n; b++) {
			DeviceKvCache m = mirrors[b];
			if (m == null || !m.live() || !m.readableThrough(positions[b]))
				continue;
			try {
				m.ensureCapacity(positions[b]);
			} catch (IllegalStateException ex) {
				if (!GpuLayerOffload.isVramOom(ex))
					throw ex;
				if (mirrorGrowthWarned.compareAndSet(false, true))
					log.warning("out of device memory growing a decode batch's KV mirror - that request attends on the CPU,"
							+ " which holds the same history. Lower --gpu-layers, or pass --gpu-attention off.");
				m.close();
			}
		}
		if (!canRun(li, positions, mirrors))
			return false;
		step.runLayer(li, x, positions, mirrors, kOut, vOut);
		for (int b = 0; b < n; b++) {
			kLayers[b].writeToken(positions[b], kOut[b]);
			vLayers[b].writeToken(positions[b], vOut[b]);
			mirrors[b].markWritten(positions[b], 1);
		}
		return true;
	}

	/** Device bytes held by every open region (the weights are the single path's). */
	long deviceBytes() {
		long total = 0;
		for (Region r : regions)
			total += r.chain.deviceBytes();
		return total;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		synchronized (p.ctx().cublasSerializationLock()) {
			for (Region r : regions)
				r.free();
			regions.clear();
			idle.clear();
		}
	}

	private void retire(Region r) {
		regions.remove(r);
		synchronized (p.ctx().cublasSerializationLock()) {
			r.free();
		}
	}

	/** One decode step over the layers it runs; closing it returns the region to the pool. */
	final class Step implements AutoCloseable {

		private final Region r;
		/** The layer whose input the device already holds, and for how many rows; -1 when it does not. */
		private int inputLayer = -1;
		private int inputRows;

		private Step(Region r) {
			this.r = r;
		}

		/**
		 * Runs layer {@code li} for the batch. {@code x} holds the layer's input rows and
		 * receives its output rows; {@code kOut} and {@code vOut} receive each row's K and
		 * V. The caller checked {@link #canRun}, writes K and V to its host KV tensors and
		 * then marks the mirrors.
		 */
		void runLayer(int li, float[][] x, int[] positions, DeviceKvCache[] mirrors, float[][] kOut, float[][] vOut) {
			int n = x.length;
			if (n > r.capacity)
				throw new IllegalArgumentException("batch of " + n + " rows in a region of capacity " + r.capacity);
			for (int pos : positions)
				p.rope().requirePosition(pos);
			synchronized (p.ctx().cublasSerializationLock()) {
				boolean done = false;
				try {
					issueLayer(li, x, positions, mirrors, n);
					r.outs[0] = kOut;
					r.outs[1] = vOut;
					r.outs[2] = x;
					try {
						ResidentActivation.materializeAll(r.kvx, r.outs);
					} finally {
						r.outs[0] = r.outs[1] = r.outs[2] = null;
					}
					done = true;
				} finally {
					if (!done)
						r.drain();
				}
			}
			inputLayer = li + 1;
			inputRows = n;
		}

		private void issueLayer(int li, float[][] x, int[] positions, DeviceKvCache[] mirrors, int n) {
			MemorySegment stream = r.chain.stream();
			DeviceSpanTimer spans = r.chain.spans();
			int h = p.hidden();
			int kv = p.kvDim();
			MemorySegment tables = writeTables(positions, mirrors, n, stream, spans);
			if (!(inputLayer == li && inputRows == n))
				r.x.upload(x, n);

			int mark = spans.begin(stream, n);
			if (!p.norm().normalizeResident(r.x, p.normWeight()[li], p.eps(), r.xn))
				throw new IllegalStateException("RMS-norm kernel failed to load after the region was built");
			spans.compute(DeviceComputeEvent.RMS_NORM, n, mark, stream);
			p.mmq().quantizeX(r.xn.devicePointer(), r.q8, n * h, stream);
			if (p.qNormWeight() != null && p.qNormWeight()[li] != null) {
				// Each head is normalized from scratch into place, out of place as in the single region.
				gemv(p.wq()[li], r.qIn, h, n);
				gemv(p.wk()[li], r.kIn, kv, n);
				headNorm(r.qIn, p.qNormWeight()[li], r.q.devicePointer(), n * p.numHeads(), n);
				headNorm(r.kIn, p.kNormWeight()[li], r.k.devicePointer(), n * kv / p.headDim(), n);
			} else {
				gemv(p.wq()[li], r.q.devicePointer(), h, n);
				gemv(p.wk()[li], r.k.devicePointer(), kv, n);
			}
			gemv(p.wv()[li], r.v.devicePointer(), kv, n);
			r.q.markWritten(n);
			r.k.markWritten(n);
			r.v.markWritten(n);
			if (p.qBias() != null && p.qBias()[li] != null) {
				p.biasKernels().addBias(r.q.devicePointer(), p.qBias()[li].devicePointer(), n, h, stream);
				p.biasKernels().addBias(r.k.devicePointer(), p.kBias()[li].devicePointer(), n, kv, stream);
				p.biasKernels().addBias(r.v.devicePointer(), p.vBias()[li].devicePointer(), n, kv, stream);
			}
			mark = spans.begin(stream, n);
			if (!p.rope().applyResidentRows(r.q, tables.asSlice(seqLensOffset(n)))
					|| !p.rope().applyResidentRows(r.k, tables.asSlice(seqLensOffset(n))))
				throw new IllegalStateException("RoPE kernel failed to load after the region was built");
			spans.compute(DeviceComputeEvent.ROPE, n, mark, stream);

			long ptrBytes = n * ADDRESS.byteSize();
			MemorySegment kPtrs = tables;
			MemorySegment vPtrs = tables.asSlice(ptrBytes);
			MemorySegment seqLens = tables.asSlice(2 * ptrBytes);
			mark = spans.begin(stream, n);
			batched.kvAppendRows(r.k.devicePointer(), r.v.devicePointer(), kPtrs, vPtrs, seqLens, n, kv, stream);
			spans.compute(DeviceComputeEvent.KV_APPEND, n, mark, stream);
			mark = spans.begin(stream, n);
			int[] windows = p.attentionWindows();
			p.attention().launch(r.q.devicePointer(), kPtrs, vPtrs, seqLens, r.attn.devicePointer(), n, p.numHeads(),
					p.gqaRatio(), p.headDim(), kv, GqaAttentionKernel.rowsPerBlock(false, n),
					windows == null ? 0 : windows[li], stream);
			spans.compute(DeviceComputeEvent.GQA_ATTENTION_REGION, n, mark, stream);
			issueTail(li, n);
		}

		/** W_o -> x += . -> norm -> W_gate, W_up -> SwiGLU -> W_down -> x += ., as {@link ResidentLayerTail#issue}. */
		private void issueTail(int li, int n) {
			MemorySegment stream = r.chain.stream();
			DeviceSpanTimer spans = r.chain.spans();
			int h = p.hidden();
			int inter = t.inter();
			project(t.wo()[li], r.attn.devicePointer(), r.attnProj, h, n, h);
			addInPlace(r.x.devicePointer(), r.attnProj, n, h);

			int mark = spans.begin(stream, n);
			if (!p.norm().normalizeResident(r.x, t.ffnNormWeight()[li], p.eps(), r.xn))
				throw new IllegalStateException("RMS-norm kernel failed to load after the region was built");
			spans.compute(DeviceComputeEvent.RMS_NORM, n, mark, stream);
			p.mmq().quantizeX(r.xn.devicePointer(), r.q8, n * h, stream);
			gemv(t.wGate()[li], r.gateUp, 2 * inter, n);
			gemv(t.wUp()[li], r.gateUp.asSlice((long) inter * Float.BYTES), 2 * inter, n);

			mark = spans.begin(stream, n);
			p.kernels().swiglu(r.gateUp, r.swiglu, n, inter, stream);
			spans.compute(DeviceComputeEvent.SWIGLU, n, mark, stream);
			project(t.wDown()[li], r.swiglu, r.out.devicePointer(), inter, n, h);

			mark = spans.begin(stream, n);
			p.kernels().residualAddBoth(r.x.devicePointer(), r.out.devicePointer(), (long) n * h, stream);
			spans.compute(DeviceComputeEvent.RESIDUAL_ADD, n, mark, stream);
			r.out.markWritten(n);
		}

		private void project(DeviceQ4KMatrix w, MemorySegment in, MemorySegment y, int inputCols, int n, int ldc) {
			p.mmq().quantizeX(in, r.q8, n * inputCols, r.chain.stream());
			gemv(w, y, ldc, n);
		}

		private void gemv(DeviceQ4KMatrix w, MemorySegment y, int ldc, int n) {
			MemorySegment stream = r.chain.stream();
			DeviceSpanTimer spans = r.chain.spans();
			int mark = spans.begin(stream, n);
			batched.gemvRows(w, r.q8, y, n, ldc, stream);
			spans.compute(DeviceComputeEvent.MMQ_PACKED, n, mark, stream);
		}

		private void headNorm(MemorySegment in, DeviceFloatMatrix weight, MemorySegment out, int headRows, int n) {
			MemorySegment stream = r.chain.stream();
			DeviceSpanTimer spans = r.chain.spans();
			RmsNormKernel kernel = RmsNormKernel.tryLoad();
			if (kernel == null)
				throw new IllegalStateException("RMS-norm kernel failed to load after the region was built");
			int mark = spans.begin(stream, n);
			kernel.launchResident(in, weight.devicePointer(), out, headRows, p.headDim(), p.eps(), stream);
			spans.compute(DeviceComputeEvent.RMS_NORM, n, mark, stream);
		}

		private void addInPlace(MemorySegment x, MemorySegment addend, int n, int h) {
			MemorySegment stream = r.chain.stream();
			DeviceSpanTimer spans = r.chain.spans();
			int mark = spans.begin(stream, n);
			p.kernels().addInPlace(x, addend, (long) n * h, stream);
			spans.compute(DeviceComputeEvent.RESIDUAL_ADD, n, mark, stream);
		}

		private long seqLensOffset(int n) {
			return 2 * n * ADDRESS.byteSize();
		}

		/**
		 * Writes the K pointers, V pointers and sequence lengths into the pinned host
		 * table and copies them to the device, before anything that reads them on the
		 * stream. Valid until the next layer's write (each layer ends with a wait).
		 */
		private MemorySegment writeTables(int[] positions, DeviceKvCache[] mirrors, int n, MemorySegment stream,
				DeviceSpanTimer spans) {
			long ptrBytes = n * ADDRESS.byteSize();
			for (int b = 0; b < n; b++) {
				r.tablesHost.setAtIndex(ADDRESS, b, mirrors[b].kPointer());
				r.tablesHost.setAtIndex(ADDRESS, n + b, mirrors[b].vPointer());
				r.tablesHost.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, positions[b] + 1);
			}
			long used = 2 * ptrBytes + (long) n * Integer.BYTES;
			int mark = spans.begin(stream, n);
			int rc;
			try {
				rc = (int) p.ctx().bindings().gpuMemcpyAsync().invokeExact(r.tables, r.tablesHost, used, GpuBindings.H2D,
						stream);
			} catch (Throwable th) {
				throw new IllegalStateException("memcpy(batch decode attention tables H2D): native call failed", th);
			}
			GpuBindings.check(rc, "memcpy(batch decode attention tables H2D)");
			spans.staging(GpuBindings.H2D, used, n, "memcpy(batch decode attention tables H2D)", mark, stream);
			return r.tables;
		}

		@Override
		public void close() {
			if (closed) {
				synchronized (p.ctx().cublasSerializationLock()) {
					r.free();
				}
				return;
			}
			idle.offer(r);
		}
	}

	/** One step's device buffers: allocated once, reused by every later step that takes them from the pool. */
	private final class Region {

		final int capacity;
		final ResidentChain chain;
		final ResidentActivation x;
		final ResidentActivation xn;
		final ResidentActivation q;
		final ResidentActivation k;
		final ResidentActivation v;
		final ResidentActivation attn;
		final ResidentActivation out;
		final ResidentActivation[] kvx;
		final float[][][] outs = new float[3][][];
		final MemorySegment q8;
		final MemorySegment attnProj;
		final MemorySegment gateUp;
		final MemorySegment swiglu;
		/** Projected q and k before their per-head norms, null without head norms. */
		final MemorySegment qIn;
		final MemorySegment kIn;
		final MemorySegment tables;
		final MemorySegment tablesHost;
		private boolean freed;

		Region(int capacity) {
			this.capacity = capacity;
			ResidentChain c = ResidentChain.open(p.ctx());
			MemorySegment host = null;
			try {
				int h = p.hidden();
				int kv = p.kvDim();
				int inter = t.inter();
				this.x = c.allocate(capacity, h, "decode batch residual");
				this.xn = c.allocate(capacity, h);
				this.q = c.allocate(capacity, h, "decode batch q");
				this.k = c.allocate(capacity, kv, "decode batch k");
				this.v = c.allocate(capacity, kv, "decode batch v");
				this.attn = c.allocate(capacity, h, "decode batch attention");
				this.out = c.allocate(capacity, h, "decode batch layer output");
				this.q8 = c.allocateScratch((long) capacity * Q4KMmqKernel.q8Bytes(Math.max(h, inter)));
				this.attnProj = c.allocateScratch((long) capacity * h * Float.BYTES);
				this.gateUp = c.allocateScratch((long) capacity * 2 * inter * Float.BYTES);
				this.swiglu = c.allocateScratch((long) capacity * inter * Float.BYTES);
				boolean headNorms = p.qNormWeight() != null;
				this.qIn = headNorms ? c.allocateScratch((long) capacity * h * Float.BYTES) : null;
				this.kIn = headNorms ? c.allocateScratch((long) capacity * kv * Float.BYTES) : null;
				long tableBytes = capacity * TABLE_BYTES_PER_ROW;
				this.tables = c.allocateScratch(tableBytes);
				host = p.ctx().bindings().hostMalloc(p.ctx().deviceIndex(), tableBytes);
				this.tablesHost = host.reinterpret(tableBytes);
			} catch (RuntimeException e) {
				if (host != null)
					p.ctx().bindings().hostFree(host);
				c.close();
				throw e;
			}
			this.chain = c;
			this.kvx = new ResidentActivation[] { k, v, out };
		}

		/** Waits for whatever was issued before a failure, so nothing queued outlives the lock. */
		void drain() {
			try {
				chain.sync();
			} catch (RuntimeException ignored) {
				// the failure that brought us here is the one the caller sees
			}
		}

		void free() {
			if (freed)
				return;
			freed = true;
			chain.close();
			p.ctx().bindings().hostFree(tablesHost);
		}
	}
}
