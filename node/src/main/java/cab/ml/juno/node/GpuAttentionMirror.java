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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * The GPU-resident attention kernel ({@link CudaGqaAttention}) and the per-request
 * device KV mirrors it reads, for a handler that keeps its own host KV tensors and
 * attention call sites (Phi-3, Qwen3). The handler asks this class at each of its
 * three call sites (prefill window, single-token decode, multi-stream decode) and
 * falls back to its own scalar attention whenever the answer is {@code false}.
 *
 * <p>The contract is the one {@code LlamaTransformerHandler} established:
 * <ul>
 * <li>The host KV tensors are written first and always. A mirror is a copy of
 * them, so it can be given up at any point without losing history.</li>
 * <li>A mirror is read only through its written-prefix watermark
 * ({@link DeviceKvCache#readableThrough}). Device memory is not zeroed, so a mirror
 * that is short of the host tensors (restored prefix, retired mid-request) is never
 * handed to the kernel.</li>
 * <li>Running out of device memory while allocating, growing or attending retires
 * the affected mirrors by closing them in place, never by unmapping them, so the
 * next token finds a closed mirror rather than a fresh empty one. The request
 * continues on the CPU, and each kind of fallback is logged once.</li>
 * </ul>
 */
final class GpuAttentionMirror {

	private static final Logger log = Logger.getLogger(GpuAttentionMirror.class.getName());

	/**
	 * Sees every kernel launch that succeeded, with the mirrors it read, its
	 * queries, lengths and output, so a test can hold the kernel at a real call
	 * site to the scalar oracle over the same FP16 rows. Tests only; {@code null}
	 * otherwise, which costs one volatile read per launch.
	 */
	interface DispatchObserver {
		void dispatched(DeviceKvCache[] mirrors, float[][] q, int[] seqLens, float[][] out, int numHeads,
				int headDim, int gqaRatio, int kvDim);
	}

	static volatile DispatchObserver observer;

	private final CudaGqaAttention gqa;
	private final String handler;
	private final int layers;
	private final int kvDim;
	private final int numHeads;
	private final int headDim;
	private final int gqaRatio;
	private final Map<String, DeviceKvCache[]> byRequest = new ConcurrentHashMap<>();
	private final AtomicBoolean allocWarned = new AtomicBoolean();
	private final AtomicBoolean growthWarned = new AtomicBoolean();
	private final AtomicBoolean kernelWarned = new AtomicBoolean();

	private GpuAttentionMirror(CudaGqaAttention gqa, String handler, int layers, int kvDim, int numHeads,
			int headDim, int gqaRatio) {
		this.gqa = gqa;
		this.handler = handler;
		this.layers = layers;
		this.kvDim = kvDim;
		this.numHeads = numHeads;
		this.headDim = headDim;
		this.gqaRatio = gqaRatio;
	}

	/**
	 * Opens the kernel path for a handler, or returns {@code null} when it does not
	 * apply: the CPU backend, {@code --gpu-attention off}, or a GPU backend without
	 * the kernel (which is announced once).
	 *
	 * @param handler label for log lines, e.g. {@code "Phi-3"}
	 * @param layers  layers this shard owns
	 */
	static GpuAttentionMirror open(MatVec backend, String handler, int layers, int kvDim, int numHeads,
			int headDim, int gqaRatio) {
		if (!(backend instanceof GpuMatVec gpu))
			return null;
		GpuAttentionOptions opts = GpuAttentionOptions.fromEnv();
		if (opts.mode() == GpuAttentionOptions.Mode.OFF)
			return null;
		CudaGqaAttention gqa = CudaGqaAttention.tryCreate(gpu.gpuContext());
		if (gqa == null) {
			GpuAttentionSupport.announceBackendWithoutKernel(gpu.gpuContext().backendLabel());
			return null;
		}
		log.info(handler + ": GPU-resident attention path active (gpu-attention=" + opts.policyLabel() + ")");
		return new GpuAttentionMirror(gqa, handler, layers, kvDim, numHeads, headDim, gqaRatio);
	}

	/**
	 * The request's mirrors, one per layer, allocated on first use; {@code null}
	 * when that allocation ran out of device memory, in which case the request runs
	 * attention on the CPU. Nothing has been written to a mirror at that point, so
	 * no history is lost, and a later request may still get one.
	 */
	DeviceKvCache[] layersFor(String requestId) {
		try {
			return byRequest.computeIfAbsent(requestId, k -> gqa.newLayers(layers, kvDim));
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			if (allocWarned.compareAndSet(false, true))
				log.warning(handler + ": out of device memory allocating the attention KV mirror"
						+ " - this request's attention runs on the CPU. Lower --gpu-layers, or pass"
						+ " --gpu-attention off.");
			return null;
		}
	}

	/**
	 * Layer {@code li}'s mirror when it may be written this call: the request has
	 * mirrors, the layer's weights are on the device, and the mirror is still live.
	 */
	static DeviceKvCache layer(DeviceKvCache[] mirrors, int li, boolean deviceLayer) {
		return mirrors != null && deviceLayer && mirrors[li].live() ? mirrors[li] : null;
	}

	/**
	 * Appends one K/V row. Returns the mirror, or {@code null} after retiring it
	 * because growing it ran out of device memory.
	 *
	 * @param width rows in the forward call that issued the write
	 */
	DeviceKvCache append(DeviceKvCache mirror, int pos, float[] k, float[] v, int width) {
		if (mirror == null)
			return null;
		try {
			mirror.appendToken(pos, k, v, width);
			return mirror;
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			if (growthWarned.compareAndSet(false, true))
				log.warning(handler + ": out of device memory growing the attention KV mirror - attention"
						+ " continues on the CPU, which holds the same history. Lower --gpu-layers, or pass"
						+ " --gpu-attention off, to keep it on the GPU.");
			mirror.close();
			return null;
		}
	}

	/**
	 * Grows {@code mirror} to hold position {@code pos} before the decode residency
	 * region casts the row into it ({@link ResidentQkvPath} never grows a mirror, so
	 * running out of device memory cannot happen inside the region). Same contract as
	 * {@link #append}: returns the mirror, or {@code null} once it has been retired for
	 * running out of device memory, before any of the layer's device work is issued.
	 */
	DeviceKvCache reserve(DeviceKvCache mirror, int pos) {
		if (mirror == null)
			return null;
		try {
			mirror.ensureCapacity(pos);
			return mirror;
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			if (growthWarned.compareAndSet(false, true))
				log.warning(handler + ": out of device memory growing the attention KV mirror - attention"
						+ " continues on the CPU, which holds the same history. Lower --gpu-layers, or pass"
						+ " --gpu-attention off, to keep it on the GPU.");
			mirror.close();
			return null;
		}
	}

	/**
	 * Appends a prefill window's K and V rows ({@code width} of them, from
	 * {@code startPos}) as one copy per tensor, after the host KV tensors hold them.
	 * Same contract as {@link #append}: returns the mirror, or {@code null} once it has
	 * been retired for running out of device memory.
	 */
	DeviceKvCache appendWindow(DeviceKvCache mirror, int startPos, float[][] k, float[][] v, int width) {
		if (mirror == null)
			return null;
		try {
			mirror.appendWindow(startPos, k, v, width);
			return mirror;
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			if (growthWarned.compareAndSet(false, true))
				log.warning(handler + ": out of device memory growing the attention KV mirror - attention"
						+ " continues on the CPU, which holds the same history. Lower --gpu-layers, or pass"
						+ " --gpu-attention off, to keep it on the GPU.");
			mirror.close();
			return null;
		}
	}

	/**
	 * Attention for a prefill window: row {@code b} of {@code q} sits at position
	 * {@code startPos + b} and attends over positions {@code [0, startPos + b]}.
	 *
	 * @return {@code true} when the kernel wrote every row of {@code out}; on
	 *         {@code false} nothing usable was written and the caller runs its own path
	 */
	boolean attendWindow(DeviceKvCache mirror, int startPos, float[][] q, float[][] out) {
		int w = q.length;
		if (mirror == null || !mirror.readableThrough(startPos + w))
			return false;
		int[] seqLens = new int[w];
		DeviceKvCache[] perRow = new DeviceKvCache[w];
		for (int b = 0; b < w; b++) {
			seqLens[b] = startPos + b + 1;
			perRow[b] = mirror;
		}
		return dispatch(perRow, q, seqLens, out);
	}

	/** Attention for one decode row at {@code pos}, written into {@code out}. */
	boolean attendOne(DeviceKvCache mirror, int pos, float[] q, float[] out) {
		if (mirror == null || !mirror.readableThrough(pos + 1))
			return false;
		return dispatch(new DeviceKvCache[] { mirror }, new float[][] { q }, new int[] { pos + 1 },
				new float[][] { out });
	}

	/**
	 * Attention for independent decode streams in one launch; stream {@code b} sits
	 * at {@code positions[b]} and reads {@code mirrors[b]}. Uses the kernel only when
	 * every stream's mirror holds its whole history.
	 */
	boolean attendStreams(DeviceKvCache[] mirrors, int[] positions, float[][] q, float[][] out) {
		int n = q.length;
		int[] seqLens = new int[n];
		for (int b = 0; b < n; b++) {
			if (mirrors[b] == null || !mirrors[b].readableThrough(positions[b] + 1))
				return false;
			seqLens[b] = positions[b] + 1;
		}
		return dispatch(mirrors, q, seqLens, out);
	}

	/**
	 * Guarded at the dispatch boundary: the kernel allocates scratch of its own,
	 * and any of it can fail on a full card. The launch covers every row at once,
	 * so which mirror it ran short on is not recoverable; retire all of them.
	 * Over-retiring costs throughput, under-retiring would leave one read past its
	 * written prefix.
	 */
	private boolean dispatch(DeviceKvCache[] mirrors, float[][] q, int[] seqLens, float[][] out) {
		try {
			boolean ok = gqa.attendBatched(mirrors, q, seqLens, out, numHeads, headDim, gqaRatio, kvDim);
			DispatchObserver o = observer;
			if (ok && o != null)
				o.dispatched(mirrors, q, seqLens, out, numHeads, headDim, gqaRatio, kvDim);
			return ok;
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			if (kernelWarned.compareAndSet(false, true))
				log.warning(handler + ": out of device memory in the GPU attention kernel - attention continues"
						+ " on the CPU, which holds the same history. Lower --gpu-layers, or pass"
						+ " --gpu-attention off.");
			for (DeviceKvCache m : mirrors)
				if (m != null)
					m.close();
			return false;
		}
	}

	/** Tests only: each layer's watermark, -1 where closed; {@code null} without mirrors. */
	int[] watermarks(String requestId) {
		DeviceKvCache[] mirrors = byRequest.get(requestId);
		if (mirrors == null)
			return null;
		int[] out = new int[mirrors.length];
		for (int i = 0; i < mirrors.length; i++)
			out[i] = mirrors[i] == null || !mirrors[i].live() ? -1 : mirrors[i].validTokens();
		return out;
	}

	/** The request's mirrors if it has any; never allocates. */
	DeviceKvCache[] existing(String requestId) {
		return byRequest.get(requestId);
	}

	/**
	 * Closes the request's mirrors but keeps them mapped, so the request goes on
	 * attending on the host instead of being handed fresh, unwritten mirrors.
	 */
	void retire(String requestId) {
		DeviceKvCache[] mirrors = byRequest.get(requestId);
		if (mirrors != null)
			for (DeviceKvCache m : mirrors)
				if (m != null)
					m.close();
	}

	/** Frees the request's mirrors. Safe for a request that never had any. */
	void evict(String requestId) {
		DeviceKvCache[] mirrors = byRequest.remove(requestId);
		if (mirrors != null)
			for (DeviceKvCache m : mirrors)
				m.close();
	}

	/** Frees every request's mirrors and the kernel's pooled scratch. */
	void close() {
		for (String id : byRequest.keySet())
			evict(id);
		gqa.close();
	}
}
