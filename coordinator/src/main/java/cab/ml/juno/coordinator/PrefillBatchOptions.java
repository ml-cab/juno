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
package cab.ml.juno.coordinator;

import java.util.function.IntToLongFunction;
import java.util.function.LongSupplier;

import cab.ml.juno.node.GpuContext;

/**
 * CLI / env resolution for prefill microbatch chunk size ({@code --prefill-batch},
 * {@code JUNO_PREFILL_BATCH}).
 *
 * <p>Chunk size applies only when {@link PrefillMode#BATCHED} is active.
 * {@code 1} processes one prompt token per {@code prefillBatch} call (sequential
 * batched path). Values {@code >= 2} split long prefills into GPU-friendly windows.
 */
public final class PrefillBatchOptions {

	public static final String ENV_PREFILL_BATCH = "JUNO_PREFILL_BATCH";
	public static final int DEFAULT_CHUNK_SIZE = 32;

	/**
	 * Fraction of currently-free device memory (post model/KV-pool residency)
	 * {@link #resolveAdaptive} is willing to spend on prefill scratch when sizing
	 * the chunk to cover a whole prompt in one window.
	 */
	static final double ADAPTIVE_HEADROOM_FRACTION = 0.5;

	/**
	 * Scratch bytes per prefill token of the host-staged batched matmul, used only
	 * when no handler reports a prefill-window footprint (its windows do not run on
	 * the device region): the largest FFN gate/up projection of the sweep models
	 * (mistral-7b, hidden=4096 to intermediate=14336) stages {@code cols(hidden)} FP16
	 * x-elements plus {@code rows(intermediate)} FP32 y-elements per batched token
	 * ({@code 4096*2 + 14336*4 = 65536} bytes/token). A window on the device region
	 * costs several times this (114 to 412 KB a row across the sweep models and a 30B
	 * Llama, before attention scores), which is why a reported footprint replaces it.
	 */
	static final long ADAPTIVE_BYTES_PER_TOKEN = 65536L;

	/** Hard ceiling on the adaptive chunk size, regardless of free VRAM. */
	static final int ADAPTIVE_CHUNK_CEILING = 65536;

	private final int chunkSize;

	private PrefillBatchOptions(int chunkSize) {
		this.chunkSize = chunkSize;
	}

	public static PrefillBatchOptions defaults() {
		return new PrefillBatchOptions(DEFAULT_CHUNK_SIZE);
	}

	/**
	 * Resolve from optional CLI override, then env, then default ({@value #DEFAULT_CHUNK_SIZE}).
	 *
	 * @param cliArg CLI {@code --prefill-batch} or null
	 */
	public static PrefillBatchOptions resolve(Integer cliArg) {
		if (cliArg != null)
			return of(cliArg);
		String env = env(ENV_PREFILL_BATCH);
		if (env != null)
			return of(parsePositiveInt(env));
		return defaults();
	}

	/**
	 * Resolve from optional CLI override, then env, then an adaptive default that
	 * sizes the chunk to cover a whole prompt in one window when {@code gpuCtx} has
	 * enough free VRAM headroom, falling back to {@value #DEFAULT_CHUNK_SIZE} on the
	 * CPU-only path ({@code gpuCtx == null}) or when the live VRAM query fails.
	 *
	 * <p>The adaptive floor is {@value #DEFAULT_CHUNK_SIZE} — this can only size the
	 * chunk as large or larger than today's fixed default, never smaller, so it
	 * cannot regress a configuration that already works.
	 *
	 * @param cliArg CLI {@code --prefill-batch} or null
	 * @param gpuCtx the GPU context prefill will run on, or null for CPU-only
	 */
	public static PrefillBatchOptions resolveAdaptive(Integer cliArg, GpuContext gpuCtx) {
		return resolveAdaptiveFrom(cliArg, gpuCtx == null ? null : gpuCtx::freeVramBytes);
	}

	/**
	 * As {@link #resolveAdaptive(Integer, GpuContext)}, with the live free-VRAM query
	 * passed as a supplier ({@code null} for CPU-only) so callers without a
	 * {@link GpuContext} instance, and tests, can drive it.
	 */
	static PrefillBatchOptions resolveAdaptiveFrom(Integer cliArg, LongSupplier freeVramBytes) {
		return resolveAdaptiveFrom(cliArg, freeVramBytes, null);
	}

	/**
	 * As {@link #resolveAdaptiveFrom(Integer, LongSupplier)}, sizing the adaptive
	 * chunk from {@code windowBytes} when given ({@link #adaptiveChunkSize(long, IntToLongFunction)}).
	 */
	static PrefillBatchOptions resolveAdaptiveFrom(Integer cliArg, LongSupplier freeVramBytes,
			IntToLongFunction windowBytes) {
		if (cliArg != null)
			return of(cliArg);
		String env = env(ENV_PREFILL_BATCH);
		if (env != null)
			return of(parsePositiveInt(env));
		if (freeVramBytes != null) {
			long freeBytes = freeVramBytes.getAsLong();
			if (freeBytes > 0)
				return of(adaptiveChunkSize(freeBytes, windowBytes));
		}
		return defaults();
	}

	/**
	 * The widest window, between {@value #DEFAULT_CHUNK_SIZE} and
	 * {@link #ADAPTIVE_CHUNK_CEILING} rows, whose device footprint fits the headroom.
	 *
	 * @param windowBytes device bytes all prefill regions on the device hold for a
	 *                    window of the given rows (non-decreasing in rows), or null;
	 *                    null or zero at the floor means no region, and the per-token
	 *                    figure {@link #ADAPTIVE_BYTES_PER_TOKEN} applies
	 */
	static int adaptiveChunkSize(long freeVramBytes, IntToLongFunction windowBytes) {
		if (windowBytes == null || windowBytes.applyAsLong(DEFAULT_CHUNK_SIZE) <= 0)
			return adaptiveChunkSize(freeVramBytes);
		long headroomBytes = (long) (freeVramBytes * ADAPTIVE_HEADROOM_FRACTION);
		int lo = DEFAULT_CHUNK_SIZE;
		int hi = ADAPTIVE_CHUNK_CEILING;
		if (windowBytes.applyAsLong(lo) > headroomBytes)
			return lo;
		while (lo < hi) {
			int mid = lo + (hi - lo + 1) / 2;
			if (windowBytes.applyAsLong(mid) <= headroomBytes)
				lo = mid;
			else
				hi = mid - 1;
		}
		return lo;
	}

	static int adaptiveChunkSize(long freeVramBytes) {
		long headroomBytes = (long) (freeVramBytes * ADAPTIVE_HEADROOM_FRACTION);
		long chunk = headroomBytes / ADAPTIVE_BYTES_PER_TOKEN;
		return (int) Math.max(DEFAULT_CHUNK_SIZE, Math.min(ADAPTIVE_CHUNK_CEILING, chunk));
	}

	public static PrefillBatchOptions of(int chunkSize) {
		if (chunkSize < 1)
			throw new IllegalArgumentException("prefill-batch must be >= 1, got: " + chunkSize);
		return new PrefillBatchOptions(chunkSize);
	}

	public int chunkSize() {
		return chunkSize;
	}

	private static String env(String key) {
		String v = System.getenv(key);
		return (v == null || v.isBlank()) ? null : v.strip();
	}

	private static int parsePositiveInt(String raw) {
		try {
			int n = Integer.parseInt(raw);
			if (n < 1)
				throw new IllegalArgumentException(ENV_PREFILL_BATCH + " must be >= 1, got: " + n);
			return n;
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("invalid " + ENV_PREFILL_BATCH + ": " + raw);
		}
	}
}
