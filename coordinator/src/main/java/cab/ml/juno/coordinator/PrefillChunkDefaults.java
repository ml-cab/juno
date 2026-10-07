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

import java.util.List;
import java.util.function.IntToLongFunction;
import java.util.function.LongSupplier;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.GpuContext;

/**
 * Default prefill chunk size ({@code --prefill-batch}) per launch surface, so every
 * entry point that builds a {@link GenerationLoop} resolves it in one place.
 *
 * <p>An explicit value (CLI, builder or {@code JUNO_PREFILL_BATCH}) wins on every
 * surface. Without one, the in-process surfaces on a GPU with the {@code static}
 * schedule size the chunk from free device memory and the prefill-window footprint
 * their shards report ({@link #resolve(Surface, Integer, boolean, GpuContext, List)}):
 * the widest window that fits half of what is free, which normally covers the whole
 * prompt in one window; everything else uses the fixed
 * {@value PrefillBatchOptions#DEFAULT_CHUNK_SIZE}. Each fixed default rests on a
 * measurement of window width on that surface (512-token prompt, TinyLlama):
 * <ul>
 * <li>CPU backend: 32, 128 and 512 within 1% of each other; a wider window only
 * holds more activation memory.</li>
 * <li>{@code continuous} schedule: the chunk is the unit decode steps interleave
 * with. 32 gives concurrent short requests the lowest time to first token; a
 * wider chunk shortens the long prompt's prefill at their expense.</li>
 * <li>Cluster REPL and standalone coordinator: the gRPC pipeline clients prefill
 * one token per call, so the chunk size does not change the work done.</li>
 * <li>LoRA training REPL: its handler's prefill does not speed up with window
 * width (1.01x at 512), and the REPL has no {@link GpuContext} to size from.</li>
 * </ul>
 */
public final class PrefillChunkDefaults {

	/** Entry points that construct a {@link GenerationLoop}. */
	public enum Surface {
		/** {@code juno local}: in-process shards. */
		LOCAL_REPL,
		/** JVM embedding facade ({@code JunoPlayer}): in-process shards. */
		EMBEDDED,
		/** {@code juno cluster}: forked node JVMs over gRPC. */
		CLUSTER_REPL,
		/** Standalone coordinator ({@code juno-master}) in front of remote nodes. */
		COORDINATOR,
		/** {@code juno lora}: chat inside the training REPL. */
		LORA_REPL
	}

	private PrefillChunkDefaults() {
	}

	/**
	 * @param cliArg         explicit chunk size, or null for the surface default
	 * @param staticSchedule true for the {@code static} serving schedule
	 * @param gpuCtx         the GPU context prefill runs on, or null for CPU-only
	 */
	public static int resolve(Surface surface, Integer cliArg, boolean staticSchedule, GpuContext gpuCtx) {
		return resolveFrom(surface, cliArg, staticSchedule, gpuCtx == null ? null : gpuCtx::freeVramBytes);
	}

	/**
	 * As {@link #resolve(Surface, Integer, boolean, GpuContext)}, sizing the adaptive
	 * chunk from the prefill-window footprint {@code handlers} report: every
	 * in-process shard keeps its own window on the device, so their footprints add up.
	 * The memory they keep free for their KV mirrors is not offered to the window.
	 */
	public static int resolve(Surface surface, Integer cliArg, boolean staticSchedule, GpuContext gpuCtx,
			List<? extends ForwardPassHandler> handlers) {
		return resolveFrom(surface, cliArg, staticSchedule, gpuCtx == null ? null : gpuCtx::freeVramBytes,
				windowBytesOf(handlers), mirrorReserveOf(handlers));
	}

	/**
	 * The summed prefill-window footprint of {@code handlers}, by window rows, or
	 * null when none of them runs its prefill windows on the device region.
	 */
	static IntToLongFunction windowBytesOf(List<? extends ForwardPassHandler> handlers) {
		List<ForwardPassHandler> regions = handlers.stream()
				.filter(h -> h.prefillWindowDeviceBytes(PrefillBatchOptions.DEFAULT_CHUNK_SIZE) > 0)
				.map(h -> (ForwardPassHandler) h).toList();
		if (regions.isEmpty())
			return null;
		return rows -> {
			long total = 0;
			for (ForwardPassHandler h : regions)
				total += h.prefillWindowDeviceBytes(rows);
			return total;
		};
	}

	static int resolveFrom(Surface surface, Integer cliArg, boolean staticSchedule, LongSupplier freeVramBytes,
			IntToLongFunction windowBytes) {
		if (sizesFromDeviceMemory(surface) && staticSchedule)
			return PrefillBatchOptions.resolveAdaptiveFrom(cliArg, freeVramBytes, windowBytes).chunkSize();
		return PrefillBatchOptions.resolve(cliArg).chunkSize();
	}

	/**
	 * As {@link #resolveFrom(Surface, Integer, boolean, LongSupplier, IntToLongFunction)},
	 * with {@code reservedBytes} of the free memory held for the shards' KV mirrors and
	 * not offered to the window. A failed free-memory query (0) stays a failed query.
	 */
	static int resolveFrom(Surface surface, Integer cliArg, boolean staticSchedule, LongSupplier freeVramBytes,
			IntToLongFunction windowBytes, long reservedBytes) {
		LongSupplier offered = freeVramBytes == null || reservedBytes <= 0 ? freeVramBytes : () -> {
			long free = freeVramBytes.getAsLong();
			return free <= 0 ? free : Math.max(1L, free - reservedBytes);
		};
		return resolveFrom(surface, cliArg, staticSchedule, offered, windowBytes);
	}

	/** The KV-mirror reserve of {@code handlers}, summed over every shard on the device. */
	static long mirrorReserveOf(List<? extends ForwardPassHandler> handlers) {
		long total = 0;
		for (ForwardPassHandler h : handlers)
			total += h.kvMirrorReserveDeviceBytes();
		return total;
	}

	static int resolveFrom(Surface surface, Integer cliArg, boolean staticSchedule, LongSupplier freeVramBytes) {
		return resolveFrom(surface, cliArg, staticSchedule, freeVramBytes, null);
	}

	private static boolean sizesFromDeviceMemory(Surface surface) {
		return surface == Surface.LOCAL_REPL || surface == Surface.EMBEDDED;
	}
}
