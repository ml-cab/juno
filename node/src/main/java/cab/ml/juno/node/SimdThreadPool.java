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

import java.util.function.IntConsumer;

/**
 * Row-parallel dispatch for the float weight-stationary CPU quantized matmul
 * kernels ({@link LlamaTransformerHandler#sgemmQ4KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ5KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ8_0WeightStationary}), which run when the
 * integer kernels do not ({@link CpuKernelSelection}).
 *
 * <p>{@link #forEachRow} runs on {@link CpuThreads}, the same thread pool and
 * thread count ({@code --threads}) as every other CPU matmul. The former
 * {@code -Djuno.simd.pool.size} property built a pool the hot path never used;
 * it is no longer read, and the launchers refuse it with an error naming
 * {@code --threads}.
 *
 * <p>This is not core-affinity pinning. Pure Java has no portable way to bind a
 * thread to a specific physical core without native code.
 */
public final class SimdThreadPool {

	/** The removed property; see {@link CpuThreads}. */
	public static final String REMOVED_POOL_SIZE_PROPERTY = "juno.simd.pool.size";

	private SimdThreadPool() {
	}

	/**
	 * Runs {@code body} once for every {@code r} in {@code [0, rows)}, and
	 * blocks until all of them complete. Any exception thrown by {@code body}
	 * propagates out of this call.
	 */
	static void forEachRow(int rows, IntConsumer body) {
		if (rows <= 0)
			return;
		CpuThreads.forEach(rows, body);
	}

	/**
	 * One-line summary of CPU matmul threading, for a single startup log line
	 * alongside {@link VectorQuantKernels#diagnosticSummary}.
	 */
	public static String diagnosticSummary() {
		return "CPU matmul threads: " + CpuThreads.threads() + " (--threads, -D" + CpuThreads.PROPERTY + " or "
				+ CpuThreads.ENV + "; availableProcessors()=" + Runtime.getRuntime().availableProcessors() + ")";
	}
}
