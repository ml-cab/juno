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

import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Dedicated {@link ForkJoinPool} for the row-parallel loop inside the SIMD
 * quantized-matmul kernels ({@link LlamaTransformerHandler#sgemmQ4KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ5KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ8_0WeightStationary}), replacing the
 * previous {@code IntStream.range(rows).parallel()} dispatch to
 * {@code ForkJoinPool.commonPool()}.
 *
 * <p>
 * This exists to let the parallelism level of that one specific hot loop be
 * tuned independently of the rest of the application, via the
 * {@code juno.simd.pool.size} system property, without touching every other
 * parallel stream in the codebase that still uses the common pool.
 *
 * <p>
 * This is not core-affinity pinning. Pure Java has no portable way to bind a
 * thread to a specific physical core without native code, and this class
 * does not attempt to distinguish performance cores from efficiency cores on
 * hybrid CPUs (e.g. Intel Alder Lake and later). What sizing the pool below
 * the total logical CPU count does is coarser and indirect: keeping the
 * number of runnable worker threads for this loop at or below the number of
 * performance cores makes it less likely the OS scheduler ends up placing
 * some of that work on slower efficiency cores under load, though the OS
 * still makes the actual placement decision. This is offered as a cheap way
 * to test that hypothesis by comparing runs at different pool sizes, not as
 * a guaranteed fix.
 *
 * <p>
 * Usage: {@code -Djuno.simd.pool.size=8} (e.g. matching the logical
 * performance-core count on a specific CPU) to override; unset or an invalid
 * value falls back to {@link Runtime#availableProcessors()}, which is the
 * same parallelism {@code ForkJoinPool.commonPool()} would have used, so the
 * default behavior of the three kernels above is unchanged.
 */
public final class SimdThreadPool {

	private static final Logger log = Logger.getLogger(SimdThreadPool.class.getName());

	private static final String POOL_SIZE_PROPERTY = "juno.simd.pool.size";

	static final ForkJoinPool POOL = build();

	private SimdThreadPool() {
	}

	private static ForkJoinPool build() {
		int fallback = Runtime.getRuntime().availableProcessors();
		int parallelism = fallback;
		String configured = System.getProperty(POOL_SIZE_PROPERTY);
		String sourceNote = "default, matches Runtime.getRuntime().availableProcessors()";

		if (configured != null) {
			try {
				int requested = Integer.parseInt(configured.trim());
				if (requested > 0) {
					parallelism = requested;
					sourceNote = "from -D" + POOL_SIZE_PROPERTY + "=" + configured;
				} else {
					log.log(Level.WARNING, POOL_SIZE_PROPERTY + "=" + configured
							+ " must be positive; ignoring and using " + fallback);
				}
			} catch (NumberFormatException e) {
				log.log(Level.WARNING, POOL_SIZE_PROPERTY + "=" + configured
						+ " is not a valid integer; ignoring and using " + fallback);
			}
		}

		log.log(Level.INFO, "SIMD row-parallel pool: parallelism=" + parallelism + " (" + sourceNote + ")");
		return new ForkJoinPool(parallelism);
	}

	/**
	 * Runs {@code body} once for every {@code r} in {@code [0, rows)} on this
	 * dedicated pool, and blocks until all of them complete. Equivalent to
	 * {@code IntStream.range(0, rows).parallel().forEach(body)} except that
	 * it dispatches to {@link #POOL} instead of {@code ForkJoinPool.commonPool()}.
	 *
	 * <p>
	 * Any exception thrown by {@code body} on a worker thread propagates out
	 * of this call as an unchecked exception (via
	 * {@link java.util.concurrent.ForkJoinTask#join()}), the same failure
	 * behavior as the {@code IntStream.parallel().forEach()} call this
	 * replaces.
	 */
	static void forEachRow(int rows, IntConsumer body) {
		POOL.submit(() -> java.util.stream.IntStream.range(0, rows).parallel().forEach(body)).join();
	}

	/**
	 * One-line summary of the current pool configuration, for a single
	 * startup log line alongside {@link VectorQuantKernels#diagnosticSummary}.
	 */
	public static String diagnosticSummary() {
		return "SIMD row-parallel pool: parallelism=" + POOL.getParallelism() + " (set -D" + POOL_SIZE_PROPERTY
				+ "=N to override; default is Runtime.getRuntime().availableProcessors()="
				+ Runtime.getRuntime().availableProcessors() + ")";
	}
}