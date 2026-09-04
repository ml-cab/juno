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
 * Row-parallel dispatch helper for the weight-stationary CPU quantized matmul
 * kernels ({@link LlamaTransformerHandler#sgemmQ4KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ5KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ8_0WeightStationary}).
 *
 * <p>
 * {@link #forEachRow} uses {@code IntStream.parallel()} on
 * {@link ForkJoinPool#commonPool()}, matching {@code matVecQ*raw}. A previous
 * {@code POOL.submit(() -> IntStream.parallel()...).join()} dispatch was kept
 * briefly for independent pool sizing via {@code juno.simd.pool.size}, but it
 * made vision-scale Q5_K prefill pathologically slow; see {@link #forEachRow}.
 * {@link #POOL} remains for diagnostics ({@link #diagnosticSummary}) and any
 * future pinned dispatch that does not wrap a nested parallel stream.
 *
 * <p>
 * This is not core-affinity pinning. Pure Java has no portable way to bind a
 * thread to a specific physical core without native code, and this class
 * does not attempt to distinguish performance cores from efficiency cores on
 * hybrid CPUs (e.g. Intel Alder Lake and later).
 *
 * <p>
 * Usage: {@code -Djuno.simd.pool.size=8} still builds {@link #POOL} at that
 * size for diagnostics; the hot path currently ignores it (common pool).
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
	 * Runs {@code body} once for every {@code r} in {@code [0, rows)}, and
	 * blocks until all of them complete.
	 *
	 * <p>
	 * Uses {@code IntStream.range(0, rows).parallel().forEach(body)} on
	 * {@link ForkJoinPool#commonPool()} — the same dispatch as
	 * {@code matVecQ5Kraw} / {@code matVecQ4Kraw}. An earlier variant wrapped
	 * that parallel stream in {@code POOL.submit(...).join()}, which on this
	 * codebase's host (and with the Vector-API hot loop) was measured ~37–260×
	 * slower than the sequential matVec path for vision-scale batches (B≈741),
	 * hanging moondream {@code forwardBatch} prefill for hours. The dedicated
	 * {@link #POOL} is retained for sizing diagnostics / future pinned
	 * dispatch; the hot path must stay on the common-pool parallel stream.
	 *
	 * <p>
	 * Any exception thrown by {@code body} on a worker thread propagates out
	 * of this call as an unchecked exception, matching
	 * {@code IntStream.parallel().forEach()}.
	 */
	static void forEachRow(int rows, IntConsumer body) {
		if (rows <= 0)
			return;
		java.util.stream.IntStream.range(0, rows).parallel().forEach(body);
	}

	/**
	 * One-line summary of row-parallel dispatch, for a single startup log line
	 * alongside {@link VectorQuantKernels#diagnosticSummary} /
	 * {@link VectorQuantKernels#policySummary}.
	 */
	public static String diagnosticSummary() {
		return "SIMD row-parallel: forEachRow uses ForkJoinPool.commonPool() IntStream.parallel(); "
				+ "diagnostic POOL parallelism=" + POOL.getParallelism() + " (set -D" + POOL_SIZE_PROPERTY
				+ "=N to size diagnostic pool; hot path ignores it; "
				+ "availableProcessors()=" + Runtime.getRuntime().availableProcessors() + ")";
	}
}