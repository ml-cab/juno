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

/**
 * CLI / env resolution for static micro-batching ({@code --parallel},
 * {@code --batch-window-ms}).
 *
 * <p>{@code parallel=1} maps to {@link BatchConfig#disabled()} for back-compat.
 * When {@code parallel > 1} and no window is set, the default window is 50 ms.
 */
public final class ServeBatchOptions {

	public static final String ENV_PARALLEL = "JUNO_PARALLEL";
	public static final String ENV_BATCH_WINDOW_MS = "JUNO_BATCH_WINDOW_MS";
	private static final long DEFAULT_WINDOW_MS = 50L;

	private final int parallel;
	private final long batchWindowMs;

	private ServeBatchOptions(int parallel, long batchWindowMs) {
		this.parallel = parallel;
		this.batchWindowMs = batchWindowMs;
	}

	public static ServeBatchOptions defaults() {
		return new ServeBatchOptions(1, 0L);
	}

	/**
	 * Resolve from optional CLI overrides, then env, then defaults.
	 *
	 * @param parallelArg      CLI {@code --parallel} or null
	 * @param batchWindowArg   CLI {@code --batch-window-ms} or null
	 */
	public static ServeBatchOptions resolve(Integer parallelArg, Long batchWindowArg) {
		int parallel = parallelArg != null ? parallelArg : parsePositiveInt(env(ENV_PARALLEL), 1);
		long window;
		if (batchWindowArg != null) {
			window = batchWindowArg;
		} else {
			String envWin = env(ENV_BATCH_WINDOW_MS);
			window = envWin != null ? parseNonNegativeLong(envWin, DEFAULT_WINDOW_MS)
					: (parallel > 1 ? DEFAULT_WINDOW_MS : 0L);
		}
		return of(parallel, window);
	}

	public static ServeBatchOptions of(int parallel, long batchWindowMs) {
		if (parallel < 1)
			throw new IllegalArgumentException("parallel must be >= 1, got: " + parallel);
		if (batchWindowMs < 0)
			throw new IllegalArgumentException("batchWindowMs must be >= 0, got: " + batchWindowMs);
		return new ServeBatchOptions(parallel, batchWindowMs);
	}

	public int parallel() {
		return parallel;
	}

	public long batchWindowMs() {
		return batchWindowMs;
	}

	public BatchConfig toBatchConfig() {
		if (parallel <= 1)
			return BatchConfig.disabled();
		return BatchConfig.of(parallel, batchWindowMs);
	}

	private static String env(String key) {
		String v = System.getenv(key);
		return (v == null || v.isBlank()) ? null : v.strip();
	}

	private static int parsePositiveInt(String raw, int defaultValue) {
		if (raw == null)
			return defaultValue;
		try {
			int n = Integer.parseInt(raw);
			if (n < 1)
				throw new IllegalArgumentException(ENV_PARALLEL + " must be >= 1, got: " + n);
			return n;
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("invalid " + ENV_PARALLEL + ": " + raw);
		}
	}

	private static long parseNonNegativeLong(String raw, long defaultValue) {
		if (raw == null)
			return defaultValue;
		try {
			long n = Long.parseLong(raw);
			if (n < 0)
				throw new IllegalArgumentException(ENV_BATCH_WINDOW_MS + " must be >= 0, got: " + n);
			return n;
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("invalid " + ENV_BATCH_WINDOW_MS + ": " + raw);
		}
	}
}
