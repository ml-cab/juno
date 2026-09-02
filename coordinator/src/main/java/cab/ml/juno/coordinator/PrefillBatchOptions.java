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
