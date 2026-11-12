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
package cab.ml.juno.kvcache;

/**
 * {@code --prefix-cache-tokens} / {@code JUNO_PREFIX_CACHE_TOKENS}: how many prompt positions of finished requests
 * the server keeps, so a later request whose prompt starts the same way starts from their KV instead of computing
 * it. Unset means the default: as many positions as a quarter of the maximum heap holds, at most one context (the
 * most a single request may hold); {@code 0} turns sharing off.
 */
public final class SharedPrefixOptions {

	public static final String ENV = "JUNO_PREFIX_CACHE_TOKENS";

	/** -1: not set, the default applies. */
	private final int tokens;

	private SharedPrefixOptions(int tokens) {
		this.tokens = tokens;
	}

	/**
	 * @param raw a non-negative integer, or null/blank for the default
	 * @throws IllegalArgumentException for anything else
	 */
	public static SharedPrefixOptions parse(String raw) {
		if (raw == null || raw.isBlank())
			return new SharedPrefixOptions(-1);
		int n;
		try {
			n = Integer.parseInt(raw.strip());
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("--prefix-cache-tokens requires a non-negative integer, got: " + raw);
		}
		if (n < 0)
			throw new IllegalArgumentException("--prefix-cache-tokens requires a non-negative integer, got: " + raw);
		return new SharedPrefixOptions(n);
	}

	public static SharedPrefixOptions fromEnv() {
		String prop = System.getProperty(ENV);
		return parse(prop != null && !prop.isBlank() ? prop : System.getenv(ENV));
	}

	/** Whether the value was given, rather than defaulted. */
	public boolean explicit() {
		return tokens >= 0;
	}

	/**
	 * Positions to keep: the given value, or by default a quarter of {@code maxHeapBytes} at {@code bytesPerToken}
	 * per position, at most {@code contextLimit} (all of it when the size per position is unknown). 0 means off.
	 */
	public int budget(int contextLimit, long maxHeapBytes, long bytesPerToken) {
		if (tokens >= 0)
			return tokens;
		int limit = Math.max(0, contextLimit);
		if (bytesPerToken <= 0)
			return limit;
		return (int) Math.min(limit, maxHeapBytes / 4 / bytesPerToken);
	}
}
