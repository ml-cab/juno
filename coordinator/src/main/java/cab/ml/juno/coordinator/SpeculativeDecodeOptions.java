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

import java.util.Locale;

/**
 * CLI / env resolution for ngram speculative decoding
 * ({@code --spec-type}, {@code --spec-ngram-n}, {@code --spec-ngram-m}).
 *
 * <p>{@link SpecType#NONE} (default) is byte-for-byte identical to the
 * pre-speculation decode path. {@link SpecType#NGRAM_SIMPLE} drafts tokens from
 * an in-memory ngram cache built from the request's own prompt and generated
 * tokens — no second model, no static corpus.
 */
public final class SpeculativeDecodeOptions {

	public enum SpecType {
		NONE, NGRAM_SIMPLE
	}

	public static final String ENV_SPEC_TYPE = "JUNO_SPEC_TYPE";
	public static final String ENV_SPEC_NGRAM_N = "JUNO_SPEC_NGRAM_N";
	public static final String ENV_SPEC_NGRAM_M = "JUNO_SPEC_NGRAM_M";
	public static final SpecType DEFAULT_SPEC_TYPE = SpecType.NONE;
	public static final int DEFAULT_NGRAM_N = 3;
	public static final int DEFAULT_NGRAM_M = 4;

	private final SpecType specType;
	private final int ngramN;
	private final int ngramM;

	private SpeculativeDecodeOptions(SpecType specType, int ngramN, int ngramM) {
		this.specType = specType;
		this.ngramN = ngramN;
		this.ngramM = ngramM;
	}

	/** {@code --spec-type none} — identical to the pre-speculation decode path. */
	public static SpeculativeDecodeOptions disabled() {
		return new SpeculativeDecodeOptions(SpecType.NONE, DEFAULT_NGRAM_N, DEFAULT_NGRAM_M);
	}

	/**
	 * Resolve from optional CLI overrides, then env, then defaults.
	 *
	 * @param cliSpecType {@code --spec-type} raw value ({@code "none"} or
	 *                    {@code "ngram-simple"}), or null
	 * @param cliNgramN   {@code --spec-ngram-n} or null
	 * @param cliNgramM   {@code --spec-ngram-m} or null
	 */
	public static SpeculativeDecodeOptions resolve(String cliSpecType, Integer cliNgramN, Integer cliNgramM) {
		SpecType type = cliSpecType != null ? parseType(cliSpecType) : resolveTypeFromEnv();
		int n = cliNgramN != null ? cliNgramN : resolvePositiveIntFromEnv(ENV_SPEC_NGRAM_N, DEFAULT_NGRAM_N);
		int m = cliNgramM != null ? cliNgramM : resolvePositiveIntFromEnv(ENV_SPEC_NGRAM_M, DEFAULT_NGRAM_M);
		return of(type, n, m);
	}

	public static SpeculativeDecodeOptions of(SpecType specType, int ngramN, int ngramM) {
		if (ngramN < 1)
			throw new IllegalArgumentException("spec-ngram-n must be >= 1, got: " + ngramN);
		if (ngramM < 1)
			throw new IllegalArgumentException("spec-ngram-m must be >= 1, got: " + ngramM);
		return new SpeculativeDecodeOptions(specType, ngramN, ngramM);
	}

	public SpecType specType() {
		return specType;
	}

	public int ngramN() {
		return ngramN;
	}

	public int ngramM() {
		return ngramM;
	}

	public boolean enabled() {
		return specType != SpecType.NONE;
	}

	/** Parse a raw {@code --spec-type}/{@code JUNO_SPEC_TYPE} value. */
	public static SpecType parseType(String raw) {
		String v = raw.strip().toLowerCase(Locale.ROOT);
		return switch (v) {
			case "none" -> SpecType.NONE;
			case "ngram-simple" -> SpecType.NGRAM_SIMPLE;
			default -> throw new IllegalArgumentException(
					"invalid --spec-type: " + raw + " (expected none|ngram-simple)");
		};
	}

	private static SpecType resolveTypeFromEnv() {
		String v = env(ENV_SPEC_TYPE);
		return v != null ? parseType(v) : DEFAULT_SPEC_TYPE;
	}

	private static int resolvePositiveIntFromEnv(String key, int fallback) {
		String v = env(key);
		if (v == null)
			return fallback;
		try {
			int n = Integer.parseInt(v);
			if (n < 1)
				throw new IllegalArgumentException(key + " must be >= 1, got: " + n);
			return n;
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("invalid " + key + ": " + v);
		}
	}

	private static String env(String key) {
		String v = System.getenv(key);
		return (v == null || v.isBlank()) ? null : v.strip();
	}
}
