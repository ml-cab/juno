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
 * CLI / env policy for {@code --cache-type-k} and {@code --cache-type-v}.
 *
 * <p>Defaults are {@link KvElementType#F16} (current float32 path) for both.
 */
public final class CacheTypeOptions {

	public static final String ENV_K = "JUNO_CACHE_TYPE_K";
	public static final String ENV_V = "JUNO_CACHE_TYPE_V";

	private final KvElementType typeK;
	private final KvElementType typeV;

	private CacheTypeOptions(KvElementType typeK, KvElementType typeV) {
		this.typeK = typeK;
		this.typeV = typeV;
	}

	public static CacheTypeOptions of(KvElementType typeK, KvElementType typeV) {
		if (typeK == null || typeV == null)
			throw new IllegalArgumentException("types must not be null");
		return new CacheTypeOptions(typeK, typeV);
	}

	public static CacheTypeOptions defaults() {
		return of(KvElementType.F16, KvElementType.F16);
	}

	/** Reads {@link #ENV_K} / {@link #ENV_V} system properties, then process env. */
	public static CacheTypeOptions fromEnv() {
		return of(
				KvElementType.parse(firstNonBlank(System.getProperty(ENV_K), System.getenv(ENV_K), "f16")),
				KvElementType.parse(firstNonBlank(System.getProperty(ENV_V), System.getenv(ENV_V), "f16")));
	}

	private static String firstNonBlank(String a, String b, String fallback) {
		if (a != null && !a.isBlank())
			return a;
		if (b != null && !b.isBlank())
			return b;
		return fallback;
	}

	public KvElementType typeK() {
		return typeK;
	}

	public KvElementType typeV() {
		return typeV;
	}

	public boolean usesQuantized() {
		return typeK == KvElementType.Q8_0 || typeV == KvElementType.Q8_0;
	}

	public String policySummary() {
		return "cache-type-k=" + typeK.cliName() + " cache-type-v=" + typeV.cliName();
	}
}
