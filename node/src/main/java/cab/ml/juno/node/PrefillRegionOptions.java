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

import java.util.Locale;

/**
 * The switch for the prefill-window device region ({@link PrefillWindowRegion}): on
 * by default on a CUDA backend, off with {@code -DJUNO_PREFILL_REGION=off} or the
 * environment variable of the same name. A diagnostic switch, for comparing the region
 * against the host window path; the two compute the same logits. Cluster launchers
 * forward it to every node they fork.
 */
public final class PrefillRegionOptions {

	/** System property (or environment variable) name. */
	public static final String ENV_PROPERTY = "JUNO_PREFILL_REGION";

	private PrefillRegionOptions() {
	}

	/**
	 * Whether the region is requested: on unless the property or environment variable
	 * says {@code off} (also {@code 0}, {@code false}, {@code no}).
	 *
	 * @throws IllegalArgumentException for any value other than on or off
	 */
	public static boolean requested() {
		String prop = System.getProperty(ENV_PROPERTY);
		String raw = prop != null && !prop.isBlank() ? prop : System.getenv(ENV_PROPERTY);
		if (raw == null || raw.isBlank())
			return true;
		return switch (raw.strip().toLowerCase(Locale.ROOT)) {
		case "on", "1", "true", "yes" -> true;
		case "off", "0", "false", "no" -> false;
		default -> throw new IllegalArgumentException(ENV_PROPERTY + " must be on or off (got " + raw + ")");
		};
	}
}
