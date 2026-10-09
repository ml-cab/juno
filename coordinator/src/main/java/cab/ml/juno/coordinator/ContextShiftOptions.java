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
 * Server default for context shifting ({@code --context-shift on|off}, off by
 * default), read from the {@value #PROPERTY} system property or environment
 * variable. A request's own choice ({@link InferenceRequest#contextShift()})
 * overrides it.
 */
public final class ContextShiftOptions {

	public static final String PROPERTY = "JUNO_CONTEXT_SHIFT";

	private ContextShiftOptions() {
	}

	/**
	 * @throws IllegalArgumentException for a value other than on/off (or true/false, 1/0, yes/no)
	 */
	public static boolean parse(String spec) {
		if (spec == null || spec.isBlank())
			return false;
		return switch (spec.strip().toLowerCase(Locale.ROOT)) {
		case "on", "true", "1", "yes" -> true;
		case "off", "false", "0", "no" -> false;
		default -> throw new IllegalArgumentException("--context-shift must be on|off (got " + spec + ")");
		};
	}

	/** The server default: the system property, else the environment variable, else off. */
	public static boolean enabledByDefault() {
		String raw = System.getProperty(PROPERTY);
		if (raw == null || raw.isBlank())
			raw = System.getenv(PROPERTY);
		return parse(raw);
	}

	/** The request's own choice, else the server default. */
	public static boolean enabledFor(InferenceRequest request) {
		return request.contextShift() != null ? request.contextShift() : enabledByDefault();
	}
}
