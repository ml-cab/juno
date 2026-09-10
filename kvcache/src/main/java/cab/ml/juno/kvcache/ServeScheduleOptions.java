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
 * Serving schedule policy for dual KV path selection.
 *
 * <p>{@code --schedule static} (default): dense in-process KV (no gather).
 * {@code --schedule continuous}: paged KV + gather-to-workspace. The continuous
 * <em>scheduler engine</em> is owned by a later step; this class only selects
 * the KV layout.
 */
public final class ServeScheduleOptions {

	public static final String ENV = "JUNO_SCHEDULE";

	public enum Mode {
		STATIC, CONTINUOUS
	}

	private final Mode mode;

	private ServeScheduleOptions(Mode mode) {
		this.mode = mode;
	}

	public static ServeScheduleOptions of(Mode mode) {
		if (mode == null)
			throw new IllegalArgumentException("mode must not be null");
		return new ServeScheduleOptions(mode);
	}

	public static ServeScheduleOptions defaults() {
		return of(Mode.STATIC);
	}

	public static ServeScheduleOptions parse(String raw) {
		if (raw == null || raw.isBlank())
			return defaults();
		return switch (raw.strip().toLowerCase()) {
		case "static" -> of(Mode.STATIC);
		case "continuous" -> of(Mode.CONTINUOUS);
		default -> throw new IllegalArgumentException(
				"unknown schedule '" + raw + "' (expected static|continuous)");
		};
	}

	public static ServeScheduleOptions fromEnv() {
		return parse(firstNonBlank(System.getProperty(ENV), System.getenv(ENV)));
	}

	private static String firstNonBlank(String a, String b) {
		if (a != null && !a.isBlank())
			return a;
		if (b != null && !b.isBlank())
			return b;
		return null;
	}

	public Mode mode() {
		return mode;
	}

	/** Continuous schedule uses {@link KvPageTable} / {@link KvBlockPool}. */
	public boolean usesPagedKv() {
		return mode == Mode.CONTINUOUS;
	}

	public String policySummary() {
		return "schedule=" + mode.name().toLowerCase();
	}
}
