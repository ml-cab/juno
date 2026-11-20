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

import java.nio.file.Path;
import java.util.Optional;

/**
 * The directory explicit session saves write to and restores read from ({@code --session-dir}); unset, saving and
 * restoring are refused. Read from the {@value #ENV} system property, else the environment variable of the same name.
 * Nothing else touches it: session KV is written to disk only when a client asks.
 */
public final class SessionDirOptions {

	public static final String ENV = "JUNO_SESSION_DIR";

	private SessionDirOptions() {
	}

	/** The configured directory, or empty when none is. */
	public static Optional<Path> fromEnv() {
		String prop = System.getProperty(ENV);
		return parse(prop != null && !prop.isBlank() ? prop : System.getenv(ENV));
	}

	/** {@code raw} as a path, or empty for null or blank. */
	public static Optional<Path> parse(String raw) {
		if (raw == null || raw.isBlank())
			return Optional.empty();
		return Optional.of(Path.of(raw.strip()));
	}
}
