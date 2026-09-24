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

package cab.ml.juno.player;

import jdk.jfr.Configuration;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.text.ParseException;

/**
 * Resolves the one JFR configuration every Juno recording is taken under.
 *
 * <p>A throughput measurement is only comparable with another one taken under the
 * same instrumentation overhead. Juno starts recordings from several places — the
 * test command in the launcher scripts, each forked cluster-node JVM, and the
 * local, LoRA and cluster-coordinator recordings this console starts in-process —
 * and those places used to name different configurations, so two published runs
 * could differ by their recording settings alone. They all resolve their settings
 * through this class now.
 *
 * <p>Two shapes are needed. An in-process recording wants a {@link Configuration};
 * a forked JVM wants a filesystem path for {@code -XX:StartFlightRecording}'s
 * {@code settings=} argument, so the packaged copy is materialized to a temporary
 * file for that case.
 *
 * <p>Resolution order: the {@value #SETTINGS_PROPERTY} system property, then the
 * copy packaged alongside this class, then the working copy under
 * {@code scripts/performance-tests/}. If none resolves, callers fall back to the
 * JDK's stock low-overhead configuration and say so on the console rather than
 * recording under different settings silently — a run that cannot state which
 * settings produced it cannot be published as a comparison.
 *
 * @author Yevhen Soldatov
 */
final class JunoJfrSettings {

	/** Overrides the packaged configuration with a file of the operator's choosing. */
	static final String SETTINGS_PROPERTY = "juno.jfr.settings";

	static final String RESOURCE_NAME = "juno-perf.jfc";

	private static final Path REPO_COPY = Path.of("scripts", "performance-tests", RESOURCE_NAME);

	/** Name of the JDK configuration used when the Juno one cannot be found. */
	private static final String FALLBACK = "default";

	private static volatile Path materialized;

	private JunoJfrSettings() {
	}

	/**
	 * The configuration for an in-process {@link jdk.jfr.Recording}, falling back to
	 * the JDK's stock low-overhead configuration if the Juno one cannot be resolved.
	 */
	static Configuration configuration() throws IOException, ParseException {
		Path file = file();
		if (file != null) {
			try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
				return Configuration.create(reader);
			}
		}
		try (InputStream in = JunoJfrSettings.class.getResourceAsStream("/" + RESOURCE_NAME)) {
			if (in != null) {
				try (Reader reader = new InputStreamReader(in, StandardCharsets.UTF_8)) {
					return Configuration.create(reader);
				}
			}
		}
		return Configuration.getConfiguration(FALLBACK);
	}

	/**
	 * A filesystem path for a forked JVM's {@code settings=} argument, or {@code null}
	 * if the configuration could not be resolved — in which case the caller passes
	 * {@value #FALLBACK} and warns.
	 *
	 * <p>The packaged copy is extracted once per process and reused; it is left for
	 * the JVM to delete on exit, because a forked node reads it after this method
	 * returns and for as long as that node runs.
	 */
	static Path fileForForkedJvm() {
		Path onDisk = file();
		if (onDisk != null)
			return onDisk;
		Path cached = materialized;
		if (cached != null)
			return cached;
		synchronized (JunoJfrSettings.class) {
			if (materialized != null)
				return materialized;
			try (InputStream in = JunoJfrSettings.class.getResourceAsStream("/" + RESOURCE_NAME)) {
				if (in == null)
					return null;
				Path tmp = Files.createTempFile("juno-perf-", ".jfc");
				Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
				tmp.toFile().deleteOnExit();
				materialized = tmp;
				return tmp;
			} catch (IOException e) {
				throw new UncheckedIOException("cannot materialize " + RESOURCE_NAME + " for a forked JVM", e);
			}
		}
	}

	/** The name a message should use for whichever settings were actually resolved. */
	static String describe() {
		Path onDisk = file();
		if (onDisk != null)
			return onDisk.toString();
		return JunoJfrSettings.class.getResource("/" + RESOURCE_NAME) != null ? RESOURCE_NAME + " (packaged)"
				: FALLBACK + " (JDK stock — " + RESOURCE_NAME + " not found)";
	}

	/** True when the Juno configuration was found, false when a caller must warn. */
	static boolean resolved() {
		return file() != null || JunoJfrSettings.class.getResource("/" + RESOURCE_NAME) != null;
	}

	static String fallbackName() {
		return FALLBACK;
	}

	private static Path file() {
		String override = System.getProperty(SETTINGS_PROPERTY);
		if (override != null && !override.isBlank()) {
			Path p = Path.of(override.trim());
			if (Files.isReadable(p))
				return p;
		}
		return Files.isReadable(REPO_COPY) ? REPO_COPY : null;
	}
}
