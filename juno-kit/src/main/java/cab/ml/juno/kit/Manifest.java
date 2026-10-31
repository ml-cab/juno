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
package cab.ml.juno.kit;


import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * The results zip's {@code manifest.json}. {@code kind} is {@code run}, or {@code build-failure} for the
 * zip {@link KitBootstrap} writes when the build fails (same fields, written without Jackson).
 * {@code schemaVersion} rises whenever a field changes meaning; readers ignore fields they do not know.
 */
record Manifest(int schemaVersion, String kind, String kitVersion, String junoCommit, boolean junoTreeDirty,
		String kitJarSha256, String suite, List<String> probes, String startedUtc, String finishedUtc,
		Map<String, Object> host) {

	static final int SCHEMA_VERSION = 1;
	static final String FILE = "manifest.json";

	void write(Path file) throws IOException {
		Json.write(file, this);
	}

	static Manifest read(Path file) throws IOException {
		return Json.read(file, Manifest.class);
	}
}
