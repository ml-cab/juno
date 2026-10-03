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

import java.io.File;

/**
 * The JVM heap for a forked node: the {@code juno.node.heap} property when set
 * (the launchers pass their own resolved heap through it), otherwise derived from
 * the model file the same way {@code run.sh}'s {@code heap_for_model} does:
 * 1.5 times the file size plus 2 GiB, rounded up to whole GiB, clamped to 4..48.
 * A node in tensor-parallel mode loads every layer, so a fixed default sized for
 * small models runs a 7B node out of heap.
 */
final class NodeHeapSize {

	/** System property a launcher sets to pass its resolved heap to forked nodes. */
	static final String PROPERTY = "juno.node.heap";

	private static final long GIB = 1024L * 1024 * 1024;
	private static final long MIN_GIB = 4;
	private static final long MAX_GIB = 48;

	private NodeHeapSize() {
	}

	/** The heap for a node serving {@code modelPath} (null in stub mode). */
	static String resolve(String explicit, String modelPath) {
		if (explicit != null && !explicit.isBlank())
			return explicit.strip();
		if (modelPath == null)
			return MIN_GIB + "g";
		File f = new File(modelPath);
		return f.isFile() ? forFileBytes(f.length()) : MIN_GIB + "g";
	}

	/** The derived heap for a model file of {@code bytes} bytes. */
	static String forFileBytes(long bytes) {
		if (bytes <= 0)
			return MIN_GIB + "g";
		long gib = (bytes * 3 / 2 + 2 * GIB + GIB - 1) / GIB;
		return Math.max(MIN_GIB, Math.min(MAX_GIB, gib)) + "g";
	}
}
