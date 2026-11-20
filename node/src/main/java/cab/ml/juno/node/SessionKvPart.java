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

/**
 * One stage's file of a saved session: the stage that wrote it (its layer range, and its rank on a tensor-parallel
 * node), the file's name in that stage's session directory, and its size in bytes.
 */
public record SessionKvPart(String stage, String file, long bytes) {

	public SessionKvPart {
		if (stage == null || stage.isBlank())
			throw new IllegalArgumentException("stage must not be blank");
		if (file == null || file.isBlank())
			throw new IllegalArgumentException("file must not be blank");
		if (bytes < 0)
			throw new IllegalArgumentException("bytes must be >= 0");
	}

	/** The file name a stage labelled {@code stage} uses for session {@code sessionId}. */
	public static String fileName(String sessionId, String stage) {
		return sessionId + "." + stage + ".kv";
	}
}
