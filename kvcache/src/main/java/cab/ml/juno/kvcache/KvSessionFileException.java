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
 * Why a session's KV could not be saved or restored, as a reason a caller maps to its own answer (the REST surface:
 * {@link Reason#NOT_FOUND} to 404, the rest to 400) and a message naming what differed.
 */
public final class KvSessionFileException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/** What went wrong. */
	public enum Reason {
		/** No file under the expected name. */
		NOT_FOUND,
		/** The file describes other KV than the reader holds: another model, adapter, stage, tokens, shape or format version. */
		MISMATCH,
		/** Not a session file, cut short, or changed since it was written (the checksum disagrees). */
		CORRUPT,
		/** The session's KV may not be saved or restored here (it holds an image, or the device KV could not reproduce it). */
		REFUSED,
		/** This handler or pipeline keeps no KV it can save. */
		UNSUPPORTED
	}

	private final Reason reason;

	public KvSessionFileException(Reason reason, String message) {
		super(message);
		if (reason == null)
			throw new IllegalArgumentException("reason must not be null");
		this.reason = reason;
	}

	public Reason reason() {
		return reason;
	}
}
