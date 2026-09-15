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

import java.util.Arrays;

/**
 * Append-only primitive int buffer for the decode hot path's per-step generated-token
 * history.
 *
 * <p>Replaces reconstructing {@code int[]} history from a {@code List<Integer>} via
 * {@code stream().mapToInt(Integer::intValue).toArray()} on every decode step (boxed
 * traversal + stream overhead, repeated once per token per active slot). {@link #append}
 * amortizes to O(1); {@link #toTrimmedArray()} is a single {@code Arrays.copyOf} call.
 */
final class GrowableIntArray {

	private int[] buf;
	private int size;

	GrowableIntArray() {
		this(16);
	}

	GrowableIntArray(int initialCapacity) {
		this.buf = new int[Math.max(1, initialCapacity)];
	}

	void append(int value) {
		if (size == buf.length)
			buf = Arrays.copyOf(buf, buf.length * 2);
		buf[size++] = value;
	}

	int size() {
		return size;
	}

	/** Returns a defensive, exact-length copy of the values appended so far. */
	int[] toTrimmedArray() {
		return Arrays.copyOf(buf, size);
	}
}
