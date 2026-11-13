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

import java.util.Arrays;

/**
 * A double-ended queue of {@code int}s in a ring buffer that doubles when full: page ids without boxing. Not
 * thread-safe; {@link KvBlockPool} uses it under its lock.
 */
final class IntRing {

	private int[] buf = new int[16];
	private int head;
	private int size;

	int size() {
		return size;
	}

	boolean isEmpty() {
		return size == 0;
	}

	void addLast(int v) {
		if (size == buf.length)
			grow();
		buf[(head + size) & (buf.length - 1)] = v;
		size++;
	}

	/** The first value, removed; the queue must not be empty. */
	int pollFirst() {
		if (size == 0)
			throw new IllegalStateException("empty");
		int v = buf[head];
		head = (head + 1) & (buf.length - 1);
		size--;
		return v;
	}

	/** The last value, removed; the queue must not be empty. */
	int pollLast() {
		if (size == 0)
			throw new IllegalStateException("empty");
		size--;
		return buf[(head + size) & (buf.length - 1)];
	}

	private void grow() {
		int[] next = new int[buf.length * 2];
		int first = Math.min(size, buf.length - head);
		System.arraycopy(buf, head, next, 0, first);
		System.arraycopy(buf, 0, next, first, size - first);
		buf = next;
		head = 0;
	}

	@Override
	public String toString() {
		int[] out = new int[size];
		for (int i = 0; i < size; i++)
			out[i] = buf[(head + i) & (buf.length - 1)];
		return Arrays.toString(out);
	}
}
