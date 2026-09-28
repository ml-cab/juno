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

import java.util.ArrayDeque;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.ToLongFunction;

/**
 * Device scratch entries shared by the callers of one GPU operation, sized by
 * how many call at once rather than by how many threads have ever called.
 *
 * <p>The request scheduler runs every request on a new thread. Scratch kept per
 * thread is therefore allocated once per request and never used again, and
 * nothing frees device memory when a thread ends. A caller here takes an idle
 * entry (or a new one if none is idle), uses it exclusively, and gives it back;
 * the pool holds at most as many entries as there were concurrent callers.
 *
 * <p>No allocation per acquire or release once the pool has grown: the idle
 * entries sit in an {@link ArrayDeque}. After {@link #close()} the idle entries
 * are freed, and an entry given back later is freed instead of kept.
 */
final class DeviceScratchPool<T> {

	private final Supplier<T> factory;
	private final Consumer<T> free;
	private final ArrayDeque<T> idle = new ArrayDeque<>();
	private boolean closed;

	/**
	 * @param factory creates an empty entry (no device memory yet; entries grow on use)
	 * @param free    releases every device and pinned host buffer an entry holds
	 */
	DeviceScratchPool(Supplier<T> factory, Consumer<T> free) {
		this.factory = factory;
		this.free = free;
	}

	/** An entry for the caller's exclusive use until {@link #release}. */
	T acquire() {
		synchronized (idle) {
			T entry = idle.pollFirst();
			if (entry != null)
				return entry;
		}
		return factory.get();
	}

	/** Gives an entry back; frees it instead if the pool is closed. */
	void release(T entry) {
		synchronized (idle) {
			if (!closed) {
				idle.addFirst(entry);
				return;
			}
		}
		free.accept(entry);
	}

	/** Frees every idle entry. Entries in use are freed when they are released. */
	void close() {
		ArrayDeque<T> drained;
		synchronized (idle) {
			closed = true;
			drained = new ArrayDeque<>(idle);
			idle.clear();
		}
		for (T entry : drained)
			free.accept(entry);
	}

	/** Sum of {@code bytes} over the idle entries. */
	long idleBytes(ToLongFunction<T> bytes) {
		synchronized (idle) {
			long total = 0;
			for (T entry : idle)
				total += bytes.applyAsLong(entry);
			return total;
		}
	}
}
