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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * A {@code cuLaunchKernel} parameter block allocated once and rewritten in place,
 * so a launch allocates nothing and boxes nothing.
 *
 * <p>The per-call launch path the kernel classes grew up with builds a confined
 * arena, allocates one native slot per argument, and invokes the driver through
 * {@code invokeWithArguments}, which boxes every argument into an {@code Object[]}.
 * That is cheap next to a synchronous host round trip, and it stops being cheap
 * once the round trip is gone: on the device-resident path a launch is most of
 * what a small operation costs. Here the slots and the {@code void*[]} array
 * pointing at them live for the life of the block, a launch only overwrites slot
 * values, and the driver is reached through {@code invokeExact} with its exact
 * signature.
 *
 * <p>The driver reads the parameter values during {@code cuLaunchKernel} itself,
 * so a block may be rewritten for the next launch as soon as the call returns.
 * A block is not safe for concurrent use; callers keep one per thread.
 */
final class KernelParams {

	/** Every kernel argument here is at most 8 bytes: a device pointer, an int or a float. */
	private static final long SLOT_BYTES = 8;

	private final int count;
	private final MemorySegment slots;
	private final MemorySegment pointers;

	KernelParams(int count) {
		if (count <= 0)
			throw new IllegalArgumentException("a kernel needs at least one parameter");
		this.count = count;
		Arena arena = Arena.ofAuto();
		this.slots = arena.allocate(SLOT_BYTES * count, SLOT_BYTES);
		this.pointers = arena.allocate(ADDRESS, count);
		for (int i = 0; i < count; i++)
			pointers.setAtIndex(ADDRESS, i, slots.asSlice(i * SLOT_BYTES, SLOT_BYTES));
	}

	KernelParams pointer(int index, MemorySegment value) {
		slots.set(ADDRESS, offset(index), value);
		return this;
	}

	KernelParams i32(int index, int value) {
		slots.set(JAVA_INT, offset(index), value);
		return this;
	}

	KernelParams i64(int index, long value) {
		slots.set(JAVA_LONG, offset(index), value);
		return this;
	}

	KernelParams f32(int index, float value) {
		slots.set(JAVA_FLOAT, offset(index), value);
		return this;
	}

	/**
	 * Launches {@code function} on a one-dimensional grid with this block's current
	 * values. {@code stream} may be {@code null} for the default stream.
	 */
	void launch(MemorySegment function, int gridX, int blockX, MemorySegment stream, String what) {
		launch(function, gridX, 1, blockX, stream, what);
	}

	/** {@link #launch(MemorySegment, int, int, MemorySegment, String)} on a two-dimensional grid. */
	void launch(MemorySegment function, int gridX, int gridY, int blockX, MemorySegment stream, String what) {
		MemorySegment onStream = stream == null ? MemorySegment.NULL : stream;
		int rc;
		try {
			rc = (int) CudaDriverBindings.instance().cuLaunchKernel.invokeExact(
					function, gridX, gridY, 1, blockX, 1, 1, 0, onStream, pointers, MemorySegment.NULL);
		} catch (Throwable t) {
			throw new IllegalStateException(what + ": native call failed", t);
		}
		CudaDriverBindings.check(rc, what);
	}

	private long offset(int index) {
		if (index < 0 || index >= count)
			throw new IndexOutOfBoundsException("parameter " + index + " of " + count);
		return index * SLOT_BYTES;
	}
}
