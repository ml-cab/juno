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

import java.lang.foreign.MemorySegment;

/**
 * One grow-on-demand scratch buffer, on the device or in pinned host memory.
 *
 * <p>Growing frees the old buffer before allocating the larger one, so the two never
 * coexist on a card at its capacity. The slot is emptied before the free: when the
 * allocation then fails (the error propagates to the caller's fallback), the slot
 * holds no pointer and no size, so no later call reuses a freed buffer and
 * {@link #free} cannot free it twice. The next {@link #ensure} allocates afresh.
 *
 * <p>Not thread-safe; owned by one pooled scratch entry or one caller at a time.
 */
final class DeviceScratchSlot {

	private final GpuBindings gpu;
	private final int device;
	private final boolean pinnedHost;
	private MemorySegment pointer;
	private long bytes;

	private DeviceScratchSlot(GpuContext ctx, boolean pinnedHost) {
		this.gpu = ctx.bindings();
		this.device = ctx.deviceIndex();
		this.pinnedHost = pinnedHost;
	}

	static DeviceScratchSlot device(GpuContext ctx) {
		return new DeviceScratchSlot(ctx, false);
	}

	static DeviceScratchSlot pinnedHost(GpuContext ctx) {
		return new DeviceScratchSlot(ctx, true);
	}

	/** At least {@code need} bytes, reusing the buffer when it is already large enough. */
	MemorySegment ensure(long need) {
		if (bytes >= need)
			return pointer;
		free();
		MemorySegment fresh = pinnedHost ? gpu.hostMalloc(device, need) : gpu.deviceMalloc(device, need);
		pointer = fresh;
		bytes = need;
		return fresh;
	}

	/** The current buffer, or {@code null} when empty. */
	MemorySegment pointer() {
		return pointer;
	}

	/** Bytes the current buffer holds, 0 when empty. */
	long bytes() {
		return bytes;
	}

	/** Frees the buffer, if any, and leaves the slot empty. */
	void free() {
		MemorySegment old = pointer;
		pointer = null;
		bytes = 0;
		if (old == null)
			return;
		if (pinnedHost)
			gpu.hostFree(old);
		else
			gpu.deviceFree(old);
	}
}
