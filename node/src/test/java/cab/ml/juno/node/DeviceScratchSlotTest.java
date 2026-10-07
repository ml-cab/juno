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

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A scratch buffer that grows on demand must never be left holding a pointer it has
 * already freed. Growing frees the old buffer before allocating the new one, and on a
 * card at its capacity the allocation can fail; a slot that kept the freed pointer and
 * its old size would hand that pointer to the next call that fits the old size (writing
 * into memory the allocator may have given to a KV mirror or a weight) and free it a
 * second time when closed.
 *
 * <p>Runs on the real card: a request larger than the whole device fails in the
 * allocator, which is the failure a full card produces at a smaller size.
 */
@Tag("gpu")
class DeviceScratchSlotTest {

	private static final long KIB = 1024L;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	/** More than the device holds, so the allocator refuses it. */
	private static long moreThanTheDevice() {
		return ctx.bindings().memGetInfo(ctx.deviceIndex())[1] * 2;
	}

	@Test
	void aRequestThatFitsKeepsTheBuffer() {
		DeviceScratchSlot slot = DeviceScratchSlot.device(ctx);
		try {
			MemorySegment first = slot.ensure(64 * KIB);
			assertThat(slot.ensure(16 * KIB)).isEqualTo(first);
			assertThat(slot.bytes()).isEqualTo(64 * KIB);
		} finally {
			slot.free();
		}
	}

	@Test
	void aGrowthThatRunsOutLeavesTheSlotEmptyNotDangling() {
		DeviceScratchSlot slot = DeviceScratchSlot.device(ctx);
		try {
			slot.ensure(64 * KIB);
			IllegalStateException oom = assertThrows(IllegalStateException.class,
					() -> slot.ensure(moreThanTheDevice()));
			assertThat(GpuLayerOffload.isVramOom(oom)).as("the failure is the allocator's").isTrue();
			assertThat(slot.bytes()).as("no size kept for a freed buffer").isZero();
			assertThat(slot.pointer()).as("no freed pointer kept").isNull();
			// The next call that fits the old size allocates afresh instead of reusing it.
			MemorySegment next = slot.ensure(16 * KIB);
			assertThat(next).isNotNull();
			assertThat(slot.bytes()).isEqualTo(16 * KIB);
		} finally {
			slot.free();
		}
		assertThat(slot.bytes()).isZero();
		assertThat(slot.pointer()).isNull();
	}

	@Test
	void aPinnedHostGrowthThatRunsOutLeavesTheSlotEmpty() {
		DeviceScratchSlot slot = DeviceScratchSlot.pinnedHost(ctx);
		try {
			slot.ensure(64 * KIB);
			// Far beyond any host's memory, so the pinned allocation fails.
			assertThrows(IllegalStateException.class, () -> slot.ensure(1L << 50));
			assertThat(slot.bytes()).isZero();
			assertThat(slot.pointer()).isNull();
			assertThat(slot.ensure(16 * KIB)).isNotNull();
		} finally {
			slot.free();
		}
	}
}
