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
 * Host-side timing for a kernel launched on the default stream between two
 * synchronous copies (the attention kernel, the FP32 BLAS GEMM), counted into
 * {@link DeviceSpanTally} as {@link DeviceComputeEvent}.
 *
 * <p>With no recording asking for the event both calls do nothing beyond the
 * enabled check. With one, a prefill-width launch is bracketed by two drains of
 * the default stream, so the figure is the kernel and not a copy still in flight
 * before it or a wait after it; the synchronous device-to-host copy that follows
 * would have waited for the kernel anyway. Decode-width launches are counted,
 * untimed.
 */
final class DeviceComputeClock {

	private DeviceComputeClock() {
	}

	/**
	 * Returns a start stamp for {@link #done}, or {@code -1} when the launch is not
	 * timed (no recording, or decode width). Drains the default stream first.
	 */
	static long start(GpuBindings gpu, int windowSize) {
		if (windowSize == 1 || !DeviceSpanTally.computeWanted())
			return -1L;
		drain(gpu);
		return System.nanoTime();
	}

	/** Counts the launch at {@code site}, timed when {@code start >= 0}. */
	static void done(GpuBindings gpu, String site, int windowSize, long start) {
		if (start < 0) {
			DeviceSpanTally.compute(site, windowSize, -1L);
			return;
		}
		drain(gpu);
		DeviceSpanTally.compute(site, windowSize, System.nanoTime() - start);
	}

	private static void drain(GpuBindings gpu) {
		GpuBindings.check(GpuBindings.callInt(gpu.gpuStreamSynchronize(), MemorySegment.NULL),
				"streamSynchronize(device compute timing)");
	}
}
