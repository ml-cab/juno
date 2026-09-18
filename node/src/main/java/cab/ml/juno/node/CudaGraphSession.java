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

/**
 * Owns a CUDA stream plus (once captured) an instantiated graph on that
 * stream — Tier 19 Phase B (see {@code docs/infra-plan/PLAN-Infra-Tier19.md}
 * "Measured finding": per-op GPU dispatch was found to cost ~11x more than
 * the CPU path it replaces due to each call's independent
 * H2D-upload/kernel-launch/D2H-download round trip; this class exists to test
 * whether replaying a pre-recorded sequence of those same operations as one
 * graph launch amortizes away the per-call CUDA driver dispatch overhead that
 * a discrete {@code cudaMemcpyAsync}/{@code cuLaunchKernel}/{@code cudaMemcpyAsync}
 * call sequence pays every time).
 *
 * <p>Usage: call {@link #stream()} and issue a <b>fixed</b> sequence of async
 * CUDA operations against fixed-address host/device buffers between
 * {@link #beginCapture()} and {@link #endCaptureAndInstantiate()} — exactly
 * once. Every subsequent {@link #launchAndSync()} replays that exact sequence;
 * only the *contents* of the source buffers may change between replays (via
 * plain host memory writes, no CUDA call), not their addresses or sizes. A
 * shape change (different row count or dimension) requires a new session —
 * this class does not support in-place re-capture.
 *
 * <p>Not thread-safe; callers must serialize {@link #launchAndSync()} calls
 * (matching how a single decode stream issues one op at a time).
 */
final class CudaGraphSession implements AutoCloseable {

	private final CudaBindings cuda;
	private final CudaDriverBindings drv;
	private final MemorySegment stream; // cudaStream_t / CUstream — same ABI handle, cudart- and driver-API-compatible
	private MemorySegment graphExec;    // CUgraphExec, null until endCaptureAndInstantiate()

	private static final int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;

	CudaGraphSession() {
		this.cuda = CudaBindings.instance();
		this.drv = CudaDriverBindings.instance();
		try (Arena a = Arena.ofConfined()) {
			MemorySegment slot = a.allocate(ADDRESS);
			CudaBindings.check(
					CudaBindings.callInt(cuda.cudaStreamCreateWithFlags, slot, 0),
					"cudaStreamCreateWithFlags");
			this.stream = slot.get(ADDRESS, 0);
		}
	}

	/** The stream to issue the captured operation sequence on. */
	MemorySegment stream() {
		return stream;
	}

	boolean isCaptured() {
		return graphExec != null;
	}

	/** Starts recording every async CUDA op issued on {@link #stream()} from this point. */
	void beginCapture() {
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuStreamBeginCapture, stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
				"cuStreamBeginCapture");
	}

	/** Ends recording and instantiates an executable graph, ready for {@link #launchAndSync()}. */
	void endCaptureAndInstantiate() {
		try (Arena a = Arena.ofConfined()) {
			MemorySegment graphSlot = a.allocate(ADDRESS);
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuStreamEndCapture, stream, graphSlot),
					"cuStreamEndCapture");
			MemorySegment graph = graphSlot.get(ADDRESS, 0);

			MemorySegment execSlot = a.allocate(ADDRESS);
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuGraphInstantiate, execSlot, graph, 0L),
					"cuGraphInstantiate");
			graphExec = execSlot.get(ADDRESS, 0);

			// The graph template is no longer needed once instantiated into an exec.
			CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuGraphDestroy, graph), "cuGraphDestroy");
		}
	}

	/** Replays the captured sequence and blocks until it completes. */
	void launchAndSync() {
		if (graphExec == null)
			throw new IllegalStateException("graph not captured — call beginCapture()/endCaptureAndInstantiate() first");
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuGraphLaunch, graphExec, stream),
				"cuGraphLaunch");
		CudaBindings.check(
				CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
				"cudaStreamSynchronize");
	}

	@Override
	public void close() {
		if (graphExec != null) {
			CudaDriverBindings.callInt(drv.cuGraphExecDestroy, graphExec);
			graphExec = null;
		}
		CudaBindings.callInt(cuda.cudaStreamDestroy, stream);
	}
}
