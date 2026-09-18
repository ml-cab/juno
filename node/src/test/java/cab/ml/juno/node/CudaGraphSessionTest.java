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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Correctness tests for {@link CudaGraphSession} (Tier 19 Phase B — see
 * {@code docs/infra-plan/PLAN-Infra-Tier19.md}'s "Measured finding" section
 * for the performance numbers this class informed: capturing and replaying
 * {@code RmsNormKernel}'s launch sequence cut per-call overhead ~3.2x vs the
 * ad-hoc path (107us -> 33us/call) but did not close the gap to the 15.5us
 * CPU scalar baseline for a single isolated op — the remaining cost is the
 * unavoidable host/device synchronization round trip, which only amortizes
 * across a *chain* of graphed ops sharing one sync point, not a single one.
 * This test class exists so the capture/replay mechanism itself stays
 * correctness-verified as reusable infrastructure regardless of that finding.
 *
 * <p>Captures {@code rms_norm}'s H2D/kernel/D2H sequence exactly as the
 * scratch benchmark did, since it is the only kernel in this codebase this
 * session added a graph-capturable async call path for.
 */
@Tag("gpu")
@DisplayName("CudaGraphSession — capture/replay correctness")
class CudaGraphSessionTest {

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		ctx = GpuContext.init(0);
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("launchAndSync before capture throws")
	void launchAndSync_beforeCapture_throws() {
		try (CudaGraphSession session = new CudaGraphSession()) {
			assertThatThrownBy(session::launchAndSync).isInstanceOf(IllegalStateException.class);
		}
	}

	@Test
	@DisplayName("isCaptured reflects lifecycle state")
	void isCaptured_reflectsLifecycle() {
		CudaBindings cuda = CudaBindings.instance();
		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		int dim = 64;
		long bytes = (long) dim * Float.BYTES;

		try (CudaGraphSession session = new CudaGraphSession();
				Arena fixedArena = Arena.ofShared()) {
			assertThat(session.isCaptured()).isFalse();

			MemorySegment dX = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
			MemorySegment dW = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
			MemorySegment dOut = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
			MemorySegment hostX = fixedArena.allocate(bytes);
			MemorySegment hostW = fixedArena.allocate(bytes);
			MemorySegment hostOut = fixedArena.allocate(bytes);
			try {
				session.beginCapture();
				MemorySegment stream = session.stream();
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpyAsync, dX, hostX, bytes, CudaBindings.H2D, stream),
						"memcpyAsync x");
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpyAsync, dW, hostW, bytes, CudaBindings.H2D, stream),
						"memcpyAsync w");
				kernel.launch(dX, dW, dOut, 1, dim, 1e-5f, stream);
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpyAsync, hostOut, dOut, bytes, CudaBindings.D2H, stream),
						"memcpyAsync out");
				session.endCaptureAndInstantiate();

				assertThat(session.isCaptured()).isTrue();
			} finally {
				cuda.deviceFree(dX);
				cuda.deviceFree(dW);
				cuda.deviceFree(dOut);
			}
		}
	}

	@Test
	@DisplayName("captured rms_norm sequence replays correctly across different inputs")
	void capturedRmsNormSequence_replaysCorrectly_acrossDifferentInputs() {
		CudaBindings cuda = CudaBindings.instance();
		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		int dim = 2048;
		float eps = 1e-5f;
		long bytes = (long) dim * Float.BYTES;

		MemorySegment dX = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
		MemorySegment dW = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
		MemorySegment dOut = cuda.deviceMalloc(ctx.deviceIndex(), bytes);
		Arena fixedArena = Arena.ofShared();
		MemorySegment hostX = fixedArena.allocate(bytes);
		MemorySegment hostW = fixedArena.allocate(bytes);
		MemorySegment hostOut = fixedArena.allocate(bytes);

		try (CudaGraphSession session = new CudaGraphSession()) {
			MemorySegment stream = session.stream();
			session.beginCapture();
			CudaBindings.check(
					CudaBindings.callInt(cuda.cudaMemcpyAsync, dX, hostX, bytes, CudaBindings.H2D, stream),
					"memcpyAsync x");
			CudaBindings.check(
					CudaBindings.callInt(cuda.cudaMemcpyAsync, dW, hostW, bytes, CudaBindings.H2D, stream),
					"memcpyAsync w");
			kernel.launch(dX, dW, dOut, 1, dim, eps, stream);
			CudaBindings.check(
					CudaBindings.callInt(cuda.cudaMemcpyAsync, hostOut, dOut, bytes, CudaBindings.D2H, stream),
					"memcpyAsync out");
			session.endCaptureAndInstantiate();

			// Replay the SAME captured graph three times with different content written
			// into the fixed host source buffers — only addresses were captured, not values.
			Random rng = new Random(11);
			for (int trial = 0; trial < 3; trial++) {
				float[] x = new float[dim];
				float[] weight = new float[dim];
				for (int i = 0; i < dim; i++) {
					x[i] = rng.nextFloat() * 4f - 2f;
					weight[i] = rng.nextFloat() * 2f - 1f;
				}
				MemorySegment.copy(x, 0, hostX, JAVA_FLOAT, 0, dim);
				MemorySegment.copy(weight, 0, hostW, JAVA_FLOAT, 0, dim);

				session.launchAndSync();

				float[] actual = new float[dim];
				MemorySegment.copy(hostOut, JAVA_FLOAT, 0, actual, 0, dim);
				float[] expected = LlamaTransformerHandler.rmsNorm(x, weight, eps);
				assertThat(actual).as("trial " + trial).containsExactly(expected, within(1e-4f));
			}
		} finally {
			cuda.deviceFree(dX);
			cuda.deviceFree(dW);
			cuda.deviceFree(dOut);
			fixedArena.close();
		}
	}
}
