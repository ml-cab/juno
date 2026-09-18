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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolates {@code RmsNormKernel.launch} (Tier 19 Phase A step 2 — see
 * {@code docs/infra-plan/PLAN-Infra-Tier19.md}) from the handler-level
 * dispatch built on top of it ({@code CudaRmsNorm} /
 * {@code LlamaTransformerHandlerElementwiseOpsJfrTest}): normalise a known
 * batch of rows on the device, download the result, and compare against
 * {@link LlamaTransformerHandler#rmsNorm} — the same scalar CPU reference the
 * whole tier is trying to offload.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=RmsNormKernelParityTest}.
 */
@Tag("gpu")
@DisplayName("RmsNormKernel.launch — parity vs LlamaTransformerHandler.rmsNorm")
class RmsNormKernelParityTest {

	/** Both sides are FP32; only summation-order rounding differs. */
	private static final float TOL = 1e-4f;

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
	@DisplayName("single row (decode-shaped, B=1) matches scalar CPU reference")
	void singleRow_matchesReference() {
		normalizeMatchesReference(1, 4096, 1e-5f, 42);
	}

	@Test
	@DisplayName("batched rows (prefill-shaped, B>1) match scalar CPU reference per row")
	void batchedRows_matchReferencePerRow() {
		normalizeMatchesReference(32, 2048, 1e-5f, 99);
	}

	@Test
	@DisplayName("non-power-of-two dim (Phi/Qwen-style hidden size) matches reference")
	void oddDimension_matchesReference() {
		normalizeMatchesReference(4, 3072, 1e-6f, 777);
	}

	private void normalizeMatchesReference(int batch, int dim, float eps, long seed) {
		Random rng = new Random(seed);
		float[][] x = new float[batch][dim];
		for (int b = 0; b < batch; b++)
			for (int i = 0; i < dim; i++)
				x[b][i] = (rng.nextFloat() * 4f) - 2f;
		float[] weight = new float[dim];
		for (int i = 0; i < dim; i++)
			weight[i] = (rng.nextFloat() * 2f) - 1f;

		float[][] expected = new float[batch][];
		for (int b = 0; b < batch; b++)
			expected[b] = LlamaTransformerHandler.rmsNorm(x[b], weight, eps);

		CudaBindings cuda = CudaBindings.instance();
		RmsNormKernel kernel = RmsNormKernel.tryLoad();
		long xBytes = (long) batch * dim * Float.BYTES;
		long weightBytes = (long) dim * Float.BYTES;

		MemorySegment dX = cuda.deviceMalloc(ctx.deviceIndex(), xBytes);
		MemorySegment dWeight = cuda.deviceMalloc(ctx.deviceIndex(), weightBytes);
		MemorySegment dOut = cuda.deviceMalloc(ctx.deviceIndex(), xBytes);
		try {
			try (Arena staging = Arena.ofConfined()) {
				MemorySegment hostX = staging.allocate(xBytes);
				for (int b = 0; b < batch; b++)
					MemorySegment.copy(x[b], 0, hostX, JAVA_FLOAT, (long) b * dim * Float.BYTES, dim);
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpy, dX, hostX, xBytes, CudaBindings.H2D),
						"cudaMemcpy(rmsNorm xBatch H2D)");

				MemorySegment hostWeight = staging.allocate(weightBytes);
				MemorySegment.copy(weight, 0, hostWeight, JAVA_FLOAT, 0, dim);
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpy, dWeight, hostWeight, weightBytes, CudaBindings.H2D),
						"cudaMemcpy(rmsNorm weight H2D)");

				// Launch on the default stream (null); the blocking cudaMemcpy below
				// synchronizes with it before reading the result.
				kernel.launch(dX, dWeight, dOut, batch, dim, eps, null);

				MemorySegment hostOut = staging.allocate(xBytes);
				CudaBindings.check(
						CudaBindings.callInt(cuda.cudaMemcpy, hostOut, dOut, xBytes, CudaBindings.D2H),
						"cudaMemcpy(rmsNorm outBatch D2H)");

				for (int b = 0; b < batch; b++) {
					float[] actual = new float[dim];
					MemorySegment.copy(hostOut, JAVA_FLOAT, (long) b * dim * Float.BYTES, actual, 0, dim);
					assertThat(actual).as("row " + b).containsExactly(expected[b], within(TOL));
				}
			}
		} finally {
			cuda.deviceFree(dX);
			cuda.deviceFree(dWeight);
			cuda.deviceFree(dOut);
		}
	}
}
