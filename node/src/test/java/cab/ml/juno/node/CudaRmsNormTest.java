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

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Validates the {@link CudaRmsNorm} handler-facing wrapper (device malloc /
 * grow-on-demand scratch, H2D/D2H staging, weight upload) end to end, on top
 * of {@link RmsNormKernelParityTest}'s isolation of the raw kernel launch
 * itself.
 *
 * <p>{@link CudaRmsNorm} is deliberately <b>not</b> constructed by
 * {@code LlamaTransformerHandler} today (see its class javadoc and
 * {@code docs/infra-plan/PLAN-Infra-Tier19.md}'s "measured regression" note —
 * a live A/B on real TinyLlama decode found the independent per-call
 * H2D/kernel/D2H round trip costs ~11x more than the scalar CPU path it would
 * replace). This test exists so the wrapper stays correctness-verified while
 * dormant, ready to re-activate once a device-resident-activation redesign
 * removes the round trip.
 */
@Tag("gpu")
@DisplayName("CudaRmsNorm.normalizeBatch — end-to-end wrapper correctness")
class CudaRmsNormTest {

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
	@DisplayName("tryCreate returns null for a non-CUDA context")
	void tryCreate_returnsNull_forNonCudaBackend() {
		assertThat(CudaRmsNorm.tryCreate(null)).isNull();
	}

	@Test
	@DisplayName("normalizeBatch matches scalar CPU reference and reuses output arrays")
	void normalizeBatch_matchesReference_andReusesOutputArrays() {
		CudaRmsNorm gpu = CudaRmsNorm.tryCreate(ctx);
		assertThat(gpu).isNotNull();

		int batch = 8;
		int dim = 2048;
		float eps = 1e-5f;
		Random rng = new Random(2026);
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

		// Pre-allocate output rows (zero-alloc workspace convention) to verify
		// normalizeBatch reuses them rather than replacing the array references.
		float[][] out = new float[batch][dim];
		float[] originalRow0 = out[0];

		boolean dispatched = gpu.normalizeBatch(x, weight, eps, out);
		assertThat(dispatched).isTrue();
		assertThat(out[0]).isSameAs(originalRow0);

		for (int b = 0; b < batch; b++)
			assertThat(out[b]).as("row " + b).containsExactly(expected[b], within(1e-4f));
	}

	@Test
	@DisplayName("normalizeBatch grows scratch buffers correctly across increasing batch/dim calls")
	void normalizeBatch_growsScratchAcrossCalls() {
		CudaRmsNorm gpu = CudaRmsNorm.tryCreate(ctx);
		assertThat(gpu).isNotNull();
		float eps = 1e-5f;
		Random rng = new Random(7);

		int[] batches = { 1, 4, 16 };
		int[] dims = { 512, 4096, 1024 };
		for (int t = 0; t < batches.length; t++) {
			int batch = batches[t];
			int dim = dims[t];
			float[][] x = new float[batch][dim];
			for (int b = 0; b < batch; b++)
				for (int i = 0; i < dim; i++)
					x[b][i] = rng.nextFloat() - 0.5f;
			float[] weight = new float[dim];
			for (int i = 0; i < dim; i++)
				weight[i] = rng.nextFloat();

			float[][] expected = new float[batch][];
			for (int b = 0; b < batch; b++)
				expected[b] = LlamaTransformerHandler.rmsNorm(x[b], weight, eps);

			float[][] out = new float[batch][];
			assertThat(gpu.normalizeBatch(x, weight, eps, out)).isTrue();
			for (int b = 0; b < batch; b++)
				assertThat(out[b]).as("iteration " + t + " row " + b).containsExactly(expected[b], within(1e-4f));
		}
	}
}
