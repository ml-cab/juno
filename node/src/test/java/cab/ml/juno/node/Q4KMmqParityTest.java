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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Parity: fused CUDA Q4_K GEMV vs CPU {@link LlamaTransformerHandler} oracle.
 */
@Tag("gpu")
@DisplayName("Q4K MMQ parity")
class Q4KMmqParityTest {

	@Test
	@DisplayName("GPU fused Q4_K GEMV matches CPU matVecQ4K within tolerance")
	void gpu_matches_cpu_oracle() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
		assumeTrue(kernel != null, "Q4K MMQ kernel failed to load");

		int rows = 64;
		int cols = 256;
		Random rnd = new Random(42);
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;

		float[] expected = new float[rows];
		LlamaTransformerHandler.matVecInto(
				new GgufReader.QuantizedTensor("t", QuantizationLayout.TYPE_Q4_K, (long) rows * cols, raw),
				x, expected, rows, cols);

		try (GpuContext ctx = GpuContext.init(0);
				DeviceQ4KMatrix dA = DeviceQ4KMatrix.upload(ctx, raw, rows, cols)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			float[] got = mv.sgemv(dA, x);
			assertThat(got).hasSize(rows);
			for (int i = 0; i < rows; i++)
				assertThat(got[i]).as("row " + i).isCloseTo(expected[i], within(1e-2f));
		}
	}
}
