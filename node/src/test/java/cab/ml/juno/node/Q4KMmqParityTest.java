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
				assertThat(got[i]).as("row " + i).isCloseTo(expected[i], within(q8Tol(expected[i])));
		}
	}

	/**
	 * The kernel maps one row per block and 16 super-blocks per iteration; exercise
	 * row counts and super-block counts that do not divide evenly, plus a
	 * single-super-block row, so tail handling is covered.
	 */
	@Test
	@DisplayName("GPU fused Q4_K GEMV matches CPU on ragged row / super-block counts")
	void gpu_matches_cpu_on_ragged_shapes() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4K MMQ kernel failed to load");

		int[][] shapes = {
				{ 1, 256 },       // one row, one super-block: only 8 of 32 lanes active
				{ 13, 768 },      // several rows, 3 super-blocks (< 16 in-flight)
				{ 3001, 1280 },   // rows not a multiple of 12, 5 super-blocks
				{ 50, 3072 },     // Phi-3 hidden width, 12 super-blocks
		};
		Random rnd = new Random(7);
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			for (int[] shape : shapes) {
				int rows = shape[0], cols = shape[1];
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
				try (DeviceQ4KMatrix dA = DeviceQ4KMatrix.upload(ctx, raw, rows, cols)) {
					float[] got = mv.sgemv(dA, x);
					assertThat(got).hasSize(rows);
					for (int i = 0; i < rows; i++)
						assertThat(got[i]).as(rows + "x" + cols + " row " + i)
								.isCloseTo(expected[i], within(q8Tol(expected[i])));
				}
			}
		}
	}

	/**
	 * Device GEMV quantizes {@code x} to Q8_1; error grows with K and with
	 * poorly-scaled 32-element blocks. Floor + relative band catch layout bugs
	 * (O(1) or sign flips) without failing on that activation quant.
	 */
	static float q8Tol(float expected) {
		return Math.max(0.15f, Math.abs(expected) * 0.12f);
	}
}
