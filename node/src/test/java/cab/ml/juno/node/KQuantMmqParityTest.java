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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Parity: fused CUDA Q5_K / Q6_K GEMV vs the CPU {@link LlamaTransformerHandler}
 * oracle, on the same ragged shapes as {@link Q4KMmqParityTest}, plus a mixed-type
 * {@code sgemvSameX} batch (one x upload, per-matrix kernel dispatch).
 */
@Tag("gpu")
@DisplayName("K-quant MMQ parity (Q5_K / Q6_K)")
class KQuantMmqParityTest {

	private static final int[][] SHAPES = {
			{ 1, 256 },
			{ 13, 768 },
			{ 3001, 1280 },
			{ 50, 3072 },
	};

	@ParameterizedTest(name = "type {0}")
	@ValueSource(ints = { QuantizationLayout.TYPE_Q5_K, QuantizationLayout.TYPE_Q6_K })
	@DisplayName("GPU fused GEMV matches CPU oracle")
	void gpu_matches_cpu_oracle(int typeId) {
		assumeGpu();
		Random rnd = new Random(11 + typeId);
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			for (int[] shape : SHAPES) {
				int rows = shape[0], cols = shape[1];
				byte[] raw = randomEncoded(rnd, rows, cols, typeId);
				float[] x = randomVector(rnd, cols);
				float[] expected = new float[rows];
				LlamaTransformerHandler.matVecInto(
						new GgufReader.QuantizedTensor("t", typeId, (long) rows * cols, raw),
						x, expected, rows, cols);
				try (DeviceQ4KMatrix dA = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId)) {
					assertThat(dA.quantType()).isEqualTo(typeId);
					float[] got = mv.sgemv(dA, x);
					assertThat(got).hasSize(rows);
					for (int i = 0; i < rows; i++)
						assertThat(got[i]).as("type " + typeId + " " + rows + "x" + cols + " row " + i)
								.isCloseTo(expected[i], within(Q4KMmqParityTest.q8Tol(expected[i])));
				}
			}
		}
	}

	@Test
	@DisplayName("sgemvSameX dispatches per-matrix kernels for a mixed Q4_K / Q5_K / Q6_K batch")
	void sameX_mixed_types() {
		assumeGpu();
		int cols = 1024;
		int[] types = { QuantizationLayout.TYPE_Q4_K, QuantizationLayout.TYPE_Q5_K, QuantizationLayout.TYPE_Q6_K };
		int[] rows = { 96, 40, 200 };
		Random rnd = new Random(5);
		float[] x = randomVector(rnd, cols);
		byte[][] raw = new byte[3][];
		float[][] expected = new float[3][];
		for (int i = 0; i < 3; i++) {
			raw[i] = randomEncoded(rnd, rows[i], cols, types[i]);
			expected[i] = new float[rows[i]];
			LlamaTransformerHandler.matVecInto(
					new GgufReader.QuantizedTensor("t" + i, types[i], (long) rows[i] * cols, raw[i]),
					x, expected[i], rows[i], cols);
		}
		try (GpuContext ctx = GpuContext.init(0);
				DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw[0], rows[0], cols, types[0]);
				DeviceQ4KMatrix b = DeviceQ4KMatrix.upload(ctx, raw[1], rows[1], cols, types[1]);
				DeviceQ4KMatrix c = DeviceQ4KMatrix.upload(ctx, raw[2], rows[2], cols, types[2])) {
			CudaMatVec mv = new CudaMatVec(ctx);
			float[][] got = mv.sgemvSameX(new DeviceQ4KMatrix[] { a, b, c }, x);
			assertThat(got).hasNumberOfRows(3);
			for (int i = 0; i < 3; i++) {
				assertThat(got[i]).hasSize(rows[i]);
				for (int r = 0; r < rows[i]; r++)
					assertThat(got[i][r]).as("matrix " + i + " row " + r)
							.isCloseTo(expected[i][r], within(Q4KMmqParityTest.q8Tol(expected[i][r])));
			}
		}
	}

	private static void assumeGpu() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "K-quant MMQ kernel failed to load");
	}

	private static byte[] randomEncoded(Random rnd, int rows, int cols, int typeId) {
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		return GgufKQuantCodec.encode(host, typeId);
	}

	private static float[] randomVector(Random rnd, int n) {
		float[] x = new float[n];
		for (int i = 0; i < n; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;
		return x;
	}
}
