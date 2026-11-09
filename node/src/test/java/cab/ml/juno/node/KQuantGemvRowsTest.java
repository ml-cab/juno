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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The decode-batch matmul over packed K-quant weights: {@code Y = A X} for N
 * activation rows, reading each weight once for all of them. A request's reply must
 * not depend on what is batched with it, so every row has to be bit for bit what one
 * request's GEMV computes, and that is what this holds it to.
 */
@Tag("gpu")
@DisplayName("Decode-batch K-quant matmul - each row bit for bit one request's GEMV")
class KQuantGemvRowsTest {

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		mv = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@ParameterizedTest(name = "GGUF type {0}")
	@ValueSource(ints = { QuantizationLayout.TYPE_Q4_K, QuantizationLayout.TYPE_Q5_K, QuantizationLayout.TYPE_Q6_K })
	@DisplayName("every row equals one request's GEMV bit for bit, at any batch size and shape")
	void rowsAreBitIdenticalToSingleRowGemv(int type) {
		Random rnd = new Random(31L + type);
		int[][] shapes = { { 256, 256 }, { 1024, 256 }, { 256, 1024 }, { 512, 2048 } };
		for (int[] shape : shapes) {
			try (DeviceQ4KMatrix a = matrix(type, shape[0], shape[1], rnd)) {
				for (int n : new int[] { 1, 2, 3, 5, 8, 9, 17 }) {
					float[][] x = rows(n, shape[1], rnd);
					float[][] y = new float[n][shape[0]];

					mv.sgemvRowsInto(a, x, y);

					for (int b = 0; b < n; b++)
						assertBits(y[b], mv.sgemv(a, x[b]), "type " + type + " " + shape[0] + "x" + shape[1] + " n=" + n
								+ " row " + b);
				}
			}
		}
	}

	@ParameterizedTest(name = "batch of {0}")
	@ValueSource(ints = { 2, 8, 9 })
	@DisplayName("a row's result does not depend on the other rows of its batch")
	void aRowDoesNotDependOnItsNeighbours(int n) {
		Random rnd = new Random(77);
		try (DeviceQ4KMatrix a = matrix(QuantizationLayout.TYPE_Q4_K, 1024, 256, rnd)) {
			float[][] x = rows(n, 256, rnd);
			float[][] first = new float[n][1024];
			mv.sgemvRowsInto(a, x, first);

			float[] kept = x[0].clone();
			for (int b = 1; b < n; b++)
				x[b] = rows(1, 256, rnd)[0];
			float[][] second = new float[n][1024];
			mv.sgemvRowsInto(a, x, second);

			assertThat(x[0]).containsExactly(kept);
			assertBits(second[0], first[0], "row 0 with other neighbours");
		}
	}

	@Test
	@DisplayName("output rows must be the matrix's width; a closed matrix is refused")
	void refusesBadArguments() {
		Random rnd = new Random(5);
		DeviceQ4KMatrix a = matrix(QuantizationLayout.TYPE_Q4_K, 256, 256, rnd);
		float[][] x = rows(2, 256, rnd);
		assertThatThrownBy(() -> mv.sgemvRowsInto(a, x, new float[2][255]))
				.isInstanceOf(IllegalArgumentException.class);
		a.close();
		assertThatThrownBy(() -> mv.sgemvRowsInto(a, x, new float[2][256])).isInstanceOf(IllegalStateException.class);
	}

	private static void assertBits(float[] actual, float[] expected, String what) {
		assertThat(actual).as(what).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": element " + i + " is " + actual[i] + ", one request's GEMV gives "
						+ expected[i]);
	}

	private static DeviceQ4KMatrix matrix(int type, int rows, int cols, Random rnd) {
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = -1f + 2f * rnd.nextFloat();
		return DeviceQ4KMatrix.upload(ctx, GgufKQuantCodec.encode(host, type), rows, cols, type);
	}

	private static float[][] rows(int n, int cols, Random rnd) {
		float[][] x = new float[n][cols];
		for (float[] row : x)
			for (int i = 0; i < cols; i++)
				row[i] = -2f + 4f * rnd.nextFloat();
		return x;
	}
}
