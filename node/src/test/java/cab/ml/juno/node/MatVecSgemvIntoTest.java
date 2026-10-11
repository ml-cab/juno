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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code MatVec.sgemvInto}: the non-allocating single-row form is bit-identical to
 * the allocating {@code sgemv} for every backend, writes into the caller's array,
 * and the allocating form keeps working.
 */
class MatVecSgemvIntoTest {

	private static final int ROWS = 384;
	private static final int COLS = 512;

	@Test
	void cpuHostWeightsMatchTheAllocatingForm() {
		MatVec mv = CpuMatVec.INSTANCE;
		float[] a = floats(ROWS * COLS, 1);
		float[] x = floats(COLS, 2);
		float[] expected = mv.sgemv(a, x, ROWS, COLS);
		float[] out = new float[ROWS + 3];
		out[ROWS] = 7f;
		mv.sgemvInto(a, x, out, ROWS, COLS);
		for (int r = 0; r < ROWS; r++)
			assertThat(out[r]).as("row %d", r).isEqualTo(expected[r]);
		assertThat(out[ROWS]).as("nothing written past rows").isEqualTo(7f);
		assertThatThrownBy(() -> mv.sgemvInto(a, x, new float[ROWS - 1], ROWS, COLS))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void theDefaultFormCopiesFromTheAllocatingOne() {
		MatVec plain = (A, x, rows, cols) -> {
			float[] y = new float[rows];
			for (int r = 0; r < rows; r++)
				y[r] = r + x[0];
			return y;
		};
		float[] out = new float[4];
		plain.sgemvInto(new float[8], new float[] { 0.5f, 0f }, out, 4, 2);
		assertThat(out).containsExactly(0.5f, 1.5f, 2.5f, 3.5f);
	}

	@Test
	@Tag("gpu")
	void cudaDeviceWeightsMatchTheAllocatingForm() {
		assumeTrue(CudaAvailability.isAvailable(), "CUDA not available");
		GpuContext ctx = GpuContext.init(0);
		CudaMatVec mv = new CudaMatVec(ctx);
		float[] a = floats(ROWS * COLS, 3);
		float[] x = floats(COLS, 4);
		try (DeviceFloatMatrix f32 = DeviceFloatMatrix.upload(ctx, a, ROWS, COLS);
				DeviceHalfMatrix f16 = DeviceHalfMatrix.uploadFromFloat32(ctx, a, ROWS, COLS);
				DeviceQ4KMatrix q4 = DeviceQ4KMatrix.upload(ctx,
						CpuIntegerKernelsTestSupport.synthetic(12, ROWS, COLS, new Random(5)).data(), ROWS, COLS)) {
			float[] out = new float[ROWS];
			mv.sgemvInto(f32, x, out);
			assertThat(out).containsExactly(mv.sgemv(f32, x));
			mv.sgemvInto(f16, x, out);
			assertThat(out).containsExactly(mv.sgemv(f16, x));
			mv.sgemvInto(q4, x, out);
			assertThat(out).containsExactly(mv.sgemv(q4, x));
		}
	}

	private static float[] floats(int n, long seed) {
		Random rnd = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) rnd.nextGaussian();
		return v;
	}
}
