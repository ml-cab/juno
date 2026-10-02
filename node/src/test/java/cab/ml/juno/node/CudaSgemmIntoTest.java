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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static cab.ml.juno.node.MatVecSgemmIntoTest.allocatedBytes;
import static cab.ml.juno.node.MatVecSgemmIntoTest.randomFloats;
import static cab.ml.juno.node.MatVecSgemmIntoTest.randomRows;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link CudaMatVec}'s non-allocating batched form for the three device weight
 * types: bit-identical to the allocating {@code sgemm} at window sizes on both
 * sides of the batched-dispatch threshold (8 rows), at a 512-token prefill window
 * and at the vision encoder's batch width (741), written into the caller's rows;
 * and at a prefill window, the output is not allocated (the interface default
 * copies from a newly allocated one).
 */
@Tag("gpu")
@DisplayName("CudaMatVec.sgemmInto - device weights, bit-identical to sgemm, no output allocation")
class CudaSgemmIntoTest {

	private static final int ROWS = 256;
	private static final int COLS = 512;
	private static final int Q4K_BLOCK_BYTES = 144;

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		mv = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@ParameterizedTest(name = "W={0}")
	@ValueSource(ints = { 1, 2, 8, 9, 32, 512, 741 })
	@DisplayName("FP16 weights")
	void half(int batch) {
		try (DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 1), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 2);
			assertSameAsAllocating(mv.sgemm(a, x), y -> mv.sgemmInto(a, x, y), batch);
		}
	}

	@ParameterizedTest(name = "W={0}")
	@ValueSource(ints = { 1, 2, 8, 9, 32, 512, 741 })
	@DisplayName("Q4_K weights")
	void q4k(int batch) {
		assumeTrue(mv.supportsQ4KMmq(), "K-quant MMQ kernel unavailable");
		try (DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 3), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 4);
			assertSameAsAllocating(mv.sgemm(a, x), y -> mv.sgemmInto(a, x, y), batch);
		}
	}

	@ParameterizedTest(name = "W={0}")
	@ValueSource(ints = { 1, 2, 9, 32, 512 })
	@DisplayName("FP32 weights")
	void fp32(int batch) {
		try (DeviceFloatMatrix a = mv.upload(randomFloats(ROWS * COLS, 5), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 6);
			assertSameAsAllocating(mv.sgemm(a, x), y -> mv.sgemmInto(a, x, y), batch);
		}
	}

	@Test
	@DisplayName("a 512-row FP16 or Q4_K window does not allocate its output")
	void prefillWindow_doesNotAllocateTheOutput() {
		int batch = 512;
		long outputBytes = (long) batch * ROWS * Float.BYTES;
		float[][] x = randomRows(batch, COLS, 7);
		float[][] y = new float[batch][ROWS];
		try (DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 8), ROWS, COLS)) {
			for (int i = 0; i < 20; i++)
				mv.sgemmInto(a, x, y);
			assertThat(allocatedBytes(() -> mv.sgemmInto(a, x, y))).as("FP16: bytes allocated (output is %d)",
					outputBytes).isLessThan(outputBytes / 8);
		}
		if (!mv.supportsQ4KMmq())
			return;
		try (DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 9), ROWS, COLS)) {
			for (int i = 0; i < 20; i++)
				mv.sgemmInto(a, x, y);
			assertThat(allocatedBytes(() -> mv.sgemmInto(a, x, y))).as("Q4_K: bytes allocated (output is %d)",
					outputBytes).isLessThan(outputBytes / 8);
		}
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	private static void assertSameAsAllocating(float[][] expected, java.util.function.Consumer<float[][]> into,
			int batch) {
		float[][] y = new float[batch][ROWS + 3];
		float[][] sameRows = y.clone();
		into.accept(y);
		for (int b = 0; b < batch; b++) {
			assertThat(y[b]).as("row " + b + " is the caller's array").isSameAs(sameRows[b]);
			assertThat(java.util.Arrays.copyOf(y[b], ROWS)).as("row " + b).containsExactly(expected[b]);
			assertThat(java.util.Arrays.copyOfRange(y[b], ROWS, ROWS + 3)).as("row " + b + " tail")
					.containsExactly(0f, 0f, 0f);
		}
	}

	/** Random Q4_K super-blocks with small finite FP16 scales, so the dequantized weights stay finite. */
	private static byte[] randomQ4K(int rows, int cols, long seed) {
		Random r = new Random(seed);
		int blocks = rows * (cols / 256);
		byte[] raw = new byte[blocks * Q4K_BLOCK_BYTES];
		r.nextBytes(raw);
		short scale = Float.floatToFloat16(0.01f);
		for (int b = 0; b < blocks; b++) {
			int off = b * Q4K_BLOCK_BYTES;
			raw[off] = (byte) scale;
			raw[off + 1] = (byte) (scale >> 8);
			raw[off + 2] = (byte) scale;
			raw[off + 3] = (byte) (scale >> 8);
		}
		return raw;
	}
}
