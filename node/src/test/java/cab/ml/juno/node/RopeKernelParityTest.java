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

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolates {@link RopeKernel#launch} from the handler-facing {@link CudaRope}:
 * rotate a known batch of rows on the device, read the result back, and compare
 * against {@link LlamaTransformerHandler#rope}, the scalar CPU path the kernel
 * replaces.
 *
 * <p>The kernel computes the angle and its sine and cosine in double, as the CPU
 * path does, and rounds each rotation step separately rather than fusing it, so
 * the two agree to within one float rounding of the sine or cosine. The
 * tolerance below is that bound for inputs in {@code [-2, 2]}; the largest
 * divergence observed is printed so a reader can see how close to exact it is.
 */
@Tag("gpu")
@DisplayName("RopeKernel.launch - parity vs LlamaTransformerHandler.rope")
class RopeKernelParityTest {

	/** One float ulp of a sine or cosine, scaled by the largest input magnitude, twice over. */
	private static final float TOL = 1e-6f;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("one decode row near the end of the context window matches the CPU path")
	void decodeRow_atLargePosition_matchesCpu() {
		// 30000 is close to the 32768 sequence cap: the angle reaches 30000 radians,
		// where a single-precision angle would already be off by about 2e-3.
		rotateAndCompare(1, 32, 64, 10000f, 30000, 11);
	}

	@Test
	@DisplayName("a prefill window of consecutive positions matches the CPU path row by row")
	void prefillWindow_consecutivePositions_matchCpu() {
		rotateAndCompare(64, 32, 64, 10000f, 1000, 12);
	}

	@Test
	@DisplayName("grouped-query K shape (few heads) matches the CPU path")
	void keyShape_fewHeads_matchesCpu() {
		rotateAndCompare(16, 4, 64, 10000f, 7, 13);
	}

	@Test
	@DisplayName("128-wide heads and a large rope base match the CPU path")
	void wideHeads_largeBase_matchCpu() {
		rotateAndCompare(8, 32, 128, 1_000_000f, 4096, 14);
	}

	@Test
	@DisplayName("position zero leaves every row unrotated, as on the CPU")
	void positionZero_isIdentity() {
		float maxDiff = rotateAndCompare(1, 8, 64, 10000f, 0, 15);
		assertThat(maxDiff).isZero();
	}

	@Test
	@DisplayName("inverse frequencies are the CPU path's own, computed in double")
	void inverseFrequencies_matchCpuExpression() {
		int headDim = 64;
		float theta = 10000f;
		double[] table = RopeKernel.inverseFrequencies(headDim, theta);
		assertThat(table).hasSize(headDim / 2);
		for (int i = 0; i < headDim / 2; i++)
			assertThat(table[i]).isEqualTo(1.0 / Math.pow(theta, (2.0 * i) / headDim));
	}

	/**
	 * Rotates {@code rows} rows of {@code nHeads * headDim} values starting at
	 * {@code startPos} on the device and on the CPU, asserts they agree within
	 * {@link #TOL}, and returns the largest divergence seen.
	 */
	private static float rotateAndCompare(int rows, int nHeads, int headDim, float theta, int startPos, long seed) {
		int width = nHeads * headDim;
		Random rng = new Random(seed);
		float[][] x = new float[rows][width];
		for (float[] row : x)
			for (int i = 0; i < width; i++)
				row[i] = rng.nextFloat() * 4f - 2f;

		float[][] expected = new float[rows][];
		for (int r = 0; r < rows; r++) {
			expected[r] = x[r].clone();
			LlamaTransformerHandler.rope(expected[r], startPos + r, nHeads, headDim, theta);
		}

		GpuBindings gpu = ctx.bindings();
		long xBytes = (long) rows * width * Float.BYTES;
		double[] invFreq = RopeKernel.inverseFrequencies(headDim, theta);
		long tableBytes = (long) invFreq.length * Double.BYTES;
		MemorySegment dX = gpu.deviceMalloc(ctx.deviceIndex(), xBytes);
		MemorySegment dTable = gpu.deviceMalloc(ctx.deviceIndex(), tableBytes);
		float[][] actual = new float[rows][width];
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment hostX = staging.allocate(xBytes);
			for (int r = 0; r < rows; r++)
				MemorySegment.copy(x[r], 0, hostX, JAVA_FLOAT, (long) r * width * Float.BYTES, width);
			MemorySegment hostTable = staging.allocate(tableBytes, Double.BYTES);
			MemorySegment.copy(invFreq, 0, hostTable, JAVA_DOUBLE, 0, invFreq.length);
			GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dX, hostX, xBytes, GpuBindings.H2D), "x H2D");
			GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dTable, hostTable, tableBytes, GpuBindings.H2D),
					"table H2D");

			RopeKernel.tryLoad().launch(dX, dTable, rows, nHeads, headDim, startPos, null);

			MemorySegment hostOut = staging.allocate(xBytes);
			GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), hostOut, dX, xBytes, GpuBindings.D2H), "x D2H");
			for (int r = 0; r < rows; r++)
				MemorySegment.copy(hostOut, JAVA_FLOAT, (long) r * width * Float.BYTES, actual[r], 0, width);
		} finally {
			gpu.deviceFree(dX);
			gpu.deviceFree(dTable);
		}

		float maxDiff = 0f;
		for (int r = 0; r < rows; r++)
			for (int i = 0; i < width; i++)
				maxDiff = Math.max(maxDiff, Math.abs(actual[r][i] - expected[r][i]));
		System.out.printf("[rope parity] rows=%d heads=%d headDim=%d theta=%.0f startPos=%d max|diff|=%.3e%n",
				rows, nHeads, headDim, theta, startPos, maxDiff);
		assertThat(maxDiff).as("largest divergence from the CPU rotation").isLessThanOrEqualTo(TOL);
		return maxDiff;
	}
}
