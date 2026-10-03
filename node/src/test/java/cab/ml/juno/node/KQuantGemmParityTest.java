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

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The tiled packed-weight GEMM ({@link KQuantGemmKernel}) against two references,
 * for Q4_K, Q5_K and Q6_K at every batch width the prefill and continuous paths
 * produce:
 *
 * <ul>
 * <li>the CPU FP32 oracle (the weights dequantized in FP32, dotted in FP32), within
 * the band that Q8_1 activation quantization allows ({@link Q4KMmqParityTest#q8Tol});
 * <li>the fused GEMV kernel, column by column. Both quantize the activation to the
 * same Q8_1 bytes and integer-dot the same packed weights, so they differ only in
 * the order of the float accumulation: a much tighter band, which catches a layout
 * slip that the oracle band would let through.
 * </ul>
 *
 * Shapes cover a square matrix, both FFN orientations, a {@code cols} of a single
 * super-block, and a row count that is not a multiple of the row tile.
 */
@Tag("gpu")
@DisplayName("Tiled K-quant GEMM parity (Q4_K / Q5_K / Q6_K)")
class KQuantGemmParityTest {

	private static final int[] WIDTHS = { 1, 2, 8, 9, 16, 32, 64, 128, 512, 741 };

	/** {rows, cols}. */
	private static final int[][] SHAPES = {
			{ 512, 512 },    // square
			{ 1376, 512 },   // FFN gate/up orientation; 1376 is not a multiple of the row tile
			{ 512, 1536 },   // FFN down orientation, six super-blocks per row
			{ 100, 256 },    // one super-block per row, rows not a multiple of the tile
	};

	private static final int[] TYPES = { QuantizationLayout.TYPE_Q4_K, QuantizationLayout.TYPE_Q5_K,
			QuantizationLayout.TYPE_Q6_K };

	private static GpuContext ctx;
	private static CudaMatVec mv;
	private static KQuantGemmKernel kernel;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA - skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "K-quant GEMV kernel unavailable");
		kernel = KQuantGemmKernel.tryLoad();
		assertThat(kernel).as("tiled K-quant GEMM kernel loads").isNotNull();
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	static java.util.stream.Stream<Arguments> typesAndWidths() {
		return Arrays.stream(TYPES).boxed()
				.flatMap(t -> Arrays.stream(WIDTHS).mapToObj(w -> Arguments.of(t, w)));
	}

	@ParameterizedTest(name = "type {0}, W={1}")
	@MethodSource("typesAndWidths")
	@DisplayName("matches the CPU FP32 oracle and the fused GEMV on every shape")
	void matchesOracleAndGemv(int typeId, int batch) {
		Random rnd = new Random(31L * typeId + batch);
		for (int[] shape : SHAPES) {
			int rows = shape[0], cols = shape[1];
			byte[] raw = randomEncoded(rnd, rows, cols, typeId);
			float[] w = GgufKQuantCodec.decodeRows(raw, typeId, rows, cols);
			float[][] x = randomRows(rnd, batch, cols);
			float[] sigma = q8NoiseSigma(w, rows, cols);
			try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId)) {
				float[][] got = runTiled(a, x, rows);
				for (int b = 0; b < batch; b++) {
					float[] oracle = dot(w, x[b], rows, cols);
					float[] gemv = mv.sgemv(a, x[b]);
					String where = "type " + typeId + " " + rows + "x" + cols + " W=" + batch + " col " + b;
					for (int r = 0; r < rows; r++) {
						assertThat(got[b][r]).as(where + " row " + r + " vs GEMV")
								.isCloseTo(gemv[r], within(accumulationTol(gemv[r])));
						assertThat(got[b][r]).as(where + " row " + r + " vs oracle")
								.isCloseTo(oracle[r], within(Math.max(6f * sigma[r], Q4KMmqParityTest.q8Tol(oracle[r]))));
					}
				}
			}
		}
	}

	@Test
	@DisplayName("an output row stride wider than the matrix writes only the matrix's column range")
	void strideWritesOnlyItsRange() {
		int rows = 192, cols = 512, batch = 37, ldc = 2 * rows;
		Random rnd = new Random(9);
		byte[] raw = randomEncoded(rnd, rows, cols, QuantizationLayout.TYPE_Q4_K);
		float[][] x = randomRows(rnd, batch, cols);
		try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, QuantizationLayout.TYPE_Q4_K)) {
			float[][] dense = runTiled(a, x, rows);
			float[][] wide = run(a, x, ldc, rows);
			for (int b = 0; b < batch; b++) {
				assertThat(Arrays.copyOfRange(wide[b], 0, rows)).as("row " + b).containsExactly(dense[b]);
				for (int r = rows; r < ldc; r++)
					assertThat(wide[b][r]).as("row " + b + " untouched column " + r).isEqualTo(SENTINEL);
			}
		}
	}

	static java.util.stream.Stream<Arguments> typesAndBatchedWidths() {
		return Arrays.stream(TYPES).boxed()
				.flatMap(t -> java.util.stream.IntStream.of(9, 64, 512).mapToObj(w -> Arguments.of(t, w)));
	}

	/**
	 * {@link CudaMatVec#sgemm(DeviceQ4KMatrix, float[][])} above eight rows stages the
	 * window as FP16, as it always has, and multiplies it with this kernel: its
	 * output is this kernel's on the FP16-rounded window, bit for bit, and the
	 * weights are never dequantized.
	 */
	@ParameterizedTest(name = "type {0}, W={1}")
	@MethodSource("typesAndBatchedWidths")
	@DisplayName("CudaMatVec's batched path is this kernel on the FP16-rounded window")
	void batchedSgemmRunsThisKernel(int typeId, int batch) {
		int rows = 320, cols = 768;
		Random rnd = new Random(77L * typeId + batch);
		byte[] raw = randomEncoded(rnd, rows, cols, typeId);
		float[][] x = randomRows(rnd, batch, cols);
		float[][] xh = new float[batch][cols];
		for (int b = 0; b < batch; b++)
			for (int i = 0; i < cols; i++)
				xh[b][i] = Float.float16ToFloat(Float.floatToFloat16(x[b][i]));
		try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId)) {
			float[][] expected = runTiled(a, xh, rows);
			float[][] got = mv.sgemm(a, x);
			for (int b = 0; b < batch; b++)
				assertThat(got[b]).as("type " + typeId + " W=" + batch + " row " + b).containsExactly(expected[b]);
		}
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	private static final float SENTINEL = -12345.5f;

	/**
	 * Q8_1 activation rounding: each activation moves by up to half a step of its
	 * block's scale ({@code amax / 127}, and every activation here is in [-1, 1]),
	 * uniformly, so output row {@code r} carries noise with standard deviation about
	 * {@code ||w_r|| / 127 / sqrt(12)}. The oracle band is six of those, or
	 * {@link Q4KMmqParityTest#q8Tol} where that is wider: across the millions of
	 * outputs this test checks, a band of one fixed width either fails on noise at
	 * wide rows or hides a layout error at narrow ones.
	 */
	private static float[] q8NoiseSigma(float[] w, int rows, int cols) {
		float[] sigma = new float[rows];
		for (int r = 0; r < rows; r++) {
			double ss = 0;
			for (int c = 0; c < cols; c++) {
				double v = w[r * cols + c];
				ss += v * v;
			}
			sigma[r] = (float) (Math.sqrt(ss) / 127.0 / Math.sqrt(12.0));
		}
		return sigma;
	}

	/** Float accumulation order only: the integer dot products are identical. */
	private static float accumulationTol(float expected) {
		return Math.max(2e-4f, Math.abs(expected) * 2e-5f);
	}

	private static float[][] runTiled(DeviceQ4KMatrix a, float[][] x, int rows) {
		return run(a, x, rows, rows);
	}

	/**
	 * Uploads {@code x}, quantizes it to Q8_1 and runs the tiled GEMM with output
	 * row stride {@code ldc} into a buffer pre-filled with {@link #SENTINEL}.
	 */
	private static float[][] run(DeviceQ4KMatrix a, float[][] x, int ldc, int rows) {
		int batch = x.length, cols = a.cols();
		try (ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation in = chain.allocate(batch, cols);
			ResidentActivation out = chain.allocate(batch, ldc);
			MemorySegment dQ8 = chain.allocateScratch(KQuantGemmKernel.q8Bytes(batch, cols));
			in.upload(x);
			float[][] fill = new float[batch][ldc];
			for (float[] row : fill)
				Arrays.fill(row, SENTINEL);
			out.upload(fill);
			kernel.multiply(a, in.devicePointer(), dQ8, out.devicePointer(), batch, ldc, chain.stream());
			chain.sync();
			out.markWritten(batch);
			float[][] result = new float[batch][];
			out.materialize(result);
			return result;
		}
	}

	private static float[] dot(float[] w, float[] x, int rows, int cols) {
		float[] y = new float[rows];
		for (int r = 0; r < rows; r++) {
			float s = 0f;
			int base = r * cols;
			for (int c = 0; c < cols; c++)
				s += w[base + c] * x[c];
			y[r] = s;
		}
		return y;
	}

	private static byte[] randomEncoded(Random rnd, int rows, int cols, int typeId) {
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		return GgufKQuantCodec.encode(host, typeId);
	}

	private static float[][] randomRows(Random rnd, int batch, int cols) {
		float[][] x = new float[batch][cols];
		for (float[] row : x)
			for (int i = 0; i < cols; i++)
				row[i] = (rnd.nextFloat() * 2f) - 1f;
		return x;
	}
}
