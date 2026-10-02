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
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The prefill-window region's elementwise kernels against the CPU window path
 * they replace: the FP16 cast against {@code Float.floatToFloat16} (the host
 * packing of every GEMM input), SwiGLU against {@code LlamaTransformerHandler.silu}
 * followed by that cast, the residual and bias adds against float addition, and
 * split-half RoPE against {@code LlamaTransformerHandler.rope(..., SPLIT_HALF)}.
 *
 * <p>Window widths span the batched-GEMM dispatch threshold (1, 8, 9, 32, 512);
 * the SwiGLU widths are every sweep model's intermediate size (TinyLlama 5632,
 * Qwen2.5-3B 11008, Phi-3.5-mini 8192, Mistral 7B 14336, Qwen3-1.7B 6144).
 */
@Tag("gpu")
@DisplayName("PrefillWindowKernels - parity vs the CPU window path")
class PrefillWindowKernelsParityTest {

	private static final int[] WINDOWS = { 1, 8, 9, 32, 512 };
	private static final int[] INTERMEDIATE = { 5632, 11008, 8192, 14336, 6144 };

	private static GpuContext ctx;
	private static PrefillWindowKernels kernels;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		kernels = PrefillWindowKernels.tryLoad();
		assumeTrue(kernels != null, "prefill-window kernels failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("the FP16 cast is bit-identical to Float.floatToFloat16, ties, subnormals and overflow included")
	void toHalf_isBitIdenticalToTheHostPacking() {
		List<Float> special = new ArrayList<>(List.of(0f, -0f, 1f, -1f, 65504f, 65520f, 65519.99f, 1e6f, -1e6f,
				Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -Float.MIN_VALUE, 5.96e-8f,
				2.98e-8f, 6.1e-5f, 6.103515625e-05f, 1.00048828125f, 1.000732421875f, 2049f, 2051f));
		Random rng = new Random(21);
		int n = 1 << 20;
		float[] x = new float[n];
		for (int i = 0; i < n; i++) {
			if (i < special.size())
				x[i] = special.get(i);
			else if ((i & 3) == 0)
				x[i] = Float.intBitsToFloat(rng.nextInt() & 0x7f7fffff) * (rng.nextBoolean() ? 1 : -1);
			else
				x[i] = (float) rng.nextGaussian() * 8f;
		}

		GpuBindings gpu = ctx.bindings();
		long inBytes = (long) n * Float.BYTES;
		long outBytes = (long) n * Short.BYTES;
		MemorySegment dIn = gpu.deviceMalloc(ctx.deviceIndex(), inBytes);
		MemorySegment dOut = gpu.deviceMalloc(ctx.deviceIndex(), outBytes);
		short[] actual = new short[n];
		try (Arena a = Arena.ofConfined()) {
			MemorySegment h = a.allocate(inBytes);
			MemorySegment.copy(x, 0, h, JAVA_FLOAT, 0, n);
			copy(gpu, dIn, h, inBytes, GpuBindings.H2D);
			kernels.toHalf(dIn, dOut, n, null);
			MemorySegment ho = a.allocate(outBytes);
			copy(gpu, ho, dOut, outBytes, GpuBindings.D2H);
			MemorySegment.copy(ho, JAVA_SHORT, 0, actual, 0, n);
		} finally {
			gpu.deviceFree(dIn);
			gpu.deviceFree(dOut);
		}
		int mismatches = 0;
		for (int i = 0; i < n; i++)
			if (actual[i] != Float.floatToFloat16(x[i]))
				mismatches++;
		assertThat(mismatches).as("elements whose FP16 bits differ from Float.floatToFloat16").isZero();
	}

	static Stream<Arguments> swigluShapes() {
		List<Arguments> out = new ArrayList<>();
		for (int w : WINDOWS)
			for (int i : INTERMEDIATE)
				out.add(Arguments.of(w, i));
		return out.stream();
	}

	@ParameterizedTest(name = "W={0} I={1}")
	@MethodSource("swigluShapes")
	@DisplayName("SwiGLU to FP16 is within one FP16 ulp of the CPU loop followed by the host cast")
	void swiglu_matchesCpuLoopThenCast(int rows, int inter) {
		Random rng = new Random(31L * rows + inter);
		float[] gu = new float[rows * 2 * inter];
		for (int i = 0; i < gu.length; i++)
			gu[i] = (float) rng.nextGaussian() * 3f;

		short[] expected = new short[rows * inter];
		for (int r = 0; r < rows; r++)
			for (int i = 0; i < inter; i++) {
				float g = gu[r * 2 * inter + i];
				float u = gu[r * 2 * inter + inter + i];
				expected[r * inter + i] = Float.floatToFloat16(LlamaTransformerHandler.silu(g) * u);
			}

		GpuBindings gpu = ctx.bindings();
		long inBytes = (long) gu.length * Float.BYTES;
		long outBytes = (long) rows * inter * Short.BYTES;
		MemorySegment dIn = gpu.deviceMalloc(ctx.deviceIndex(), inBytes);
		MemorySegment dOut = gpu.deviceMalloc(ctx.deviceIndex(), outBytes);
		short[] actual = new short[rows * inter];
		try (Arena a = Arena.ofConfined()) {
			MemorySegment h = a.allocate(inBytes);
			MemorySegment.copy(gu, 0, h, JAVA_FLOAT, 0, gu.length);
			copy(gpu, dIn, h, inBytes, GpuBindings.H2D);
			kernels.swigluToHalf(dIn, dOut, rows, inter, null);
			MemorySegment ho = a.allocate(outBytes);
			copy(gpu, ho, dOut, outBytes, GpuBindings.D2H);
			MemorySegment.copy(ho, JAVA_SHORT, 0, actual, 0, actual.length);
		} finally {
			gpu.deviceFree(dIn);
			gpu.deviceFree(dOut);
		}
		int maxUlp = 0;
		int exact = 0;
		for (int i = 0; i < expected.length; i++) {
			int ulp = Math.abs(orderedHalf(actual[i]) - orderedHalf(expected[i]));
			maxUlp = Math.max(maxUlp, ulp);
			if (ulp == 0)
				exact++;
		}
		System.out.printf("[swiglu parity] W=%d I=%d exact %.4f%% max %d FP16 ulp%n", rows, inter,
				100.0 * exact / expected.length, maxUlp);
		assertThat(maxUlp).as("largest FP16 ulp distance from the CPU loop then the host cast").isLessThanOrEqualTo(1);
	}

	@ParameterizedTest(name = "W={0}")
	@MethodSource("normShapes")
	@DisplayName("the host-order RMS norm equals LlamaTransformerHandler.rmsNormInto bit for bit")
	void rmsNormHostOrder_isBitIdenticalToTheHostNorm(int rows, int dim) {
		Random rng = new Random(71L * rows + dim);
		float[] x = new float[rows * dim];
		float[] w = new float[dim];
		for (int i = 0; i < x.length; i++)
			x[i] = (float) rng.nextGaussian() * (i % 97 == 0 ? 40f : 1.5f); // a few outlier channels, as in real layers
		for (int i = 0; i < dim; i++)
			w[i] = 0.5f + rng.nextFloat();
		float eps = 1e-6f;

		GpuBindings gpu = ctx.bindings();
		long xBytes = (long) x.length * Float.BYTES;
		long wBytes = (long) dim * Float.BYTES;
		MemorySegment dx = gpu.deviceMalloc(ctx.deviceIndex(), xBytes);
		MemorySegment dw = gpu.deviceMalloc(ctx.deviceIndex(), wBytes);
		MemorySegment dOut = gpu.deviceMalloc(ctx.deviceIndex(), xBytes);
		float[] actual = new float[x.length];
		try (Arena a = Arena.ofConfined()) {
			MemorySegment hx = a.allocate(xBytes);
			MemorySegment hw = a.allocate(wBytes);
			MemorySegment.copy(x, 0, hx, JAVA_FLOAT, 0, x.length);
			MemorySegment.copy(w, 0, hw, JAVA_FLOAT, 0, dim);
			copy(gpu, dx, hx, xBytes, GpuBindings.H2D);
			copy(gpu, dw, hw, wBytes, GpuBindings.H2D);
			kernels.rmsNormHostOrder(dx, dw, dOut, rows, dim, eps, null);
			copy(gpu, hx, dOut, xBytes, GpuBindings.D2H);
			MemorySegment.copy(hx, JAVA_FLOAT, 0, actual, 0, x.length);
		} finally {
			gpu.deviceFree(dx);
			gpu.deviceFree(dw);
			gpu.deviceFree(dOut);
		}
		float[] row = new float[dim];
		float[] out = new float[dim];
		int mismatches = 0;
		for (int r = 0; r < rows; r++) {
			System.arraycopy(x, r * dim, row, 0, dim);
			LlamaTransformerHandler.rmsNormInto(row, w, eps, out);
			for (int i = 0; i < dim; i++)
				if (Float.floatToRawIntBits(actual[r * dim + i]) != Float.floatToRawIntBits(out[i]))
					mismatches++;
		}
		assertThat(mismatches).as("elements whose bits differ from the host norm").isZero();
	}

	static Stream<Arguments> normShapes() {
		List<Arguments> out = new ArrayList<>();
		for (int w : WINDOWS)
			for (int dim : new int[] { 2048, 2560, 3072, 4096 })
				out.add(Arguments.of(w, dim));
		return out.stream();
	}

	@Test
	@DisplayName("the residual add equals float addition bit for bit")
	void addInPlace_isFloatAddition() {
		int rows = 512;
		int dim = 4096;
		Random rng = new Random(41);
		float[] x = new float[rows * dim];
		float[] y = new float[rows * dim];
		for (int i = 0; i < x.length; i++) {
			x[i] = (float) rng.nextGaussian() * 10f;
			y[i] = (float) rng.nextGaussian();
		}
		float[] actual = runBinary(x, y, (dx, dy) -> kernels.addInPlace(dx, dy, x.length, null));
		for (int i = 0; i < x.length; i++)
			assertThat(Float.floatToRawIntBits(actual[i])).as("element %d", i)
					.isEqualTo(Float.floatToRawIntBits(x[i] + y[i]));
	}

	@Test
	@DisplayName("the bias add broadcasts one row over the window, bit for bit")
	void addBias_broadcastsOverRows() {
		for (int rows : WINDOWS) {
			int dim = 2048 + 256 + 256;
			Random rng = new Random(51 + rows);
			float[] x = new float[rows * dim];
			float[] bias = new float[dim];
			for (int i = 0; i < x.length; i++)
				x[i] = (float) rng.nextGaussian();
			for (int i = 0; i < dim; i++)
				bias[i] = (float) rng.nextGaussian() * 0.1f;
			float[] actual = runBinary(x, bias, (dx, db) -> kernels.addBias(dx, db, rows, dim, null));
			for (int r = 0; r < rows; r++)
				for (int j = 0; j < dim; j++)
					assertThat(Float.floatToRawIntBits(actual[r * dim + j])).as("row %d col %d", r, j)
							.isEqualTo(Float.floatToRawIntBits(x[r * dim + j] + bias[j]));
		}
	}

	@Test
	@DisplayName("split-half RoPE matches the CPU rotate-half rotation within one float rounding")
	void splitHalfRope_matchesCpu() {
		// Qwen2.5-3B: 16 query heads and 2 KV heads of 128, base 1e6; a 512-wide window at 0 and a later chunk.
		rotateSplitHalf(512, 16, 128, 1_000_000f, 0, 61);
		rotateSplitHalf(32, 2, 128, 1_000_000f, 4000, 62);
		rotateSplitHalf(9, 32, 64, 10000f, 30000, 63);
		assertThat(rotateSplitHalf(1, 8, 64, 10000f, 0, 64)).as("position zero is the identity").isZero();
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	private interface Launch {
		void run(MemorySegment a, MemorySegment b);
	}

	/** Uploads {@code x} and {@code y}, runs the kernel, and returns what is left in {@code x}'s buffer. */
	private static float[] runBinary(float[] x, float[] y, Launch launch) {
		GpuBindings gpu = ctx.bindings();
		long xBytes = (long) x.length * Float.BYTES;
		long yBytes = (long) y.length * Float.BYTES;
		MemorySegment dx = gpu.deviceMalloc(ctx.deviceIndex(), xBytes);
		MemorySegment dy = gpu.deviceMalloc(ctx.deviceIndex(), yBytes);
		float[] out = new float[x.length];
		try (Arena a = Arena.ofConfined()) {
			MemorySegment hx = a.allocate(xBytes);
			MemorySegment hy = a.allocate(yBytes);
			MemorySegment.copy(x, 0, hx, JAVA_FLOAT, 0, x.length);
			MemorySegment.copy(y, 0, hy, JAVA_FLOAT, 0, y.length);
			copy(gpu, dx, hx, xBytes, GpuBindings.H2D);
			copy(gpu, dy, hy, yBytes, GpuBindings.H2D);
			launch.run(dx, dy);
			copy(gpu, hx, dx, xBytes, GpuBindings.D2H);
			MemorySegment.copy(hx, JAVA_FLOAT, 0, out, 0, x.length);
		} finally {
			gpu.deviceFree(dx);
			gpu.deviceFree(dy);
		}
		return out;
	}

	private static float rotateSplitHalf(int rows, int nHeads, int headDim, float theta, int startPos, long seed) {
		int width = nHeads * headDim;
		Random rng = new Random(seed);
		float[] x = new float[rows * width];
		for (int i = 0; i < x.length; i++)
			x[i] = rng.nextFloat() * 4f - 2f;
		float[] expected = x.clone();
		float[] row = new float[width];
		for (int r = 0; r < rows; r++) {
			System.arraycopy(expected, r * width, row, 0, width);
			LlamaTransformerHandler.rope(row, startPos + r, nHeads, headDim, theta, RopePairing.SPLIT_HALF);
			System.arraycopy(row, 0, expected, r * width, width);
		}

		double[] invFreq = RopeKernel.inverseFrequencies(headDim, theta);
		GpuBindings gpu = ctx.bindings();
		long tableBytes = (long) invFreq.length * Double.BYTES;
		MemorySegment dTable = gpu.deviceMalloc(ctx.deviceIndex(), tableBytes);
		float[] actual;
		try (Arena a = Arena.ofConfined()) {
			MemorySegment ht = a.allocate(tableBytes, Double.BYTES);
			MemorySegment.copy(invFreq, 0, ht, JAVA_DOUBLE, 0, invFreq.length);
			copy(gpu, dTable, ht, tableBytes, GpuBindings.H2D);
			actual = runBinary(x, new float[1], (dx, unused) -> RopeKernel.tryLoad().launchSplitHalf(dx, dTable, rows,
					nHeads, headDim, startPos, null));
		} finally {
			gpu.deviceFree(dTable);
		}
		float maxDiff = 0f;
		for (int i = 0; i < x.length; i++)
			maxDiff = Math.max(maxDiff, Math.abs(actual[i] - expected[i]));
		System.out.printf("[split-half rope parity] rows=%d heads=%d headDim=%d startPos=%d max|diff|=%.3e%n", rows,
				nHeads, headDim, startPos, maxDiff);
		assertThat(maxDiff).as("largest divergence from the CPU rotate-half rotation").isLessThanOrEqualTo(1e-6f);
		return maxDiff;
	}

	/** FP16 bits mapped to a monotonic integer, so the difference of two is their ulp distance. */
	private static int orderedHalf(short h) {
		int bits = h & 0xffff;
		return (bits & 0x8000) != 0 ? -(bits & 0x7fff) : bits;
	}

	private static void copy(GpuBindings gpu, MemorySegment dst, MemorySegment src, long bytes, int kind) {
		GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dst, src, bytes, kind), "memcpy");
	}
}
