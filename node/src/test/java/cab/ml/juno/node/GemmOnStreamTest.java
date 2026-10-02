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

import java.lang.foreign.MemorySegment;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link CudaMatVec#gemmOnStream}: the prefill-window region's matmul, which reads
 * an FP16 input already on the device and leaves its FP32 output there. Fed the
 * same bits, it runs the same kernels as {@link CudaMatVec#sgemm}'s batched path
 * (device dequantization for K-quant weights, then the tiled FP16 GEMM), so its
 * output must be bit-identical to that path's; and an output row stride wider
 * than the matrix must place the result in its column range and touch nothing
 * else, which is how the region writes the gate and up projections side by side.
 */
@Tag("gpu")
@DisplayName("CudaMatVec.gemmOnStream - device operands, bit-identical to sgemm")
class GemmOnStreamTest {

	private static final int ROWS = 256;
	private static final int COLS = 512;
	private static final int Q4K_BLOCK_BYTES = 144;

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "K-quant MMQ kernel unavailable");
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@ParameterizedTest(name = "W={0}")
	@ValueSource(ints = { 9, 32, 512 })
	@DisplayName("an FP16-weight GEMM on device operands equals sgemm bit for bit")
	void halfWeights_matchSgemm(int batch) {
		try (DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 1), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 2);
			float[][] expected = mv.sgemm(a, x);
			float[][] actual = runOnStream(x, ROWS,
					(dXh, dY, stream, timer) -> mv.gemmOnStream(a, dXh, dY, ROWS, batch, stream, timer));
			for (int b = 0; b < batch; b++)
				assertThat(actual[b]).as("row " + b).containsExactly(expected[b]);
		}
	}

	@ParameterizedTest(name = "W={0}")
	@ValueSource(ints = { 9, 32, 512 })
	@DisplayName("a Q4_K GEMM on device operands equals sgemm bit for bit")
	void q4kWeights_matchSgemm(int batch) {
		try (DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 3), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 4);
			float[][] expected = mv.sgemm(a, x);
			float[][] actual = runOnStream(x, ROWS,
					(dXh, dY, stream, timer) -> mv.gemmOnStream(a, dXh, dY, ROWS, batch, stream, timer));
			for (int b = 0; b < batch; b++)
				assertThat(actual[b]).as("row " + b).containsExactly(expected[b]);
		}
	}

	@Test
	@DisplayName("two GEMMs with a doubled row stride land side by side, each equal to its own sgemm")
	void sideBySide_gateAndUp() {
		int batch = 33;
		try (DeviceQ4KMatrix gate = mv.uploadQ4K(randomQ4K(ROWS, COLS, 5), ROWS, COLS);
				DeviceHalfMatrix up = mv.uploadHalf(randomFloats(ROWS * COLS, 6), ROWS, COLS)) {
			float[][] x = randomRows(batch, COLS, 7);
			float[][] g = mv.sgemm(gate, x);
			float[][] u = mv.sgemm(up, x);
			float[][] both = runOnStream(x, 2 * ROWS, (dXh, dY, stream, timer) -> {
				mv.gemmOnStream(gate, dXh, dY, 2 * ROWS, batch, stream, timer);
				mv.gemmOnStream(up, dXh, dY.asSlice((long) ROWS * Float.BYTES), 2 * ROWS, batch, stream, timer);
			});
			for (int b = 0; b < batch; b++) {
				float[] gateHalf = java.util.Arrays.copyOfRange(both[b], 0, ROWS);
				float[] upHalf = java.util.Arrays.copyOfRange(both[b], ROWS, 2 * ROWS);
				assertThat(gateHalf).as("gate, row " + b).containsExactly(g[b]);
				assertThat(upHalf).as("up, row " + b).containsExactly(u[b]);
			}
		}
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	private interface Body {
		void run(MemorySegment dXh, MemorySegment dY, MemorySegment stream, DeviceSpanTimer timer);
	}

	/**
	 * Uploads {@code x} into a resident activation, casts it to FP16 on the device,
	 * runs {@code body} under the serialization lock on the chain's stream with an
	 * output of {@code outWidth} floats per row, and returns that output.
	 */
	private static float[][] runOnStream(float[][] x, int outWidth, Body body) {
		int batch = x.length;
		PrefillWindowKernels kernels = PrefillWindowKernels.tryLoad();
		try (ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation in = chain.allocate(batch, COLS);
			ResidentActivation out = chain.allocate(batch, outWidth);
			MemorySegment dXh = chain.allocateScratch((long) batch * COLS * Short.BYTES);
			in.upload(x);
			synchronized (ctx.cublasSerializationLock()) {
				kernels.toHalf(in.devicePointer(), dXh, (long) batch * COLS, chain.stream());
				body.run(dXh, out.devicePointer(), chain.stream(), chain.spans());
				chain.sync();
			}
			out.markWritten(batch);
			float[][] result = new float[batch][];
			out.materialize(result);
			return result;
		}
	}

	private static float[] randomFloats(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = r.nextFloat() * 2f - 1f;
		return v;
	}

	private static float[][] randomRows(int rows, int cols, long seed) {
		float[][] x = new float[rows][];
		for (int i = 0; i < rows; i++)
			x[i] = randomFloats(cols, seed * 1000 + i);
		return x;
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
