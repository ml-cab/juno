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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The FFN gate and up projections in one launch ({@code q4k_gemv_pair}) against two {@code q4k_gemv} launches over
 * the same Q8_1 input: equal bit for bit, on both of the kernel's layouts (two row indices a block up to 8 super-blocks
 * a row, one above), at odd and even row counts.
 */
@Tag("gpu")
@DisplayName("Paired Q4_K gate and up GEMV")
class Q4KGemvPairTest {

	private static final int H2D = 1;
	private static final int D2H = 2;
	private static final float SENTINEL = -12345.5f;

	/** {rows, cols}: 1 to 8 super-blocks take the two-row layout, 9 and up the one-row layout; 17 needs two passes. */
	private static final int[][] SHAPES = { { 1, 256 }, { 2, 256 }, { 3, 768 }, { 6, 1280 }, { 257, 2048 },
			{ 1000, 2048 }, { 64, 2304 }, { 51, 3072 }, { 33, 4096 }, { 17, 4352 } };

	@Test
	@DisplayName("gate and up equal two single-matrix launches bit for bit, and no row past the last is written")
	void pairEqualsTwoLaunches() {
		assumeTrue(CudaAvailability.isAvailable() && Q4KMmqKernel.isAvailable(), "no CUDA device");
		Q4KMmqKernel single = Q4KMmqKernel.tryLoad();
		Q4KGemvPairKernel pair = Q4KGemvPairKernel.tryLoad();
		assumeTrue(single != null, "single GEMV kernel unavailable");
		assertThat(pair).as("paired kernel loads").isNotNull();
		CudaBindings cuda = CudaBindings.instance();
		Random rng = new Random(20261117);
		try (GpuContext ctx = GpuContext.init(0); Arena arena = Arena.ofConfined()) {
			for (int[] shape : SHAPES) {
				int rows = shape[0], cols = shape[1];
				byte[] gateRaw = GgufKQuantCodec.encode(floats(rng, rows * cols), QuantizationLayout.TYPE_Q4_K);
				byte[] upRaw = GgufKQuantCodec.encode(floats(rng, rows * cols), QuantizationLayout.TYPE_Q4_K);
				float[] x = floats(rng, cols);
				try (DeviceQ4KMatrix gate = DeviceQ4KMatrix.upload(ctx, gateRaw, rows, cols);
						DeviceQ4KMatrix up = DeviceQ4KMatrix.upload(ctx, upRaw, rows, cols)) {
					assertThat(Q4KGemvPairKernel.pairs(gate, up)).isTrue();
					MemorySegment dX = upload(cuda, arena, x);
					MemorySegment dQ8 = cuda.deviceMalloc(0, Q4KMmqKernel.q8Bytes(cols));
					MemorySegment[] y = new MemorySegment[4];
					for (int i = 0; i < y.length; i++)
						y[i] = upload(cuda, arena, sentinels(rows + 1));
					try {
						single.quantizeX(dX, dQ8, cols, null);
						single.launchPacked(gate.devicePointer(), dQ8, y[0], rows, cols, QuantizationLayout.TYPE_Q4_K, null);
						single.launchPacked(up.devicePointer(), dQ8, y[1], rows, cols, QuantizationLayout.TYPE_Q4_K, null);
						pair.launch(gate.devicePointer(), up.devicePointer(), dQ8, y[2], y[3], rows, cols, null);
						float[] gateOne = download(cuda, arena, y[0], rows + 1);
						float[] upOne = download(cuda, arena, y[1], rows + 1);
						float[] gatePair = download(cuda, arena, y[2], rows + 1);
						float[] upPair = download(cuda, arena, y[3], rows + 1);
						String at = rows + " x " + cols;
						for (int r = 0; r < rows; r++) {
							assertThat(Float.floatToRawIntBits(gatePair[r])).as("gate row %d of %s", r, at)
									.isEqualTo(Float.floatToRawIntBits(gateOne[r]));
							assertThat(Float.floatToRawIntBits(upPair[r])).as("up row %d of %s", r, at)
									.isEqualTo(Float.floatToRawIntBits(upOne[r]));
						}
						assertThat(gateOne[0]).as("the reference wrote row 0 of %s", at).isNotEqualTo(SENTINEL);
						assertThat(gatePair[rows]).as("gate past the last row of %s", at).isEqualTo(SENTINEL);
						assertThat(upPair[rows]).as("up past the last row of %s", at).isEqualTo(SENTINEL);
					} finally {
						cuda.deviceFree(dX);
						cuda.deviceFree(dQ8);
						for (MemorySegment s : y)
							cuda.deviceFree(s);
					}
				}
			}
		}
	}

	private static float[] floats(Random rng, int n) {
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = rng.nextFloat() * 2f - 1f;
		return v;
	}

	private static float[] sentinels(int n) {
		float[] v = new float[n];
		java.util.Arrays.fill(v, SENTINEL);
		return v;
	}

	private static MemorySegment upload(CudaBindings cuda, Arena arena, float[] values) {
		long bytes = (long) values.length * Float.BYTES;
		MemorySegment d = cuda.deviceMalloc(0, bytes);
		MemorySegment host = arena.allocate(bytes);
		MemorySegment.copy(values, 0, host, JAVA_FLOAT, 0, values.length);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, bytes, H2D), "cudaMemcpy");
		return d;
	}

	private static float[] download(CudaBindings cuda, Arena arena, MemorySegment d, int n) {
		long bytes = (long) n * Float.BYTES;
		MemorySegment host = arena.allocate(bytes);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, host, d, bytes, D2H), "cudaMemcpy");
		float[] v = new float[n];
		MemorySegment.copy(host, JAVA_FLOAT, 0, v, 0, n);
		return v;
	}
}
