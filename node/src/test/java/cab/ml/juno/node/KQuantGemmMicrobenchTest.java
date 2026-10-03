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

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Informal device time per prefill matmul on Mistral 7B's Q4_K shapes: the tiled
 * packed GEMM (Q8_1 quantize + {@link KQuantGemmKernel}) against the dequant route
 * (dequantize to FP16 + FP16 GEMM), both from the same FP16-or-FP32 device input
 * the prefill-window region holds. Not a pass/fail gate: prints ms per matmul for
 * kernel triage. The tier's gate is the pinned end-to-end A/B.
 */
@Tag("gpu")
@DisplayName("Tiled K-quant GEMM microbench")
class KQuantGemmMicrobenchTest {

	/** {rows, cols}: attention Q/O, FFN gate/up, FFN down. */
	private static final int[][] SHAPES = { { 4096, 4096 }, { 14336, 4096 }, { 4096, 14336 } };
	private static final int[] WIDTHS = { 16, 32, 64, 128, 512 };
	private static final int ITERS = 10;
	private static final int Q4K_BLOCK_BYTES = 144;

	@Test
	@DisplayName("print packed vs dequant ms per matmul")
	void printMsPerMatmul() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA - skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			KQuantGemmKernel kernel = KQuantGemmKernel.tryLoad();
			PrefillWindowKernels window = PrefillWindowKernels.tryLoad();
			assumeTrue(mv.supportsQ4KMmq() && kernel != null && window != null, "kernels unavailable");
			System.out.println("rows cols W | packed ms | dequant+fp16 ms | speedup");
			for (int[] s : SHAPES) {
				int rows = s[0], cols = s[1];
				try (DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(rows, cols, rows + cols), rows, cols);
						ResidentChain chain = ResidentChain.open(ctx)) {
					int maxW = WIDTHS[WIDTHS.length - 1];
					ResidentActivation in = chain.allocate(maxW, cols);
					ResidentActivation out = chain.allocate(maxW, rows);
					MemorySegment dQ8 = chain.allocateScratch(KQuantGemmKernel.q8Bytes(maxW, cols));
					MemorySegment dXh = chain.allocateScratch((long) maxW * cols * Short.BYTES);
					in.upload(randomRows(maxW, cols));
					for (int w : WIDTHS) {
						synchronized (ctx.cublasSerializationLock()) {
							double packed = time(chain, () -> kernel.multiply(a, in.devicePointer(), dQ8,
									out.devicePointer(), w, rows, chain.stream()));
							double dequant = time(chain, () -> {
								window.toHalf(in.devicePointer(), dXh, (long) w * cols, chain.stream());
								MemorySegment dW = mv.dequantOnStream(a, w, chain.stream(), chain.spans());
								mv.gemmHalfOnStream(dW, rows, cols, dXh, out.devicePointer(), rows, w,
										chain.stream(), chain.spans());
							});
							System.out.printf("%d %d %d | %.3f | %.3f | %.2fx%n", rows, cols, w, packed, dequant,
									dequant / packed);
						}
					}
				}
			}
			mv.releaseScratch();
		}
	}

	private static double time(ResidentChain chain, Runnable body) {
		for (int i = 0; i < 3; i++)
			body.run();
		chain.sync();
		long t0 = System.nanoTime();
		for (int i = 0; i < ITERS; i++)
			body.run();
		chain.sync();
		return (System.nanoTime() - t0) / 1e6 / ITERS;
	}

	private static float[][] randomRows(int rows, int cols) {
		Random r = new Random(5);
		float[][] x = new float[rows][cols];
		for (float[] row : x)
			for (int i = 0; i < cols; i++)
				row[i] = r.nextFloat() * 2f - 1f;
		return x;
	}

	/** Random Q4_K super-blocks with small finite FP16 scales; the timing does not depend on the values. */
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
