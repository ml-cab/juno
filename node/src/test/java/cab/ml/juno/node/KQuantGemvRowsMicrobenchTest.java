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
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Times the decode-batch matmul against what it replaces (one GEMV per row) and the
 * tiled GEMM that a batch wider than the GEMV path's threshold takes, on stream, with
 * no copies, so only the kernels are compared. Prints; holds nothing to a number.
 */
@Tag("gpu")
@DisplayName("Decode-batch K-quant matmul microbench")
class KQuantGemvRowsMicrobenchTest {

	/** {rows, cols}: attention and FFN projections of the 256-wide and the 2048-wide sweep models. */
	private static final int[][] SHAPES = { { 256, 256 }, { 1024, 256 }, { 256, 1024 }, { 2048, 2048 }, { 8192, 2048 },
			{ 2048, 8192 } };
	private static final int[] WIDTHS = { 1, 2, 4, 8 };
	private static final int ITERS = 15;

	@Test
	@DisplayName("print ms per matmul: rows kernel, one GEMV per row, tiled GEMM")
	void printMsPerMatmul() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA - skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			Q4KMmqKernel mmq = Q4KMmqKernel.tryLoad();
			BatchedDecodeKernels rowsKernel = BatchedDecodeKernels.tryLoad();
			KQuantGemmKernel tiled = KQuantGemmKernel.tryLoad();
			assumeTrue(mmq != null && rowsKernel != null && tiled != null, "kernels unavailable");
			System.out.println("rows cols N | rows-kernel ms | N x GEMV ms | tiled GEMM ms");
			for (int[] s : SHAPES) {
				int rows = s[0], cols = s[1];
				try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, randomQ4K(rows, cols, rows + cols), rows, cols,
						QuantizationLayout.TYPE_Q4_K); ResidentChain chain = ResidentChain.open(ctx)) {
					int maxW = WIDTHS[WIDTHS.length - 1];
					ResidentActivation in = chain.allocate(maxW, cols);
					ResidentActivation out = chain.allocate(maxW, rows);
					MemorySegment dQ8 = chain.allocateScratch(maxW * Q4KMmqKernel.q8Bytes(cols));
					MemorySegment dQ8t = chain.allocateScratch(KQuantGemmKernel.q8Bytes(maxW, cols));
					in.upload(randomRows(maxW, cols));
					synchronized (ctx.cublasSerializationLock()) {
						mmq.quantizeX(in.devicePointer(), dQ8, maxW * cols, chain.stream());
						chain.sync();
					}
					long q8Row = Q4KMmqKernel.q8Bytes(cols);
					for (int w : WIDTHS) {
						synchronized (ctx.cublasSerializationLock()) {
							double kernel = time(chain, () -> rowsKernel.gemvRows(a, dQ8, out.devicePointer(), w, rows,
									chain.stream()));
							double serial = time(chain, () -> {
								for (int b = 0; b < w; b++)
									mmq.launchPacked(a.devicePointer(), dQ8.asSlice(b * q8Row),
											out.devicePointer().asSlice((long) b * rows * Float.BYTES), rows, cols,
											a.quantType(), chain.stream());
							});
							double gemm = time(chain,
									() -> tiled.multiply(a, in.devicePointer(), dQ8t, out.devicePointer(), w, rows,
											chain.stream()));
							System.out.printf("%d %d %d | %.3f | %.3f | %.3f%n", rows, cols, w, kernel, serial, gemm);
						}
					}
				}
			}
			mv.releaseScratch();
		}
	}

	private static double time(ResidentChain chain, Runnable body) {
		double[] ms = new double[ITERS];
		for (int i = 0; i < ITERS; i++) {
			long t0 = System.nanoTime();
			body.run();
			chain.sync();
			ms[i] = (System.nanoTime() - t0) / 1e6;
		}
		Arrays.sort(ms);
		return ms[ITERS / 2];
	}

	private static byte[] randomQ4K(int rows, int cols, long seed) {
		Random rnd = new Random(seed);
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = -1f + 2f * rnd.nextFloat();
		return GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
	}

	private static float[][] randomRows(int n, int cols) {
		Random rnd = new Random(n * 31L + cols);
		float[][] x = new float[n][cols];
		for (float[] row : x)
			for (int i = 0; i < cols; i++)
				row[i] = -2f + 4f * rnd.nextFloat();
		return x;
	}
}
