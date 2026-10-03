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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Numerical quality of the prefill matmul routes for packed K-quant weights,
 * against the same CPU FP32 oracle on the same matrices and activations, and the
 * threshold the packed route is held to: mean relative error no more than the
 * fused decode GEMV's (1.001x, for float accumulation order) and no more than
 * 0.5%. The FP16 dequant route is reported beside them; it is about 14x more
 * accurate, because it rounds activations to an 11-bit mantissa rather than 8 bits.
 *
 * <ul>
 * <li>packed: activations quantized to Q8_1, integer dot over the packed weights
 * ({@link KQuantGemmKernel});
 * <li>dequant: weights expanded to FP16, activations cast to FP16, FP16 GEMM with
 * FP32 compute ({@link CudaMatVec#dequantOnStream} then
 * {@link CudaMatVec#gemmHalfOnStream}).
 * </ul>
 *
 * Error is relative to the oracle's magnitude: mean = sum |err| / sum |ref|, max =
 * max |err| / max |ref|, over every output of a shape. Prints one line per shape
 * for the published side-by-side table.
 */
@Tag("gpu")
@DisplayName("Tiled K-quant GEMM numerical quality against the FP16 dequant route")
class KQuantGemmQualityTest {

	private static final int[] TYPES = { QuantizationLayout.TYPE_Q4_K, QuantizationLayout.TYPE_Q5_K,
			QuantizationLayout.TYPE_Q6_K };
	/** {rows, cols, batch}. */
	private static final int[][] CASES = { { 512, 512, 16 }, { 1376, 512, 64 }, { 512, 1536, 128 },
			{ 512, 4096, 512 } };

	private static GpuContext ctx;
	private static CudaMatVec mv;
	private static KQuantGemmKernel kernel;
	private static PrefillWindowKernels window;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA - skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "K-quant GEMV kernel unavailable");
		window = PrefillWindowKernels.tryLoad();
		assumeTrue(window != null, "prefill-window kernels failed to load");
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

	@Test
	@DisplayName("packed error is the fused GEMV's and under 0.5%; all three routes reported")
	void reportBothRoutes() {
		System.out.println("type rows cols batch | packed mean max | gemv mean max | dequant-fp16 mean max"
				+ " | packed/gemv | packed/fp16");
		for (int typeId : TYPES) {
			for (int[] c : CASES) {
				int rows = c[0], cols = c[1], batch = c[2];
				Random rnd = new Random(1000L * typeId + rows + cols + batch);
				float[] host = new float[rows * cols];
				for (int i = 0; i < host.length; i++)
					host[i] = (rnd.nextFloat() * 2f) - 1f;
				byte[] raw = GgufKQuantCodec.encode(host, typeId);
				float[] w = GgufKQuantCodec.decodeRows(raw, typeId, rows, cols);
				float[][] x = new float[batch][cols];
				for (float[] row : x)
					for (int i = 0; i < cols; i++)
						row[i] = (rnd.nextFloat() * 2f) - 1f;
				float[][] ref = new float[batch][];
				for (int b = 0; b < batch; b++)
					ref[b] = dot(w, x[b], rows, cols);
				try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId)) {
					double[] packed = error(run(a, x, true), ref);
					double[] dequant = error(run(a, x, false), ref);
					float[][] gemvOut = new float[batch][];
					for (int b = 0; b < batch; b++)
						gemvOut[b] = mv.sgemv(a, x[b]);
					double[] gemv = error(gemvOut, ref);
					System.out.printf("%d %d %d %d | %.3e %.3e | %.3e %.3e | %.3e %.3e | %.4f | %.2f%n", typeId,
							rows, cols, batch, packed[0], packed[1], gemv[0], gemv[1], dequant[0], dequant[1],
							packed[0] / gemv[0], packed[0] / dequant[0]);
					String where = "type " + typeId + " " + rows + "x" + cols + " W=" + batch;
					assertThat(packed[0]).as(where + ": packed mean error <= 1.001x the fused GEMV's")
							.isLessThanOrEqualTo(gemv[0] * 1.001);
					assertThat(packed[0]).as(where + ": packed mean error <= 0.5%").isLessThanOrEqualTo(0.005);
				}
			}
		}
	}

	private static float[][] run(DeviceQ4KMatrix a, float[][] x, boolean packed) {
		int batch = x.length, rows = a.rows(), cols = a.cols();
		try (ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation in = chain.allocate(batch, cols);
			ResidentActivation out = chain.allocate(batch, rows);
			in.upload(x);
			synchronized (ctx.cublasSerializationLock()) {
				if (packed) {
					MemorySegment dQ8 = chain.allocateScratch(KQuantGemmKernel.q8Bytes(batch, cols));
					kernel.multiply(a, in.devicePointer(), dQ8, out.devicePointer(), batch, rows, chain.stream());
				} else {
					MemorySegment dXh = chain.allocateScratch((long) batch * cols * Short.BYTES);
					window.toHalf(in.devicePointer(), dXh, (long) batch * cols, chain.stream());
					MemorySegment dW = mv.dequantOnStream(a, batch, chain.stream(), chain.spans());
					mv.gemmHalfOnStream(dW, rows, cols, dXh, out.devicePointer(), rows, batch, chain.stream(),
							chain.spans());
				}
				chain.sync();
			}
			out.markWritten(batch);
			float[][] result = new float[batch][];
			out.materialize(result);
			return result;
		}
	}

	private static double[] error(float[][] got, float[][] ref) {
		double sumErr = 0, sumRef = 0, maxErr = 0, maxRef = 0;
		for (int b = 0; b < ref.length; b++)
			for (int r = 0; r < ref[b].length; r++) {
				double e = Math.abs(got[b][r] - ref[b][r]);
				double m = Math.abs(ref[b][r]);
				sumErr += e;
				sumRef += m;
				maxErr = Math.max(maxErr, e);
				maxRef = Math.max(maxRef, m);
			}
		return new double[] { sumErr / sumRef, maxErr / maxRef };
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
}
