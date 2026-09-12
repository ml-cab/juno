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

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Informal GEMV wall time: packed Q4_K + Q8_1 vs FP16-resident cuBLAS.
 * Not a pass/fail gate — prints ms/call for bake-off triage.
 */
@Tag("gpu")
@DisplayName("Q4K MMQ microbench")
class Q4KMmqMicrobenchTest {

	private static final int[][] SHAPES = {
			{ 3072, 3072 },
			{ 8192, 3072 },
			{ 3072, 8192 },
			{ 32064, 3072 },
	};

	@Test
	@DisplayName("print Q4_K vs FP16 GEMV ms/call")
	void print_ms_per_call() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "MMQ kernel failed to load");
		Random rnd = new Random(1);
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			for (int[] shape : SHAPES) {
				int rows = shape[0], cols = shape[1];
				float[] host = new float[rows * cols];
				for (int i = 0; i < host.length; i++)
					host[i] = (rnd.nextFloat() * 2f) - 1f;
				byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
				float[] x = new float[cols];
				for (int i = 0; i < cols; i++)
					x[i] = (rnd.nextFloat() * 2f) - 1f;
				try (DeviceQ4KMatrix q4 = DeviceQ4KMatrix.upload(ctx, raw, rows, cols);
						DeviceHalfMatrix half = DeviceHalfMatrix.uploadFromFloat32(ctx, host, rows, cols)) {
					double q4ms = timeMs(mv, q4, x);
					double fp16ms = timeMs(mv, half, x);
					System.out.printf("GEMV %dx%d  q4=%.3f ms  fp16=%.3f ms  q4/fp16=%.2f%n",
							rows, cols, q4ms, fp16ms, q4ms / fp16ms);
				}
			}
		}
	}

	private static double timeMs(CudaMatVec mv, DeviceQ4KMatrix A, float[] x) {
		for (int i = 0; i < 5; i++)
			mv.sgemv(A, x);
		int n = 20;
		long t0 = System.nanoTime();
		for (int i = 0; i < n; i++)
			mv.sgemv(A, x);
		return (System.nanoTime() - t0) / 1e6 / n;
	}

	private static double timeMs(CudaMatVec mv, DeviceHalfMatrix A, float[] x) {
		for (int i = 0; i < 5; i++)
			mv.sgemv(A, x);
		int n = 20;
		long t0 = System.nanoTime();
		for (int i = 0; i < n; i++)
			mv.sgemv(A, x);
		return (System.nanoTime() - t0) / 1e6 / n;
	}
}
