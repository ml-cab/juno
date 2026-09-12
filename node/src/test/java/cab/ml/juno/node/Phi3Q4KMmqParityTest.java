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

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Parity: fused Phi-3-style QKV Q4_K device GEMV + host slice vs CPU row-range matVec.
 */
@Tag("gpu")
@DisplayName("Phi-3 fused Q4_K MMQ parity")
class Phi3Q4KMmqParityTest {

	@Test
	@DisplayName("fused QKV: one Q4 GEMV + slice matches CPU row-range")
	void fused_qkv_gemv_slice_matches_cpu() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
		assumeTrue(kernel != null, "Q4K MMQ kernel failed to load");

		int H = 256;
		int kvDim = 128;
		int rows = H + 2 * kvDim;
		int cols = H;
		Random rnd = new Random(7);
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
		var qt = new GgufReader.QuantizedTensor("attn_qkv", QuantizationLayout.TYPE_Q4_K,
				(long) rows * cols, raw);

		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;

		float[] qCpu = LlamaTransformerHandler.matVec(qt, x, 0, H, cols);
		float[] kCpu = LlamaTransformerHandler.matVec(qt, x, H, H + kvDim, cols);
		float[] vCpu = LlamaTransformerHandler.matVec(qt, x, H + kvDim, rows, cols);

		try (GpuContext ctx = GpuContext.init(0);
				DeviceQ4KMatrix dA = DeviceQ4KMatrix.upload(ctx, raw, rows, cols)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			float[] qkv = mv.sgemv(dA, x);
			assertThat(qkv).hasSize(rows);
			for (int i = 0; i < H; i++)
				assertThat(qkv[i]).as("q[" + i + "]").isCloseTo(qCpu[i], within(Q4KMmqParityTest.q8Tol(qCpu[i])));
			for (int i = 0; i < kvDim; i++)
				assertThat(qkv[H + i]).as("k[" + i + "]").isCloseTo(kCpu[i], within(Q4KMmqParityTest.q8Tol(kCpu[i])));
			for (int i = 0; i < kvDim; i++)
				assertThat(qkv[H + kvDim + i]).as("v[" + i + "]")
						.isCloseTo(vCpu[i], within(Q4KMmqParityTest.q8Tol(vCpu[i])));
		}
	}
}
