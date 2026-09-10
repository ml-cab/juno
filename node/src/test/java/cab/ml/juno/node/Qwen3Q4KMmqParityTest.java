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
 * Parity: Qwen3-style separate-tensor Q4_K device GEMV vs CPU matVec
 * (qDim may differ from hidden; cols = hidden).
 */
@Tag("gpu")
@DisplayName("Qwen3 Q4_K MMQ parity")
class Qwen3Q4KMmqParityTest {

	@Test
	@DisplayName("attn_q-shaped Q4 GEMV matches CPU")
	void attn_q_gemv_matches_cpu() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
		assumeTrue(kernel != null, "Q4K MMQ kernel failed to load");

		int qDim = 512;
		int H = 256;
		Random rnd = new Random(11);
		float[] host = new float[qDim * H];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
		var qt = new GgufReader.QuantizedTensor("attn_q", QuantizationLayout.TYPE_Q4_K,
				(long) qDim * H, raw);

		float[] x = new float[H];
		for (int i = 0; i < H; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;

		float[] expected = LlamaTransformerHandler.matVec(qt, x, qDim, H);

		try (GpuContext ctx = GpuContext.init(0);
				DeviceQ4KMatrix dA = DeviceQ4KMatrix.upload(ctx, raw, qDim, H)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			float[] got = mv.sgemv(dA, x);
			assertThat(got).hasSize(qDim);
			for (int i = 0; i < qDim; i++)
				assertThat(got[i]).as("row " + i).isCloseTo(expected[i], within(1e-2f));
		}
	}
}
