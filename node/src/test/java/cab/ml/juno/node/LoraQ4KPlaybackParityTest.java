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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterConfig;

/**
 * Playback MMQ: packed Q4 GEMV then LoRA {@code += B(Ax)} vs CPU oracle.
 */
@Tag("gpu")
@DisplayName("LoRA Q4_K playback parity")
class LoraQ4KPlaybackParityTest {

	@Test
	@DisplayName("ResidentQ4KWeight GEMV + LoRA delta matches CPU within MMQ tolerance")
	void gpu_q4_then_lora_delta_matches_cpu() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
		Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
		assumeTrue(kernel != null, "Q4K MMQ kernel failed to load");

		int rows = 64;
		int cols = 256;
		int rank = 4;
		Random rnd = new Random(7);
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = (rnd.nextFloat() * 2f) - 1f;
		byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
		GgufReader.QuantizedTensor quant = new GgufReader.QuantizedTensor(
				"w", QuantizationLayout.TYPE_Q4_K, (long) rows * cols, raw);
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;
		float[] a = random(rnd, rank * cols);
		float[] b = random(rnd, rows * rank);
		LoraAdapter lora = LoraAdapter.fromWeights(LoraAdapterConfig.legacy(rank, 8f), cols, rows, a, b);

		float[] expected = new float[rows];
		LlamaTransformerHandler.matVecInto(quant, x, expected, rows, cols);
		float[] delta = lora.forward(x);
		for (int i = 0; i < rows; i++)
			expected[i] += delta[i];

		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			ResidentQ4KWeight w = ResidentQ4KWeight.upload(mv, raw, rows, cols);
			try {
				float[] got = LoraResidentWeights.matVec(quant, w, null, x, rows, cols);
				assertThat(got).hasSize(rows);
				float[] playDelta = lora.forward(x);
				for (int i = 0; i < rows; i++)
					got[i] += playDelta[i];
				for (int i = 0; i < rows; i++)
					assertThat(got[i]).as("row " + i).isCloseTo(expected[i], within(1e-2f));
			} finally {
				w.close();
				assertThat(w.isClosed()).isTrue();
				w.close();
				assertThat(w.isClosed()).isTrue();
				assertThatThrownBy(() -> w.sgemv(x))
						.isInstanceOf(IllegalStateException.class)
						.hasMessageContaining("closed");
			}
		}
	}

	private static float[] random(Random rnd, int n) {
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (rnd.nextFloat() * 2f) - 1f;
		return v;
	}
}
