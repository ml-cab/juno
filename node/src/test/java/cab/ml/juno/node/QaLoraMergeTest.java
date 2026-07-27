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

import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraInitialization;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraScaling;
import cab.ml.juno.lora.MergeCapability;
import cab.ml.juno.lora.QaLoraAdapter;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

@DisplayName("QA-LoRA merge formulas")
class QaLoraMergeTest {

	@Test
	@DisplayName("applyQaDelta matches dense expansion add")
	void apply_qa_delta() {
		LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 4f, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM,
				LoraMode.QA_LORA);
		QaLoraAdapter qa = new QaLoraAdapter(cfg, 8, 4, 4, new Random(3));
		Arrays.fill(qa.b(), 0.05f);
		float[] w = new float[4 * 8];
		float[] w2 = w.clone();
		LoraMerge.applyQaDelta(w, qa, 4, 8);
		float[] dense = qa.expandDenseDelta();
		for (int i = 0; i < w2.length; i++)
			w2[i] += dense[i];
		assertThat(w).usingComparatorWithPrecision(1e-6f).containsExactly(w2);
	}

	@Test
	@DisplayName("EXACT_AFFINE rejected in QaLoraEntryMeta")
	void exact_affine_rejected() {
		assertThatThrownBy(() -> new cab.ml.juno.lora.QaLoraEntryMeta(32, 4, 12, "juno-kquant-v1",
				MergeCapability.EXACT_AFFINE, QaLoraAdapter.PoolingOp.SUM))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("EXACT_AFFINE");
	}

	@Test
	@DisplayName("group width auto: Q4_K=32 Q6_K=16")
	void group_width_auto() {
		assertThat(QaLoraInitializer.resolveGroupWidth(12, 0)).isEqualTo(32);
		assertThat(QaLoraInitializer.resolveGroupWidth(13, 0)).isEqualTo(32);
		assertThat(QaLoraInitializer.resolveGroupWidth(14, 0)).isEqualTo(16);
		assertThat(QaLoraInitializer.resolveGroupWidth(12, 64)).isEqualTo(64);
	}

	@Test
	@DisplayName("projected encode reports requantization metrics helpers")
	void projected_metrics() {
		float[] data = new float[256];
		Arrays.fill(data, 0.25f);
		byte[] enc = GgufQuantCodec.encode(data, QuantizationLayout.TYPE_Q4_K);
		float[] dec = GgufQuantCodec.decode(enc, QuantizationLayout.TYPE_Q4_K);
		var m = QuantizedMergeMetrics.ofReconstruction(data, dec);
		assertThat(m.rmse()).isGreaterThanOrEqualTo(0);
		assertThat(QuantizedMergeMetrics.deltaRetention(new float[] { 1f }, new float[] { 1f }))
				.isCloseTo(1.0, within(1e-12));
	}
}
