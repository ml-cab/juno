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

import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraInitialization;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraScaling;
import cab.ml.juno.lora.MergeCapability;

class LoraMetricsIdentityTest {

	@Test
	void cliVocabulary_matchesModes() {
		LoraMetricsIdentity id = LoraMetricsIdentity.of(LoraMode.QA_LORA, LoraScaling.RANK_STABILIZED,
				LoraInitialization.LEGACY_NORMAL, "llama", "cuda", 8, 16f, List.of(LoraProjection.WQ, LoraProjection.WV),
				32, MergeCapability.SOURCE_TYPE_PROJECTED);
		assertThat(id.algorithm).isEqualTo("qa-lora");
		assertThat(id.scaling).isEqualTo("rslora");
		assertThat(id.initialization).isEqualTo("legacy-normal");
		assertThat(id.targets).isEqualTo("wq,wv");
		assertThat(id.groupWidth).isEqualTo(32);
		assertThat(id.mergeCapability).isEqualTo("source-type-projected");
		assertThat(id.effectiveScale).isEqualTo(16f / (float) Math.sqrt(8));
	}

	@Test
	void doraAndLora_labels() {
		assertThat(LoraMetricsIdentity.algorithmLabel(LoraMode.DORA)).isEqualTo("dora");
		assertThat(LoraMetricsIdentity.algorithmLabel(LoraMode.LORA)).isEqualTo("lora");
		assertThat(LoraMetricsIdentity.scalingLabel(LoraScaling.STANDARD)).isEqualTo("standard");
	}

	@Test
	void apply_copiesOntoTrainEvent() {
		LoraMetricsIdentity id = LoraMetricsIdentity.of(LoraMode.DORA, LoraScaling.STANDARD,
				LoraInitialization.KAIMING_UNIFORM, "qwen2", "cpu", 4, 4f, List.of(LoraProjection.WV), 0,
				MergeCapability.F32_PRESERVE);
		LoraTrainEvent ev = new LoraTrainEvent();
		id.apply(ev);
		assertThat(ev.algorithm).isEqualTo("dora");
		assertThat(ev.architecture).isEqualTo("qwen2");
		assertThat(ev.rank).isEqualTo(4);
		assertThat(ev.targets).isEqualTo("wv");
	}
}
