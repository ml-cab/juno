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
package cab.ml.juno.lora;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("QA-LoRA checkpoint round-trip")
class QaLoraCheckpointTest {

	@Test
	@DisplayName("v2 save/load preserves grouped A and metadata")
	void round_trip(@TempDir Path dir) throws Exception {
		LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 4f, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM,
				LoraMode.QA_LORA);
		QaLoraAdapter a = new QaLoraAdapter(cfg, 8, 4, 4, new Random(1));
		for (int i = 0; i < a.b().length; i++)
			a.b()[i] = 0.01f * i;

		LoraAdapterSet set = new LoraAdapterSet();
		set.addQa(0, "wq", a, QaLoraEntryMeta.of(4, 2, 12, "juno-kquant-v1", MergeCapability.F32_PRESERVE));

		Path path = dir.resolve("qa.lora");
		set.save(path);

		LoraAdapterSet loaded = LoraAdapterSet.load(path);
		assertThat(loaded.all()).isEmpty();
		assertThat(loaded.allQa()).hasSize(1);
		QaLoraAdapter b = loaded.getQa(0, "wq");
		assertThat(b.groupWidth).isEqualTo(4);
		assertThat(b.a()).containsExactly(a.a());
		assertThat(b.b()).containsExactly(a.b());
		QaLoraEntryMeta meta = loaded.getQaMeta(0, "wq");
		assertThat(meta.ggmlType()).isEqualTo(12);
		assertThat(meta.encoderId()).isEqualTo("juno-kquant-v1");
		assertThat(meta.mergeCapability()).isEqualTo(MergeCapability.F32_PRESERVE);
	}

	@Test
	@DisplayName("v1 export rejects QA-LoRA")
	void reject_v1(@TempDir Path dir) {
		LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL,
				LoraMode.QA_LORA);
		LoraAdapterSet set = new LoraAdapterSet();
		set.addQa(0, "wv", new QaLoraAdapter(cfg, 8, 4, 2, new Random(2)),
				QaLoraEntryMeta.of(2, 4, 14, "juno-kquant-v1", MergeCapability.SOURCE_TYPE_PROJECTED));
		assertThatThrownBy(() -> set.saveLegacyV1(dir.resolve("bad.lora")))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("QA-LoRA");
	}
}
