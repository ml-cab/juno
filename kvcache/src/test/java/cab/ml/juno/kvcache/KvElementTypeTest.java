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
package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link KvElementType}: {@code f16} names IEEE half storage (two bytes a value) and {@code f32} the float32 storage,
 * whose spellings keep their meaning; serialized blocks name a type by ordinal, so the existing ordinals stay put and
 * {@code F32} is appended.
 */
@DisplayName("KvElementType - names, sizes and ordinals")
class KvElementTypeTest {

	@Test
	@DisplayName("f16 and fp16 are half storage; f32, fp32 and float are float32 storage")
	void parses() {
		assertThat(KvElementType.parse("f16")).isEqualTo(KvElementType.F16);
		assertThat(KvElementType.parse("FP16")).isEqualTo(KvElementType.F16);
		assertThat(KvElementType.parse(null)).isEqualTo(KvElementType.F16);
		assertThat(KvElementType.parse(" ")).isEqualTo(KvElementType.F16);
		assertThat(KvElementType.parse("f32")).isEqualTo(KvElementType.F32);
		assertThat(KvElementType.parse("fp32")).isEqualTo(KvElementType.F32);
		assertThat(KvElementType.parse("float")).isEqualTo(KvElementType.F32);
		assertThat(KvElementType.parse("q8_0")).isEqualTo(KvElementType.Q8_0);
		assertThat(KvElementType.parse("q4")).isEqualTo(KvElementType.Q4_0);
	}

	@Test
	@DisplayName("an unknown type names every accepted one")
	void refusesUnknown() {
		assertThatThrownBy(() -> KvElementType.parse("bf16")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("f16|f32|q8_0|q4_0");
	}

	@Test
	@DisplayName("CLI names round-trip")
	void cliNames() {
		for (KvElementType t : KvElementType.values())
			assertThat(KvElementType.parse(t.cliName())).isEqualTo(t);
		assertThat(KvElementType.F32.cliName()).isEqualTo("f32");
	}

	@Test
	@DisplayName("f16 stores 2 bytes a value and f32 4")
	void sizes() {
		assertThat(KvElementType.F16.bytesPerElement()).isEqualTo(2.0);
		assertThat(KvElementType.F32.bytesPerElement()).isEqualTo(4.0);
		assertThat(KvRowCodec.rowBytes(KvElementType.F32, 96)).isEqualTo(96 * Float.BYTES);
	}

	@Test
	@DisplayName("ordinals: F16 0, Q8_0 1, Q4_0 2, F32 appended as 3")
	void ordinals() {
		assertThat(KvElementType.F16.ordinal()).isZero();
		assertThat(KvElementType.Q8_0.ordinal()).isEqualTo(1);
		assertThat(KvElementType.Q4_0.ordinal()).isEqualTo(2);
		assertThat(KvElementType.F32.ordinal()).isEqualTo(3);
	}
}
