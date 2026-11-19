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

@DisplayName("CacheTypeOptions / KvElementType")
class CacheTypeOptionsTest {

	@Test
	void parse_aliases() {
		assertThat(KvElementType.parse("f16")).isEqualTo(KvElementType.F16);
		assertThat(KvElementType.parse("F32")).isEqualTo(KvElementType.F32);
		assertThat(KvElementType.parse("q8_0")).isEqualTo(KvElementType.Q8_0);
		assertThat(KvElementType.parse("q4_0")).isEqualTo(KvElementType.Q4_0);
		assertThat(KvElementType.parse(" Q4 ")).isEqualTo(KvElementType.Q4_0);
		assertThat(KvElementType.parse("")).isEqualTo(KvElementType.F16);
	}

	@Test
	void cli_names_round_trip() {
		for (KvElementType t : KvElementType.values())
			assertThat(KvElementType.parse(t.cliName())).isEqualTo(t);
		assertThat(KvElementType.Q4_0.cliName()).isEqualTo("q4_0");
	}

	/** The serialized KV block names a type by its ordinal, so existing blocks keep reading as written. */
	@Test
	void ordinals_are_stable() {
		assertThat(KvElementType.F16.ordinal()).isZero();
		assertThat(KvElementType.Q8_0.ordinal()).isEqualTo(1);
		assertThat(KvElementType.Q4_0.ordinal()).isEqualTo(2);
	}

	@Test
	void parse_rejects_unknown() {
		assertThatThrownBy(() -> KvElementType.parse("q4_1"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("f16|f32|q8_0|q4_0");
	}

	@Test
	void fromEnv_defaults() {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		try {
			System.clearProperty(CacheTypeOptions.ENV_K);
			System.clearProperty(CacheTypeOptions.ENV_V);
			CacheTypeOptions o = CacheTypeOptions.fromEnv();
			assertThat(o.typeK()).isEqualTo(KvElementType.F16);
			assertThat(o.typeV()).isEqualTo(KvElementType.F16);
			assertThat(o.usesQuantized()).isFalse();
			assertThat(o.policySummary()).contains("f16");
		} finally {
			restore(CacheTypeOptions.ENV_K, prevK);
			restore(CacheTypeOptions.ENV_V, prevV);
		}
	}

	@Test
	void fromEnv_q8() {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		try {
			System.setProperty(CacheTypeOptions.ENV_K, "q8_0");
			System.setProperty(CacheTypeOptions.ENV_V, "q8_0");
			CacheTypeOptions o = CacheTypeOptions.fromEnv();
			assertThat(o.usesQuantized()).isTrue();
			assertThat(o.typeK()).isEqualTo(KvElementType.Q8_0);
		} finally {
			restore(CacheTypeOptions.ENV_K, prevK);
			restore(CacheTypeOptions.ENV_V, prevV);
		}
	}

	@Test
	void fromEnv_q4_on_one_side_is_quantized() {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		try {
			System.setProperty(CacheTypeOptions.ENV_K, "f16");
			System.setProperty(CacheTypeOptions.ENV_V, "q4_0");
			CacheTypeOptions o = CacheTypeOptions.fromEnv();
			assertThat(o.usesQuantized()).isTrue();
			assertThat(o.typeV()).isEqualTo(KvElementType.Q4_0);
			assertThat(o.policySummary()).isEqualTo("cache-type-k=f16 cache-type-v=q4_0");
		} finally {
			restore(CacheTypeOptions.ENV_K, prevK);
			restore(CacheTypeOptions.ENV_V, prevV);
		}
	}

	private static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
