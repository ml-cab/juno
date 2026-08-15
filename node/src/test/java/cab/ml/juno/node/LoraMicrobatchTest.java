/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraMicrobatch")
class LoraMicrobatchTest {

	private String prev;

	@BeforeEach
	void save() {
		prev = System.getProperty(LoraMicrobatch.PROPERTY);
	}

	@AfterEach
	void restore() {
		if (prev == null)
			System.clearProperty(LoraMicrobatch.PROPERTY);
		else
			System.setProperty(LoraMicrobatch.PROPERTY, prev);
	}

	@Test
	@DisplayName("validate accepts 1..MAX and rejects out of range")
	void validate_bounds() {
		assertThat(LoraMicrobatch.validate(1)).isEqualTo(1);
		assertThat(LoraMicrobatch.validate(LoraMicrobatch.DEFAULT)).isEqualTo(8);
		assertThat(LoraMicrobatch.validate(LoraMicrobatch.MAX)).isEqualTo(128);
		assertThatThrownBy(() -> LoraMicrobatch.validate(0))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("1..128");
		assertThatThrownBy(() -> LoraMicrobatch.validate(129))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("1..128");
	}

	@Test
	@DisplayName("normalize blank/null → DEFAULT; parses integers")
	void normalize() {
		assertThat(LoraMicrobatch.normalize(null)).isEqualTo(LoraMicrobatch.DEFAULT);
		assertThat(LoraMicrobatch.normalize("")).isEqualTo(LoraMicrobatch.DEFAULT);
		assertThat(LoraMicrobatch.normalize("  ")).isEqualTo(LoraMicrobatch.DEFAULT);
		assertThat(LoraMicrobatch.normalize("1")).isEqualTo(1);
		assertThat(LoraMicrobatch.normalize(" 16 ")).isEqualTo(16);
		assertThatThrownBy(() -> LoraMicrobatch.normalize("x"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("lora-microbatch");
	}

	@Test
	@DisplayName("apply sets property; current reads it; blank property → DEFAULT")
	void apply_and_current() {
		System.clearProperty(LoraMicrobatch.PROPERTY);
		assertThat(LoraMicrobatch.current()).isEqualTo(LoraMicrobatch.DEFAULT);

		LoraMicrobatch.apply(1);
		assertThat(System.getProperty(LoraMicrobatch.PROPERTY)).isEqualTo("1");
		assertThat(LoraMicrobatch.current()).isEqualTo(1);

		LoraMicrobatch.apply(8);
		assertThat(LoraMicrobatch.current()).isEqualTo(8);

		System.setProperty(LoraMicrobatch.PROPERTY, "0");
		assertThat(LoraMicrobatch.current()).isEqualTo(1);
	}
}
