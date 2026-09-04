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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("MmqOptions — parse and prefer")
class MmqOptionsTest {

	private String original;

	@AfterEach
	void restore() {
		if (original == null)
			System.clearProperty(MmqOptions.ENV_PROPERTY);
		else
			System.setProperty(MmqOptions.ENV_PROPERTY, original);
	}

	@Test
	@DisplayName("parse on|off|auto and aliases")
	void parse_values() {
		assertThat(MmqOptions.parse("off").mode()).isEqualTo(MmqOptions.Mode.OFF);
		assertThat(MmqOptions.parse("ON").mode()).isEqualTo(MmqOptions.Mode.ON);
		assertThat(MmqOptions.parse("auto").mode()).isEqualTo(MmqOptions.Mode.AUTO);
		assertThat(MmqOptions.parse("1").mode()).isEqualTo(MmqOptions.Mode.ON);
		assertThat(MmqOptions.parse("0").mode()).isEqualTo(MmqOptions.Mode.OFF);
		assertThat(MmqOptions.parse(null).mode()).isEqualTo(MmqOptions.Mode.OFF);
		assertThat(MmqOptions.parse("").mode()).isEqualTo(MmqOptions.Mode.OFF);
	}

	@Test
	@DisplayName("invalid spec fails closed")
	void parse_invalid() {
		assertThatThrownBy(() -> MmqOptions.parse("maybe"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("on|off|auto");
	}

	@Test
	@DisplayName("preferMmq respects mode")
	void prefer_mmq() {
		assertThat(MmqOptions.off().preferMmq()).isFalse();
		assertThat(MmqOptions.on().preferMmq()).isTrue();
		assertThat(MmqOptions.auto().preferMmq()).isEqualTo(CudaAvailability.isAvailable());
	}

	@Test
	@DisplayName("fromEnv reads JUNO_MMQ")
	void from_env() {
		original = System.getProperty(MmqOptions.ENV_PROPERTY);
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		assertThat(MmqOptions.fromEnv().mode()).isEqualTo(MmqOptions.Mode.ON);
	}
}
