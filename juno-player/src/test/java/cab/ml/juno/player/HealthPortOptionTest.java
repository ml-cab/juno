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

package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * {@code --health-port N} names the port of the health sidecar {@code --health} starts. Before this option existed
 * the launchers accepted the flag and printed the port while the sidecar always listened on its default.
 */
class HealthPortOptionTest {

	@AfterEach
	void restoreDefaultPort() throws Exception {
		healthPortField().setInt(null, cab.ml.juno.health.HealthMain.DEFAULT_PORT);
	}

	@Test
	void a_port_number_parses() {
		assertThat(HealthPortOption.parse("9090")).isEqualTo(9090);
		assertThat(HealthPortOption.parse(" 1 ")).isEqualTo(1);
		assertThat(HealthPortOption.parse("65535")).isEqualTo(65535);
	}

	@Test
	void a_value_that_is_not_a_port_is_refused_with_the_flag_named() {
		for (String bad : new String[] { "0", "65536", "-1", "http", "" })
			assertThatThrownBy(() -> HealthPortOption.parse(bad)).isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining("--health-port").hasMessageContaining("'" + bad + "'");
	}

	@Test
	void the_console_takes_the_sidecar_port_from_the_flag() throws Exception {
		Method parseArgs = ConsoleMain.class.getDeclaredMethod("parseArgs", String[].class);
		parseArgs.setAccessible(true);
		parseArgs.invoke(null, (Object) new String[] { "--health-port", "9191" });
		assertThat(healthPortField().getInt(null)).isEqualTo(9191);
	}

	private static Field healthPortField() throws NoSuchFieldException {
		Field f = ConsoleMain.class.getDeclaredField("healthPort");
		f.setAccessible(true);
		return f;
	}
}
