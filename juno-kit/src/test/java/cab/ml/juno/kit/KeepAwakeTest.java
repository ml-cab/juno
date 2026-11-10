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
package cab.ml.juno.kit;


import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class KeepAwakeTest {

	@Test
	void eachOsGetsAHelperThatEndsWithTheKit() {
		assertThat(KeepAwake.command("Mac OS X", 42, false)).containsExactly("caffeinate", "-i", "-w", "42");
		assertThat(KeepAwake.command("Linux", 42, true)).contains("systemd-inhibit", "--what=idle:sleep", "--pid=42");
		assertThat(KeepAwake.command("Linux", 42, false)).isEmpty();
		assertThat(KeepAwake.command("Windows 11", 42, true)).isEmpty(); // FFM call instead of a helper
	}

	@Test
	void theWindowsCallSaysWhetherItHeld() {
		// SetThreadExecutionState returns the previous state, or 0 when it failed.
		assertThat(KeepAwake.windowsMessage(0x80000000)).contains("Keeping this machine awake");
		assertThat(KeepAwake.windowsMessage(0)).contains("Could not keep Windows awake", "SetThreadExecutionState");
	}
}
