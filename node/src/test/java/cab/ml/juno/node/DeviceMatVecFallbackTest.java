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

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Only a device out-of-memory failure sends a resident-weight product to the CPU. */
@DisplayName("DeviceMatVecFallback")
class DeviceMatVecFallbackTest {

	@Test
	@DisplayName("a device allocation failure is absorbed, every time, on both GPU vendors")
	void absorbsOutOfMemory() {
		DeviceMatVecFallback f = new DeviceMatVecFallback("test");
		assertThatCode(() -> {
			f.absorb(new IllegalStateException("cudaMalloc failed: rc=2"));
			f.absorb(new IllegalStateException("cudaMalloc failed: rc=2"));
			f.absorb(new IllegalStateException("hipMalloc failed: rc=2"));
		}).doesNotThrowAnyException();
	}

	@Test
	@DisplayName("any other failure is rethrown unchanged, so a real defect still fails the request")
	void rethrowsEverythingElse() {
		DeviceMatVecFallback f = new DeviceMatVecFallback("test");
		IllegalStateException closed = new IllegalStateException("DeviceQ4KMatrix is closed");
		assertThatThrownBy(() -> f.absorb(closed)).isSameAs(closed);
		IllegalStateException kernel = new IllegalStateException("cuLaunchKernel failed: rc=719");
		assertThatThrownBy(() -> f.absorb(kernel)).isSameAs(kernel);
		IllegalStateException noMessage = new IllegalStateException();
		assertThatThrownBy(() -> f.absorb(noMessage)).isSameAs(noMessage);
	}
}
