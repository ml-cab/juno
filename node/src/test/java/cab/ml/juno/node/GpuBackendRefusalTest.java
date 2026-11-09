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

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * An explicitly chosen GPU backend that cannot run refuses with the backend's own one-line reason (the
 * platform refusal on Windows or macOS, the missing library on Linux), not a bare "not available".
 */
@DisplayName("GpuContext.selectBindings: an explicit backend that cannot run says why")
class GpuBackendRefusalTest {

	private String previous;

	@BeforeEach
	void saveBackend() {
		previous = System.getProperty("juno.gpu.backend");
	}

	@AfterEach
	void restoreBackend() {
		if (previous == null)
			System.clearProperty("juno.gpu.backend");
		else
			System.setProperty("juno.gpu.backend", previous);
	}

	@Test
	@DisplayName("juno.gpu.backend=rocm on a Linux host without ROCm names the library it could not load")
	void forcedRocmWithoutRocmSaysWhy() {
		assumeTrue(GpuPlatform.current() == GpuPlatform.LINUX && !RocmBindings.isAvailable(),
				"Skipping: needs Linux without ROCm");
		System.setProperty("juno.gpu.backend", "rocm");
		assertThatThrownBy(GpuContext::selectBindings).isInstanceOf(IllegalStateException.class)
				.hasMessage("juno.gpu.backend=rocm but ROCm not available: "
						+ "Could not load GPU library — tried: [libamdhip64.so]");
	}
}
