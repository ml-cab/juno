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
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("GpuPlatform: which GPU backend each operating system supports")
class GpuPlatformTest {

	@Test
	@DisplayName("os.name values map to the platform")
	void osNamesMapToThePlatform() {
		assertThat(GpuPlatform.of("Linux")).isEqualTo(GpuPlatform.LINUX);
		assertThat(GpuPlatform.of("Windows 11")).isEqualTo(GpuPlatform.WINDOWS);
		assertThat(GpuPlatform.of("Windows Server 2022")).isEqualTo(GpuPlatform.WINDOWS);
		assertThat(GpuPlatform.of("Mac OS X")).isEqualTo(GpuPlatform.MACOS);
		assertThat(GpuPlatform.of("FreeBSD")).isEqualTo(GpuPlatform.OTHER);
		assertThat(GpuPlatform.of("")).isEqualTo(GpuPlatform.OTHER);
		assertThat(GpuPlatform.of(null)).isEqualTo(GpuPlatform.OTHER);
	}

	@Test
	@DisplayName("CUDA runs on Linux and Windows and is refused on macOS with one line")
	void cudaIsRefusedOnMacOs() {
		assertThat(GpuPlatform.LINUX.cudaRefusal()).isNull();
		assertThat(GpuPlatform.WINDOWS.cudaRefusal()).isNull();
		assertThatCode(GpuPlatform.LINUX::requireCuda).doesNotThrowAnyException();
		assertThatCode(GpuPlatform.WINDOWS::requireCuda).doesNotThrowAnyException();

		assertThat(GpuPlatform.MACOS.cudaRefusal())
				.isEqualTo("CUDA is not available on macOS: NVIDIA's CUDA does not support it, so Juno runs on the CPU here.");
		assertThatThrownBy(GpuPlatform.MACOS::requireCuda).isInstanceOf(IllegalStateException.class)
				.hasMessage(GpuPlatform.MACOS.cudaRefusal());
		assertThat(GpuPlatform.OTHER.cudaRefusal()).isEqualTo("CUDA is supported on Linux and Windows only.");
	}

	@Test
	@DisplayName("ROCm runs on Linux only and is refused on Windows and macOS with one line")
	void rocmIsRefusedOutsideLinux() {
		assertThat(GpuPlatform.LINUX.rocmRefusal()).isNull();
		assertThatCode(GpuPlatform.LINUX::requireRocm).doesNotThrowAnyException();

		assertThat(GpuPlatform.WINDOWS.rocmRefusal())
				.isEqualTo("ROCm is not supported on Windows: Juno's ROCm backend runs on Linux only. "
						+ "On Windows Juno uses CUDA or the CPU.");
		assertThatThrownBy(GpuPlatform.WINDOWS::requireRocm).isInstanceOf(IllegalStateException.class)
				.hasMessage(GpuPlatform.WINDOWS.rocmRefusal());
		assertThat(GpuPlatform.MACOS.rocmRefusal())
				.isEqualTo("ROCm is not supported on macOS: Juno's ROCm backend runs on Linux only.");
		assertThat(GpuPlatform.OTHER.rocmRefusal()).isEqualTo("ROCm is supported on Linux only.");
	}

	@Test
	@DisplayName("every refusal is a single line")
	void refusalsAreOneLine() {
		for (GpuPlatform p : GpuPlatform.values()) {
			if (p.cudaRefusal() != null)
				assertThat(p.cudaRefusal()).doesNotContain("\n");
			if (p.rocmRefusal() != null)
				assertThat(p.rocmRefusal()).doesNotContain("\n");
		}
	}
}
