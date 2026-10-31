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

import java.util.List;
import java.util.Map;
import java.util.Set;

/** Detected hosts for tests: a 128-bit CPU-only machine and a 256-bit CUDA machine. */
final class HostFixtures {

	private HostFixtures() {
	}

	static HostInfo cpuOnly128() {
		return new HostInfo("Linux", "6.8", "amd64", "Test CPU", 6, 12, 64L << 30, 500L << 30, Set.of("avx"), 128,
				128, "1", false, false, List.of(), new HostInfo.Power(false, false, -1), "no", "hpet", Map.of());
	}

	static HostInfo cuda256() {
		return new HostInfo("Linux", "6.8", "amd64", "Test CPU", 8, 16, 32L << 30, 500L << 30, Set.of("avx", "avx2"),
				256, 256, "2", true, false, List.of(new HostInfo.Gpu("NVIDIA", "Test GPU", 8L << 30, "580")),
				new HostInfo.Power(false, false, -1), "no", "tsc", Map.of());
	}
}
