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

/**
 * User-facing policy for whether inference uses CUDA ({@link CudaMatVec}) or stays
 * on the CPU ({@link CpuMatVec}).
 */
public enum ComputeBackendPreference {

	/**
	 * Use CUDA when {@link CudaAvailability#isAvailable()}; otherwise CPU only.
	 */
	AUTO,

	/** Never initialise CUDA or load GPU-resident weights. */
	DISABLE,

	/**
	 * CUDA must be available; otherwise startup or {@code loadShard} fails (no
	 * silent CPU fallback for real-model loads).
	 */
	REQUIRE;

	/**
	 * Value for {@code -DJUNO_USE_GPU=} on forked node JVMs and
	 * {@link System#setProperty(String, String)} on the coordinator.
	 */
	public String toPropertyToken() {
		return switch (this) {
		case DISABLE -> "false";
		case AUTO -> "auto";
		case REQUIRE -> "require";
		};
	}
}
