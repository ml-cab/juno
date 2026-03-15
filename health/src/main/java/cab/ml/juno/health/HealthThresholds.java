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

package cab.ml.juno.health;

/**
 * Configurable health thresholds.
 *
 * Defaults match the architecture spec: vramWarning = 0.90 (90%) → evict cold
 * KV blocks, reduce batch size vramCritical = 0.98 (98%) → open circuit
 * breaker, trigger reshard staleAfterMs = 15_000 → 3 missed probes (probe
 * interval = 5s)
 */
public record HealthThresholds(double vramWarning, double vramCritical, long staleAfterMs) {

	public HealthThresholds {
		if (vramWarning <= 0.0 || vramWarning >= 1.0)
			throw new IllegalArgumentException("vramWarning must be in (0.0, 1.0)");
		if (vramCritical <= 0.0 || vramCritical > 1.0)
			throw new IllegalArgumentException("vramCritical must be in (0.0, 1.0]");
		if (vramWarning >= vramCritical)
			throw new IllegalArgumentException("vramWarning must be less than vramCritical");
		if (staleAfterMs < 1)
			throw new IllegalArgumentException("staleAfterMs must be >= 1");
	}

	public static HealthThresholds defaults() {
		return new HealthThresholds(0.90, 0.98, 15_000);
	}
}
