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

import java.io.Serializable;
import java.time.Instant;

/**
 * Immutable health snapshot for a single inference node. Published by each
 * node's GPU health probe every 5s. Stored in Hazelcast IMap("node-health")
 * keyed by nodeId.
 */
public record NodeHealth(String nodeId, double vramPressure, // 0.0 → 1.0
		long vramFreeBytes, long vramTotalBytes, double temperatureCelsius, // -1.0 if unavailable
		double inferenceLatencyP99Ms, // -1.0 if no data yet
		Instant sampledAt) implements Serializable {

	public NodeHealth {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (vramPressure < 0.0 || vramPressure > 1.0)
			throw new IllegalArgumentException("vramPressure must be in [0.0, 1.0]");
	}

	/** Whether VRAM is above the warning threshold (default 90%). */
	public boolean isVramWarning(double threshold) {
		return vramPressure >= threshold;
	}

	/** Whether VRAM is above the critical threshold (default 98%). */
	public boolean isVramCritical(double threshold) {
		return vramPressure >= threshold;
	}

	/** How stale this snapshot is. */
	public long ageMillis() {
		return Instant.now().toEpochMilli() - sampledAt.toEpochMilli();
	}
}
