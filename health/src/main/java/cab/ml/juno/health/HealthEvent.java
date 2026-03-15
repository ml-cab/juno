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

import java.time.Instant;

/**
 * A health state transition event emitted by HealthEvaluator. Consumed by the
 * coordinator and registry to trigger appropriate reactions.
 *
 * Event types map to the cascade from the architecture doc: VRAM_WARNING →
 * evict cold KV blocks, reduce batch size VRAM_CRITICAL → open circuit breaker,
 * trigger reshard NODE_STALE → treat as offline, trigger reshard NODE_RECOVERED
 * → reshard back in
 */
public record HealthEvent(String nodeId, EventType type, String detail, Instant occurredAt) {

	public enum EventType {
		VRAM_WARNING, VRAM_CRITICAL, NODE_STALE, NODE_RECOVERED
	}

	public static HealthEvent vramWarning(String nodeId, double pressure) {
		return new HealthEvent(nodeId, EventType.VRAM_WARNING, String.format("VRAM pressure %.1f%%", pressure * 100),
				Instant.now());
	}

	public static HealthEvent vramCritical(String nodeId, double pressure) {
		return new HealthEvent(nodeId, EventType.VRAM_CRITICAL,
				String.format("VRAM pressure %.1f%% — circuit opening", pressure * 100), Instant.now());
	}

	public static HealthEvent stale(String nodeId, long ageMs) {
		return new HealthEvent(nodeId, EventType.NODE_STALE, String.format("No health probe for %dms", ageMs),
				Instant.now());
	}

	public static HealthEvent recovered(String nodeId) {
		return new HealthEvent(nodeId, EventType.NODE_RECOVERED, "Health probe resumed", Instant.now());
	}
}
