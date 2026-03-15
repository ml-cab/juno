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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Pure domain evaluator — no Hazelcast, no JCuda, no I/O.
 *
 * Receives NodeHealth snapshots (from whatever source — Hazelcast IMap listener
 * in production, direct calls in tests) and emits HealthEvents when thresholds
 * are crossed.
 *
 * State tracked: - Previous health level per node (to avoid duplicate events) -
 * Whether a node was previously stale (to emit NODE_RECOVERED)
 *
 * Thread-safe — ConcurrentHashMap for state, evaluate() is idempotent.
 */
public final class HealthEvaluator {

	private final HealthThresholds thresholds;

	// Track previous state to detect transitions, not just current values
	private final Map<String, HealthLevel> previousLevel = new ConcurrentHashMap<>();
	private final Map<String, Boolean> wasStale = new ConcurrentHashMap<>();

	public HealthEvaluator(HealthThresholds thresholds) {
		if (thresholds == null)
			throw new IllegalArgumentException("thresholds must not be null");
		this.thresholds = thresholds;
	}

	/**
	 * Evaluate a fresh NodeHealth snapshot. Returns zero or more HealthEvents
	 * representing state transitions.
	 */
	public List<HealthEvent> evaluate(NodeHealth health) {
		List<HealthEvent> events = new ArrayList<>();
		String nodeId = health.nodeId();

		// ── Staleness check ───────────────────────────────────────────────────
		boolean stale = health.ageMillis() > thresholds.staleAfterMs();
		boolean wasPreviouslyStale = wasStale.getOrDefault(nodeId, false);

		if (stale && !wasPreviouslyStale) {
			events.add(HealthEvent.stale(nodeId, health.ageMillis()));
			wasStale.put(nodeId, true);
			return events; // no further evaluation on stale data
		}
		if (!stale && wasPreviouslyStale) {
			events.add(HealthEvent.recovered(nodeId));
			wasStale.put(nodeId, false);
		}

		if (stale)
			return events;

		// ── VRAM threshold evaluation ─────────────────────────────────────────
		HealthLevel current = levelFor(health);
		HealthLevel previous = previousLevel.getOrDefault(nodeId, HealthLevel.OK);

		if (current == HealthLevel.CRITICAL && previous != HealthLevel.CRITICAL) {
			events.add(HealthEvent.vramCritical(nodeId, health.vramPressure()));
		} else if (current == HealthLevel.WARNING && previous == HealthLevel.OK) {
			events.add(HealthEvent.vramWarning(nodeId, health.vramPressure()));
		} else if (current == HealthLevel.OK && previous != HealthLevel.OK) {
			events.add(HealthEvent.recovered(nodeId));
		}

		previousLevel.put(nodeId, current);
		return events;
	}

	/** Reset tracked state for a node (called when node leaves the cluster). */
	public void forget(String nodeId) {
		previousLevel.remove(nodeId);
		wasStale.remove(nodeId);
	}

	private HealthLevel levelFor(NodeHealth health) {
		if (health.isVramCritical(thresholds.vramCritical()))
			return HealthLevel.CRITICAL;
		if (health.isVramWarning(thresholds.vramWarning()))
			return HealthLevel.WARNING;
		return HealthLevel.OK;
	}

	private enum HealthLevel {
		OK, WARNING, CRITICAL
	}
}
