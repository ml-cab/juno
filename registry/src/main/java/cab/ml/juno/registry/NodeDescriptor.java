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
package cab.ml.juno.registry;

import java.io.Serializable;
import java.time.Instant;

/**
 * Immutable snapshot of an inference node's identity, resources and health.
 * Stored in Hazelcast IMap("node-registry") — must be Serializable.
 *
 * Nodes self-register on startup by writing their descriptor into the IMap.
 * Updated periodically by the health monitor (vram, temperature, status).
 */
public record NodeDescriptor(String nodeId, // unique node identifier e.g. "node-192.168.1.10-0"
		String host, // hostname or IP
		int grpcPort, // gRPC port for forward pass calls
		long vramTotalBytes, // total GPU VRAM
		long vramFreeBytes, // available GPU VRAM (updated by health probe)
		NodeStatus status, double seedScore, // IMQ-inspired election score (updated by registry)
		Instant registeredAt, Instant lastSeenAt) implements Serializable {

	public NodeDescriptor {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (host == null || host.isBlank())
			throw new IllegalArgumentException("host must not be blank");
		if (grpcPort < 1 || grpcPort > 65535)
			throw new IllegalArgumentException("grpcPort out of range: " + grpcPort);
		if (vramTotalBytes < 0)
			throw new IllegalArgumentException("vramTotalBytes must be >= 0");
		if (vramFreeBytes < 0 || vramFreeBytes > vramTotalBytes)
			throw new IllegalArgumentException("vramFreeBytes invalid: " + vramFreeBytes);
		if (status == null)
			throw new IllegalArgumentException("status must not be null");
	}

	/** VRAM utilization fraction 0.0 → 1.0 */
	public double vramPressure() {
		if (vramTotalBytes == 0)
			return 0.0;
		return 1.0 - ((double) vramFreeBytes / vramTotalBytes);
	}

	/** Whether this node can accept new shard assignments. */
	public boolean isAvailable() {
		return status == NodeStatus.IDLE || status == NodeStatus.READY;
	}

	/** How many bytes of VRAM are reserved (10% headroom). */
	public long usableVramBytes() {
		return (long) (vramTotalBytes * 0.90);
	}

	// ── Fluent updaters ───────────────────────────────────────────────────────

	public NodeDescriptor withStatus(NodeStatus status) {
		return new NodeDescriptor(nodeId, host, grpcPort, vramTotalBytes, vramFreeBytes, status, seedScore,
				registeredAt, lastSeenAt);
	}

	public NodeDescriptor withVramFree(long vramFreeBytes) {
		return new NodeDescriptor(nodeId, host, grpcPort, vramTotalBytes, vramFreeBytes, status, seedScore,
				registeredAt, lastSeenAt);
	}

	public NodeDescriptor withSeedScore(double seedScore) {
		return new NodeDescriptor(nodeId, host, grpcPort, vramTotalBytes, vramFreeBytes, status, seedScore,
				registeredAt, lastSeenAt);
	}

	public NodeDescriptor withLastSeen(Instant lastSeenAt) {
		return new NodeDescriptor(nodeId, host, grpcPort, vramTotalBytes, vramFreeBytes, status, seedScore,
				registeredAt, lastSeenAt);
	}
}
