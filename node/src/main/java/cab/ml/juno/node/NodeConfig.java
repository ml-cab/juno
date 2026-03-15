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

/**
 * Immutable configuration for an inference node. Loaded from
 * cluster-config.yaml on startup.
 */
public record NodeConfig(String nodeId, String host, int grpcPort, int deviceId, // CUDA device index (0-based)
		double vramHeadroomFraction // reserve this fraction of VRAM (default 0.10)
) {

	public NodeConfig {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (host == null || host.isBlank())
			throw new IllegalArgumentException("host must not be blank");
		if (grpcPort < 1 || grpcPort > 65535)
			throw new IllegalArgumentException("grpcPort out of range: " + grpcPort);
		if (deviceId < 0)
			throw new IllegalArgumentException("deviceId must be >= 0");
		if (vramHeadroomFraction < 0.0 || vramHeadroomFraction >= 1.0)
			throw new IllegalArgumentException("vramHeadroomFraction must be in [0.0, 1.0)");
	}

	public static NodeConfig defaults(String nodeId, String host) {
		return new NodeConfig(nodeId, host, 9091, 0, 0.10);
	}
}
