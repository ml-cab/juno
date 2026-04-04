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

package cab.ml.juno.player;

import java.util.logging.Logger;

import cab.ml.juno.node.ComputeBackendPolicy;
import cab.ml.juno.node.ComputeBackendPreference;

/**
 * Entry point for a standalone node JVM process.
 *
 * Launched by ClusterHarness via ProcessBuilder — one JVM per node. Listens on
 * the given gRPC port and serves NodeService via EmbeddedNodeServer.
 *
 * Usage (ClusterHarness handles this automatically): java ...
 * cab.ml.juno.integration.NodeMain <nodeId> <port> [modelPath]
 *
 * When modelPath is supplied, EmbeddedNodeServer uses LlamaTransformerHandler
 * (real transformer math) instead of CyclicForwardPassHandler.
 *
 * Manual launch for debugging: mvn exec:java -pl integration \
 * -Dexec.mainClass=cab.ml.juno.integration.NodeMain \ -Dexec.args="node-1 9092
 * /models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
 */
public final class NodeMain {

	private static final Logger log = Logger.getLogger(NodeMain.class.getName());

	public static void main(String[] args) throws Exception {
		if (args.length < 2) {
			System.err.println("Usage: NodeMain <nodeId> <port> [modelPath]");
			System.exit(1);
		}

		String nodeId = args[0];
		int port = Integer.parseInt(args[1]);
		String modelPath = args.length >= 3 ? args[2] : null;
		ComputeBackendPreference pref = ComputeBackendPolicy
				.parsePreference(System.getProperty("JUNO_USE_GPU", ComputeBackendPreference.AUTO.toPropertyToken()));

		EmbeddedNodeServer server = new EmbeddedNodeServer(nodeId, port, modelPath, pref);
		server.start();

		// Signal readiness to the parent process (ClusterHarness polls for this line)
		System.out.println("READY:" + nodeId + ":" + port);
		System.out.flush();

		log.info("Node [" + nodeId + "] running on port " + port + " — waiting for requests");
		server.blockUntilShutdown();
	}
}