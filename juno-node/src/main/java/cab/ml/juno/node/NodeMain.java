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

import java.util.logging.Logger;

/**
 * Entry point for a standalone node JVM process.
 *
 * Configuration is read from system properties (no command-line arguments):
 *   -Dnode.id=<nodeId>          required
 *   -Dnode.port=<port>          required
 *   -Dmodel.path=<modelPath>    optional (if missing, runs with dummy handler)
 *   -DJUNO_USE_GPU=<true|false> optional, defaults to true
 *   -Djuno.cuda.device=<0..N-1> optional CUDA device index for {@link cab.ml.juno.node.GpuContext#shared(int)}
 *
 * Example:
 *   java -Dnode.id=node-1 -Dnode.port=9092 -Dmodel.path=/models/model.gguf \
 *        -jar juno-node.jar cab.ml.juno.node.NodeMain
 *
 * The old command-line signature (nodeId port modelPath) is still supported
 * for backward compatibility, but system properties take precedence.
 */
public final class NodeMain {

    private static final Logger log = Logger.getLogger(NodeMain.class.getName());

    public static void main(String[] args) throws Exception {
        // Try system properties first
        String nodeId = System.getProperty("node.id");
        String portStr = System.getProperty("node.port");
        String modelPath = System.getProperty("model.path");

        // Fallback to command-line arguments (backward compatibility)
        if (nodeId == null && args.length >= 1) {
            nodeId = args[0];
        }
        if (portStr == null && args.length >= 2) {
            portStr = args[1];
        }
        if (modelPath == null && args.length >= 3) {
            modelPath = args[2];
        }

        // Validate
        if (nodeId == null || nodeId.isBlank()) {
            System.err.println("Missing required property: -Dnode.id=<nodeId>");
            System.exit(1);
        }
        if (portStr == null || portStr.isBlank()) {
            System.err.println("Missing required property: -Dnode.port=<port>");
            System.exit(1);
        }

        int port;
        try {
            port = Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            System.err.println("Invalid port: " + portStr);
            System.exit(1);
            return;
        }

        boolean useGpu = "true".equalsIgnoreCase(System.getProperty("JUNO_USE_GPU", "true"));

        // Forward JUNO_LORA_PLAY_PATH env var as system property so EmbeddedNodeServer
        // can pick it up without requiring a JVM flag relaunch.
        String loraPlayPath = System.getenv("JUNO_LORA_PLAY_PATH");
        if (loraPlayPath != null && !loraPlayPath.isBlank()) {
            System.setProperty("juno.lora.play.path", loraPlayPath);
            log.info("LoRA inference overlay enabled: " + loraPlayPath);
        }

        EmbeddedNodeServer server = new EmbeddedNodeServer(nodeId, port, modelPath, useGpu);
        server.start();

        // Signal readiness to the parent process (ClusterHarness polls for this line)
        System.out.println("READY:" + nodeId + ":" + port);
        System.out.flush();

        log.info("Node [" + nodeId + "] running on port " + port + " — waiting for requests");
        server.blockUntilShutdown();
    }
}