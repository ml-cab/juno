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
import cab.ml.juno.health.HealthMain;
import cab.ml.juno.health.HealthThresholds;

/**
 * Entry point for a standalone node JVM process.
 *
 * Configuration is read from system properties (no command-line arguments):
 *   -Dnode.id=<nodeId>          required
 *   -Dnode.port=<port>          required
 *   -Dmodel.path=<modelPath>    optional (if missing, runs with dummy handler)
 *   -DJUNO_USE_GPU=<true|false> optional, defaults to true
 *
 * Health sidecar (optional — starts alongside the node, does NOT replace it):
 *   -Djuno.health=true          enable the health HTTP sidecar
 *   -Djuno.health.port=8081     port for the health sidecar (default: 8081)
 *   -Djuno.health.staleMs=15000 staleness threshold in ms (default: 15000)
 *   -Djuno.health.warn=0.90     VRAM warning threshold (default: 0.90)
 *   -Djuno.health.critical=0.98 VRAM critical threshold (default: 0.98)
 *
 * Example:
 *   java -Dnode.id=node-1 -Dnode.port=9092 -Dmodel.path=/models/model.gguf \
 *        -Djuno.health=true -Djuno.health.port=8082 \
 *        -jar juno-node.jar
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

        // ── Health sidecar (optional) ─────────────────────────────────────────
        if ("true".equalsIgnoreCase(System.getProperty("juno.health", "false"))) {
            int    healthPort    = parsePropInt   ("juno.health.port",     HealthMain.DEFAULT_PORT);
            long   healthStaleMs = parsePropLong  ("juno.health.staleMs",  15_000L);
            double healthWarn    = parsePropDouble("juno.health.warn",     0.90);
            double healthCrit    = parsePropDouble("juno.health.critical", 0.98);
            HealthMain.startBackground(healthPort,
                new HealthThresholds(healthWarn, healthCrit, healthStaleMs));
            log.info("Health sidecar started on :" + healthPort);
        }

        EmbeddedNodeServer server = new EmbeddedNodeServer(nodeId, port, modelPath, useGpu);
        server.start();

        // Signal readiness to the parent process (ClusterHarness polls for this line)
        System.out.println("READY:" + nodeId + ":" + port);
        System.out.flush();

        log.info("Node [" + nodeId + "] running on port " + port + " — waiting for requests");
        server.blockUntilShutdown();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static int parsePropInt(String key, int def) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return def;
        try { return Integer.parseInt(v.trim()); } catch (NumberFormatException e) { return def; }
    }

    private static long parsePropLong(String key, long def) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return def;
        try { return Long.parseLong(v.trim()); } catch (NumberFormatException e) { return def; }
    }

    private static double parsePropDouble(String key, double def) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return def;
        try { return Double.parseDouble(v.trim()); } catch (NumberFormatException e) { return def; }
    }
}