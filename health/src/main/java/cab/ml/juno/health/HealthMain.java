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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import io.javalin.http.Context;
import io.javalin.Javalin;

/**
 * Standalone health-monitor HTTP server.
 *
 * <p>Launched when the player fat-jar is invoked with {@code --health}, or
 * directly via {@code java -jar juno-health.jar [--port N]}.
 *
 * <h2>API surface</h2>
 * <pre>
 *   POST /health/probe          Accept a NodeHealth snapshot from any node.
 *                               Body: JSON matching {@link NodeHealthDto}.
 *                               Runs the HealthEvaluator and updates the
 *                               per-node circuit breaker accordingly.
 *
 *   GET  /health                Cluster overview — all known nodes, their last
 *                               snapshot, circuit state, and a top-level
 *                               "status" field (HEALTHY / DEGRADED / DOWN).
 *
 *   GET  /health/nodes/{nodeId} Single-node detail (404 if unknown).
 *
 *   GET  /health/circuits       Per-node circuit-breaker states only (compact).
 * </pre>
 *
 * <h2>Configuration</h2>
 * <pre>
 *   --port N          HTTP listen port (default: {@value #DEFAULT_PORT})
 *   --stale-ms N      ms before a node is considered stale (default: 15000)
 *   --warn F          VRAM warning threshold 0.0–1.0 (default: 0.90)
 *   --critical F      VRAM critical threshold 0.0–1.0 (default: 0.98)
 * </pre>
 *
 * <h2>Thread safety</h2>
 * All mutable state ({@code nodeSnapshots}, {@code circuits}) is held in
 * {@link ConcurrentHashMap}s. {@link HealthEvaluator} uses the same internally.
 * Javalin routes are served on virtual threads so individual route handlers may
 * block without stalling the server.
 */
public final class HealthMain {

    private static final Logger log = Logger.getLogger(HealthMain.class.getName());

    public static final int DEFAULT_PORT = 8081;

    // ── per-node state ────────────────────────────────────────────────────────
    private final Map<String, NodeHealth>      nodeSnapshots = new ConcurrentHashMap<>();
    private final Map<String, CircuitBreaker>  circuits      = new ConcurrentHashMap<>();

    private final HealthEvaluator evaluator;
    private final int port;

    // ── constructor ───────────────────────────────────────────────────────────

    public HealthMain(int port, HealthThresholds thresholds) {
        this.port      = port;
        this.evaluator = new HealthEvaluator(thresholds);
    }

    // ── entry point ───────────────────────────────────────────────────────────

    /**
     * CLI entry point.  Parses {@code --port}, {@code --stale-ms},
     * {@code --warn}, {@code --critical} and starts the server.
     *
     * @param args command-line arguments (may include {@code --health} which
     *             is already consumed by the caller before delegation here)
     */
    public static void main(String[] args) {
        int    port       = envInt   ("HEALTH_PORT",     DEFAULT_PORT);
        long   staleMs    = envLong  ("HEALTH_STALE_MS", 15_000L);
        double warn       = envDouble("HEALTH_WARN",     0.90);
        double critical   = envDouble("HEALTH_CRITICAL", 0.98);

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port"      -> { if (i + 1 < args.length) port     = Integer.parseInt(args[++i]); }
                case "--stale-ms"  -> { if (i + 1 < args.length) staleMs  = Long.parseLong(args[++i]); }
                case "--warn"      -> { if (i + 1 < args.length) warn     = Double.parseDouble(args[++i]); }
                case "--critical"  -> { if (i + 1 < args.length) critical = Double.parseDouble(args[++i]); }
                case "--health"    -> { /* consumed by ConsoleMain, safe to skip */ }
                default            -> log.fine("Ignoring unknown arg: " + args[i]);
            }
        }

        HealthThresholds thresholds = new HealthThresholds(warn, critical, staleMs);
        new HealthMain(port, thresholds).start();
    }

    // ── server lifecycle ──────────────────────────────────────────────────────

    /**
     * Builds and starts the Javalin server. Blocks until the process is killed.
     */
    public void start() {
        Javalin app = Javalin.create(config -> {
            config.showJavalinBanner = false;
            config.useVirtualThreads = true;
        });

        app.post("/health/probe",             this::handleProbe);
        app.get ("/health",                   this::handleClusterOverview);
        app.get ("/health/nodes/{nodeId}",    this::handleNodeDetail);
        app.get ("/health/circuits",          this::handleCircuits);

        app.start(port);
        log.info("Juno health server listening on :" + port);

        // Register shutdown hook so the server drains cleanly on Ctrl-C / SIGTERM.
        Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
            log.info("Health server shutting down…");
            app.stop();
        }));

        // Block the main thread so the process stays alive.
        try {
            Thread.currentThread().join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ── route handlers ────────────────────────────────────────────────────────

    /**
     * {@code POST /health/probe}
     *
     * <p>Accepts a {@link NodeHealthDto} JSON body, converts it to a
     * {@link NodeHealth}, stores the snapshot, runs the {@link HealthEvaluator},
     * and updates the node's {@link CircuitBreaker} based on any emitted
     * {@link HealthEvent}s.
     *
     * <p>Returns {@code 200} with an "events" array listing every state
     * transition that fired as a result of this probe.
     */
    private void handleProbe(Context ctx) {
        NodeHealthDto dto;
        try {
            dto = ctx.bodyAsClass(NodeHealthDto.class);
        } catch (Exception e) {
            ctx.status(400).json(errorBody("BAD_REQUEST", "Invalid NodeHealth body: " + e.getMessage()));
            return;
        }
        if (dto.nodeId() == null || dto.nodeId().isBlank()) {
            ctx.status(400).json(errorBody("BAD_REQUEST", "nodeId must not be blank"));
            return;
        }
        if (dto.vramPressure() < 0.0 || dto.vramPressure() > 1.0) {
            ctx.status(400).json(errorBody("BAD_REQUEST", "vramPressure must be in [0.0, 1.0]"));
            return;
        }

        NodeHealth probe = dto.toNodeHealth();
        nodeSnapshots.put(probe.nodeId(), probe);

        // Ensure a circuit exists for this node before evaluation.
        circuits.computeIfAbsent(probe.nodeId(), CircuitBreaker::forNode);

        List<HealthEvent> events = evaluator.evaluate(probe);
        for (HealthEvent event : events) {
            reactToEvent(event);
        }

        ctx.json(Map.of(
            "nodeId", probe.nodeId(),
            "accepted", true,
            "events", events.stream().map(this::eventToMap).toList()
        ));
    }

    /**
     * {@code GET /health}
     *
     * <p>Returns a cluster-level overview:
     * <ul>
     *   <li>{@code status} — {@code HEALTHY} if every known node's circuit is
     *       CLOSED, {@code DEGRADED} if any circuit is OPEN or HALF_OPEN but at
     *       least one is CLOSED, {@code DOWN} if all circuits are OPEN.</li>
     *   <li>{@code nodeCount} — number of nodes that have ever sent a probe.</li>
     *   <li>{@code nodes} — list of per-node detail objects.</li>
     * </ul>
     */
    private void handleClusterOverview(Context ctx) {
        List<Map<String, Object>> nodes = buildNodeList();
        String status = deriveClusterStatus();
        ctx.json(Map.of(
            "status",    status,
            "nodeCount", nodeSnapshots.size(),
            "nodes",     nodes
        ));
    }

    /**
     * {@code GET /health/nodes/{nodeId}}
     *
     * <p>Returns full detail for a single node, or 404 if no probe has been
     * received from that node.
     */
    private void handleNodeDetail(Context ctx) {
        String nodeId = ctx.pathParam("nodeId");
        NodeHealth snap = nodeSnapshots.get(nodeId);
        if (snap == null) {
            ctx.status(404).json(errorBody("NOT_FOUND", "No health data for node: " + nodeId));
            return;
        }
        ctx.json(buildNodeDetail(nodeId, snap));
    }

    /**
     * {@code GET /health/circuits}
     *
     * <p>Returns only the circuit-breaker state for each known node — useful for
     * the coordinator's routing layer to poll cheaply.
     */
    private void handleCircuits(Context ctx) {
        List<Map<String, String>> result = circuits.entrySet().stream()
            .map(e -> Map.of(
                "nodeId", e.getKey(),
                "state",  e.getValue().state().name()
            ))
            .toList();
        ctx.json(Map.of("circuits", result));
    }

    // ── reaction ──────────────────────────────────────────────────────────────

    private void reactToEvent(HealthEvent event) {
        CircuitBreaker cb = circuits.computeIfAbsent(event.nodeId(), CircuitBreaker::forNode);
        switch (event.type()) {
            case VRAM_CRITICAL -> {
                log.warning("VRAM_CRITICAL on " + event.nodeId() + " — " + event.detail() + " — opening circuit");
                cb.forceOpen();
            }
            case NODE_STALE -> {
                log.warning("NODE_STALE on " + event.nodeId() + " — " + event.detail() + " — opening circuit");
                cb.forceOpen();
            }
            case NODE_RECOVERED -> {
                log.info("NODE_RECOVERED on " + event.nodeId() + " — resetting circuit");
                cb.reset();
            }
            case VRAM_WARNING ->
                log.info("VRAM_WARNING on " + event.nodeId() + " — " + event.detail());
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private List<Map<String, Object>> buildNodeList() {
        List<Map<String, Object>> list = new ArrayList<>();
        for (Map.Entry<String, NodeHealth> e : nodeSnapshots.entrySet()) {
            list.add(buildNodeDetail(e.getKey(), e.getValue()));
        }
        return list;
    }

    private Map<String, Object> buildNodeDetail(String nodeId, NodeHealth snap) {
        CircuitBreaker cb = circuits.computeIfAbsent(nodeId, CircuitBreaker::forNode);
        return Map.of(
            "nodeId",               nodeId,
            "circuit",              cb.state().name(),
            "callPermitted",        cb.isCallPermitted(),
            "vramPressure",         snap.vramPressure(),
            "vramFreeBytes",        snap.vramFreeBytes(),
            "vramTotalBytes",       snap.vramTotalBytes(),
            "temperatureCelsius",   snap.temperatureCelsius(),
            "inferenceLatencyP99",  snap.inferenceLatencyP99Ms(),
            "ageMs",                snap.ageMillis(),
            "sampledAt",            snap.sampledAt().toString()
        );
    }

    private String deriveClusterStatus() {
        if (circuits.isEmpty()) return "HEALTHY";
        long open   = circuits.values().stream().filter(cb -> cb.state() == CircuitState.OPEN).count();
        long total  = circuits.size();
        if (open == 0)     return "HEALTHY";
        if (open == total) return "DOWN";
        return "DEGRADED";
    }

    private Map<String, Object> eventToMap(HealthEvent event) {
        return Map.of(
            "nodeId",     event.nodeId(),
            "type",       event.type().name(),
            "detail",     event.detail(),
            "occurredAt", event.occurredAt().toString()
        );
    }

    private static Map<String, String> errorBody(String code, String message) {
        return Map.of("error", code, "message", message);
    }

    // ── env helpers ───────────────────────────────────────────────────────────

    private static int envInt(String key, int def) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) return def;
        try { return Integer.parseInt(v.trim()); } catch (NumberFormatException e) { return def; }
    }

    private static long envLong(String key, long def) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) return def;
        try { return Long.parseLong(v.trim()); } catch (NumberFormatException e) { return def; }
    }

    private static double envDouble(String key, double def) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) return def;
        try { return Double.parseDouble(v.trim()); } catch (NumberFormatException e) { return def; }
    }

    // ── DTO ───────────────────────────────────────────────────────────────────

    /**
     * JSON-deserializable counterpart to {@link NodeHealth}.
     *
     * <p>{@code NodeHealth} is a Java record with a compact constructor —
     * Jackson cannot bind constructor arguments by name without {@code
     * @JsonCreator}. Using a mutable JavaBean DTO keeps the domain record clean
     * and avoids pulling Jackson annotations into the health module.
     */
    public static final class NodeHealthDto {

        private String nodeId;
        private double vramPressure;
        private long   vramFreeBytes;
        private long   vramTotalBytes;
        private double temperatureCelsius   = -1.0;
        private double inferenceLatencyP99Ms = -1.0;
        private String sampledAt; // ISO-8601; null → now

        // Jackson needs a no-arg constructor + setters (or public fields).
        public NodeHealthDto() {}

        public String getNodeId()               { return nodeId; }
        public void   setNodeId(String v)       { this.nodeId = v; }

        public double getVramPressure()          { return vramPressure; }
        public void   setVramPressure(double v)  { this.vramPressure = v; }

        public long   getVramFreeBytes()         { return vramFreeBytes; }
        public void   setVramFreeBytes(long v)   { this.vramFreeBytes = v; }

        public long   getVramTotalBytes()        { return vramTotalBytes; }
        public void   setVramTotalBytes(long v)  { this.vramTotalBytes = v; }

        public double getTemperatureCelsius()            { return temperatureCelsius; }
        public void   setTemperatureCelsius(double v)    { this.temperatureCelsius = v; }

        public double getInferenceLatencyP99Ms()          { return inferenceLatencyP99Ms; }
        public void   setInferenceLatencyP99Ms(double v)  { this.inferenceLatencyP99Ms = v; }

        public String getSampledAt()             { return sampledAt; }
        public void   setSampledAt(String v)     { this.sampledAt = v; }

        // ── accessors for handler ──────────────────────────────────────────

        String nodeId()        { return nodeId; }
        double vramPressure()  { return vramPressure; }

        /** Convert to the canonical domain record. */
        NodeHealth toNodeHealth() {
            Instant ts = (sampledAt != null && !sampledAt.isBlank())
                ? Instant.parse(sampledAt)
                : Instant.now();
            return new NodeHealth(
                nodeId,
                vramPressure,
                vramFreeBytes,
                vramTotalBytes,
                temperatureCelsius,
                inferenceLatencyP99Ms,
                ts
            );
        }
    }
}