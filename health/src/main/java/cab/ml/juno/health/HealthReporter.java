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

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Periodically POSTs {@code NodeHealth} snapshots to the local health sidecar.
 *
 * <h2>Why this exists</h2>
 * The health sidecar is a passive HTTP server — it only knows about a node when
 * that node pushes a probe to {@code POST /health/probe}. Without an active
 * reporter the dashboard stays empty. This class fills that gap.
 *
 * <h2>In-process (local mode)</h2>
 * {@code ConsoleMain} creates one {@code HealthReporter} per virtual node and
 * calls {@link #startBackground}. Each reporter uses JVM heap occupancy as a
 * proxy for VRAM pressure (good enough when CUDA is not present).
 *
 * <h2>Forked cluster mode</h2>
 * {@code NodeMain} reads {@code -Djuno.health.url=http://host:port} and starts
 * a {@code HealthReporter} for the node's own ID. This way the forked process
 * pushes real per-JVM heap stats to the coordinator's sidecar.
 *
 * <h2>Thread model</h2>
 * A single daemon virtual-thread scheduled executor fires every
 * {@value #DEFAULT_INTERVAL_MS} ms. If the HTTP call fails the error is logged
 * at FINE level and the next cycle retries.
 */
public final class HealthReporter {

    private static final Logger log = Logger.getLogger(HealthReporter.class.getName());

    public static final long DEFAULT_INTERVAL_MS = 5_000L;

    private final String nodeId;
    private final String healthUrl;   // e.g. "http://localhost:8081/health/probe"
    private final long intervalMs;

    private final HttpClient http = HttpClient.newHttpClient();
    private ScheduledExecutorService scheduler;

    public HealthReporter(String nodeId, String healthBaseUrl) {
        this(nodeId, healthBaseUrl, DEFAULT_INTERVAL_MS);
    }

    public HealthReporter(String nodeId, String healthBaseUrl, long intervalMs) {
        this.nodeId      = nodeId;
        this.healthUrl   = healthBaseUrl.replaceAll("/+$", "") + "/health/probe";
        this.intervalMs  = intervalMs;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Start emitting probes in a background daemon thread.
     * Returns immediately; call {@link #stop()} to shut down.
     */
    public void startBackground() {
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = Thread.ofVirtual().name("juno-health-reporter[" + nodeId + "]").unstarted(r);
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::pushProbe, 0, intervalMs, TimeUnit.MILLISECONDS);
        log.info("HealthReporter started for node [" + nodeId + "] → " + healthUrl
                + "  interval=" + intervalMs + "ms");
    }

    /** Stop the background reporter. */
    public void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
        }
    }

    // ── Factory helpers ───────────────────────────────────────────────────────

    /**
     * Convenience: create and start a reporter for one node, using JVM heap as
     * the VRAM proxy. Returns immediately.
     *
     * @param nodeId       logical node ID (shown on the dashboard)
     * @param healthBase   base URL of the health sidecar, e.g. {@code "http://localhost:8081"}
     */
    public static HealthReporter startForNode(String nodeId, String healthBase) {
        HealthReporter r = new HealthReporter(nodeId, healthBase);
        r.startBackground();
        return r;
    }

    // ── Probe emission ────────────────────────────────────────────────────────

    private void pushProbe() {
        try {
            String body = buildProbeJson();
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(healthUrl))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<Void> resp = http.send(req, HttpResponse.BodyHandlers.discarding());
            if (resp.statusCode() >= 300) {
                log.fine("Health probe rejected: HTTP " + resp.statusCode());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            log.fine("Health probe failed (sidecar not ready?): " + e.getMessage());
        }
    }

    /**
     * Build a JSON body for {@code POST /health/probe} using JVM heap stats as
     * a VRAM proxy. Works even without a real GPU.
     *
     * <p>Mapping:
     * <ul>
     *   <li>{@code vramTotalBytes} = {@link Runtime#maxMemory()}
     *   <li>{@code vramFreeBytes}  = max − (total − free)  (= truly available heap)
     *   <li>{@code vramPressure}   = heap used / heap max  (0.0–1.0)
     * </ul>
     */
    private String buildProbeJson() {
        Runtime rt = Runtime.getRuntime();
        long maxMem   = rt.maxMemory();
        long totalMem = rt.totalMemory();
        long freeMem  = rt.freeMemory();
        long usedMem  = totalMem - freeMem;
        long trueFree = Math.max(0L, maxMem - usedMem);

        double pressure = maxMem > 0 ? (double) usedMem / maxMem : 0.0;
        // Clamp to [0.0, 1.0] in case of edge cases
        pressure = Math.min(1.0, Math.max(0.0, pressure));

        return "{"
                + "\"nodeId\":\"" + escape(nodeId) + "\","
                + "\"vramPressure\":" + String.format("%.6f", pressure) + ","
                + "\"vramFreeBytes\":"  + trueFree + ","
                + "\"vramTotalBytes\":" + maxMem + ","
                + "\"temperatureCelsius\":-1.0,"
                + "\"inferenceLatencyP99Ms\":-1.0,"
                + "\"sampledAt\":\"" + Instant.now() + "\""
                + "}";
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}