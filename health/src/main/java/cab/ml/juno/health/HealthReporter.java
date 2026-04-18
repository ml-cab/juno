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

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.logging.Logger;

/**
 * Periodically POSTs {@code NodeHealth} snapshots to the local health sidecar.
 *
 * <h2>Metrics collected</h2>
 * <ul>
 *   <li><b>Heap pressure</b> — JVM {@code usedMemory / maxMemory} as VRAM proxy.
 *   <li><b>Temperature</b> — reads {@code /sys/class/thermal/thermal_zone/temp}
 *       on Linux, preferring package-level sensors. Returns -1 on other platforms.
 *   <li><b>Latency P99</b> — sliding window of the last 128 inference durations;
 *       callers record each generation via {@link #recordLatency(long)}.
 * </ul>
 *
 * <h2>Usage (local mode)</h2>
 * {@code ConsoleMain} starts one reporter per shard and calls
 * {@link #recordLatency(long)} from the REPL after every generation. The
 * returned P99 value shows up in the dashboard's "Latency P99" column.
 *
 * <h2>Usage (cluster mode)</h2>
 * {@code NodeMain} picks up {@code -Djuno.health.url=http://host:port} and
 * starts a reporter inside each forked JVM so real per-node heap stats flow
 * to the sidecar independently.
 */

public final class HealthReporter {
	 
    private static final Logger log = Logger.getLogger(HealthReporter.class.getName());
 
    public static final long DEFAULT_INTERVAL_MS = 5_000L;
 
    /** Sliding window size for P99 latency. */
    private static final int LATENCY_WINDOW = 128;
 
    private final String nodeId;
    private final String healthUrl;   // full probe URL, e.g. "http://localhost:8081/health/probe"
    private final long   intervalMs;
 
    private final HttpClient http = HttpClient.newHttpClient();
    private ScheduledExecutorService scheduler;
 
    // ── Latency sliding window ────────────────────────────────────────────────
    private final AtomicLongArray latencyWindow = new AtomicLongArray(LATENCY_WINDOW);
    private final AtomicLong      latencyIdx    = new AtomicLong(0);
    private final AtomicLong      latencyCount  = new AtomicLong(0);
 
    // ── Temperature cache ─────────────────────────────────────────────────────
    private volatile Path    thermalPath   = null;
    private volatile boolean thermalProbed = false;
 
    // ── Constructor ───────────────────────────────────────────────────────────
 
    public HealthReporter(String nodeId, String healthBaseUrl) {
        this(nodeId, healthBaseUrl, DEFAULT_INTERVAL_MS);
    }
 
    public HealthReporter(String nodeId, String healthBaseUrl, long intervalMs) {
        this.nodeId     = nodeId;
        this.healthUrl  = healthBaseUrl.replaceAll("/+$", "") + "/health/probe";
        this.intervalMs = intervalMs;
    }
 
    // ── Public API ────────────────────────────────────────────────────────────
 
    /**
     * Record one inference duration. Thread-safe; call from the REPL after each
     * completed generation to populate the Latency P99 dashboard column.
     *
     * @param durationMs elapsed wall-clock time in milliseconds
     */
    public void recordLatency(long durationMs) {
        long idx = latencyIdx.getAndIncrement() % LATENCY_WINDOW;
        latencyWindow.set((int) idx, durationMs);
        latencyCount.incrementAndGet();
    }
 
    /** Start emitting probes every {@code intervalMs} ms on a background daemon thread. */
    public void startBackground() {
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = Thread.ofVirtual()
                    .name("juno-health-reporter[" + nodeId + "]")
                    .unstarted(r);
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::pushProbe, 0, intervalMs, TimeUnit.MILLISECONDS);
        log.info("HealthReporter started for [" + nodeId + "] → " + healthUrl
                + "  interval=" + intervalMs + " ms");
    }
 
    /** Stop the reporter. Safe to call multiple times. */
    public void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
        }
    }
 
    /**
     * Create, start and return a reporter. The caller should retain the reference
     * to call {@link #recordLatency(long)} after each inference.
     */
    public static HealthReporter startForNode(String nodeId, String healthBase) {
        HealthReporter r = new HealthReporter(nodeId, healthBase);
        r.startBackground();
        return r;
    }
 
    // ── Probe push ────────────────────────────────────────────────────────────
 
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
 
    // ── Metric collection ─────────────────────────────────────────────────────
 
    private String buildProbeJson() {
        // Heap / VRAM proxy
        Runtime rt    = Runtime.getRuntime();
        long maxMem   = rt.maxMemory();
        long usedMem  = rt.totalMemory() - rt.freeMemory();
        long trueFree = Math.max(0L, maxMem - usedMem);
        double pressure = maxMem > 0
                ? Math.min(1.0, Math.max(0.0, (double) usedMem / maxMem))
                : 0.0;
 
        double tempC = readTemperatureCelsius();
        double p99   = computeP99();
 
        return "{"
                + "\"nodeId\":\""              + escape(nodeId)                      + "\","
                + "\"vramPressure\":"           + String.format("%.6f", pressure)    + ","
                + "\"vramFreeBytes\":"          + trueFree                           + ","
                + "\"vramTotalBytes\":"         + maxMem                             + ","
                + "\"temperatureCelsius\":"     + String.format("%.1f", tempC)       + ","
                + "\"inferenceLatencyP99Ms\":"  + String.format("%.1f", p99)         + ","
                + "\"sampledAt\":\""            + Instant.now()                      + "\""
                + "}";
    }
 
    // ── Temperature ───────────────────────────────────────────────────────────
 
    /**
     * Read CPU temperature from Linux sysfs.
     * Strategy (in order):
     * 1. {@code /sys/class/thermal/thermal_zone temp} — preferred, prefers x86_pkg zones.
     * 2. {@code /sys/class/hwmon/hwmon temp_input} — fallback, common on EC2 where
     *    thermal_zone is absent. Reads the first "input" file that looks like a CPU temp
     *    (millidegrees > 0).
     * Returns -1.0 if no readable sensor is found (expected on most VMs).
     */
    private double readTemperatureCelsius() {
        if (!thermalProbed) {
            thermalProbed = true;
            thermalPath   = findThermalZone();
            if (thermalPath == null) thermalPath = findHwmonTemp();
        }
        if (thermalPath == null) return -1.0;
        try {
            long milliC = Long.parseLong(Files.readString(thermalPath).trim());
            // hwmon values may be in millidegrees (>1000) or degrees (<200) depending on driver
            return milliC > 1000 ? milliC / 1000.0 : milliC;
        } catch (IOException | NumberFormatException e) {
            return -1.0;
        }
    }
 
    private static Path findThermalZone() {
        Path base = Path.of("/sys/class/thermal");
        if (!Files.isDirectory(base)) return null;
 
        Path fallback = null;
        try (var ds = Files.newDirectoryStream(base, "thermal_zone*")) {
            for (Path zone : ds) {
                Path tempFile = zone.resolve("temp");
                if (!Files.isReadable(tempFile)) continue;
                if (fallback == null) fallback = tempFile;
 
                Path typeFile = zone.resolve("type");
                if (Files.isReadable(typeFile)) {
                    String type = Files.readString(typeFile).trim().toLowerCase();
                    if (type.contains("x86_pkg") || type.contains("pkg_temp")
                            || type.contains("coretemp") || type.contains("cpu")) {
                        return tempFile;
                    }
                }
            }
        } catch (IOException e) {
            return null;
        }
        return fallback;
    }
 
    /**
     * Fallback: read temperature from {@code /sys/class/hwmon/hwmon temp*_input}.
     * Common on EC2 (especially bare-metal and Nitro instances) where
     * {@code /sys/class/thermal/} has no zones.
     */
    private static Path findHwmonTemp() {
        Path base = Path.of("/sys/class/hwmon");
        if (!Files.isDirectory(base)) return null;
        try (var ds = Files.newDirectoryStream(base)) {
            for (Path hwmon : ds) {
                // Prefer hwmon devices labelled "coretemp" or "k10temp"
                Path nameFile = hwmon.resolve("name");
                boolean preferred = false;
                if (Files.isReadable(nameFile)) {
                    String name = Files.readString(nameFile).trim().toLowerCase();
                    preferred = name.contains("coretemp") || name.contains("k10temp")
                             || name.contains("acpitz")   || name.contains("cpu");
                }
                // Look for temp*_input files
                try (var fs = Files.newDirectoryStream(hwmon, "temp*_input")) {
                    for (Path f : fs) {
                        if (!Files.isReadable(f)) continue;
                        String raw = Files.readString(f).trim();
                        long val = Long.parseLong(raw);
                        if (val > 0) return f;   // first readable non-zero sensor
                        if (!preferred) break;   // skip zeros unless it's a preferred device
                    }
                } catch (IOException ignored) {}
            }
        } catch (IOException e) {
            return null;
        }
        return null;
    }
 
    // ── Latency P99 ───────────────────────────────────────────────────────────
 
    private double computeP99() {
        long count = latencyCount.get();
        if (count == 0) return -1.0;
 
        int filled = (int) Math.min(count, LATENCY_WINDOW);
        long[] samples = new long[filled];
        for (int i = 0; i < filled; i++) {
            samples[i] = latencyWindow.get(i);
        }
        Arrays.sort(samples);
 
        int p99idx = Math.max(0, (int) Math.ceil(0.99 * filled) - 1);
        return samples[Math.min(p99idx, filled - 1)];
    }
 
    // ── Helpers ───────────────────────────────────────────────────────────────
 
    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}