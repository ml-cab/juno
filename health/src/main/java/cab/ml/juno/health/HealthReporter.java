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
 *   <li><b>RAM pressure</b> — physical host memory ({@code usedPhysical / totalPhysical})
 *       via {@code OperatingSystemMXBean}. Matches what {@code free -m} and {@code top}
 *       report, and is consistent with what {@code juno-deploy.sh} displays. Exposed on
 *       the dashboard as "VRAM" for forward-compatibility with GPU builds where it will
 *       reflect actual device VRAM.
 *   <li><b>CPU load</b> — system-wide CPU utilisation via
 *       {@code OperatingSystemMXBean.getCpuLoad()}, 0.0–1.0. This is the whole-host
 *       average across all cores, equivalent to the {@code us+sy} columns in {@code top}.
 *       Available on all JVM platforms; no sysfs dependency.
 *   <li><b>Latency P99</b> — sliding window of the last 128 inference durations;
 *       callers record each generation via {@link #recordLatency(long)}.
 *       Meaningful only on the coordinator — nodes report throughput instead.
 *   <li><b>Throughput</b> — activation bytes sent per second, accumulated via
 *       {@link #recordBytes(long)} from the node's gRPC handler after each
 *       forward-pass response. Displayed on worker-node cards.
 * </ul>
 *
 * <h2>Usage (cluster mode)</h2>
 * Coordinator: constructed with {@code nodeRole="coordinator"}, wired to
 * {@code InferenceApiServer} via {@link #recordLatency(long)}.
 * Worker nodes: constructed with {@code nodeRole="node"}, wired to
 * {@code EmbeddedNodeServer} via {@link #recordBytes(long)}.
 */
public final class HealthReporter {

    private static final Logger log = Logger.getLogger(HealthReporter.class.getName());

    public static final long DEFAULT_INTERVAL_MS = 5_000L;

    /** Sliding window size for P99 latency. */
    private static final int LATENCY_WINDOW = 128;

    private final String nodeId;
    private final String nodeRole;    // "coordinator" | "node"
    private final String healthUrl;   // full probe URL, e.g. "http://localhost:8081/health/probe"
    private final long   intervalMs;

    private final HttpClient http = HttpClient.newHttpClient();
    private ScheduledExecutorService scheduler;

    // ── Latency sliding window (coordinator) ─────────────────────────────────
    private final AtomicLongArray latencyWindow = new AtomicLongArray(LATENCY_WINDOW);
    private final AtomicLong      latencyIdx    = new AtomicLong(0);
    private final AtomicLong      latencyCount  = new AtomicLong(0);

    // ── Throughput counter (nodes) ────────────────────────────────────────────
    private final AtomicLong throughputBytesAcc  = new AtomicLong(0);
    private volatile long    throughputWindowStart = System.currentTimeMillis();

    // ── Constructor ───────────────────────────────────────────────────────────

    /** Backward-compatible 2-arg constructor — defaults to {@code nodeRole="node"}. */
    public HealthReporter(String nodeId, String healthBaseUrl) {
        this(nodeId, "node", healthBaseUrl, DEFAULT_INTERVAL_MS);
    }

    /** Backward-compatible 3-arg constructor — defaults to {@code nodeRole="node"}. */
    public HealthReporter(String nodeId, String healthBaseUrl, long intervalMs) {
        this(nodeId, "node", healthBaseUrl, intervalMs);
    }

    /** Full constructor. {@code nodeRole} must be {@code "coordinator"} or {@code "node"}. */
    public HealthReporter(String nodeId, String nodeRole, String healthBaseUrl, long intervalMs) {
        this.nodeId     = nodeId;
        this.nodeRole   = (nodeRole == null || nodeRole.isBlank()) ? "node" : nodeRole;
        this.healthUrl  = healthBaseUrl.replaceAll("/+$", "") + "/health/probe";
        this.intervalMs = intervalMs;
    }
 
    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Record one inference duration (coordinator only). Thread-safe.
     *
     * @param durationMs elapsed wall-clock time in milliseconds
     */
    public void recordLatency(long durationMs) {
        long idx = latencyIdx.getAndIncrement() % LATENCY_WINDOW;
        latencyWindow.set((int) idx, durationMs);
        latencyCount.incrementAndGet();
    }

    /**
     * Accumulate bytes sent in one forward-pass response (worker nodes only).
     * Called from {@code EmbeddedNodeServer.forwardPass()} after each gRPC response.
     * Thread-safe — uses {@code AtomicLong}.
     *
     * @param n byte count of the encoded activation payload
     */
    public void recordBytes(long n) {
        throughputBytesAcc.addAndGet(n);
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
     * Create, start and return a reporter for a worker node.
     * Call {@link #recordBytes(long)} from the gRPC handler to populate throughput.
     */
    public static HealthReporter startForNode(String nodeId, String healthBase) {
        HealthReporter r = new HealthReporter(nodeId, "node", healthBase, DEFAULT_INTERVAL_MS);
        r.startBackground();
        return r;
    }

    /**
     * Create, start and return a coordinator reporter.
     * Call {@link #recordLatency(long)} after each inference to populate Latency P99.
     */
    public static HealthReporter startForCoordinator(String healthBase) {
        HealthReporter r = new HealthReporter("coordinator", "coordinator", healthBase, DEFAULT_INTERVAL_MS);
        r.startBackground();
        return r;
    }
 
    // ── Metric collection ─────────────────────────────────────────────────────

    private String buildProbeJson() {
        // Physical host RAM — matches `free -m` / `top`, consistent with juno-deploy.sh output.
        // Previously used JVM heap (rt.maxMemory / rt.totalMemory) which reported the
        // configured -Xmx ceiling (e.g. 12.9 GB) rather than real host RAM (e.g. 7.7 GB),
        // making the dashboard numbers incomparable to OS-level monitoring tools.
        var os      = (com.sun.management.OperatingSystemMXBean)
                      java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        long maxMem = os.getTotalMemorySize();

        // Use MemAvailable from /proc/meminfo rather than os.getFreeMemorySize() (MemFree).
        // MemFree is raw unallocated pages only — it ignores the buff/cache that the kernel
        // will reclaim on demand (typically 2–4 GB on an active node). This caused pressure
        // to read ~5× higher than reality (50% vs 10%) and would have fired false circuit-
        // breaker trips on nodes with large disk caches. MemAvailable is the kernel's own
        // estimate of what is truly allocatable without swapping, and matches what
        // `free -m`'s "available" column and `top`'s "avail Mem" show.
        long trueFree = readMemAvailableBytes(maxMem);
        long usedMem  = maxMem - trueFree;
        double pressure = maxMem > 0
                ? Math.min(1.0, Math.max(0.0, (double) usedMem / maxMem))
                : 0.0;

        double cpuLoad      = readCpuLoad();
        double p99          = computeP99();
        double throughput   = drainThroughput();

        return "{"
                + "\"nodeId\":\""              + escape(nodeId)                        + "\","
                + "\"nodeRole\":\""            + escape(nodeRole)                      + "\","
                + "\"vramPressure\":"          + String.format("%.6f", pressure)       + ","
                + "\"vramFreeBytes\":"         + trueFree                              + ","
                + "\"vramTotalBytes\":"        + maxMem                                + ","
                + "\"cpuLoad\":"              + String.format("%.4f", cpuLoad)         + ","
                + "\"inferenceLatencyP99Ms\":" + String.format("%.1f", p99)            + ","
                + "\"throughputBytesPerSec\":" + String.format("%.1f", throughput)     + ","
                + "\"sampledAt\":\""           + Instant.now()                         + "\""
                + "}";
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

    // ── CPU load ──────────────────────────────────────────────────────────────

    /**
     * Read system-wide CPU utilisation via {@code OperatingSystemMXBean.getCpuLoad()}.
     * Returns 0.0–1.0 (whole-host average across all cores), equivalent to the
     * {@code us+sy} percentage shown by {@code top}. Returns 0.0 instead of the
     * transient -1.0 the JVM emits before its first sample (~200 ms after startup).
     * Works on all platforms — no {@code /proc/stat} or sysfs dependency.
     *
     * <p><b>Note:</b> this is NOT per-process CPU ({@code getProcessCpuLoad()});
     * it reflects total host load including all other processes, which is what an
     * operator expects to see in the dashboard.
     */
    private static double readCpuLoad() {
        var os = (com.sun.management.OperatingSystemMXBean)
                 java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        double load = os.getCpuLoad();
        return load < 0 ? 0.0 : load;
    }

    // ── Memory available ─────────────────────────────────────────────────────────────

    /**
     * Return the kernel's {@code MemAvailable} figure from {@code /proc/meminfo} in bytes.
     *
     * <p>Why not {@code OperatingSystemMXBean.getFreeMemorySize()}?  That maps to Linux
     * {@code MemFree} — raw unallocated pages only.  On a typical JVM node, 2–4 GB of
     * buff/cache sits between {@code MemFree} and what the kernel would actually hand to
     * a new allocation, making pressure appear 3–5× higher than reality and risking false
     * circuit-breaker trips.
     *
     * <p>{@code MemAvailable} (added in Linux 3.14, available on all modern distributions)
     * is the kernel's own estimate of what can be allocated without swapping.  It accounts
     * for reclaimable cache and is identical to what {@code free -m}'s "available" column
     * and {@code top}'s "avail Mem" field report.
     *
     * <p>Falls back to {@code OperatingSystemMXBean.getFreeMemorySize()} on non-Linux
     * platforms (macOS, Windows) where {@code /proc/meminfo} does not exist.
     *
     * @param totalBytes total physical memory in bytes (used only for the fallback path)
     * @return estimated allocatable bytes
     */
    private static long readMemAvailableBytes(long totalBytes) {
        try (var br = new java.io.BufferedReader(new java.io.FileReader("/proc/meminfo"))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("MemAvailable:")) {
                    // Format: "MemAvailable:   6812340 kB"
                    String[] parts = line.split("\\s+");
                    return Long.parseLong(parts[1]) * 1024L;
                }
            }
        } catch (Exception ignored) {
            // /proc/meminfo not available (non-Linux); fall through to JVM API
        }
        // Fallback: MemFree from JVM — less accurate but universally available
        var os = (com.sun.management.OperatingSystemMXBean)
                 java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        return os.getFreeMemorySize();
    }

    // ── Throughput ────────────────────────────────────────────────────────────

    /**
     * Drain accumulated bytes and return bytes-per-second since last call.
     * Called once per probe interval, so interval = {@code intervalMs}.
     */
    private double drainThroughput() {
        long now     = System.currentTimeMillis();
        long bytes   = throughputBytesAcc.getAndSet(0);
        long elapsed = now - throughputWindowStart;
        throughputWindowStart = now;
        if (elapsed <= 0) return -1.0;
        return bytes * 1000.0 / elapsed; // bytes → bytes/s
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