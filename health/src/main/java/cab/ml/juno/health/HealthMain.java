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
 * Health-monitor HTTP server — runs as a sidecar alongside any Juno process
 * ({@code --health} option) or as a standalone fat-jar.
 *
 * <h2>API surface</h2>
 * <pre>
 *   GET  /                      HTML dashboard (auto-refreshing, dark-themed)
 *   POST /health/probe          Accept a NodeHealth snapshot from any node.
 *   GET  /health                Cluster overview JSON.
 *   GET  /health/nodes/{nodeId} Single-node detail (404 if unknown).
 *   GET  /health/circuits       Per-node circuit-breaker states (compact).
 * </pre>
 *
 * <h2>Embedding in another process</h2>
 * <pre>{@code
 *   // Start on port 8081 with default thresholds, in a background virtual thread.
 *   HealthMain.startBackground(8081, HealthThresholds.defaults());
 * }</pre>
 *
 * <h2>Configuration (CLI / env)</h2>
 * <pre>
 *   --port N          HTTP listen port (default: {@value #DEFAULT_PORT})
 *   --stale-ms N      ms before a node is considered stale (default: 15000)
 *   --warn F          VRAM warning threshold 0.0–1.0 (default: 0.90)
 *   --critical F      VRAM critical threshold 0.0–1.0 (default: 0.98)
 *
 *   Env equivalents: HEALTH_PORT  HEALTH_STALE_MS  HEALTH_WARN  HEALTH_CRITICAL
 * </pre>
 */
public final class HealthMain {

    private static final Logger log = Logger.getLogger(HealthMain.class.getName());

    public static final int DEFAULT_PORT = 8081;

    // ── per-node state ────────────────────────────────────────────────────────
    private final Map<String, NodeHealth>     nodeSnapshots = new ConcurrentHashMap<>();
    private final Map<String, CircuitBreaker> circuits      = new ConcurrentHashMap<>();

    private final HealthEvaluator evaluator;
    private final int port;

    // ── constructor ───────────────────────────────────────────────────────────

    public HealthMain(int port, HealthThresholds thresholds) {
        this.port      = port;
        this.evaluator = new HealthEvaluator(thresholds);
    }

    // ── entry points ──────────────────────────────────────────────────────────

    /**
     * Standalone CLI entry point. Parses args, builds thresholds, and starts the
     * server — blocking the calling thread until the process is killed.
     *
     * @param args command-line arguments (may contain {@code --health} which is
     *             already consumed by the caller before delegation here)
     */
    public static void main(String[] args) {
        int    port     = envInt   ("HEALTH_PORT",     DEFAULT_PORT);
        long   staleMs  = envLong  ("HEALTH_STALE_MS", 15_000L);
        double warn     = envDouble("HEALTH_WARN",     0.90);
        double critical = envDouble("HEALTH_CRITICAL", 0.98);

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port"      -> { if (i + 1 < args.length) port     = Integer.parseInt(args[++i]); }
                case "--stale-ms"  -> { if (i + 1 < args.length) staleMs  = Long.parseLong(args[++i]); }
                case "--warn"      -> { if (i + 1 < args.length) warn     = Double.parseDouble(args[++i]); }
                case "--critical"  -> { if (i + 1 < args.length) critical = Double.parseDouble(args[++i]); }
                case "--health"    -> { /* consumed by caller — safe to skip */ }
                default            -> log.fine("Ignoring unknown arg: " + args[i]);
            }
        }

        HealthThresholds thresholds = new HealthThresholds(warn, critical, staleMs);
        new HealthMain(port, thresholds).startBlocking();
    }

    /**
     * Start the health server in a background virtual thread and return
     * immediately. Intended for sidecar use inside NodeMain / CoordinatorMain /
     * ConsoleMain when {@code --health} is passed as an option.
     *
     * @param port       HTTP port to bind
     * @param thresholds VRAM / staleness thresholds
     */
    public static void startBackground(int port, HealthThresholds thresholds) {
        HealthMain server = new HealthMain(port, thresholds);
        Thread t = Thread.ofVirtual()
                .name("juno-health-server")
                .start(server::startBlocking);
        // Background — caller continues
        log.info("Health sidecar starting on :" + port + " (background)");
        // Ensure the thread doesn't keep the JVM alive on its own
        //t.setDaemon(false); // we want clean shutdown hooks to still fire
    }

    // ── server lifecycle ──────────────────────────────────────────────────────

    /**
     * Builds and starts the Javalin server. Blocks the calling thread until the
     * process is killed (i.e. {@link Thread#join()} returns).
     */
    public void startBlocking() {
        Javalin app = Javalin.create(config -> {
            config.showJavalinBanner = false;
            config.useVirtualThreads = true;
        });

        app.get ("/",                         this::handleDashboard);
        app.post("/health/probe",             this::handleProbe);
        app.get ("/health",                   this::handleClusterOverview);
        app.get ("/health/nodes/{nodeId}",    this::handleNodeDetail);
        app.get ("/health/circuits",          this::handleCircuits);

        app.start(port);
        log.info("Juno health server listening on :" + port);

        Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
            log.info("Health server shutting down…");
            app.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ── route handlers ────────────────────────────────────────────────────────

    /** {@code GET /} — serve the HTML dashboard. */
    private void handleDashboard(Context ctx) {
        ctx.contentType("text/html; charset=utf-8")
           .result(DASHBOARD_HTML);
    }

    /**
     * {@code POST /health/probe}
     * Accepts a {@link NodeHealthDto}, stores the snapshot, evaluates it and
     * drives the circuit breaker. Returns the list of events that fired.
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
        circuits.computeIfAbsent(probe.nodeId(), CircuitBreaker::forNode);

        List<HealthEvent> events = evaluator.evaluate(probe);
        for (HealthEvent event : events) {
            reactToEvent(event);
        }

        ctx.json(Map.of(
            "nodeId",   probe.nodeId(),
            "accepted", true,
            "events",   events.stream().map(this::eventToMap).toList()
        ));
    }

    /** {@code GET /health} — cluster overview JSON. */
    private void handleClusterOverview(Context ctx) {
        ctx.json(Map.of(
            "status",    deriveClusterStatus(),
            "nodeCount", nodeSnapshots.size(),
            "nodes",     buildNodeList()
        ));
    }

    /** {@code GET /health/nodes/{nodeId}} — single-node detail (404 if unknown). */
    private void handleNodeDetail(Context ctx) {
        String nodeId = ctx.pathParam("nodeId");
        NodeHealth snap = nodeSnapshots.get(nodeId);
        if (snap == null) {
            ctx.status(404).json(errorBody("NOT_FOUND", "No health data for node: " + nodeId));
            return;
        }
        ctx.json(buildNodeDetail(nodeId, snap));
    }

    /** {@code GET /health/circuits} — compact circuit states for all nodes. */
    private void handleCircuits(Context ctx) {
        List<Map<String, String>> result = circuits.entrySet().stream()
            .map(e -> Map.of(
                "nodeId", e.getKey(),
                "state",  e.getValue().state().name()
            ))
            .toList();
        ctx.json(Map.of("circuits", result));
    }

    // ── circuit reaction ──────────────────────────────────────────────────────

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
        for (var e : nodeSnapshots.entrySet()) {
            list.add(buildNodeDetail(e.getKey(), e.getValue()));
        }
        return list;
    }

    private Map<String, Object> buildNodeDetail(String nodeId, NodeHealth snap) {
        CircuitBreaker cb = circuits.computeIfAbsent(nodeId, CircuitBreaker::forNode);
        return Map.of(
            "nodeId",              nodeId,
            "circuit",             cb.state().name(),
            "callPermitted",       cb.isCallPermitted(),
            "vramPressure",        snap.vramPressure(),
            "vramFreeBytes",       snap.vramFreeBytes(),
            "vramTotalBytes",      snap.vramTotalBytes(),
            "temperatureCelsius",  snap.temperatureCelsius(),
            "inferenceLatencyP99", snap.inferenceLatencyP99Ms(),
            "ageMs",               snap.ageMillis(),
            "sampledAt",           snap.sampledAt().toString()
        );
    }

    private String deriveClusterStatus() {
        if (circuits.isEmpty()) return "HEALTHY";
        long open  = circuits.values().stream().filter(cb -> cb.state() == CircuitState.OPEN).count();
        long total = circuits.size();
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
     * <p>{@code NodeHealth} is a Java record — Jackson cannot bind constructor
     * arguments by name without {@code @JsonCreator}. This mutable JavaBean DTO
     * keeps the domain record annotation-free.
     */
    public static final class NodeHealthDto {

        private String nodeId;
        private double vramPressure;
        private long   vramFreeBytes;
        private long   vramTotalBytes;
        private double temperatureCelsius    = -1.0;
        private double inferenceLatencyP99Ms = -1.0;
        private String sampledAt; // ISO-8601; null → now

        public NodeHealthDto() {}

        public String getNodeId()                        { return nodeId; }
        public void   setNodeId(String v)                { this.nodeId = v; }
        public double getVramPressure()                  { return vramPressure; }
        public void   setVramPressure(double v)          { this.vramPressure = v; }
        public long   getVramFreeBytes()                 { return vramFreeBytes; }
        public void   setVramFreeBytes(long v)           { this.vramFreeBytes = v; }
        public long   getVramTotalBytes()                { return vramTotalBytes; }
        public void   setVramTotalBytes(long v)          { this.vramTotalBytes = v; }
        public double getTemperatureCelsius()            { return temperatureCelsius; }
        public void   setTemperatureCelsius(double v)    { this.temperatureCelsius = v; }
        public double getInferenceLatencyP99Ms()         { return inferenceLatencyP99Ms; }
        public void   setInferenceLatencyP99Ms(double v) { this.inferenceLatencyP99Ms = v; }
        public String getSampledAt()                     { return sampledAt; }
        public void   setSampledAt(String v)             { this.sampledAt = v; }

        // package-private for handleProbe
        String nodeId()       { return nodeId; }
        double vramPressure() { return vramPressure; }

        NodeHealth toNodeHealth() {
            Instant ts = (sampledAt != null && !sampledAt.isBlank())
                ? Instant.parse(sampledAt)
                : Instant.now();
            return new NodeHealth(
                nodeId, vramPressure, vramFreeBytes, vramTotalBytes,
                temperatureCelsius, inferenceLatencyP99Ms, ts);
        }
    }

    // ── Dashboard HTML ────────────────────────────────────────────────────────

    private static final String DASHBOARD_HTML = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Juno Health</title>
          <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            :root {
              --bg:      #0d1117;
              --surface: #161b22;
              --border:  #30363d;
              --text:    #c9d1d9;
              --muted:   #8b949e;
              --green:   #3fb950;
              --yellow:  #d29922;
              --red:     #f85149;
              --blue:    #58a6ff;
              --accent:  #1f6feb;
            }
            body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
            header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
            header h1 { font-size: 1.25rem; font-weight: 600; letter-spacing: .02em; }
            header span { font-size: .75rem; color: var(--muted); }
            .badge { display: inline-flex; align-items: center; gap: .4rem; padding: .25rem .75rem; border-radius: 2rem; font-size: .75rem; font-weight: 600; letter-spacing: .05em; }
            .badge.healthy  { background: #0d2a16; color: var(--green);  border: 1px solid var(--green); }
            .badge.degraded { background: #2a1f05; color: var(--yellow); border: 1px solid var(--yellow); }
            .badge.down     { background: #2a0b0b; color: var(--red);    border: 1px solid var(--red); }
            .badge.unknown  { background: #1a1f27; color: var(--muted);  border: 1px solid var(--border); }
            .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
            main { padding: 2rem; max-width: 1200px; margin: 0 auto; }
            .summary { display: flex; align-items: center; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }
            .summary .stat { background: var(--surface); border: 1px solid var(--border); border-radius: .5rem; padding: .75rem 1.25rem; }
            .summary .stat .label { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: .25rem; }
            .summary .stat .value { font-size: 1.5rem; font-weight: 700; }
            .nodes { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
            .node-card { background: var(--surface); border: 1px solid var(--border); border-radius: .75rem; padding: 1.25rem; transition: border-color .15s; }
            .node-card:hover { border-color: var(--accent); }
            .node-card.circuit-open { border-color: var(--red); }
            .node-card.circuit-half_open { border-color: var(--yellow); }
            .node-card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; }
            .node-id { font-weight: 600; font-size: .95rem; }
            .circuit-badge { font-size: .65rem; font-weight: 700; padding: .2rem .5rem; border-radius: .25rem; letter-spacing: .06em; }
            .circuit-CLOSED    { background: #0d2a16; color: var(--green); }
            .circuit-OPEN      { background: #2a0b0b; color: var(--red); }
            .circuit-HALF_OPEN { background: #2a1f05; color: var(--yellow); }
            .metric-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: .6rem; font-size: .83rem; }
            .metric-label { color: var(--muted); }
            .metric-value { font-weight: 500; font-variant-numeric: tabular-nums; }
            .vram-bar-wrap { margin-top: .75rem; }
            .vram-bar-label { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); margin-bottom: .3rem; }
            .vram-bar-bg { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
            .vram-bar-fill { height: 100%; border-radius: 4px; transition: width .4s ease; }
            .bar-ok   { background: var(--green); }
            .bar-warn { background: var(--yellow); }
            .bar-crit { background: var(--red); }
            .empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); }
            .empty-state h2 { font-size: 1.1rem; margin-bottom: .5rem; }
            .empty-state code { background: var(--surface); border: 1px solid var(--border); border-radius: .25rem; padding: .15rem .4rem; font-size: .8rem; }
            .refresh-ticker { margin-left: auto; font-size: .72rem; color: var(--muted); display: flex; align-items: center; gap: .4rem; }
            footer { text-align: center; padding: 2rem; font-size: .75rem; color: var(--muted); border-top: 1px solid var(--border); margin-top: 2rem; }
            footer a { color: var(--blue); text-decoration: none; }
            footer a:hover { text-decoration: underline; }
          </style>
        </head>
        <body>
          <header>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
              <rect x="2" y="2" width="20" height="20" rx="4" fill="#1f6feb" opacity=".15"/>
              <path d="M7 17l3-6 2 4 2-7 3 9" stroke="#58a6ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h1>Juno Health</h1>
            <div id="clusterBadge" class="badge unknown"><span class="dot"></span>LOADING</div>
            <div class="refresh-ticker"><span id="countdown">5</span>s</div>
          </header>

          <main>
            <div class="summary">
              <div class="stat"><div class="label">Nodes</div><div class="value" id="nodeCount">—</div></div>
              <div class="stat"><div class="label">Open Circuits</div><div class="value" id="openCount" style="color:var(--red)">—</div></div>
              <div class="stat"><div class="label">Avg VRAM</div><div class="value" id="avgVram">—</div></div>
            </div>
            <div id="nodeGrid" class="nodes"></div>
          </main>

          <footer>
            <a href="/health">/health</a> &nbsp;·&nbsp;
            <a href="/health/circuits">/health/circuits</a> &nbsp;·&nbsp;
            POST <a href="#" onclick="return false">/health/probe</a>
            &nbsp;—&nbsp; Juno Health Monitor
          </footer>

          <script>
            let countdown = 5;

            function statusClass(s) {
              return s === 'HEALTHY' ? 'healthy' : s === 'DEGRADED' ? 'degraded' : s === 'DOWN' ? 'down' : 'unknown';
            }

            function circuitClass(s) {
              return s === 'CLOSED' ? 'circuit-CLOSED' : s === 'OPEN' ? 'circuit-OPEN' : 'circuit-HALF_OPEN';
            }

            function barClass(p) {
              return p >= 0.98 ? 'bar-crit' : p >= 0.90 ? 'bar-warn' : 'bar-ok';
            }

            function fmtBytes(b) {
              if (b <= 0) return '—';
              if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
              return (b / 1e6).toFixed(0) + ' MB';
            }

            function fmtAge(ms) {
              if (ms < 2000) return ms + ' ms';
              if (ms < 60000) return (ms / 1000).toFixed(1) + ' s';
              return (ms / 60000).toFixed(1) + ' m';
            }

            async function refresh() {
              try {
                const r = await fetch('/health');
                if (!r.ok) throw new Error('HTTP ' + r.status);
                const data = await r.json();

                // Summary
                const badge = document.getElementById('clusterBadge');
                badge.className = 'badge ' + statusClass(data.status);
                badge.innerHTML = '<span class="dot"></span>' + (data.status || 'UNKNOWN');

                document.getElementById('nodeCount').textContent = data.nodeCount ?? 0;

                const nodes = data.nodes || [];
                const openCircuits = nodes.filter(n => n.circuit === 'OPEN').length;
                document.getElementById('openCount').textContent = openCircuits;

                const totalVram = nodes.reduce((s, n) => s + (n.vramPressure || 0), 0);
                const avgVram = nodes.length ? (totalVram / nodes.length * 100).toFixed(1) + '%' : '—';
                document.getElementById('avgVram').textContent = avgVram;

                // Node cards
                const grid = document.getElementById('nodeGrid');
                if (nodes.length === 0) {
                  grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h2>No nodes yet</h2><p>Push a probe: <code>POST /health/probe</code></p></div>';
                } else {
                  grid.innerHTML = nodes.map(n => {
                    const pct = ((n.vramPressure || 0) * 100).toFixed(1);
                    const barW = Math.min(100, pct);
                    return `
                      <div class="node-card circuit-${n.circuit}">
                        <div class="node-card-header">
                          <div class="node-id">${n.nodeId}</div>
                          <span class="circuit-badge ${circuitClass(n.circuit)}">${n.circuit}</span>
                        </div>
                        <div class="metric-row">
                          <span class="metric-label">VRAM free</span>
                          <span class="metric-value">${fmtBytes(n.vramFreeBytes)} / ${fmtBytes(n.vramTotalBytes)}</span>
                        </div>
                        <div class="metric-row">
                          <span class="metric-label">Temperature</span>
                          <span class="metric-value">${n.temperatureCelsius >= 0 ? n.temperatureCelsius.toFixed(1) + ' °C' : '—'}</span>
                        </div>
                        <div class="metric-row">
                          <span class="metric-label">Latency P99</span>
                          <span class="metric-value">${n.inferenceLatencyP99 >= 0 ? n.inferenceLatencyP99.toFixed(1) + ' ms' : '—'}</span>
                        </div>
                        <div class="metric-row">
                          <span class="metric-label">Last seen</span>
                          <span class="metric-value" style="color:${n.ageMs > 10000 ? 'var(--red)' : 'var(--muted)'}">${fmtAge(n.ageMs)}</span>
                        </div>
                        <div class="vram-bar-wrap">
                          <div class="vram-bar-label"><span>VRAM pressure</span><span>${pct}%</span></div>
                          <div class="vram-bar-bg">
                            <div class="vram-bar-fill ${barClass(n.vramPressure)}" style="width:${barW}%"></div>
                          </div>
                        </div>
                      </div>`;
                  }).join('');
                }
              } catch (e) {
                const badge = document.getElementById('clusterBadge');
                badge.className = 'badge unknown';
                badge.innerHTML = '<span class="dot"></span>UNREACHABLE';
              }
            }

            refresh();
            setInterval(() => {
              countdown--;
              if (countdown <= 0) { countdown = 5; refresh(); }
              document.getElementById('countdown').textContent = countdown;
            }, 1000);
          </script>
        </body>
        </html>
        """;
}