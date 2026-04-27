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

package cab.ml.juno.coordinator;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import io.javalin.Javalin;
import io.javalin.http.Context;

/**
 * Javalin-based REST API server for the juno coordinator.
 *
 * Implements the OpenAPI spec at api/src/main/resources/openapi.yaml.
 *
 * Routes: POST /v1/inference — blocking, returns full InferenceResponse POST
 * /v1/inference/stream — SSE, streams TokenEvent per generated token GET
 * /v1/models — list all registered models GET /v1/models/{modelId} — get model
 * by ID DELETE /v1/models/{modelId} — unload model GET /v1/cluster/health —
 * cluster health overview
 *
 * Thread model: Javalin uses Virtual Threads (configured via
 * VirtualThreadPool). Both blocking and SSE handlers call scheduler.submit()
 * which dispatches generation on its own virtual thread. The SSE handler joins
 * the future, blocking its virtual thread (cheap — no OS thread pinned).
 *
 * Error handling: 400 — bad request (missing/empty messages) 404 — model not
 * found 429 — scheduler queue full 503 — model not loaded / cluster unavailable
 * 500 — unexpected inference error
 */
public final class InferenceApiServer {

	private static final Logger log = Logger.getLogger(InferenceApiServer.class.getName());

	private final RequestScheduler scheduler;
	private final ModelRegistry modelRegistry;
	private final OpenAiChatHandler openAiChatHandler;
	private Javalin app;
	private String byteOrder;

	/**
	 * Optional health sidecar URL (e.g. "http://localhost:8081").
	 * When set, /health/probe and /health-data are proxied to the sidecar so
	 * the health dashboard can be served from the coordinator's own port.
	 */
	private String healthSidecarUrl;

	/**
	 * Optional reporter that receives per-request inference latency so the
	 * coordinator node shows Latency P99 on the health dashboard.
	 */
	private cab.ml.juno.health.HealthReporter latencyReporter;

	public InferenceApiServer(RequestScheduler scheduler, ModelRegistry modelRegistry, String byteOrder) {
		this.byteOrder = byteOrder != null ? byteOrder : "BE";
		if (scheduler == null)
			throw new IllegalArgumentException("scheduler must not be null");
		if (modelRegistry == null)
			throw new IllegalArgumentException("modelRegistry must not be null");
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
		this.openAiChatHandler = new OpenAiChatHandler(scheduler, modelRegistry, ms -> {
			if (latencyReporter != null)
				latencyReporter.recordLatency(ms);
		});
	}

	public void start(int port) {
		app = Javalin.create(config -> {
			config.useVirtualThreads = true;
			config.showJavalinBanner = false;
		});

		// ── Web console ───────────────────────────────────────────────────────
		app.get("/", this::handleConsole);
		app.get("/health-ui", this::handleHealthDashboard);

		// ── Health probe proxy — nodes POST here, dashboard polls GET /health-data ──
		// The coordinator proxies probe storage to the sidecar when running on AWS.
		// In standalone mode the sidecar runs on a separate port; /health-ui fetches
		// from window.location.origin so routes live on the same :8080 host.
		app.post("/health/probe",    this::handleHealthProbeProxy);
		app.get ("/health-data",     this::handleHealthDataProxy);

		// ── Inference ─────────────────────────────────────────────────────────
		app.post("/v1/inference", this::handleBlockingInference);
		app.post("/v1/inference/stream", this::handleStreamingInference);
		app.post("/v1/chat/completions", openAiChatHandler::handleChatCompletion);

		// ── Models ────────────────────────────────────────────────────────────
		app.get("/v1/models", openAiChatHandler::handleListModels);
		app.get("/v1/models/{modelId}", openAiChatHandler::handleGetModel);
		app.delete("/v1/models/{modelId}", this::handleUnloadModel);

		// ── Cluster ───────────────────────────────────────────────────────────
		app.get("/v1/cluster/health", this::handleClusterHealth);

		// ── Error handlers ────────────────────────────────────────────────────
		app.exception(RequestScheduler.QueueFullException.class, (e, ctx) -> {
			ctx.status(429).json(Map.of("code", 429, "error", "QUEUE_FULL", "message", e.getMessage(), "retryAfterMs",
					e.retryAfterSeconds() * 1000));
		});
		app.exception(Exception.class, (e, ctx) -> {
			log.warning("Unhandled exception: " + e.getMessage());
			ctx.status(500).json(Map.of("code", 500, "error", "INTERNAL_ERROR", "message",
					e.getMessage() != null ? e.getMessage() : "Unexpected error"));
		});

		app.start(port);
		log.info("InferenceApiServer started on port " + port);
	}

	public void stop() {
		if (app != null) {
			app.stop();
			log.info("InferenceApiServer stopped");
		}
	}

	/**
	 * Optionally wire a health sidecar so the coordinator proxies node health
	 * probes and serves the dashboard at {@code GET /health-ui}.
	 *
	 * @param baseUrl base URL of the sidecar, e.g. {@code "http://localhost:8081"}
	 */
	public void setHealthSidecarUrl(String baseUrl) {
		this.healthSidecarUrl = baseUrl != null ? baseUrl.replaceAll("/+$", "") : null;
	}

	/**
	 * Wire a {@link cab.ml.juno.health.HealthReporter} that will have
	 * {@code recordLatency(ms)} called after every completed inference request.
	 * This populates the Latency P99 column in the health dashboard for the
	 * coordinator node.
	 *
	 * @param reporter an already-started reporter (caller owns lifecycle)
	 */
	public void setLatencyReporter(cab.ml.juno.health.HealthReporter reporter) {
		this.latencyReporter = reporter;
	}

	// ── Web console ───────────────────────────────────────────────────────────

	private void handleConsole(Context ctx) {
		ctx.contentType("text/html; charset=UTF-8").result(CONSOLE_HTML);
	}

	private void handleHealthDashboard(Context ctx) {
		ctx.contentType("text/html; charset=UTF-8").result(HEALTH_DASHBOARD_HTML);
	}

	/**
	 * Proxy {@code POST /health/probe} → sidecar {@code POST /health/probe}.
	 * Nodes deployed by juno-deploy.sh point {@code JUNO_HEALTH_URL} at the
	 * coordinator (port 8080) so all health data flows through one endpoint.
	 */
	private void handleHealthProbeProxy(Context ctx) {
		if (healthSidecarUrl == null) {
			ctx.status(503).json(java.util.Map.of("error", "health sidecar not configured"));
			return;
		}
		try {
			java.net.http.HttpClient http = java.net.http.HttpClient.newHttpClient();
			java.net.http.HttpRequest req = java.net.http.HttpRequest.newBuilder()
					.uri(java.net.URI.create(healthSidecarUrl + "/health/probe"))
					.header("Content-Type", "application/json")
					.POST(java.net.http.HttpRequest.BodyPublishers.ofString(ctx.body()))
					.build();
			java.net.http.HttpResponse<String> resp =
					http.send(req, java.net.http.HttpResponse.BodyHandlers.ofString());
			ctx.status(resp.statusCode()).contentType("application/json").result(resp.body());
		} catch (Exception e) {
			ctx.status(502).json(java.util.Map.of("error", "sidecar unreachable: " + e.getMessage()));
		}
	}

	/**
	 * Proxy {@code GET /health-data} → sidecar {@code GET /health}.
	 * The health dashboard JavaScript fetches from the same origin (/health-data)
	 * so CORS is never an issue regardless of which port the user connects to.
	 */
	private void handleHealthDataProxy(Context ctx) {
		if (healthSidecarUrl == null) {
			ctx.status(200).contentType("application/json")
			   .result("{\"status\":\"HEALTHY\",\"nodeCount\":0,\"nodes\":[]}");
			return;
		}
		try {
			java.net.http.HttpClient http = java.net.http.HttpClient.newHttpClient();
			java.net.http.HttpRequest req = java.net.http.HttpRequest.newBuilder()
					.uri(java.net.URI.create(healthSidecarUrl + "/health"))
					.GET().build();
			java.net.http.HttpResponse<String> resp =
					http.send(req, java.net.http.HttpResponse.BodyHandlers.ofString());
			ctx.status(resp.statusCode()).contentType("application/json").result(resp.body());
		} catch (Exception e) {
			ctx.status(200).contentType("application/json")
			   .result("{\"status\":\"UNREACHABLE\",\"nodeCount\":0,\"nodes\":[]}");
		}
	}

	/**
	 * Self-contained HTML5 chat console.
	 *
	 * Served at GET / — no external dependencies. Polls /v1/cluster/health every
	 * 10 s; fetches /v1/models on load. Streams tokens via POST /v1/inference/stream
	 * using the Fetch ReadableStream API to consume the SSE response body.
	 *
	 * SSE event shape (from SseTokenConsumer):
	 *   data: {"requestId":"...","token":" hi","tokenId":1234,"isComplete":false}
	 *   data: {"requestId":"...","token":"","tokenId":0,"isComplete":true,"finishReason":"stop"}
	 */
	private static final String CONSOLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Juno Console</title>
<style>
:root{
  --bg:#ffffff;--surface:#f8fafc;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;--accent:#6366f1;
  --user-bg:#3b82f6;--user-text:#fff;
  --bot-bg:#f1f5f9;--bot-text:#0f172a;
  --ok:#22c55e;--warn:#f59e0b;--err:#ef4444;
  --radius:12px;--hh:52px;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--surface:#161b22;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;
  --bot-bg:#21262d;--bot-text:#e6edf3;
}}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;font-size:14px;
  background:var(--bg);color:var(--text);
  display:flex;flex-direction:column;height:100dvh;overflow:hidden}

/* ── header ── */
header{
  height:var(--hh);min-height:var(--hh);
  display:flex;align-items:center;gap:10px;padding:0 16px;
  background:var(--surface);border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.logo{font-family:monospace;font-weight:700;font-size:18px;
  color:var(--accent);letter-spacing:-.5px;margin-right:4px}
#model-select{
  border:1px solid var(--border);border-radius:6px;
  background:var(--bg);color:var(--text);
  padding:4px 28px 4px 8px;font-size:13px;cursor:pointer;
  appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24'%3E%3Cpath fill='%238b949e' d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 6px center;
}
.health-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0}
.health-dot.ok{background:var(--ok)}.health-dot.err{background:var(--err)}
#health-text{font-size:12px;color:var(--muted)}
.spacer{flex:1}
#queue-badge{
  font-size:11px;color:var(--muted);padding:2px 8px;
  border:1px solid var(--border);border-radius:20px;white-space:nowrap
}
#nodes-badge{
  font-size:11px;color:var(--muted);padding:2px 8px;
  border:1px solid var(--border);border-radius:20px;white-space:nowrap
}

/* ── chat area ── */
#chat{
  flex:1;overflow-y:auto;padding:20px 16px;
  display:flex;flex-direction:column;gap:16px;
}
.msg-row{display:flex;max-width:720px;margin:0 auto;width:100%}
.msg-row.user{justify-content:flex-end}
.msg-row.bot{justify-content:flex-start}
.bubble{
  padding:10px 14px;border-radius:var(--radius);
  line-height:1.55;white-space:pre-wrap;word-break:break-word;
  max-width:82%;font-size:14px;
}
.bubble.user{background:var(--user-bg);color:var(--user-text);border-bottom-right-radius:4px}
.bubble.bot{background:var(--bot-bg);color:var(--bot-text);border-bottom-left-radius:4px}
.bubble.streaming::after{
  content:'▌';animation:blink .6s step-end infinite;display:inline
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.meta{font-size:11px;color:var(--muted);margin-top:4px;text-align:right}
.meta.bot{text-align:left}
.empty-state{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;color:var(--muted);gap:8px;
}
.empty-state .icon{font-size:40px;opacity:.4}
.empty-state p{font-size:13px}

/* ── footer ── */
footer{
  background:var(--surface);border-top:1px solid var(--border);
  padding:10px 16px 14px;display:flex;flex-direction:column;gap:8px;
  flex-shrink:0;
}

/* params bar */
#params-bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.param-group{display:flex;align-items:center;gap:5px}
.param-group label{font-size:11px;color:var(--muted);white-space:nowrap}
.param-group input[type=range]{width:80px;accent-color:var(--accent)}
.param-group input[type=number]{
  width:60px;padding:2px 5px;border:1px solid var(--border);
  border-radius:5px;background:var(--bg);color:var(--text);font-size:12px;
}
#val-temp,#val-top-p{font-size:11px;color:var(--accent);min-width:24px}
.param-sep{width:1px;height:16px;background:var(--border)}

/* input row */
#input-row{display:flex;gap:8px;align-items:flex-end}
#prompt{
  flex:1;resize:none;border:1px solid var(--border);border-radius:10px;
  padding:9px 12px;font-family:inherit;font-size:14px;line-height:1.45;
  background:var(--bg);color:var(--text);outline:none;
  max-height:160px;overflow-y:auto;
  transition:border-color .15s;
}
#prompt:focus{border-color:var(--accent)}
#send-btn{
  padding:9px 18px;border-radius:10px;border:none;cursor:pointer;
  background:var(--accent);color:#fff;font-size:14px;font-weight:500;
  white-space:nowrap;flex-shrink:0;
  transition:opacity .15s,background .15s;
}
#send-btn:disabled{opacity:.45;cursor:default}
#send-btn.cancel{background:var(--err)}
</style>
</head>
<body>

<header>
  <span class="logo">juno</span>
  <select id="model-select"><option value="">loading models…</option></select>
  <div class="health-dot" id="hdot"></div>
  <span id="health-text">–</span>
  <div class="spacer"></div>
  <span id="queue-badge">queue –/–</span>
  <span id="nodes-badge">models –</span>
  <span id="byte-order-badge" style="font-size:11px;color:var(--muted);padding:2px 8px;border:1px solid var(--border);border-radius:20px;\">order –</span>
</header>

<div id="chat">
  <div class="empty-state" id="empty">
    <div class="icon">◈</div>
    <p>Type a message to start a conversation</p>
  </div>
</div>

<footer>
  <div id="params-bar">
    <div class="param-group">
      <label>temp</label>
      <input type="range" id="p-temp" min="0" max="2" step="0.05" value="0.7"
             oninput="document.getElementById('val-temp').textContent=this.value">
      <span id="val-temp">0.7</span>
    </div>
    <div class="param-sep"></div>
    <div class="param-group">
      <label>top-p</label>
      <input type="range" id="p-topp" min="0" max="1" step="0.05" value="0.95"
             oninput="document.getElementById('val-top-p').textContent=this.value">
      <span id="val-top-p">0.95</span>
    </div>
    <div class="param-sep"></div>
    <div class="param-group">
      <label>top-k</label>
      <input type="number" id="p-topk" value="20" min="0" max="200" style="width:52px">
    </div>
    <div class="param-sep"></div>
    <div class="param-group">
      <label>max tokens</label>
      <input type="number" id="p-maxt" value="512" min="1" max="4096" style="width:62px">
    </div>
    <div class="param-sep"></div>
    <div class="param-group">
      <label>priority</label>
      <select id="p-prio" style="font-size:12px;padding:2px 5px;border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--text)">
        <option>NORMAL</option><option>HIGH</option><option>LOW</option>
      </select>
    </div>
  </div>
  <div id="input-row">
    <textarea id="prompt" rows="1" placeholder="Send a message…"></textarea>
    <button id="send-btn" onclick="onSend()">Send</button>
  </div>
</footer>

<script>
const chat    = document.getElementById('chat');
const empty   = document.getElementById('empty');
const prompt  = document.getElementById('prompt');
const sendBtn = document.getElementById('send-btn');
const hdot    = document.getElementById('hdot');
const htext   = document.getElementById('health-text');
const qBadge  = document.getElementById('queue-badge');
const nBadge  = document.getElementById('nodes-badge');
const mSelect = document.getElementById('model-select');

let history   = [];   // [{role,content}]
let abortCtrl = null;
let generating = false;
let tokenCount = 0;
let genStart   = 0;

// ── auto-resize textarea ──────────────────────────────────────────────────────
prompt.addEventListener('input', () => {
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 160) + 'px';
});
prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
});

// ── health polling ────────────────────────────────────────────────────────────
async function fetchHealth() {
  try {
    const r = await fetch('/v1/cluster/health');
    if (!r.ok) throw new Error();
    const d = await r.json();
    const ok = d.status === 'HEALTHY';
    hdot.className = 'health-dot ' + (ok ? 'ok' : 'err');
    htext.textContent = d.status;
    htext.style.color = ok ? 'var(--ok)' : 'var(--err)';
    qBadge.textContent = 'queue ' + d.queueDepth + '/' + d.maxQueue;
    nBadge.textContent = 'models ' + d.loadedModels;
    document.getElementById('byte-order-badge').textContent = 'byteOrder ' + (d.byteOrder || '–');
  } catch {
    hdot.className = 'health-dot err';
    htext.textContent = 'UNREACHABLE';
    htext.style.color = 'var(--err)';
  }
}

// ── model list ────────────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    const r = await fetch('/v1/models');
    if (!r.ok) return;
    const d = await r.json();
    const loaded = (d.models || []).filter(m => m.status === 'LOADED');
    mSelect.innerHTML = '';
    if (loaded.length === 0) {
      mSelect.innerHTML = '<option value="">no models loaded</option>';
      return;
    }
    for (const m of loaded) {
      const o = document.createElement('option');
      o.value = m.modelId;
      o.textContent = m.modelId + '  (' + m.architecture + '  ' + m.quantization
                    + '  layers=' + m.totalLayers + ')';
      mSelect.appendChild(o);
    }
  } catch {
    mSelect.innerHTML = '<option value="">error loading models</option>';
  }
}

// ── chat helpers ──────────────────────────────────────────────────────────────
function addBubble(role, text) {
  empty.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const bub = document.createElement('div');
  bub.className = 'bubble ' + role;
  bub.textContent = text;
  row.appendChild(bub);
  chat.appendChild(row);
  return bub;
}

function scrollBottom() {
  chat.scrollTop = chat.scrollHeight;
}

// ── send ──────────────────────────────────────────────────────────────────────
function onSend() {
  if (generating) { cancelGen(); return; }
  const text = prompt.value.trim();
  if (!text) return;
  prompt.value = '';
  prompt.style.height = 'auto';
  history.push({role:'user', content:text});
  addBubble('user', text);
  scrollBottom();
  startGen();
}

async function startGen() {
  generating = true;
  tokenCount = 0;
  genStart = Date.now();
  sendBtn.textContent = 'Stop';
  sendBtn.className = 'cancel';

  const botBub = addBubble('bot', '');
  botBub.classList.add('streaming');
  scrollBottom();

  const body = {
    messages: history.slice(),
    modelId: mSelect.value || undefined,
    sampling: {
      temperature: parseFloat(document.getElementById('p-temp').value),
      topK:        parseInt(document.getElementById('p-topk').value)   || 20,
      topP:        parseFloat(document.getElementById('p-topp').value),
      maxTokens:   parseInt(document.getElementById('p-maxt').value)   || 512,
      priority:    document.getElementById('p-prio').value,
    }
  };

  abortCtrl = new AbortController();
  let fullText = '';

  try {
    const resp = await fetch('/v1/inference/stream', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
      signal: abortCtrl.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({message:'HTTP ' + resp.status}));
      botBub.textContent = '[error: ' + (err.message || resp.statusText) + ']';
      botBub.style.color = 'var(--err)';
      finishGen(botBub, fullText, 'error');
      return;
    }

    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf      = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split('\\n\\n');
      buf = parts.pop();                // keep the trailing incomplete chunk
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }
        if (evt.isComplete) {
          finishGen(botBub, fullText, evt.finishReason);
          return;
        }
        fullText += evt.token;
        botBub.textContent = fullText;
        tokenCount++;
        scrollBottom();
      }
    }
    // stream ended without isComplete event
    finishGen(botBub, fullText, 'stop');

  } catch (e) {
    if (e.name !== 'AbortError') {
      botBub.textContent = '[connection error: ' + e.message + ']';
      botBub.style.color = 'var(--err)';
    }
    finishGen(botBub, fullText, 'cancelled');
  }
}

function finishGen(bub, fullText, reason) {
  bub.classList.remove('streaming');
  if (fullText) history.push({role:'assistant', content: fullText});

  const elapsed = ((Date.now() - genStart) / 1000).toFixed(1);
  const tps     = tokenCount > 0 ? (tokenCount / (elapsed || 1)).toFixed(1) : '–';
  const meta    = document.createElement('div');
  meta.className = 'meta bot';
  meta.textContent = tokenCount + ' tokens  ·  ' + elapsed + 's  ·  ' + tps + ' tok/s'
                   + (reason && reason !== 'stop' ? '  ·  ' + reason : '');
  bub.parentNode.appendChild(meta);

  generating = false;
  abortCtrl  = null;
  sendBtn.textContent = 'Send';
  sendBtn.className   = '';
  prompt.focus();
  fetchHealth();
}

function cancelGen() {
  if (abortCtrl) abortCtrl.abort();
}

// ── init ──────────────────────────────────────────────────────────────────────
fetchHealth();
fetchModels();
setInterval(fetchHealth, 10000);
prompt.focus();
</script>
</body>
</html>
""";

	// ── Health dashboard HTML ─────────────────────────────────────────────────
	// Served at GET /health-ui. Fetches data from /health-data (same-origin proxy
	// to the sidecar) — no CORS issues, no extra port needed in AWS security groups.

	private static final String HEALTH_DASHBOARD_HTML = """
		<!DOCTYPE html>
		<html lang="en">
		<head>
		  <meta charset="UTF-8">
		  <meta name="viewport" content="width=device-width, initial-scale=1.0">
		  <title>Juno Health</title>
		  <style>
		    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
		    :root {
		      --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#c9d1d9;
		      --muted:#8b949e; --green:#3fb950; --yellow:#d29922; --red:#f85149;
		      --blue:#58a6ff; --accent:#1f6feb;
		    }
		    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; min-height:100vh; }
		    header { background:var(--surface); border-bottom:1px solid var(--border); padding:1rem 2rem; display:flex; align-items:center; gap:1rem; }
		    header h1 { font-size:1.25rem; font-weight:600; }
		    header a { margin-left:auto; font-size:.75rem; color:var(--blue); text-decoration:none; }
		    header a:hover { text-decoration:underline; }
		    .badge { display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .75rem; border-radius:2rem; font-size:.75rem; font-weight:600; letter-spacing:.05em; }
		    .badge.healthy  { background:#0d2a16; color:var(--green); border:1px solid var(--green); }
		    .badge.degraded { background:#2a1f05; color:var(--yellow); border:1px solid var(--yellow); }
		    .badge.down     { background:#2a0b0b; color:var(--red); border:1px solid var(--red); }
		    .badge.unknown  { background:#1a1f27; color:var(--muted); border:1px solid var(--border); }
		    .dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
		    main { padding:2rem; max-width:1200px; margin:0 auto; }
		    .summary { display:flex; align-items:center; gap:1.5rem; margin-bottom:2rem; flex-wrap:wrap; }
		    .stat { background:var(--surface); border:1px solid var(--border); border-radius:.5rem; padding:.75rem 1.25rem; }
		    .stat .label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.25rem; }
		    .stat .value { font-size:1.5rem; font-weight:700; }
		    .nodes { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:1rem; }
		    .node-card { background:var(--surface); border:1px solid var(--border); border-radius:.75rem; padding:1.25rem; transition:border-color .15s; }
		    .node-card:hover { border-color:var(--accent); }
		    .node-card.circuit-OPEN      { border-color:var(--red); }
		    .node-card.circuit-HALF_OPEN { border-color:var(--yellow); }
		    .node-card-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; }
		    .node-id { font-weight:600; font-size:.95rem; }
		    .circuit-badge { font-size:.65rem; font-weight:700; padding:.2rem .5rem; border-radius:.25rem; letter-spacing:.06em; }
		    .circuit-CLOSED    { background:#0d2a16; color:var(--green); }
		    .circuit-OPEN      { background:#2a0b0b; color:var(--red); }
		    .circuit-HALF_OPEN { background:#2a1f05; color:var(--yellow); }
		    .metric-row { display:flex; justify-content:space-between; margin-bottom:.6rem; font-size:.83rem; }
		    .metric-label { color:var(--muted); }
		    .metric-value { font-weight:500; font-variant-numeric:tabular-nums; }
		    .vram-bar-wrap { margin-top:.75rem; }
		    .vram-bar-label { display:flex; justify-content:space-between; font-size:.72rem; color:var(--muted); margin-bottom:.3rem; }
		    .vram-bar-bg { background:var(--border); border-radius:4px; height:6px; overflow:hidden; }
		    .vram-bar-fill { height:100%; border-radius:4px; transition:width .4s ease; }
		    .bar-ok   { background:var(--green); }
		    .bar-warn { background:var(--yellow); }
		    .bar-crit { background:var(--red); }
		    .refresh-ticker { font-size:.72rem; color:var(--muted); display:flex; align-items:center; gap:.4rem; }
		    .empty-state { text-align:center; padding:4rem 2rem; color:var(--muted); }
		    footer { text-align:center; padding:2rem; font-size:.75rem; color:var(--muted); border-top:1px solid var(--border); margin-top:2rem; }
		    footer a { color:var(--blue); text-decoration:none; }
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
		    <a href="/">← Console</a>
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
		    <a href="/health-data">/health-data</a> &nbsp;·&nbsp; POST /health/probe &nbsp;—&nbsp; Juno Health Monitor
		  </footer>
		  <script>
		    let countdown = 5;
		    const sc = s => s==='HEALTHY'?'healthy':s==='DEGRADED'?'degraded':s==='DOWN'?'down':'unknown';
		    const bc = p => p>=0.98?'bar-crit':p>=0.90?'bar-warn':'bar-ok';
		    const fb = b => b<=0?'—':b>=1e9?(b/1e9).toFixed(1)+' GB':(b/1e6).toFixed(0)+' MB';
		    const fa = ms => ms<2000?ms+' ms':ms<60000?(ms/1000).toFixed(1)+' s':(ms/60000).toFixed(1)+' m';
		    const fc = v => v>=0?(v*100).toFixed(1)+' %':'—';
		    const fl = ms => ms>=0?ms.toFixed(0)+' ms':'—';
		    const fth = bps => bps>=0?(bps/1048576).toFixed(2)+' MB/s':'—';
		    async function refresh() {
		      try {
		        const r = await fetch('/health-data');
		        if (!r.ok) throw new Error('HTTP '+r.status);
		        const d = await r.json();
		        const badge = document.getElementById('clusterBadge');
		        badge.className = 'badge '+sc(d.status);
		        badge.innerHTML = '<span class="dot"></span>'+(d.status||'UNKNOWN');
		        document.getElementById('nodeCount').textContent = d.nodeCount??0;
		        const nodes = d.nodes||[];
		        document.getElementById('openCount').textContent = nodes.filter(n=>n.circuit==='OPEN').length;
		        const avg = nodes.length?(nodes.reduce((s,n)=>s+(n.vramPressure||0),0)/nodes.length*100).toFixed(1)+'%':'—';
		        document.getElementById('avgVram').textContent = avg;
		        const grid = document.getElementById('nodeGrid');
		        if (!nodes.length) {
		          grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h2>No nodes yet</h2><p>Nodes push probes to POST /health/probe</p></div>';
		        } else {
		          grid.innerHTML = nodes.map(n => {
		            const pct = ((n.vramPressure||0)*100).toFixed(1);
		            return '<div class="node-card circuit-'+n.circuit+'">'
		              +'<div class="node-card-header"><div class="node-id">'+n.nodeId+'</div>'
		              +'<span class="circuit-badge circuit-'+n.circuit+'">'+n.circuit+'</span></div>'
		              +'<div class="metric-row"><span class="metric-label">VRAM free</span><span class="metric-value">'+fb(n.vramFreeBytes)+' / '+fb(n.vramTotalBytes)+'</span></div>'
		              +'<div class="metric-row"><span class="metric-label">CPU load</span><span class="metric-value">'+fc(n.cpuLoad)+'</span></div>'
		              +'<div class="metric-row"><span class="metric-label">'+(n.nodeRole==='coordinator'?'Latency P99':'Throughput')+'</span>'
		              +'<span class="metric-value">'+(n.nodeRole==='coordinator'?fl(n.inferenceLatencyP99):fth(n.throughputBytesPerSec))+'</span></div>'
		              +'<div class="metric-row"><span class="metric-label">Last seen</span>'
		              +'<span class="metric-value" style="color:'+(n.ageMs>10000?'var(--red)':'var(--muted)')+'">'+fa(n.ageMs)+'</span></div>'
		              +'<div class="vram-bar-wrap"><div class="vram-bar-label"><span>VRAM pressure</span><span>'+pct+'%</span></div>'
		              +'<div class="vram-bar-bg"><div class="vram-bar-fill '+bc(n.vramPressure)+'" style="width:'+Math.min(100,pct)+'%"></div></div></div>'
		              +'</div>';
		          }).join('');
		        }
		      } catch(e) {
		        const badge = document.getElementById('clusterBadge');
		        badge.className = 'badge unknown';
		        badge.innerHTML = '<span class="dot"></span>UNREACHABLE';
		      }
		    }
		    refresh();
		    setInterval(()=>{countdown--;if(countdown<=0){countdown=5;refresh();}document.getElementById('countdown').textContent=countdown;},1000);
		  </script>
		</body>
		</html>
		""";

	// ── Route handlers ────────────────────────────────────────────────────────

	private void handleBlockingInference(Context ctx) {
		InferenceRequest request = parseRequest(ctx);
		if (request == null)
			return;

		long start = System.currentTimeMillis();
		GenerationResult result = scheduler.submitAndWait(request);
		if (latencyReporter != null)
			latencyReporter.recordLatency(System.currentTimeMillis() - start);
		ctx.json(toResponse(result, request.modelId()));
	}

	private void handleStreamingInference(Context ctx) {
		// Parse request body before switching to SSE mode
		ApiInferenceRequest body = parseBody(ctx);
		if (body == null)
			return;

		String modelId = resolveModelId(body.modelId());
		if (modelId == null) {
			ctx.status(503).json(errorBody(503, "SERVICE_UNAVAILABLE", "No model is currently loaded"));
			return;
		}

		if (!modelRegistry.isLoaded(modelId)) {
			ctx.status(503).json(errorBody(503, "MODEL_NOT_LOADED", "Model '" + modelId + "' is not loaded"));
			return;
		}

		InferenceRequest request = toInferenceRequest(body, modelId);

		// Set SSE headers manually — Javalin's sse() API doesn't support
		// POST bodies, so we drive SSE by hand on a regular POST route.
		ctx.res().setContentType("text/event-stream");
		ctx.res().setCharacterEncoding("UTF-8");
		ctx.res().setHeader("Cache-Control", "no-cache");
		ctx.res().setHeader("X-Accel-Buffering", "no");

		SseTokenConsumer consumer = new SseTokenConsumer(request.requestId(), data -> {
			try {
				ctx.res().getWriter().write("data: " + data + "\n\n");
				ctx.res().getWriter().flush();
			} catch (Exception e) {
				log.fine("SSE write failed (client disconnected?): " + e.getMessage());
			}
		});

		try {
			long start = System.currentTimeMillis();
			GenerationResult result = scheduler.submit(request, consumer).join();
			if (latencyReporter != null)
				latencyReporter.recordLatency(System.currentTimeMillis() - start);
			String finishReason = toFinishReason(result.stopReason());
			consumer.sendComplete(finishReason);
			ctx.res().getWriter().flush();
		} catch (Exception e) {
			consumer.sendComplete("error");
			log.warning("SSE generation error for " + request.requestId() + ": " + e.getMessage());
		}
	}

	private void handleListModels(Context ctx) {
		List<ModelDescriptor> models = modelRegistry.listModels();
		ctx.json(Map.of("models", models.stream().map(this::toModelResponse).toList(), "total", models.size()));
	}

	private void handleGetModel(Context ctx) {
		String modelId = ctx.pathParam("modelId");
		modelRegistry.getModel(modelId).ifPresentOrElse(m -> ctx.json(toModelResponse(m)),
				() -> ctx.status(404).json(errorBody(404, "NOT_FOUND", "Model '" + modelId + "' not found")));
	}

	private void handleUnloadModel(Context ctx) {
		String modelId = ctx.pathParam("modelId");
		if (modelRegistry.getModel(modelId).isEmpty()) {
			ctx.status(404).json(errorBody(404, "NOT_FOUND", "Model '" + modelId + "' not found"));
			return;
		}
		modelRegistry.unregister(modelId);
		ctx.status(204);
	}

	private void handleClusterHealth(Context ctx) {
		ctx.json(Map.of(
			    "status", "HEALTHY",
			    "queueDepth", scheduler.queueDepth(),
			    "maxQueue", scheduler.maxQueueDepth(),
			    "loadedModels", modelRegistry.modelCount(),
			    "byteOrder", byteOrder
			));
	}

	// ── Request parsing ───────────────────────────────────────────────────────

	/** Parse and validate, set error response and return null on failure. */
	private InferenceRequest parseRequest(Context ctx) {
		ApiInferenceRequest body = parseBody(ctx);
		if (body == null)
			return null;

		String modelId = resolveModelId(body.modelId());
		if (modelId == null) {
			ctx.status(503).json(errorBody(503, "SERVICE_UNAVAILABLE", "No model is currently loaded"));
			return null;
		}
		if (!modelRegistry.isLoaded(modelId)) {
			ctx.status(503).json(errorBody(503, "MODEL_NOT_LOADED", "Model '" + modelId + "' is not loaded"));
			return null;
		}
		return toInferenceRequest(body, modelId);
	}

	private ApiInferenceRequest parseBody(Context ctx) {
		try {
			ApiInferenceRequest body = ctx.bodyAsClass(ApiInferenceRequest.class);
			if (body.messages() == null || body.messages().isEmpty()) {
				ctx.status(400).json(errorBody(400, "BAD_REQUEST", "messages must not be empty"));
				return null;
			}
			return body;
		} catch (Exception e) {
			ctx.status(400).json(errorBody(400, "BAD_REQUEST", "Invalid request body: " + e.getMessage()));
			return null;
		}
	}

	private String resolveModelId(String requested) {
		if (requested != null && !requested.isBlank())
			return requested;
		// Default to first loaded model
		return modelRegistry.listModels().stream().filter(m -> modelRegistry.isLoaded(m.modelId()))
				.map(ModelDescriptor::modelId).findFirst().orElse(null);
	}

	private InferenceRequest toInferenceRequest(ApiInferenceRequest body, String modelId) {
		List<ChatMessage> messages = body.messages().stream().map(m -> new ChatMessage(m.role(), m.content())).toList();

		SamplingParams params = buildSamplingParams(body.sampling());
		RequestPriority priority = parsePriority(body.sampling() != null ? body.sampling().priority() : null);

		return InferenceRequest.of(modelId, messages, params, priority);
	}

	private SamplingParams buildSamplingParams(ApiSampling s) {
		if (s == null)
			return SamplingParams.defaults();
		SamplingParams params = SamplingParams.defaults();
		if (s.maxTokens() != null)
			params = params.withMaxTokens(s.maxTokens());
		if (s.temperature() != null)
			params = params.withTemperature(s.temperature());
		if (s.topK() != null)
			params = params.withTopK(s.topK());
		if (s.topP() != null)
			params = params.withTopP(s.topP());
		return params;
	}

	private RequestPriority parsePriority(String priority) {
		if (priority == null)
			return RequestPriority.NORMAL;
		return switch (priority.toUpperCase()) {
		case "HIGH" -> RequestPriority.HIGH;
		case "LOW" -> RequestPriority.LOW;
		default -> RequestPriority.NORMAL;
		};
	}

	// ── Response builders ─────────────────────────────────────────────────────

	private Map<String, Object> toResponse(GenerationResult result, String modelId) {
		return Map.of("requestId", result.requestId(), "text", result.text(), "tokenCount", result.generatedTokens(),
				"promptTokenCount", result.promptTokens(), "finishReason", toFinishReason(result.stopReason()),
				"modelId", modelId, "latencyMs", result.latency().toMillis());
	}

	private Map<String, Object> toModelResponse(ModelDescriptor m) {
		return Map.of("modelId", m.modelId(), "architecture", m.architecture(), "quantization",
				m.quantization().displayName(), "totalLayers", m.totalLayers(), "hiddenDim", m.hiddenDim(), "vocabSize",
				m.vocabSize(), "status", m.status().name(), "estimatedVram", m.humanReadableSize());
	}

	private static String toFinishReason(GenerationResult.StopReason reason) {
		return switch (reason) {
		case EOS_TOKEN, STOP_TOKEN -> "stop";
		case MAX_TOKENS -> "length";
		case ERROR -> "error";
		};
	}

	private static Map<String, Object> errorBody(int code, String error, String message) {
		return Map.of("code", code, "error", error, "message", message);
	}

	// ── DTOs (parsed by Javalin/Jackson from request body) ───────────────────

	public record ApiInferenceRequest(String requestId, String modelId, List<ApiMessage> messages,
			ApiSampling sampling) {
	}

	public record ApiMessage(String role, String content) {
	}

	public record ApiSampling(Float temperature, Integer topK, Float topP, Integer maxTokens, String priority) {
	}
}