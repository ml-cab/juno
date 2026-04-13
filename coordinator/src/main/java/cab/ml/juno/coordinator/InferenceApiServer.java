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
	private Javalin app;
	private String byteOrder;

	public InferenceApiServer(RequestScheduler scheduler, ModelRegistry modelRegistry, String byteOrder) {
		this.byteOrder = byteOrder != null ? byteOrder : "BE";
		if (scheduler == null)
			throw new IllegalArgumentException("scheduler must not be null");
		if (modelRegistry == null)
			throw new IllegalArgumentException("modelRegistry must not be null");
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
	}

	public void start(int port) {
		app = Javalin.create(config -> {
			config.useVirtualThreads = true;
			config.showJavalinBanner = false;
		});

		// ── Web console ───────────────────────────────────────────────────────
		app.get("/", this::handleConsole);

		// ── Inference ─────────────────────────────────────────────────────────
		app.post("/v1/inference", this::handleBlockingInference);
		app.post("/v1/inference/stream", this::handleStreamingInference);

		// ── Models ────────────────────────────────────────────────────────────
		app.get("/v1/models", this::handleListModels);
		app.get("/v1/models/{modelId}", this::handleGetModel);
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

	// ── Web console ───────────────────────────────────────────────────────────

	private void handleConsole(Context ctx) {
		ctx.contentType("text/html; charset=UTF-8").result(CONSOLE_HTML);
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

	// ── Route handlers ────────────────────────────────────────────────────────

	private void handleBlockingInference(Context ctx) {
		InferenceRequest request = parseRequest(ctx);
		if (request == null)
			return; // parseRequest already set error response

		GenerationResult result = scheduler.submitAndWait(request);
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
			GenerationResult result = scheduler.submit(request, consumer).join();
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