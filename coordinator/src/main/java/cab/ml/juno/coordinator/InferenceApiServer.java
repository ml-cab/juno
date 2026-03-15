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

	public InferenceApiServer(RequestScheduler scheduler, ModelRegistry modelRegistry) {
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
		ctx.json(Map.of("status", "HEALTHY", "queueDepth", scheduler.queueDepth(), "maxQueue",
				scheduler.maxQueueDepth(), "loadedModels", modelRegistry.modelCount()));
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