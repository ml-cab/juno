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

package cab.ml.juno.apiserver;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import io.javalin.http.Context;

/**
 * OpenAI-compatible HTTP handlers for {@code juno-api.yaml}: chat completions and model listing.
 */
public final class OpenAiChatHandler {

	private static final Logger log = Logger.getLogger(OpenAiChatHandler.class.getName());
	private static final ObjectMapper JSON = new ObjectMapper();

	private final RequestScheduler scheduler;
	private final ModelRegistry modelRegistry;
	private final java.util.function.LongConsumer latencyCallback;

	public OpenAiChatHandler(RequestScheduler scheduler, ModelRegistry modelRegistry,
			java.util.function.LongConsumer latencyCallback) {
		if (scheduler == null)
			throw new IllegalArgumentException("scheduler must not be null");
		if (modelRegistry == null)
			throw new IllegalArgumentException("modelRegistry must not be null");
		if (latencyCallback == null)
			throw new IllegalArgumentException("latencyCallback must not be null");
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
		this.latencyCallback = latencyCallback;
	}

	// ── POST /v1/chat/completions ─────────────────────────────────────────────

	public void handleChatCompletion(Context ctx) {
		OaiChatCompletionRequest body;
		try {
			body = JSON.readValue(ctx.body(), OaiChatCompletionRequest.class);
		} catch (Exception e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request",
					"Invalid request body: " + e.getMessage(), null);
			return;
		}

		String nError = OpenAiAdapter.validateCompletionsN(body.n());
		if (nError != null) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", nError, "n");
			return;
		}

		if (body.messages() == null || body.messages().isEmpty()) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", "messages must not be empty", "messages");
			return;
		}

		List<ChatMessage> messages = new ArrayList<>();
		for (OaiMessage m : body.messages()) {
			if (m == null || m.role() == null || m.role().isBlank()) {
				openAiError(ctx, 400, "invalid_request_error", "invalid_request", "each message needs a non-blank role",
						"messages");
				return;
			}
			String text = extractTextContent(m.content());
			if (text == null) {
				openAiError(ctx, 400, "invalid_request_error", "invalid_request",
						"Only string text content is supported in messages[].content", "messages");
				return;
			}
			messages.add(new ChatMessage(m.role(), text));
		}

		if (body.stop() != null && !body.stop().isNull()) {
			log.fine("OpenAI request: stop sequences are not implemented; ignoring.");
		}

		String modelId = resolveModelId(body.model());
		if (modelId == null) {
			openAiError(ctx, 503, "service_unavailable_error", "service_unavailable", "No model is currently loaded",
					null);
			return;
		}
		if (!modelRegistry.isLoaded(modelId)) {
			openAiError(ctx, 503, "service_unavailable_error", "service_unavailable",
					"Model '" + modelId + "' is not loaded", "model");
			return;
		}

		SamplingParams sampling;
		try {
			sampling = buildSamplingParams(body);
		} catch (IllegalArgumentException e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), null);
			return;
		}

		RequestPriority priority = parsePriority(body.xJunoPriority());

		InferenceRequest request;
		String sid = body.xJunoSessionId();
		if (sid != null && !sid.isBlank()) {
			request = InferenceRequest.ofSession(sid.strip(), modelId, messages, sampling, priority);
		} else {
			request = InferenceRequest.of(modelId, messages, sampling, priority);
		}

		boolean stream = Boolean.TRUE.equals(body.stream());
		if (stream) {
			handleStreamingChat(ctx, request, modelId);
		} else {
			handleBlockingChat(ctx, request, modelId);
		}
	}

	private void handleBlockingChat(Context ctx, InferenceRequest request, String modelId) {
		try {
			long start = System.currentTimeMillis();
			GenerationResult result = scheduler.submitAndWait(request);
			latencyCallback.accept(System.currentTimeMillis() - start);

			String completionId = OpenAiAdapter.chatCompletionId(result.requestId());
			long created = request.receivedAt().getEpochSecond();
			String finish = OpenAiAdapter.toOpenAiFinishReason(result.stopReason());

			Map<String, Object> message = Map.of("role", "assistant", "content", result.text());
			Map<String, Object> choice = new LinkedHashMap<>();
			choice.put("index", 0);
			choice.put("message", message);
			choice.put("finish_reason", finish);

			Map<String, Object> usage = Map.of("prompt_tokens", result.promptTokens(), "completion_tokens",
					result.generatedTokens(), "total_tokens", result.promptTokens() + result.generatedTokens());

			Map<String, Object> root = new LinkedHashMap<>();
			root.put("id", completionId);
			root.put("object", "chat.completion");
			root.put("created", created);
			root.put("model", modelId);
			root.put("choices", List.of(choice));
			root.put("usage", usage);
			root.put("x_juno_latency_ms", result.latency().toMillis());

			ctx.header("X-Juno-Latency-Ms", Long.toString(result.latency().toMillis()));
			ctx.status(200).json(root);
		} catch (RequestScheduler.QueueFullException e) {
			queueFull(ctx, e);
		} catch (Exception e) {
			log.warning("OpenAI chat completion error: " + e.getMessage());
			openAiError(ctx, 500, "internal_error", "internal_error",
					e.getMessage() != null ? e.getMessage() : "Unexpected error", null);
		}
	}

	private void handleStreamingChat(Context ctx, InferenceRequest request, String modelId) {
		if (scheduler.queueDepth() >= scheduler.maxQueueDepth()) {
			queueFull(ctx, new RequestScheduler.QueueFullException(
					"Request queue full (" + scheduler.maxQueueDepth() + "). Retry later.",
					Math.max(1, scheduler.queueDepth() * 2)));
			return;
		}

		String completionId = OpenAiAdapter.chatCompletionId(request.requestId());
		long created = request.receivedAt().getEpochSecond();

		ctx.res().setContentType("text/event-stream");
		ctx.res().setCharacterEncoding("UTF-8");
		ctx.res().setHeader("Cache-Control", "no-cache");
		ctx.res().setHeader("X-Accel-Buffering", "no");

		final java.io.PrintWriter writer;
		try {
			writer = ctx.res().getWriter();
		} catch (IOException e) {
			openAiError(ctx, 500, "internal_error", "internal_error", "Could not open response writer", null);
			return;
		}

		java.util.concurrent.CompletableFuture<GenerationResult> future;
		try {
			TokenConsumer consumer = new TokenConsumer() {
				@Override
				public void onPrefillComplete() {
					try {
						writeSseChunk(writer, firstChunk(completionId, created, modelId));
					} catch (IOException e) {
						log.fine("OpenAI SSE first chunk failed: " + e.getMessage());
					}
				}

				@Override
				public void onToken(String piece, int tokenId, int position) {
					try {
						writeSseChunk(writer, tokenChunk(completionId, created, modelId, piece));
					} catch (IOException ex) {
						log.fine("OpenAI SSE token write failed: " + ex.getMessage());
					}
				}
			};
			future = scheduler.submit(request, consumer);
		} catch (RequestScheduler.QueueFullException e) {
			writeJsonQueueFull(ctx, e);
			return;
		}

		try {

			long start = System.currentTimeMillis();
			GenerationResult result = future.join();
			latencyCallback.accept(System.currentTimeMillis() - start);

			String finish = OpenAiAdapter.toOpenAiFinishReason(result.stopReason());
			writeSseChunk(writer, finalChunk(completionId, created, modelId, finish));
			writer.write("data: [DONE]\n\n");
			writer.flush();
		} catch (Exception e) {
			Throwable root = e instanceof java.util.concurrent.CompletionException && e.getCause() != null
					? e.getCause()
					: e;
			log.warning("OpenAI streaming chat error: " + root.getMessage());
			try {
				writeSseChunk(writer, finalChunk(completionId, created, modelId, "error"));
				writer.write("data: [DONE]\n\n");
				writer.flush();
			} catch (IOException ioe) {
				log.fine("OpenAI SSE error tail failed: " + ioe.getMessage());
			}
		}
	}

	private void writeJsonQueueFull(Context ctx, RequestScheduler.QueueFullException e) {
		try {
			ctx.res().resetBuffer();
		} catch (Exception ignored) {
			// committed — best-effort
		}
		ctx.status(429);
		ctx.header("Retry-After", Integer.toString(e.retryAfterSeconds()));
		ctx.res().setContentType("application/json");
		ctx.res().setCharacterEncoding("UTF-8");
		ctx.json(queueFullBody(e));
	}

	private static Map<String, Object> firstChunk(String id, long created, String modelId) {
		Map<String, Object> delta = new LinkedHashMap<>();
		delta.put("role", "assistant");
		delta.put("content", "");
		Map<String, Object> choice = chunkChoice(0, delta, null);
		return chunkRoot(id, created, modelId, List.of(choice));
	}

	private static Map<String, Object> tokenChunk(String id, long created, String modelId, String piece) {
		Map<String, Object> delta = new LinkedHashMap<>();
		delta.put("content", piece != null ? piece : "");
		Map<String, Object> choice = chunkChoice(0, delta, null);
		return chunkRoot(id, created, modelId, List.of(choice));
	}

	private static Map<String, Object> finalChunk(String id, long created, String modelId, String finishReason) {
		Map<String, Object> delta = new LinkedHashMap<>();
		delta.put("content", "");
		Map<String, Object> choice = chunkChoice(0, delta, finishReason);
		return chunkRoot(id, created, modelId, List.of(choice));
	}

	private static Map<String, Object> chunkRoot(String id, long created, String modelId,
			List<Map<String, Object>> choices) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("id", id);
		m.put("object", "chat.completion.chunk");
		m.put("created", created);
		m.put("model", modelId);
		m.put("choices", choices);
		return m;
	}

	private static Map<String, Object> chunkChoice(int index, Map<String, Object> delta, String finishReason) {
		Map<String, Object> c = new LinkedHashMap<>();
		c.put("index", index);
		c.put("delta", delta);
		c.put("finish_reason", finishReason);
		return c;
	}

	private static void writeSseChunk(java.io.Writer writer, Map<String, Object> chunk) throws IOException {
		writer.write("data: ");
		writer.write(JSON.writeValueAsString(chunk));
		writer.write("\n\n");
		writer.flush();
	}

	// ── GET /v1/models (OpenAI list) ────────────────────────────────────────

	public void handleListModels(Context ctx) {
		List<ModelDescriptor> all = modelRegistry.listModels();
		List<Map<String, Object>> data = all.stream().filter(m -> m.status() == ModelStatus.LOADED)
				.map(this::toOpenAiModel).toList();
		ctx.json(Map.of("object", "list", "data", data));
	}

	public void handleGetModel(Context ctx) {
		String modelId = ctx.pathParam("modelId");
		modelRegistry.getModel(modelId).ifPresentOrElse(m -> ctx.json(toOpenAiModel(m)), () -> openAiError(ctx, 404,
				"invalid_request_error", "model_not_found", "Model '" + modelId + "' not found", "model"));
	}

	private Map<String, Object> toOpenAiModel(ModelDescriptor m) {
		Map<String, Object> o = new LinkedHashMap<>();
		o.put("id", m.modelId());
		o.put("object", "model");
		o.put("created", m.registeredAt().getEpochSecond());
		o.put("owned_by", "juno");
		o.put("x_juno_architecture", m.architecture());
		o.put("x_juno_quantization", m.quantization().displayName());
		o.put("x_juno_total_layers", m.totalLayers());
		o.put("x_juno_hidden_dim", m.hiddenDim());
		o.put("x_juno_vocab_size", m.vocabSize());
		o.put("x_juno_status", m.status().name());
		return o;
	}

	// ── Request parsing / translation ─────────────────────────────────────────

	private String resolveModelId(String requested) {
		if (requested != null && !requested.isBlank())
			return requested.strip();
		return modelRegistry.listModels().stream().filter(m -> modelRegistry.isLoaded(m.modelId()))
				.map(ModelDescriptor::modelId).findFirst().orElse(null);
	}

	private static String extractTextContent(JsonNode content) {
		if (content == null || content.isNull())
			return null;
		if (content.isTextual())
			return content.asText();
		return null;
	}

	private SamplingParams buildSamplingParams(OaiChatCompletionRequest body) {
		SamplingParams p = SamplingParams.defaults();

		Integer maxTok = body.maxCompletionTokens() != null ? body.maxCompletionTokens() : body.maxTokens();
		if (maxTok != null)
			p = p.withMaxTokens(maxTok);

		if (body.temperature() != null)
			p = p.withTemperature(body.temperature().floatValue());

		if (body.topP() != null)
			p = p.withTopP(body.topP().floatValue());

		if (body.xJunoTopK() != null)
			p = p.withTopK(body.xJunoTopK());

		if (body.frequencyPenalty() != null) {
			float rep = OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(body.frequencyPenalty().floatValue());
			p = p.withRepetitionPenalty(rep);
		}

		return p;
	}

	private static RequestPriority parsePriority(String priority) {
		if (priority == null || priority.isBlank())
			return RequestPriority.NORMAL;
		return switch (priority.strip().toUpperCase()) {
		case "HIGH" -> RequestPriority.HIGH;
		case "LOW" -> RequestPriority.LOW;
		default -> RequestPriority.NORMAL;
		};
	}

	private static void openAiError(Context ctx, int status, String type, String code, String message, String param) {
		Map<String, Object> err = new LinkedHashMap<>();
		err.put("message", message);
		err.put("type", type);
		err.put("code", code);
		if (param != null)
			err.put("param", param);
		ctx.status(status).json(Map.of("error", err));
	}

	private void queueFull(Context ctx, RequestScheduler.QueueFullException e) {
		ctx.status(429);
		ctx.header("Retry-After", Integer.toString(e.retryAfterSeconds()));
		ctx.json(queueFullBody(e));
	}

	private static Map<String, Object> queueFullBody(RequestScheduler.QueueFullException e) {
		Map<String, Object> err = new LinkedHashMap<>();
		err.put("message", e.getMessage());
		err.put("type", "rate_limit_error");
		err.put("code", "rate_limit_exceeded");
		err.put("x_juno_retry_after_ms", e.retryAfterSeconds() * 1000L);
		return Map.of("error", err);
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	public record OaiChatCompletionRequest(@JsonProperty("model") String model,
			@JsonProperty("messages") List<OaiMessage> messages, @JsonProperty("temperature") Double temperature,
			@JsonProperty("top_p") Double topP, @JsonProperty("max_tokens") Integer maxTokens,
			@JsonProperty("max_completion_tokens") Integer maxCompletionTokens, @JsonProperty("stream") Boolean stream,
			@JsonProperty("n") Integer n, @JsonProperty("frequency_penalty") Double frequencyPenalty,
			@JsonProperty("stop") JsonNode stop, @JsonProperty("x_juno_priority") String xJunoPriority,
			@JsonProperty("x_juno_session_id") String xJunoSessionId, @JsonProperty("x_juno_top_k") Integer xJunoTopK) {
	}

	public record OaiMessage(@JsonProperty("role") String role, @JsonProperty("content") JsonNode content) {
	}
}
