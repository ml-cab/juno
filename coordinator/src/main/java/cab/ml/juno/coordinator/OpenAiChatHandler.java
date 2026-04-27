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
 * OpenAI-compatible handlers for chat completions and model listing.
 */
public final class OpenAiChatHandler {

	private static final Logger log = Logger.getLogger(OpenAiChatHandler.class.getName());
	private static final ObjectMapper JSON = new ObjectMapper();

	private final RequestScheduler scheduler;
	private final ModelRegistry modelRegistry;
	private final java.util.function.LongConsumer latencyCallback;

	public OpenAiChatHandler(RequestScheduler scheduler, ModelRegistry modelRegistry,
			java.util.function.LongConsumer latencyCallback) {
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
		this.latencyCallback = latencyCallback;
	}

	public void handleChatCompletion(Context ctx) {
		OaiChatCompletionRequest body;
		try {
			body = JSON.readValue(ctx.body(), OaiChatCompletionRequest.class);
		} catch (Exception e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", "Invalid request body: " + e.getMessage(), null);
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
				openAiError(ctx, 400, "invalid_request_error", "invalid_request", "each message needs a non-blank role", "messages");
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

		String modelId = resolveModelId(body.model());
		if (modelId == null) {
			openAiError(ctx, 503, "service_unavailable_error", "service_unavailable", "No model is currently loaded", null);
			return;
		}
		if (!modelRegistry.isLoaded(modelId)) {
			openAiError(ctx, 503, "service_unavailable_error", "service_unavailable", "Model '" + modelId + "' is not loaded", "model");
			return;
		}

		SamplingParams sampling = buildSamplingParams(body);
		RequestPriority priority = parsePriority(body.xJunoPriority());
		InferenceRequest request = (body.xJunoSessionId() != null && !body.xJunoSessionId().isBlank())
				? InferenceRequest.ofSession(body.xJunoSessionId().strip(), modelId, messages, sampling, priority)
				: InferenceRequest.of(modelId, messages, sampling, priority);

		if (Boolean.TRUE.equals(body.stream())) {
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

			Map<String, Object> choice = Map.of("index", 0, "message", Map.of("role", "assistant", "content", result.text()),
					"finish_reason", finish);
			Map<String, Object> usage = Map.of("prompt_tokens", result.promptTokens(), "completion_tokens", result.generatedTokens(),
					"total_tokens", result.promptTokens() + result.generatedTokens());
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
			openAiError(ctx, 500, "internal_error", "internal_error", e.getMessage() != null ? e.getMessage() : "Unexpected error", null);
		}
	}

	private void handleStreamingChat(Context ctx, InferenceRequest request, String modelId) {
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

		try {
			TokenConsumer consumer = new TokenConsumer() {
				@Override
				public void onPrefillComplete() {
					writeChunkQuietly(writer, chunkRoot(completionId, created, modelId,
							List.of(chunkChoice(0, Map.of("role", "assistant", "content", ""), null))));
				}

				@Override
				public void onToken(String piece, int tokenId, int position) {
					writeChunkQuietly(writer, chunkRoot(completionId, created, modelId,
							List.of(chunkChoice(0, Map.of("content", piece != null ? piece : ""), null))));
				}
			};

			long start = System.currentTimeMillis();
			GenerationResult result = scheduler.submit(request, consumer).join();
			latencyCallback.accept(System.currentTimeMillis() - start);

			writeSseChunk(writer, chunkRoot(completionId, created, modelId, List.of(chunkChoice(0, Map.of("content", ""),
					OpenAiAdapter.toOpenAiFinishReason(result.stopReason())))));
			writer.write("data: [DONE]\n\n");
			writer.flush();
		} catch (RequestScheduler.QueueFullException e) {
			writeJsonQueueFull(ctx, e);
		} catch (Exception e) {
			log.warning("OpenAI streaming error: " + e.getMessage());
		}
	}

	private static void writeChunkQuietly(java.io.Writer writer, Map<String, Object> chunk) {
		try {
			writeSseChunk(writer, chunk);
		} catch (IOException e) {
			log.fine("SSE write failed: " + e.getMessage());
		}
	}

	private static Map<String, Object> chunkRoot(String id, long created, String modelId, List<Map<String, Object>> choices) {
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

	public void handleListModels(Context ctx) {
		List<Map<String, Object>> data = modelRegistry.listModels().stream().filter(m -> m.status() == ModelStatus.LOADED)
				.map(this::toOpenAiModel).toList();
		ctx.json(Map.of("object", "list", "data", data));
	}

	public void handleGetModel(Context ctx) {
		String modelId = ctx.pathParam("modelId");
		modelRegistry.getModel(modelId).ifPresentOrElse(m -> ctx.json(toOpenAiModel(m)),
				() -> openAiError(ctx, 404, "invalid_request_error", "model_not_found", "Model '" + modelId + "' not found", "model"));
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

	private String resolveModelId(String requested) {
		if (requested != null && !requested.isBlank())
			return requested.strip();
		return modelRegistry.listModels().stream().filter(m -> modelRegistry.isLoaded(m.modelId())).map(ModelDescriptor::modelId)
				.findFirst().orElse(null);
	}

	private static String extractTextContent(JsonNode content) {
		if (content == null || content.isNull() || !content.isTextual())
			return null;
		return content.asText();
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
		if (body.frequencyPenalty() != null)
			p = p.withRepetitionPenalty(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(body.frequencyPenalty().floatValue()));
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

	private void writeJsonQueueFull(Context ctx, RequestScheduler.QueueFullException e) {
		try {
			ctx.res().resetBuffer();
		} catch (Exception ignored) {
		}
		queueFull(ctx, e);
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
	public record OaiChatCompletionRequest(@JsonProperty("model") String model, @JsonProperty("messages") List<OaiMessage> messages,
			@JsonProperty("temperature") Double temperature, @JsonProperty("top_p") Double topP,
			@JsonProperty("max_tokens") Integer maxTokens, @JsonProperty("max_completion_tokens") Integer maxCompletionTokens,
			@JsonProperty("stream") Boolean stream, @JsonProperty("n") Integer n,
			@JsonProperty("frequency_penalty") Double frequencyPenalty, @JsonProperty("stop") JsonNode stop,
			@JsonProperty("x_juno_priority") String xJunoPriority, @JsonProperty("x_juno_session_id") String xJunoSessionId,
			@JsonProperty("x_juno_top_k") Integer xJunoTopK) {
	}

	public record OaiMessage(@JsonProperty("role") String role, @JsonProperty("content") JsonNode content) {
	}
}
