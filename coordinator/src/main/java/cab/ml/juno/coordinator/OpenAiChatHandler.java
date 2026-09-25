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
import cab.ml.juno.registry.ModelIdResolver;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.sampler.GbnfGrammar;
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
	private final GbnfGrammar defaultGrammar;

	public OpenAiChatHandler(RequestScheduler scheduler, ModelRegistry modelRegistry,
			java.util.function.LongConsumer latencyCallback) {
		this(scheduler, modelRegistry, latencyCallback, null);
	}

	public OpenAiChatHandler(RequestScheduler scheduler, ModelRegistry modelRegistry,
			java.util.function.LongConsumer latencyCallback, GbnfGrammar defaultGrammar) {
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
		this.latencyCallback = latencyCallback;
		this.defaultGrammar = defaultGrammar;
	}

	public void handleChatCompletion(Context ctx) {
		OaiChatCompletionRequest body;
		try {
			body = JSON.readValue(ctx.body(), OaiChatCompletionRequest.class);
		} catch (Exception e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", "Invalid request body: " + e.getMessage(),
					null);
			return;
		}

		String nError = OpenAiAdapter.validateCompletionsN(body.n());
		if (nError != null) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", nError, "n");
			return;
		}
		String responseFormatError = OpenAiAdapter.validateResponseFormat(body.responseFormat());
		if (responseFormatError != null) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", responseFormatError, "response_format");
			return;
		}
		final String[] stopStrings;
		try {
			stopStrings = OpenAiAdapter.parseStop(body.stop());
		} catch (IllegalArgumentException e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "stop");
			return;
		}
		final OpenAiTools.ParsedRequest toolsReq;
		try {
			toolsReq = OpenAiTools.parse(body.tools(), body.toolChoice());
		} catch (IllegalArgumentException e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "tools");
			return;
		}
		if (body.messages() == null || body.messages().isEmpty()) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", "messages must not be empty", "messages");
			return;
		}

		List<ChatMessage> messages = new ArrayList<>();
		for (OaiMessage m : body.messages()) {
			if (m == null) {
				openAiError(ctx, 400, "invalid_request_error", "invalid_request", "each message needs a non-blank role",
						"messages");
				return;
			}
			try {
				messages.add(OpenAiTools.toChatMessage(m.role(), m.content(), m.toolCalls(), m.toolCallId()));
			} catch (IllegalArgumentException e) {
				openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "messages");
				return;
			}
		}

		String modelId = resolveModelId(ctx, body.model());
		if (modelId == null)
			return;

		try {
			OpenAiTools.rejectGrammarConflict(toolsReq, body.responseFormat(), body.xJunoGrammar(),
					defaultGrammar != null);
			messages = new ArrayList<>(OpenAiTools.bindPrompt(messages, toolsReq, modelId));
		} catch (IllegalArgumentException e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "tools");
			return;
		}

		SamplingParams sampling;
		try {
			sampling = buildSamplingParams(body, stopStrings, toolsReq);
		} catch (IllegalArgumentException e) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), null);
			return;
		}
		RequestPriority priority = parsePriority(body.xJunoPriority());
		InferenceRequest request = (body.xJunoSessionId() != null && !body.xJunoSessionId().isBlank())
				? InferenceRequest.ofSession(body.xJunoSessionId().strip(), modelId, messages, sampling, priority)
				: InferenceRequest.of(modelId, messages, sampling, priority);

		boolean disclosureEnabled = AiDisclosure.isEnabled(body.xJunoDisclosure());

		if (ContinuousLoraPolicy.forbidden(hasPerRequestLoras(body.xJunoLoras()),
				cab.ml.juno.kvcache.ServeScheduleOptions.fromEnv())) {
			openAiError(ctx, 400, "invalid_request_error", "invalid_request", ContinuousLoraPolicy.ERROR,
					"x_juno_loras");
			return;
		}

		if (Boolean.TRUE.equals(body.stream())) {
			if (toolsReq.active()) {
				handleStreamingToolsChat(ctx, request, modelId, disclosureEnabled, toolsReq);
			} else {
				handleStreamingChat(ctx, request, modelId, disclosureEnabled);
			}
		} else {
			handleBlockingChat(ctx, request, modelId, disclosureEnabled, toolsReq);
		}
	}

	private void handleBlockingChat(Context ctx, InferenceRequest request, String modelId,
			boolean disclosureEnabled, OpenAiTools.ParsedRequest toolsReq) {
		try {
			long start = System.currentTimeMillis();
			GenerationResult result = scheduler.submitAndWait(request);
			latencyCallback.accept(System.currentTimeMillis() - start);

			String completionId = OpenAiAdapter.chatCompletionId(result.requestId());
			long created = request.receivedAt().getEpochSecond();
			List<Map<String, Object>> toolCalls = parseToolCalls(result, toolsReq);
			String finish = OpenAiTools.finishReason(result.stopReason(), !toolCalls.isEmpty());

			Map<String, Object> choice = new LinkedHashMap<>();
			choice.put("index", 0);
			choice.put("message", OpenAiTools.assistantMessage(result.text(), toolCalls));
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
			if (disclosureEnabled) {
				root.put(AiDisclosure.FIELD_NAME, AiDisclosure.DISCLOSURE_TEXT);
			}

			ctx.header("X-Juno-Latency-Ms", Long.toString(result.latency().toMillis()));
			ctx.status(200).json(root);
		} catch (RequestScheduler.QueueFullException e) {
			queueFull(ctx, e);
		} catch (Exception e) {
			openAiError(ctx, 500, "internal_error", "internal_error",
					e.getMessage() != null ? e.getMessage() : "Unexpected error", null);
		}
	}

	private void handleStreamingChat(Context ctx, InferenceRequest request, String modelId,
			boolean disclosureEnabled) {
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
					Map<String, Object> firstChunk = chunkRoot(completionId, created, modelId,
							List.of(chunkChoice(0, Map.of("role", "assistant", "content", ""), null)));
					if (disclosureEnabled) {
						firstChunk.put(AiDisclosure.FIELD_NAME, AiDisclosure.DISCLOSURE_TEXT);
					}
					writeChunkQuietly(writer, firstChunk);
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

			writeSseChunk(writer, chunkRoot(completionId, created, modelId, List.of(
					chunkChoice(0, Map.of("content", ""), OpenAiAdapter.toOpenAiFinishReason(result.stopReason())))));
			writer.write("data: [DONE]\n\n");
			writer.flush();
		} catch (RequestScheduler.QueueFullException e) {
			writeJsonQueueFull(ctx, e);
		} catch (Exception e) {
			log.warning("OpenAI streaming error: " + e.getMessage());
		}
	}

	private void handleStreamingToolsChat(Context ctx, InferenceRequest request, String modelId,
			boolean disclosureEnabled, OpenAiTools.ParsedRequest toolsReq) {
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
			long start = System.currentTimeMillis();
			GenerationResult result = scheduler.submitAndWait(request);
			latencyCallback.accept(System.currentTimeMillis() - start);

			Map<String, Object> first = chunkRoot(completionId, created, modelId,
					List.of(chunkChoice(0, Map.of("role", "assistant", "content", ""), null)));
			if (disclosureEnabled) {
				first.put(AiDisclosure.FIELD_NAME, AiDisclosure.DISCLOSURE_TEXT);
			}
			writeSseChunk(writer, first);

			List<Map<String, Object>> toolCalls = parseToolCalls(result, toolsReq);
			String finish = OpenAiTools.finishReason(result.stopReason(), !toolCalls.isEmpty());
			if (!toolCalls.isEmpty()) {
				List<Map<String, Object>> indexed = new ArrayList<>(toolCalls.size());
				for (int i = 0; i < toolCalls.size(); i++) {
					Map<String, Object> tc = new LinkedHashMap<>(toolCalls.get(i));
					tc.put("index", i);
					indexed.add(tc);
				}
				Map<String, Object> delta = new LinkedHashMap<>();
				delta.put("tool_calls", indexed);
				writeSseChunk(writer, chunkRoot(completionId, created, modelId, List.of(chunkChoice(0, delta, finish))));
			} else {
				writeSseChunk(writer, chunkRoot(completionId, created, modelId,
						List.of(chunkChoice(0, Map.of("content", result.text() != null ? result.text() : ""), null))));
				writeSseChunk(writer, chunkRoot(completionId, created, modelId,
						List.of(chunkChoice(0, Map.of("content", ""), finish))));
			}
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

	public void handleListModels(Context ctx) {
		List<Map<String, Object>> data = modelRegistry.listModels().stream()
				.filter(m -> m.status() == ModelStatus.LOADED).map(this::toOpenAiModel).toList();
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

	/**
	 * Resolves and validates the requested model id, writing an OpenAI-style 503
	 * error response itself on failure. See {@link ModelIdResolver} for
	 * fallback/ambiguity rules.
	 *
	 * @return the resolved model id, or {@code null} if an error response was
	 *         already written to {@code ctx}
	 */
	private String resolveModelId(Context ctx, String requested) {
		ModelIdResolver.Resolution res = ModelIdResolver.resolve(modelRegistry, requested,
				ModelIdResolver.FallbackPolicy.SINGLE_MODEL_FALLBACK);
		if (res.isError()) {
			openAiError(ctx, 503, "service_unavailable_error", "service_unavailable", res.errorMessage(),
					requested == null || requested.isBlank() ? null : "model");
			return null;
		}
		if (res.warning() != null) {
			log.warning(res.warning());
		}
		return res.modelId();
	}

	SamplingParams buildSamplingParams(OaiChatCompletionRequest body, String[] stopStrings,
			OpenAiTools.ParsedRequest toolsReq) {
		SamplingParams p = SamplingParams.defaults();
		Integer maxTok = body.maxCompletionTokens() != null ? body.maxCompletionTokens() : body.maxTokens();
		if (maxTok != null)
			p = p.withMaxTokens(maxTok);
		// After the maximum, because the parameters reject a minimum above it and the
		// caller should hear that rather than have one of the two quietly clamped.
		if (body.minTokens() != null)
			p = p.withMinTokens(body.minTokens());
		if (body.temperature() != null)
			p = p.withTemperature(body.temperature().floatValue());
		if (body.topP() != null)
			p = p.withTopP(body.topP().floatValue());
		if (body.xJunoTopK() != null)
			p = p.withTopK(body.xJunoTopK());
		if (body.frequencyPenalty() != null)
			p = p.withRepetitionPenalty(
					OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(body.frequencyPenalty().floatValue()));
		if (body.presencePenalty() != null)
			p = p.withPresencePenalty(body.presencePenalty().floatValue());
		if (body.seed() != null)
			p = p.withSeed(body.seed());
		if (stopStrings != null && stopStrings.length > 0)
			p = p.withStopStrings(stopStrings);
		GbnfGrammar grammar = OpenAiResponseFormat.compile(body.responseFormat(), body.xJunoGrammar(), defaultGrammar);
		GbnfGrammar toolGrammar = OpenAiTools.grammar(toolsReq);
		p = p.withGrammar(toolGrammar != null ? toolGrammar : grammar);
		return p;
	}

	private static List<Map<String, Object>> parseToolCalls(GenerationResult result, OpenAiTools.ParsedRequest toolsReq) {
		if (toolsReq == null || !toolsReq.active())
			return List.of();
		return OpenAiTools.toOpenAiToolCalls(result.requestId(), ToolCallParser.parse(result.text()));
	}

	private static boolean hasPerRequestLoras(JsonNode node) {
		if (node == null || node.isNull() || node.isMissingNode())
			return false;
		if (node.isArray())
			return node.size() > 0;
		return true;
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
	public record OaiChatCompletionRequest(@JsonProperty("model") String model,
			@JsonProperty("messages") List<OaiMessage> messages, @JsonProperty("temperature") Double temperature,
			@JsonProperty("top_p") Double topP, @JsonProperty("max_tokens") Integer maxTokens,
			@JsonProperty("max_completion_tokens") Integer maxCompletionTokens,
			@JsonProperty("min_tokens") Integer minTokens, @JsonProperty("stream") Boolean stream,
			@JsonProperty("n") Integer n, @JsonProperty("frequency_penalty") Double frequencyPenalty,
			@JsonProperty("presence_penalty") Double presencePenalty, @JsonProperty("stop") JsonNode stop,
			@JsonProperty("seed") Long seed, @JsonProperty("response_format") JsonNode responseFormat,
			@JsonProperty("tools") JsonNode tools, @JsonProperty("tool_choice") JsonNode toolChoice,
			@JsonProperty("x_juno_priority") String xJunoPriority,
			@JsonProperty("x_juno_session_id") String xJunoSessionId, @JsonProperty("x_juno_top_k") Integer xJunoTopK,
			@JsonProperty("x_juno_disclosure") Boolean xJunoDisclosure, @JsonProperty("x_juno_loras") JsonNode xJunoLoras,
			@JsonProperty("x_juno_grammar") String xJunoGrammar) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	public record OaiMessage(@JsonProperty("role") String role, @JsonProperty("content") JsonNode content,
			@JsonProperty("tool_calls") JsonNode toolCalls, @JsonProperty("tool_call_id") String toolCallId) {
	}
}