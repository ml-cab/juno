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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.node.EmbeddingPooling;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.PoolingMode;
import cab.ml.juno.registry.ModelIdResolver;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.tokenizer.Tokenizer;
import io.javalin.http.Context;

/**
 * OpenAI-compatible {@code POST /v1/embeddings}.
 *
 * <p>Bypasses {@link RequestScheduler} / {@link GenerationLoop}'s sampler-driven
 * token generation entirely: extracts the RMS/LayerNorm-normalized hidden state
 * at every prompt position via {@link InferencePipeline#embedTokens} and reduces
 * it to one vector per input with {@link EmbeddingPooling}. Runs directly on the
 * Javalin request's own virtual thread (see {@link InferenceApiServer} class doc)
 * — v1 does not share the chat completions queue-depth limit or 429 semantics;
 * see {@code docs/howto.md}.
 *
 * <p>Disabled by default. Requires the server to be started with
 * {@code --embeddings}; distributed pipelines (cluster / tensor-parallel /
 * pipeline-parallel) fail closed with a clear error instead of silently
 * returning a wrong vector — {@link InferencePipeline#embedTokens} throws
 * {@link UnsupportedOperationException} there, which this handler turns into a
 * 400 rather than a bare 500.
 */
public final class EmbeddingsHandler {

	private static final Logger log = Logger.getLogger(EmbeddingsHandler.class.getName());
	private static final ObjectMapper JSON = new ObjectMapper();

	private final RequestScheduler scheduler;
	private final ModelRegistry modelRegistry;
	private final boolean enabled;
	private final PoolingMode defaultPooling;
	private final AtomicBoolean warnedLastPooling = new AtomicBoolean(false);

	public EmbeddingsHandler(RequestScheduler scheduler, ModelRegistry modelRegistry, boolean enabled,
			PoolingMode defaultPooling) {
		this.scheduler = scheduler;
		this.modelRegistry = modelRegistry;
		this.enabled = enabled;
		this.defaultPooling = defaultPooling;
	}

	public void handleEmbeddings(Context ctx) {
		if (!enabled) {
			error(ctx, 400, "invalid_request_error", "embeddings_disabled",
					"Embeddings are disabled on this server; start with --embeddings to enable POST /v1/embeddings.",
					null);
			return;
		}

		EmbeddingsRequest body;
		try {
			body = JSON.readValue(ctx.body(), EmbeddingsRequest.class);
		} catch (Exception e) {
			error(ctx, 400, "invalid_request_error", "invalid_request", "Invalid request body: " + e.getMessage(),
					null);
			return;
		}

		List<String> inputs;
		try {
			inputs = parseInput(body.input());
		} catch (IllegalArgumentException e) {
			error(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "input");
			return;
		}

		PoolingMode pooling;
		try {
			pooling = body.xJunoPooling() != null ? PoolingMode.parse(body.xJunoPooling()) : defaultPooling;
		} catch (IllegalArgumentException e) {
			error(ctx, 400, "invalid_request_error", "invalid_request", e.getMessage(), "x_juno_pooling");
			return;
		}
		if (pooling == PoolingMode.LAST && warnedLastPooling.compareAndSet(false, true)) {
			log.warning("POST /v1/embeddings using last-token pooling: on chat-tuned (non embedding-trained) "
					+ "models this tends to under-represent earlier prompt tokens; prefer --pooling mean.");
		}

		ModelIdResolver.Resolution res = ModelIdResolver.resolve(modelRegistry, body.model(),
				ModelIdResolver.FallbackPolicy.SINGLE_MODEL_FALLBACK);
		if (res.isError()) {
			error(ctx, 503, "service_unavailable_error", "service_unavailable", res.errorMessage(),
					body.model() == null || body.model().isBlank() ? null : "model");
			return;
		}
		if (res.warning() != null)
			log.warning(res.warning());
		String modelId = res.modelId();

		Tokenizer tokenizer = scheduler.generationLoop().tokenizer();
		InferencePipeline pipeline = scheduler.generationLoop().pipeline();

		List<float[]> embeddings = new ArrayList<>(inputs.size());
		long promptTokens = 0;
		try {
			for (String input : inputs) {
				int[] tokens = tokenizer.encode(input);
				if (tokens.length == 0) {
					error(ctx, 400, "invalid_request_error", "invalid_request", "input must not tokenize to empty",
							"input");
					return;
				}
				promptTokens += tokens.length;
				float[][] hidden = pipeline.embedTokens(UUID.randomUUID().toString(), tokens);
				embeddings.add(EmbeddingPooling.pool(hidden, pooling));
			}
		} catch (UnsupportedOperationException e) {
			error(ctx, 400, "invalid_request_error", "embeddings_unsupported", e.getMessage(), null);
			return;
		}

		List<Map<String, Object>> data = new ArrayList<>(embeddings.size());
		for (int i = 0; i < embeddings.size(); i++) {
			float[] v = embeddings.get(i);
			List<Float> boxed = new ArrayList<>(v.length);
			for (float f : v)
				boxed.add(f);
			data.add(Map.of("object", "embedding", "index", i, "embedding", boxed));
		}

		Map<String, Object> usage = Map.of("prompt_tokens", promptTokens, "total_tokens", promptTokens);
		Map<String, Object> responseBody = new LinkedHashMap<>();
		responseBody.put("object", "list");
		responseBody.put("data", data);
		responseBody.put("model", modelId);
		responseBody.put("usage", usage);
		ctx.json(responseBody);
	}

	/**
	 * OpenAI {@code input} accepts a single string or an array of strings. Jackson
	 * gives us the raw {@link JsonNode} so we can accept either shape.
	 */
	private static List<String> parseInput(JsonNode input) {
		if (input == null || input.isNull())
			throw new IllegalArgumentException("input is required");
		if (input.isTextual()) {
			String s = input.asText();
			if (s.isBlank())
				throw new IllegalArgumentException("input must not be blank");
			return List.of(s);
		}
		if (input.isArray()) {
			if (input.isEmpty())
				throw new IllegalArgumentException("input array must not be empty");
			List<String> out = new ArrayList<>(input.size());
			for (JsonNode item : input) {
				if (!item.isTextual() || item.asText().isBlank())
					throw new IllegalArgumentException("every input array element must be a non-blank string");
				out.add(item.asText());
			}
			return out;
		}
		throw new IllegalArgumentException("input must be a string or an array of strings");
	}

	private static void error(Context ctx, int status, String type, String code, String message, String param) {
		Map<String, Object> err = new LinkedHashMap<>();
		err.put("message", message);
		err.put("type", type);
		err.put("code", code);
		if (param != null)
			err.put("param", param);
		ctx.status(status).json(Map.of("error", err));
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	public record EmbeddingsRequest(String model, JsonNode input, @JsonProperty("encoding_format") String encodingFormat,
			@JsonProperty("x_juno_pooling") String xJunoPooling) {
	}
}
