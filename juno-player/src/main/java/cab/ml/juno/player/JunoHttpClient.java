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

package cab.ml.juno.player;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.tokenizer.ChatMessage;

/**
 * JDK {@link HttpClient} wrapper for the local Juno REST API (blocking and streaming).
 */
public final class JunoHttpClient {

	private final HttpClient http = HttpClient.newHttpClient();
	private final String baseV1;
	private final ObjectMapper json = new ObjectMapper();

	public JunoHttpClient(URI base) {
		String s = base.toString();
		if (s.endsWith("/"))
			s = s.substring(0, s.length() - 1);
		baseV1 = s.endsWith("/v1") ? s : s + "/v1";
	}

	/**
	 * As {@link #blockingInference(String, List, Integer)}, with a minimum number of
	 * tokens the response must contain before an end-of-sequence token may end it.
	 *
	 * @param minTokens minimum tokens to generate; null or 0 leaves the model free
	 *                  to stop whenever it likes
	 */
	public String blockingInference(String modelId, List<ChatMessage> messages, Integer maxTokens, Integer minTokens)
			throws Exception {
		String body = postJson(baseV1 + "/inference",
				buildInferenceJsonUnchecked(modelId, messages, maxTokens, minTokens));
		return json.readTree(body).path("text").asText("");
	}

	public String blockingInference(String modelId, List<ChatMessage> messages, Integer maxTokens) throws Exception {
		return blockingInference(modelId, messages, maxTokens, null);
	}

	/**
	 * Native SSE from {@code POST /v1/inference/stream} — JSON lines with
	 * {@code token}, {@code isComplete}, {@code finishReason}.
	 */
	public Flow.Publisher<String> streamingInference(String modelId, List<ChatMessage> messages, Integer maxTokens) {
		return startSsePublisher(baseV1 + "/inference/stream", buildInferenceJsonUnchecked(modelId, messages, maxTokens),
				line -> parseNativeToken(line, json));
	}

	/**
	 * As {@link #blockingOpenAiChat(String, List, Integer, Float)}, with a minimum
	 * number of tokens the response must contain.
	 *
	 * @param minTokens minimum tokens to generate; null or 0 leaves the model free
	 *                  to stop whenever it likes
	 */
	public String blockingOpenAiChat(String model, List<ChatMessage> messages, Integer maxTokens, Float temperature,
			Integer minTokens) throws Exception {
		String body = postJson(baseV1 + "/chat/completions",
				buildOpenAiChatJsonUnchecked(model, messages, false, maxTokens, temperature, minTokens));
		return json.readTree(body).path("choices").path(0).path("message").path("content").asText("");
	}

	public String blockingOpenAiChat(String model, List<ChatMessage> messages, Integer maxTokens, Float temperature)
			throws Exception {
		return blockingOpenAiChat(model, messages, maxTokens, temperature, null);
	}

	/** One JSON POST, returning the response body and failing on any non-200. */
	private String postJson(String url, String payload) throws Exception {
		HttpRequest req = HttpRequest.newBuilder(URI.create(url)).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8)).build();
		HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
		if (res.statusCode() != 200)
			throw new IllegalStateException("HTTP " + res.statusCode() + ": " + res.body());
		return res.body();
	}

	/** OpenAI-compatible SSE ({@code choices[0].delta.content}). */
	public Flow.Publisher<String> streamingOpenAiChat(String model, List<ChatMessage> messages, Integer maxTokens,
			Float temperature) {
		String payload = buildOpenAiChatJsonUnchecked(model, messages, true, maxTokens, temperature);
		return startSsePublisher(baseV1 + "/chat/completions", payload, line -> parseOpenAiDelta(line, json));
	}

	private Flow.Publisher<String> startSsePublisher(String url, String jsonBody,
			java.util.function.Function<String, String> lineToPiece) {
		Executor executor = Executors.newVirtualThreadPerTaskExecutor();
		SubmissionPublisher<String> publisher = new SubmissionPublisher<>(executor, Flow.defaultBufferSize());
		HttpRequest req = HttpRequest.newBuilder(URI.create(url)).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(jsonBody, StandardCharsets.UTF_8)).build();
		http.sendAsync(req, HttpResponse.BodyHandlers.ofInputStream()).whenComplete((resp, err) -> {
			if (err != null) {
				publisher.closeExceptionally(err);
				return;
			}
			if (resp.statusCode() != 200) {
				try (BufferedReader br = new BufferedReader(
						new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
					String body = br.lines().reduce("", (a, b) -> a + b + "\n");
					publisher.closeExceptionally(new IllegalStateException("HTTP " + resp.statusCode() + ": " + body));
				} catch (Exception e) {
					publisher.closeExceptionally(e);
				}
				return;
			}
			try (BufferedReader reader = new BufferedReader(
					new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
				String line;
				while ((line = reader.readLine()) != null) {
					if (!line.startsWith("data: "))
						continue;
					String data = line.substring(6).strip();
					if ("[DONE]".equals(data))
						break;
					String piece = lineToPiece.apply(data);
					if (piece != null && !piece.isEmpty())
						publisher.submit(piece);
				}
				publisher.close();
			} catch (Exception e) {
				publisher.closeExceptionally(e);
			}
		});
		return publisher;
	}

	private static String parseNativeToken(String data, ObjectMapper json) {
		try {
			JsonNode n = json.readTree(data);
			if (n.path("isComplete").asBoolean(false))
				return null;
			return n.path("token").asText("");
		} catch (Exception e) {
			return null;
		}
	}

	private static String parseOpenAiDelta(String data, ObjectMapper json) {
		try {
			JsonNode n = json.readTree(data);
			JsonNode delta = n.path("choices").path(0).path("delta");
			if (delta.hasNonNull("content"))
				return delta.path("content").asText("");
		} catch (Exception e) {
			return null;
		}
		return null;
	}

	private String buildInferenceJsonUnchecked(String modelId, List<ChatMessage> messages, Integer maxTokens) {
		return buildInferenceJsonUnchecked(modelId, messages, maxTokens, null);
	}

	private String buildInferenceJsonUnchecked(String modelId, List<ChatMessage> messages, Integer maxTokens,
			Integer minTokens) {
		try {
			ObjectNode root = json.createObjectNode();
			ArrayNode arr = root.putArray("messages");
			for (ChatMessage m : messages) {
				ObjectNode o = arr.addObject();
				o.put("role", m.role());
				o.put("content", m.content());
			}
			if (modelId != null && !modelId.isBlank())
				root.put("modelId", modelId);
			if (maxTokens != null || minTokens != null) {
				ObjectNode sampling = root.putObject("sampling");
				if (maxTokens != null)
					sampling.put("maxTokens", maxTokens);
				if (minTokens != null)
					sampling.put("minTokens", minTokens);
			}
			return json.writeValueAsString(root);
		} catch (Exception e) {
			throw new IllegalStateException(e);
		}
	}

	private String buildOpenAiChatJsonUnchecked(String model, List<ChatMessage> messages, boolean stream,
			Integer maxTokens, Float temperature) {
		return buildOpenAiChatJsonUnchecked(model, messages, stream, maxTokens, temperature, null);
	}

	private String buildOpenAiChatJsonUnchecked(String model, List<ChatMessage> messages, boolean stream,
			Integer maxTokens, Float temperature, Integer minTokens) {
		try {
			ObjectNode root = json.createObjectNode();
			if (model != null && !model.isBlank())
				root.put("model", model);
			ArrayNode arr = root.putArray("messages");
			for (ChatMessage m : messages) {
				ObjectNode o = arr.addObject();
				o.put("role", m.role());
				o.put("content", m.content());
			}
			root.put("stream", stream);
			if (maxTokens != null)
				root.put("max_completion_tokens", maxTokens);
			if (minTokens != null)
				root.put("min_tokens", minTokens);
			if (temperature != null)
				root.put("temperature", temperature);
			return json.writeValueAsString(root);
		} catch (Exception e) {
			throw new IllegalStateException(e);
		}
	}
}
