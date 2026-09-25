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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.coordinator.InferenceApiServer.ApiSampling;
import cab.ml.juno.coordinator.OpenAiChatHandler.OaiChatCompletionRequest;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * A minimum token count has to arrive through the surfaces callers actually use,
 * not only through the sampler that implements it. A field that parses but never
 * reaches the sampling parameters is the silent no-op this project treats as a
 * defect rather than a gap, so each surface is checked end to end from its own
 * request shape.
 */
@DisplayName("min_tokens on the request surfaces")
class MinTokensRequestFieldTest {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private OpenAiChatHandler chatHandler;
	private InferenceApiServer nativeServer;

	@BeforeEach
	void setUp() {
		var kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
		var loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), new StubInferencePipeline(), kvCache);
		var scheduler = new RequestScheduler(8, loop);
		var registry = new ModelRegistry(ShardPlanner.create());
		chatHandler = new OpenAiChatHandler(scheduler, registry, ms -> {
		});
		nativeServer = new InferenceApiServer(scheduler, registry, "BE");
	}

	private static OaiChatCompletionRequest chatBody(String json) throws Exception {
		return MAPPER.readValue(json, OaiChatCompletionRequest.class);
	}

	@Test
	@DisplayName("chat completions accepts min_tokens and carries it into the sampling parameters")
	void chatCompletionsCarriesMinTokens() throws Exception {
		OaiChatCompletionRequest body = chatBody("""
				{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"min_tokens":64}
				""");

		assertThat(body.minTokens()).isEqualTo(64);
		assertThat(chatHandler.buildSamplingParams(body, new String[0], null).minTokens()).isEqualTo(64);
	}

	@Test
	@DisplayName("chat completions without min_tokens leaves the model free to stop")
	void chatCompletionsDefaultsToNoMinimum() throws Exception {
		OaiChatCompletionRequest body = chatBody("""
				{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":64}
				""");

		assertThat(body.minTokens()).isNull();
		assertThat(chatHandler.buildSamplingParams(body, new String[0], null).minTokens()).isZero();
	}

	@Test
	@DisplayName("a minimum above the maximum is rejected rather than silently clamped")
	void aMinimumAboveTheMaximumIsRejected() throws Exception {
		OaiChatCompletionRequest body = chatBody("""
				{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"min_tokens":64}
				""");

		assertThatThrownBy(() -> chatHandler.buildSamplingParams(body, new String[0], null))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("minTokens");
	}

	@Test
	@DisplayName("the native surface carries minTokens too")
	void nativeSurfaceCarriesMinTokens() {
		var params = nativeServer.buildSamplingParams(new ApiSampling(0.0f, 1, 1.0f, 64, 64, "NORMAL"));

		assertThat(params.minTokens()).isEqualTo(64);
		assertThat(params.maxTokens()).isEqualTo(64);
	}

	@Test
	@DisplayName("the native surface without a minimum leaves the model free to stop")
	void nativeSurfaceDefaultsToNoMinimum() {
		assertThat(nativeServer.buildSamplingParams(new ApiSampling(0.0f, 1, 1.0f, 64, null, "NORMAL")).minTokens())
				.isZero();
	}

	@Test
	@DisplayName("the native surface rejects a minimum above the maximum")
	void nativeSurfaceRejectsAMinimumAboveTheMaximum() {
		assertThatThrownBy(() -> nativeServer.buildSamplingParams(new ApiSampling(0.0f, 1, 1.0f, 8, 64, "NORMAL")))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("minTokens");
	}
}
