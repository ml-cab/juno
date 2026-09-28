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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * A minimum token count must hold against every way the model itself can end a
 * turn, not only the end-of-sequence id: a chat turn-marker id the vocabulary
 * declares, and a role header spelled out as text. A stop the caller asked for
 * still ends the request below its minimum, as the published contract says.
 *
 * <p>Each case runs on all three generation paths: a single request, a static
 * batch, and the continuous engine. The pipeline scores its chosen ending token
 * highest on every step and a plain filler token second, so the counts are
 * exact rather than statistical.
 */
@DisplayName("Generation — minimum token floor against turn markers")
class GenerationLoopMinTokensTurnMarkerTest {

	private static final String MODEL = "tinyllama";
	private static final int MIN = 5;
	private static final int VOCAB = 1000;
	private static final int FILLER = 400;
	/** A vocabulary turn-marker id, e.g. {@code <|im_start|>}. */
	private static final int TURN_ID = 500;
	/** A plain token whose text is a role header, as TinyLlama spells {@code <|user|>}. */
	private static final int HEADER_TEXT = 450;

	private RequestScheduler scheduler;

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "single", "static", "continuous" })
	@DisplayName("a turn-marker id proposed from the first step is held back until the minimum")
	void turnMarkerIdIsHeldBackUntilTheMinimum(String path) throws Exception {
		GenerationResult r = run(path, TURN_ID, params(MIN));

		assertThat(r.generatedTokens()).isEqualTo(MIN);
		assertThat(r.tokenIds()).containsOnly(FILLER);
		assertThat(r.stopReason()).isEqualTo(GenerationResult.StopReason.STOP_TOKEN);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "single", "static", "continuous" })
	@DisplayName("a turn-marker id the caller asked to stop on still ends the request below its minimum")
	void callerRequestedStopIdEndsBelowTheMinimum(String path) throws Exception {
		GenerationResult r = run(path, TURN_ID, params(MIN).withStopTokenIds(TURN_ID));

		assertThat(r.generatedTokens()).isZero();
		assertThat(r.stopReason()).isEqualTo(GenerationResult.StopReason.STOP_TOKEN);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "single", "static", "continuous" })
	@DisplayName("a role header in the decoded text does not end the request below its minimum")
	void roleHeaderTextDoesNotEndBelowTheMinimum(String path) throws Exception {
		List<String> streamed = new ArrayList<>();
		GenerationResult r = run(path, HEADER_TEXT, params(MIN), streamed);

		assertThat(r.generatedTokens()).isEqualTo(MIN);
		assertThat(r.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
		assertThat(r.text()).as("text held back below the minimum is emitted, not dropped")
				.isEqualTo("<|user|>".repeat(MIN));
		assertThat(String.join("", streamed)).isEqualTo(r.text());
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "single", "static", "continuous" })
	@DisplayName("control: without a minimum both ending signals still end the request at once")
	void withoutAMinimumTheModelEndsAtOnce(String path) throws Exception {
		assertThat(run(path, TURN_ID, params(0)).generatedTokens()).isZero();
		assertThat(run(path, HEADER_TEXT, params(0)).generatedTokens()).isZero();
	}

	// ── harness ──────────────────────────────────────────────────────────────

	private static SamplingParams params(int minTokens) {
		return SamplingParams.deterministic().withMaxTokens(20).withMinTokens(minTokens);
	}

	private GenerationResult run(String path, int endingToken, SamplingParams params) throws Exception {
		return run(path, endingToken, params, new ArrayList<>());
	}

	private GenerationResult run(String path, int endingToken, SamplingParams params, List<String> streamed)
			throws Exception {
		if (scheduler != null) {
			scheduler.shutdown();
			scheduler = null;
		}
		GenerationLoop loop = new GenerationLoop(new MarkerTokenizer(), Sampler.create(),
				new FavouringPipeline(endingToken),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		InferenceRequest req = InferenceRequest.of(MODEL, List.of(ChatMessage.user("hi")), params,
				RequestPriority.NORMAL);
		TokenConsumer consumer = (piece, id, step) -> streamed.add(piece);
		if ("single".equals(path))
			return loop.generate(req, consumer);

		scheduler = new RequestScheduler(32, loop, BatchConfig.of(4, 20), ServeScheduleOptions.parse(path));
		CompletableFuture<GenerationResult> a = scheduler.submit(req, consumer);
		// A second request so the static schedule takes its batched path.
		CompletableFuture<GenerationResult> b = scheduler.submit(
				InferenceRequest.of(MODEL, List.of(ChatMessage.user("other")), params, RequestPriority.NORMAL),
				TokenConsumer.discard());
		GenerationResult result = a.get(30, TimeUnit.SECONDS);
		b.get(30, TimeUnit.SECONDS);
		return result;
	}

	/** Scores one ending token highest and a filler second, at every position. */
	private static final class FavouringPipeline implements InferencePipeline {
		private final int ending;

		FavouringPipeline(int ending) {
			this.ending = ending;
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			float[] logits = new float[VOCAB];
			logits[ending] = 100.0f;
			logits[FILLER] = 50.0f;
			return logits;
		}

		@Override
		public int vocabSize() {
			return VOCAB;
		}
	}

	/** Declares one turn-marker id and one token that decodes to a role header. */
	private static final class MarkerTokenizer implements Tokenizer {
		private final SimpleTokenizer delegate = new SimpleTokenizer();

		@Override
		public int[] chatTurnTokenIds() {
			return new int[] { TURN_ID };
		}

		@Override
		public int[] encode(String text) {
			return delegate.encode(text);
		}

		@Override
		public String decode(int[] ids) {
			return delegate.decode(ids);
		}

		@Override
		public String decodeToken(int tokenId) {
			if (tokenId == HEADER_TEXT)
				return "<|user|>";
			if (tokenId == FILLER)
				return "x";
			return delegate.decodeToken(tokenId);
		}

		@Override
		public int bosTokenId() {
			return delegate.bosTokenId();
		}

		@Override
		public int eosTokenId() {
			return delegate.eosTokenId();
		}

		@Override
		public int padTokenId() {
			return delegate.padTokenId();
		}

		@Override
		public int vocabSize() {
			return VOCAB;
		}

		@Override
		public String modelType() {
			return delegate.modelType();
		}

		@Override
		public boolean isReady() {
			return true;
		}
	}
}
