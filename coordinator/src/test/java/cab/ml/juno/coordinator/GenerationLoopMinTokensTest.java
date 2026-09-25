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

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * A request asking for a minimum number of tokens must reach it on every
 * schedule, not only on the one that happened to be tested.
 *
 * <p>The pipeline here always scores end-of-sequence highest, so a request with
 * no minimum stops with nothing generated. That makes the assertion exact rather
 * than statistical: with a minimum of N the sequence must produce exactly N
 * tokens, because the token is held back for N steps and wins immediately once
 * released.
 */
@DisplayName("Generation — minimum token floor across schedules")
class GenerationLoopMinTokensTest {

	private static final String MODEL = "tinyllama";
	private static final int MIN = 6;

	private SimpleTokenizer tokenizer;
	private AlwaysEosPipeline pipeline;
	private RequestScheduler scheduler;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		pipeline = new AlwaysEosPipeline(tokenizer.eosTokenId());
	}

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	/** Scores end-of-sequence highest on every step, whatever the position. */
	private static final class AlwaysEosPipeline implements InferencePipeline {
		private static final int VOCAB = 1000;
		private final int eos;

		AlwaysEosPipeline(int eos) {
			this.eos = eos;
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			float[] logits = new float[VOCAB];
			logits[eos] = 100.0f;
			return logits;
		}

		@Override
		public int vocabSize() {
			return VOCAB;
		}
	}

	private GenerationLoop loop() {
		return new GenerationLoop(tokenizer, Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private RequestScheduler scheduler(String schedule, int maxBatch) {
		scheduler = new RequestScheduler(32, loop(), BatchConfig.of(maxBatch, 20),
				ServeScheduleOptions.parse(schedule));
		return scheduler;
	}

	private static InferenceRequest request(String prompt, int minTokens) {
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(32).withMinTokens(minTokens);
		return InferenceRequest.of(MODEL, List.of(ChatMessage.user(prompt)), params, RequestPriority.NORMAL);
	}

	@Test
	@DisplayName("control: with no minimum, an immediate stop token ends the sequence at zero tokens")
	void withoutAMinimumTheSequenceStopsImmediately() {
		GenerationResult result = loop().generate(request("hi", 0), TokenConsumer.discard());

		assertThat(result.generatedTokens()).isZero();
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
	}

	@Test
	@DisplayName("single request: the floor holds the sequence open to exactly the minimum")
	void singleRequestReachesTheMinimum() {
		GenerationResult result = loop().generate(request("hi", MIN), TokenConsumer.discard());

		assertThat(result.generatedTokens()).isEqualTo(MIN);
	}

	@Test
	@DisplayName("static schedule, batched: every request in the batch reaches the minimum")
	void everyRequestInAStaticBatchReachesTheMinimum() throws Exception {
		RequestScheduler s = scheduler("static", 4);

		CompletableFuture<GenerationResult> a = s.submit(request("first question", MIN), TokenConsumer.discard());
		CompletableFuture<GenerationResult> b = s.submit(request("second question", MIN), TokenConsumer.discard());

		assertThat(a.get(30, TimeUnit.SECONDS).generatedTokens()).isEqualTo(MIN);
		assertThat(b.get(30, TimeUnit.SECONDS).generatedTokens()).isEqualTo(MIN);
	}

	@Test
	@DisplayName("static schedule: a request without a minimum is unaffected by one that has it")
	void aMinimumOnOneRequestDoesNotLeakToAnother() throws Exception {
		RequestScheduler s = scheduler("static", 4);

		CompletableFuture<GenerationResult> floored = s.submit(request("floored", MIN), TokenConsumer.discard());
		CompletableFuture<GenerationResult> free = s.submit(request("free", 0), TokenConsumer.discard());

		assertThat(floored.get(30, TimeUnit.SECONDS).generatedTokens()).isEqualTo(MIN);
		assertThat(free.get(30, TimeUnit.SECONDS).generatedTokens()).as("per-request, not per-batch").isZero();
	}

	@Test
	@DisplayName("continuous schedule: the floor holds across mixed prefill and decode steps")
	void continuousScheduleReachesTheMinimum() throws Exception {
		RequestScheduler s = scheduler("continuous", 4);

		GenerationResult result = s.submit(request("hi", MIN), TokenConsumer.discard()).get(30, TimeUnit.SECONDS);

		assertThat(result.generatedTokens()).isEqualTo(MIN);
	}

	@Test
	@DisplayName("continuous schedule: a request without a minimum still stops immediately")
	void continuousScheduleWithoutAMinimumStopsImmediately() throws Exception {
		RequestScheduler s = scheduler("continuous", 4);

		GenerationResult result = s.submit(request("hi", 0), TokenConsumer.discard()).get(30, TimeUnit.SECONDS);

		assertThat(result.generatedTokens()).isZero();
	}

	@Test
	@DisplayName("the minimum never overrides the maximum")
	void theMaximumStillWins() {
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(3).withMinTokens(3);
		GenerationResult result = loop().generate(
				InferenceRequest.of(MODEL, List.of(ChatMessage.user("hi")), params, RequestPriority.NORMAL),
				TokenConsumer.discard());

		assertThat(result.generatedTokens()).isEqualTo(3);
	}
}
