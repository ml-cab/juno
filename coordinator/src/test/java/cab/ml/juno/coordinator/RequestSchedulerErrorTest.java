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

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
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
 * A request whose generation throws an {@link Error}, such as running out of Java
 * heap, ends with that error instead of never ending, on every dispatch path, and the
 * scheduler keeps serving. Before, each path caught {@code Exception} only: the
 * request's future was never completed, so the HTTP client waited until its own
 * timeout, and on the continuous schedule the engine thread died and every later
 * request hung too.
 */
@DisplayName("RequestScheduler: an Error in generation fails the request, never hangs it")
class RequestSchedulerErrorTest {

	private RequestScheduler scheduler;

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	@Test
	@DisplayName("single dispatch: the request fails with the error, and the next request succeeds")
	void singleDispatch() throws Exception {
		FailingPipeline pipeline = new FailingPipeline();
		scheduler = new RequestScheduler(10, loop(pipeline), BatchConfig.disabled());
		assertFailsThenRecovers(pipeline);
	}

	@Test
	@DisplayName("batched dispatch: the request fails with the error, and the next request succeeds")
	void batchedDispatch() throws Exception {
		FailingPipeline pipeline = new FailingPipeline();
		scheduler = new RequestScheduler(10, loop(pipeline), BatchConfig.of(4, 20));
		assertFailsThenRecovers(pipeline);
	}

	@Test
	@DisplayName("continuous schedule: the request fails with the error, and the engine keeps running")
	void continuousSchedule() throws Exception {
		FailingPipeline pipeline = new FailingPipeline();
		scheduler = new RequestScheduler(10, loop(pipeline), BatchConfig.of(4, 20),
				ServeScheduleOptions.parse("continuous"));
		assertFailsThenRecovers(pipeline);
	}

	private void assertFailsThenRecovers(FailingPipeline pipeline) throws Exception {
		CompletableFuture<GenerationResult> failing = scheduler.submit(req(), TokenConsumer.discard());
		assertThatThrownBy(() -> failing.get(10, TimeUnit.SECONDS))
				.as("the failing request ends within 10 s, with the error as its cause")
				.isInstanceOf(ExecutionException.class)
				.cause().isInstanceOf(OutOfMemoryError.class);

		pipeline.failing = false;
		GenerationResult next = scheduler.submit(req(), TokenConsumer.discard()).get(10, TimeUnit.SECONDS);
		assertThat(next.generatedTokens()).as("the next request is served").isPositive();
	}

	private static GenerationLoop loop(InferencePipeline pipeline) {
		return new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private static InferenceRequest req() {
		return InferenceRequest.of("model", List.of(ChatMessage.user("hi")), SamplingParams.defaults().withMaxTokens(2),
				RequestPriority.NORMAL);
	}

	/** Throws {@link OutOfMemoryError} from every forward pass while {@link #failing} is set. */
	private static final class FailingPipeline implements InferencePipeline {
		private final StubInferencePipeline stub = new StubInferencePipeline();
		volatile boolean failing = true;

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			if (failing)
				throw new OutOfMemoryError("Java heap space (simulated)");
			return stub.forward(requestId, tokens, startPos);
		}

		@Override
		public int vocabSize() {
			return stub.vocabSize();
		}
	}
}
