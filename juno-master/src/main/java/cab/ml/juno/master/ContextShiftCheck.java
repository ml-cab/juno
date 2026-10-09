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

package cab.ml.juno.master;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

import cab.ml.juno.coordinator.BatchConfig;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PrefillMode;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * Check 10: a request that opts in to context shifting runs past the context
 * limit on the real model, on both schedules. Reaching a model's real limit
 * (32768 positions for most) is hours of generation, so the pipeline reports a
 * limit of {@value #LIMIT} positions; the handlers underneath shift and re-rotate
 * their KV exactly as they would at the real one. A minimum token count holds the
 * request open past the limit whatever the model would rather do.
 */
final class ContextShiftCheck {

	static final int LIMIT = 192;
	private static final int PROMPT_TOKENS = 96;
	private static final int GENERATED = 200;

	private ContextShiftCheck() {
	}

	/** Reports {@value #LIMIT} as its limit and records what reaches the real pipeline. */
	private static final class LimitedPipeline implements InferencePipeline {
		private final InferencePipeline inner;
		final AtomicInteger shifts = new AtomicInteger();
		final AtomicInteger maxPosition = new AtomicInteger(-1);

		LimitedPipeline(InferencePipeline inner) {
			this.inner = inner;
		}

		private void saw(int lastPosition) {
			maxPosition.accumulateAndGet(lastPosition, Math::max);
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			saw(startPos);
			return inner.forward(requestId, tokens, startPos);
		}

		@Override
		public float[][] forwardBatch(List<String> ids, List<int[]> tokens, List<Integer> positions) {
			positions.forEach(this::saw);
			return inner.forwardBatch(ids, tokens, positions);
		}

		@Override
		public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
			saw(startPosition + newTokens.length - 1);
			inner.prefillBatch(requestId, newTokens, startPosition);
		}

		@Override
		public float[][] verifyDraft(String requestId, int[] draftTokens, int startPosition) {
			saw(startPosition + draftTokens.length - 1);
			return inner.verifyDraft(requestId, draftTokens, startPosition);
		}

		@Override
		public int vocabSize() {
			return inner.vocabSize();
		}

		@Override
		public void evict(String requestId) {
			inner.evict(requestId);
		}

		@Override
		public int contextLimit() {
			return Math.min(LIMIT, inner.contextLimit());
		}

		@Override
		public boolean supportsContextShift() {
			return inner.supportsContextShift();
		}

		@Override
		public void shiftKv(String requestId, int seqLen, int keep, int discard) {
			shifts.incrementAndGet();
			inner.shiftKv(requestId, seqLen, keep, discard);
		}
	}

	static String run(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer) throws IOException {
		InferenceRequest request = InferenceRequest.of("model",
				List.of(ChatMessage.system("You are a careful assistant. Answer in complete sentences."),
						ChatMessage.user(ModelLiveChecks.longPrompt(tokenizer, PROMPT_TOKENS))),
				SamplingParams.deterministic().withMaxTokens(GENERATED).withMinTokens(GENERATED),
				RequestPriority.NORMAL).withContextShift(true);
		String savedSchedule = System.getProperty(ServeScheduleOptions.ENV);
		GpuContext gpu = CudaAvailability.isAvailable() ? GpuContext.init(0) : null;
		try {
			MatVec backend = gpu != null ? gpu.createMatVec() : ForwardPassHandlerLoader.selectBackend();

			System.setProperty(ServeScheduleOptions.ENV, "static");
			ForwardPassHandler dense = ModelLiveChecks.loadSingleShard(modelPath, cfg, backend);
			LimitedPipeline staticPipe = new LimitedPipeline(ModelLiveChecks.singleShardPipeline(cfg, dense));
			GenerationResult stat;
			try {
				stat = new GenerationLoop(tokenizer, Sampler.create(), staticPipe, ModelLiveChecks.newKvCache(4096),
						PrefillMode.BATCHED, 64).generate(request, TokenConsumer.discard());
			} finally {
				dense.releaseGpuResources();
			}

			System.setProperty(ServeScheduleOptions.ENV, "continuous");
			ForwardPassHandler paged = ModelLiveChecks.loadSingleShard(modelPath, cfg, backend);
			LimitedPipeline contPipe = new LimitedPipeline(ModelLiveChecks.singleShardPipeline(cfg, paged));
			GenerationResult cont;
			RequestScheduler scheduler = null;
			try {
				scheduler = new RequestScheduler(16,
						new GenerationLoop(tokenizer, Sampler.create(), contPipe, ModelLiveChecks.newKvCache(4096),
								PrefillMode.BATCHED, 32),
						BatchConfig.disabled(), ServeScheduleOptions.of(ServeScheduleOptions.Mode.CONTINUOUS));
				cont = scheduler.submitAndWait(request);
			} finally {
				if (scheduler != null)
					scheduler.shutdown();
				paged.releaseGpuResources();
			}

			String detail = String.format(Locale.ROOT,
					"backend=%s limit=%d prompt_tokens=%d generated static=%d continuous=%d shifts static=%d"
							+ " continuous=%d max_position static=%d continuous=%d",
					gpu != null ? "cuda" : "cpu", LIMIT, stat.promptTokens(), stat.generatedTokens(),
					cont.generatedTokens(), staticPipe.shifts.get(), contPipe.shifts.get(),
					staticPipe.maxPosition.get(), contPipe.maxPosition.get());
			ModelLiveChecks.check(stat.generatedTokens() == GENERATED && cont.generatedTokens() == GENERATED,
					"both schedules must generate all " + GENERATED + " tokens (" + detail + ")");
			ModelLiveChecks.check(staticPipe.shifts.get() >= 1 && contPipe.shifts.get() >= 1,
					"both schedules must shift at least once (" + detail + ")");
			ModelLiveChecks.check(staticPipe.maxPosition.get() < LIMIT && contPipe.maxPosition.get() < LIMIT,
					"no position may reach the limit of " + LIMIT + " (" + detail + ")");
			ModelLiveChecks.check(!ModelLiveChecks.cleanText(stat.text()).isEmpty()
					&& !ModelLiveChecks.cleanText(cont.text()).isEmpty(), "both responses must be non-empty (" + detail
							+ ")");
			return detail;
		} finally {
			if (savedSchedule == null)
				System.clearProperty(ServeScheduleOptions.ENV);
			else
				System.setProperty(ServeScheduleOptions.ENV, savedSchedule);
			if (gpu != null)
				gpu.close();
		}
	}
}
