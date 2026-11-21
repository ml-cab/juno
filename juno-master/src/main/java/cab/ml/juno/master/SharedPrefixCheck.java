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
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import cab.ml.juno.coordinator.BatchConfig;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PrefillMode;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.kvcache.SharedPrefixOptions;
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
 * Check 12: requests from different clients start from a finished request's prompt, on both schedules, in this
 * process. With shared prompts on, a server keeps finished requests' prompts, and a later request whose prompt begins
 * with one starts from its KV instead of computing those positions. For each schedule, on one loaded model: a first
 * request alone, then two requests that open with the same system prompt submitted together (the static schedule
 * batches them, the continuous one runs them in one running set), first with sharing off and then with it on. With
 * sharing on, both later requests must start from the first one's kept prompt (two shared-prefix hits), and every reply
 * must equal the reply computed with sharing off, token for token. Greedy.
 *
 * <p>On CUDA both runs prefill one position at a time: GPU prefill rounds a position by the width of the window it is
 * computed in, and starting from a shared prefix moves the windows, so only one-position windows compare the sharing
 * itself. A cluster pipeline does not share (each node keeps its own layers' KV), so this check runs in process only.
 */
final class SharedPrefixCheck {

	private static final SamplingParams GREEDY = SamplingParams.deterministic().withMaxTokens(24);
	private static final String SYSTEM = "You are a patient tutor for a class of curious students. Answer every question"
			+ " in two short sentences, use plain words, and give one concrete example from everyday life.";
	private static final String[] QUESTIONS = { "Why is the sky blue?", "How does a refrigerator keep food cold?",
			"What makes a rainbow appear after rain?" };
	/** Prompt positions kept for sharing in the run with it on. */
	private static final String BUDGET_ON = "4096";
	private static final long WAIT_MINUTES = 30;

	private SharedPrefixCheck() {
	}

	/** One run's three replies, in question order, and its shared-prefix counters. */
	private record Run(List<GenerationResult> replies, long hits, long sharedTokens) {
	}

	static String run(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer, boolean cuda) throws IOException {
		String savedSchedule = System.getProperty(ServeScheduleOptions.ENV);
		String savedBudget = System.getProperty(SharedPrefixOptions.ENV);
		GpuContext gpu = cuda ? GpuContext.init(0) : null;
		try {
			MatVec backend = gpu != null ? gpu.createMatVec() : ForwardPassHandlerLoader.selectBackend();
			int window = gpu != null ? 1 : 32;
			String stat = schedule(modelPath, cfg, tokenizer, backend, window, ServeScheduleOptions.Mode.STATIC);
			String cont = schedule(modelPath, cfg, tokenizer, backend, window, ServeScheduleOptions.Mode.CONTINUOUS);
			return "backend=" + (gpu != null ? "cuda" : "cpu") + " prefill_window=" + window + "; " + stat + "; " + cont;
		} finally {
			restore(ServeScheduleOptions.ENV, savedSchedule);
			restore(SharedPrefixOptions.ENV, savedBudget);
			if (gpu != null)
				gpu.close();
		}
	}

	private static String schedule(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer, MatVec backend,
			int window, ServeScheduleOptions.Mode mode) throws IOException {
		String label = mode.name().toLowerCase(Locale.ROOT);
		System.setProperty(ServeScheduleOptions.ENV, label);
		ForwardPassHandler handler = ModelLiveChecks.loadSingleShard(modelPath, cfg, backend);
		try {
			InferencePipeline pipeline = ModelLiveChecks.singleShardPipeline(cfg, handler);
			Run off = submit(tokenizer, pipeline, window, mode, "0");
			Run on = submit(tokenizer, pipeline, window, mode, BUDGET_ON);
			String detail = String.format(Locale.ROOT,
					"%s: shared-prefix hits off=%d on=%d, positions shared on=%d, prompt tokens %d/%d/%d", label,
					off.hits(), on.hits(), on.sharedTokens(), on.replies().get(0).promptTokens(),
					on.replies().get(1).promptTokens(), on.replies().get(2).promptTokens());
			ModelLiveChecks.check(off.hits() == 0, "with sharing off, shared-prefix hits were counted (" + detail + ")");
			ModelLiveChecks.check(on.hits() == 2,
					"both later requests must start from the first one's kept prompt (" + detail + ")");
			for (int i = 0; i < QUESTIONS.length; i++) {
				GenerationResult a = off.replies().get(i), b = on.replies().get(i);
				ModelLiveChecks.check(b.tokenIds().equals(a.tokenIds()), label + ": reply " + (i + 1) + " with sharing \""
						+ b.text() + "\", without \"" + a.text() + "\" (" + detail + ")");
				ModelLiveChecks.check(!ModelLiveChecks.cleanText(b.text()).isEmpty(),
						label + ": reply " + (i + 1) + " is empty (" + detail + ")");
			}
			return detail + ", replies equal";
		} finally {
			handler.releaseGpuResources();
		}
	}

	/** The first question alone, then the other two together, on a scheduler whose loop keeps {@code budget} positions. */
	private static Run submit(GgufTokenizer tokenizer, InferencePipeline pipeline, int window,
			ServeScheduleOptions.Mode mode, String budget) throws IOException {
		System.setProperty(SharedPrefixOptions.ENV, budget);
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, ModelLiveChecks.newKvCache(4096),
				PrefillMode.BATCHED, window);
		RequestScheduler scheduler = new RequestScheduler(16, loop, BatchConfig.of(2, 500),
				ServeScheduleOptions.of(mode));
		try {
			List<GenerationResult> replies = new ArrayList<>();
			replies.add(scheduler.submitAndWait(request(0)));
			CompletableFuture<GenerationResult> second = scheduler.submit(request(1), TokenConsumer.discard());
			CompletableFuture<GenerationResult> third = scheduler.submit(request(2), TokenConsumer.discard());
			replies.add(second.get(WAIT_MINUTES, TimeUnit.MINUTES));
			replies.add(third.get(WAIT_MINUTES, TimeUnit.MINUTES));
			return new Run(replies, scheduler.sharedPrefixHits(), scheduler.sharedPrefixTokens());
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException(e);
		} catch (ExecutionException | TimeoutException e) {
			throw new IOException("a request did not complete: " + ModelLiveChecks.failureDetail(e), e);
		} finally {
			scheduler.shutdown();
		}
	}

	private static InferenceRequest request(int question) {
		return InferenceRequest.of("model", List.of(ChatMessage.system(SYSTEM), ChatMessage.user(QUESTIONS[question])),
				GREEDY, RequestPriority.NORMAL);
	}

	private static void restore(String key, String saved) {
		if (saved == null)
			System.clearProperty(key);
		else
			System.setProperty(key, saved);
	}
}
