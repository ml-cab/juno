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
package cab.ml.juno.kit;


import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Flow;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import cab.ml.juno.player.JunoHttpClient;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Greedy streamed OpenAI chats through {@link JunoHttpClient}, each content piece stamped on arrival, as the
 * schedule scripts stamp their stream chunks: time to first token is the first arrival, time per output token the
 * mean gap between later ones.
 */
final class StreamTiming {

	/** One stream: its text, each piece's arrival in ms after the request was sent, and an error if it failed. */
	record Result(String text, List<Double> arrivalsMs, String error) {

		Result {
			arrivalsMs = List.copyOf(arrivalsMs);
		}

		boolean ok() {
			return error == null && !arrivalsMs.isEmpty();
		}

		double ttftMs() {
			return arrivalsMs.isEmpty() ? Double.NaN : arrivalsMs.get(0);
		}

		/** Mean gap between consecutive pieces after the first; NaN with fewer than two pieces. */
		double tpotMs() {
			int n = arrivalsMs.size();
			return n < 2 ? Double.NaN : (arrivalsMs.get(n - 1) - arrivalsMs.get(0)) / (n - 1);
		}
	}

	private StreamTiming() {
	}

	static Result stream(JunoHttpClient client, String model, String prompt, int maxTokens)
			throws InterruptedException {
		StringBuilder text = new StringBuilder();
		List<Double> arrivals = new ArrayList<>();
		String[] error = { null };
		CountDownLatch done = new CountDownLatch(1);
		long t0 = System.nanoTime();
		client.streamingOpenAiChat(model, List.of(ChatMessage.user(prompt)), maxTokens, 0f)
				.subscribe(new Flow.Subscriber<String>() {
					@Override
					public void onSubscribe(Flow.Subscription sub) {
						sub.request(Long.MAX_VALUE);
					}

					@Override
					public void onNext(String piece) {
						synchronized (arrivals) {
							arrivals.add((System.nanoTime() - t0) / 1e6);
							text.append(piece);
						}
					}

					@Override
					public void onError(Throwable t) {
						error[0] = String.valueOf(t.getMessage());
						done.countDown();
					}

					@Override
					public void onComplete() {
						done.countDown();
					}
				});
		if (!done.await(20, TimeUnit.MINUTES))
			error[0] = "the stream did not finish within 20 minutes";
		synchronized (arrivals) {
			return new Result(text.toString(), arrivals, error[0]);
		}
	}

	/** One stream per prompt, released together on threads of their own; results in prompt order. */
	static List<Result> concurrent(JunoHttpClient client, String model, List<String> prompts, int maxTokens)
			throws Exception {
		CountDownLatch go = new CountDownLatch(1);
		List<Future<Result>> futures = new ArrayList<>();
		try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
			for (String p : prompts)
				futures.add(pool.submit(() -> {
					go.await();
					return stream(client, model, p, maxTokens);
				}));
			go.countDown();
		}
		List<Result> out = new ArrayList<>();
		for (Future<Result> f : futures)
			out.add(f.get());
		return out;
	}

	/** The mean of {@code values} that are numbers; NaN when none is. */
	static double mean(List<Double> values) {
		return values.stream().filter(v -> !v.isNaN()).mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
	}
}
