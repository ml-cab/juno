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


import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * {@code sessions} greedy blocking chats of one prompt, released together on threads of their own, as the parallel
 * and schedule scripts send them. Aggregate throughput is every completion token over the round's wall time.
 */
final class ConcurrentChats {

	/** One round: its wall time and, per request, the answer and its own wall time. */
	record Round(double wallMs, List<JunoProcess.Answer> answers, List<Double> latenciesMs) {

		Round {
			answers = List.copyOf(answers);
			latenciesMs = List.copyOf(latenciesMs);
		}

		/** Every request answered HTTP 200 with at least one completion token. */
		boolean allOk() {
			return answers.stream().allMatch(a -> a.status() == 200 && tokens(a) > 0);
		}

		int completionTokens() {
			return answers.stream().filter(a -> a.status() == 200).mapToInt(ConcurrentChats::tokens).sum();
		}

		double aggregateTps() {
			return wallMs > 0 ? completionTokens() / (wallMs / 1000.0) : 0;
		}

		double meanLatencyMs() {
			return latenciesMs.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
		}

		/** The distinct reply texts, in first-seen order. */
		Set<String> replies() {
			Set<String> s = new LinkedHashSet<>();
			for (JunoProcess.Answer a : answers)
				s.add(GrammarProbe.content(a));
			return s;
		}

		/** The requests that did not answer, for a failure message. */
		String failures() {
			List<String> out = new ArrayList<>();
			for (int i = 0; i < answers.size(); i++)
				if (answers.get(i).status() != 200 || tokens(answers.get(i)) == 0)
					out.add("request " + (i + 1) + ": " + GrammarProbe.describe(answers.get(i)));
			return out.isEmpty() ? "none" : String.join("; ", out);
		}
	}

	private ConcurrentChats() {
	}

	static Round run(JunoProcess s, String model, String prompt, int sessions, int maxTokens) throws Exception {
		String body = GrammarProbe.body(model, prompt, maxTokens).toString();
		CountDownLatch go = new CountDownLatch(1);
		List<Future<double[]>> latencies = new ArrayList<>();
		JunoProcess.Answer[] answers = new JunoProcess.Answer[sessions];
		long t0;
		long t1;
		try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
			for (int i = 0; i < sessions; i++) {
				int idx = i;
				latencies.add(pool.submit(() -> {
					go.await();
					long start = System.nanoTime();
					try {
						answers[idx] = s.post(GrammarProbe.PATH, body);
					} catch (IOException e) {
						answers[idx] = new JunoProcess.Answer(0, String.valueOf(e.getMessage()));
					}
					return new double[] { (System.nanoTime() - start) / 1e6 };
				}));
			}
			t0 = System.nanoTime();
			go.countDown();
			for (Future<double[]> f : latencies)
				f.get();
			t1 = System.nanoTime();
		}
		List<Double> ms = new ArrayList<>();
		for (Future<double[]> f : latencies)
			ms.add(f.get()[0]);
		return new Round((t1 - t0) / 1e6, List.of(answers), ms);
	}

	private static int tokens(JunoProcess.Answer a) {
		try {
			return a.status() == 200 ? a.json().path("usage").path("completion_tokens").asInt(0) : 0;
		} catch (IOException e) {
			return 0;
		}
	}
}
