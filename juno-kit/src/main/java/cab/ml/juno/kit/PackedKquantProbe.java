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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * The packed K-quant smoke's checks on one model and schedule: a greedy streamed request prefills about
 * {@code nPrompt} tokens and generates exactly {@code nGen}; the prompt count is within 10%, the generated
 * count exact, one stream piece per token; the same request unstreamed returns the same text; the engine
 * log shows the tiled K-quant kernel loaded and the fused packed weights enabled, with no dequantizing
 * fallback. Time to first token and the process's peak device memory are recorded.
 */
final class PackedKquantProbe implements KitProbe {

	static final String KERNEL_LOADED = "Tiled K-quant GEMM kernels loaded";
	static final String PACKED_ENABLED = "Fused Q4_K MMQ enabled";
	static final String FALLBACK = "Tiled K-quant GEMM kernel unavailable";

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String schedule = args.getOrDefault("schedule", "static");
		int nPrompt = KitProbe.intArg(args, "nPrompt", 512);
		int nGen = KitProbe.intArg(args, "nGen", 512);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "true"));
		System.setProperty(ServeScheduleOptions.ENV, schedule);
		AtomicLong peak = new AtomicLong();
		Thread sampler = Thread.ofPlatform().daemon().start(() -> {
			Set<Long> self = Set.of(ProcessHandle.current().pid());
			try {
				while (true) {
					peak.accumulateAndGet(GpuLoad.computeApps(GpuLoad.query(GpuLoad.COMPUTE_APPS), self).ownMiB(),
							Math::max);
					Thread.sleep(500);
				}
			} catch (InterruptedException ignored) {
			}
		});
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		try (NoticeCapture log = NoticeCapture.install();
				JunoPlayer player = JunoPlayer.builder(java.nio.file.Path.of(args.get("model"))).useGpu(gpu)
						.samplingParams(SamplingParams.deterministic().withMaxTokens(nGen).withMinTokens(nGen))
						.build()) {
			int overhead = player.scheduler().submitAndWait(InferenceRequest.of(player.modelId(),
					List.of(ChatMessage.user(ThroughputProbe.words(nPrompt))),
					SamplingParams.deterministic().withMaxTokens(1), RequestPriority.NORMAL)).promptTokens() - nPrompt;
			List<ChatMessage> messages = List.of(ChatMessage.user(ThroughputProbe.words(Math.max(1, nPrompt - overhead))));

			StringBuilder streamed = new StringBuilder();
			int[] pieces = { 0 };
			long[] ttftNanos = { -1 };
			CountDownLatch done = new CountDownLatch(1);
			long t0 = System.nanoTime();
			player.streamPublisher(messages).subscribe(new Flow.Subscriber<String>() {
				@Override
				public void onSubscribe(Flow.Subscription s) {
					s.request(Long.MAX_VALUE);
				}

				@Override
				public void onNext(String piece) {
					if (ttftNanos[0] < 0)
						ttftNanos[0] = System.nanoTime() - t0;
					pieces[0]++;
					streamed.append(piece);
				}

				@Override
				public void onError(Throwable t) {
					done.countDown();
				}

				@Override
				public void onComplete() {
					done.countDown();
				}
			});
			if (!done.await(60, TimeUnit.MINUTES))
				throw new IllegalStateException("streamed request did not complete");
			GenerationResult res = player.chat(messages);

			checks.add(new ProbeReport.Check("prompt of about " + nPrompt + " tokens",
					Math.abs(res.promptTokens() - nPrompt) <= 0.10 * nPrompt, res.promptTokens() + " prompt tokens"));
			checks.add(new ProbeReport.Check("exactly " + nGen + " tokens generated", res.generatedTokens() == nGen,
					res.generatedTokens() + " generated"));
			checks.add(new ProbeReport.Check("one stream piece per generated token", pieces[0] == res.generatedTokens(),
					pieces[0] + " pieces"));
			checks.add(new ProbeReport.Check("streamed and unstreamed greedy text identical",
					streamed.toString().equals(res.text()),
					streamed.toString().equals(res.text()) ? res.text().length() + " characters"
							: "first difference at word " + GpuResidencyProbe.firstDiff(streamed.toString(), res.text())));
			boolean packed = log.first(KERNEL_LOADED) != null && log.first(PACKED_ENABLED) != null
					&& log.first(FALLBACK) == null;
			checks.add(new ProbeReport.Check("packed weights resident and the tiled kernel loaded, no dequantizing fallback",
					packed, packed ? log.first(KERNEL_LOADED) : "the engine log does not show the packed path active"
							+ (log.first(FALLBACK) != null ? ": " + log.first(FALLBACK) : "")));
			facts.put("schedule", schedule);
			facts.put("promptTokens", res.promptTokens());
			facts.put("ttftMs", ttftNanos[0] / 1e6);
		} finally {
			sampler.interrupt();
		}
		facts.put("peakDeviceMiB", peak.get());
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}
}
