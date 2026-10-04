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

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * Every request records its prompt encoding as one {@code juno.PromptEncode} event, on the static
 * and the continuous schedule. Encoding runs before the first forward pass, so without the event
 * its time is part of the request that no span covers, and a benchmark's check that the spans
 * account for their request reads it as a timestamp error. On long prompts it reaches hundreds of
 * milliseconds.
 */
@DisplayName("juno.PromptEncode: one event per request, on both schedules")
class PromptEncodeEventTest {

	@TempDir
	Path tmp;

	private RequestScheduler scheduler;

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	@Test
	@DisplayName("static schedule: one event carrying the prompt's token count")
	void staticSchedule() throws Exception {
		scheduler = new RequestScheduler(10, loop(), BatchConfig.disabled());
		assertOneEventPerRequest();
	}

	@Test
	@DisplayName("continuous schedule: one event carrying the prompt's token count")
	void continuousSchedule() throws Exception {
		scheduler = new RequestScheduler(10, loop(), BatchConfig.of(4, 20), ServeScheduleOptions.parse("continuous"));
		assertOneEventPerRequest();
	}

	private void assertOneEventPerRequest() throws Exception {
		Path jfr = tmp.resolve("encode-" + System.nanoTime() + ".jfr");
		GenerationResult result;
		try (Recording rec = new Recording()) {
			rec.enable("juno.PromptEncode").withoutThreshold();
			rec.setDestination(jfr);
			rec.start();
			result = scheduler.submit(req(), TokenConsumer.discard()).get(10, TimeUnit.SECONDS);
			rec.stop();
		}
		List<RecordedEvent> events = RecordingFile.readAllEvents(jfr).stream()
				.filter(e -> e.getEventType().getName().equals("juno.PromptEncode")).toList();
		assertThat(events).as("prompt encode events").hasSize(1);
		RecordedEvent ev = events.get(0);
		assertThat(ev.getInt("tokens")).as("tokens").isEqualTo(result.promptTokens());
		assertThat(ev.getInt("characters")).as("characters").isPositive();
		assertThat(ev.getString("requestId")).as("request id").isNotBlank();
	}

	private static GenerationLoop loop() {
		return new GenerationLoop(new SimpleTokenizer(), Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private static InferenceRequest req() {
		return InferenceRequest.of("model", List.of(ChatMessage.user("hello there, how are you")),
				SamplingParams.defaults().withMaxTokens(2), RequestPriority.NORMAL);
	}
}
