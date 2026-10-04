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
package cab.ml.juno.metrics;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Category;
import jdk.jfr.Configuration;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Recording;
import jdk.jfr.StackTrace;

/**
 * Synthetic JFR fixtures for {@code juno.PromptEncode}: how many prompts were encoded and how long
 * encoding took in total. A benchmark subtracts that time from what a request's forward-pass spans
 * leave unaccounted for, since prompt encoding runs before the first forward pass.
 */
class JfrMetricsExtractorPromptEncodeTest {

	@TempDir
	Path tmp;

	@Test
	void noEncode_writesBothKeysAsZero() throws Exception {
		Map<String, Double> metrics = extract(record(() -> {
		}));
		assertThat(metrics.get("juno.PromptEncode.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.PromptEncode.total_ms")).isEqualTo(0.0);
	}

	@Test
	void encodes_areCountedAndTheirDurationsSummed() throws Exception {
		Map<String, Double> metrics = extract(record(() -> {
			commit(30);
			commit(20);
		}));
		assertThat(metrics.get("juno.PromptEncode.count")).isEqualTo(2.0);
		// The events span sleeps of 30 and 20 ms. Sleep granularity and the recorder's clock let a
		// span read slightly short of its sleep (49.89 ms was seen), so the sum is checked in a band.
		assertThat(metrics.get("juno.PromptEncode.total_ms")).isBetween(45.0, 250.0);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static void commit(long sleepMs) {
		SyntheticPromptEncode ev = new SyntheticPromptEncode();
		ev.begin();
		try {
			Thread.sleep(sleepMs);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		ev.requestId = "test";
		ev.characters = 100;
		ev.tokens = 25;
		ev.commit();
	}

	private Path record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.PromptEncode").withoutThreshold();
			rec.setDestination(jfr);
			rec.start();
			body.run();
			Thread.sleep(20);
			rec.stop();
		}
		assertThat(Files.size(jfr)).isGreaterThan(0);
		return jfr;
	}

	private static Map<String, Double> extract(Path jfr) throws Exception {
		ModelsConfig.ModelEntry entry = new ModelsConfig.ModelEntry("tiny", "tiny.gguf");
		return JfrMetricsExtractor.extract(jfr, entry).getMetrics();
	}

	@Name("juno.PromptEncode")
	@Label("Prompt Encode")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticPromptEncode extends Event {
		public String requestId;
		public int characters;
		public int tokens;
	}
}
