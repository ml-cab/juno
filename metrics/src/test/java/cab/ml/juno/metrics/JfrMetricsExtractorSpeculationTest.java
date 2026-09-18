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
 * Synthetic JFR fixtures for {@code juno.Speculation} extraction (ngram-simple
 * speculative decoding, see {@code PLAN-Infra-Tier9.md}).
 */
class JfrMetricsExtractorSpeculationTest {

	@TempDir
	Path tmp;

	@Test
	void emptyRecording_yieldsZerosNotExceptions() throws Exception {
		Path jfr = record(rec -> {
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Speculation.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Speculation.draftTokens.sum")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Speculation.acceptedTokens.sum")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Speculation.acceptanceRate")).isEqualTo(0.0);
	}

	@Test
	void fullAcceptanceRounds_giveAcceptanceRateOfOne() throws Exception {
		Path jfr = record(rec -> {
			commit(4, 4);
			commit(4, 4);
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Speculation.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Speculation.draftTokens.sum")).isEqualTo(8.0);
		assertThat(metrics.get("juno.Speculation.acceptedTokens.sum")).isEqualTo(8.0);
		assertThat(metrics.get("juno.Speculation.acceptanceRate")).isEqualTo(1.0);
	}

	@Test
	void partialAcceptance_computesFractionalAcceptanceRate() throws Exception {
		Path jfr = record(rec -> {
			commit(4, 2); // half accepted
			commit(4, 0); // total miss
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Speculation.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Speculation.draftTokens.sum")).isEqualTo(8.0);
		assertThat(metrics.get("juno.Speculation.acceptedTokens.sum")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Speculation.acceptanceRate")).isEqualTo(0.25);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private void commit(int draftTokens, int acceptedTokens) {
		SyntheticSpeculation ev = new SyntheticSpeculation();
		ev.begin();
		ev.requestId = "test";
		ev.draftTokens = draftTokens;
		ev.acceptedTokens = acceptedTokens;
		ev.commit();
	}

	private Path record(ThrowingConsumer<Recording> body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.Speculation");
			rec.setDestination(jfr);
			rec.start();
			body.accept(rec);
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

	@FunctionalInterface
	private interface ThrowingConsumer<T> {
		void accept(T t) throws Exception;
	}

	@Name("juno.Speculation")
	@Label("Speculation")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticSpeculation extends Event {
		public String requestId;
		public int draftTokens;
		public int acceptedTokens;
	}
}
