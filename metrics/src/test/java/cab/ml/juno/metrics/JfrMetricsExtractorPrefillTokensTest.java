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
 * Synthetic JFR fixtures for the prompt-token coverage of {@code juno.PrefillBatch}: how many
 * tokens the recorded windows prefilled and the lowest position any of them started at. A
 * benchmark reads both to confirm a measured request prefilled its whole prompt rather than
 * resuming from KV it did not write.
 */
class JfrMetricsExtractorPrefillTokensTest {

	@TempDir
	Path tmp;

	@Test
	void noPrefillWindow_writesBothKeysWithTheirEmptyValues() throws Exception {
		Map<String, Double> metrics = extract(record(() -> {
		}));
		assertThat(metrics.get("juno.PrefillBatch.tokens")).isEqualTo(0.0);
		assertThat(metrics.get("juno.PrefillBatch.min_start_position")).isEqualTo(-1.0);
	}

	@Test
	void wholePromptInOneWindow_countsEveryTokenFromPositionZero() throws Exception {
		Map<String, Double> metrics = extract(record(() -> commit(127, 0)));
		assertThat(metrics.get("juno.PrefillBatch.tokens")).isEqualTo(127.0);
		assertThat(metrics.get("juno.PrefillBatch.min_start_position")).isEqualTo(0.0);
	}

	@Test
	void chunkedPrefill_sumsTheWindowsAndKeepsTheFirstStart() throws Exception {
		Map<String, Double> metrics = extract(record(() -> {
			commit(32, 0);
			commit(32, 32);
			commit(32, 64);
			commit(31, 96);
		}));
		assertThat(metrics.get("juno.PrefillBatch.tokens")).isEqualTo(127.0);
		assertThat(metrics.get("juno.PrefillBatch.min_start_position")).isEqualTo(0.0);
	}

	@Test
	void resumedFromAReusedPrefix_reportsWhereThePrefillStarted() throws Exception {
		Map<String, Double> metrics = extract(record(() -> commit(27, 100)));
		assertThat(metrics.get("juno.PrefillBatch.tokens")).isEqualTo(27.0);
		assertThat(metrics.get("juno.PrefillBatch.min_start_position")).isEqualTo(100.0);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static void commit(int windowSize, int startPosition) {
		SyntheticPrefillBatch ev = new SyntheticPrefillBatch();
		ev.begin();
		ev.requestId = "test";
		ev.windowSize = windowSize;
		ev.startPosition = startPosition;
		ev.commit();
	}

	private Path record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.PrefillBatch");
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

	@Name("juno.PrefillBatch")
	@Label("Prefill Batch")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticPrefillBatch extends Event {
		public String requestId;
		public int windowSize;
		public int startPosition;
	}
}
