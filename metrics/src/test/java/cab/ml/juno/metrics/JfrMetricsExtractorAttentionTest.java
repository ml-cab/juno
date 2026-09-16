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
 * Synthetic JFR fixtures for {@code juno.Attention} extraction — a narrower span
 * around {@code gqa}/{@code gqaInto} added so attention cost (softmax/RoPE/QK^T,
 * scalar CPU regardless of GPU layer offload) is visible separately from
 * {@code juno.MatVec}, in particular the {@code O(seq^2)} cost at long prefill
 * windows (see {@code docs/perf-compare/README.md} — "GPU batched-prefill GEMM
 * bake-off — Tier 17", long-window caveat).
 */
class JfrMetricsExtractorAttentionTest {

	@TempDir
	Path tmp;

	@Test
	void emptyRecording_yieldsZerosNotExceptions() throws Exception {
		Path jfr = record(rec -> {
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Attention.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Attention.prefill.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Attention.decode.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Attention.duration.total_ms")).isEqualTo(0.0);
	}

	@Test
	void batchedPrefillCall_windowSizeAboveOne_bucketsAsPrefill() throws Exception {
		Path jfr = record(rec -> {
			commit(64, 0, 64);   // first prefill window, layer call over 64 positions
			commit(64, 64, 128); // second prefill window (chunked), starts mid-context
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Attention.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Attention.prefill.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Attention.decode.count")).isEqualTo(0.0);
	}

	@Test
	void singlePositionCall_atStartZero_bucketsAsPrefill() throws Exception {
		// transformerLayer() sequential-fallback prefill: windowSize=1, startPosition=0.
		Path jfr = record(rec -> commit(1, 0, 1));
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Attention.prefill.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.Attention.decode.count")).isEqualTo(0.0);
	}

	@Test
	void singlePositionCall_atStartAboveZero_bucketsAsDecode() throws Exception {
		Path jfr = record(rec -> {
			commit(1, 5, 6);
			commit(1, 6, 7);
			commit(1, 7, 8);
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Attention.count")).isEqualTo(3.0);
		assertThat(metrics.get("juno.Attention.prefill.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.Attention.decode.count")).isEqualTo(3.0);
	}

	@Test
	void mixedPrefillAndDecode_aggregatesIndependently() throws Exception {
		Path jfr = record(rec -> {
			commit(128, 0, 128); // one batched prefill window
			commit(1, 128, 129); // subsequent decode tokens
			commit(1, 129, 130);
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get("juno.Attention.count")).isEqualTo(3.0);
		assertThat(metrics.get("juno.Attention.prefill.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.Attention.decode.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.Attention.duration.total_ms"))
				.isEqualTo(metrics.get("juno.Attention.prefill.total_ms")
						+ metrics.get("juno.Attention.decode.total_ms"));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private void commit(int windowSize, int startPosition, int contextLength) {
		SyntheticAttention ev = new SyntheticAttention();
		ev.begin();
		ev.windowSize = windowSize;
		ev.startPosition = startPosition;
		ev.contextLength = contextLength;
		ev.commit();
	}

	private Path record(ThrowingConsumer<Recording> body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.Attention");
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

	@Name("juno.Attention")
	@Label("Attention")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticAttention extends Event {
		public int windowSize;
		public int startPosition;
		public int contextLength;
	}
}
