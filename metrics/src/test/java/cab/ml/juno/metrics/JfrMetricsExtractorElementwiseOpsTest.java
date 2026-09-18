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
import java.util.stream.Stream;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import jdk.jfr.Category;
import jdk.jfr.Configuration;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Recording;
import jdk.jfr.StackTrace;

/**
 * Synthetic JFR fixtures for the four scalar-CPU elementwise-op events
 * ({@code juno.RmsNorm}/{@code Rope}/{@code ResidualAdd}/{@code SwiGlu}) added
 * as the Tier 19 baseline measurement — a JFR span around {@code rmsNorm}/
 * {@code rope}/residual-add/{@code silu(gate)*up} so their scalar CPU cost is
 * visible in the same {@code count}/{@code prefill}/{@code decode}/
 * {@code total_ms}/{@code p95_ms} shape as {@code juno.Attention}, before any
 * GPU-kernel work moves them off the CPU.
 *
 * <p>All four events share {@link JfrMetricsExtractorAttentionTest}'s
 * windowSize/startPosition prefill-vs-decode bucketing heuristic, so this
 * class runs the same scenario matrix once per event name rather than
 * duplicating the test bodies four times.
 */
class JfrMetricsExtractorElementwiseOpsTest {

	@TempDir
	Path tmp;

	static Stream<String> eventNames() {
		return Stream.of("juno.RmsNorm", "juno.Rope", "juno.ResidualAdd", "juno.SwiGlu");
	}

	@ParameterizedTest
	@MethodSource("eventNames")
	void emptyRecording_yieldsZerosNotExceptions(String eventName) throws Exception {
		Path jfr = record(eventName, () -> {
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get(eventName + ".count")).isEqualTo(0.0);
		assertThat(metrics.get(eventName + ".prefill.count")).isEqualTo(0.0);
		assertThat(metrics.get(eventName + ".decode.count")).isEqualTo(0.0);
		assertThat(metrics.get(eventName + ".duration.total_ms")).isEqualTo(0.0);
	}

	@ParameterizedTest
	@MethodSource("eventNames")
	void batchedPrefillCall_windowSizeAboveOne_bucketsAsPrefill(String eventName) throws Exception {
		Path jfr = record(eventName, () -> {
			commit(eventName, 32, 0, 4096);   // one prefill window over 32 rows
			commit(eventName, 32, 32, 4096);  // a second chunked window
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get(eventName + ".count")).isEqualTo(2.0);
		assertThat(metrics.get(eventName + ".prefill.count")).isEqualTo(2.0);
		assertThat(metrics.get(eventName + ".decode.count")).isEqualTo(0.0);
	}

	@ParameterizedTest
	@MethodSource("eventNames")
	void singleRowCall_atStartAboveZero_bucketsAsDecode(String eventName) throws Exception {
		Path jfr = record(eventName, () -> {
			commit(eventName, 1, 5, 4096);
			commit(eventName, 1, 6, 4096);
			commit(eventName, 1, 7, 4096);
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get(eventName + ".count")).isEqualTo(3.0);
		assertThat(metrics.get(eventName + ".prefill.count")).isEqualTo(0.0);
		assertThat(metrics.get(eventName + ".decode.count")).isEqualTo(3.0);
	}

	@ParameterizedTest
	@MethodSource("eventNames")
	void mixedPrefillAndDecode_aggregatesIndependently(String eventName) throws Exception {
		Path jfr = record(eventName, () -> {
			commit(eventName, 64, 0, 4096);  // one batched prefill window
			commit(eventName, 1, 64, 4096);  // subsequent decode tokens
			commit(eventName, 1, 65, 4096);
		});
		Map<String, Double> metrics = extract(jfr);
		assertThat(metrics.get(eventName + ".count")).isEqualTo(3.0);
		assertThat(metrics.get(eventName + ".prefill.count")).isEqualTo(1.0);
		assertThat(metrics.get(eventName + ".decode.count")).isEqualTo(2.0);
		// isCloseTo, not isEqualTo: total_ms is summed over the combined list while
		// prefill/decode totals are summed separately then added here — floating-point
		// addition is not associative, so real (non-fixed) JFR-measured nanos can differ
		// in the last bit between the two summation orders.
		assertThat(metrics.get(eventName + ".duration.total_ms"))
				.isCloseTo(metrics.get(eventName + ".prefill.total_ms")
						+ metrics.get(eventName + ".decode.total_ms"), org.assertj.core.data.Offset.offset(1e-6));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private void commit(String eventName, int windowSize, int startPosition, int dimension) {
		switch (eventName) {
			case "juno.RmsNorm" -> {
				SyntheticRmsNorm ev = new SyntheticRmsNorm();
				ev.begin();
				ev.windowSize = windowSize;
				ev.startPosition = startPosition;
				ev.dimension = dimension;
				ev.commit();
			}
			case "juno.Rope" -> {
				SyntheticRope ev = new SyntheticRope();
				ev.begin();
				ev.windowSize = windowSize;
				ev.startPosition = startPosition;
				ev.dimension = dimension;
				ev.commit();
			}
			case "juno.ResidualAdd" -> {
				SyntheticResidualAdd ev = new SyntheticResidualAdd();
				ev.begin();
				ev.windowSize = windowSize;
				ev.startPosition = startPosition;
				ev.dimension = dimension;
				ev.commit();
			}
			case "juno.SwiGlu" -> {
				SyntheticSwiGlu ev = new SyntheticSwiGlu();
				ev.begin();
				ev.windowSize = windowSize;
				ev.startPosition = startPosition;
				ev.dimension = dimension;
				ev.commit();
			}
			default -> throw new IllegalArgumentException("unknown event: " + eventName);
		}
	}

	private Path record(String eventName, ThrowingRunnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable(eventName);
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

	@FunctionalInterface
	private interface ThrowingRunnable {
		void run() throws Exception;
	}

	@Name("juno.RmsNorm")
	@Label("RMS Normalization")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticRmsNorm extends Event {
		public int windowSize;
		public int startPosition;
		public int dimension;
	}

	@Name("juno.Rope")
	@Label("Rotary Position Embedding")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticRope extends Event {
		public int windowSize;
		public int startPosition;
		public int dimension;
	}

	@Name("juno.ResidualAdd")
	@Label("Residual Add")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticResidualAdd extends Event {
		public int windowSize;
		public int startPosition;
		public int dimension;
	}

	@Name("juno.SwiGlu")
	@Label("SwiGLU")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticSwiGlu extends Event {
		public int windowSize;
		public int startPosition;
		public int dimension;
	}
}
