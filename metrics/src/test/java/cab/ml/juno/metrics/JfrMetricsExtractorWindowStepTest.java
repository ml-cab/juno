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
import java.util.List;
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
 * Synthetic JFR fixtures for the two inputs a residue-free prefill breakdown needs beyond the
 * per-op spans: {@code juno.WindowStep}, the window work no other span covers (embedding, each
 * projection call including its copy-out, bias adds, the KV write, the LM head), and the
 * prefill/decode split of {@code juno.MatVec}, which a prefill recording needs because it also
 * holds the decode-width pass at the last prompt position.
 */
class JfrMetricsExtractorWindowStepTest {

	private static final List<String> STEPS = List.of("embed", "projection", "bias_add", "kv_write", "lm_head");

	@TempDir
	Path tmp;

	@Test
	void noEvents_writesEveryStepAndMatVecPhaseKeyAsZero() throws Exception {
		Map<String, Double> m = extract(record(() -> {
		}));
		for (String step : STEPS)
			for (String k : List.of(".count", ".prefill.count", ".decode.count", ".prefill.total_ms",
					".decode.total_ms"))
				assertThat(m.get("juno.WindowStep." + step + k)).as(step + k).isEqualTo(0.0);
		for (String k : List.of(".prefill.count", ".decode.count", ".prefill.total_ms", ".decode.total_ms"))
			assertThat(m.get("juno.MatVec" + k)).as("juno.MatVec" + k).isEqualTo(0.0);
	}

	@Test
	void windowStep_bucketsByWindowWidthPerStep() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			step("projection", 511, 0);
			step("projection", 511, 0);
			step("kv_write", 511, 0);
			step("projection", 1, 511);
		}));
		assertThat(m.get("juno.WindowStep.projection.count")).isEqualTo(3.0);
		assertThat(m.get("juno.WindowStep.projection.prefill.count")).isEqualTo(2.0);
		assertThat(m.get("juno.WindowStep.projection.decode.count")).isEqualTo(1.0);
		assertThat(m.get("juno.WindowStep.projection.prefill.total_ms")).isGreaterThan(0.0);
		assertThat(m.get("juno.WindowStep.kv_write.prefill.count")).isEqualTo(1.0);
		assertThat(m.get("juno.WindowStep.lm_head.count")).isEqualTo(0.0);
	}

	@Test
	void windowStep_reportsAStepOutsideTheFixedList() throws Exception {
		Map<String, Double> m = extract(record(() -> step("unflatten", 32, 0)));
		assertThat(m.get("juno.WindowStep.unflatten.prefill.count")).isEqualTo(1.0);
	}

	@Test
	void matVec_splitsPrefillFromDecodeByItsBatchWidth() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			matVec(511);
			matVec(511);
			matVec(1);
		}));
		assertThat(m.get("juno.MatVec.count")).isEqualTo(3.0);
		assertThat(m.get("juno.MatVec.prefill.count")).isEqualTo(2.0);
		assertThat(m.get("juno.MatVec.decode.count")).isEqualTo(1.0);
		assertThat(m.get("juno.MatVec.prefill.total_ms")).isGreaterThan(0.0);
	}

	@Test
	void matVec_withoutTheWidthField_countsInNeitherPhase() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			LegacyMatVec ev = new LegacyMatVec();
			ev.begin();
			ev.backend = "cuda";
			ev.commit();
		}));
		assertThat(m.get("juno.MatVec.count")).isEqualTo(1.0);
		assertThat(m.get("juno.MatVec.prefill.count")).isEqualTo(0.0);
		assertThat(m.get("juno.MatVec.decode.count")).isEqualTo(0.0);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static void spin() {
		long until = System.nanoTime() + 200_000;
		while (System.nanoTime() < until)
			Thread.onSpinWait();
	}

	private static void step(String step, int windowSize, int startPosition) {
		SyntheticWindowStep ev = new SyntheticWindowStep();
		ev.begin();
		spin();
		ev.step = step;
		ev.windowSize = windowSize;
		ev.startPosition = startPosition;
		ev.commit();
	}

	private static void matVec(int windowSize) {
		SyntheticMatVec ev = new SyntheticMatVec();
		ev.begin();
		spin();
		ev.backend = "cuda";
		ev.windowSize = windowSize;
		ev.commit();
	}

	private Path record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording(Configuration.getConfiguration("default"))) {
			rec.enable("juno.WindowStep");
			rec.enable("juno.MatVec");
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

	@Name("juno.WindowStep")
	@Label("Window Step")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticWindowStep extends Event {
		public String step;
		public int windowSize;
		public int startPosition;
	}

	@Name("juno.MatVec")
	@Label("MatVec")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class SyntheticMatVec extends Event {
		public String backend;
		public int rows;
		public int cols;
		public int windowSize;
	}

	/** A {@code juno.MatVec} as builds before the width field recorded it. */
	@Name("juno.MatVec")
	@Label("MatVec")
	@Category({ "Juno", "Inference" })
	@StackTrace(false)
	public static class LegacyMatVec extends Event {
		public String backend;
		public int rows;
		public int cols;
	}
}
