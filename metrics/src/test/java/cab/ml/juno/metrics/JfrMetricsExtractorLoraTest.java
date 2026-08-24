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
 * Synthetic JFR fixtures for LoRA extractor coverage (old + new event shapes).
 */
class JfrMetricsExtractorLoraTest {

	@TempDir
	Path tmp;

	@Test
	void emptyRecording_yieldsZerosNotExceptions() throws Exception {
		Path jfr = record(rec -> {
		});
		var metrics = extract(jfr);
		assertThat(metrics.get("juno.LoraTrainStep.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraValidation.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraMerge.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraNormRefresh.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraPlayback.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraCheckpoint.count")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraTrainStep.forward_ms.p95")).isEqualTo(0.0);
		assertThat(metrics.keySet()).noneMatch(k -> k.startsWith("juno.LoraTrainStep.by_algorithm."));
	}

	@Test
	void oldTrainStepShape_withoutIdentity_stillExtracts() throws Exception {
		Path jfr = record(rec -> {
			LegacyTrainStep ev = new LegacyTrainStep();
			ev.begin();
			ev.forwardMs = 10;
			ev.backwardMs = 20;
			ev.optimizerMs = 5;
			ev.loss = 1.5f;
			ev.commit();
		});
		var metrics = extract(jfr);
		assertThat(metrics.get("juno.LoraTrainStep.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraTrainStep.forward_ms.p95")).isEqualTo(10.0);
		assertThat(metrics.get("juno.LoraTrainStep.backward_ms.p95")).isEqualTo(20.0);
		assertThat(metrics.get("juno.LoraTrainStep.optimizer_ms.p95")).isEqualTo(5.0);
		assertThat(metrics.get("juno.LoraTrainStep.loss.last")).isEqualTo(1.5);
		assertThat(metrics.keySet()).noneMatch(k -> k.startsWith("juno.LoraTrainStep.by_algorithm."));
	}

	@Test
	void newTrainStep_withIdentity_andValidation_andOperations() throws Exception {
		Path jfr = record(rec -> {
			commitTrain("lora", "standard", 0, 2.0f, 100, false);
			commitTrain("dora", "standard", 0, 1.5f, 80, true);
			commitTrain("qa-lora", "standard", 32, 1.2f, 90, false);
			commitTrain("lora", "rslora", 0, 1.8f, 70, false);

			RichValidation val = new RichValidation();
			val.begin();
			val.loss = 1.1f;
			val.durationMs = 40;
			val.bestSoFar = true;
			val.algorithm = "dora";
			val.commit();

			NormRefresh nr = new NormRefresh();
			nr.begin();
			nr.durationMs = 12;
			nr.algorithm = "dora";
			nr.commit();

			Merge merge = new Merge();
			merge.begin();
			merge.durationMs = 200;
			merge.rmse = 0.01f;
			merge.deltaRetention = 0.95f;
			merge.success = true;
			merge.commit();

			Playback play = new Playback();
			play.begin();
			play.loadMs = 15;
			play.commit();

			Checkpoint ck = new Checkpoint();
			ck.begin();
			ck.operation = "save";
			ck.commit();
		});

		var metrics = extract(jfr);
		assertThat(metrics.get("juno.LoraTrainStep.count")).isEqualTo(4.0);
		assertThat(metrics.get("juno.LoraTrainStep.loss.last")).isCloseTo(1.8, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraTrainStep.loss.mean"))
				.isCloseTo((2.0 + 1.5 + 1.2 + 1.8) / 4.0, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraTrainStep.clipped.fraction")).isEqualTo(0.25);
		assertThat(metrics.get("juno.LoraTrainStep.tokens.total")).isEqualTo(0.0);
		assertThat(metrics.get("juno.LoraTrainStep.total_ms.p95")).isGreaterThan(0.0);
		assertThat(metrics.get("juno.LoraTrainStep.frozen_forward_ms.p95")).isEqualTo(0.0);

		assertThat(metrics.get("juno.LoraTrainStep.by_algorithm.lora.count")).isEqualTo(2.0);
		assertThat(metrics.get("juno.LoraTrainStep.by_algorithm.dora.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraTrainStep.by_algorithm.qa-lora.count")).isEqualTo(1.0);

		assertThat(metrics.get("juno.LoraValidation.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraValidation.loss.last")).isCloseTo(1.1, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraValidation.loss.best")).isCloseTo(1.1, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraValidation.duration_ms.p95")).isEqualTo(40.0);

		assertThat(metrics.get("juno.LoraNormRefresh.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraNormRefresh.duration_ms.p95")).isEqualTo(12.0);
		assertThat(metrics.get("juno.LoraMerge.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraMerge.duration_ms.p95")).isEqualTo(200.0);
		assertThat(metrics.get("juno.LoraMerge.rmse.last")).isCloseTo(0.01, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraMerge.delta_retention.last"))
				.isCloseTo(0.95, org.assertj.core.data.Offset.offset(1e-5));
		assertThat(metrics.get("juno.LoraPlayback.count")).isEqualTo(1.0);
		assertThat(metrics.get("juno.LoraPlayback.load_ms.p95")).isEqualTo(15.0);
		assertThat(metrics.get("juno.LoraCheckpoint.count")).isEqualTo(1.0);
	}

	private static void commitTrain(String algorithm, String scaling, int groupWidth, float loss, long totalMs,
			boolean clipped) {
		RichTrainStep ev = new RichTrainStep();
		ev.begin();
		ev.forwardMs = totalMs / 2;
		ev.backwardMs = totalMs / 4;
		ev.optimizerMs = totalMs / 4;
		ev.totalMs = totalMs;
		ev.loss = loss;
		ev.globalGradNorm = 1.0f;
		ev.clipped = clipped;
		ev.numTokens = 0;
		ev.predictionCount = 0;
		ev.algorithm = algorithm;
		ev.scaling = scaling;
		ev.groupWidth = groupWidth;
		ev.commit();
	}

	private Path record(ThrowingConsumer<Recording> body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.LoraTrainStep");
			rec.enable("juno.LoraValidation");
			rec.enable("juno.LoraNormRefresh");
			rec.enable("juno.LoraMerge");
			rec.enable("juno.LoraPlayback");
			rec.enable("juno.LoraCheckpoint");
			rec.setDestination(jfr);
			rec.start();
			body.accept(rec);
			Thread.sleep(20);
			rec.stop();
		}
		assertThat(Files.size(jfr)).isGreaterThan(0);
		return jfr;
	}

	private static java.util.Map<String, Double> extract(Path jfr) throws Exception {
		ModelsConfig.ModelEntry entry = new ModelsConfig.ModelEntry("tiny", "tiny.gguf");
		return JfrMetricsExtractor.extract(jfr, entry).getMetrics();
	}

	@FunctionalInterface
	private interface ThrowingConsumer<T> {
		void accept(T t) throws Exception;
	}

	@Name("juno.LoraTrainStep")
	@Label("LoRA Train Step")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class LegacyTrainStep extends Event {
		@Label("Forward ms")
		public long forwardMs;
		@Label("Backward ms")
		public long backwardMs;
		@Label("Optimizer ms")
		public long optimizerMs;
		@Label("Loss")
		public float loss;
	}

	@Name("juno.LoraTrainStep")
	@Label("LoRA Train Step")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class RichTrainStep extends Event {
		public long forwardMs;
		public long backwardMs;
		public long optimizerMs;
		public long totalMs;
		public float loss;
		public float globalGradNorm;
		public boolean clipped;
		public int numTokens;
		public int predictionCount;
		public String algorithm = "";
		public String scaling = "";
		public int groupWidth;
	}

	@Name("juno.LoraValidation")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class RichValidation extends Event {
		public float loss;
		public long durationMs;
		public boolean bestSoFar;
		public String algorithm = "";
	}

	@Name("juno.LoraNormRefresh")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class NormRefresh extends Event {
		public long durationMs;
		public String algorithm = "";
	}

	@Name("juno.LoraMerge")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class Merge extends Event {
		public long durationMs;
		public float rmse;
		public float deltaRetention;
		public boolean success;
	}

	@Name("juno.LoraPlayback")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class Playback extends Event {
		public long loadMs;
	}

	@Name("juno.LoraCheckpoint")
	@Category({ "Juno", "LoRA" })
	@StackTrace(false)
	public static class Checkpoint extends Event {
		public String operation = "";
	}
}
