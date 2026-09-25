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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Configuration;
import jdk.jfr.Event;
import jdk.jfr.Name;
import jdk.jfr.Recording;

/**
 * Extraction from one named recording to one named output file.
 *
 * <p>The two existing entry points cannot serve a measurement harness that
 * scopes its own recording window: one scans the working directory and maps what
 * it finds against {@code models.json}, and both write to a fixed relative path,
 * so the caller has to control the working directory to control the output. A
 * harness that starts and stops a recording around a single measured request
 * needs to name both the recording and where its metrics land.
 *
 * <p>The failure cases matter as much as the success case. A missing or empty
 * recording that produced an all-zero metrics file would read downstream as a run
 * with no collection pauses and no tokens, which is indistinguishable from a
 * clean fast run — so both fail rather than write anything.
 */
class JfrMetricsCliTest {

	@TempDir
	Path tmp;

	@Name("juno.TokenProduced")
	static class SyntheticTokenProduced extends Event {
	}

	@Test
	void writesTheNamedRecordingsMetricsToTheNamedOutput() throws Exception {
		Path jfr = recordTwoTokens(tmp.resolve("scoped-window.jfr"));
		Path out = tmp.resolve("nested").resolve("metrics.json");

		JfrMetricsCli.main(new String[] { jfr.toString(), out.toString(), "tinyllama-1.1b", "tinyllama-1.1b.gguf" });

		String json = Files.readString(out, StandardCharsets.UTF_8);
		assertThat(json).contains("\"tinyllama-1.1b\"").contains("tinyllama-1.1b.gguf")
				.contains("scoped-window.jfr");
		assertThat(metric(json, "juno.TokenProduced.count")).isEqualTo(2.0);
	}

	@Test
	void emitsTheJdkAccountingSoAGateCanReadGcAndAllocation() throws Exception {
		Path jfr = recordTwoTokens(tmp.resolve("juno-tinyllama-20260924-120000.jfr"));
		Path out = tmp.resolve("metrics.json");

		JfrMetricsCli.main(new String[] { jfr.toString(), out.toString() });

		String json = Files.readString(out, StandardCharsets.UTF_8);
		for (String key : new String[] { "jdk.GCPhasePause.count", "jdk.GCPhasePause.max_ms",
				"jdk.ThreadAllocationStatistics.bytes_total", "jdk.ExecutionSample.count" })
			assertThat(json).as("%s present so a published run can state what to read its throughput against", key)
					.contains("\"" + key + "\"");
	}

	@Test
	void derivesTheModelStemFromTheRecordingNameWhenNotGiven() throws Exception {
		Path jfr = recordTwoTokens(tmp.resolve("juno-mistral-7b-instruct-20260924-120000.jfr"));
		Path out = tmp.resolve("metrics.json");

		JfrMetricsCli.main(new String[] { jfr.toString(), out.toString() });

		assertThat(Files.readString(out, StandardCharsets.UTF_8)).contains("\"mistral-7b-instruct\"");
	}

	@Test
	void missingRecordingFailsNamingTheFileAndWritesNothing() {
		Path absent = tmp.resolve("never-written.jfr");
		Path out = tmp.resolve("metrics.json");

		assertThatThrownBy(() -> JfrMetricsCli.main(new String[] { absent.toString(), out.toString() }))
				.hasMessageContaining("never-written.jfr");
		assertThat(out).doesNotExist();
	}

	@Test
	void emptyRecordingFailsRatherThanPublishingAnAllZeroResult() throws Exception {
		Path empty = Files.createFile(tmp.resolve("truncated.jfr"));
		Path out = tmp.resolve("metrics.json");

		assertThatThrownBy(() -> JfrMetricsCli.main(new String[] { empty.toString(), out.toString() }))
				.hasMessageContaining("truncated.jfr");
		assertThat(out).doesNotExist();
	}

	@Test
	void missingArgumentsFailWithUsageRatherThanScanningTheWorkingDirectory() {
		assertThatThrownBy(() -> JfrMetricsCli.main(new String[] { "only-one-argument.jfr" }))
				.hasMessageContaining("usage");
	}

	/** Two events, so the extractor has a span to divide by and reports a rate. */
	private Path recordTwoTokens(Path dest) throws Exception {
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.TokenProduced");
			rec.setDestination(dest);
			rec.start();
			new SyntheticTokenProduced().commit();
			Thread.sleep(20);
			new SyntheticTokenProduced().commit();
			Thread.sleep(50);
			rec.stop();
		}
		assertThat(Files.size(dest)).isGreaterThan(0);
		return dest;
	}

	private static double metric(String json, String key) {
		var matcher = java.util.regex.Pattern.compile("\"" + java.util.regex.Pattern.quote(key) + "\"\\s*:\\s*([-0-9.eE]+)")
				.matcher(json);
		assertThat(matcher.find()).as("metric %s present in %s", key, json).isTrue();
		return Double.parseDouble(matcher.group(1));
	}
}
