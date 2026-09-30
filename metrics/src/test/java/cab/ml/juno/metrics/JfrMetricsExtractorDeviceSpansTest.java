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
import static org.assertj.core.api.Assertions.within;

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
 * Synthetic JFR fixtures for {@code juno.DeviceStaging} and {@code juno.WeightDequant}: the
 * host-device copies and weight dequantizations that {@code juno.MatVec} otherwise hides inside
 * its own span. The engine commits both as totals per site and phase at the end of a recording
 * chunk; durations are the events' measured fields, over the copies that were timed.
 */
class JfrMetricsExtractorDeviceSpansTest {

	private static final List<String> DIRECTIONS = List.of("H2D", "D2H", "D2D");
	private static final List<String> PHASES = List.of("prefill", "decode", "other");
	private static final List<String> FORMATS = List.of("F32", "F16", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
			"Q6_K");

	@TempDir
	Path tmp;

	@Test
	void noEvents_writesEveryKeyAsZero() throws Exception {
		Map<String, Double> m = extract(record(() -> {
		}));
		for (String dir : DIRECTIONS) {
			assertZero(m, "juno.DeviceStaging." + dir);
			for (String phase : PHASES)
				assertZero(m, "juno.DeviceStaging." + dir + "." + phase);
		}
		assertThat(m.get("juno.WeightDequant.count")).isEqualTo(0.0);
		assertThat(m.get("juno.WeightDequant.timed_count")).isEqualTo(0.0);
		assertThat(m.get("juno.WeightDequant.total_ms")).isEqualTo(0.0);
		for (String timing : List.of("device", "host")) {
			assertThat(m.get("juno.WeightDequant." + timing + ".count")).isEqualTo(0.0);
			assertThat(m.get("juno.WeightDequant." + timing + ".total_ms")).isEqualTo(0.0);
		}
		for (String format : FORMATS) {
			assertThat(m.get("juno.WeightDequant.format." + format + ".count")).isEqualTo(0.0);
			assertThat(m.get("juno.WeightDequant.format." + format + ".total_ms")).isEqualTo(0.0);
		}
	}

	@Test
	void staging_sumsTotalsPerDirectionAndPhase() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			staging("cudaMemcpyAsync(xh H2D batched-gemm)", "H2D", "prefill", 154, 160_000_000, 154, 2_000_000);
			staging("memcpy(K row H2D)", "H2D", "prefill", 11_264, 2_883_584, 11_264, 1_000_000);
			staging("cudaMemcpyAsync(y D2H batched-gemm)", "D2H", "prefill", 154, 800_000_000, 154, 3_500_000);
		}));
		assertThat(m.get("juno.DeviceStaging.H2D.count")).isEqualTo(11_418.0);
		assertThat(m.get("juno.DeviceStaging.H2D.bytes")).isEqualTo(162_883_584.0);
		assertThat(m.get("juno.DeviceStaging.H2D.timed_count")).isEqualTo(11_418.0);
		assertThat(m.get("juno.DeviceStaging.H2D.total_ms")).isCloseTo(3.0, within(1e-9));
		assertThat(m.get("juno.DeviceStaging.H2D.prefill.bytes")).isEqualTo(162_883_584.0);
		assertThat(m.get("juno.DeviceStaging.D2H.prefill.count")).isEqualTo(154.0);
		assertThat(m.get("juno.DeviceStaging.D2H.prefill.total_ms")).isCloseTo(3.5, within(1e-9));
		assertZero(m, "juno.DeviceStaging.D2D");
	}

	@Test
	void staging_decodeIsCountedButNotTimed_andOtherStaysApart() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			staging("cudaMemcpyAsync(xh H2D)", "H2D", "decode", 7_000, 56_000_000, 0, 0);
			staging("memcpy(A FP16 H2D)", "H2D", "other", 155, 1_200_000_000, 155, 900_000_000);
		}));
		assertThat(m.get("juno.DeviceStaging.H2D.decode.count")).isEqualTo(7_000.0);
		assertThat(m.get("juno.DeviceStaging.H2D.decode.bytes")).isEqualTo(56_000_000.0);
		assertThat(m.get("juno.DeviceStaging.H2D.decode.timed_count")).isEqualTo(0.0);
		assertThat(m.get("juno.DeviceStaging.H2D.decode.total_ms")).isEqualTo(0.0);
		assertThat(m.get("juno.DeviceStaging.H2D.other.bytes")).isEqualTo(1_200_000_000.0);
		assertThat(m.get("juno.DeviceStaging.H2D.other.total_ms")).isCloseTo(900.0, within(1e-9));
		assertThat(m.get("juno.DeviceStaging.H2D.prefill.count")).isEqualTo(0.0);
	}

	@Test
	void staging_estimatesASampledSiteFromItsMeanAndKeepsTheMeasuredSumApart() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			// KV rows: 1,600 copies, 100 of them timed at 2 us each.
			staging("memcpy(K row H2D)", "H2D", "prefill", 1_600, 409_600, 100, 200_000);
			// Matmul activations: every copy timed.
			staging("cudaMemcpyAsync(xh H2D batched-gemm)", "H2D", "prefill", 10, 10_000_000, 10, 5_000_000);
		}));
		assertThat(m.get("juno.DeviceStaging.H2D.prefill.total_ms")).isCloseTo(5.2, within(1e-9));
		assertThat(m.get("juno.DeviceStaging.H2D.prefill.estimated_total_ms")).isCloseTo(8.2, within(1e-9));
		assertThat(m.get("juno.DeviceStaging.site.memcpy_k_row_h2d.prefill.estimated_total_ms"))
				.isCloseTo(3.2, within(1e-9));
	}

	@Test
	void staging_isAttributedPerSite() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			staging("memcpy(K row H2D)", "H2D", "prefill", 100, 25_600, 100, 50_000);
			staging("memcpy(K row H2D)", "H2D", "prefill", 20, 5_120, 20, 10_000);
			staging("memcpy(K row H2D)", "H2D", "decode", 1, 256, 0, 0);
		}));
		assertThat(m.get("juno.DeviceStaging.site.memcpy_k_row_h2d.prefill.count")).isEqualTo(120.0);
		assertThat(m.get("juno.DeviceStaging.site.memcpy_k_row_h2d.prefill.bytes")).isEqualTo(30_720.0);
		assertThat(m.get("juno.DeviceStaging.site.memcpy_k_row_h2d.decode.count")).isEqualTo(1.0);
	}

	@Test
	void dequant_aggregatesPerFormatAndPerTimingSource() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			dequant("Q4_K", "device", 110, 110, 3_000_000);
			dequant("Q6_K", "device", 44, 44, 500_000);
			dequant("Q8_0", "host", 1, 1, 40_000_000);
		}));
		assertThat(m.get("juno.WeightDequant.count")).isEqualTo(155.0);
		assertThat(m.get("juno.WeightDequant.total_ms")).isCloseTo(43.5, within(1e-9));
		assertThat(m.get("juno.WeightDequant.device.count")).isEqualTo(154.0);
		assertThat(m.get("juno.WeightDequant.device.total_ms")).isCloseTo(3.5, within(1e-9));
		assertThat(m.get("juno.WeightDequant.host.count")).isEqualTo(1.0);
		assertThat(m.get("juno.WeightDequant.format.Q4_K.count")).isEqualTo(110.0);
		assertThat(m.get("juno.WeightDequant.format.Q4_K.total_ms")).isCloseTo(3.0, within(1e-9));
		assertThat(m.get("juno.WeightDequant.format.Q8_0.total_ms")).isCloseTo(40.0, within(1e-9));
		assertThat(m.get("juno.WeightDequant.format.Q5_K.count")).isEqualTo(0.0);
	}

	@Test
	void dequant_reportsAFormatOutsideTheFixedList() throws Exception {
		Map<String, Double> m = extract(record(() -> dequant("IQ4_XS", "device", 2, 2, 700_000)));
		assertThat(m.get("juno.WeightDequant.format.IQ4_XS.count")).isEqualTo(2.0);
		assertThat(m.get("juno.WeightDequant.format.IQ4_XS.total_ms")).isCloseTo(0.7, within(1e-9));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static void assertZero(Map<String, Double> m, String prefix) {
		for (String k : List.of(".count", ".bytes", ".timed_count", ".total_ms", ".estimated_total_ms"))
			assertThat(m.get(prefix + k)).as(prefix + k).isEqualTo(0.0);
	}

	private static void staging(String site, String direction, String phase, long copies, long bytes,
			long timedCopies, long transferNanos) {
		SyntheticDeviceStaging ev = new SyntheticDeviceStaging();
		ev.site = site;
		ev.direction = direction;
		ev.phase = phase;
		ev.copies = copies;
		ev.bytes = bytes;
		ev.timedCopies = timedCopies;
		ev.transferNanos = transferNanos;
		ev.commit();
	}

	private static void dequant(String format, String timing, long count, long timedCount, long dequantNanos) {
		SyntheticWeightDequant ev = new SyntheticWeightDequant();
		ev.format = format;
		ev.timing = timing;
		ev.count = count;
		ev.timedCount = timedCount;
		ev.dequantNanos = dequantNanos;
		ev.commit();
	}

	private Path record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.DeviceStaging");
			rec.enable("juno.WeightDequant");
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

	@Name("juno.DeviceStaging")
	@Label("Device Staging")
	@Category({ "Juno", "GPU" })
	@StackTrace(false)
	public static class SyntheticDeviceStaging extends Event {
		public String site;
		public String direction;
		public String phase;
		public long copies;
		public long bytes;
		public long timedCopies;
		public long transferNanos;
	}

	@Name("juno.WeightDequant")
	@Label("Weight Dequant")
	@Category({ "Juno", "GPU" })
	@StackTrace(false)
	public static class SyntheticWeightDequant extends Event {
		public String format;
		public String timing;
		public long count;
		public long timedCount;
		public long dequantNanos;
	}
}
