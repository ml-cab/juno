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
 * Synthetic JFR fixtures for {@code juno.DeviceCompute}, the device kernels a prefill
 * breakdown has to read rather than infer (the GEMMs and the attention kernel), and for
 * the host FP16 packing counted under {@code juno.DeviceStaging} with direction
 * {@code HOST}, which must stay apart from the bytes that actually cross the bus.
 */
class JfrMetricsExtractorDeviceComputeTest {

	/** The matmul and attention sites, then the prefill-window device region's own operations. */
	private static final List<String> SITES = List.of("gemm_half", "gemv_half_batched", "gemm_fp32", "mmq_packed",
			"gqa_attention", "rms_norm", "convert_fp16", "bias_add", "rope", "kv_append", "gqa_attention_region",
			"swiglu", "residual_add");
	private static final List<String> PHASES = List.of("prefill", "decode", "other");

	@TempDir
	Path tmp;

	@Test
	void noEvents_writesEveryComputeKeyAsZero() throws Exception {
		Map<String, Double> m = extract(record(() -> {
		}));
		assertZero(m, "juno.DeviceCompute");
		for (String phase : PHASES)
			assertZero(m, "juno.DeviceCompute." + phase);
		for (String site : SITES)
			for (String phase : PHASES)
				assertZero(m, "juno.DeviceCompute.site." + site + "." + phase);
		for (String k : List.of(".count", ".bytes", ".timed_count", ".total_ms", ".estimated_total_ms"))
			assertThat(m.get("juno.DeviceStaging.HOST" + k)).as("juno.DeviceStaging.HOST" + k).isEqualTo(0.0);
	}

	@Test
	void compute_sumsTotalsPerSiteAndPhase() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			compute("gemm_half", "prefill", 154, 154, 300_000_000);
			compute("gemm_half", "prefill", 22, 22, 50_000_000);
			compute("gqa_attention", "prefill", 22, 22, 40_000_000);
		}));
		assertThat(m.get("juno.DeviceCompute.count")).isEqualTo(198.0);
		assertThat(m.get("juno.DeviceCompute.total_ms")).isCloseTo(390.0, within(1e-9));
		assertThat(m.get("juno.DeviceCompute.prefill.timed_count")).isEqualTo(198.0);
		assertThat(m.get("juno.DeviceCompute.site.gemm_half.prefill.count")).isEqualTo(176.0);
		assertThat(m.get("juno.DeviceCompute.site.gemm_half.prefill.total_ms")).isCloseTo(350.0, within(1e-9));
		assertThat(m.get("juno.DeviceCompute.site.gqa_attention.prefill.total_ms")).isCloseTo(40.0, within(1e-9));
		assertZero(m, "juno.DeviceCompute.site.gemm_fp32.prefill");
	}

	@Test
	void compute_decodeIsCountedButNotTimed() throws Exception {
		Map<String, Double> m = extract(record(() -> compute("mmq_packed", "decode", 7_000, 0, 0)));
		assertThat(m.get("juno.DeviceCompute.decode.count")).isEqualTo(7_000.0);
		assertThat(m.get("juno.DeviceCompute.decode.timed_count")).isEqualTo(0.0);
		assertThat(m.get("juno.DeviceCompute.decode.total_ms")).isEqualTo(0.0);
		assertThat(m.get("juno.DeviceCompute.site.mmq_packed.decode.count")).isEqualTo(7_000.0);
		assertThat(m.get("juno.DeviceCompute.prefill.count")).isEqualTo(0.0);
	}

	@Test
	void compute_reportsASiteOutsideTheFixedList() throws Exception {
		Map<String, Double> m = extract(record(() -> compute("swiglu_window", "prefill", 22, 22, 9_000_000)));
		assertThat(m.get("juno.DeviceCompute.site.swiglu_window.prefill.count")).isEqualTo(22.0);
		assertThat(m.get("juno.DeviceCompute.site.swiglu_window.prefill.total_ms")).isCloseTo(9.0, within(1e-9));
	}

	@Test
	void hostPacking_isItsOwnDirectionAndLeavesTheBusBytesAlone() throws Exception {
		Map<String, Double> m = extract(record(() -> {
			staging("pack_fp16_host", "HOST", "prefill", 154, 80_000_000, 154, 150_000_000);
			staging("cudaMemcpyAsync(xh H2D batched-gemm)", "H2D", "prefill", 154, 80_000_000, 154, 20_000_000);
		}));
		assertThat(m.get("juno.DeviceStaging.HOST.prefill.total_ms")).isCloseTo(150.0, within(1e-9));
		assertThat(m.get("juno.DeviceStaging.site.pack_fp16_host.prefill.count")).isEqualTo(154.0);
		assertThat(m.get("juno.DeviceStaging.H2D.bytes")).isEqualTo(80_000_000.0);
		assertThat(m.get("juno.DeviceStaging.H2D.total_ms")).isCloseTo(20.0, within(1e-9));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static void assertZero(Map<String, Double> m, String prefix) {
		for (String k : List.of(".count", ".timed_count", ".total_ms"))
			assertThat(m.get(prefix + k)).as(prefix + k).isEqualTo(0.0);
	}

	private static void compute(String site, String phase, long count, long timedCount, long computeNanos) {
		SyntheticDeviceCompute ev = new SyntheticDeviceCompute();
		ev.site = site;
		ev.phase = phase;
		ev.count = count;
		ev.timedCount = timedCount;
		ev.computeNanos = computeNanos;
		ev.commit();
	}

	private static void staging(String site, String direction, String phase, long copies, long bytes,
			long timedCopies, long transferNanos) {
		JfrMetricsExtractorDeviceSpansTest.SyntheticDeviceStaging ev = new JfrMetricsExtractorDeviceSpansTest.SyntheticDeviceStaging();
		ev.site = site;
		ev.direction = direction;
		ev.phase = phase;
		ev.copies = copies;
		ev.bytes = bytes;
		ev.timedCopies = timedCopies;
		ev.transferNanos = transferNanos;
		ev.commit();
	}

	private Path record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("test-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			rec.enable("juno.DeviceCompute");
			rec.enable("juno.DeviceStaging");
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

	@Name("juno.DeviceCompute")
	@Label("Device Compute")
	@Category({ "Juno", "GPU" })
	@StackTrace(false)
	public static class SyntheticDeviceCompute extends Event {
		public String site;
		public String phase;
		public long count;
		public long timedCount;
		public long computeNanos;
	}
}
