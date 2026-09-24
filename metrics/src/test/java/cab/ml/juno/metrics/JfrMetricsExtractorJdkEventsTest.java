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
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.LockSupport;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordingFile;

/**
 * The JDK-event side of the extractor: GC pauses, allocation, hot methods and
 * lock/park time. Until these are consumed, no performance gate in this project
 * can report a GC pause or an allocation rate, so a short measurement window
 * contaminated by one long pause is indistinguishable from a real regression.
 *
 * <p>The sampled and periodic JDK events cannot be synthesised the way the
 * {@code juno.*} events are — they are emitted by the JVM, not by this code, and
 * how many of them land in a given window is timing-dependent. So rather than
 * asserting an absolute count, every value assertion here re-reads the same
 * recording with {@link RecordingFile} and computes what the extractor should
 * have produced from it. The assertions are then exact whether the window caught
 * three events or three hundred, and they still hold when it caught none.
 */
class JfrMetricsExtractorJdkEventsTest {

	private static final String GC_PAUSE = "jdk.GCPhasePause";
	private static final String THREAD_ALLOC = "jdk.ThreadAllocationStatistics";
	private static final String ALLOC_SAMPLE = "jdk.ObjectAllocationSample";
	private static final String EXEC_SAMPLE = "jdk.ExecutionSample";
	private static final String MONITOR_ENTER = "jdk.JavaMonitorEnter";
	private static final String THREAD_PARK = "jdk.ThreadPark";

	@TempDir
	Path tmp;

	/** Sink for the allocation loops, so the JIT cannot delete them. */
	private static volatile Object blackhole;

	@Test
	void recordingWithNoJdkEvents_yieldsZeroesNotMissingKeys() throws Exception {
		// The stock "default" configuration already enables several of these, so a
		// recording that genuinely contains none has to turn them off by name.
		Path jfr = record(rec -> {
			rec.enable("juno.TokenProduced");
			for (String jdkEvent : List.of(GC_PAUSE, THREAD_ALLOC, ALLOC_SAMPLE, EXEC_SAMPLE, MONITOR_ENTER,
					THREAD_PARK))
				rec.disable(jdkEvent);
		}, () -> {
		});
		Map<String, Double> m = extract(jfr);

		assertThat(m).containsKeys(
				"jdk.GCPhasePause.count", "jdk.GCPhasePause.max_ms", "jdk.GCPhasePause.total_ms",
				"jdk.ThreadAllocationStatistics.bytes_total",
				"jdk.ObjectAllocationSample.count", "jdk.ObjectAllocationSample.weight_total_bytes",
				"jdk.ExecutionSample.count",
				"jdk.JavaMonitorEnter.count", "jdk.JavaMonitorEnter.total_ms",
				"jdk.ThreadPark.count", "jdk.ThreadPark.total_ms");
		assertThat(m.get("jdk.GCPhasePause.count")).isZero();
		assertThat(m.get("jdk.GCPhasePause.max_ms")).isZero();
		assertThat(m.get("jdk.GCPhasePause.total_ms")).isZero();
		assertThat(m.get("jdk.ThreadAllocationStatistics.bytes_total")).isZero();
		assertThat(m.get("jdk.ObjectAllocationSample.count")).isZero();
		assertThat(m.get("jdk.ExecutionSample.count")).isZero();
		assertThat(m.get("jdk.JavaMonitorEnter.total_ms")).isZero();
		assertThat(m.get("jdk.ThreadPark.total_ms")).isZero();
	}

	@Test
	void gcPauses_areCountedAndTheLongestIsReportedSeparatelyFromTheTotal() throws Exception {
		Path jfr = record(rec -> rec.enable(GC_PAUSE), () -> {
			for (int i = 0; i < 3; i++) {
				blackhole = new byte[8 * 1024 * 1024];
				System.gc();
			}
		});

		List<RecordedEvent> pauses = eventsOfType(jfr, GC_PAUSE);
		Map<String, Double> m = extract(jfr);

		double expectedTotalMs = pauses.stream().mapToDouble(e -> e.getDuration().toNanos() / 1_000_000.0).sum();
		double expectedMaxMs = pauses.stream().mapToDouble(e -> e.getDuration().toNanos() / 1_000_000.0).max()
				.orElse(0.0);

		assertThat(m.get("jdk.GCPhasePause.count")).isEqualTo((double) pauses.size());
		assertThat(m.get("jdk.GCPhasePause.total_ms")).isCloseTo(expectedTotalMs, org.assertj.core.data.Offset.offset(1e-6));
		assertThat(m.get("jdk.GCPhasePause.max_ms")).isCloseTo(expectedMaxMs, org.assertj.core.data.Offset.offset(1e-6));

		// Three explicit collections in the window: if this is zero the JVM ignored
		// System.gc() and the rest of the assertions above are vacuous.
		assertThat(pauses).as("explicit System.gc() produced no GCPhasePause events").isNotEmpty();
		assertThat(m.get("jdk.GCPhasePause.max_ms")).isLessThanOrEqualTo(m.get("jdk.GCPhasePause.total_ms"));
	}

	/**
	 * {@code jdk.ThreadAllocationStatistics#allocated} is a running per-thread total,
	 * not a per-event delta, so the bytes a run allocated is the sum over threads of
	 * each thread's largest sample. Summing every event instead multiplies the answer
	 * by roughly the number of samples per thread, which is what makes a bytes-per-token
	 * ceiling read as passing when it is not.
	 */
	@Test
	void threadAllocationStatistics_sumsEachThreadsLatestTotalNotEverySample() throws Exception {
		Path jfr = record(
				rec -> rec.enable(THREAD_ALLOC).with("period", "100 ms"),
				() -> {
					for (int i = 0; i < 40; i++) {
						blackhole = new byte[1024 * 1024];
						Thread.sleep(10);
					}
				});

		List<RecordedEvent> events = eventsOfType(jfr, THREAD_ALLOC);
		Map<String, Double> m = extract(jfr);

		Map<Long, Long> maxPerThread = new HashMap<>();
		double naiveSum = 0.0;
		for (RecordedEvent ev : events) {
			long allocated = ev.getLong("allocated");
			naiveSum += allocated;
			long threadId = ev.getThread("thread") == null ? -1L : ev.getThread("thread").getJavaThreadId();
			maxPerThread.merge(threadId, allocated, Math::max);
		}
		double expected = maxPerThread.values().stream().mapToDouble(Long::doubleValue).sum();

		assertThat(events).as("no ThreadAllocationStatistics events in the window").isNotEmpty();
		assertThat(m.get("jdk.ThreadAllocationStatistics.bytes_total")).isEqualTo(expected);

		// The whole point of the per-thread-maximum rule: where any thread was sampled
		// more than once, the naive sum is strictly larger and would be wrong.
		if (events.size() > maxPerThread.size())
			assertThat(m.get("jdk.ThreadAllocationStatistics.bytes_total")).isLessThan(naiveSum);
	}

	@Test
	void objectAllocationSamples_areWeighedAndAttributedToTheirTopSite() throws Exception {
		Path jfr = record(
				rec -> rec.enable(ALLOC_SAMPLE).with("throttle", "300/s"),
				() -> allocateHard(600));

		List<RecordedEvent> samples = eventsOfType(jfr, ALLOC_SAMPLE);
		Map<String, Double> m = extract(jfr);

		double expectedWeight = samples.stream().mapToDouble(e -> e.getLong("weight")).sum();
		assertThat(m.get("jdk.ObjectAllocationSample.count")).isEqualTo((double) samples.size());
		assertThat(m.get("jdk.ObjectAllocationSample.weight_total_bytes")).isEqualTo(expectedWeight);

		assertThat(samples).as("sustained allocation produced no ObjectAllocationSample events").isNotEmpty();
		Map<String, Double> sites = labelsUnder(m, "jdk.ObjectAllocationSample.top_sites.", ".bytes");
		assertThat(sites).isNotEmpty();
		assertThat(sites.values().stream().mapToDouble(Double::doubleValue).sum())
				.isLessThanOrEqualTo(expectedWeight);
		assertThat(sites.keySet()).allSatisfy(k -> assertThat(k).matches("[A-Za-z0-9_]+"));
	}

	@Test
	void executionSamples_areAttributedToTheirTopMethod() throws Exception {
		Path jfr = record(
				rec -> rec.enable(EXEC_SAMPLE).with("period", "1 ms"),
				() -> burnCpu(Duration.ofMillis(400)));

		List<RecordedEvent> samples = eventsOfType(jfr, EXEC_SAMPLE);
		Map<String, Double> m = extract(jfr);

		assertThat(m.get("jdk.ExecutionSample.count")).isEqualTo((double) samples.size());
		assertThat(samples).as("400ms of busy CPU produced no ExecutionSample events").isNotEmpty();

		Map<String, Double> methods = labelsUnder(m, "jdk.ExecutionSample.top_methods.", ".samples");
		assertThat(methods).isNotEmpty();
		assertThat(methods.values().stream().mapToDouble(Double::doubleValue).sum())
				.isLessThanOrEqualTo((double) samples.size());
		assertThat(methods.keySet()).allSatisfy(k -> assertThat(k).matches("[A-Za-z0-9_]+"));

		String expectedTop = topFrameHistogram(samples).entrySet().stream()
				.max(Map.Entry.comparingByValue())
				.map(Map.Entry::getKey)
				.orElseThrow();
		assertThat(methods).containsKey(expectedTop);
	}

	/**
	 * Attribution is for reading, not for machine gates, so it stays bounded: an
	 * unbounded histogram would put one JSON key per distinct method into every
	 * published result file.
	 */
	@Test
	void attributionHistogramsAreBoundedToTheTopTen() throws Exception {
		Path jfr = record(
				rec -> {
					rec.enable(EXEC_SAMPLE).with("period", "1 ms");
					rec.enable(ALLOC_SAMPLE).with("throttle", "300/s");
				},
				() -> {
					burnCpu(Duration.ofMillis(200));
					allocateHard(300);
				});
		Map<String, Double> m = extract(jfr);

		assertThat(labelsUnder(m, "jdk.ExecutionSample.top_methods.", ".samples")).hasSizeLessThanOrEqualTo(10);
		assertThat(labelsUnder(m, "jdk.ObjectAllocationSample.top_sites.", ".bytes")).hasSizeLessThanOrEqualTo(10);
	}

	@Test
	void monitorContentionAndParkTimeAreSummed() throws Exception {
		Object lock = new Object();
		Path jfr = record(
				rec -> {
					rec.enable(MONITOR_ENTER).withThreshold(Duration.ofMillis(10));
					rec.enable(THREAD_PARK).withThreshold(Duration.ofMillis(10));
				},
				() -> {
					Thread blocked = new Thread(() -> {
						synchronized (lock) {
							blackhole = lock;
						}
					}, "jdk-events-test-blocked");
					synchronized (lock) {
						blocked.start();
						Thread.sleep(120);
					}
					blocked.join();
					LockSupport.parkNanos(Duration.ofMillis(120).toNanos());
				});

		Map<String, Double> m = extract(jfr);
		List<RecordedEvent> enters = eventsOfType(jfr, MONITOR_ENTER);
		List<RecordedEvent> parks = eventsOfType(jfr, THREAD_PARK);

		assertThat(m.get("jdk.JavaMonitorEnter.count")).isEqualTo((double) enters.size());
		assertThat(m.get("jdk.ThreadPark.count")).isEqualTo((double) parks.size());
		assertThat(m.get("jdk.JavaMonitorEnter.total_ms")).isCloseTo(
				enters.stream().mapToDouble(e -> e.getDuration().toNanos() / 1_000_000.0).sum(),
				org.assertj.core.data.Offset.offset(1e-6));
		assertThat(m.get("jdk.ThreadPark.total_ms")).isCloseTo(
				parks.stream().mapToDouble(e -> e.getDuration().toNanos() / 1_000_000.0).sum(),
				org.assertj.core.data.Offset.offset(1e-6));

		assertThat(enters).as("a 120ms held monitor produced no JavaMonitorEnter event").isNotEmpty();
		assertThat(parks).as("a 120ms park produced no ThreadPark event").isNotEmpty();
	}

	/** The JDK bucket must not disturb the juno.* metrics sharing the same pass. */
	@Test
	void junoMetricsAreUnaffectedByTheJdkBucket() throws Exception {
		Path jfr = record(rec -> {
			rec.enable(GC_PAUSE);
			rec.enable("juno.TokenProduced");
		}, System::gc);
		Map<String, Double> m = extract(jfr);
		assertThat(m.get("juno.TokenProduced.count")).isZero();
		assertThat(m.get("juno.MatVec.count")).isZero();
		assertThat(m.get("juno.ForwardPass.count")).isZero();
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private Path record(RecordingSetup setup, ThrowingRunnable body) throws Exception {
		Path jfr = tmp.resolve("jdk-events-" + System.nanoTime() + ".jfr");
		Configuration cfg = Configuration.getConfiguration("default");
		try (Recording rec = new Recording(cfg)) {
			setup.apply(rec);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			Thread.sleep(50);
			rec.stop();
		}
		assertThat(Files.size(jfr)).isGreaterThan(0);
		return jfr;
	}

	private static List<RecordedEvent> eventsOfType(Path jfr, String typeName) throws Exception {
		List<RecordedEvent> out = new ArrayList<>();
		try (RecordingFile rf = new RecordingFile(jfr)) {
			while (rf.hasMoreEvents()) {
				RecordedEvent ev = rf.readEvent();
				if (typeName.equals(ev.getEventType().getName()))
					out.add(ev);
			}
		}
		return out;
	}

	private static Map<String, Integer> topFrameHistogram(List<RecordedEvent> samples) {
		Map<String, Integer> hist = new HashMap<>();
		for (RecordedEvent ev : samples) {
			RecordedStackTrace st = ev.getStackTrace();
			if (st == null || st.getFrames().isEmpty())
				continue;
			RecordedFrame top = st.getFrames().get(0);
			String name = top.getMethod().getType().getName() + "." + top.getMethod().getName();
			hist.merge(name.replaceAll("[^A-Za-z0-9]", "_"), 1, Integer::sum);
		}
		return hist;
	}

	/** The attribution labels under {@code prefix}, with the metric suffix stripped off. */
	private static Map<String, Double> labelsUnder(Map<String, Double> m, String prefix, String suffix) {
		Map<String, Double> out = new HashMap<>();
		for (Map.Entry<String, Double> e : m.entrySet()) {
			String k = e.getKey();
			if (k.startsWith(prefix) && k.endsWith(suffix))
				out.put(k.substring(prefix.length(), k.length() - suffix.length()), e.getValue());
		}
		return out;
	}

	private static void allocateHard(int megabytes) {
		for (int i = 0; i < megabytes; i++) {
			byte[] b = new byte[1024 * 1024];
			b[0] = (byte) i;
			blackhole = b;
		}
	}

	private static void burnCpu(Duration budget) {
		long deadline = System.nanoTime() + budget.toNanos();
		long acc = 0;
		while (System.nanoTime() < deadline)
			for (int i = 0; i < 100_000; i++)
				acc += i * 31L;
		blackhole = acc;
	}

	private static Map<String, Double> extract(Path jfr) throws Exception {
		ModelsConfig.ModelEntry entry = new ModelsConfig.ModelEntry("tiny", "tiny.gguf");
		return JfrMetricsExtractor.extract(jfr, entry).getMetrics();
	}

	@FunctionalInterface
	private interface ThrowingRunnable {
		void run() throws Exception;
	}

	@FunctionalInterface
	private interface RecordingSetup {
		void apply(Recording rec);
	}
}
