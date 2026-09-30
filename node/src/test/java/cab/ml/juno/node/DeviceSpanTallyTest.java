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
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * The running totals behind {@code juno.DeviceStaging} and {@code juno.WeightDequant}: one event per
 * site and phase at the end of a recording, carrying exactly the work done while that recording
 * ran. No GPU needed; the counts are fed in directly.
 */
class DeviceSpanTallyTest {

	private static final String SITE = "test(copy H2D)";

	@TempDir
	Path tmp;

	@Test
	void aRecordingHoldsOneTotalPerSiteAndPhase() throws Exception {
		List<RecordedEvent> events = recordAll("a", () -> {
			DeviceSpanTally.staging(SITE, GpuBindings.H2D, 512, 1_000, 2_000);
			DeviceSpanTally.staging(SITE, GpuBindings.H2D, 512, 3_000, 4_000);
			DeviceSpanTally.staging(SITE, GpuBindings.H2D, 1, 50, -1);
		});
		RecordedEvent prefill = only(events, "prefill");
		assertThat(prefill.getString("direction")).isEqualTo("H2D");
		assertThat(prefill.getLong("copies")).isEqualTo(2);
		assertThat(prefill.getLong("bytes")).isEqualTo(4_000);
		assertThat(prefill.getLong("timedCopies")).isEqualTo(2);
		assertThat(prefill.getLong("transferNanos")).isEqualTo(6_000);
		RecordedEvent decode = only(events, "decode");
		assertThat(decode.getLong("copies")).isEqualTo(1);
		assertThat(decode.getLong("bytes")).isEqualTo(50);
		assertThat(decode.getLong("timedCopies")).isZero();
		assertThat(decode.getLong("transferNanos")).isZero();
	}

	@Test
	void workBeforeARecordingStartsIsNotInIt() throws Exception {
		Path first = tmp.resolve("outer.jfr");
		Path second = tmp.resolve("inner.jfr");
		try (Recording outer = start(first)) {
			// Counted while only the outer recording runs: must not reach the inner one.
			DeviceSpanTally.staging(SITE, GpuBindings.H2D, 512, 7_777, 1);
			try (Recording inner = start(second)) {
				DeviceSpanTally.staging(SITE, GpuBindings.H2D, 512, 100, 1);
				inner.stop();
			}
			outer.stop();
		}
		List<RecordedEvent> inner = staging(RecordingFile.readAllEvents(second));
		assertThat(inner).hasSize(1);
		assertThat(inner.get(0).getLong("bytes")).isEqualTo(100);
		long outerBytes = staging(RecordingFile.readAllEvents(first)).stream().mapToLong(e -> e.getLong("bytes"))
				.sum();
		assertThat(outerBytes).isEqualTo(7_877);
	}

	@Test
	void nothingIsCountedWithoutARecording() throws Exception {
		DeviceSpanTally.staging(SITE, GpuBindings.D2H, 512, 999_999, 1);
		DeviceSpanTally.dequant(12, DeviceStagingEvent.TIMING_DEVICE, 1);
		List<RecordedEvent> events = recordAll("empty", () -> {
		});
		assertThat(staging(events)).isEmpty();
		assertThat(events.stream().filter(e -> e.getEventType().getName().equals("juno.WeightDequant")))
				.isEmpty();
	}

	@Test
	void dequantIsTotalledPerFormatAndTiming() throws Exception {
		List<RecordedEvent> events = recordAll("dq", () -> {
			DeviceSpanTally.dequant(12, DeviceStagingEvent.TIMING_DEVICE, 1_000);
			DeviceSpanTally.dequant(12, DeviceStagingEvent.TIMING_DEVICE, 3_000);
			DeviceSpanTally.dequant(14, DeviceStagingEvent.TIMING_HOST, 500);
		});
		List<RecordedEvent> dq = events.stream().filter(e -> e.getEventType().getName().equals("juno.WeightDequant"))
				.toList();
		assertThat(dq).hasSize(2);
		RecordedEvent q4k = dq.stream().filter(e -> e.getString("format").equals("Q4_K")).findFirst().orElseThrow();
		assertThat(q4k.getString("timing")).isEqualTo("device");
		assertThat(q4k.getLong("count")).isEqualTo(2);
		assertThat(q4k.getLong("dequantNanos")).isEqualTo(4_000);
		RecordedEvent q6k = dq.stream().filter(e -> e.getString("format").equals("Q6_K")).findFirst().orElseThrow();
		assertThat(q6k.getString("timing")).isEqualTo("host");
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private List<RecordedEvent> recordAll(String name, Runnable body) throws Exception {
		Path jfr = tmp.resolve(name + ".jfr");
		try (Recording rec = start(jfr)) {
			body.run();
			rec.stop();
		}
		return RecordingFile.readAllEvents(jfr);
	}

	private static Recording start(Path destination) throws Exception {
		Recording rec = new Recording();
		rec.enable("juno.DeviceStaging");
		rec.enable("juno.WeightDequant");
		rec.setDestination(destination);
		rec.start();
		return rec;
	}

	private static List<RecordedEvent> staging(List<RecordedEvent> events) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> SITE.equals(e.getString("site"))).toList();
	}

	private static RecordedEvent only(List<RecordedEvent> events, String phase) {
		List<RecordedEvent> matching = staging(events).stream().filter(e -> phase.equals(e.getString("phase")))
				.toList();
		assertThat(matching).as(phase).hasSize(1);
		return matching.get(0);
	}
}
