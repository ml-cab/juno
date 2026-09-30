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
import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * The host-side, load-time dequantization that feeds every FP16 weight upload is recorded as
 * {@code juno.WeightDequant} with its format and a host-measured duration. No GPU needed.
 */
class WeightDequantEventTest {

	@TempDir
	Path tmp;

	@Test
	void hostDequantize_recordsFormatShapeAndDuration() throws Exception {
		int rows = 4;
		int cols = 64; // two Q8_0 blocks per row
		byte[] raw = new byte[rows * (cols / 32) * 34];
		GgufReader.QuantizedTensor t = new GgufReader.QuantizedTensor("w", 8, (long) rows * cols, raw);

		Path jfr = tmp.resolve("dequant.jfr");
		float[] out;
		try (Recording rec = new Recording()) {
			rec.enable("juno.WeightDequant").withThreshold(Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			out = LlamaTransformerHandler.dequantize(t, rows, cols);
			rec.stop();
		}
		assertThat(out).hasSize(rows * cols);

		List<RecordedEvent> events = RecordingFile.readAllEvents(jfr).stream()
				.filter(e -> e.getEventType().getName().equals("juno.WeightDequant")).toList();
		assertThat(events).hasSize(1);
		RecordedEvent e = events.get(0);
		assertThat(e.getString("format")).isEqualTo("Q8_0");
		assertThat(e.getString("timing")).isEqualTo("host");
		assertThat(e.getLong("count")).isEqualTo(1);
		assertThat(e.getLong("timedCount")).isEqualTo(1);
		assertThat(e.getLong("dequantNanos")).isPositive();
	}
}
