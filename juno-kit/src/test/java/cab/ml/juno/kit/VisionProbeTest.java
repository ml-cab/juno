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
package cab.ml.juno.kit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

class VisionProbeTest {

	/** Two requests in one recording, every prompt token through the single-token pass (moondream2, 12 threads). */
	private static ObjectNode recording() {
		ObjectNode m = Json.MAPPER.createObjectNode();
		m.put("juno.ForwardPass.decode.count", 1535.0);
		m.put("juno.ForwardPass.decode.total_ms", 281557.419432);
		m.put("juno.ForwardPass.prefill.count", 2.0);
		m.put("juno.ForwardPass.prefill.total_ms", 1706.306547);
		m.put("juno.MatVec.backend.cpu.count", 222865.0);
		m.put("juno.MatVec.backend.cuda.count", 0.0);
		m.put("juno.MatVec.backend.cuda_resident.count", 0.0);
		return m;
	}

	@Test
	void decodeRateIsTheMeanSingleTokenPassNotGeneratedTokensOverTheWholeRecording() {
		var r = VisionProbe.readings("threads 12: ", false, 207398, recording());
		assertThat(r).extracting(ProbeReport.Reading::name).containsExactly("threads 12: latency",
				"threads 12: decode t/s");
		assertThat(r.get(0).values()).containsExactly(207398.0);
		assertThat(r.get(1).values().get(0)).isCloseTo(1535 / 281.557419432, within(1e-9));
	}

	@Test
	void aGpuRunWhoseMatmulsAllRanOnTheHostGivesNoReadings() {
		JsonNode host = recording();
		assertThat(VisionProbe.deviceMatVecs(host)).isZero();
		assertThat(VisionProbe.readings("GPU: ", true, 205151, host)).isEmpty();

		ObjectNode device = recording();
		device.put("juno.MatVec.backend.cuda.count", 40.0);
		device.put("juno.MatVec.backend.cuda_resident.count", 2.0);
		assertThat(VisionProbe.deviceMatVecs(device)).isEqualTo(42);
		assertThat(VisionProbe.readings("GPU: ", true, 205151, device)).hasSize(2);
	}

	@Test
	void noRecordingGivesNoDecodeValue() {
		var r = VisionProbe.readings("threads 4: ", false, Double.NaN, Json.MAPPER.missingNode());
		assertThat(r).hasSize(2);
		assertThat(r).allSatisfy(x -> assertThat(x.values()).isEmpty());
	}
}
