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
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The host KV footprint probe: its reading of a run's line, the exact-growth check, and one real run on the kit's
 * synthetic model on the CPU backend (no model file, no GPU).
 */
class KvFootprintProbeTest {

	@Test
	void theRunsLineIsParsed() throws IOException {
		String out = "loading\n" + KvFootprintRunMain.PREFIX + "{\"positions\":2112,\"hostKvBytes\":830472192,"
				+ "\"hostKvBytesPerPosition\":393216,\"liveHeapAfterLoadBytes\":3613392896,"
				+ "\"liveHeapWithRequestBytes\":5000000000,\"maxHeapBytes\":12884901888,"
				+ "\"kv\":\"cache-type-k=f16 cache-type-v=f16\",\"gpuAttention\":true}\nbye\n";
		KvFootprintProbe.Run run = KvFootprintProbe.parse(out, 0);
		assertThat(run.positions()).isEqualTo(2112);
		assertThat(run.hostKvBytes()).isEqualTo(830_472_192L);
		assertThat(run.gpuAttention()).isTrue();
		assertThatCode(() -> KvFootprintProbe.requireExactGrowth(run)).doesNotThrowAnyException();
		List<ProbeReport.Reading> r = KvFootprintProbe.readings(run);
		assertThat(r).extracting(ProbeReport.Reading::name).containsExactly("host KV of the request, 2112 positions",
				"host KV bytes per position", "live heap after load", "live heap with the request held");
		assertThat(r.get(0).values()).containsExactly(830_472_192 / (double) (1 << 20));
		assertThat(r).allSatisfy(x -> assertThat(x.scored()).isFalse());
	}

	@Test
	void aRunWithoutItsLineFailsNamingTheExit() {
		assertThatThrownBy(() -> KvFootprintProbe.parse("OutOfMemoryError\n", 1)).isInstanceOf(IOException.class)
				.hasMessageContaining("exit 1");
	}

	@Test
	void aHostKvPastTheRequestsPositionsFails() {
		KvFootprintProbe.Run doubled = new KvFootprintProbe.Run(2112, 4096L * 393216, 393216, 1, 1, 1, "f16", true);
		assertThatThrownBy(() -> KvFootprintProbe.requireExactGrowth(doubled)).hasMessageContaining("grew past");
		KvFootprintProbe.Run unknown = new KvFootprintProbe.Run(2112, 1, 0, 1, 1, 1, "f16", true);
		assertThatThrownBy(() -> KvFootprintProbe.requireExactGrowth(unknown)).hasMessageContaining("per position");
	}

	/** One real run, CPU backend, synthetic model: the request holds exactly its positions of host KV. */
	@Test
	void theSyntheticModelHoldsExactlyItsPositions(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new KvFootprintProbe().run("gpu.kv-footprint.synthetic",
				Map.of("model", model.toString(), "tokens", "90", "generated", "6", "gpu", "false", "childHeap", "1g"));
		assertThat(r.readings()).hasSize(4);
		assertThat(r.readings().get(0).values()).singleElement().satisfies(v -> assertThat(v).isPositive());
		assertThat(r.facts()).containsEntry("tokens", 90).containsEntry("generated", 6)
				.containsEntry("backend", "cpu").containsEntry("kv", "cache-type-k=f16 cache-type-v=f16");
	}
}
