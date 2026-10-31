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

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

class ProbeCatalogTest {

	private static final Map<String, Probe> BY_ID = ProbeCatalog.all().stream()
			.collect(Collectors.toMap(Probe::id, Function.identity()));

	/** Timeouts are fixed seconds; the reference host is slow, but a tester's may be slower still. */
	@Test
	void everyTimeoutIsAtLeastThreeTimesTheProbesReferenceCost() {
		assertThat(ProbeCatalog.all()).allSatisfy(p -> assertThat(p.timeoutSeconds())
				.as("%s: timeout %d s, reference cost %d s", p.id(), p.timeoutSeconds(), p.refCostSeconds())
				.isGreaterThanOrEqualTo(3 * p.refCostSeconds()));
	}

	@Test
	void everyMeasuredDevicePeakNamesAGpuProbeOfTheCatalog() {
		assertThat(ProbeCatalog.MEASURED_DEVICE_PEAK_MIB.keySet()).allSatisfy(id -> {
			assertThat(BY_ID).containsKey(id);
			assertThat(BY_ID.get(id).requirements().needsGpu()).as(id).isTrue();
		});
	}

	@Test
	void aGpuProbeNeedsItsMeasuredPeakRoundedUpTo512MiB() {
		assertThat(BY_ID.get("gpu.live.mistral-7b").requirements().minDeviceBytes()).isEqualTo(7680L << 20);
		assertThat(BY_ID.get("gpu.vision.moondream2").requirements().minDeviceBytes()).isEqualTo(512L << 20);
		assertThat(ProbeCatalog.deviceMinimum(1024)).isEqualTo(1024L << 20);
		assertThat(BY_ID.get("cpu.live.mistral-7b").requirements().minDeviceBytes()).isZero();
	}

	/** Phi-3.5-mini has no grouped-query attention: at 2048 prompt tokens its dense KV alone is about 3 GiB. */
	@Test
	void phiLongPromptEntriesHaveTheHeapTheirMeasuredLiveSetNeeds() {
		assertThat(BY_ID.get("cpu.throughput.phi-3.5-mini.2048").heap()).isEqualTo("12g");
		assertThat(BY_ID.get("gpu.throughput.phi-3.5-mini.2048").heap()).isEqualTo("12g");
		assertThat(BY_ID.get("gpu.live.phi-3.5-mini").heap()).isEqualTo("12g");
	}

	/**
	 * Three turns of 85 notes cross Phi-3.5-mini's 4,096 positions at turn 2 and end at about 6,700 prompt tokens,
	 * the longest prompt of ten turns of 25; every turn sends its whole history again, so fewer turns prefill less.
	 */
	@Test
	void contextPolicyEntriesGrowTheConversationInThreeLargeTurns() {
		for (String id : new String[] { "context-policy.phi-3.5-mini", "gpu.context-policy.phi-3.5-mini" })
			assertThat(BY_ID.get(id).args()).as(id).containsEntry("turns", "3").containsEntry("notesPerTurn", "85");
	}
}
