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

import cab.ml.juno.node.RmsNormRoundTripMicrobench.Cell;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.DeviceMemoryDelta;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.Reading;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static cab.ml.juno.node.ResidentChainMicrobench.LANE_DEVICE_ONLY;
import static cab.ml.juno.node.ResidentChainMicrobench.LANE_OP_AT_A_TIME;
import static cab.ml.juno.node.ResidentChainMicrobench.LANE_RESIDENT_CHAIN;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.LANE_CPU_SCALAR;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * The scoring and reporting half of {@link ResidentChainMicrobench}, covered
 * without a device: which lane a ratio is taken against and in which direction
 * decides what the residency measurement is read as, so it is pinned here
 * rather than trusted.
 */
@DisplayName("ResidentChainMicrobench - chain scoring and report")
class ResidentChainMicrobenchTest {

	private static Reading lane(String name, double... repMs) {
		return new Reading(name, repMs);
	}

	/** A width with all four lanes, CPU at 1.0 ms. */
	private static Cell cell(String label, int batch, double opAtATime, double chain, double deviceOnly) {
		return new Cell(label, batch, 2048, List.of(
				lane(LANE_CPU_SCALAR, 1.0, 1.0, 1.0),
				lane(LANE_OP_AT_A_TIME, opAtATime, opAtATime, opAtATime),
				lane(LANE_RESIDENT_CHAIN, chain, chain, chain),
				lane(LANE_DEVICE_ONLY, deviceOnly, deviceOnly, deviceOnly)));
	}

	@Test
	@DisplayName("cost against op-at-a-time is lane over op-at-a-time: a cheaper chain reads below 1")
	void costOverOpAtATime_isLaneOverBaseline() {
		Cell c = cell("decode", 1, 0.040, 0.020, 0.010);
		assertThat(ResidentChainMicrobench.costOverOpAtATime(c, LANE_RESIDENT_CHAIN)).isCloseTo(0.5, within(1e-12));
		assertThat(ResidentChainMicrobench.costOverOpAtATime(c, LANE_OP_AT_A_TIME)).isCloseTo(1.0, within(1e-12));
		assertThat(ResidentChainMicrobench.costOverOpAtATime(c, LANE_DEVICE_ONLY)).isCloseTo(0.25, within(1e-12));
	}

	@Test
	@DisplayName("speedup against the CPU chain keeps the step-one orientation: faster reads above 1")
	void speedupOverCpu_isCpuOverLane() {
		Cell c = cell("prefill", 512, 4.0, 0.5, 0.1);
		assertThat(c.speedupOverCpu(LANE_RESIDENT_CHAIN)).isCloseTo(2.0, within(1e-12));
		assertThat(c.speedupOverCpu(LANE_OP_AT_A_TIME)).isCloseTo(0.25, within(1e-12));
	}

	@Test
	@DisplayName("a width without the op-at-a-time lane cannot be scored for chaining")
	void missingOpAtATimeLane_isRejected() {
		Cell c = new Cell("decode", 1, 2048, List.of(
				lane(LANE_CPU_SCALAR, 1.0),
				lane(LANE_RESIDENT_CHAIN, 0.5)));
		assertThatThrownBy(() -> ResidentChainMicrobench.costOverOpAtATime(c, LANE_RESIDENT_CHAIN))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining(LANE_OP_AT_A_TIME);
	}

	@Test
	@DisplayName("the report carries both widths, all four lanes, both ratio columns and the chain definition")
	void report_carriesBothWidthsAndAllLanes() {
		List<Cell> cells = List.of(cell("decode", 1, 0.040, 0.020, 0.010), cell("prefill", 512, 4.0, 0.5, 0.1));
		String report = ResidentChainMicrobench.formatReport(cells,
				new DeviceMemoryDelta(8_000_000_000L, 8_000_000_000L, 8_500_000_000L), "test-host", 512, 64, 10000f);

		assertThat(report).contains("| decode | 1 | 2048 |", "| prefill | 512 | 2048 |");
		for (String name : List.of(LANE_CPU_SCALAR, LANE_OP_AT_A_TIME, LANE_RESIDENT_CHAIN, LANE_DEVICE_ONLY))
			assertThat(report).contains("| " + name + " |");
		assertThat(report).contains("speedup vs CPU scalar", "cost vs op-at-a-time");
		// Resident chain at decode: 1.0 / 0.020 = 50.00x faster than CPU, 0.020 / 0.040 = 0.50 of op-at-a-time.
		assertThat(report).contains("| 50.00x | 0.50 |");
		assertThat(report).contains("RMS norm", "RoPE", "head size 64", "base 10000", "position 512");
		assertThat(report).contains("not returned: 0 bytes");
	}

	@Test
	@DisplayName("the per-width reading names the two ratios the residency result is read against")
	void report_statesPerWidthReading() {
		String report = ResidentChainMicrobench.formatReport(
				List.of(cell("decode", 1, 0.040, 0.030, 0.010)),
				new DeviceMemoryDelta(1L, 1L, 2L), "h", 512, 64, 10000f);
		assertThat(report).contains("decode (batch 1): resident chain 33.33x the CPU scalar chain, 0.75 of op-at-a-time");
	}

	@Test
	@DisplayName("an unscorable row is marked, not averaged into the reading")
	void unscorableRow_isMarked() {
		Cell noisy = new Cell("decode", 1, 2048, List.of(
				lane(LANE_CPU_SCALAR, 1.0, 1.0, 1.0),
				lane(LANE_OP_AT_A_TIME, 0.04, 0.04, 0.04),
				lane(LANE_RESIDENT_CHAIN, 0.02, 0.03, 0.02),
				lane(LANE_DEVICE_ONLY, 0.01, 0.01, 0.01)));
		String report = ResidentChainMicrobench.formatReport(List.of(noisy),
				new DeviceMemoryDelta(1L, 1L, 2L), "h", 512, 64, 10000f);
		assertThat(report).containsPattern("\\| " + LANE_RESIDENT_CHAIN + " \\|.*\\| no \\|");
		// Named explicitly, so the rule sentence in the header cannot satisfy this on its own.
		assertThat(report).contains("Rows not scorable, re-run before reading them: decode " + LANE_RESIDENT_CHAIN);
	}

	@Test
	@DisplayName("a clean run names no unscorable rows")
	void cleanRun_namesNoUnscorableRows() {
		String report = ResidentChainMicrobench.formatReport(
				List.of(cell("decode", 1, 0.040, 0.020, 0.010)),
				new DeviceMemoryDelta(1L, 1L, 2L), "h", 512, 64, 10000f);
		assertThat(report).doesNotContain("Rows not scorable");
	}
}
