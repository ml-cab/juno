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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.RmsNormRoundTripMicrobench.Cell;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.DeviceMemoryDelta;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.Reading;

/**
 * Scoring and reporting logic of the RMS-norm host-round-trip microbench.
 *
 * <p>Measurement itself needs a CUDA device; everything that decides whether a
 * reading is usable and what ratio it reports does not, and that is what the
 * activation-residency decision is read off, so it is covered here.
 */
@DisplayName("RMS-norm round-trip microbench scoring")
class RmsNormRoundTripMicrobenchTest {

	private static Reading reading(String lane, double... repMs) {
		return new Reading(lane, repMs);
	}

	@Nested
	@DisplayName("Reading")
	class ReadingTest {

		@Test
		@DisplayName("median of an odd rep count is the middle value, regardless of order")
		void median_odd() {
			assertEquals(5.0, reading("cpu", 9.0, 1.0, 5.0).medianMs(), 1e-9);
		}

		@Test
		@DisplayName("median of an even rep count averages the two middle values")
		void median_even() {
			assertEquals(3.0, reading("cpu", 4.0, 1.0, 2.0, 5.0).medianMs(), 1e-9);
		}

		@Test
		@DisplayName("min and max are the extremes, not the first and last rep")
		void min_max() {
			Reading r = reading("cpu", 5.0, 1.0, 9.0);
			assertEquals(1.0, r.minMs(), 1e-9);
			assertEquals(9.0, r.maxMs(), 1e-9);
		}

		@Test
		@DisplayName("spread is max-minus-min as a percentage of the median")
		void spread_pct() {
			// median 10, min 9, max 12 -> 3/10 = 30%
			assertEquals(30.0, reading("cpu", 9.0, 10.0, 12.0).spreadPct(), 1e-9);
		}

		@Test
		@DisplayName("a row whose reps disagree by more than 15% of their median is not scorable")
		void scorable_threshold() {
			// spread 15.0% exactly -> still scorable (the rule rejects "more than 15%")
			assertTrue(reading("cpu", 10.0, 10.0, 11.5).scorable());
			// spread 20% -> not scorable
			assertFalse(reading("cpu", 10.0, 10.0, 12.0).scorable());
		}

		@Test
		@DisplayName("a reading needs at least one repetition")
		void rejects_empty() {
			assertThrows(IllegalArgumentException.class, () -> reading("cpu"));
		}

		@Test
		@DisplayName("repetitions are defensively copied, so a caller cannot mutate a reading")
		void defensive_copy() {
			double[] reps = { 1.0, 2.0, 3.0 };
			Reading r = new Reading("cpu", reps);
			reps[0] = 99.0;
			r.repMs()[1] = 99.0;
			assertEquals(2.0, r.medianMs(), 1e-9);
		}
	}

	@Nested
	@DisplayName("Cell")
	class CellTest {

		private Cell decodeCell() {
			return new Cell("decode", 1, 2048, List.of(
					reading(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR, 0.015, 0.015, 0.015),
					reading(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP, 0.168, 0.168, 0.168)));
		}

		@Test
		@DisplayName("a lane slower than CPU scalar scores below 1.0x")
		void slower_lane_scores_below_one() {
			// 0.015 / 0.168 = 0.0893x -- the ~11x-slower decode finding, in the units
			// the tier threshold is written in.
			assertEquals(0.0893, decodeCell().speedupOverCpu(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP), 1e-4);
		}

		@Test
		@DisplayName("the CPU scalar lane scores exactly 1.0x against itself")
		void cpu_lane_is_unity() {
			assertEquals(1.0, decodeCell().speedupOverCpu(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR), 1e-9);
		}

		@Test
		@DisplayName("a cell without a CPU scalar lane cannot produce a ratio")
		void requires_cpu_lane() {
			Cell c = new Cell("decode", 1, 2048, List.of(reading("gpu-round-trip", 1.0)));
			assertThrows(IllegalStateException.class, () -> c.speedupOverCpu("gpu-round-trip"));
		}

		@Test
		@DisplayName("an unknown lane name is rejected rather than silently scored")
		void rejects_unknown_lane() {
			assertThrows(IllegalArgumentException.class, () -> decodeCell().speedupOverCpu("no-such-lane"));
		}
	}

	@Nested
	@DisplayName("Device memory")
	class DeviceMemoryTest {

		@Test
		@DisplayName("free VRAM returning to its starting level reports no leak")
		void no_leak() {
			assertEquals(0L, new DeviceMemoryDelta(8_000_000_000L, 8_000_000_000L, 8_589_934_592L).leakedBytes());
		}

		@Test
		@DisplayName("free VRAM ending lower than it started reports the shortfall as leaked")
		void leak_is_the_shortfall() {
			assertEquals(1024L, new DeviceMemoryDelta(8_000_000_000L, 7_999_998_976L, 8_589_934_592L).leakedBytes());
		}

		@Test
		@DisplayName("free VRAM ending higher than it started is not a negative leak")
		void freed_more_than_taken() {
			assertEquals(0L, new DeviceMemoryDelta(7_000_000_000L, 8_000_000_000L, 8_589_934_592L).leakedBytes());
		}

		@Test
		@DisplayName("retention too large to be this harness's scratch is flagged, not scored")
		void unaccounted_retention_is_flagged() {
			// 4.1 GB unreturned on an 8.6 GB device: a stray process on the GPU, not
			// two activation buffers. Observed for real during this harness's own bring-up.
			assertTrue(new DeviceMemoryDelta(4_317_249_536L, 175_702_016L, 8_589_934_592L)
					.retentionUnaccounted());
			// 3.9 MB unreturned is the per-thread device scratch, and must not be flagged.
			assertFalse(new DeviceMemoryDelta(7_784_890_368L, 7_780_958_208L, 8_589_934_592L)
					.retentionUnaccounted());
		}

		@Test
		@DisplayName("a failed device query is not reported as a clean run")
		void failed_query_is_not_a_clean_run() {
			// memGetInfo answers {0, 0} when the query fails; leakedBytes() is then 0,
			// which must not be published as evidence of no retention.
			assertFalse(new DeviceMemoryDelta(0L, 0L, 0L).queried());
			assertTrue(new DeviceMemoryDelta(8_000_000_000L, 8_000_000_000L, 8_589_934_592L).queried());
		}
	}

	@Nested
	@DisplayName("Report")
	class ReportTest {

		private String report() {
			Cell decode = new Cell("decode", 1, 2048, List.of(
					reading(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR, 0.015, 0.015, 0.015),
					reading(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP, 0.160, 0.168, 0.176)));
			Cell prefill = new Cell("prefill", 512, 2048, List.of(
					reading(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR, 7.5, 7.5, 7.5),
					reading(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP, 6.0, 12.0, 18.0)));
			return RmsNormRoundTripMicrobench.formatReport(List.of(decode, prefill),
					new DeviceMemoryDelta(8_000_000_000L, 8_000_000_000L, 8_589_934_592L), "GTX 1080");
		}

		@Test
		@DisplayName("both widths appear, so a decode result is never read as a prefill one")
		void both_widths_reported() {
			String md = report();
			assertTrue(md.contains("decode"), md);
			assertTrue(md.contains("prefill"), md);
			assertTrue(md.contains("512"), md);
		}

		@Test
		@DisplayName("each lane's ratio against CPU scalar is printed")
		void ratios_printed() {
			assertTrue(report().contains("0.09x"), report());
		}

		@Test
		@DisplayName("a row whose repetitions disagree too much is marked unscorable in the table")
		void unscorable_row_marked() {
			// the prefill GPU row spans 6..18 around a median of 12 -> 100% spread,
			// while every other row's reps are identical
			String md = report();
			assertTrue(md.contains("| no |"), md);
			assertTrue(md.contains("| yes |"), md);
		}

		@Test
		@DisplayName("the device-memory delta is reported, so a leak is visible without a crash")
		void device_memory_reported() {
			assertTrue(report().toLowerCase(java.util.Locale.ROOT).contains("vram"), report());
		}

		@Test
		@DisplayName("the host label is recorded, since a ratio is only meaningful per host")
		void host_label_reported() {
			assertTrue(report().contains("GTX 1080"), report());
		}

		@Test
		@DisplayName("an unanswered device query prints no retention figure at all")
		void failed_device_query_prints_no_figure() {
			Cell decode = new Cell("decode", 1, 2048, List.of(
					reading(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR, 0.015),
					reading(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP, 0.168)));
			String md = RmsNormRoundTripMicrobench.formatReport(List.of(decode),
					new DeviceMemoryDelta(0L, 0L, 0L), "GTX 1080");
			assertTrue(md.contains("makes no claim about device-memory retention"), md);
			assertFalse(md.contains("not returned: 0 bytes"), md);
		}

		@Test
		@DisplayName("a retention figure too large to be ours says so in the report")
		void unaccounted_retention_reported() {
			Cell decode = new Cell("decode", 1, 2048, List.of(
					reading(RmsNormRoundTripMicrobench.LANE_CPU_SCALAR, 0.015),
					reading(RmsNormRoundTripMicrobench.LANE_GPU_ROUND_TRIP, 0.168)));
			String md = RmsNormRoundTripMicrobench.formatReport(List.of(decode),
					new DeviceMemoryDelta(4_317_249_536L, 175_702_016L, 8_589_934_592L), "GTX 1080");
			assertTrue(md.contains("Re-run on an idle device"), md);
			assertTrue(md.contains("far more than this harness allocates"), md);
		}
	}
}
