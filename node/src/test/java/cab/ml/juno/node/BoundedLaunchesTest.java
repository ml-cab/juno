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

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link BoundedLaunches}: how a prefill window's attention and its tiled K-quant matmuls split into launches of
 * bounded work. No GPU: the split's bit-for-bit parity with one launch is {@code BoundedLaunchParityTest}'s.
 */
@DisplayName("BoundedLaunches")
class BoundedLaunchesTest {

	private static final long BUDGET = BoundedLaunches.ATTENTION_WORK_PER_LAUNCH;

	@Test
	@DisplayName("a window of up to 2048 rows is one launch on every sweep model's heads")
	void aSweepWindowIsOneLaunch() {
		int[] seqLens = window(0, 2048);
		for (GpuKernelTimings.AttentionShape s : GpuKernelTimings.ATTENTION) {
			int rpb = GqaAttentionKernel.rowsPerBlock(true, 2048);
			long perKey = BoundedLaunches.attentionWorkPerKey(s.numHeads(), s.headDim());
			assertThat(BoundedLaunches.attentionEnd(0, seqLens, 2048, rpb, 0, perKey, BUDGET)).as(s.model())
					.isEqualTo(2048);
		}
	}

	@Test
	@DisplayName("the longest window splits into whole blocks, each launch within the budget, covering every row once")
	void theLongestWindowSplitsIntoWholeBlocksWithinTheBudget() {
		int rows = DeviceKvCache.MAX_SEQ_LEN;
		int[] seqLens = window(0, rows);
		int rpb = GqaAttentionKernel.rowsPerBlock(true, rows);
		long perKey = BoundedLaunches.attentionWorkPerKey(32, 128);
		List<int[]> launches = plan(seqLens, rows, rpb, 0, perKey, BUDGET);
		assertThat(launches.size()).isGreaterThan(100);
		int expectedStart = 0;
		for (int[] l : launches) {
			assertThat(l[0]).isEqualTo(expectedStart);
			assertThat(l[0] % rpb).as("launch at row %d starts a block", l[0]).isZero();
			assertThat(BoundedLaunches.attentionWork(seqLens, l[0], l[1], 0, perKey)).as("rows %d to %d", l[0], l[1])
					.isLessThanOrEqualTo(BUDGET);
			expectedStart = l[1];
		}
		assertThat(expectedStart).isEqualTo(rows);
	}

	@Test
	@DisplayName("each launch takes as many whole blocks as fit: one more block would pass the budget")
	void eachLaunchIsAsWideAsTheBudgetAllows() {
		int rows = 8192;
		int[] seqLens = window(0, rows);
		int rpb = 32;
		long perKey = BoundedLaunches.attentionWorkPerKey(32, 128);
		for (int[] l : plan(seqLens, rows, rpb, 0, perKey, BUDGET)) {
			if (l[1] == rows)
				continue;
			int wider = Math.min(rows, l[1] + rpb);
			assertThat(BoundedLaunches.attentionWork(seqLens, l[0], wider, 0, perKey)).isGreaterThan(BUDGET);
		}
	}

	@Test
	@DisplayName("a sliding window counts only the keys a row attends")
	void aSlidingWindowCountsOnlyTheAttendedKeys() {
		int rows = 16384;
		int[] seqLens = window(0, rows);
		long perKey = BoundedLaunches.attentionWorkPerKey(32, 96);
		assertThat(BoundedLaunches.attentionWork(seqLens, 8000, 8001, 2047, perKey)).isEqualTo(2047 * perKey);
		assertThat(BoundedLaunches.attentionWork(seqLens, 100, 101, 2047, perKey)).isEqualTo(101 * perKey);
		int unwindowed = plan(seqLens, rows, 32, 0, perKey, BUDGET).size();
		int windowed = plan(seqLens, rows, 32, 2047, perKey, BUDGET).size();
		assertThat(windowed).isLessThan(unwindowed);
	}

	@Test
	@DisplayName("a later chunk of a long prompt splits by the keys it attends, not by its rows")
	void aLaterChunkSplitsByItsKeys() {
		int startPos = 30000, rows = 512;
		int[] seqLens = window(startPos, rows);
		long perKey = BoundedLaunches.attentionWorkPerKey(32, 128);
		List<int[]> launches = plan(seqLens, rows, GqaAttentionKernel.rowsPerBlock(true, rows), 0, perKey, BUDGET);
		assertThat(launches.size()).isGreaterThan(1);
	}

	@Test
	@DisplayName("a block heavier than the budget still launches, one block at a time")
	void aBlockHeavierThanTheBudgetStillLaunches() {
		int[] seqLens = window(0, 100);
		List<int[]> launches = plan(seqLens, 100, 32, 0, 1, 1);
		assertThat(launches).extracting(l -> l[1] - l[0]).containsExactly(32, 32, 32, 4);
	}

	@Test
	@DisplayName("decode rows over separate caches at the longest context stay one launch")
	void decodeRowsStayOneLaunch() {
		int[] seqLens = new int[8];
		java.util.Arrays.fill(seqLens, DeviceKvCache.MAX_SEQ_LEN);
		long perKey = BoundedLaunches.attentionWorkPerKey(32, 128);
		assertThat(BoundedLaunches.attentionEnd(0, seqLens, 8, 1, 0, perKey, BUDGET)).isEqualTo(8);
	}

	@Test
	@DisplayName("the work per key is the heads times the head width the kernel holds in registers")
	void workPerKeyUsesTheRegisterWidth() {
		assertThat(BoundedLaunches.attentionWorkPerKey(32, 64)).isEqualTo(32L * 64);
		assertThat(BoundedLaunches.attentionWorkPerKey(32, 96)).isEqualTo(32L * 128);
		assertThat(BoundedLaunches.attentionWorkPerKey(16, 128)).isEqualTo(16L * 128);
		assertThat(BoundedLaunches.attentionWorkPerKey(8, 160)).isEqualTo(8L * 256);
	}

	@Test
	@DisplayName("matmul launches take whole 64-row column tiles within the budget, and the sweep's 2048 rows in one")
	void gemmRowsAreWholeColumnTilesWithinTheBudget() {
		for (GpuKernelTimings.MatrixShape m : GpuKernelTimings.GEMM) {
			int rows = BoundedLaunches.gemmRows(m.rows(), m.cols(), BoundedLaunches.GEMM_WORK_PER_LAUNCH);
			assertThat(rows % BoundedLaunches.GEMM_ROW_STEP).as(m.label()).isZero();
			assertThat(rows).as(m.label()).isGreaterThanOrEqualTo(2048);
			assertThat((long) rows * m.rows() * m.cols()).as(m.label())
					.isLessThanOrEqualTo(BoundedLaunches.GEMM_WORK_PER_LAUNCH);
			assertThat((long) (rows + BoundedLaunches.GEMM_ROW_STEP) * m.rows() * m.cols()).as(m.label())
					.isGreaterThan(BoundedLaunches.GEMM_WORK_PER_LAUNCH);
		}
		assertThat(BoundedLaunches.gemmRows(1 << 20, 1 << 16, BoundedLaunches.GEMM_WORK_PER_LAUNCH))
				.isEqualTo(BoundedLaunches.GEMM_ROW_STEP);
	}

	/**
	 * The budgets against the slowest GPU a kit run has measured (a GeForce MX150): a launch's device time follows
	 * its work, so the heaviest launch the budgets allow, scaled from that GPU's 2048-token readings, stays under
	 * the kernel-timing probe's 500 ms bar with a fifth to spare. Readings: Mistral 7B's attention over a
	 * 2048-token window 268.75 ms; the tiled GEMM over Mistral 7B's 4096x14336 Q6_K down matrix at 2048 rows
	 * 209.69 ms.
	 */
	@Test
	@DisplayName("the budgets keep the slowest measured GPU's heaviest launch under 400 ms")
	void theBudgetsKeepTheSlowestMeasuredGpuUnderTheProbeBar() {
		long attention2048 = BoundedLaunches.attentionWork(window(0, 2048), 0, 2048, 0,
				BoundedLaunches.attentionWorkPerKey(32, 128));
		assertThat(268.75 * BUDGET / attention2048).isLessThanOrEqualTo(400.0);
		long gemm2048 = 2048L * 4096 * 14336;
		assertThat(209.69 * BoundedLaunches.GEMM_WORK_PER_LAUNCH / gemm2048).isLessThanOrEqualTo(400.0);
	}

	private static int[] window(int startPos, int rows) {
		int[] s = new int[rows];
		for (int b = 0; b < rows; b++)
			s[b] = startPos + b + 1;
		return s;
	}

	private static List<int[]> plan(int[] seqLens, int batch, int rpb, int window, long perKey, long budget) {
		List<int[]> out = new ArrayList<>();
		for (int start = 0, end; start < batch; start = end) {
			end = BoundedLaunches.attentionEnd(start, seqLens, batch, rpb, window, perKey, budget);
			assertThat(end).isGreaterThan(start);
			out.add(new int[] { start, end });
		}
		return out;
	}
}
