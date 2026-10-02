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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.management.ManagementFactory;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link MatVec#sgemmInto(float[], float[][], float[][], int, int)}, the
 * non-allocating batched form, on the CPU backend and on a backend that does not
 * override it (the interface default, which is what ROCm runs today): the same
 * bits as the allocating {@code sgemm}, written into the caller's rows, with the
 * argument check made before any work. {@link CpuMatVec} must not allocate the
 * output, which the default (a copy from the allocating form) does.
 */
@DisplayName("MatVec.sgemmInto - the non-allocating batched form")
class MatVecSgemmIntoTest {

	private static final int ROWS = 384;
	private static final int COLS = 96;

	@ParameterizedTest(name = "B={0}")
	@ValueSource(ints = { 1, 2, 8, 9, 32 })
	@DisplayName("CpuMatVec: bit-identical to sgemm, written into the caller's rows")
	void cpu_matchesAllocatingForm(int batch) {
		MatVec mv = new CpuMatVec();
		float[] a = randomFloats(ROWS * COLS, 1);
		float[][] x = randomRows(batch, COLS, 2);
		float[][] expected = mv.sgemm(a, x, ROWS, COLS);
		float[][] y = new float[batch][ROWS];
		float[][] sameRows = y.clone();
		mv.sgemmInto(a, x, y, ROWS, COLS);
		for (int b = 0; b < batch; b++) {
			assertThat(y[b]).as("row " + b + " is the caller's array").isSameAs(sameRows[b]);
			assertThat(y[b]).as("row " + b).containsExactly(expected[b]);
		}
	}

	@Test
	@DisplayName("a backend that does not override it (the interface default) is bit-identical to its sgemm")
	void defaultForm_matchesAllocatingForm() {
		MatVec serial = (a, x, rows, cols) -> new CpuMatVec().sgemv(a, x, rows, cols);
		float[] a = randomFloats(ROWS * COLS, 3);
		float[][] x = randomRows(9, COLS, 4);
		float[][] expected = serial.sgemm(a, x, ROWS, COLS);
		float[][] y = new float[9][ROWS];
		serial.sgemmInto(a, x, y, ROWS, COLS);
		for (int b = 0; b < 9; b++)
			assertThat(y[b]).as("row " + b).containsExactly(expected[b]);
	}

	@Test
	@DisplayName("output rows wider than the result keep their tail")
	void widerRows_keepTheirTail() {
		MatVec mv = new CpuMatVec();
		float[] a = randomFloats(ROWS * COLS, 5);
		float[][] x = randomRows(3, COLS, 6);
		float[][] expected = mv.sgemm(a, x, ROWS, COLS);
		float[][] y = new float[3][ROWS + 7];
		for (float[] row : y)
			java.util.Arrays.fill(row, ROWS, ROWS + 7, -42f);
		mv.sgemmInto(a, x, y, ROWS, COLS);
		for (int b = 0; b < 3; b++) {
			assertThat(java.util.Arrays.copyOf(y[b], ROWS)).as("row " + b).containsExactly(expected[b]);
			for (int i = ROWS; i < ROWS + 7; i++)
				assertThat(y[b][i]).as("row " + b + " tail " + i).isEqualTo(-42f);
		}
	}

	@Test
	@DisplayName("too few output rows, a null row or a short row is rejected before any work")
	void shortOutput_isRejected() {
		MatVec mv = new CpuMatVec();
		float[] a = randomFloats(ROWS * COLS, 7);
		float[][] x = randomRows(4, COLS, 8);
		assertThatThrownBy(() -> mv.sgemmInto(a, x, new float[3][ROWS], ROWS, COLS))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("3 rows for a batch of 4");
		float[][] withNull = new float[4][ROWS];
		withNull[2] = null;
		assertThatThrownBy(() -> mv.sgemmInto(a, x, withNull, ROWS, COLS))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("row 2 is null");
		float[][] shortRow = new float[4][ROWS];
		shortRow[1] = new float[ROWS - 1];
		assertThatThrownBy(() -> mv.sgemmInto(a, x, shortRow, ROWS, COLS))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("row 1 is " + (ROWS - 1));
	}

	@Test
	@DisplayName("CpuMatVec does not allocate the output (the default copies from a newly allocated one)")
	void cpu_doesNotAllocateTheOutput() {
		MatVec mv = new CpuMatVec();
		int batch = 32;
		float[] a = randomFloats(ROWS * COLS, 9);
		float[][] x = randomRows(batch, COLS, 10);
		float[][] y = new float[batch][ROWS];
		for (int i = 0; i < 50; i++) // compile the path before measuring
			mv.sgemmInto(a, x, y, ROWS, COLS);
		long outputBytes = (long) batch * ROWS * Float.BYTES;
		long allocated = allocatedBytes(() -> mv.sgemmInto(a, x, y, ROWS, COLS));
		assertThat(allocated).as("bytes the calling thread allocated for one call (the output is %d)", outputBytes)
				.isLessThan(outputBytes / 8);
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	/** Bytes the calling thread allocated while running {@code body}. */
	static long allocatedBytes(Runnable body) {
		com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		long before = bean.getCurrentThreadAllocatedBytes();
		body.run();
		return bean.getCurrentThreadAllocatedBytes() - before;
	}

	static float[] randomFloats(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = r.nextFloat() * 2f - 1f;
		return v;
	}

	static float[][] randomRows(int rows, int cols, long seed) {
		float[][] x = new float[rows][];
		for (int i = 0; i < rows; i++)
			x[i] = randomFloats(cols, seed * 1000 + i);
		return x;
	}
}
