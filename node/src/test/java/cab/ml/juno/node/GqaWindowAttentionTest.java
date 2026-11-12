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

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRows;
import cab.ml.juno.node.PagedKvAttentionOracleTest.Shape;
import cab.ml.juno.node.PagedKvAttentionOracleTest.TwinKv;

/**
 * {@link GqaWindowAttention}, a prefill window's attention in one pooled call, against the per-row form it replaces:
 * {@link GqaMath#attendPooled} on each row over the rows gathered into one array, decoded by the codec. Paged and
 * dense KV, every (K type, V type) pair and page size, windows, a window that starts at 0 and ones that start later,
 * and both head forms; bit for bit.
 */
class GqaWindowAttentionTest {

	private static final Shape[] SHAPES = { new Shape(4, 4, 8), new Shape(8, 2, 16), new Shape(16, 2, 32),
			new Shape(8, 1, 32) };
	private static final int N = 37;
	private static final int[] STARTS = { 0, 1, 16, 30 };
	private static final int[] WINDOWS = { 0, 1, 5, 64 };

	@ParameterizedTest(name = "K {0}, V {1}, page {2}")
	@MethodSource("cab.ml.juno.node.PagedKvAttentionOracleTest#typePairsAndPages")
	@DisplayName("a window attended in one call equals each row attended alone over the gathered rows, bit for bit")
	void windowEqualsRowByRow(KvElementType kType, KvElementType vType, int pageSize) {
		boolean[] heads = GqaMath.VECTOR_HEAD ? new boolean[] { false, true } : new boolean[] { false };
		for (Shape s : SHAPES) {
			TwinKv kv = new TwinKv(kType, vType, pageSize, s.kvDim(), 17L * pageSize + s.kvDim());
			kv.writeThrough(N);
			KvRows gatheredK = KvRows.of(kv.pagedK.toFloatArray(N), s.kvDim());
			KvRows gatheredV = KvRows.of(kv.pagedV.toFloatArray(N), s.kvDim());
			for (int start : STARTS) {
				int rows = N - start;
				float[][] q = queries(s, rows, 31L * start + s.qDim());
				for (int window : WINDOWS) {
					for (boolean vectorHead : heads) {
						float[][] expected = new float[rows][s.qDim()];
						for (int b = 0; b < rows; b++)
							GqaMath.attendPooled(q[b], gatheredK, gatheredV, start + b + 1, expected[b], s.heads(),
									s.headDim(), s.gqaRatio(), s.kvDim(), window, vectorHead);
						String label = s + ", start " + start + ", window " + window + ", vector " + vectorHead;
						assertWindow(expected, q, kv.pagedK.attentionRows(N), kv.pagedV.attentionRows(N), s, start,
								window, vectorHead, "paged, " + label);
						assertWindow(expected, q, kv.denseK.attentionRows(N), kv.denseV.attentionRows(N), s, start,
								window, vectorHead, "dense, " + label);
					}
				}
			}
		}
	}

	private static void assertWindow(float[][] expected, float[][] q, KvRows k, KvRows v, Shape s, int start,
			int window, boolean vectorHead, String label) {
		int rows = expected.length;
		float[][] out = new float[rows][s.qDim()];
		for (float[] row : out)
			Arrays.fill(row, Float.NaN); // every head's slice must be written
		GqaWindowAttention.attend(q, k, v, start, rows, out, s.heads(), s.headDim(), s.gqaRatio(), s.kvDim(), window,
				vectorHead);
		for (int b = 0; b < rows; b++)
			for (int i = 0; i < s.qDim(); i++)
				assertEquals(Float.floatToRawIntBits(expected[b][i]), Float.floatToRawIntBits(out[b][i]),
						label + ", row " + b + ", element " + i);
	}

	private static float[][] queries(Shape s, int count, long seed) {
		Random rng = new Random(seed);
		float[][] q = new float[count][s.qDim()];
		for (float[] row : q)
			for (int i = 0; i < row.length; i++)
				row[i] = (float) rng.nextGaussian();
		return q;
	}
}
