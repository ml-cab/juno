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

import java.util.Arrays;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRows;
import cab.ml.juno.kvcache.Q8_0KvCodec;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * One query head of grouped-query attention ({@link GqaMath}) with the weighted sum of value rows held in
 * vector registers.
 *
 * <p>The serial loop adds each key's weighted value row into the output array, so every output element is
 * loaded, updated and stored once per key. Here the output is built in blocks of 32 elements (eight 128-bit
 * float vectors; then a block of 16, then a scalar tail) that stay in registers across every key and are
 * stored once; each element receives the same products in the same order as in the serial loop. Each key's
 * dot product sums in four 128-bit accumulators (16 elements per step, then 4, then a scalar tail) and then
 * across the lanes in a fixed order, so the scores, and with them the output, differ from
 * {@link GqaMath#attend} by float rounding only, within its oracle bound, and are the same bits on every call
 * and at every thread count. The softmax is the serial loop's. Every lane operation is written in place: C2
 * boxes a vector passed to a helper it does not inline.
 *
 * <p>K and V are read in place ({@link KvRows}), page by page; an accumulator carried across keys is carried across
 * pages too. An accumulator is only ever the argument of a lane operation ({@code o0 = term.add(o0)}) and its lanes
 * are read through {@code zero(F).add(o0)}: C2 types a vector carried round a loop inexactly, and a call on one can
 * be bound through the JDK's shared type profile and boxed. Float addition commutes, and an accumulator that starts
 * at zero never holds -0, so both forms give the bits of {@code o0.add(term)}. A {@code Q8_0} key slice is decoded
 * into a per-thread row and scored as one row; {@code Q8_0} value slices are added by the serial loop, whose sums are
 * the register blocks' bit for bit.
 *
 * <p>Loading this class needs {@code jdk.incubator.vector}; {@link GqaMath} calls it only when
 * {@link VectorQuantKernels#AVAILABLE} is true.
 */
final class GqaVectorHead {

	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;

	private GqaVectorHead() {
	}

	/**
	 * Head {@code h} of the row: writes {@code out[h * headDim, (h + 1) * headDim)} from keys
	 * {@code [lo, seqLen)}, using {@code scores} (at least {@code seqLen - lo} long) as scratch.
	 */
	static void attend(float[] q, KvRows k, KvRows v, int seqLen, int lo, float[] out, int h, int headDim,
			int gqaRatio, int kvDim, float scale, float[] scores) {
		int kBase = (h / gqaRatio) * headDim;
		scoreKeys(q, h * headDim, k, kBase, lo, seqLen, headDim, kvDim, scale, scores);
		LlamaTransformerHandler.softmax(scores, seqLen - lo);
		addValues(v, kBase, lo, seqLen, out, h * headDim, headDim, kvDim, scores);
	}

	/** {@code scores[t - lo]} for keys {@code [lo, seqLen)}: the query slice at {@code qBase} against key slice {@code kBase}. */
	static void scoreKeys(float[] q, int qBase, KvRows k, int kBase, int lo, int seqLen, int headDim, int kvDim,
			float scale, float[] scores) {
		int perK = k.rowsPerPage();
		if (k.type() == KvElementType.F16) {
			for (int t = lo; t < seqLen;) {
				int page = t / perK;
				int slot = t - page * perK;
				int count = Math.min(seqLen - t, perK - slot);
				scoreRun(q, qBase, k.floatPage(page), slot * kvDim + kBase, kvDim, count, headDim, scale, scores, t - lo);
				t += count;
			}
		} else {
			float[] row = GqaMath.row(headDim);
			int bytesPerRow = k.bytesPerRow();
			for (int t = lo; t < seqLen;) {
				int page = t / perK;
				int slot = t - page * perK;
				int count = Math.min(seqLen - t, perK - slot);
				byte[] rows = k.bytePage(page);
				for (int r = 0, ro = slot * bytesPerRow; r < count; r++, ro += bytesPerRow) {
					Q8_0KvCodec.decodeRange(rows, ro, kBase, headDim, row, 0);
					scoreRun(q, qBase, row, 0, 0, 1, headDim, scale, scores, t - lo + r);
				}
				t += count;
			}
		}
	}

	/**
	 * Writes {@code out[outBase, outBase + headDim)}: the value slices at {@code kBase} of keys {@code [lo, seqLen)}
	 * weighted by {@code scores[t - lo]}, each output element summed in key order.
	 */
	static void addValues(KvRows v, int kBase, int lo, int seqLen, float[] out, int outBase, int headDim, int kvDim,
			float[] scores) {
		if (v.type() != KvElementType.F16) {
			Arrays.fill(out, outBase, outBase + headDim, 0f);
			GqaMath.addValues(v, kBase, lo, seqLen, out, outBase, headDim, kvDim, scores);
			return;
		}
		int perV = v.rowsPerPage();
		int c = 0;
		for (; c + 32 <= headDim; c += 32) {
			FloatVector o0 = FloatVector.zero(F), o1 = FloatVector.zero(F), o2 = FloatVector.zero(F), o3 = FloatVector.zero(F);
			FloatVector o4 = FloatVector.zero(F), o5 = FloatVector.zero(F), o6 = FloatVector.zero(F), o7 = FloatVector.zero(F);
			for (int t = lo; t < seqLen;) {
				int page = t / perV;
				int slot = t - page * perV;
				int end = t + Math.min(seqLen - t, perV - slot);
				float[] rows = v.floatPage(page);
				for (int vo = slot * kvDim + kBase + c; t < end; t++, vo += kvDim) {
					float w = scores[t - lo];
					o0 = FloatVector.fromArray(F, rows, vo).mul(w).add(o0);
					o1 = FloatVector.fromArray(F, rows, vo + 4).mul(w).add(o1);
					o2 = FloatVector.fromArray(F, rows, vo + 8).mul(w).add(o2);
					o3 = FloatVector.fromArray(F, rows, vo + 12).mul(w).add(o3);
					o4 = FloatVector.fromArray(F, rows, vo + 16).mul(w).add(o4);
					o5 = FloatVector.fromArray(F, rows, vo + 20).mul(w).add(o5);
					o6 = FloatVector.fromArray(F, rows, vo + 24).mul(w).add(o6);
					o7 = FloatVector.fromArray(F, rows, vo + 28).mul(w).add(o7);
				}
			}
			int ob = outBase + c;
			FloatVector.zero(F).add(o0).intoArray(out, ob);
			FloatVector.zero(F).add(o1).intoArray(out, ob + 4);
			FloatVector.zero(F).add(o2).intoArray(out, ob + 8);
			FloatVector.zero(F).add(o3).intoArray(out, ob + 12);
			FloatVector.zero(F).add(o4).intoArray(out, ob + 16);
			FloatVector.zero(F).add(o5).intoArray(out, ob + 20);
			FloatVector.zero(F).add(o6).intoArray(out, ob + 24);
			FloatVector.zero(F).add(o7).intoArray(out, ob + 28);
		}
		for (; c + 16 <= headDim; c += 16) {
			FloatVector o0 = FloatVector.zero(F), o1 = FloatVector.zero(F), o2 = FloatVector.zero(F), o3 = FloatVector.zero(F);
			for (int t = lo; t < seqLen;) {
				int page = t / perV;
				int slot = t - page * perV;
				int end = t + Math.min(seqLen - t, perV - slot);
				float[] rows = v.floatPage(page);
				for (int vo = slot * kvDim + kBase + c; t < end; t++, vo += kvDim) {
					float w = scores[t - lo];
					o0 = FloatVector.fromArray(F, rows, vo).mul(w).add(o0);
					o1 = FloatVector.fromArray(F, rows, vo + 4).mul(w).add(o1);
					o2 = FloatVector.fromArray(F, rows, vo + 8).mul(w).add(o2);
					o3 = FloatVector.fromArray(F, rows, vo + 12).mul(w).add(o3);
				}
			}
			int ob = outBase + c;
			FloatVector.zero(F).add(o0).intoArray(out, ob);
			FloatVector.zero(F).add(o1).intoArray(out, ob + 4);
			FloatVector.zero(F).add(o2).intoArray(out, ob + 8);
			FloatVector.zero(F).add(o3).intoArray(out, ob + 12);
		}
		for (; c < headDim; c++) {
			float o = 0f;
			for (int t = lo; t < seqLen;) {
				int page = t / perV;
				int slot = t - page * perV;
				int end = t + Math.min(seqLen - t, perV - slot);
				float[] rows = v.floatPage(page);
				for (int vo = slot * kvDim + kBase + c; t < end; t++, vo += kvDim)
					o += scores[t - lo] * rows[vo];
			}
			out[outBase + c] = o;
		}
	}

	/**
	 * Scores of {@code count} keys whose slices start at {@code rows[off]}, {@code stride} apart, into
	 * {@code scores[s0..)}: four accumulators over the head's width, then across the lanes in a fixed order.
	 */
	static void scoreRun(float[] q, int qBase, float[] rows, int off, int stride, int count, int headDim,
			float scale, float[] scores, int s0) {
		for (int r = 0; r < count; r++, off += stride) {
			FloatVector a0 = FloatVector.zero(F), a1 = FloatVector.zero(F), a2 = FloatVector.zero(F), a3 = FloatVector.zero(F);
			int d = 0;
			for (; d + 16 <= headDim; d += 16) {
				a0 = FloatVector.fromArray(F, q, qBase + d).mul(FloatVector.fromArray(F, rows, off + d)).add(a0);
				a1 = FloatVector.fromArray(F, q, qBase + d + 4).mul(FloatVector.fromArray(F, rows, off + d + 4)).add(a1);
				a2 = FloatVector.fromArray(F, q, qBase + d + 8).mul(FloatVector.fromArray(F, rows, off + d + 8)).add(a2);
				a3 = FloatVector.fromArray(F, q, qBase + d + 12).mul(FloatVector.fromArray(F, rows, off + d + 12)).add(a3);
			}
			for (; d + 4 <= headDim; d += 4)
				a0 = FloatVector.fromArray(F, q, qBase + d).mul(FloatVector.fromArray(F, rows, off + d)).add(a0);
			FloatVector a = FloatVector.zero(F).add(a0).add(a1).add(FloatVector.zero(F).add(a2).add(a3));
			float dot = (a.lane(0) + a.lane(1)) + (a.lane(2) + a.lane(3));
			for (; d < headDim; d++)
				dot += q[qBase + d] * rows[off + d];
			scores[s0 + r] = dot * scale;
		}
	}
}
