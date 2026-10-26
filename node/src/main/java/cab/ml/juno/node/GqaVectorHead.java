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
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen, int lo, float[] out, int h, int headDim,
			int gqaRatio, int kvDim, float scale, float[] scores) {
		int n = seqLen - lo;
		int kBase = (h / gqaRatio) * headDim;
		int qBase = h * headDim;
		for (int t = lo; t < seqLen; t++) {
			int kOffset = t * kvDim + kBase;
			FloatVector a0 = FloatVector.zero(F), a1 = FloatVector.zero(F), a2 = FloatVector.zero(F), a3 = FloatVector.zero(F);
			int d = 0;
			for (; d + 16 <= headDim; d += 16) {
				a0 = a0.add(FloatVector.fromArray(F, q, qBase + d).mul(FloatVector.fromArray(F, kCache, kOffset + d)));
				a1 = a1.add(FloatVector.fromArray(F, q, qBase + d + 4).mul(FloatVector.fromArray(F, kCache, kOffset + d + 4)));
				a2 = a2.add(FloatVector.fromArray(F, q, qBase + d + 8).mul(FloatVector.fromArray(F, kCache, kOffset + d + 8)));
				a3 = a3.add(FloatVector.fromArray(F, q, qBase + d + 12).mul(FloatVector.fromArray(F, kCache, kOffset + d + 12)));
			}
			for (; d + 4 <= headDim; d += 4)
				a0 = a0.add(FloatVector.fromArray(F, q, qBase + d).mul(FloatVector.fromArray(F, kCache, kOffset + d)));
			FloatVector a = a0.add(a1).add(a2.add(a3));
			float dot = (a.lane(0) + a.lane(1)) + (a.lane(2) + a.lane(3));
			for (; d < headDim; d++)
				dot += q[qBase + d] * kCache[kOffset + d];
			scores[t - lo] = dot * scale;
		}
		LlamaTransformerHandler.softmax(scores, n);

		int outBase = h * headDim;
		int c = 0;
		for (; c + 32 <= headDim; c += 32) {
			FloatVector o0 = FloatVector.zero(F), o1 = FloatVector.zero(F), o2 = FloatVector.zero(F), o3 = FloatVector.zero(F);
			FloatVector o4 = FloatVector.zero(F), o5 = FloatVector.zero(F), o6 = FloatVector.zero(F), o7 = FloatVector.zero(F);
			for (int t = lo; t < seqLen; t++) {
				int vo = t * kvDim + kBase + c;
				float w = scores[t - lo];
				o0 = o0.add(FloatVector.fromArray(F, vCache, vo).mul(w));
				o1 = o1.add(FloatVector.fromArray(F, vCache, vo + 4).mul(w));
				o2 = o2.add(FloatVector.fromArray(F, vCache, vo + 8).mul(w));
				o3 = o3.add(FloatVector.fromArray(F, vCache, vo + 12).mul(w));
				o4 = o4.add(FloatVector.fromArray(F, vCache, vo + 16).mul(w));
				o5 = o5.add(FloatVector.fromArray(F, vCache, vo + 20).mul(w));
				o6 = o6.add(FloatVector.fromArray(F, vCache, vo + 24).mul(w));
				o7 = o7.add(FloatVector.fromArray(F, vCache, vo + 28).mul(w));
			}
			int ob = outBase + c;
			o0.intoArray(out, ob);
			o1.intoArray(out, ob + 4);
			o2.intoArray(out, ob + 8);
			o3.intoArray(out, ob + 12);
			o4.intoArray(out, ob + 16);
			o5.intoArray(out, ob + 20);
			o6.intoArray(out, ob + 24);
			o7.intoArray(out, ob + 28);
		}
		for (; c + 16 <= headDim; c += 16) {
			FloatVector o0 = FloatVector.zero(F), o1 = FloatVector.zero(F), o2 = FloatVector.zero(F), o3 = FloatVector.zero(F);
			for (int t = lo; t < seqLen; t++) {
				int vo = t * kvDim + kBase + c;
				float w = scores[t - lo];
				o0 = o0.add(FloatVector.fromArray(F, vCache, vo).mul(w));
				o1 = o1.add(FloatVector.fromArray(F, vCache, vo + 4).mul(w));
				o2 = o2.add(FloatVector.fromArray(F, vCache, vo + 8).mul(w));
				o3 = o3.add(FloatVector.fromArray(F, vCache, vo + 12).mul(w));
			}
			int ob = outBase + c;
			o0.intoArray(out, ob);
			o1.intoArray(out, ob + 4);
			o2.intoArray(out, ob + 8);
			o3.intoArray(out, ob + 12);
		}
		for (; c < headDim; c++) {
			float o = 0f;
			for (int t = lo; t < seqLen; t++)
				o += scores[t - lo] * vCache[t * kvDim + kBase + c];
			out[outBase + c] = o;
		}
	}
}
