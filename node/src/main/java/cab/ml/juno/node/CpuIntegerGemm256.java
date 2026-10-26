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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The tile and weight unpack of {@link CpuIntegerGemm} at 256-bit lanes, for
 * hosts with 256-bit integer instructions; chosen by {@link CpuIntegerWidth}.
 * The call structure, activations, scratch and task split are
 * {@link CpuIntegerGemm}'s.
 *
 * <p>The unpack extracts 32 packed bytes per vector where the 128-bit form takes
 * 16, and writes the same 16-bit weights and factors. The tile multiplies a
 * 16-weight sub-block as one sixteen-lane vector (two products per lane for
 * Q4_K and Q5_K's 32-weight sub-blocks, one for Q6_K and Q8_0), scales its exact
 * integer sum in eight float lanes, takes the offset term eight sub-blocks at a
 * time, and sums each output across lanes in a fixed order. Against the 128-bit
 * tile this changes float rounding only. As in {@link CpuIntegerGemm}, the batch is
 * padded to a multiple of four rows, each sub-block width's four-row form is a
 * method of its own, and every lane operation is written in place, so no vector
 * crosses a call.
 */
final class CpuIntegerGemm256 {

	private static final VectorSpecies<Byte> BY = ByteVector.SPECIES_256;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_256;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_256;

	private CpuIntegerGemm256() {
	}

	static void tile(CpuIntegerGemm.TileScratch ts, int type, byte[] raw, CpuIntegerGemm.Activations act, int batch,
			int cols, int r0, int r1, float[][] Y, int rowStart) {
		int nr = r1 - r0;
		int pb = act.padded;
		if (ts.acc.length < CpuIntegerGemm.TILE_ROWS * pb)
			ts.acc = new float[CpuIntegerGemm.TILE_ROWS * pb];
		float[] acc = ts.acc;
		java.util.Arrays.fill(acc, 0, nr * pb, 0f);
		int blocks = cols / 256;
		boolean sub16 = type == 14 || type == 8;
		float[] xds = sub16 ? act.dsum16 : act.dsum32;
		short[] xq = act.q;
		float[] xd = act.d;
		short[] w = ts.w;
		float[] sc = ts.scale;
		float[] off = ts.offset;
		for (int kb = 0; kb < blocks; kb++) {
			for (int r = 0; r < nr; r++)
				unpack(type, raw, r0 + r, kb, cols, w, r * 256, sc, off, r * 16);
			if (sub16)
				for (int b = 0; b < pb; b += 4)
					fourRows16(w, sc, off, xq, xd, xds, acc, nr, pb, b, b * cols + kb * 256, cols);
			else
				for (int b = 0; b < pb; b += 4)
					fourRows32(w, sc, off, xq, xd, xds, acc, nr, pb, b, b * cols + kb * 256, cols);
		}
		for (int r = 0; r < nr; r++) {
			int yr = r0 + r - rowStart;
			for (int b = 0; b < batch; b++)
				Y[b][yr] = acc[r * pb + b];
		}
	}

	/** {@link CpuIntegerGemm}'s four-row form at 256 bits, 16-wide sub-blocks. */
	private static void fourRows16(short[] w, float[] sc, float[] off, short[] xq, float[] xd, float[] xds,
			float[] acc, int nr, int batch, int b, int xo0, int cols) {
		int xo1 = xo0 + 1 * cols;
		int xo2 = xo0 + 2 * cols;
		int xo3 = xo0 + 3 * cols;
		int xd0 = xo0 >> 5;
		int xs0 = xo0 >> 4;
		int xd1 = xo1 >> 5;
		int xs1 = xo1 >> 4;
		int xd2 = xo2 >> 5;
		int xs2 = xo2 >> 4;
		int xd3 = xo3 >> 5;
		int xs3 = xo3 >> 4;
		for (int r = 0; r < nr; r++) {
			int wo = r * 256;
			int so = r * 16;
			FloatVector f0 = FloatVector.zero(F);
			FloatVector f1 = FloatVector.zero(F);
			FloatVector f2 = FloatVector.zero(F);
			FloatVector f3 = FloatVector.zero(F);
			for (int j = 0; j < 16; j++) {
				int e = j * 16;
				ShortVector w0 = ShortVector.fromArray(S, w, wo + e);
				IntVector v0 = w0.mul(ShortVector.fromArray(S, xq, xo0 + e)).reinterpretAsInts();
				f0 = f0.add(((FloatVector) v0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v0.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd0 + (j >> 1)]));
				IntVector v1 = w0.mul(ShortVector.fromArray(S, xq, xo1 + e)).reinterpretAsInts();
				f1 = f1.add(((FloatVector) v1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v1.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd1 + (j >> 1)]));
				IntVector v2 = w0.mul(ShortVector.fromArray(S, xq, xo2 + e)).reinterpretAsInts();
				f2 = f2.add(((FloatVector) v2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v2.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd2 + (j >> 1)]));
				IntVector v3 = w0.mul(ShortVector.fromArray(S, xq, xo3 + e)).reinterpretAsInts();
				f3 = f3.add(((FloatVector) v3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v3.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd3 + (j >> 1)]));
			}
			for (int j = 0; j < 16; j += 8) {
				FloatVector o = FloatVector.fromArray(F, off, so + j);
				f0 = f0.sub(o.mul(FloatVector.fromArray(F, xds, xs0 + j)));
				f1 = f1.sub(o.mul(FloatVector.fromArray(F, xds, xs1 + j)));
				f2 = f2.sub(o.mul(FloatVector.fromArray(F, xds, xs2 + j)));
				f3 = f3.sub(o.mul(FloatVector.fromArray(F, xds, xs3 + j)));
			}
			acc[r * batch + b + 0] += (((f0.lane(0) + f0.lane(1)) + (f0.lane(2) + f0.lane(3))) + ((f0.lane(4) + f0.lane(5)) + (f0.lane(6) + f0.lane(7))));
			acc[r * batch + b + 1] += (((f1.lane(0) + f1.lane(1)) + (f1.lane(2) + f1.lane(3))) + ((f1.lane(4) + f1.lane(5)) + (f1.lane(6) + f1.lane(7))));
			acc[r * batch + b + 2] += (((f2.lane(0) + f2.lane(1)) + (f2.lane(2) + f2.lane(3))) + ((f2.lane(4) + f2.lane(5)) + (f2.lane(6) + f2.lane(7))));
			acc[r * batch + b + 3] += (((f3.lane(0) + f3.lane(1)) + (f3.lane(2) + f3.lane(3))) + ((f3.lane(4) + f3.lane(5)) + (f3.lane(6) + f3.lane(7))));
		}
	}

	/** {@link CpuIntegerGemm}'s four-row form at 256 bits, 32-wide sub-blocks. */
	private static void fourRows32(short[] w, float[] sc, float[] off, short[] xq, float[] xd, float[] xds,
			float[] acc, int nr, int batch, int b, int xo0, int cols) {
		int xo1 = xo0 + 1 * cols;
		int xo2 = xo0 + 2 * cols;
		int xo3 = xo0 + 3 * cols;
		int xd0 = xo0 >> 5;
		int xs0 = xd0;
		int xd1 = xo1 >> 5;
		int xs1 = xd1;
		int xd2 = xo2 >> 5;
		int xs2 = xd2;
		int xd3 = xo3 >> 5;
		int xs3 = xd3;
		for (int r = 0; r < nr; r++) {
			int wo = r * 256;
			int so = r * 16;
			FloatVector f0 = FloatVector.zero(F);
			FloatVector f1 = FloatVector.zero(F);
			FloatVector f2 = FloatVector.zero(F);
			FloatVector f3 = FloatVector.zero(F);
			for (int j = 0; j < 8; j++) {
				int e = j * 32;
				ShortVector w0 = ShortVector.fromArray(S, w, wo + e);
				ShortVector w16 = ShortVector.fromArray(S, w, wo + e + 16);
				IntVector v0 = w0.mul(ShortVector.fromArray(S, xq, xo0 + e))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo0 + e + 16))).reinterpretAsInts();
				f0 = f0.add(((FloatVector) v0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v0.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd0 + j]));
				IntVector v1 = w0.mul(ShortVector.fromArray(S, xq, xo1 + e))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo1 + e + 16))).reinterpretAsInts();
				f1 = f1.add(((FloatVector) v1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v1.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd1 + j]));
				IntVector v2 = w0.mul(ShortVector.fromArray(S, xq, xo2 + e))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo2 + e + 16))).reinterpretAsInts();
				f2 = f2.add(((FloatVector) v2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v2.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd2 + j]));
				IntVector v3 = w0.mul(ShortVector.fromArray(S, xq, xo3 + e))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo3 + e + 16))).reinterpretAsInts();
				f3 = f3.add(((FloatVector) v3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v3.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd3 + j]));
			}
			for (int j = 0; j < 8; j += 8) {
				FloatVector o = FloatVector.fromArray(F, off, so + j);
				f0 = f0.sub(o.mul(FloatVector.fromArray(F, xds, xs0 + j)));
				f1 = f1.sub(o.mul(FloatVector.fromArray(F, xds, xs1 + j)));
				f2 = f2.sub(o.mul(FloatVector.fromArray(F, xds, xs2 + j)));
				f3 = f3.sub(o.mul(FloatVector.fromArray(F, xds, xs3 + j)));
			}
			acc[r * batch + b + 0] += (((f0.lane(0) + f0.lane(1)) + (f0.lane(2) + f0.lane(3))) + ((f0.lane(4) + f0.lane(5)) + (f0.lane(6) + f0.lane(7))));
			acc[r * batch + b + 1] += (((f1.lane(0) + f1.lane(1)) + (f1.lane(2) + f1.lane(3))) + ((f1.lane(4) + f1.lane(5)) + (f1.lane(6) + f1.lane(7))));
			acc[r * batch + b + 2] += (((f2.lane(0) + f2.lane(1)) + (f2.lane(2) + f2.lane(3))) + ((f2.lane(4) + f2.lane(5)) + (f2.lane(6) + f2.lane(7))));
			acc[r * batch + b + 3] += (((f3.lane(0) + f3.lane(1)) + (f3.lane(2) + f3.lane(3))) + ((f3.lane(4) + f3.lane(5)) + (f3.lane(6) + f3.lane(7))));
		}
	}

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	/** {@link CpuIntegerGemm#unpack} with 32 packed bytes per vector; the same weights and factors. */
	static void unpack(int type, byte[] raw, int row, int kb, int cols, short[] w, int wo, float[] sc, float[] off,
			int so) {
		switch (type) {
		case 12 -> {
			int bo = (row * (cols / 256) + kb) * 144;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int e = wo + g * 64;
				ByteVector v = ByteVector.fromArray(BY, raw, bo + 16 + g * 32);
				ByteVector lo = v.and((byte) 0x0F);
				ByteVector hi = v.lanewise(VectorOperators.LSHR, 4);
				((ShortVector) lo.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e);
				((ShortVector) lo.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 16);
				((ShortVector) hi.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 32);
				((ShortVector) hi.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 48);
				sc[so + 2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
				sc[so + 2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
				off[so + 2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
				off[so + 2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
			}
		}
		case 13 -> {
			int bo = (row * (cols / 256) + kb) * 176;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			ByteVector h = ByteVector.fromArray(BY, raw, bo + 16);
			for (int g = 0; g < 4; g++) {
				int e = wo + g * 64;
				ByteVector v = ByteVector.fromArray(BY, raw, bo + 48 + g * 32);
				ByteVector lo = v.and((byte) 0x0F)
						.or(h.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
				ByteVector hi = v.lanewise(VectorOperators.LSHR, 4)
						.or(h.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
				((ShortVector) lo.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e);
				((ShortVector) lo.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 16);
				((ShortVector) hi.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 32);
				((ShortVector) hi.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 48);
				sc[so + 2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
				sc[so + 2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
				off[so + 2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
				off[so + 2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
			}
		}
		case 14 -> {
			// Stored unsigned (0..63); the offset of 32 is the activation-sum term.
			int bo = (row * (cols / 256) + kb) * 210;
			float d = f16(raw, bo + 208);
			for (int half = 0; half < 2; half++) {
				int qlOff = bo + half * 64;
				int scOff = bo + 192 + half * 8;
				int e = wo + half * 128;
				ByteVector ql = ByteVector.fromArray(BY, raw, qlOff);
				ByteVector ql2 = ByteVector.fromArray(BY, raw, qlOff + 32);
				ByteVector qh = ByteVector.fromArray(BY, raw, bo + 128 + half * 32);
				ByteVector w0 = ql.and((byte) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and((byte) 0x30));
				ByteVector w1 = ql2.and((byte) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and((byte) 0x30));
				ByteVector w2 = ql.lanewise(VectorOperators.LSHR, 4).or(qh.and((byte) 0x30));
				ByteVector w3 = ql2.lanewise(VectorOperators.LSHR, 4).or(qh.lanewise(VectorOperators.LSHR, 2).and((byte) 0x30));
				((ShortVector) w0.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e);
				((ShortVector) w0.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 16);
				((ShortVector) w1.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 32);
				((ShortVector) w1.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 48);
				((ShortVector) w2.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 64);
				((ShortVector) w2.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 80);
				((ShortVector) w3.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 96);
				((ShortVector) w3.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 112);
				// The half's eight sub-blocks of 16, in order, take its eight scales in order.
				for (int jj = 0; jj < 8; jj++) {
					float s = d * raw[scOff + jj];
					sc[so + half * 8 + jj] = s;
					off[so + half * 8 + jj] = 32f * s;
				}
			}
		}
		case 8 -> {
			int bo = (row * (cols / 32) + kb * 8) * 34;
			for (int k = 0; k < 8; k++) {
				int o = bo + k * 34;
				float d = f16(raw, o);
				int e = wo + k * 32;
				ByteVector v = ByteVector.fromArray(BY, raw, o + 2);
				((ShortVector) v.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e);
				((ShortVector) v.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 16);
				sc[so + 2 * k] = d;
				sc[so + 2 * k + 1] = d;
				off[so + 2 * k] = 0f;
				off[so + 2 * k + 1] = 0f;
			}
		}
		default -> throw new IllegalArgumentException("type " + type);
		}
	}
}
