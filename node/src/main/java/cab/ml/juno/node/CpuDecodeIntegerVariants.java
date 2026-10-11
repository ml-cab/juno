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

import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Integer-dot decode kernel variants for {@link CpuDecodeKernelVariantMicrobench}:
 * the activation row is quantized once per call to 8-bit integers with one
 * float scale per 32 values, and each 32-weight (Q4_K) or 16-weight (Q6_K)
 * sub-block becomes an integer dot product whose scale is applied once.
 *
 * <ul>
 * <li>{@code c}: scalar integer loops, written so C2's auto-vectorizer can
 * take them;</li>
 * <li>{@code d}: explicit Vector API at the 128-bit species, 16-bit lanes,
 * per-sub-block partial sums widened to int and scaled as floats;</li>
 * <li>{@code d2}: as {@code d}, but the packed weight bytes are reinterpreted
 * as 16-bit lanes (no conversion) and the even and odd bytes' fields are
 * extracted with shifts and masks; the activation is stored de-interleaved per
 * 16 values (even positions, then odd) so lanes line up without a shuffle, and
 * Q6_K's offset of 32 is taken out of the inner loop as a per-sub-block
 * correction from activation sums over 16 values;</li>
 * <li>{@code d3}: as {@code d2}, but the activation stays 8-bit in natural
 * order (half the cache footprint) and is split into even and odd 16-bit lanes
 * with the same reinterpretation (shift left 8 then arithmetic shift right 8,
 * and arithmetic shift right 8).</li>
 * </ul>
 * Measurement only; not on any product path.
 */
final class CpuDecodeIntegerVariants {

	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
	private static final VectorSpecies<Integer> I = IntVector.SPECIES_128;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;

	private CpuDecodeIntegerVariants() {
	}

	/** Activation quantized per 32 values: q in [-127, 127], scale d, block sum of q. */
	record Q8(byte[] qb, short[] qs, float[] d, int[] sum) {
	}

	static Q8 quantize(float[] x) {
		int n = x.length;
		int nb = n / 32;
		byte[] qb = new byte[n];
		short[] qs = new short[n];
		float[] d = new float[nb];
		int[] sum = new int[nb];
		for (int b = 0; b < nb; b++) {
			int o = b * 32;
			float amax = 0f;
			for (int i = 0; i < 32; i++)
				amax = Math.max(amax, Math.abs(x[o + i]));
			float scale = amax / 127f;
			float inv = scale == 0f ? 0f : 1f / scale;
			int s = 0;
			for (int i = 0; i < 32; i++) {
				int v = Math.round(x[o + i] * inv);
				qb[o + i] = (byte) v;
				qs[o + i] = (short) v;
				s += v;
			}
			d[b] = scale;
			sum[b] = s;
		}
		return new Q8(qb, qs, d, sum);
	}

	/** De-interleaved activation for {@code d2}: per 16 values, even positions then odd. */
	record Q8Perm(short[] q, float[] d, int[] sum32, int[] sum16) {
	}

	static Q8Perm quantizePermuted(float[] x) {
		Q8 a = quantize(x);
		int n = x.length;
		short[] q = new short[n];
		int[] sum16 = new int[n / 16];
		for (int base = 0; base < n; base += 16) {
			int s = 0;
			for (int k = 0; k < 8; k++) {
				q[base + k] = a.qs()[base + 2 * k];
				q[base + 8 + k] = a.qs()[base + 2 * k + 1];
			}
			for (int i = 0; i < 16; i++)
				s += a.qs()[base + i];
			sum16[base >> 4] = s;
		}
		return new Q8Perm(q, a.d(), a.sum(), sum16);
	}

	/** Pairwise-sums a short vector's lanes into four int lanes (sign-extended) and converts them to float. */
	private static FloatVector widenPairs(ShortVector s) {
		IntVector w = s.reinterpretAsInts();
		IntVector lo = w.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16);
		IntVector hi = w.lanewise(VectorOperators.ASHR, 16);
		return (FloatVector) lo.add(hi).convertShape(VectorOperators.I2F, F, 0);
	}

	private static ShortVector evenBytes(ShortVector v) {
		return v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8);
	}

	private static ShortVector oddBytes(ShortVector v) {
		return v.lanewise(VectorOperators.ASHR, 8);
	}

	/** Activation sums over 16 values, from the per-value quantization. */
	static int[] sum16(Q8 a) {
		int[] s16 = new int[a.qb().length / 16];
		for (int i = 0; i < a.qb().length; i++)
			s16[i >> 4] += a.qb()[i];
		return s16;
	}

	static void q4kVector3(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final byte[] xq = a.qb();
		final float[] xd = a.d();
		final int[] xs = a.sum();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			float macc = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					int q = bo + 16 + g * 32;
					int x0 = b * 256 + g * 64;
					ShortVector aLo = ShortVector.zero(S);
					ShortVector aHi = ShortVector.zero(S);
					for (int h = 0; h < 32; h += 16) {
						ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
						ShortVector xl = ByteVector.fromArray(B, xq, x0 + h).reinterpretAsShorts();
						ShortVector xh = ByteVector.fromArray(B, xq, x0 + 32 + h).reinterpretAsShorts();
						aLo = aLo.add(w.and((short) 0x0F).mul(evenBytes(xl)))
								.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).mul(oddBytes(xl)));
						aHi = aHi.add(w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).mul(evenBytes(xh)))
								.add(w.lanewise(VectorOperators.LSHR, 12).mul(oddBytes(xh)));
					}
					int ab = x0 >> 5;
					float sc0 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					float sc1 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					float m0 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					float m1 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
					facc = facc.add(widenPairs(aLo).mul(d * sc0 * xd[ab]));
					facc = facc.add(widenPairs(aHi).mul(d * sc1 * xd[ab + 1]));
					macc += dmin * (m0 * xd[ab] * xs[ab] + m1 * xd[ab + 1] * xs[ab + 1]);
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD) - macc;
		});
	}

	static void q6kVector3(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final byte[] xq = a.qb();
		final float[] xd = a.d();
		final int[] xs16 = sum16(a);
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			float corr = 0f;
			ShortVector m30 = ShortVector.broadcast(S, (short) 0x30);
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = b * 256 + half * 128;
					for (int is = 0; is < 2; is++) {
						int l0 = is * 16;
						ShortVector qa = ByteVector.fromArray(B, raw, qlOff + l0).reinterpretAsShorts();
						ShortVector qb = ByteVector.fromArray(B, raw, qlOff + l0 + 32).reinterpretAsShorts();
						ShortVector qh = ByteVector.fromArray(B, raw, qhOff + l0).reinterpretAsShorts();
						int xo = xOff + l0;
						ShortVector x1 = ByteVector.fromArray(B, xq, xo).reinterpretAsShorts();
						ShortVector x2 = ByteVector.fromArray(B, xq, xo + 32).reinterpretAsShorts();
						ShortVector x3 = ByteVector.fromArray(B, xq, xo + 64).reinterpretAsShorts();
						ShortVector x4 = ByteVector.fromArray(B, xq, xo + 96).reinterpretAsShorts();
						ShortVector s1 = qa.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and(m30)).mul(evenBytes(x1))
								.add(qa.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
										.or(qh.lanewise(VectorOperators.LSHR, 4).and(m30)).mul(oddBytes(x1)));
						ShortVector s2 = qb.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and(m30)).mul(evenBytes(x2))
								.add(qb.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
										.or(qh.lanewise(VectorOperators.LSHR, 6).and(m30)).mul(oddBytes(x2)));
						ShortVector s3 = qa.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).or(qh.and(m30)).mul(evenBytes(x3))
								.add(qa.lanewise(VectorOperators.LSHR, 12)
										.or(qh.lanewise(VectorOperators.LSHR, 8).and(m30)).mul(oddBytes(x3)));
						ShortVector s4 = qb.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 2).and(m30)).mul(evenBytes(x4))
								.add(qb.lanewise(VectorOperators.LSHR, 12)
										.or(qh.lanewise(VectorOperators.LSHR, 10).and(m30)).mul(oddBytes(x4)));
						int ab = xo >> 5;
						int a16 = xo >> 4;
						float f1 = d * raw[scOff + is] * xd[ab];
						float f2 = d * raw[scOff + is + 2] * xd[ab + 1];
						float f3 = d * raw[scOff + is + 4] * xd[ab + 2];
						float f4 = d * raw[scOff + is + 6] * xd[ab + 3];
						facc = facc.add(widenPairs(s1).mul(f1)).add(widenPairs(s2).mul(f2))
								.add(widenPairs(s3).mul(f3)).add(widenPairs(s4).mul(f4));
						corr += 32f * (f1 * xs16[a16] + f2 * xs16[a16 + 2] + f3 * xs16[a16 + 4] + f4 * xs16[a16 + 6]);
					}
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD) - corr;
		});
	}

	// ── Q4_K ────────────────────────────────────────────────────────────────

	static void q4kVector2(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8Perm a = quantizePermuted(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final short[] xq = a.q();
		final float[] xd = a.d();
		final int[] xs = a.sum32();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			float macc = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					int q = bo + 16 + g * 32;
					int x0 = b * 256 + g * 64;
					ShortVector aLo = ShortVector.zero(S);
					ShortVector aHi = ShortVector.zero(S);
					for (int h = 0; h < 32; h += 16) {
						ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
						ShortVector e = w.and((short) 0x0F);
						ShortVector o = w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F);
						ShortVector he = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
						ShortVector ho = w.lanewise(VectorOperators.LSHR, 12);
						int xo = x0 + h;
						aLo = aLo.add(e.mul(ShortVector.fromArray(S, xq, xo)))
								.add(o.mul(ShortVector.fromArray(S, xq, xo + 8)));
						aHi = aHi.add(he.mul(ShortVector.fromArray(S, xq, xo + 32)))
								.add(ho.mul(ShortVector.fromArray(S, xq, xo + 40)));
					}
					int ab = x0 >> 5;
					float sc0 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					float sc1 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					float m0 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					float m1 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
					facc = facc.add(widenPairs(aLo).mul(d * sc0 * xd[ab]));
					facc = facc.add(widenPairs(aHi).mul(d * sc1 * xd[ab + 1]));
					macc += dmin * (m0 * xd[ab] * xs[ab] + m1 * xd[ab + 1] * xs[ab + 1]);
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD) - macc;
		});
	}

	static void q6kVector2(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8Perm a = quantizePermuted(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final short[] xq = a.q();
		final float[] xd = a.d();
		final int[] xs16 = a.sum16();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			float corr = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = b * 256 + half * 128;
					for (int is = 0; is < 2; is++) {
						int l0 = is * 16;
						ShortVector qa = ByteVector.fromArray(B, raw, qlOff + l0).reinterpretAsShorts();
						ShortVector qb = ByteVector.fromArray(B, raw, qlOff + l0 + 32).reinterpretAsShorts();
						ShortVector qh = ByteVector.fromArray(B, raw, qhOff + l0).reinterpretAsShorts();
						ShortVector m30 = ShortVector.broadcast(S, (short) 0x30);
						ShortVector q1e = qa.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and(m30));
						ShortVector q1o = qa.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 4).and(m30));
						ShortVector q2e = qb.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and(m30));
						ShortVector q2o = qb.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 6).and(m30));
						ShortVector q3e = qa.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).or(qh.and(m30));
						ShortVector q3o = qa.lanewise(VectorOperators.LSHR, 12)
								.or(qh.lanewise(VectorOperators.LSHR, 8).and(m30));
						ShortVector q4e = qb.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 2).and(m30));
						ShortVector q4o = qb.lanewise(VectorOperators.LSHR, 12)
								.or(qh.lanewise(VectorOperators.LSHR, 10).and(m30));
						int xo = xOff + l0;
						ShortVector s1 = q1e.mul(ShortVector.fromArray(S, xq, xo))
								.add(q1o.mul(ShortVector.fromArray(S, xq, xo + 8)));
						ShortVector s2 = q2e.mul(ShortVector.fromArray(S, xq, xo + 32))
								.add(q2o.mul(ShortVector.fromArray(S, xq, xo + 40)));
						ShortVector s3 = q3e.mul(ShortVector.fromArray(S, xq, xo + 64))
								.add(q3o.mul(ShortVector.fromArray(S, xq, xo + 72)));
						ShortVector s4 = q4e.mul(ShortVector.fromArray(S, xq, xo + 96))
								.add(q4o.mul(ShortVector.fromArray(S, xq, xo + 104)));
						int ab = xo >> 5;
						int a16 = xo >> 4;
						float f1 = d * raw[scOff + is] * xd[ab];
						float f2 = d * raw[scOff + is + 2] * xd[ab + 1];
						float f3 = d * raw[scOff + is + 4] * xd[ab + 2];
						float f4 = d * raw[scOff + is + 6] * xd[ab + 3];
						facc = facc.add(widenPairs(s1).mul(f1)).add(widenPairs(s2).mul(f2))
								.add(widenPairs(s3).mul(f3)).add(widenPairs(s4).mul(f4));
						corr += 32f * (f1 * xs16[a16] + f2 * xs16[a16 + 2] + f3 * xs16[a16 + 4] + f4 * xs16[a16 + 6]);
					}
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD) - corr;
		});
	}

	static void q4kScalar(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final byte[] xq = a.qb();
		final float[] xd = a.d();
		final int[] xs = a.sum();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float acc = 0f;
			float macc = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					int q = bo + 16 + g * 32;
					int x0 = b * 256 + g * 64;
					int i0 = 0;
					int i1 = 0;
					for (int i = 0; i < 32; i++) {
						int v = raw[q + i];
						i0 += (v & 0x0F) * xq[x0 + i];
						i1 += ((v >> 4) & 0x0F) * xq[x0 + 32 + i];
					}
					int ab = x0 >> 5;
					float sc0 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					float sc1 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					float m0 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					float m1 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
					acc += d * (sc0 * xd[ab] * i0 + sc1 * xd[ab + 1] * i1);
					macc += dmin * (m0 * xd[ab] * xs[ab] + m1 * xd[ab + 1] * xs[ab + 1]);
				}
			}
			y[r] = acc - macc;
		});
	}

	static void q4kVector(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final short[] xq = a.qs();
		final float[] xd = a.d();
		final int[] xs = a.sum();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			float macc = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					int q = bo + 16 + g * 32;
					int x0 = b * 256 + g * 64;
					ShortVector aLo = ShortVector.zero(S);
					ShortVector aHi = ShortVector.zero(S);
					for (int h = 0; h < 32; h += 16) {
						ByteVector v = ByteVector.fromArray(B, raw, q + h);
						for (int part = 0; part < 2; part++) {
							ShortVector w = (ShortVector) v.convertShape(VectorOperators.B2S, S, part);
							ShortVector lo = w.and((short) 0x0F);
							ShortVector hi = w.lanewise(VectorOperators.ASHR, 4).and((short) 0x0F);
							int xo = x0 + h + part * 8;
							aLo = aLo.add(lo.mul(ShortVector.fromArray(S, xq, xo)));
							aHi = aHi.add(hi.mul(ShortVector.fromArray(S, xq, xo + 32)));
						}
					}
					IntVector iLo = ((IntVector) aLo.convertShape(VectorOperators.S2I, I, 0))
							.add((IntVector) aLo.convertShape(VectorOperators.S2I, I, 1));
					IntVector iHi = ((IntVector) aHi.convertShape(VectorOperators.S2I, I, 0))
							.add((IntVector) aHi.convertShape(VectorOperators.S2I, I, 1));
					int ab = x0 >> 5;
					float sc0 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					float sc1 = LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					float m0 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					float m1 = LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
					facc = facc.add(((FloatVector) iLo.convertShape(VectorOperators.I2F, F, 0)).mul(d * sc0 * xd[ab]));
					facc = facc.add(((FloatVector) iHi.convertShape(VectorOperators.I2F, F, 0)).mul(d * sc1 * xd[ab + 1]));
					macc += dmin * (m0 * xd[ab] * xs[ab] + m1 * xd[ab + 1] * xs[ab + 1]);
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD) - macc;
		});
	}

	// ── Q6_K ────────────────────────────────────────────────────────────────

	static void q6kScalar(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final byte[] xq = a.qb();
		final float[] xd = a.d();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float acc = 0f;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = b * 256 + half * 128;
					for (int is = 0; is < 2; is++) {
						int s1 = 0, s2 = 0, s3 = 0, s4 = 0;
						for (int l = is * 16; l < is * 16 + 16; l++) {
							int qlL = raw[qlOff + l] & 0xFF;
							int qlL2 = raw[qlOff + l + 32] & 0xFF;
							int qhL = raw[qhOff + l] & 0xFF;
							s1 += (((qlL & 0x0F) | ((qhL & 3) << 4)) - 32) * xq[xOff + l];
							s2 += (((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32) * xq[xOff + l + 32];
							s3 += (((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32) * xq[xOff + l + 64];
							s4 += (((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32) * xq[xOff + l + 96];
						}
						int ab = (xOff + is * 16) >> 5;
						acc += d * (raw[scOff + is] * xd[ab] * s1 + raw[scOff + is + 2] * xd[ab + 1] * s2
								+ raw[scOff + is + 4] * xd[ab + 2] * s3 + raw[scOff + is + 6] * xd[ab + 3] * s4);
					}
				}
			}
			y[r] = acc;
		});
	}

	static void q6kVector(byte[] raw, float[] x, float[] y, int rows, int cols) {
		Q8 a = quantize(x);
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final short[] xq = a.qs();
		final float[] xd = a.d();
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			FloatVector facc = FloatVector.zero(F);
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = b * 256 + half * 128;
					for (int is = 0; is < 2; is++) {
						int l0 = is * 16;
						ByteVector qa = ByteVector.fromArray(B, raw, qlOff + l0);
						ByteVector qb = ByteVector.fromArray(B, raw, qlOff + l0 + 32);
						ByteVector qh = ByteVector.fromArray(B, raw, qhOff + l0);
						ShortVector s1 = ShortVector.zero(S), s2 = ShortVector.zero(S);
						ShortVector s3 = ShortVector.zero(S), s4 = ShortVector.zero(S);
						for (int part = 0; part < 2; part++) {
							ShortVector wa = (ShortVector) qa.convertShape(VectorOperators.B2S, S, part);
							ShortVector wb = (ShortVector) qb.convertShape(VectorOperators.B2S, S, part);
							ShortVector wh = (ShortVector) qh.convertShape(VectorOperators.B2S, S, part);
							ShortVector q1 = wa.and((short) 0x0F).or(wh.and((short) 3).lanewise(VectorOperators.LSHL, 4))
									.sub((short) 32);
							ShortVector q2 = wb.and((short) 0x0F)
									.or(wh.lanewise(VectorOperators.ASHR, 2).and((short) 3).lanewise(VectorOperators.LSHL, 4))
									.sub((short) 32);
							ShortVector q3 = wa.lanewise(VectorOperators.ASHR, 4).and((short) 0x0F)
									.or(wh.lanewise(VectorOperators.ASHR, 4).and((short) 3).lanewise(VectorOperators.LSHL, 4))
									.sub((short) 32);
							ShortVector q4 = wb.lanewise(VectorOperators.ASHR, 4).and((short) 0x0F)
									.or(wh.lanewise(VectorOperators.ASHR, 6).and((short) 3).lanewise(VectorOperators.LSHL, 4))
									.sub((short) 32);
							int xo = xOff + l0 + part * 8;
							s1 = s1.add(q1.mul(ShortVector.fromArray(S, xq, xo)));
							s2 = s2.add(q2.mul(ShortVector.fromArray(S, xq, xo + 32)));
							s3 = s3.add(q3.mul(ShortVector.fromArray(S, xq, xo + 64)));
							s4 = s4.add(q4.mul(ShortVector.fromArray(S, xq, xo + 96)));
						}
						int ab = (xOff + l0) >> 5;
						facc = facc.add(widen(s1).mul(d * raw[scOff + is] * xd[ab]));
						facc = facc.add(widen(s2).mul(d * raw[scOff + is + 2] * xd[ab + 1]));
						facc = facc.add(widen(s3).mul(d * raw[scOff + is + 4] * xd[ab + 2]));
						facc = facc.add(widen(s4).mul(d * raw[scOff + is + 6] * xd[ab + 3]));
					}
				}
			}
			y[r] = facc.reduceLanes(VectorOperators.ADD);
		});
	}

	/** Sums a short vector's two halves into four int lanes and converts them to float. */
	private static FloatVector widen(ShortVector s) {
		IntVector i = ((IntVector) s.convertShape(VectorOperators.S2I, I, 0))
				.add((IntVector) s.convertShape(VectorOperators.S2I, I, 1));
		return (FloatVector) i.convertShape(VectorOperators.I2F, F, 0);
	}
}
