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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Q4_K, Q5_K and Q6_K integer decode kernels of {@link CpuIntegerKernels} at
 * 256-bit lanes, for hosts with 256-bit integer instructions; chosen by
 * {@link CpuIntegerWidth}. Q8_0 stays on the scalar kernel at either width.
 *
 * <p>Each kernel is its 128-bit form with one 32-byte load where that form takes
 * two 16-byte loads; the integer sums are the same. Q5_K and Q6_K reduce each
 * sub-block to an int and keep the 128-bit float expression, so they return the
 * same bits. Q4_K scales its sub-block sums in eight float lanes instead of four
 * and sums the row across them in a fixed order, so it differs from the 128-bit
 * kernel by float rounding only. Every lane operation is written in place: C2
 * boxes a vector passed to a helper it does not inline.
 */
final class CpuIntegerKernels256 {

	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_256;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_256;
	private static final VectorSpecies<Integer> I = IntVector.SPECIES_256;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_256;

	private static final VarHandle LE_LONG = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
	private static final VarHandle LE_INT = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

	/** All ones in the low four int lanes: selects the first of the two 16-weight sub-blocks a Q6_K load covers. */
	private static final int[] LOW_HALF = { -1, -1, -1, -1, 0, 0, 0, 0 };

	private CpuIntegerKernels256() {
	}

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	// ── Q4_K ────────────────────────────────────────────────────────────────

	static void q4k(byte[] raw, CpuIntegerKernels.Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final short[] xe = a.xe;
		final short[] xo = a.xo;
		final float[] xd = a.d;
		final int[] xs = a.sum32;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q4kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xe, xo, xd, xs));
	}

	/**
	 * One Q4_K row; {@link CpuIntegerKernels#q4kRow} with each sub-block's 32 weight bytes in one load (sixteen
	 * 16-bit lanes), its exact integer sum in eight int lanes, scaled in eight float lanes.
	 */
	static float q4kRow(byte[] raw, int rowByteOffset, int blocksPerRow, short[] xe, short[] xo, float[] xd, int[] xs) {
		FloatVector accV = FloatVector.zero(F);
		float minTot = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			long lo = (long) LE_LONG.get(raw, bo + 4);
			long hi = (int) LE_INT.get(raw, bo + 12) & 0xFFFFFFFFL;
			long low6 = lo & 0x3F3F3F3F3F3F3F3FL;
			long top2 = (lo >>> 6) & 0x0303030303030303L;
			long sc = (low6 & 0xFFFFFFFFL) | (((hi & 0x0F0F0F0FL) | ((top2 & 0xFFFFFFFFL) << 4)) << 32);
			long mn = (low6 >>> 32) | ((((hi >>> 4) & 0x0F0F0F0FL) | ((top2 >>> 32) << 4)) << 32);
			float minAcc = 0f;
			for (int g = 0; g < 4; g++) {
				int x0 = b * 256 + g * 64;
				int xl = x0 >> 1;
				int xh = (x0 + 32) >> 1;
				ShortVector w = ByteVector.fromArray(B, raw, bo + 16 + g * 32).reinterpretAsShorts();
				ShortVector aLo = w.and((short) 0x0F).mul(ShortVector.fromArray(S, xe, xl))
						.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).mul(ShortVector.fromArray(S, xo, xl)));
				ShortVector aHi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).mul(ShortVector.fromArray(S, xe, xh))
						.add(w.lanewise(VectorOperators.LSHR, 12).mul(ShortVector.fromArray(S, xo, xh)));
				IntVector iLo = aLo.reinterpretAsInts();
				IntVector iHi = aHi.reinterpretAsInts();
				IntVector sLo = iLo.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iLo.lanewise(VectorOperators.ASHR, 16));
				IntVector sHi = iHi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iHi.lanewise(VectorOperators.ASHR, 16));
				int ab = x0 >> 5;
				int sh = g * 16;
				float fLo = xd[ab] * d * (float) ((int) (sc >>> sh) & 0xFF);
				float fHi = xd[ab + 1] * d * (float) ((int) (sc >>> (sh + 8)) & 0xFF);
				accV = accV.add(((FloatVector) sLo.convertShape(VectorOperators.I2F, F, 0)).mul(fLo))
						.add(((FloatVector) sHi.convertShape(VectorOperators.I2F, F, 0)).mul(fHi));
				minAcc += xd[ab] * ((int) (mn >>> sh) & 0xFF) * xs[ab]
						+ xd[ab + 1] * ((int) (mn >>> (sh + 8)) & 0xFF) * xs[ab + 1];
			}
			minTot += dmin * minAcc;
		}
		return ((accV.lane(0) + accV.lane(1)) + (accV.lane(2) + accV.lane(3)))
				+ ((accV.lane(4) + accV.lane(5)) + (accV.lane(6) + accV.lane(7))) - minTot;
	}

	// ── Q5_K ────────────────────────────────────────────────────────────────

	static void q5k(byte[] raw, CpuIntegerKernels.Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 176;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		final int[] xs = a.sum32;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q5kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, xs));
	}

	/** One Q5_K row; {@link CpuIntegerKernels#q5kRow} with each sub-block's bytes in one load. Same bits. */
	static float q5kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 176;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			ShortVector qh = ByteVector.fromArray(B, raw, bo + 16).reinterpretAsShorts();
			for (int g = 0; g < 4; g++) {
				int x0 = b * 256 + g * 64;
				int bit = 2 * g;
				ShortVector w = ByteVector.fromArray(B, raw, bo + 48 + g * 32).reinterpretAsShorts();
				ShortVector t0 = qh.lanewise(VectorOperators.LSHR, bit);
				ShortVector t1 = qh.lanewise(VectorOperators.LSHR, bit + 1);
				ShortVector xl = ByteVector.fromArray(B, xq, x0).reinterpretAsShorts();
				ShortVector xh = ByteVector.fromArray(B, xq, x0 + 32).reinterpretAsShorts();
				ShortVector aLo = w.and((short) 0x0F).or(t0.lanewise(VectorOperators.LSHL, 4).and((short) 0x10))
						.mul(xl.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
								.or(t0.lanewise(VectorOperators.LSHR, 4).and((short) 0x10))
								.mul(xl.lanewise(VectorOperators.ASHR, 8)));
				ShortVector aHi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
						.or(t1.lanewise(VectorOperators.LSHL, 4).and((short) 0x10))
						.mul(xh.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(w.lanewise(VectorOperators.LSHR, 12)
								.or(t1.lanewise(VectorOperators.LSHR, 4).and((short) 0x10))
								.mul(xh.lanewise(VectorOperators.ASHR, 8)));
				IntVector iLo = aLo.reinterpretAsInts();
				IntVector iHi = aHi.reinterpretAsInts();
				int dotLo = iLo.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iLo.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
				int dotHi = iHi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iHi.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
				int ab = x0 >> 5;
				acc += xd[ab] * (d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g) * dotLo
						- dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g) * xs[ab]);
				acc += xd[ab + 1] * (d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1) * dotHi
						- dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1) * xs[ab + 1]);
			}
		}
		return acc;
	}

	// ── Q6_K ────────────────────────────────────────────────────────────────

	static void q6k(byte[] raw, CpuIntegerKernels.Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		final int[] xs = a.sum16;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q6kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, xs));
	}

	/**
	 * One Q6_K row; {@link CpuIntegerKernels#q6kRow} with both of a pass's 16-byte halves in one load. Each product
	 * vector then holds two 16-weight sub-blocks, one per 128-bit half; the low half's sum is the reduction of the
	 * low four int lanes and the high half's is the whole reduction minus it, both exact. The float expression is
	 * the 128-bit kernel's, the first half's pass added before the second's, so the result is the same bits.
	 */
	static float q6kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		IntVector low = IntVector.fromArray(I, LOW_HALF, 0);
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 210;
			float d = f16(raw, bo + 208);
			for (int half = 0; half < 2; half++) {
				int qlOff = bo + half * 64;
				int scOff = bo + 192 + half * 8;
				int xo = b * 256 + half * 128;
				ShortVector qa = ByteVector.fromArray(B, raw, qlOff).reinterpretAsShorts();
				ShortVector qb = ByteVector.fromArray(B, raw, qlOff + 32).reinterpretAsShorts();
				ShortVector qh = ByteVector.fromArray(B, raw, bo + 128 + half * 32).reinterpretAsShorts();
				ShortVector x1 = ByteVector.fromArray(B, xq, xo).reinterpretAsShorts();
				IntVector i1 = qa.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and((short) 0x30))
						.mul(x1.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(qa.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 4).and((short) 0x30)).mul(x1.lanewise(VectorOperators.ASHR, 8)))
						.reinterpretAsInts();
				IntVector v1 = i1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(i1.lanewise(VectorOperators.ASHR, 16));
				int t1 = v1.reduceLanes(VectorOperators.ADD);
				int a1 = v1.and(low).reduceLanes(VectorOperators.ADD);
				ShortVector x2 = ByteVector.fromArray(B, xq, xo + 32).reinterpretAsShorts();
				IntVector i2 = qb.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and((short) 0x30))
						.mul(x2.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(qb.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
								.or(qh.lanewise(VectorOperators.LSHR, 6).and((short) 0x30)).mul(x2.lanewise(VectorOperators.ASHR, 8)))
						.reinterpretAsInts();
				IntVector v2 = i2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(i2.lanewise(VectorOperators.ASHR, 16));
				int t2 = v2.reduceLanes(VectorOperators.ADD);
				int a2 = v2.and(low).reduceLanes(VectorOperators.ADD);
				ShortVector x3 = ByteVector.fromArray(B, xq, xo + 64).reinterpretAsShorts();
				IntVector i3 = qa.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).or(qh.and((short) 0x30))
						.mul(x3.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(qa.lanewise(VectorOperators.LSHR, 12)
								.or(qh.lanewise(VectorOperators.LSHR, 8).and((short) 0x30)).mul(x3.lanewise(VectorOperators.ASHR, 8)))
						.reinterpretAsInts();
				IntVector v3 = i3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(i3.lanewise(VectorOperators.ASHR, 16));
				int t3 = v3.reduceLanes(VectorOperators.ADD);
				int a3 = v3.and(low).reduceLanes(VectorOperators.ADD);
				ShortVector x4 = ByteVector.fromArray(B, xq, xo + 96).reinterpretAsShorts();
				IntVector i4 = qb.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
						.or(qh.lanewise(VectorOperators.LSHR, 2).and((short) 0x30))
						.mul(x4.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
						.add(qb.lanewise(VectorOperators.LSHR, 12)
								.or(qh.lanewise(VectorOperators.LSHR, 10).and((short) 0x30)).mul(x4.lanewise(VectorOperators.ASHR, 8)))
						.reinterpretAsInts();
				IntVector v4 = i4.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(i4.lanewise(VectorOperators.ASHR, 16));
				int t4 = v4.reduceLanes(VectorOperators.ADD);
				int a4 = v4.and(low).reduceLanes(VectorOperators.ADD);
				int s = xo >> 4;
				int ab = xo >> 5;
				// first half (is = 0): weights l = 0..15 of each 32-weight quarter; second half (is = 1): l = 16..31
				acc += d * (raw[scOff] * xd[ab] * (a1 - 32 * xs[s]) + raw[scOff + 2] * xd[ab + 1] * (a2 - 32 * xs[s + 2])
						+ raw[scOff + 4] * xd[ab + 2] * (a3 - 32 * xs[s + 4]) + raw[scOff + 6] * xd[ab + 3] * (a4 - 32 * xs[s + 6]));
				acc += d * (raw[scOff + 1] * xd[ab] * (t1 - a1 - 32 * xs[s + 1])
						+ raw[scOff + 3] * xd[ab + 1] * (t2 - a2 - 32 * xs[s + 3])
						+ raw[scOff + 5] * xd[ab + 2] * (t3 - a3 - 32 * xs[s + 5])
						+ raw[scOff + 7] * xd[ab + 3] * (t4 - a4 - 32 * xs[s + 7]));
			}
		}
		return acc;
	}
}
