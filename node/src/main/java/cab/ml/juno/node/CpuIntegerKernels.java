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
 * Integer-dot CPU matrix-vector kernels for Q4_K, Q5_K, Q6_K and Q8_0 weights.
 *
 * <p>The activation row is quantized once per call to 8-bit integers with one
 * float scale per 32 values ({@link Q8Activation}). Each weight sub-block (32
 * values for Q4_K, Q5_K and Q8_0, 16 for Q6_K) then becomes an integer dot
 * product against the packed weights, and its scales are applied once. Integer
 * accumulation is exact, so the only numerical change against a float
 * activation is the activation's quantization.
 *
 * <p>Q4_K and Q5_K use the Vector API at the 128-bit species: 16 packed weight
 * bytes are reinterpreted as eight 16-bit lanes, the even and odd bytes'
 * nibbles are extracted with shifts and masks, and the 8-bit activation is
 * split into the matching even and odd lanes the same way, so no conversion or
 * shuffle runs in the inner loop. Lane products accumulate in 16 bits within a
 * sub-block (bounded: at most four products of at most 31 x 127 per lane), and
 * are widened to int and scaled as floats once per sub-block. Q6_K and Q8_0 use
 * scalar integer loops, which measured faster than their vector forms at the
 * decode thread count on the baseline host (AVX without AVX2).
 *
 * <p>The species width is fixed at 128 bits in code: wider integer lanes do not
 * compile to hardware instructions on a host without AVX2, and on a wider host
 * the 128-bit form stays correct. Rows run on the common pool, as the float
 * kernels do.
 */
final class CpuIntegerKernels {

	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;

	private static final ThreadLocal<Q8Activation> SCRATCH = ThreadLocal.withInitial(Q8Activation::new);

	private CpuIntegerKernels() {
	}

	/** True for the GGUF tensor types these kernels implement; see {@link CpuKernelSelection#integerFormat}. */
	static boolean supports(int ggmlType) {
		return CpuKernelSelection.integerFormat(ggmlType);
	}

	/**
	 * {@code y[r - rowStart] = dot(A[r], x)} for {@code r} in
	 * {@code [rowStart, rowEnd)}, over an 8-bit quantization of {@code x}.
	 * Callers choose this kernel through {@link CpuKernelSelection}.
	 *
	 * @throws IllegalArgumentException if {@link #supports} is false for the type
	 */
	static void matVecInto(GgufReader.QuantizedTensor A, float[] x, int rowStart, int rowEnd, float[] y, int cols) {
		if (!supports(A.type()))
			throw new IllegalArgumentException("no integer kernel for GGUF tensor type " + A.type());
		// A common-pool worker waiting on its rows may run another task that calls in
		// here on the same thread; that call must not overwrite this one's activation.
		Q8Activation a = SCRATCH.get();
		if (a.inUse)
			a = new Q8Activation();
		a.inUse = true;
		try {
			a.quantizeFrom(x, cols);
			byte[] raw = A.data();
			switch (A.type()) {
			case 12 -> q4k(raw, a, rowStart, rowEnd, y, cols);
			case 13 -> q5k(raw, a, rowStart, rowEnd, y, cols);
			case 14 -> q6k(raw, a, rowStart, rowEnd, y, cols);
			default -> q8_0(raw, a, rowStart, rowEnd, y, cols);
			}
		} finally {
			a.inUse = false;
		}
	}

	// ── activation ──────────────────────────────────────────────────────────

	/**
	 * An activation row quantized per 32 values: {@code q} in [-127, 127],
	 * scale {@code d} (so the value is {@code q * d}), and the sums of
	 * {@code q} over each 32 and each 16 values, which the Q4_K/Q5_K minimum
	 * and the Q6_K offset terms need. Buffers grow and are reused.
	 */
	static final class Q8Activation {
		byte[] q = new byte[0];
		float[] d = new float[0];
		int[] sum32 = new int[0];
		int[] sum16 = new int[0];
		int cols;
		boolean inUse;

		static Q8Activation quantize(float[] x, int cols) {
			Q8Activation a = new Q8Activation();
			a.quantizeFrom(x, cols);
			return a;
		}

		void quantizeFrom(float[] x, int cols) {
			if (cols % 32 != 0)
				throw new IllegalArgumentException("cols must be a multiple of 32, got " + cols);
			if (q.length < cols) {
				q = new byte[cols];
				d = new float[cols / 32];
				sum32 = new int[cols / 32];
				sum16 = new int[cols / 16];
			}
			this.cols = cols;
			for (int b = 0, o = 0; o < cols; b++, o += 32) {
				float amax = 0f;
				for (int i = 0; i < 32; i++)
					amax = Math.max(amax, Math.abs(x[o + i]));
				float scale = amax / 127f;
				float inv = scale == 0f ? 0f : 1f / scale;
				int s0 = 0;
				int s1 = 0;
				for (int i = 0; i < 16; i++) {
					int v = Math.round(x[o + i] * inv);
					q[o + i] = (byte) v;
					s0 += v;
				}
				for (int i = 16; i < 32; i++) {
					int v = Math.round(x[o + i] * inv);
					q[o + i] = (byte) v;
					s1 += v;
				}
				d[b] = scale;
				sum16[2 * b] = s0;
				sum16[2 * b + 1] = s1;
				sum32[b] = s0 + s1;
			}
		}

		/** The values the kernels actually multiply, {@code q * d}; for tests. */
		float[] dequantized() {
			float[] out = new float[cols];
			for (int i = 0; i < cols; i++)
				out[i] = q[i] * d[i >> 5];
			return out;
		}
	}

	// ── lane helpers ────────────────────────────────────────────────────────

	private static ShortVector evenBytes(ShortVector v) {
		return v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8);
	}

	private static ShortVector oddBytes(ShortVector v) {
		return v.lanewise(VectorOperators.ASHR, 8);
	}

	/**
	 * Sums the four float lanes in a fixed order. {@code reduceLanes(ADD)} leaves the
	 * order unspecified, and the interpreted and compiled forms differ, so a row's
	 * result would depend on which compilation tier ran it.
	 */
	static float sumLanes(FloatVector v) {
		return (v.lane(0) + v.lane(1)) + (v.lane(2) + v.lane(3));
	}

	/** Sums adjacent 16-bit lanes into four sign-extended int lanes, as floats. */
	private static FloatVector widenPairs(ShortVector s) {
		IntVector w = s.reinterpretAsInts();
		IntVector lo = w.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16);
		IntVector hi = w.lanewise(VectorOperators.ASHR, 16);
		return (FloatVector) lo.add(hi).convertShape(VectorOperators.I2F, F, 0);
	}

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	// ── Q4_K: [d f16][dmin f16][scales 12][qs 128] per 256 ───────────────────

	private static void q4k(byte[] raw, Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		final int[] xs = a.sum32;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q4kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, xs));
	}

	/** One Q4_K row's dot with the quantized activation. */
	static float q4kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 16 + g * 32;
				int x0 = b * 256 + g * 64;
				ShortVector aLo = ShortVector.zero(S);
				ShortVector aHi = ShortVector.zero(S);
				for (int h = 0; h < 32; h += 16) {
					ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
					ShortVector xl = ByteVector.fromArray(B, xq, x0 + h).reinterpretAsShorts();
					ShortVector xh = ByteVector.fromArray(B, xq, x0 + 32 + h).reinterpretAsShorts();
					aLo = aLo.add(w.and((short) 0x0F)
							.mul(xl.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
									.mul(xl.lanewise(VectorOperators.ASHR, 8)));
					aHi = aHi.add(w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
							.mul(xh.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 12).mul(xh.lanewise(VectorOperators.ASHR, 8)));
				}
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

	// ── Q5_K: [d f16][dmin f16][scales 12][qh 32][qs 128] per 256 ────────────

	private static void q5k(byte[] raw, Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 176;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		final int[] xs = a.sum32;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q5kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, xs));
	}

	/** One Q5_K row's dot with the quantized activation. */
	static float q5kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 176;
			int scBase = bo + 4;
			int qhBase = bo + 16;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 48 + g * 32;
				int x0 = b * 256 + g * 64;
				int bit = 2 * g;
				ShortVector aLo = ShortVector.zero(S);
				ShortVector aHi = ShortVector.zero(S);
				for (int h = 0; h < 32; h += 16) {
					ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
					ShortVector qh = ByteVector.fromArray(B, raw, qhBase + h).reinterpretAsShorts();
					ShortVector t0 = qh.lanewise(VectorOperators.LSHR, bit);
					ShortVector t1 = qh.lanewise(VectorOperators.LSHR, bit + 1);
					ShortVector xl = ByteVector.fromArray(B, xq, x0 + h).reinterpretAsShorts();
					ShortVector xh = ByteVector.fromArray(B, xq, x0 + 32 + h).reinterpretAsShorts();
					aLo = aLo.add(w.and((short) 0x0F).or(t0.lanewise(VectorOperators.LSHL, 4).and((short) 0x10))
							.mul(xl.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
									.or(t0.lanewise(VectorOperators.LSHR, 4).and((short) 0x10))
									.mul(xl.lanewise(VectorOperators.ASHR, 8)));
					aHi = aHi.add(w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
							.or(t1.lanewise(VectorOperators.LSHL, 4).and((short) 0x10))
							.mul(xh.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 12)
									.or(t1.lanewise(VectorOperators.LSHR, 4).and((short) 0x10))
									.mul(xh.lanewise(VectorOperators.ASHR, 8)));
				}
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

	// ── Q6_K: [ql 128][qh 64][scales 16][d f16] per 256 ──────────────────────

	private static void q6k(byte[] raw, Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q6kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, null));
	}

	/** One Q6_K row's dot with the quantized activation. */
	static float q6kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 210;
			float d = f16(raw, bo + 208);
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
		return acc;
	}

	// ── Q8_0: [d f16][qs 32] per 32 ──────────────────────────────────────────

	private static void q8_0(byte[] raw, Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
		final int blocksPerRow = cols / 32;
		final int bytesPerRow = blocksPerRow * 34;
		final byte[] xq = a.q;
		final float[] xd = a.d;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q8_0Row(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, null));
	}

	/** One Q8_0 row's dot with the quantized activation. */
	static float q8_0Row(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 34;
			int xo = b * 32;
			int s = 0;
			for (int i = 0; i < 32; i++)
				s += raw[bo + 2 + i] * xq[xo + i];
			acc += f16(raw, bo) * xd[b] * s;
		}
		return acc;
	}
}
