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
 * are widened to int and scaled as floats once per sub-block. Q6_K extracts its
 * 6-bit weights the same way, one 16-weight sub-block sum at a time (see
 * {@link #q6kRow}). Q8_0 uses a scalar integer loop, which measured faster than
 * its vector form at the decode thread count on the baseline host (AVX without
 * AVX2).
 *
 * <p>This class is the 128-bit form, and the reference for the 256-bit form
 * ({@link CpuIntegerKernels256}), which runs instead where
 * {@link CpuIntegerWidth} selects 256 bits (a host with 256-bit integer
 * instructions, or {@code -Djuno.cpu.integerWidth=256}); wider integer lanes do
 * not compile to hardware instructions on a host without AVX2. Rows run on the
 * worker pool.
 */
final class CpuIntegerKernels {

	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;

	private static final VarHandle LE_LONG = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
	private static final VarHandle LE_INT = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

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
		matVecInto(A, x, rowStart, rowEnd, y, cols, CpuIntegerWidth.WIDE);
	}

	/**
	 * {@link #matVecInto(GgufReader.QuantizedTensor, float[], int, int, float[], int)} at a given lane width:
	 * {@code wide} runs {@link CpuIntegerKernels256}. For tests that compare the widths.
	 */
	static void matVecInto(GgufReader.QuantizedTensor A, float[] x, int rowStart, int rowEnd, float[] y, int cols,
			boolean wide) {
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
			case 12 -> {
				if (wide)
					CpuIntegerKernels256.q4k(raw, a, rowStart, rowEnd, y, cols);
				else
					q4k(raw, a, rowStart, rowEnd, y, cols);
			}
			case 13 -> {
				if (wide)
					CpuIntegerKernels256.q5k(raw, a, rowStart, rowEnd, y, cols);
				else
					q5k(raw, a, rowStart, rowEnd, y, cols);
			}
			case 14 -> {
				if (wide)
					CpuIntegerKernels256.q6k(raw, a, rowStart, rowEnd, y, cols);
				else
					q6k(raw, a, rowStart, rowEnd, y, cols);
			}
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
	 * and the Q6_K offset terms need. {@code xe} and {@code xo} hold the even and
	 * odd values of {@code q} sign-extended to 16 bits ({@code xe[i] = q[2i]},
	 * {@code xo[i] = q[2i + 1]}), split once per row instead of once per weight row
	 * by the Q4_K kernel. Buffers grow and are reused.
	 */
	static final class Q8Activation {
		byte[] q = new byte[0];
		float[] d = new float[0];
		int[] sum32 = new int[0];
		int[] sum16 = new int[0];
		short[] xe = new short[0];
		short[] xo = new short[0];
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
				xe = new short[cols / 2];
				xo = new short[cols / 2];
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
			for (int i = 0; i < cols / 2; i++) {
				xe[i] = q[2 * i];
				xo[i] = q[2 * i + 1];
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

	// ── helpers ─────────────────────────────────────────────────────────────
	// No method here takes or returns a vector: C2 boxes a vector that crosses a call it does not inline
	// (CpuKernelVectorEscapeTest). Lane sums are written in place, in a fixed order, because reduceLanes(ADD)
	// leaves the order unspecified and its interpreted and compiled forms differ.

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	// ── Q4_K: [d f16][dmin f16][scales 12][qs 128] per 256 ───────────────────

	private static void q4k(byte[] raw, Q8Activation a, int rowStart, int rowEnd, float[] y, int cols) {
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
	 * One Q4_K row's dot with the quantized activation.
	 *
	 * <p>The block's eight 6-bit scales and eight 6-bit minimums are decoded once, without branches, from its twelve
	 * scale bytes read as one little-endian long and one int, into two longs holding one value per byte. Each
	 * 32-weight sub-block's exact integer sum stays in four int lanes, is converted to float lanes and multiplied by
	 * the sub-block's scale times the activation scale, and accumulates in a float vector; the row is summed across
	 * lanes once, in a fixed order. The minimum term is accumulated per block as a scalar. Against summing each
	 * sub-block as a scalar in row order this changes float rounding only (about 1e-7 of the row's magnitude), and the
	 * result is the same on every call. The activation's even and odd bytes come pre-split and sign-extended
	 * ({@link Q8Activation#xe}, {@link Q8Activation#xo}), so each 16 weight bytes take only the weight's own nibble
	 * extraction. Every lane operation is written in place: a vector passed to a helper method is boxed by C2.
	 */
	static float q4kRow(byte[] raw, int rowByteOffset, int blocksPerRow, short[] xe, short[] xo, float[] xd, int[] xs) {
		FloatVector accV = FloatVector.zero(F);
		float minTot = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			// scale byte k: k < 4 holds sc[k] and k 4..7 holds mn[k - 4] in its low 6 bits, and the top 2 bits of
			// sc[k + 4] (k < 4) or mn[k] (k 4..7); bytes 8..11 hold the low 4 bits of sc[4..7] and mn[4..7]
			long lo = (long) LE_LONG.get(raw, bo + 4);
			long hi = (int) LE_INT.get(raw, bo + 12) & 0xFFFFFFFFL;
			long low6 = lo & 0x3F3F3F3F3F3F3F3FL;
			long top2 = (lo >>> 6) & 0x0303030303030303L;
			long sc = (low6 & 0xFFFFFFFFL) | (((hi & 0x0F0F0F0FL) | ((top2 & 0xFFFFFFFFL) << 4)) << 32);
			long mn = (low6 >>> 32) | ((((hi >>> 4) & 0x0F0F0F0FL) | ((top2 >>> 32) << 4)) << 32);
			float minAcc = 0f;
			for (int g = 0; g < 4; g++) {
				int q = bo + 16 + g * 32;
				int x0 = b * 256 + g * 64;
				ShortVector aLo = ShortVector.zero(S);
				ShortVector aHi = ShortVector.zero(S);
				for (int h = 0; h < 32; h += 16) {
					ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
					int xl = (x0 + h) >> 1;
					int xh = (x0 + 32 + h) >> 1;
					aLo = aLo.add(w.and((short) 0x0F).mul(ShortVector.fromArray(S, xe, xl)))
							.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).mul(ShortVector.fromArray(S, xo, xl)));
					aHi = aHi.add(w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).mul(ShortVector.fromArray(S, xe, xh)))
							.add(w.lanewise(VectorOperators.LSHR, 12).mul(ShortVector.fromArray(S, xo, xh)));
				}
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
		return (accV.lane(0) + accV.lane(1)) + (accV.lane(2) + accV.lane(3)) - minTot;
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
		final int[] xs = a.sum16;
		CpuThreads.forEach(rowEnd - rowStart,
				r -> y[r] = q6kRow(raw, (rowStart + r) * bytesPerRow, blocksPerRow, xq, xd, xs));
	}

	/**
	 * One Q6_K row's dot with the quantized activation; {@code xs} is the activation's sums per 16 values.
	 *
	 * <p>Each 16-weight sub-block's integer sum is formed in vector lanes: its 16 low-nibble bytes and 16 high-bit
	 * bytes are read as eight 16-bit lanes, the even and odd bytes' unsigned 6-bit weights (0 to 63) are extracted
	 * with shifts and masks and multiplied by the matching even and odd activation bytes, and the lane sums (at most
	 * two products of 63 x 127 each) are widened to int and reduced. The stored offset of 32 is subtracted once per
	 * sub-block as 32 times the activation sum. The four sums of a pass are the scalar loop's sums exactly, and the
	 * float scaling expression is the scalar loop's (the pass returns {@code d} times the bracket and the row adds it, as
	 * the scalar loop's {@code acc += d * (...)} did), so the result is bit-identical to it. The sums are formed one
	 * at a time, so few vector registers are live at once.
	 */
	static float q6kRow(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 210;
			float d = f16(raw, bo + 208);
			for (int half = 0; half < 2; half++)
				for (int is = 0; is < 2; is++)
					acc += q6kPass(raw, bo + half * 64, bo + 128 + half * 32, bo + 192 + half * 8, b * 256 + half * 128,
							is, d, xq, xd, xs);
		}
		return acc;
	}

	/**
	 * One pass of {@link #q6kRow}: the four 16-weight sub-blocks that share a pass's {@code ql} and {@code qh}
	 * bytes, as {@code d} times their scaled sum. Every lane operation is written out in place: passing a vector to
	 * a helper method made C2 box the vectors (about 350 MB allocated per Mistral 7B FFN matrix call, and 1.9 times
	 * slower than the scalar loop at 12 threads), as it did for the earlier vector form of this kernel.
	 */
	private static float q6kPass(byte[] raw, int qlOff, int qhOff, int scOff, int xOff, int is, float d, byte[] xq,
			float[] xd, int[] xs) {
		int l0 = is * 16;
		ShortVector qa = ByteVector.fromArray(B, raw, qlOff + l0).reinterpretAsShorts();
		ShortVector qb = ByteVector.fromArray(B, raw, qlOff + l0 + 32).reinterpretAsShorts();
		ShortVector qh = ByteVector.fromArray(B, raw, qhOff + l0).reinterpretAsShorts();
		int xo = xOff + l0;
		ShortVector x1 = ByteVector.fromArray(B, xq, xo).reinterpretAsShorts();
		ShortVector p1 = qa.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and((short) 0x30))
				.mul(x1.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
				.add(qa.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
						.or(qh.lanewise(VectorOperators.LSHR, 4).and((short) 0x30)).mul(x1.lanewise(VectorOperators.ASHR, 8)));
		IntVector i1 = p1.reinterpretAsInts();
		int s1 = i1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
				.add(i1.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD) - 32 * xs[xo >> 4];
		ShortVector x2 = ByteVector.fromArray(B, xq, xo + 32).reinterpretAsShorts();
		ShortVector p2 = qb.and((short) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and((short) 0x30))
				.mul(x2.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
				.add(qb.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
						.or(qh.lanewise(VectorOperators.LSHR, 6).and((short) 0x30)).mul(x2.lanewise(VectorOperators.ASHR, 8)));
		IntVector i2 = p2.reinterpretAsInts();
		int s2 = i2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
				.add(i2.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD) - 32 * xs[(xo >> 4) + 2];
		ShortVector x3 = ByteVector.fromArray(B, xq, xo + 64).reinterpretAsShorts();
		ShortVector p3 = qa.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).or(qh.and((short) 0x30))
				.mul(x3.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
				.add(qa.lanewise(VectorOperators.LSHR, 12)
						.or(qh.lanewise(VectorOperators.LSHR, 8).and((short) 0x30)).mul(x3.lanewise(VectorOperators.ASHR, 8)));
		IntVector i3 = p3.reinterpretAsInts();
		int s3 = i3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
				.add(i3.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD) - 32 * xs[(xo >> 4) + 4];
		ShortVector x4 = ByteVector.fromArray(B, xq, xo + 96).reinterpretAsShorts();
		ShortVector p4 = qb.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
				.or(qh.lanewise(VectorOperators.LSHR, 2).and((short) 0x30)).mul(x4.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8))
				.add(qb.lanewise(VectorOperators.LSHR, 12)
						.or(qh.lanewise(VectorOperators.LSHR, 10).and((short) 0x30)).mul(x4.lanewise(VectorOperators.ASHR, 8)));
		IntVector i4 = p4.reinterpretAsInts();
		int s4 = i4.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
				.add(i4.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD) - 32 * xs[(xo >> 4) + 6];
		int ab = xo >> 5;
		return d * (raw[scOff + is] * xd[ab] * s1 + raw[scOff + is + 2] * xd[ab + 1] * s2
				+ raw[scOff + is + 4] * xd[ab + 2] * s3 + raw[scOff + is + 6] * xd[ab + 3] * s4);
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
