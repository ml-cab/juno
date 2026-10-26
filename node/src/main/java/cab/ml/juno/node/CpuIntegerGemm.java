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
 * Cache-blocked integer-dot CPU matmul {@code Y[b] = A * X[b]} for Q4_K, Q5_K,
 * Q6_K and Q8_0 weights, at prefill width.
 *
 * <p>Every activation row is quantized once per call to 8-bit integers with one
 * float scale per 32 values (the same scheme as {@link CpuIntegerKernels}, so a
 * row gives the same products in either kernel), and held as 16-bit values.
 * Weight rows are split into tiles of {@link #TILE_ROWS}; one task owns a tile.
 * For each 256-column block the task unpacks its rows' weights once into a
 * 16-bit scratch, in natural order, with one float scale per sub-block (and the
 * Q4_K/Q5_K minimum or Q6_K offset term), extracting 16 packed bytes per vector
 * and widening them to 16 bits, then multiplies that block against
 * every activation row while it sits in L1. Each sub-block is an integer dot
 * product over 16-bit lanes, scaled once: 32 values for Q4_K and Q5_K (at most
 * four products of at most 31 x 127 per lane), 16 for Q6_K and Q8_0 (at most two
 * products of at most 63 x 127 or 128 x 127). Activation rows are taken four at
 * a time, the batch padded with zero rows to a multiple of four, so each loaded
 * weight vector serves four. The minimum or offset term is
 * taken against each activation sub-block's scale times its sum, computed once
 * per call, as four-lane float products subtracted from the row's lane sums.
 * Sums for a (row, activation row) pair accumulate in a tile-local buffer and are
 * written once, so no two tasks write the same cache line of {@code Y}. Every
 * lane operation, in the four-row loop and in the remainder, is written in place:
 * C2 boxes a vector passed to a helper it does not inline, and once the tile is
 * this large it does not inline one (a remainder helper allocated about a
 * gigabyte per request in a server before it was removed).
 *
 * <p>The tile here is the 128-bit form and the reference for
 * {@link CpuIntegerGemm256}'s, which runs where {@link CpuIntegerWidth} selects
 * 256 bits, for the reason {@link CpuIntegerKernels} gives. Requires
 * {@code cols % 256 == 0}; {@link #supports} says so.
 */
final class CpuIntegerGemm {

	/** Weight rows per task. 16 rows of one 256-column block are 8 KB of 16-bit weights. */
	static final int TILE_ROWS = 16;

	private static final VectorSpecies<Byte> BY = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;

	private static final ThreadLocal<Activations> ACTIVATIONS = ThreadLocal.withInitial(Activations::new);

	/**
	 * Tile scratch, shared by all tasks: a task takes one, runs its range of tiles,
	 * and returns it. A per-thread scratch would not survive: the common pool clears
	 * a worker's thread-locals after each top-level task.
	 */
	private static final java.util.concurrent.ConcurrentLinkedQueue<TileScratch> SCRATCH = new java.util.concurrent.ConcurrentLinkedQueue<>();

	/** Calls per format (Q8_0, Q4_K, Q5_K, Q6_K), so a test can tell a batched path from a per-row fallback. */
	private static final java.util.concurrent.atomic.LongAdder[] CALLS = {
			new java.util.concurrent.atomic.LongAdder(), new java.util.concurrent.atomic.LongAdder(),
			new java.util.concurrent.atomic.LongAdder(), new java.util.concurrent.atomic.LongAdder() };

	private CpuIntegerGemm() {
	}

	private static int formatIndex(int ggmlType) {
		return switch (ggmlType) {
		case 8 -> 0;
		case 12 -> 1;
		case 13 -> 2;
		default -> 3;
		};
	}

	/** Number of {@link #sgemmInto} calls over {@code ggmlType} since the last {@link #resetCalls}. */
	static long calls(int ggmlType) {
		return CALLS[formatIndex(ggmlType)].sum();
	}

	static void resetCalls() {
		for (java.util.concurrent.atomic.LongAdder a : CALLS)
			a.reset();
	}

	/** True when this matmul implements the type at this width. */
	static boolean supports(int ggmlType, int cols) {
		return CpuKernelSelection.integerFormat(ggmlType) && cols % 256 == 0;
	}

	/**
	 * {@code Y[b][r - rowStart] = dot(A[r], X[b])} for {@code r} in
	 * {@code [rowStart, rowEnd)} and every {@code b}.
	 *
	 * @throws IllegalArgumentException if {@link #supports} is false
	 */
	static void sgemmInto(GgufReader.QuantizedTensor A, float[][] X, float[][] Y, int rowStart, int rowEnd, int cols) {
		sgemmInto(A, X, Y, rowStart, rowEnd, cols, CpuIntegerWidth.WIDE);
	}

	/**
	 * {@link #sgemmInto(GgufReader.QuantizedTensor, float[][], float[][], int, int, int)} at a given lane width:
	 * {@code wide} runs {@link CpuIntegerGemm256}'s tile. For tests that compare the widths.
	 */
	static void sgemmInto(GgufReader.QuantizedTensor A, float[][] X, float[][] Y, int rowStart, int rowEnd, int cols,
			boolean wide) {
		if (!supports(A.type(), cols))
			throw new IllegalArgumentException("no integer matmul for GGUF tensor type " + A.type() + " at cols=" + cols);
		int batch = X.length;
		if (batch == 0 || rowEnd <= rowStart)
			return;
		CALLS[formatIndex(A.type())].increment();
		// A common-pool worker waiting on this call may run another task that calls in
		// here on the same thread; that call must not overwrite this one's activations.
		Activations held = ACTIVATIONS.get();
		final Activations act = held.inUse ? new Activations() : held;
		act.inUse = true;
		try {
			act.quantize(X, cols);
			final int type = A.type();
			final byte[] raw = A.data();
			final int rows = rowEnd - rowStart;
			final int tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;
			// A few tasks per thread, each a contiguous range of tiles, so scratch is
			// taken once per task rather than once per tile.
			final int tasks = Math.min(tiles, 4 * Math.max(CpuThreads.threads(),
					java.util.concurrent.ForkJoinPool.getCommonPoolParallelism() + 1));
			CpuThreads.forEach(tasks, k -> {
				TileScratch ts = SCRATCH.poll();
				if (ts == null)
					ts = new TileScratch();
				try {
					for (int t = (int) ((long) tiles * k / tasks), end = (int) ((long) tiles * (k + 1) / tasks); t < end; t++) {
						int r0 = rowStart + t * TILE_ROWS;
						int r1 = Math.min(rowEnd, r0 + TILE_ROWS);
						if (wide)
							CpuIntegerGemm256.tile(ts, type, raw, act, batch, cols, r0, r1, Y, rowStart);
						else
							tile(ts, type, raw, act, batch, cols, r0, r1, Y, rowStart);
					}
				} finally {
					SCRATCH.offer(ts);
				}
			});
		} finally {
			act.inUse = false;
		}
	}

	// ── activations ─────────────────────────────────────────────────────────

	/**
	 * All activation rows quantized per 32 values, as 16-bit values; buffers grow and are reused. The rows are padded
	 * with zero rows to a multiple of four ({@link #padded}), so the tile runs its four-row form on every row.
	 */
	static final class Activations {
		short[] q = new short[0];
		float[] d = new float[0];
		int[] sum32 = new int[0];
		int[] sum16 = new int[0];
		/** Per 32 and per 16 values: the scale times the sum of the quantized values, for the offset term. */
		float[] dsum32 = new float[0];
		float[] dsum16 = new float[0];
		int cols;
		/** The row count rounded up to a multiple of four; rows past the real ones are all zero. */
		int padded;
		boolean inUse;

		void quantize(float[][] X, int cols) {
			int batch = X.length;
			int rows = (batch + 3) & ~3;
			int n = rows * cols;
			if (q.length < n) {
				q = new short[n];
				d = new float[n / 32];
				sum32 = new int[n / 32];
				sum16 = new int[n / 16];
				dsum32 = new float[n / 32];
				dsum16 = new float[n / 16];
			}
			this.cols = cols;
			this.padded = rows;
			if (rows > batch) {
				// reused buffers: the padding rows are reset to zero activations; their outputs are discarded, and
				// zeroing keeps that work the same on every call
				int from = batch * cols;
				java.util.Arrays.fill(q, from, n, (short) 0);
				java.util.Arrays.fill(d, from >> 5, n >> 5, 0f);
				java.util.Arrays.fill(sum32, from >> 5, n >> 5, 0);
				java.util.Arrays.fill(dsum32, from >> 5, n >> 5, 0f);
				java.util.Arrays.fill(sum16, from >> 4, n >> 4, 0);
				java.util.Arrays.fill(dsum16, from >> 4, n >> 4, 0f);
			}
			final short[] qq = q;
			final float[] dd = d;
			final int[] s32 = sum32;
			final int[] s16 = sum16;
			final float[] ds32 = dsum32;
			final float[] ds16 = dsum16;
			CpuThreads.forEach(batch, b -> {
				float[] x = X[b];
				int base = b * cols;
				for (int o = 0; o < cols; o += 32) {
					float amax = 0f;
					for (int i = 0; i < 32; i++)
						amax = Math.max(amax, Math.abs(x[o + i]));
					float scale = amax / 127f;
					float inv = scale == 0f ? 0f : 1f / scale;
					int lo = 0;
					int hi = 0;
					for (int i = 0; i < 16; i++) {
						int v = Math.round(x[o + i] * inv);
						qq[base + o + i] = (short) v;
						lo += v;
					}
					for (int i = 16; i < 32; i++) {
						int v = Math.round(x[o + i] * inv);
						qq[base + o + i] = (short) v;
						hi += v;
					}
					int blk = (base + o) >> 5;
					dd[blk] = scale;
					s16[2 * blk] = lo;
					s16[2 * blk + 1] = hi;
					s32[blk] = lo + hi;
					ds32[blk] = scale * (lo + hi);
					ds16[2 * blk] = scale * lo;
					ds16[2 * blk + 1] = scale * hi;
				}
			});
		}
	}

	// ── tile ────────────────────────────────────────────────────────────────

	/** Per-task scratch: one 256-column block of a tile's weights, its scales, and the tile's sums. */
	static final class TileScratch {
		final short[] w = new short[TILE_ROWS * 256];
		/** Per row and sub-block: the factor on the integer dot (times the activation scale). */
		final float[] scale = new float[TILE_ROWS * 16];
		/** Per row and sub-block: the factor on the activation sum (minimum or offset term). */
		final float[] offset = new float[TILE_ROWS * 16];
		float[] acc = new float[0];
	}

	/**
	 * One tile: each 256-column block of its rows is unpacked, then multiplied against the activation rows four at a
	 * time, the batch padded with zero rows to a multiple of four ({@link Activations#padded}); only the real rows
	 * are written out. Each sub-block width's four-row form is a method of its own, small enough for C2 to keep every
	 * lane operation in registers. Two kinds of code ran the Vector API's allocating generic path in a server: one
	 * method holding every row form, once every path had run, and a separate one-row form for the remainder, which
	 * runs too rarely to be compiled reliably. Padding gives every row the same, always-hot code; a padded row's
	 * outputs are discarded, and a real row's arithmetic is the same in any position of the four. No vector crosses
	 * a call.
	 */
	private static void tile(TileScratch ts, int type, byte[] raw, Activations act, int batch, int cols, int r0, int r1,
			float[][] Y, int rowStart) {
		int nr = r1 - r0;
		int pb = act.padded;
		if (ts.acc.length < TILE_ROWS * pb)
			ts.acc = new float[TILE_ROWS * pb];
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

	/**
	 * Activation rows {@code b} to {@code b + 3} (the first at offset {@code xo0}) against every row of the
	 * unpacked block, 16-wide: 16-weight sub-blocks (Q6_K, Q8_0). Each loaded weight vector serves the four activation rows.
	 */
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
				ShortVector w8 = ShortVector.fromArray(S, w, wo + e + 8);
				ShortVector s0 = w0.mul(ShortVector.fromArray(S, xq, xo0 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo0 + e + 8)));
				IntVector v0 = s0.reinterpretAsInts();
				f0 = f0.add(((FloatVector) v0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v0.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd0 + (j >> 1)]));
				ShortVector s1 = w0.mul(ShortVector.fromArray(S, xq, xo1 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo1 + e + 8)));
				IntVector v1 = s1.reinterpretAsInts();
				f1 = f1.add(((FloatVector) v1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v1.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd1 + (j >> 1)]));
				ShortVector s2 = w0.mul(ShortVector.fromArray(S, xq, xo2 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo2 + e + 8)));
				IntVector v2 = s2.reinterpretAsInts();
				f2 = f2.add(((FloatVector) v2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v2.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd2 + (j >> 1)]));
				ShortVector s3 = w0.mul(ShortVector.fromArray(S, xq, xo3 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo3 + e + 8)));
				IntVector v3 = s3.reinterpretAsInts();
				f3 = f3.add(((FloatVector) v3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v3.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd3 + (j >> 1)]));
			}
			for (int j = 0; j < 16; j += 4) {
				FloatVector o = FloatVector.fromArray(F, off, so + j);
				f0 = f0.sub(o.mul(FloatVector.fromArray(F, xds, xs0 + j)));
				f1 = f1.sub(o.mul(FloatVector.fromArray(F, xds, xs1 + j)));
				f2 = f2.sub(o.mul(FloatVector.fromArray(F, xds, xs2 + j)));
				f3 = f3.sub(o.mul(FloatVector.fromArray(F, xds, xs3 + j)));
			}
			acc[r * batch + b + 0] += (f0.lane(0) + f0.lane(1)) + (f0.lane(2) + f0.lane(3));
			acc[r * batch + b + 1] += (f1.lane(0) + f1.lane(1)) + (f1.lane(2) + f1.lane(3));
			acc[r * batch + b + 2] += (f2.lane(0) + f2.lane(1)) + (f2.lane(2) + f2.lane(3));
			acc[r * batch + b + 3] += (f3.lane(0) + f3.lane(1)) + (f3.lane(2) + f3.lane(3));
		}
	}

	/**
	 * Activation rows {@code b} to {@code b + 3} (the first at offset {@code xo0}) against every row of the
	 * unpacked block, 32-wide: 32-weight sub-blocks (Q4_K, Q5_K). Each loaded weight vector serves the four activation rows.
	 */
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
				ShortVector w8 = ShortVector.fromArray(S, w, wo + e + 8);
				ShortVector w16 = ShortVector.fromArray(S, w, wo + e + 16);
				ShortVector w24 = ShortVector.fromArray(S, w, wo + e + 24);
				ShortVector s0 = w0.mul(ShortVector.fromArray(S, xq, xo0 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo0 + e + 8)))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo0 + e + 16)))
						.add(w24.mul(ShortVector.fromArray(S, xq, xo0 + e + 24)));
				IntVector v0 = s0.reinterpretAsInts();
				f0 = f0.add(((FloatVector) v0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v0.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd0 + j]));
				ShortVector s1 = w0.mul(ShortVector.fromArray(S, xq, xo1 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo1 + e + 8)))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo1 + e + 16)))
						.add(w24.mul(ShortVector.fromArray(S, xq, xo1 + e + 24)));
				IntVector v1 = s1.reinterpretAsInts();
				f1 = f1.add(((FloatVector) v1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v1.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd1 + j]));
				ShortVector s2 = w0.mul(ShortVector.fromArray(S, xq, xo2 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo2 + e + 8)))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo2 + e + 16)))
						.add(w24.mul(ShortVector.fromArray(S, xq, xo2 + e + 24)));
				IntVector v2 = s2.reinterpretAsInts();
				f2 = f2.add(((FloatVector) v2.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v2.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd2 + j]));
				ShortVector s3 = w0.mul(ShortVector.fromArray(S, xq, xo3 + e))
						.add(w8.mul(ShortVector.fromArray(S, xq, xo3 + e + 8)))
						.add(w16.mul(ShortVector.fromArray(S, xq, xo3 + e + 16)))
						.add(w24.mul(ShortVector.fromArray(S, xq, xo3 + e + 24)));
				IntVector v3 = s3.reinterpretAsInts();
				f3 = f3.add(((FloatVector) v3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(v3.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, F, 0))
						.mul(sc[so + j] * xd[xd3 + j]));
			}
			for (int j = 0; j < 8; j += 4) {
				FloatVector o = FloatVector.fromArray(F, off, so + j);
				f0 = f0.sub(o.mul(FloatVector.fromArray(F, xds, xs0 + j)));
				f1 = f1.sub(o.mul(FloatVector.fromArray(F, xds, xs1 + j)));
				f2 = f2.sub(o.mul(FloatVector.fromArray(F, xds, xs2 + j)));
				f3 = f3.sub(o.mul(FloatVector.fromArray(F, xds, xs3 + j)));
			}
			acc[r * batch + b + 0] += (f0.lane(0) + f0.lane(1)) + (f0.lane(2) + f0.lane(3));
			acc[r * batch + b + 1] += (f1.lane(0) + f1.lane(1)) + (f1.lane(2) + f1.lane(3));
			acc[r * batch + b + 2] += (f2.lane(0) + f2.lane(1)) + (f2.lane(2) + f2.lane(3));
			acc[r * batch + b + 3] += (f3.lane(0) + f3.lane(1)) + (f3.lane(2) + f3.lane(3));
		}
	}

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	/**
	 * Unpacks one 256-column block of row {@code row} into {@code w[wo, wo+256)} as
	 * 16-bit integers, and each sub-block's factors into {@code sc} and
	 * {@code off} at {@code so}: the dot contributes {@code sc * xd * dot} and the
	 * activation sum {@code -off * xd * sum}.
	 */
	static void unpack(int type, byte[] raw, int row, int kb, int cols, short[] w, int wo, float[] sc,
			float[] off, int so) {
		switch (type) {
		case 12 -> {
			int bo = (row * (cols / 256) + kb) * 144;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 16 + g * 32;
				int e = wo + g * 64;
				for (int i = 0; i < 32; i += 16) {
					ByteVector v = ByteVector.fromArray(BY, raw, q + i);
					ByteVector lo = v.and((byte) 0x0F);
					ByteVector hi = v.lanewise(VectorOperators.LSHR, 4);
					((ShortVector) lo.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + i);
					((ShortVector) lo.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + i + 8);
					((ShortVector) hi.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 32 + i);
					((ShortVector) hi.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 40 + i);
				}
				sc[so + 2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
				sc[so + 2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
				off[so + 2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
				off[so + 2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
			}
		}
		case 13 -> {
			int bo = (row * (cols / 256) + kb) * 176;
			int scBase = bo + 4;
			int qhBase = bo + 16;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 48 + g * 32;
				int e = wo + g * 64;
				for (int i = 0; i < 32; i += 16) {
					ByteVector v = ByteVector.fromArray(BY, raw, q + i);
					ByteVector h = ByteVector.fromArray(BY, raw, qhBase + i);
					ByteVector lo = v.and((byte) 0x0F)
							.or(h.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
					ByteVector hi = v.lanewise(VectorOperators.LSHR, 4)
							.or(h.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
					((ShortVector) lo.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + i);
					((ShortVector) lo.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + i + 8);
					((ShortVector) hi.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + 32 + i);
					((ShortVector) hi.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + 40 + i);
				}
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
				int qhOff = bo + 128 + half * 32;
				int scOff = bo + 192 + half * 8;
				int e = wo + half * 128;
				for (int l = 0; l < 32; l += 16) {
					ByteVector ql = ByteVector.fromArray(BY, raw, qlOff + l);
					ByteVector ql2 = ByteVector.fromArray(BY, raw, qlOff + l + 32);
					ByteVector qh = ByteVector.fromArray(BY, raw, qhOff + l);
					ByteVector w0 = ql.and((byte) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 4).and((byte) 0x30));
					ByteVector w1 = ql2.and((byte) 0x0F).or(qh.lanewise(VectorOperators.LSHL, 2).and((byte) 0x30));
					ByteVector w2 = ql.lanewise(VectorOperators.LSHR, 4).or(qh.and((byte) 0x30));
					ByteVector w3 = ql2.lanewise(VectorOperators.LSHR, 4).or(qh.lanewise(VectorOperators.LSHR, 2).and((byte) 0x30));
					((ShortVector) w0.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + l);
					((ShortVector) w0.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + l + 8);
					((ShortVector) w1.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + l + 32);
					((ShortVector) w1.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + l + 40);
					((ShortVector) w2.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + l + 64);
					((ShortVector) w2.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + l + 72);
					((ShortVector) w3.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + l + 96);
					((ShortVector) w3.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + l + 104);
				}
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
				for (int i = 0; i < 32; i += 16) {
					ByteVector v = ByteVector.fromArray(BY, raw, o + 2 + i);
					((ShortVector) v.convertShape(VectorOperators.B2S, S, 0)).intoArray(w, e + i);
					((ShortVector) v.convertShape(VectorOperators.B2S, S, 1)).intoArray(w, e + i + 8);
				}
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
