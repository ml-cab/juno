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
 * Q4_K/Q5_K minimum or Q6_K offset term), then multiplies that block against
 * every activation row while it sits in L1. Each sub-block is an integer dot
 * product over 16-bit lanes, scaled once: 32 values for Q4_K and Q5_K (at most
 * four products of at most 31 x 127 per lane), 16 for Q6_K and Q8_0 (at most two
 * products of at most 63 x 127 or 128 x 127). Sums for a (row, activation row)
 * pair accumulate in a tile-local buffer and are written once, so no two tasks
 * write the same cache line of {@code Y}.
 *
 * <p>The species is fixed at 128 bits for the reason {@link CpuIntegerKernels}
 * gives. Requires {@code cols % 256 == 0}; {@link #supports} says so.
 */
final class CpuIntegerGemm {

	/** Weight rows per task. 16 rows of one 256-column block are 8 KB of 16-bit weights. */
	static final int TILE_ROWS = 16;

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

	/** All activation rows quantized per 32 values, as 16-bit values; buffers grow and are reused. */
	static final class Activations {
		short[] q = new short[0];
		float[] d = new float[0];
		int[] sum32 = new int[0];
		int[] sum16 = new int[0];
		int cols;
		boolean inUse;

		void quantize(float[][] X, int cols) {
			int batch = X.length;
			int n = batch * cols;
			if (q.length < n) {
				q = new short[n];
				d = new float[n / 32];
				sum32 = new int[n / 32];
				sum16 = new int[n / 16];
			}
			this.cols = cols;
			final short[] qq = q;
			final float[] dd = d;
			final int[] s32 = sum32;
			final int[] s16 = sum16;
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

	private static void tile(TileScratch ts, int type, byte[] raw, Activations act, int batch, int cols, int r0, int r1,
			float[][] Y, int rowStart) {
		int nr = r1 - r0;
		if (ts.acc.length < TILE_ROWS * batch)
			ts.acc = new float[TILE_ROWS * batch];
		float[] acc = ts.acc;
		java.util.Arrays.fill(acc, 0, nr * batch, 0f);
		int blocks = cols / 256;
		boolean sub16 = type == 14 || type == 8;
		int subs = sub16 ? 16 : 8;
		int[] xsum = sub16 ? act.sum16 : act.sum32;
		short[] xq = act.q;
		float[] xd = act.d;
		short[] w = ts.w;
		float[] sc = ts.scale;
		float[] off = ts.offset;
		for (int kb = 0; kb < blocks; kb++) {
			for (int r = 0; r < nr; r++)
				unpack(type, raw, r0 + r, kb, cols, w, r * 256, sc, off, r * 16);
			for (int b = 0; b < batch; b++) {
				int xo = b * cols + kb * 256;
				int x32 = xo >> 5;
				int xs = sub16 ? xo >> 4 : x32;
				for (int r = 0; r < nr; r++) {
					int wo = r * 256;
					int so = r * 16;
					FloatVector f = FloatVector.zero(F);
					float m = 0f;
					if (sub16) {
						for (int j = 0; j < 16; j++) {
							int e = j * 16;
							ShortVector s = ShortVector.fromArray(S, w, wo + e).mul(ShortVector.fromArray(S, xq, xo + e))
									.add(ShortVector.fromArray(S, w, wo + e + 8).mul(ShortVector.fromArray(S, xq, xo + e + 8)));
							float xdj = xd[x32 + (j >> 1)];
							f = f.add(widenPairs(s).mul(sc[so + j] * xdj));
							m += off[so + j] * xdj * xsum[xs + j];
						}
					} else {
						for (int j = 0; j < 8; j++) {
							int e = j * 32;
							ShortVector s = ShortVector.fromArray(S, w, wo + e).mul(ShortVector.fromArray(S, xq, xo + e))
									.add(ShortVector.fromArray(S, w, wo + e + 8).mul(ShortVector.fromArray(S, xq, xo + e + 8)))
									.add(ShortVector.fromArray(S, w, wo + e + 16).mul(ShortVector.fromArray(S, xq, xo + e + 16)))
									.add(ShortVector.fromArray(S, w, wo + e + 24).mul(ShortVector.fromArray(S, xq, xo + e + 24)));
							float xdj = xd[x32 + j];
							f = f.add(widenPairs(s).mul(sc[so + j] * xdj));
							m += off[so + j] * xdj * xsum[xs + j];
						}
					}
					acc[r * batch + b] += CpuIntegerKernels.sumLanes(f) - m;
				}
			}
		}
		for (int r = 0; r < nr; r++) {
			int yr = r0 + r - rowStart;
			for (int b = 0; b < batch; b++)
				Y[b][yr] = acc[r * batch + b];
		}
	}

	/** Sums adjacent 16-bit lanes into four sign-extended int lanes, as floats. */
	private static FloatVector widenPairs(ShortVector s) {
		IntVector v = s.reinterpretAsInts();
		IntVector lo = v.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16);
		IntVector hi = v.lanewise(VectorOperators.ASHR, 16);
		return (FloatVector) lo.add(hi).convertShape(VectorOperators.I2F, F, 0);
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
	private static void unpack(int type, byte[] raw, int row, int kb, int cols, short[] w, int wo, float[] sc,
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
				for (int i = 0; i < 32; i++) {
					int v = raw[q + i];
					w[e + i] = (short) (v & 0x0F);
					w[e + 32 + i] = (short) ((v >> 4) & 0x0F);
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
				for (int i = 0; i < 32; i++) {
					int v = raw[q + i];
					int h = raw[qhBase + i];
					w[e + i] = (short) ((v & 0x0F) | (((h >> (2 * g)) & 1) << 4));
					w[e + 32 + i] = (short) (((v >> 4) & 0x0F) | (((h >> (2 * g + 1)) & 1) << 4));
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
				for (int l = 0; l < 32; l++) {
					int ql = raw[qlOff + l] & 0xFF;
					int ql2 = raw[qlOff + l + 32] & 0xFF;
					int qh = raw[qhOff + l] & 0xFF;
					w[e + l] = (short) ((ql & 0x0F) | ((qh & 3) << 4));
					w[e + l + 32] = (short) ((ql2 & 0x0F) | (((qh >> 2) & 3) << 4));
					w[e + l + 64] = (short) ((ql >> 4) | (((qh >> 4) & 3) << 4));
					w[e + l + 96] = (short) ((ql2 >> 4) | (((qh >> 6) & 3) << 4));
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
				for (int i = 0; i < 32; i++)
					w[e + i] = raw[o + 2 + i];
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
