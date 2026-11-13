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

/**
 * How a GPU prefill window's attention and its tiled K-quant matmuls split into launches of bounded work, so that
 * no single kernel runs long enough for a display driver's watchdog to reset the device (two seconds by default on
 * Windows, for a GPU that drives a display; the reset ends the CUDA context, and the request with it).
 *
 * <p>A window's attention grows with the square of its rows and its matmuls with its rows, and the window is the
 * prompt, up to the device KV's {@link DeviceKvCache#MAX_SEQ_LEN} positions. The split keeps each launch's work
 * under a fixed budget at any length:
 * <ul>
 * <li><b>Attention</b> splits between whole blocks of query rows ({@link GqaAttentionKernel#rowsPerBlock}), and
 * every launch keeps the block geometry the whole window would have, so each block computes on the same rows in the
 * same order and the output is bit for bit one launch's. A launch's work is the keys its rows attend (a sliding
 * window counts only its own) times the heads times the head width the kernel holds in registers.
 * <li><b>The tiled K-quant GEMM</b> splits into launches of whole 64-row column tiles; activation rows are
 * independent, so the output is bit for bit one launch's. A launch's work is its rows times the weight matrix's
 * rows times its columns.
 * </ul>
 *
 * <p>The budgets are sized from measured launch times, which follow the work: scaled from its 2048-token readings,
 * the slowest GPU a kit run has measured (a GeForce MX150) runs the heaviest launch they allow in under 400 ms,
 * against a watchdog of 2 s. A window of up to 2048 rows on any standing sweep model is one launch of each, as
 * before the split.
 */
final class BoundedLaunches {

	/** Attention work per launch: keys attended, times heads, times the register head width. */
	static final long ATTENTION_WORK_PER_LAUNCH = 12_000_000_000L;
	/** Matmul work per launch: activation rows, times weight rows, times weight columns. */
	static final long GEMM_WORK_PER_LAUNCH = 200_000_000_000L;
	/** The GEMM splits at multiples of its widest column tile, so every launch but the last fills its tiles. */
	static final int GEMM_ROW_STEP = 64;

	private BoundedLaunches() {
	}

	/** Attention work one key costs one query row: the heads times the head width the kernel holds in registers. */
	static long attentionWorkPerKey(int numHeads, int headDim) {
		return (long) numHeads * GqaAttentionKernel.registerWidth(headDim);
	}

	/**
	 * The end (exclusive) of the launch that starts at row {@code start}: as many whole blocks of
	 * {@code rowsPerBlock} rows as fit in {@code budget}, and always at least one. Row {@code b} attends
	 * {@code seqLens[b]} keys, or its last {@code window} ({@code 0}: no window). {@code start} is 0 or an end
	 * this method returned.
	 */
	static int attentionEnd(int start, int[] seqLens, int batch, int rowsPerBlock, int window, long workPerKey,
			long budget) {
		long work = 0;
		int end = start;
		while (end < batch) {
			int blockEnd = Math.min(end + rowsPerBlock, batch);
			long block = attentionWork(seqLens, end, blockEnd, window, workPerKey);
			if (end > start && work + block > budget)
				break;
			work += block;
			end = blockEnd;
		}
		return end;
	}

	/** Attention work of rows {@code [from, to)}. */
	static long attentionWork(int[] seqLens, int from, int to, int window, long workPerKey) {
		long keys = 0;
		for (int b = from; b < to; b++) {
			int len = seqLens[b];
			keys += window > 0 && len > window ? window : len;
		}
		return keys * workPerKey;
	}

	/**
	 * Activation rows per GEMM launch over a {@code weightRows x weightCols} matrix: the most whole
	 * {@link #GEMM_ROW_STEP}-row tiles whose work fits in {@code budget}, and always at least one tile.
	 */
	static int gemmRows(int weightRows, int weightCols, long budget) {
		long perRow = (long) weightRows * weightCols;
		long tiles = budget / (perRow * GEMM_ROW_STEP);
		return (int) Math.max(1, Math.min(tiles, Integer.MAX_VALUE / GEMM_ROW_STEP)) * GEMM_ROW_STEP;
	}
}
