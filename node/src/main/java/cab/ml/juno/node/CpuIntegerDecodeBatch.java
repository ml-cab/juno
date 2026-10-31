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
 * The CPU integer matmul for a decode batch, {@code Y[b] = A * X[b]} over Q4_K, Q5_K, Q6_K and Q8_0 weights, where
 * every activation row's result is, bit for bit, what {@link CpuIntegerKernels} gives that row alone at the same lane
 * width.
 *
 * <p>A request in a static or continuous decode batch is decoded with this matmul while the batch has two or more
 * active rows, and with {@link CpuIntegerKernels} once it is the only one. The integer kernels agree with
 * {@link CpuIntegerGemm} only to about 1e-5 of a row's magnitude, and each layer quantizes its input to 8 bits, so
 * that difference can cross a rounding step and reach the logits: a greedy answer at a near tie then depends on what
 * else was being served. Computing each row with the single-row kernel's own arithmetic removes the dependence.
 *
 * <p>Each activation row is quantized once per call, as the single-row kernel quantizes it. The weight rows are split
 * across the CPU worker pool; a task takes one weight row and runs the single-row kernel's row function against every
 * activation row in turn, so the weight row is read from memory once and from cache for the rest.
 */
final class CpuIntegerDecodeBatch {

	private static final ThreadLocal<Activations> SCRATCH = ThreadLocal.withInitial(Activations::new);

	/** Calls per format (Q8_0, Q4_K, Q5_K, Q6_K), so a test can tell which matmul a decode batch took. */
	private static final java.util.concurrent.atomic.LongAdder[] CALLS = {
			new java.util.concurrent.atomic.LongAdder(), new java.util.concurrent.atomic.LongAdder(),
			new java.util.concurrent.atomic.LongAdder(), new java.util.concurrent.atomic.LongAdder() };

	private CpuIntegerDecodeBatch() {
	}

	private static int formatIndex(int ggmlType) {
		return switch (ggmlType) {
		case 8 -> 0;
		case 12 -> 1;
		case 13 -> 2;
		default -> 3;
		};
	}

	/** Number of {@link #matmulInto} calls over {@code ggmlType} since the last {@link #resetCalls}. */
	static long calls(int ggmlType) {
		return CALLS[formatIndex(ggmlType)].sum();
	}

	static void resetCalls() {
		for (java.util.concurrent.atomic.LongAdder a : CALLS)
			a.reset();
	}

	/** True when this matmul implements the type: the types {@link CpuIntegerKernels} implements. */
	static boolean supports(int ggmlType) {
		return CpuIntegerKernels.supports(ggmlType);
	}

	/**
	 * {@code Y[b][r - rowStart] = dot(A[r], X[b])} for {@code r} in {@code [rowStart, rowEnd)} and every {@code b}, at
	 * the lane width the single-row kernel runs at.
	 */
	static void matmulInto(GgufReader.QuantizedTensor A, float[][] X, float[][] Y, int rowStart, int rowEnd, int cols) {
		matmulInto(A, X, Y, rowStart, rowEnd, cols, CpuIntegerWidth.WIDE);
	}

	/**
	 * {@link #matmulInto(GgufReader.QuantizedTensor, float[][], float[][], int, int, int)} matching the single-row
	 * kernel at a given lane width: {@code wide} matches {@link CpuIntegerKernels256}. For tests.
	 *
	 * @throws IllegalArgumentException if {@link #supports} is false
	 */
	static void matmulInto(GgufReader.QuantizedTensor A, float[][] X, float[][] Y, int rowStart, int rowEnd, int cols,
			boolean wide) {
		if (!supports(A.type()))
			throw new IllegalArgumentException("no integer decode-batch matmul for GGUF tensor type " + A.type());
		final int batch = X.length;
		if (batch == 0 || rowEnd <= rowStart)
			return;
		CALLS[formatIndex(A.type())].increment();
		// A pool worker waiting on this call may run another task that calls in here on the same thread; that call
		// must not overwrite this one's activations.
		Activations held = SCRATCH.get();
		final Activations act = held.inUse ? new Activations() : held;
		act.inUse = true;
		try {
			final CpuIntegerKernels.Q8Activation[] a = act.quantize(X, cols);
			final byte[] raw = A.data();
			switch (A.type()) {
			case 12 -> {
				final int blocksPerRow = cols / 256;
				final int bytesPerRow = blocksPerRow * 144;
				CpuThreads.forEach(rowEnd - rowStart, r -> {
					int off = (rowStart + r) * bytesPerRow;
					for (int b = 0; b < batch; b++) {
						CpuIntegerKernels.Q8Activation x = a[b];
						Y[b][r] = wide ? CpuIntegerKernels256.q4kRow(raw, off, blocksPerRow, x.xe, x.xo, x.d, x.sum32)
								: CpuIntegerKernels.q4kRow(raw, off, blocksPerRow, x.xe, x.xo, x.d, x.sum32);
					}
				});
			}
			case 13 -> {
				final int blocksPerRow = cols / 256;
				final int bytesPerRow = blocksPerRow * 176;
				CpuThreads.forEach(rowEnd - rowStart, r -> {
					int off = (rowStart + r) * bytesPerRow;
					for (int b = 0; b < batch; b++) {
						CpuIntegerKernels.Q8Activation x = a[b];
						Y[b][r] = wide ? CpuIntegerKernels256.q5kRow(raw, off, blocksPerRow, x.q, x.d, x.sum32)
								: CpuIntegerKernels.q5kRow(raw, off, blocksPerRow, x.q, x.d, x.sum32);
					}
				});
			}
			case 14 -> {
				final int blocksPerRow = cols / 256;
				final int bytesPerRow = blocksPerRow * 210;
				CpuThreads.forEach(rowEnd - rowStart, r -> {
					int off = (rowStart + r) * bytesPerRow;
					for (int b = 0; b < batch; b++) {
						CpuIntegerKernels.Q8Activation x = a[b];
						Y[b][r] = wide ? CpuIntegerKernels256.q6kRow(raw, off, blocksPerRow, x.q, x.d, x.sum16)
								: CpuIntegerKernels.q6kRow(raw, off, blocksPerRow, x.q, x.d, x.sum16);
					}
				});
			}
			default -> {
				final int blocksPerRow = cols / 32;
				final int bytesPerRow = blocksPerRow * 34;
				CpuThreads.forEach(rowEnd - rowStart, r -> {
					int off = (rowStart + r) * bytesPerRow;
					for (int b = 0; b < batch; b++)
						Y[b][r] = CpuIntegerKernels.q8_0Row(raw, off, blocksPerRow, a[b].q, a[b].d, null);
				});
			}
			}
		} finally {
			act.inUse = false;
		}
	}

	/** One quantized activation per batch row; grown, never shrunk, and reused across calls. */
	private static final class Activations {
		CpuIntegerKernels.Q8Activation[] rows = new CpuIntegerKernels.Q8Activation[0];
		boolean inUse;

		CpuIntegerKernels.Q8Activation[] quantize(float[][] X, int cols) {
			if (rows.length < X.length) {
				CpuIntegerKernels.Q8Activation[] grown = java.util.Arrays.copyOf(rows, X.length);
				for (int b = rows.length; b < X.length; b++)
					grown[b] = new CpuIntegerKernels.Q8Activation();
				rows = grown;
			}
			for (int b = 0; b < X.length; b++)
				rows[b].quantizeFrom(X[b], cols);
			return rows;
		}
	}
}
