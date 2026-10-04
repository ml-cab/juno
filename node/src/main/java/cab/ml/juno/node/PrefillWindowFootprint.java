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

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * Device bytes a prefill window of a given width costs on the
 * {@link PrefillWindowRegion}: the window's own buffers, the attention scores, and
 * the matmul backend's Q8_1 copy of the window. The upload reserve
 * ({@link DeviceScratchBudget}) keeps the narrowest window free, and the adaptive
 * prefill chunk is sized from the widest one that fits.
 *
 * <p>The window's buffers mirror {@code PrefillWindowRegion.Window}'s constructor
 * term for term ({@code PrefillReserveDeviceTest} holds the two equal). They are
 * allocated at the window's capacity, its row count rounded up to
 * {@link PrefillWindowRegion#CAPACITY_STEP}. The scores buffer is grown to the rows
 * actually used and to the context the window attends over, so it grows with rows
 * times context: for a window that starts the prompt, with the square of its width.
 * A window deeper into a longer prompt attends over more context than its own width
 * and needs more; when that does not fit, the region's attention falls back to the
 * host for that window.
 *
 * @author Yevhen Soldatov
 */
final class PrefillWindowFootprint {

	private PrefillWindowFootprint() {
	}

	/** Rows a window of {@code rows} rows is allocated for. */
	static int capacity(int rows) {
		requirePositive("rows", rows);
		int step = PrefillWindowRegion.CAPACITY_STEP;
		return (rows + step - 1) / step * step;
	}

	/** Device bytes of the window's own buffers, allocated when it is opened. */
	static long windowBytes(PrefillWindowRegion.Shape shape, boolean fusedQkv, int rows) {
		long c = capacity(rows);
		long activations = (long) shape.hidden() + 2L * shape.qDim() + 2L * shape.kvDim()
				+ (fusedQkv ? shape.qDim() + 2L * shape.kvDim() : 0L);
		long scratch = 3L * shape.hidden() + 2L * shape.inter();
		long tables = 2L * ADDRESS.byteSize() + Integer.BYTES;
		return c * ((activations + scratch) * Float.BYTES + widestInput(shape) * Short.BYTES + tables);
	}

	/** Device bytes of the attention scores for {@code rows} rows over {@code seqLen} positions. */
	static long scoresBytes(PrefillWindowRegion.Shape shape, int rows, int seqLen) {
		requirePositive("rows", rows);
		requirePositive("seqLen", seqLen);
		return (long) rows * shape.numHeads() * seqLen * Float.BYTES;
	}

	/** Device bytes of the backend's Q8_1 copy of the window at the widest matmul input. */
	static long q8Bytes(PrefillWindowRegion.Shape shape, int rows) {
		requirePositive("rows", rows);
		return KQuantGemmKernel.q8Bytes(rows, widestInput(shape));
	}

	/** Everything a window of {@code rows} rows attending over {@code seqLen} positions holds on the device. */
	static long bytes(PrefillWindowRegion.Shape shape, boolean fusedQkv, int rows, int seqLen) {
		return windowBytes(shape, fusedQkv, rows) + scoresBytes(shape, rows, seqLen) + q8Bytes(shape, rows);
	}

	/** The widest matmul input: hidden (Q/K/V, gate, up), query width (O) or FFN (down). */
	private static int widestInput(PrefillWindowRegion.Shape shape) {
		return Math.max(shape.hidden(), Math.max(shape.qDim(), shape.inter()));
	}

	private static void requirePositive(String name, int value) {
		if (value < 1)
			throw new IllegalArgumentException(name + " must be >= 1 (got " + value + ")");
	}
}
