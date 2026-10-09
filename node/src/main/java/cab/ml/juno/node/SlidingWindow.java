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

import java.util.Arrays;

/**
 * Which layers attend within a sliding window, and how wide it is, as a model
 * file declares it. A windowed layer's row at context length {@code seqLen}
 * attends over keys {@code [max(0, seqLen - width), seqLen)}; a global layer over
 * all of them. Every handler reads this once at load and gives each of its
 * layers {@link #forShard} of it, which the CPU attention ({@link GqaMath}) and
 * the GPU kernel ({@link GqaAttentionKernel}) take as a per-call window, 0
 * meaning none.
 *
 * <p>Two keys, read as exporters write them:
 * <ul>
 * <li>{@code <arch>.attention.sliding_window}: the width. Absent or 0 means no
 * window on any layer, whatever the pattern says.</li>
 * <li>{@code <arch>.attention.sliding_window_pattern}: which layers. Absent means
 * every layer is windowed (the uniform case). A boolean array, one entry per
 * layer ({@code true} = windowed), must have {@code <arch>.block_count} entries.
 * An integer period {@code N} makes the last layer of each period global
 * ({@code layer % N == N - 1}), so {@code N = 1} is every layer global and
 * {@code N = 0} every layer windowed.</li>
 * </ul>
 * Anything else (another value type, an array of the wrong length, a negative
 * width or period) is refused at load: guessing which layers a file meant to
 * window would produce wrong output that still reads fluently.
 *
 * <p>A window no shorter than a row's context reads exactly the keys of no
 * window, so a file whose window exceeds every context it can reach (Phi-3.5's
 * 262144 against its 4096-token cap) computes what it computed before.
 *
 * <p>The cache keeps every row; the window narrows what attention reads, not
 * what is stored.
 */
final class SlidingWindow {

	/** No window on any layer. */
	static final SlidingWindow NONE = new SlidingWindow(0, null);

	/**
	 * Replaces what {@link #read} returns, so a test can run a real model file under
	 * a window it does not declare. Tests only; {@code null} otherwise.
	 */
	static volatile SlidingWindow testOverride;

	private final int width;
	/** {@code windowed[layer]}; {@code null} with a width means every layer. */
	private final boolean[] windowed;

	private SlidingWindow(int width, boolean[] windowed) {
		this.width = width;
		this.windowed = windowed;
	}

	/**
	 * A window of {@code width} on the layers {@code windowed} marks (indexed by
	 * global layer), or on every layer when {@code windowed} is {@code null}.
	 */
	static SlidingWindow of(int width, boolean[] windowed) {
		if (width < 0)
			throw new IllegalArgumentException("width must be >= 0 (got " + width + ")");
		return width == 0 ? NONE : new SlidingWindow(width, windowed == null ? null : windowed.clone());
	}

	/**
	 * The window the file declares, {@link #NONE} when it declares none.
	 *
	 * @throws UnsupportedModelException naming the key when a window key is malformed
	 */
	static SlidingWindow read(GgufReader r) throws UnsupportedModelException {
		SlidingWindow override = testOverride;
		if (override != null)
			return override;
		String arch = r.metaString("general.architecture");
		if (arch == null)
			return NONE;
		String widthKey = arch + ".attention.sliding_window";
		String patternKey = arch + ".attention.sliding_window_pattern";
		Object w = r.meta(widthKey);
		if (w == null)
			return NONE;
		if (!(w instanceof Integer || w instanceof Long))
			throw new UnsupportedModelException(widthKey + " must be an integer (got " + describe(w) + ")");
		long width = ((Number) w).longValue();
		if (width < 0 || width > Integer.MAX_VALUE)
			throw new UnsupportedModelException(widthKey + " must be a non-negative int (got " + width + ")");
		if (width == 0)
			return NONE;
		Object pattern = r.meta(patternKey);
		if (pattern == null)
			return new SlidingWindow((int) width, null);
		return new SlidingWindow((int) width, layers(r, arch, patternKey, pattern));
	}

	/**
	 * The file's window for the layers {@code shard} holds ({@link #forShard}),
	 * logged once when it declares one: what a handler reads at load.
	 *
	 * @throws UnsupportedModelException naming the key when a window key is malformed
	 */
	static int[] forShard(GgufReader r, ShardContext shard, java.util.logging.Logger log)
			throws UnsupportedModelException {
		SlidingWindow window = read(r);
		if (window.declared())
			log.info("Sliding-window attention: " + window);
		return window.forShard(shard.startLayer(), shard.endLayer() - shard.startLayer());
	}

	private static boolean[] layers(GgufReader r, String arch, String patternKey, Object pattern)
			throws UnsupportedModelException {
		String countKey = arch + ".block_count";
		long count = r.metaLong(countKey, -1);
		if (count < 1)
			throw new UnsupportedModelException(patternKey + " needs " + countKey + " to place it, and the file has none");
		boolean[] windowed = new boolean[(int) count];
		if (pattern instanceof Object[] arr) {
			if (arr.length != count)
				throw new UnsupportedModelException(patternKey + " has " + arr.length + " entries for " + count
						+ " layers (" + countKey + ")");
			for (int i = 0; i < arr.length; i++) {
				if (!(arr[i] instanceof Boolean b))
					throw new UnsupportedModelException(patternKey + " must be an array of booleans (entry " + i
							+ " is " + describe(arr[i]) + ")");
				windowed[i] = b;
			}
			return windowed;
		}
		if (pattern instanceof Integer || pattern instanceof Long) {
			long period = ((Number) pattern).longValue();
			if (period < 0)
				throw new UnsupportedModelException(patternKey + " must be >= 0 (got " + period + ")");
			for (int i = 0; i < count; i++)
				windowed[i] = period == 0 || i % period != period - 1;
			return windowed;
		}
		throw new UnsupportedModelException(patternKey + " must be a boolean array or an integer period (got "
				+ describe(pattern) + ")");
	}

	private static String describe(Object v) {
		return v instanceof Object[] a ? "an array of " + a.length : v.getClass().getSimpleName() + " " + v;
	}

	/** Whether any layer is windowed. */
	boolean declared() {
		return width > 0;
	}

	/** The window width, 0 when there is none. */
	int width() {
		return width;
	}

	/** The window of global layer {@code globalLayer}: {@link #width()} when it is windowed, else 0. */
	int forLayer(int globalLayer) {
		if (width == 0)
			return 0;
		if (windowed == null)
			return width;
		if (globalLayer < 0 || globalLayer >= windowed.length)
			throw new IllegalArgumentException("layer " + globalLayer + " outside the pattern's " + windowed.length
					+ " layers");
		return windowed[globalLayer] ? width : 0;
	}

	/**
	 * Each local layer's window for a shard holding global layers
	 * {@code [startLayer, startLayer + layers)}: the pattern is read at the global
	 * index, so a pipeline shard windows the layers the whole model windows.
	 */
	int[] forShard(int startLayer, int layers) {
		int[] out = new int[layers];
		for (int li = 0; li < layers; li++)
			out[li] = forLayer(startLayer + li);
		return out;
	}

	@Override
	public String toString() {
		if (width == 0)
			return "none";
		if (windowed == null)
			return width + " (every layer)";
		int global = 0;
		for (boolean b : windowed)
			if (!b)
				global++;
		return width + " (" + (windowed.length - global) + " of " + windowed.length + " layers; global: "
				+ Arrays.toString(globalIndices()) + ")";
	}

	private int[] globalIndices() {
		return java.util.stream.IntStream.range(0, windowed.length).filter(i -> !windowed[i]).toArray();
	}
}
