package cab.ml.juno.node;

import java.nio.file.Path;
import java.util.Random;

import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Drives one real handler, loaded under a forced patterned window
 * ({@link SlidingWindow#testOverride}: period 2, so even layers are windowed and
 * odd layers global), through the KV oracle: after a prompt of {@code n} tokens,
 * the rows the next token's window excludes are overwritten in one layer, and the
 * next logits are compared with an untouched request. A windowed layer must not
 * move them at all; a global layer must.
 */
final class SlidingWindowLiveCheck {

	/** {@code windowedDiff} must be 0; {@code globalDiff} must be large; {@code windowEffect}: windowed vs none. */
	record Result(double windowedDiff, double globalDiff, double windowEffect) {
	}

	private SlidingWindowLiveCheck() {
	}

	interface Loader {
		ForwardPassHandler load() throws Exception;
	}

	/** Period-2 pattern over {@code layers} layers: even layers windowed. */
	static SlidingWindow periodTwo(int width, int layers) {
		boolean[] windowed = new boolean[layers];
		for (int i = 0; i < layers; i++)
			windowed[i] = i % 2 == 0;
		return SlidingWindow.of(width, windowed);
	}

	/** Loads with {@code window} forced for the duration of the load only. */
	static ForwardPassHandler loadUnder(SlidingWindow window, Loader loader) throws Exception {
		SlidingWindow.testOverride = window;
		try {
			return loader.load();
		} finally {
			SlidingWindow.testOverride = null;
		}
	}

	static Result run(Loader loader, ShardContext shard, int layers, int kvDim, int n, int width) throws Exception {
		ForwardPassHandler windowed = loadUnder(periodTwo(width, layers), loader);
		int[] prompt = new int[n];
		for (int i = 0; i < n; i++)
			prompt[i] = ContextShiftLiveCheck.token(i);
		int next = ContextShiftLiveCheck.token(n);
		int outside = n + 1 - width;
		try {
			for (String id : new String[] { "clean", "w", "g" }) {
				windowed.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard);
				ContextShiftLiveCheck.retireDeviceKv(windowed, id);
			}
			overwrite(ContextShiftLiveCheck.hostKv(windowed, "w"), 0, outside, kvDim);
			overwrite(ContextShiftLiveCheck.hostKv(windowed, "g"), 1, outside, kvDim);
			float[] clean = windowed.forward(ForwardRequest.withTokens("clean", new int[] { next }, n), shard).logits();
			float[] w = windowed.forward(ForwardRequest.withTokens("w", new int[] { next }, n), shard).logits();
			float[] g = windowed.forward(ForwardRequest.withTokens("g", new int[] { next }, n), shard).logits();
			float[] none = unwindowed(loader, shard, prompt, next, n);
			Result r = new Result(maxAbsDiff(w, clean), maxAbsDiff(g, clean), maxAbsDiff(clean, none));
			System.out.printf("%s: windowed-layer overwrite %.6f, global-layer overwrite %.4f, window vs none %.4f%n",
					windowed.getClass().getSimpleName(), r.windowedDiff(), r.globalDiff(), r.windowEffect());
			return r;
		} finally {
			for (String id : new String[] { "clean", "w", "g" })
				windowed.evict(id);
			windowed.releaseGpuResources();
		}
	}

	private static float[] unwindowed(Loader loader, ShardContext shard, int[] prompt, int next, int n)
			throws Exception {
		ForwardPassHandler h = loadUnder(SlidingWindow.NONE, loader);
		try {
			h.forwardBatch(BatchForwardRequest.withTokens("none", prompt, 0), shard);
			return h.forward(ForwardRequest.withTokens("none", new int[] { next }, n), shard).logits();
		} finally {
			h.evict("none");
			h.releaseGpuResources();
		}
	}

	private static void overwrite(SessionKvTensor[][] kv, int layer, int rows, int kvDim) {
		Random rng = new Random(layer * 31L + rows);
		float[] row = new float[kvDim];
		for (int pos = 0; pos < rows; pos++)
			for (int which = 0; which < 2; which++) {
				for (int i = 0; i < kvDim; i++)
					row[i] = (float) rng.nextGaussian() * 50f;
				kv[which][layer].writeToken(pos, row);
			}
	}

	static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	static Path model(String file) {
		return ContextShiftLiveTest.model(file);
	}
}
