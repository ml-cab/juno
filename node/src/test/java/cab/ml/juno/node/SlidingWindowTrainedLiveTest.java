package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.Locale;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * Sliding-window attention on a model trained with a window that binds, CPU
 * backend: Phi-3-mini-4k ({@value #FILE}, its July 2024 update), whose file
 * declares {@code phi3.attention.sliding_window = 2047} on every layer of a
 * 4096-token context. {@link SlidingWindowLiveTest} holds every handler to a
 * forced window on models that declare none; this holds the Phi-3 handler to
 * the window the model was trained with, at a 3,000-token context:
 * <ul>
 * <li>the file's window as the handler reads it;</li>
 * <li>the KV oracle of {@link SlidingWindowLiveCheck} at width 2047 on a
 * 3,000-token prompt, under each schedule (dense KV and paged KV);</li>
 * <li>teacher-forced NLL over the 3,000 tokens of {@code model-quality-nll.txt}
 * with the file's window against no window. Before position 2047 the window
 * reads every key, so the two agree; past it the windowed run must read the
 * lower NLL, which is the evidence the window is applied where the model
 * expects it.</li>
 * </ul>
 * Opt-in: {@code -Djuno.test.trainedWindow=true}, with a heap of about 32 GB
 * (the oracle holds four 3,000-token sessions of host KV on a model with 32 KV
 * heads); about three hours on the reference host, most of it the NLL's 6,000
 * decode-width forwards. Skips when the file is absent.
 */
@EnabledIfSystemProperty(named = "juno.test.trainedWindow", matches = "true")
@DisplayName("Sliding-window attention - a model trained with a binding window, CPU")
class SlidingWindowTrainedLiveTest {

	static final String FILE = "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf";
	/** {@code model-quality-nll.txt} under this model's tokenizer, 3,001 ids. */
	static final String IDS = "model-quality-nll.phi3-mini-4k.ids";
	static final int WIDTH = 2047;
	static final int N = 3000;
	/** A global layer's overwritten rows (values around 50) move the logits by far more than this. */
	static final double MOVED = 1e-2;
	/** Before the window binds, both runs read the same keys through the same code. */
	static final double SAME_BEFORE_WIDTH = 1e-6;

	static Path present() {
		Path path = SlidingWindowLiveCheck.model(FILE);
		assumeTrue(path.toFile().exists(), FILE + " not present");
		return path;
	}

	static LlamaConfig config(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			return LlamaConfig.from(r);
		}
	}

	static ShardContext shard(LlamaConfig cfg) {
		return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(), cfg.numHeads());
	}

	/** Runs {@code body} with {@code schedule} set for the handler loads inside it. */
	static <T> T underSchedule(String schedule, java.util.concurrent.Callable<T> body) throws Exception {
		String prev = System.getProperty(ServeScheduleOptions.ENV);
		System.setProperty(ServeScheduleOptions.ENV, schedule);
		try {
			return body.call();
		} finally {
			if (prev == null)
				System.clearProperty(ServeScheduleOptions.ENV);
			else
				System.setProperty(ServeScheduleOptions.ENV, prev);
		}
	}

	static void assertOracle(SlidingWindowLiveCheck.Result r) {
		assertThat(r.windowedDiff()).as("rows outside a windowed layer's window are not read").isZero();
		assertThat(r.globalDiff()).as("a global layer reads them").isGreaterThan(MOVED);
		assertThat(r.windowEffect()).as("the window changes the output of a prompt longer than it")
				.isGreaterThan(MOVED);
	}

	/**
	 * NLL with the file's window ({@code load} as is) and with none ({@code load}
	 * under {@link SlidingWindow#NONE}), one handler at a time, per range; asserts
	 * agreement before the width and a gain past it.
	 */
	static void assertWindowLowersNll(String label, SlidingWindowLiveCheck.Loader load, ShardContext shard)
			throws Exception {
		int[] ids = TeacherForcedNll.readIds(IDS, N + 1);
		long t0 = System.nanoTime();
		double[] on = score(load.load(), shard, ids, "nll-window");
		long t1 = System.nanoTime();
		double[] off = score(SlidingWindowLiveCheck.loadUnder(SlidingWindow.NONE, load), shard, ids, "nll-none");
		long t2 = System.nanoTime();
		double beforeOn = TeacherForcedNll.mean(on, 0, WIDTH);
		double beforeOff = TeacherForcedNll.mean(off, 0, WIDTH);
		double pastOn = TeacherForcedNll.mean(on, WIDTH, N);
		double pastOff = TeacherForcedNll.mean(off, WIDTH, N);
		System.out.printf(Locale.ROOT,
				"TRAINED-WINDOW-NLL %s %s width=%d tokens=%d before_width: window=%.5f none=%.5f diff=%.3e | "
						+ "past_width (%d positions): window=%.5f none=%.5f ratio=%.5f ppl_window=%.4f ppl_none=%.4f | "
						+ "window_s=%.1f none_s=%.1f%n",
				label, FILE, WIDTH, N, beforeOn, beforeOff, beforeOn - beforeOff, N - WIDTH, pastOn, pastOff,
				pastOn / pastOff, Math.exp(pastOn), Math.exp(pastOff), (t1 - t0) / 1e9, (t2 - t1) / 1e9);
		int[][] bins = { { 0, 1024 }, { 1024, WIDTH }, { WIDTH, 2560 }, { 2560, N } };
		for (int[] b : bins)
			System.out.printf(Locale.ROOT, "TRAINED-WINDOW-NLL %s positions [%d, %d): window=%.5f none=%.5f%n", label,
					b[0], b[1], TeacherForcedNll.mean(on, b[0], b[1]), TeacherForcedNll.mean(off, b[0], b[1]));
		assertThat(beforeOn).as("before position %d the window reads every key", WIDTH).isCloseTo(beforeOff,
				within(SAME_BEFORE_WIDTH));
		assertThat(pastOn).as("past position %d the trained window reads lower NLL than no window", WIDTH)
				.isLessThan(pastOff);
	}

	private static double[] score(ForwardPassHandler h, ShardContext shard, int[] ids, String session) {
		try {
			return TeacherForcedNll.perPosition(h, shard, ids, session);
		} finally {
			h.releaseGpuResources();
		}
	}

	@Test
	@DisplayName("The file declares a 2047-token window on every layer")
	void declaresItsWindow() throws Exception {
		Path path = present();
		LlamaConfig cfg = config(path);
		SlidingWindow window;
		long context;
		try (GgufReader r = GgufReader.open(path)) {
			window = SlidingWindow.read(r);
			context = r.metaLong("phi3.context_length", -1);
		}
		System.out.println(FILE + ": sliding window " + window + ", context " + context);
		assertThat(window.width()).isEqualTo(WIDTH);
		for (int layer = 0; layer < cfg.numLayers(); layer++)
			assertThat(window.forLayer(layer)).as("layer %d", layer).isEqualTo(WIDTH);
		assertThat((long) N).as("the prompt is longer than the window and fits the context").isGreaterThan(WIDTH)
				.isLessThan(context);
	}

	@ParameterizedTest(name = "schedule {0}")
	@ValueSource(strings = { "static", "continuous" })
	@DisplayName("KV oracle at width 2047 on a 3,000-token prompt")
	void kvOracle(String schedule) throws Exception {
		Path path = present();
		LlamaConfig cfg = config(path);
		ShardContext shard = shard(cfg);
		SlidingWindowLiveCheck.Result r = underSchedule(schedule, () -> SlidingWindowLiveCheck.run(
				() -> Phi3TransformerHandler.load(path, shard, CpuMatVec.INSTANCE), shard, cfg.numLayers(),
				cfg.kvDim(), N, WIDTH));
		System.out.printf(Locale.ROOT, "TRAINED-WINDOW-ORACLE cpu %s: windowed %.6f global %.4f effect %.4f%n",
				schedule, r.windowedDiff(), r.globalDiff(), r.windowEffect());
		assertOracle(r);
	}

	@Test
	@DisplayName("Teacher-forced NLL over 3,000 tokens: the file's window reads lower NLL past 2047 than none")
	void windowLowersNllPastItsWidth() throws Exception {
		Path path = present();
		ShardContext shard = shard(config(path));
		assertWindowLowersNll("cpu", () -> Phi3TransformerHandler.load(path, shard, CpuMatVec.INSTANCE), shard);
	}
}
