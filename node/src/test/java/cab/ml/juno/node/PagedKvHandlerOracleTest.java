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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * The handler-level oracle for a block-table attention read: on every handler that
 * attends over host KV, the continuous schedule's paged KV gives the static schedule's
 * dense-KV logits bit for bit, at each cache type. Today the paged side gathers its
 * pages before every attention call; a handler that reads the block table instead
 * must still meet this test unchanged.
 *
 * <p>The sequence crosses page boundaries on every attention call site a handler has
 * on the CPU: a prefill window over several pages with a partly filled last page; a
 * second prompt prefilled in two chunks, so the second chunk's rows attend over pages
 * the first chunk wrote; single-token decode across a page boundary; and a decode
 * batch of three requests at different lengths. Pages are {@value #PAGE_SIZE} tokens and
 * every request shares the handler's K and V pools with the others.
 *
 * <p>The families include the three LoRA handlers the loader returns when adapters are
 * given (--lora-play), since they attend over host KV through call sites of their own.
 *
 * <p>Every model has random weights ({@link RandomWeightGgufs}, or the LLaMA test
 * instance), and the test first checks that its logits depend on the context: a
 * zero-weight model gives the same logits whatever attention reads, and would pass
 * against any defect.
 */
@DisplayName("Paged KV handlers - static dense against continuous paged, bit for bit")
class PagedKvHandlerOracleTest {

	private static final int PAGE_SIZE = 4;
	private static final int PROMPT = 11;
	private static final int SHORT_PROMPT = 5;
	private static final int CHUNK = 6;
	private static final int VOCAB = RandomWeightGgufs.VOCAB;
	private static final int H = RandomWeightGgufs.H;
	private static final int HEADS = RandomWeightGgufs.HEADS;

	private final String prevSchedule = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevPage = System.getProperty(KvPageSizeOptions.ENV);
	private final String prevTypeK = System.getProperty(CacheTypeOptions.ENV_K);
	private final String prevTypeV = System.getProperty(CacheTypeOptions.ENV_V);

	@TempDir
	Path tmp;

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSchedule);
		restoreProp(KvPageSizeOptions.ENV, prevPage);
		restoreProp(CacheTypeOptions.ENV_K, prevTypeK);
		restoreProp(CacheTypeOptions.ENV_V, prevTypeV);
	}

	/** A handler family: how to build one from a scratch folder, and its shard context. */
	record Family(String name, Function<Path, ForwardPassHandler> build, ShardContext ctx) {
		@Override
		public String toString() {
			return name;
		}
	}

	private static ShardContext ctx(int layers, int vocab, int hidden, int heads) {
		return new ShardContext("n0", 0, layers, true, true, vocab, hidden, heads);
	}

	private static Function<Path, ForwardPassHandler> fromGguf(IoFunction<Path, Path> writer, int layers) {
		return dir -> {
			try {
				return ForwardPassHandlerLoader.load(writer.apply(dir), ctx(layers, VOCAB, H, HEADS));
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
		};
	}

	/**
	 * Loads through the loader with an adapter set, the way --lora-play loads a model; the loader refuses an
	 * empty set. Rank-2 adapters on Q and V, initialized as training starts them.
	 */
	private static Function<Path, ForwardPassHandler> withAdapters(IoFunction<Path, Path> writer, int layers,
			IoFunction<GgufReader, LoraModelLayout> layout) {
		return dir -> {
			try {
				Path gguf = writer.apply(dir);
				LoraAdapterSet adapters;
				try (GgufReader r = GgufReader.open(gguf)) {
					adapters = LoraInitializer.create(layout.apply(r), LoraProjection.qv(),
							LoraAdapterConfig.legacy(2, 2f), new Random(2));
				}
				return ForwardPassHandlerLoader.load(gguf, ctx(layers, VOCAB, H, HEADS), CpuMatVec.INSTANCE, adapters);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
		};
	}

	@FunctionalInterface
	interface IoFunction<A, B> {
		B apply(A a) throws IOException;
	}

	static Stream<Family> families() {
		boolean[] allWindowed = new boolean[4];
		Arrays.fill(allWindowed, true);
		return Stream.of(
				new Family("llama", dir -> LlamaTransformerHandler.newTestInstance(64, 32, 4, 2, 4, 0, 4, true, true,
						null, 2.0f), ctx(4, 64, 32, 4)),
				// A window of 5 over 4-token pages starts mid-page, and moves a page boundary at a time.
				new Family("llama, window 5", dir -> LlamaTransformerHandler.newTestInstance(64, 32, 4, 2, 4, 0, 4,
						true, true, null, 2.0f, SlidingWindow.of(5, allWindowed)), ctx(4, 64, 32, 4)),
				new Family("phi3", fromGguf(dir -> RandomWeightGgufs.phi3(dir, 2), 2), ctx(2, VOCAB, H, HEADS)),
				new Family("phi2", fromGguf(dir -> RandomWeightGgufs.phi2(dir, 2), 2), ctx(2, VOCAB, H, HEADS)),
				new Family("qwen3", fromGguf(dir -> Qwen3LoraTrainableHandlerTest.buildQwen3F32Gguf(dir, H, HEADS, 4,
						H / HEADS, RandomWeightGgufs.I, VOCAB, 2), 2), ctx(2, VOCAB, H, HEADS)),
				new Family("qwen3moe", fromGguf(dir -> RandomWeightGgufs.qwen3moe(dir, 2), 2),
						ctx(2, VOCAB, H, HEADS)),
				// With adapters the loader returns the LoRA handlers --lora-play serves on, each with its own attention.
				new Family("llama, LoRA handler", withAdapters(dir -> RandomWeightGgufs.llama(dir, 2), 2,
						r -> LoraModelLayout.llama(LlamaConfig.from(r))),
						ctx(2, VOCAB, H, HEADS)),
				new Family("phi3, LoRA handler", withAdapters(dir -> RandomWeightGgufs.phi3(dir, 2), 2,
						r -> LoraModelLayout.phi3(LlamaConfig.from(r))),
						ctx(2, VOCAB, H, HEADS)),
				new Family("qwen3, LoRA handler", withAdapters(dir -> Qwen3LoraTrainableHandlerTest
						.buildQwen3F32Gguf(dir, H, HEADS, 4, H / HEADS, RandomWeightGgufs.I, VOCAB, 2), 2,
						r -> LoraModelLayout.qwen3(Qwen3Config.from(r))), ctx(2, VOCAB, H, HEADS)));
	}

	static Stream<Arguments> familiesAndTypes() {
		List<Arguments> out = new ArrayList<>();
		families().forEach(f -> {
			out.add(Arguments.of(f, "f16"));
			out.add(Arguments.of(f, "q8_0"));
		});
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, {1}")
	@MethodSource("familiesAndTypes")
	@DisplayName("prefill, chunked prefill, decode across a page boundary and a decode batch read the same logits")
	void pagedLogitsEqualDenseLogits(Family family, String cacheType) throws IOException {
		List<float[]> dense = run(family, cacheType, false);
		List<float[]> paged = run(family, cacheType, true);
		String[] steps = { "prefill a (" + PROMPT + " tokens)", "prefill b (" + SHORT_PROMPT + " tokens)",
				"chunked prefill c, first chunk", "chunked prefill c, second chunk", "decode a at " + PROMPT,
				"decode a at " + (PROMPT + 1) + " (page boundary)", "decode batch a", "decode batch b",
				"decode batch c", "prefill d (a's length and last token, other context)" };
		if (dense.size() != steps.length || paged.size() != steps.length)
			throw new AssertionError("expected " + steps.length + " readings, got " + dense.size() + " dense and "
					+ paged.size() + " paged");
		// The oracle is only as good as its model: logits must depend on the context attention reads.
		double contextEffect = maxAbsDiff(dense.get(0), dense.get(steps.length - 1));
		if (!(contextEffect > 1e-4))
			throw new AssertionError(family + ": prompts a and d differ before their shared last token, yet their"
					+ " logits differ by only " + contextEffect + "; attention does not reach the logits");
		for (int i = 0; i < steps.length; i++)
			assertBitsEqual(paged.get(i), dense.get(i), family + ", " + cacheType + ", " + steps[i]);
	}

	/** Runs the sequence on a fresh handler under one schedule and returns every step's logits. */
	private List<float[]> run(Family family, String cacheType, boolean paged) throws IOException {
		if (paged) {
			System.setProperty(ServeScheduleOptions.ENV, "continuous");
			System.setProperty(KvPageSizeOptions.ENV, Integer.toString(PAGE_SIZE));
		} else {
			System.setProperty(ServeScheduleOptions.ENV, "static");
			System.clearProperty(KvPageSizeOptions.ENV);
		}
		System.setProperty(CacheTypeOptions.ENV_K, cacheType);
		System.setProperty(CacheTypeOptions.ENV_V, cacheType);
		Path dir = Files.createDirectories(tmp.resolve(family.name().replaceAll("[^a-z0-9]", "_")
				+ (paged ? "-paged" : "-dense") + "-" + cacheType));
		ForwardPassHandler h = family.build().apply(dir);
		ShardContext ctx = family.ctx();
		int vocab = ctx.vocabSize();

		List<float[]> out = new ArrayList<>();
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("a", prompt(PROMPT, 0, vocab), 0), ctx).lastLogits());
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("b", prompt(SHORT_PROMPT, 3, vocab), 0), ctx)
				.lastLogits());
		int[] c = prompt(PROMPT, 5, vocab);
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("c", Arrays.copyOfRange(c, 0, CHUNK), 0), ctx)
				.lastLogits());
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("c", Arrays.copyOfRange(c, CHUNK, PROMPT), CHUNK), ctx)
				.lastLogits());
		out.add(h.forward(ForwardRequest.withTokens("a", new int[] { token(PROMPT, vocab) }, PROMPT), ctx).logits());
		out.add(h.forward(ForwardRequest.withTokens("a", new int[] { token(PROMPT + 1, vocab) }, PROMPT + 1), ctx)
				.logits());
		float[][] batch = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of("a", "b", "c"),
				new int[] { token(PROMPT + 2, vocab), token(SHORT_PROMPT, vocab), token(PROMPT + 7, vocab) },
				new int[] { PROMPT + 2, SHORT_PROMPT, PROMPT }), ctx).logits();
		out.addAll(Arrays.asList(batch));
		int[] d = prompt(PROMPT, 40, vocab);
		d[PROMPT - 1] = token(PROMPT - 1, vocab);
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("d", d, 0), ctx).lastLogits());
		return out;
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	private static int token(int i, int vocab) {
		return (i * 7 + 3) % vocab;
	}

	private static int[] prompt(int n, int shift, int vocab) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = token(i + shift, vocab);
		return t;
	}

	private static void assertBitsEqual(float[] actual, float[] expected, String what) {
		if (actual == null || expected == null)
			throw new AssertionError(what + ": no logits (" + (actual == null ? "paged" : "dense") + ")");
		if (actual.length != expected.length)
			throw new AssertionError(what + ": " + actual.length + " logits, dense has " + expected.length);
		for (int i = 0; i < actual.length; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": logit " + i + " is " + actual[i] + ", dense reads " + expected[i]);
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
