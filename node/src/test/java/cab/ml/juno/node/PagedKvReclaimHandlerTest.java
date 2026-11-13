package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvBlockPool;
import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.PagedKvArena;
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * Reclaiming idle KV pages while a request is live changes nothing it reads: on every handler family that reads
 * host KV, a request prefilled before two other requests finish and their pages are reclaimed decodes, alone and in
 * a decode batch, to the same logits bit for bit as without the reclaim; and a request started afterwards, whose
 * pages are new storage at the released ids, reads the same logits as one started on reused pages.
 */
@DisplayName("Paged KV handlers - reclaiming idle pages under a live request")
class PagedKvReclaimHandlerTest {

	private static final int PAGE_SIZE = 4;
	private static final int PROMPT = 11;

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

	static Stream<Arguments> familiesAndTypes() {
		List<Arguments> out = new ArrayList<>();
		PagedKvHandlerOracleTest.families().forEach(f -> {
			out.add(Arguments.of(f, "f16"));
			out.add(Arguments.of(f, "q8_0"));
			out.add(Arguments.of(f, "q4_0"));
		});
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, {1}")
	@MethodSource("familiesAndTypes")
	@DisplayName("a live request and one started after the reclaim read the logits they read without it")
	void reclaimChangesNoLogits(PagedKvHandlerOracleTest.Family family, String cacheType) throws IOException {
		List<float[]> kept = run(family, cacheType, false);
		List<float[]> reclaimed = run(family, cacheType, true);
		String[] steps = { "prefill a", "prefill c (after the reclaim)", "decode a at " + PROMPT,
				"decode a at " + (PROMPT + 1) + " (page boundary)", "decode batch a", "decode batch c",
				"prefill d (a's length, other context)" };
		assertThat(kept).hasSize(steps.length);
		assertThat(reclaimed).hasSize(steps.length);
		for (int i = 0; i < steps.length; i++)
			assertThat(Arrays.equals(reclaimed.get(i), kept.get(i)))
					.as(family + ", " + cacheType + ", " + steps[i] + ": logits bit for bit").isTrue();
	}

	private List<float[]> run(PagedKvHandlerOracleTest.Family family, String cacheType, boolean reclaim)
			throws IOException {
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		System.setProperty(KvPageSizeOptions.ENV, Integer.toString(PAGE_SIZE));
		System.setProperty(CacheTypeOptions.ENV_K, cacheType);
		System.setProperty(CacheTypeOptions.ENV_V, cacheType);
		Path dir = Files.createDirectories(tmp.resolve(family.name().replaceAll("[^a-z0-9]", "_") + "-" + cacheType
				+ (reclaim ? "-reclaim" : "-kept")));
		ForwardPassHandler h = family.build().apply(dir);
		ShardContext ctx = family.ctx();
		int vocab = ctx.vocabSize();
		PagedKvArena arena = h.pagedKvArena();
		assertThat(arena).as(family + ": a continuous-schedule handler reports its page arena").isNotNull();

		List<float[]> out = new ArrayList<>();
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("a", prompt(PROMPT, 0, vocab), 0), ctx).lastLogits());
		h.forwardBatch(BatchForwardRequest.withTokens("x", prompt(23, 11, vocab), 0), ctx);
		h.forwardBatch(BatchForwardRequest.withTokens("y", prompt(9, 17, vocab), 0), ctx);
		h.evict("x");
		h.evict("y");
		KvBlockPool k = arena.kPool();
		KvBlockPool v = arena.vPool();
		int liveK = k.liveBlocks();
		if (reclaim) {
			for (KvBlockPool pool : new KvBlockPool[] { k, v }) {
				pool.reclaimIdle();
				pool.reclaimIdle();
				assertThat(pool.reclaimedPages()).as(family + ": x's and y's pages released").isPositive();
				assertThat(pool.idlePages()).isZero();
			}
			assertThat(k.liveBlocks()).as("a's pages stay live").isEqualTo(liveK);
		}
		int[] c = prompt(17, 5, vocab);
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("c", c, 0), ctx).lastLogits());
		out.add(h.forward(ForwardRequest.withTokens("a", new int[] { token(PROMPT, vocab) }, PROMPT), ctx).logits());
		out.add(h.forward(ForwardRequest.withTokens("a", new int[] { token(PROMPT + 1, vocab) }, PROMPT + 1), ctx)
				.logits());
		float[][] batch = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of("a", "c"),
				new int[] { token(PROMPT + 2, vocab), token(17, vocab) }, new int[] { PROMPT + 2, 17 }), ctx)
				.logits();
		out.addAll(Arrays.asList(batch));
		int[] d = prompt(PROMPT, 40, vocab);
		d[PROMPT - 1] = token(PROMPT - 1, vocab);
		out.add(h.forwardBatch(BatchForwardRequest.withTokens("d", d, 0), ctx).lastLogits());
		return out;
	}

	private static int token(int i, int vocab) {
		return (i * 7 + 3) % vocab;
	}

	private static int[] prompt(int n, int shift, int vocab) {
		int[] p = new int[n];
		for (int i = 0; i < n; i++)
			p[i] = token(i + shift, vocab);
		return p;
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
