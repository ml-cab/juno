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

import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.PagedKvArena;
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * {@link ForwardPassHandler#evict} drops the request's KV on every handler family, the LoRA handlers
 * {@code --lora-play} serves on included: a pass for that id afterwards does not read the evicted rows, and on the
 * continuous schedule the request's pages go back to the pool. A handler whose evict kept the KV would hold every
 * finished request's rows for the life of the process.
 */
@DisplayName("Handlers - evict releases the request's KV")
class HandlerEvictReleasesKvTest {

	private static final int PROMPT = 11;

	private final String prevSchedule = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevPage = System.getProperty(KvPageSizeOptions.ENV);

	@TempDir
	Path tmp;

	@AfterEach
	void restore() {
		if (prevSchedule == null)
			System.clearProperty(ServeScheduleOptions.ENV);
		else
			System.setProperty(ServeScheduleOptions.ENV, prevSchedule);
		if (prevPage == null)
			System.clearProperty(KvPageSizeOptions.ENV);
		else
			System.setProperty(KvPageSizeOptions.ENV, prevPage);
	}

	static Stream<Arguments> familiesAndSchedules() {
		List<Arguments> out = new ArrayList<>();
		PagedKvHandlerOracleTest.families().forEach(f -> {
			out.add(Arguments.of(f, "static"));
			out.add(Arguments.of(f, "continuous"));
		});
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, {1}")
	@MethodSource("familiesAndSchedules")
	@DisplayName("a pass after evict reads none of the evicted rows; paged pages go back to the pool")
	void evictDropsTheRows(PagedKvHandlerOracleTest.Family family, String schedule) throws IOException {
		System.setProperty(ServeScheduleOptions.ENV, schedule);
		System.setProperty(KvPageSizeOptions.ENV, "4");
		ShardContext ctx = family.ctx();
		int vocab = ctx.vocabSize();
		int[] prompt = new int[PROMPT];
		for (int i = 0; i < PROMPT; i++)
			prompt[i] = (i * 7 + 3) % vocab;
		int next = (PROMPT * 7 + 3) % vocab;

		ForwardPassHandler kept = build(family, schedule + "-kept");
		kept.forwardBatch(BatchForwardRequest.withTokens("x", prompt, 0), ctx);
		float[] withContext = kept.forward(ForwardRequest.withTokens("x", new int[] { next }, PROMPT), ctx).logits();

		ForwardPassHandler h = build(family, schedule + "-evicted");
		h.forwardBatch(BatchForwardRequest.withTokens("x", prompt, 0), ctx);
		h.evict("x");
		PagedKvArena arena = h.pagedKvArena();
		if (schedule.equals("continuous")) {
			assertThat(arena).isNotNull();
			assertThat(arena.kPool().liveBlocks() + arena.vPool().liveBlocks())
					.as(family + ": pages held after evict").isZero();
		}
		float[] afterEvict = h.forward(ForwardRequest.withTokens("x", new int[] { next }, PROMPT), ctx).logits();
		assertThat(Arrays.equals(afterEvict, withContext))
				.as(family + ", " + schedule + ": the pass after evict read the evicted prompt's rows").isFalse();
	}

	private ForwardPassHandler build(PagedKvHandlerOracleTest.Family family, String tag) throws IOException {
		Path dir = Files.createDirectories(tmp.resolve(family.name().replaceAll("[^a-z0-9]", "_") + "-" + tag));
		return family.build().apply(dir);
	}
}
