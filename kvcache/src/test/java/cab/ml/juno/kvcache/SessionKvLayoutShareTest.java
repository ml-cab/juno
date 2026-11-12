package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * A request started from another request's prefix ({@link SessionKvLayout#sharePrefixK}) reads exactly the source's
 * rows there, and neither tensor ever sees the other's later writes: on the paged layout because a page held by two
 * tables is copied before either writes it, on the dense layout because the rows are copied.
 */
class SessionKvLayoutShareTest {

	private static final int KV_DIM = 64;
	private static final int PAGE = 16;
	private static final int LAYERS = 3;

	private static SessionKvLayout layout(String schedule, String type) {
		KvElementType t = KvElementType.parse(type);
		ServeScheduleOptions s = ServeScheduleOptions.parse(schedule);
		PagedKvArena arena = s.usesPagedKv() ? new PagedKvArena(PAGE, KV_DIM, t, t) : null;
		return SessionKvLayout.of(s, KvPageSizeOptions.of(PAGE), CacheTypeOptions.of(t, t), KV_DIM, arena);
	}

	private static float[] row(Random rnd) {
		float[] r = new float[KV_DIM];
		for (int i = 0; i < KV_DIM; i++)
			r[i] = (float) rnd.nextGaussian();
		return r;
	}

	private static SessionKvTensor[] written(SessionKvLayout layout, int rows, long seed) {
		SessionKvTensor[] k = layout.newKLayers(LAYERS);
		Random rnd = new Random(seed);
		for (SessionKvTensor t : k)
			for (int p = 0; p < rows; p++)
				t.writeToken(p, row(rnd));
		return k;
	}

	@ParameterizedTest(name = "{0} {1}, {2} of {3} rows")
	@CsvSource({ "static, f16, 40, 50", "static, q8_0, 40, 50", "continuous, f16, 40, 50",
			"continuous, q8_0, 40, 50", "continuous, f16, 32, 50", "continuous, q8_0, 1, 50",
			"static, f16, 50, 50", "continuous, f16, 50, 50" })
	void sharedPrefixReadsTheSourceRowsAndWritesStayApart(String schedule, String type, int len, int rows) {
		SessionKvLayout layout = layout(schedule, type);
		SessionKvTensor[] src = written(layout, rows, 7);
		float[][] before = new float[LAYERS][];
		for (int i = 0; i < LAYERS; i++)
			before[i] = src[i].toFloatArray(rows);

		SessionKvTensor[] shared = layout.sharePrefixK(src, len);

		assertThat(shared).hasSize(LAYERS);
		Random rnd = new Random(11);
		for (int i = 0; i < LAYERS; i++) {
			assertThat(shared[i].toFloatArray(len)).containsExactly(java.util.Arrays.copyOf(before[i], len * KV_DIM));
			// Each side writes where the other holds rows: the new request past the prefix (into a shared partial
			// page on the paged layout), the source inside it.
			float[] mine = row(rnd);
			float[] theirs = row(rnd);
			shared[i].writeToken(len, mine);
			src[i].writeToken(len - 1, theirs);
			float[] read = new float[KV_DIM];
			shared[i].readToken(len - 1, read);
			assertThat(read).containsExactly(java.util.Arrays.copyOfRange(before[i], (len - 1) * KV_DIM, len * KV_DIM));
			if (len < rows) {
				src[i].readToken(len, read);
				assertThat(read).containsExactly(java.util.Arrays.copyOfRange(before[i], len * KV_DIM, (len + 1) * KV_DIM));
			}
		}
		SessionKvLayout.releaseLayers(src);
		SessionKvLayout.releaseLayers(shared);
		layout.arena().ifPresent(a -> assertThat(a.kPool().liveBlocks()).isZero());
	}

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "static", "continuous" })
	void aSourceShorterThanTheRequestSharesNothing(String schedule) {
		SessionKvLayout layout = layout(schedule, "f16");
		SessionKvTensor[] src = written(layout, 20, 3);

		assertThat(layout.sharePrefixK(src, 21)).isNull();
		assertThat(layout.sharePrefixK(src, 0)).isNull();

		SessionKvLayout.releaseLayers(src);
		layout.arena().ifPresent(a -> assertThat(a.kPool().liveBlocks()).isZero());
	}

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "f16", "q8_0" })
	void thePagedPrefixHoldsTheSourcePagesUntilBothRelease(String type) {
		SessionKvLayout layout = layout("continuous", type);
		SessionKvTensor[] src = written(layout, 40, 5); // 3 pages per layer
		KvBlockPool pool = layout.arena().orElseThrow().kPool();
		int held = pool.liveBlocks();

		SessionKvTensor[] shared = layout.sharePrefixK(src, 33);
		assertThat(pool.liveBlocks()).isEqualTo(held);

		SessionKvLayout.releaseLayers(src);
		assertThat(pool.liveBlocks()).isEqualTo(3 * LAYERS);
		float[] read = new float[KV_DIM];
		shared[0].readToken(32, read);
		SessionKvLayout.releaseLayers(shared);
		assertThat(pool.liveBlocks()).isZero();
	}
}
