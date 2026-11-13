package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Reclaiming idle KV pages: a page every holder has dropped keeps its storage for reuse until it has stayed unused
 * for a whole reclaim period ({@link KvBlockPool#reclaimIdle}), then the storage goes back to the heap. Live and
 * shared pages are never touched, whatever their position among the idle ones.
 */
@DisplayName("KvBlockPool - reclaiming idle pages")
class KvBlockPoolReclaimTest {

	private static final int PAGE = 4;
	private static final int KV_DIM = 64;

	private static float[] row(int seed) {
		float[] r = new float[KV_DIM];
		for (int i = 0; i < KV_DIM; i++)
			r[i] = (float) Math.sin(seed * 0.37 + i * 0.11) * (1 + seed % 7);
		return r;
	}

	private static void fill(KvBlockPool pool, int id, int seed) {
		for (int s = 0; s < PAGE; s++)
			pool.writeToken(id, s, row(seed * PAGE + s));
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a page idle for a whole period loses its storage; the live pages among the idle ones keep their bytes")
	void idle_pages_are_released_after_a_whole_period(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		int[] ids = new int[12];
		for (int i = 0; i < ids.length; i++) {
			ids[i] = pool.allocate();
			fill(pool, ids[i], i);
		}
		// Fragment the pool: free scattered pages, keep the rest live in between.
		int[] freed = { 0, 2, 4, 7, 9, 11 };
		Map<Integer, byte[]> liveBytes = new HashMap<>();
		for (int i = 0; i < ids.length; i++)
			liveBytes.put(ids[i], pool.pageBytes(ids[i]));
		for (int i : freed) {
			pool.free(ids[i]);
			liveBytes.remove(ids[i]);
		}
		assertThat(pool.idlePages()).isEqualTo(6);
		assertThat(pool.storedPages()).isEqualTo(12);

		// The pages became idle during this period, so it releases nothing.
		assertThat(pool.reclaimIdle()).isZero();
		assertThat(pool.storedPages()).isEqualTo(12);

		// Unused for the whole of the next one: released.
		assertThat(pool.reclaimIdle()).isEqualTo(6);
		assertThat(pool.idlePages()).isZero();
		assertThat(pool.storedPages()).isEqualTo(6);
		assertThat(pool.liveBlocks()).isEqualTo(6);
		assertThat(pool.reclaimedPages()).isEqualTo(6);
		assertThat(pool.storedBytes()).isEqualTo(6L * PAGE * pool.bytesPerToken());
		assertThat(pool.allocatedBytes()).isEqualTo(6L * PAGE * pool.bytesPerToken());
		for (Map.Entry<Integer, byte[]> e : liveBytes.entrySet())
			assertThat(pool.pageBytes(e.getKey())).as("live page " + e.getKey()).isEqualTo(e.getValue());

		assertThat(pool.reclaimIdle()).isZero();
	}

	@Test
	@DisplayName("pages taken during the period are kept; only those unused throughout are released")
	void pages_used_during_the_period_are_kept() {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, KvElementType.F16);
		int[] ids = new int[8];
		for (int i = 0; i < ids.length; i++)
			ids[i] = pool.allocate();
		for (int id : ids)
			pool.free(id);
		assertThat(pool.reclaimIdle()).isZero();

		// A burst takes three idle pages and returns them: three were needed, five were not.
		int[] burst = { pool.allocate(), pool.allocate(), pool.allocate() };
		for (int id : burst)
			pool.free(id);
		assertThat(pool.idlePages()).isEqualTo(8);
		assertThat(pool.reclaimIdle()).isEqualTo(5);
		assertThat(pool.idlePages()).isEqualTo(3);
		assertThat(pool.storedPages()).isEqualTo(3);

		assertThat(pool.reclaimIdle()).isEqualTo(3);
		assertThat(pool.storedPages()).isZero();
		assertThat(pool.reclaimedPages()).isEqualTo(8);
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("an idle page with storage is reused first; a released id comes back with zeroed storage")
	void released_ids_are_reused_with_fresh_storage(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		int a = pool.allocate();
		int b = pool.allocate();
		fill(pool, a, 1);
		fill(pool, b, 2);
		pool.free(a);
		pool.reclaimIdle();
		pool.reclaimIdle();
		assertThat(pool.storedPages()).isEqualTo(1);

		pool.free(b);
		// b still holds its storage, a does not: b is taken first.
		assertThat(pool.allocate()).isEqualTo(b);
		int again = pool.allocate();
		assertThat(again).isEqualTo(a);
		assertThat(pool.pageBytes(again)).containsOnly((byte) 0);
		assertThat(pool.storedPages()).isEqualTo(2);
		fill(pool, again, 9);
		float[] got = new float[KV_DIM];
		float[] want = new float[KV_DIM];
		for (int s = 0; s < PAGE; s++) {
			pool.readToken(again, s, got);
			KvBlockPool ref = new KvBlockPool(PAGE, KV_DIM, type);
			int r = ref.allocate();
			ref.writeToken(r, s, row(9 * PAGE + s));
			ref.readToken(r, s, want);
			assertThat(got).as("slot " + s).containsExactly(want);
		}
		// Ids stay dense: nothing new was made past the two.
		assertThat(pool.allocate()).isEqualTo(2);
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("live and shared tables read the same rows after a reclaim, and copy-on-write still separates them")
	void live_and_shared_tables_survive_reclaim(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable a = new KvPageTable(pool);
		KvPageTable done = new KvPageTable(pool);
		for (int p = 0; p < 10; p++) {
			a.appendToken(row(p));
			done.appendToken(row(100 + p));
		}
		KvPageTable b = a.sharePrefix(6); // pages 0 and 1, the second partly
		done.release();
		float[] beforeA = new float[10 * KV_DIM];
		float[] beforeB = new float[6 * KV_DIM];
		a.gather(beforeA);
		b.gather(beforeB);

		pool.reclaimIdle();
		assertThat(pool.reclaimIdle()).isEqualTo(3);
		assertThat(pool.storedPages()).isEqualTo(3);

		float[] afterA = new float[10 * KV_DIM];
		float[] afterB = new float[6 * KV_DIM];
		a.gather(afterA);
		b.gather(afterB);
		assertThat(afterA).containsExactly(beforeA);
		assertThat(afterB).containsExactly(beforeB);
		assertViewMatchesGather(a, afterA);
		assertViewMatchesGather(b, afterB);

		// b appends into the shared, partly filled page: it takes a copy, which reuses a released id.
		b.appendToken(row(500));
		assertThat(pool.storedPages()).isEqualTo(4);
		a.gather(afterA);
		assertThat(afterA).containsExactly(beforeA);
		float[] row6 = new float[KV_DIM];
		float[] want = new float[KV_DIM];
		b.readToken(6, row6);
		KvPageTable ref = new KvPageTable(new KvBlockPool(PAGE, KV_DIM, type));
		ref.appendToken(row(500));
		ref.readToken(0, want);
		assertThat(row6).containsExactly(want);
		assertViewMatchesGather(a, afterA);
	}

	/** The attention view's pages hold, row for row, what {@link KvPageTable#gather} copies out. */
	private static void assertViewMatchesGather(KvPageTable t, float[] gathered) {
		KvRows rows = t.rows();
		for (int pos = 0; pos < t.seqLen(); pos++) {
			int page = pos / rows.rowsPerPage();
			int slot = pos % rows.rowsPerPage();
			float[] got = new float[KV_DIM];
			if (rows.type() == KvElementType.F16)
				System.arraycopy(rows.floatPage(page), slot * KV_DIM, got, 0, KV_DIM);
			else
				rows.decodeRange(rows.bytePage(page), slot * rows.bytesPerRow(), 0, KV_DIM, got, 0);
			float[] want = new float[KV_DIM];
			System.arraycopy(gathered, pos * KV_DIM, want, 0, KV_DIM);
			assertThat(got).as("view row " + pos).containsExactly(want);
		}
	}

	@Test
	@DisplayName("a reclaim larger than one lock hold releases every idle page")
	void a_large_reclaim_spans_several_lock_holds() {
		KvBlockPool pool = new KvBlockPool(1, 8, KvElementType.F16);
		int n = 3 * KvBlockPool.RECLAIM_CHUNK + 5;
		List<Integer> ids = new ArrayList<>();
		for (int i = 0; i < n; i++)
			ids.add(pool.allocate());
		int keep = pool.allocate();
		for (int id : ids)
			pool.free(id);
		pool.reclaimIdle();
		assertThat(pool.reclaimIdle()).isEqualTo(n);
		assertThat(pool.storedPages()).isEqualTo(1);
		assertThat(pool.refCount(keep)).isEqualTo(1);
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("tables written, read, truncated and released on other threads while reclaims run read what they wrote")
	void concurrent_reclaim_leaves_live_rows_intact(KvElementType type) throws Exception {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		int writers = 4;
		ExecutorService exec = Executors.newFixedThreadPool(writers + 1);
		AtomicBoolean stop = new AtomicBoolean();
		try {
			Future<?> reclaimer = exec.submit(() -> {
				while (!stop.get())
					pool.reclaimIdle();
				return null;
			});
			List<Future<?>> work = new ArrayList<>();
			for (int w = 0; w < writers; w++) {
				int seed = w;
				work.add(exec.submit(() -> {
					KvPageTable t = new KvPageTable(pool);
					KvPageTable ref = new KvPageTable(new KvBlockPool(PAGE, KV_DIM, type));
					float[] got = new float[KV_DIM];
					float[] want = new float[KV_DIM];
					for (int round = 0; round < 60; round++) {
						int len = 5 + (round * 7 + seed * 3) % 23;
						for (int p = t.seqLen(); p < len; p++) {
							float[] r = row(seed * 10_000 + round * 100 + p);
							t.appendToken(r);
							ref.appendToken(r);
						}
						for (int p = 0; p < t.seqLen(); p++) {
							t.readToken(p, got);
							ref.readToken(p, want);
							if (!java.util.Arrays.equals(got, want))
								throw new AssertionError("writer " + seed + " round " + round + " row " + p);
						}
						if (round % 3 == 2) {
							t.release();
							ref.release();
						} else {
							int cut = t.seqLen() / 2;
							t.truncate(cut);
							ref.truncate(cut);
						}
					}
					t.release();
					return null;
				}));
			}
			for (Future<?> f : work)
				f.get(60, TimeUnit.SECONDS);
			stop.set(true);
			reclaimer.get(10, TimeUnit.SECONDS);
		} finally {
			stop.set(true);
			exec.shutdownNow();
		}
		assertThat(pool.liveBlocks()).isZero();
		pool.reclaimIdle();
		pool.reclaimIdle();
		assertThat(pool.storedPages()).isZero();
		assertThat(pool.storedBytes()).isZero();
	}
}
