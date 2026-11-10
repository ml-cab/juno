package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@DisplayName("KvPageTable")
class KvPageTableTest {

	@Test
	@DisplayName("append grows pages; memory scales with used tokens at page granularity")
	void append_scales_with_pages() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		assertThat(table.seqLen()).isZero();
		assertThat(table.pageCount()).isZero();

		float[] tok = new float[8];
		for (int p = 0; p < 9; p++) {
			for (int i = 0; i < 8; i++)
				tok[i] = p * 10 + i;
			table.appendToken(tok);
		}
		assertThat(table.seqLen()).isEqualTo(9);
		assertThat(table.pageCount()).isEqualTo(3); // ceil(9/4)
		assertThat(pool.liveBlocks()).isEqualTo(3);
		assertThat(pool.allocatedBytes()).isEqualTo(3L * 4 * 8 * Float.BYTES);
	}

	@Test
	@DisplayName("gather copies contiguous prefix matching written tokens")
	void gather_matches_written() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		float[] tok = new float[8];
		for (int p = 0; p < 6; p++) {
			for (int i = 0; i < 8; i++)
				tok[i] = p + i * 0.1f;
			table.appendToken(tok);
		}
		float[] workspace = new float[6 * 8];
		table.gather(workspace);
		for (int p = 0; p < 6; p++) {
			for (int i = 0; i < 8; i++)
				assertThat(workspace[p * 8 + i]).isCloseTo(p + i * 0.1f, offset(1e-5f));
		}
	}

	@Test
	@DisplayName("release returns all pages to the pool")
	void release_frees_pool() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		float[] tok = new float[8];
		for (int p = 0; p < 5; p++)
			table.appendToken(tok);
		assertThat(pool.liveBlocks()).isEqualTo(2);
		table.release();
		assertThat(table.seqLen()).isZero();
		assertThat(table.pageCount()).isZero();
		assertThat(pool.liveBlocks()).isZero();
	}

	// Sharing: a table can start from another table's prefix, holding the same pages.
	// A page held by more than one table is never written: whichever table writes
	// into it first takes a copy. Every expected content below is a table written
	// with the same rows and no sharing, in a pool of its own, compared bit for bit.

	private static final int PAGE = 4;
	private static final int KV_DIM = 64;

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("sharing a prefix holds the same pages and copies none")
	void share_prefix_copies_no_page(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable source = written(pool, salts(9, 0));

		KvPageTable whole = source.sharePrefix(9);
		KvPageTable part = source.sharePrefix(6);
		assertThat(pool.liveBlocks()).isEqualTo(3);
		assertThat(whole.seqLen()).isEqualTo(9);
		assertThat(whole.pageCount()).isEqualTo(3);
		assertThat(part.seqLen()).isEqualTo(6);
		assertThat(part.pageCount()).isEqualTo(2);
		assertThat(contents(whole)).containsExactly(contents(reference(type, salts(9, 0))));
		assertThat(contents(part)).containsExactly(contents(reference(type, salts(6, 0))));
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("after sharing, each table's writes are its own, on either side of a partly shared page")
	void writes_after_sharing_are_private(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable source = written(pool, salts(9, 0));
		KvPageTable sharer = source.sharePrefix(6); // page 1 holds the sharer's rows 4-5 and the source's 4-7

		appendRows(sharer, 6, 11, 1); // copies page 1, then a page of its own for rows 8-10
		appendRows(source, 9, 11, 0); // page 2 is the source's alone: written in place
		assertThat(contents(sharer)).containsExactly(contents(reference(type, concat(salts(6, 0), salts(5, 1)))));
		assertThat(contents(source)).containsExactly(contents(reference(type, salts(11, 0))));
		assertThat(pool.liveBlocks()).as("source 3 pages, sharer 1 copy and 1 new").isEqualTo(5);

		// The source writing first into a page the sharer also holds copies it the same way.
		KvBlockPool pool2 = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable src2 = written(pool2, salts(6, 0));
		KvPageTable sh2 = src2.sharePrefix(6);
		appendRows(src2, 6, 8, 0);
		appendRows(sh2, 6, 8, 1); // page 1 is now the sharer's alone: written in place
		assertThat(contents(src2)).containsExactly(contents(reference(type, salts(8, 0))));
		assertThat(contents(sh2)).containsExactly(contents(reference(type, concat(salts(6, 0), salts(2, 1)))));
		assertThat(pool2.liveBlocks()).as("page 0 shared, page 1 and its copy").isEqualTo(3);
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a page returns to the pool only when the last table holding it lets go")
	void page_freed_when_last_table_releases(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable source = written(pool, salts(9, 0));
		KvPageTable sharer = source.sharePrefix(9);
		KvPageTable shorter = sharer.sharePrefix(4);

		source.release();
		assertThat(pool.liveBlocks()).isEqualTo(3);
		assertThat(contents(sharer)).containsExactly(contents(reference(type, salts(9, 0))));

		sharer.truncate(5);
		assertThat(pool.liveBlocks()).isEqualTo(2);
		sharer.release();
		assertThat(pool.liveBlocks()).as("page 0 is still the shorter table's").isEqualTo(1);
		assertThat(contents(shorter)).containsExactly(contents(reference(type, salts(4, 0))));
		shorter.release();
		assertThat(pool.liveBlocks()).isZero();
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a context shift or an overwrite on a sharer leaves the source's rows intact")
	void shift_and_overwrite_on_a_sharer_leave_the_source_intact(KvElementType type) {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, type);
		KvPageTable source = written(pool, salts(12, 0));

		KvPageTable shifted = source.sharePrefix(12);
		shifted.compact(2, 5); // keeps rows 0-1 and 7-11
		KvPageTable expected = new KvPageTable(new KvBlockPool(PAGE, KV_DIM, type));
		for (int p : new int[] { 0, 1, 7, 8, 9, 10, 11 })
			expected.appendToken(row(p, 0));
		assertThat(contents(shifted)).containsExactly(contents(expected));

		KvPageTable overwritten = source.sharePrefix(12);
		overwritten.writeToken(2, row(2, 7)); // in place at a position below seqLen, as a redrafted row
		int[] salts = salts(12, 0);
		salts[2] = 7;
		assertThat(contents(overwritten)).containsExactly(contents(reference(type, salts)));

		assertThat(contents(source)).containsExactly(contents(reference(type, salts(12, 0))));
	}

	@Test
	@DisplayName("sharing an empty prefix holds no page; a prefix past the written rows is refused")
	void share_prefix_bounds() {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, KvElementType.F16);
		KvPageTable source = written(pool, salts(6, 0));
		KvPageTable empty = source.sharePrefix(0);
		assertThat(empty.seqLen()).isZero();
		assertThat(empty.pageCount()).isZero();
		assertThat(pool.liveBlocks()).isEqualTo(2);
		empty.appendToken(row(0, 3));
		assertThat(pool.liveBlocks()).isEqualTo(3);

		assertThatThrownBy(() -> source.sharePrefix(-1)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> source.sharePrefix(7)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("two sessions sharing a prefix write and read on two threads while a third reads the source")
	void two_sessions_sharing_a_prefix_on_two_threads() throws Exception {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, KvElementType.F16);
		KvPageTable source = written(pool, salts(10, 0)); // last page partly filled, held by all three
		int extra = 200;
		KvPageTable[] sharers = { source.sharePrefix(10), source.sharePrefix(10) };
		float[] sourceRows = contents(reference(KvElementType.F16, salts(10, 0)));

		var exec = Executors.newFixedThreadPool(3);
		CountDownLatch start = new CountDownLatch(1);
		List<Future<?>> futures = new ArrayList<>();
		for (int t = 0; t < 2; t++) {
			KvPageTable mine = sharers[t];
			int salt = t + 1;
			futures.add(exec.submit(() -> {
				start.await();
				for (int p = 10; p < 10 + extra; p++) {
					mine.appendToken(row(p, salt));
					if (p % 10 == 0)
						assertThat(contents(mine))
								.containsExactly(contents(reference(KvElementType.F16, concat(salts(10, 0),
										salts(p - 9, salt)))));
				}
				return null;
			}));
		}
		futures.add(exec.submit(() -> {
			start.await();
			for (int i = 0; i < 2_000; i++)
				assertThat(contents(source)).containsExactly(sourceRows);
			return null;
		}));
		start.countDown();
		for (Future<?> f : futures)
			f.get(60, TimeUnit.SECONDS);
		exec.shutdownNow();

		for (int t = 0; t < 2; t++)
			assertThat(contents(sharers[t]))
					.containsExactly(contents(reference(KvElementType.F16, concat(salts(10, 0), salts(extra, t + 1)))));
		source.release();
		sharers[0].release();
		sharers[1].release();
		assertThat(pool.liveBlocks()).isZero();
	}

	/** A row that depends on its position and a salt, so each writer's rows differ. */
	private static float[] row(int pos, int salt) {
		float[] r = new float[KV_DIM];
		for (int i = 0; i < KV_DIM; i++)
			r[i] = (float) Math.sin(pos * 0.37 + i * 0.11 + salt * 1.7);
		return r;
	}

	private static int[] salts(int n, int salt) {
		int[] s = new int[n];
		java.util.Arrays.fill(s, salt);
		return s;
	}

	private static int[] concat(int[] a, int[] b) {
		int[] out = java.util.Arrays.copyOf(a, a.length + b.length);
		System.arraycopy(b, 0, out, a.length, b.length);
		return out;
	}

	/** Rows {@code 0..salts.length-1}, row {@code p} salted with {@code salts[p]}. */
	private static KvPageTable written(KvBlockPool pool, int[] salts) {
		KvPageTable t = new KvPageTable(pool);
		for (int p = 0; p < salts.length; p++)
			t.appendToken(row(p, salts[p]));
		return t;
	}

	private static KvPageTable reference(KvElementType type, int[] salts) {
		return written(new KvBlockPool(PAGE, KV_DIM, type), salts);
	}

	private static void appendRows(KvPageTable t, int from, int to, int salt) {
		for (int p = from; p < to; p++)
			t.appendToken(row(p, salt));
	}

	private static float[] contents(KvPageTable t) {
		float[] out = new float[t.seqLen() * t.kvDim()];
		if (t.seqLen() > 0)
			t.gather(out, t.seqLen());
		return out;
	}
}
