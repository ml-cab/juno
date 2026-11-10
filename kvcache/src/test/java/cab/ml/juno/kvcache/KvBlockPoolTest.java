package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@DisplayName("KvBlockPool")
class KvBlockPoolTest {

	@Test
	@DisplayName("allocate returns distinct ids and free recycles")
	void allocate_and_free_recycles() {
		KvBlockPool pool = new KvBlockPool(16, 64, KvElementType.F16);
		int a = pool.allocate();
		int b = pool.allocate();
		assertThat(a).isNotEqualTo(b);
		assertThat(pool.liveBlocks()).isEqualTo(2);
		assertThat(pool.allocatedBytes()).isEqualTo(2L * 16 * 64 * Float.BYTES);

		pool.free(a);
		assertThat(pool.liveBlocks()).isEqualTo(1);
		int c = pool.allocate();
		assertThat(c).isEqualTo(a); // recycled
		assertThat(pool.liveBlocks()).isEqualTo(2);
	}

	@Test
	@DisplayName("writeToken / readToken roundtrip within a page")
	void write_read_roundtrip() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		int id = pool.allocate();
		float[] tok = new float[8];
		for (int i = 0; i < 8; i++)
			tok[i] = i + 0.5f;
		pool.writeToken(id, 2, tok);
		float[] got = new float[8];
		pool.readToken(id, 2, got);
		assertThat(got).containsExactly(tok);
	}

	@Test
	@DisplayName("reject out-of-range slot and double-free")
	void rejects_bad_ops() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		int id = pool.allocate();
		float[] tok = new float[8];
		assertThatThrownBy(() -> pool.writeToken(id, 4, tok))
				.isInstanceOf(IllegalArgumentException.class);
		pool.free(id);
		assertThatThrownBy(() -> pool.free(id))
				.isInstanceOf(IllegalStateException.class);
	}

	@Test
	@DisplayName("Q8_0 write/read roundtrip and smaller pages than F16")
	void q8_roundtrip_and_size() {
		int kvDim = 256;
		int pageSize = 8;
		KvBlockPool f16 = new KvBlockPool(pageSize, kvDim, KvElementType.F16);
		KvBlockPool q8 = new KvBlockPool(pageSize, kvDim, KvElementType.Q8_0);
		int id = q8.allocate();
		float[] tok = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			tok[i] = (float) Math.cos(i * 0.02);
		q8.writeToken(id, 3, tok);
		float[] got = new float[kvDim];
		q8.readToken(id, 3, got);
		for (int i = 0; i < kvDim; i++)
			assertThat(got[i]).isCloseTo(tok[i], org.assertj.core.data.Offset.offset(0.05f));

		f16.allocate();
		assertThat(q8.bytesPerToken() * 2L).isLessThanOrEqualTo(f16.bytesPerToken());
		assertThat(q8.allocatedBytes()).isLessThan(f16.allocatedBytes());
	}

	@Test
	@DisplayName("concurrent allocate/free keeps unique live ids")
	void concurrent_allocate_free() throws Exception {
		KvBlockPool pool = new KvBlockPool(8, 16, KvElementType.F16);
		int threads = 8;
		int ops = 200;
		var exec = Executors.newFixedThreadPool(threads);
		CountDownLatch start = new CountDownLatch(1);
		List<Future<List<Integer>>> futures = new ArrayList<>();
		for (int t = 0; t < threads; t++) {
			futures.add(exec.submit(() -> {
				start.await();
				List<Integer> held = new ArrayList<>();
				for (int i = 0; i < ops; i++) {
					held.add(pool.allocate());
					if (held.size() > 4) {
						pool.free(held.remove(0));
					}
				}
				return held;
			}));
		}
		start.countDown();
		Set<Integer> allLive = new HashSet<>();
		for (Future<List<Integer>> f : futures) {
			List<Integer> held = f.get(10, TimeUnit.SECONDS);
			for (int id : held) {
				assertThat(allLive.add(id)).as("duplicate live id %s", id).isTrue();
			}
		}
		exec.shutdownNow();
		assertThat(pool.liveBlocks()).isEqualTo(allLive.size());
	}

	@Test
	@DisplayName("a retained page stays live until its last reference is freed, then recycles")
	void retained_page_lives_until_last_free() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		int a = pool.allocate();
		assertThat(pool.refCount(a)).isEqualTo(1);
		pool.retain(a);
		assertThat(pool.refCount(a)).isEqualTo(2);
		assertThat(pool.liveBlocks()).isEqualTo(1);
		assertThat(pool.allocatedBytes()).isEqualTo(4L * 8 * Float.BYTES);

		pool.free(a);
		assertThat(pool.refCount(a)).isEqualTo(1);
		int b = pool.allocate();
		assertThat(b).as("a page still referenced is not handed out again").isNotEqualTo(a);
		pool.free(b);

		pool.free(a);
		assertThat(pool.refCount(a)).isZero();
		assertThatThrownBy(() -> pool.free(a)).isInstanceOf(IllegalStateException.class);
		assertThatThrownBy(() -> pool.retain(a)).isInstanceOf(IllegalStateException.class);
		assertThat(pool.liveBlocks()).isZero();
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a shared page refuses writes; exclusive() hands the writer a copy, byte for byte")
	void shared_page_is_read_only_until_made_exclusive(KvElementType type) {
		int kvDim = 64;
		KvBlockPool pool = new KvBlockPool(4, kvDim, type);
		int a = pool.allocate();
		for (int slot = 0; slot < 4; slot++)
			pool.writeToken(a, slot, row(slot, kvDim));
		pool.retain(a);

		assertThatThrownBy(() -> pool.writeToken(a, 1, row(9, kvDim))).isInstanceOf(IllegalStateException.class);
		int other = pool.allocate();
		assertThatThrownBy(() -> pool.copyToken(other, 0, a, 2)).isInstanceOf(IllegalStateException.class);
		pool.copyToken(a, 2, other, 0); // reading a shared page is allowed
		float[] read = new float[kvDim];
		pool.readToken(a, 1, read);

		byte[] before = pool.pageBytes(a).clone();
		int copy = pool.exclusive(a);
		assertThat(copy).isNotEqualTo(a);
		assertThat(pool.refCount(a)).isEqualTo(1);
		assertThat(pool.refCount(copy)).isEqualTo(1);
		assertThat(pool.pageBytes(copy)).isEqualTo(before);

		pool.writeToken(copy, 1, row(9, kvDim));
		assertThat(pool.pageBytes(a)).as("the other holder's page is untouched").isEqualTo(before);
		assertThat(pool.exclusive(a)).as("a sole holder keeps its page").isEqualTo(a);
		pool.writeToken(a, 1, row(9, kvDim));
	}

	@Test
	@DisplayName("holders of one page making it exclusive at once each end with a page of their own")
	void concurrent_exclusive_gives_each_holder_its_own_page() throws Exception {
		int holders = 8;
		KvBlockPool pool = new KvBlockPool(4, 16, KvElementType.F16);
		int shared = pool.allocate();
		for (int slot = 0; slot < 4; slot++)
			pool.writeToken(shared, slot, row(slot, 16));
		byte[] content = pool.pageBytes(shared).clone();
		for (int i = 1; i < holders; i++)
			pool.retain(shared);

		var exec = Executors.newFixedThreadPool(holders);
		CountDownLatch start = new CountDownLatch(1);
		List<Future<Integer>> futures = new ArrayList<>();
		for (int t = 0; t < holders; t++)
			futures.add(exec.submit(() -> {
				start.await();
				return pool.exclusive(shared);
			}));
		start.countDown();
		Set<Integer> owned = new HashSet<>();
		for (Future<Integer> f : futures)
			assertThat(owned.add(f.get(10, TimeUnit.SECONDS))).isTrue();
		exec.shutdownNow();

		assertThat(owned).hasSize(holders).contains(shared);
		for (int id : owned) {
			assertThat(pool.refCount(id)).isEqualTo(1);
			assertThat(pool.pageBytes(id)).isEqualTo(content);
		}
		assertThat(pool.liveBlocks()).isEqualTo(holders);
	}

	@Test
	@DisplayName("concurrent retain and free keep every page's count")
	void concurrent_retain_and_free_keep_counts() throws Exception {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		int pages = 16;
		int threads = 8;
		int rounds = 300;
		int[] ids = new int[pages];
		for (int i = 0; i < pages; i++)
			ids[i] = pool.allocate();
		var exec = Executors.newFixedThreadPool(threads);
		CountDownLatch start = new CountDownLatch(1);
		List<Future<?>> futures = new ArrayList<>();
		for (int t = 0; t < threads; t++)
			futures.add(exec.submit(() -> {
				start.await();
				for (int r = 0; r < rounds; r++) {
					for (int id : ids)
						pool.retain(id);
					for (int id : ids)
						pool.free(id);
				}
				return null;
			}));
		start.countDown();
		for (Future<?> f : futures)
			f.get(30, TimeUnit.SECONDS);
		exec.shutdownNow();

		for (int id : ids)
			assertThat(pool.refCount(id)).isEqualTo(1);
		for (int id : ids)
			pool.free(id);
		assertThat(pool.liveBlocks()).isZero();
	}

	private static float[] row(int salt, int kvDim) {
		float[] r = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			r[i] = (float) Math.sin(salt * 0.7 + i * 0.13);
		return r;
	}
}
