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
}
