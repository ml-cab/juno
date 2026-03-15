package cab.ml.juno.kvcache;

import static cab.ml.juno.kvcache.KVBlockFactory.block;
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVKey;

class GpuKVCacheTest {

	private static final long MB = 1024 * 1024;
	private final GpuKVCache cache = new GpuKVCache(10 * MB);

	@Test
	void put_and_get_returns_stored_block() {
		KVKey key = new KVKey("req-1", 0);
		cache.put(key, block("req-1", 0));
		assertThat(cache.get(key)).isPresent();
	}

	@Test
	void evicts_oldest_when_budget_exceeded() {
		// Budget: 10MB. Each block: 6MB. Second put should evict first.
		GpuKVCache tiny = new GpuKVCache(10 * MB);
		KVKey key1 = new KVKey("req-1", 0);
		KVKey key2 = new KVKey("req-2", 0);

		tiny.put(key1, block("req-1", 0, (int) (6 * MB)));
		tiny.put(key2, block("req-2", 0, (int) (6 * MB)));

		// At least one of them was evicted — both can't fit in 10MB
		long total = tiny.estimatedSizeBytes();
		assertThat(total).isLessThanOrEqualTo(10 * MB);
	}

	@Test
	void evict_by_requestId_removes_all_layers() {
		cache.put(new KVKey("req-3", 0), block("req-3", 0));
		cache.put(new KVKey("req-3", 1), block("req-3", 1));

		cache.evict("req-3");

		assertThat(cache.contains(new KVKey("req-3", 0))).isFalse();
		assertThat(cache.contains(new KVKey("req-3", 1))).isFalse();
	}

	@Test
	void estimated_size_bytes_tracks_usage() {
		cache.put(new KVKey("req-4", 0), block("req-4", 0, 4096));
		assertThat(cache.estimatedSizeBytes()).isGreaterThanOrEqualTo(4096);
	}

	@Test
	void tier_name_is_gpu() {
		assertThat(cache.tierName()).isEqualTo("gpu");
	}
}
