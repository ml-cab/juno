package cab.ml.juno.kvcache;

import static cab.ml.juno.kvcache.KVBlockFactory.block;
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class CpuKVCacheTest {

	private final CpuKVCache cache = new CpuKVCache(100);

	@Test
	void put_and_get_returns_stored_block() {
		KVKey key = new KVKey("req-1", 0);
		KVBlock block = block("req-1", 0);

		cache.put(key, block);

		assertThat(cache.get(key)).isPresent();
	}

	@Test
	void get_returns_empty_for_missing_key() {
		assertThat(cache.get(new KVKey("missing", 0))).isEmpty();
	}

	@Test
	void contains_returns_true_after_put() {
		KVKey key = new KVKey("req-2", 1);
		cache.put(key, block("req-2", 1));
		assertThat(cache.contains(key)).isTrue();
	}

	@Test
	void evict_removes_all_blocks_for_request() {
		cache.put(new KVKey("req-3", 0), block("req-3", 0));
		cache.put(new KVKey("req-3", 1), block("req-3", 1));
		cache.put(new KVKey("req-3", 2), block("req-3", 2));

		cache.evict("req-3");

		assertThat(cache.contains(new KVKey("req-3", 0))).isFalse();
		assertThat(cache.contains(new KVKey("req-3", 1))).isFalse();
		assertThat(cache.contains(new KVKey("req-3", 2))).isFalse();
	}

	@Test
	void evict_only_removes_target_request() {
		cache.put(new KVKey("req-a", 0), block("req-a", 0));
		cache.put(new KVKey("req-b", 0), block("req-b", 0));

		cache.evict("req-a");

		assertThat(cache.contains(new KVKey("req-b", 0))).isTrue();
	}

	@Test
	void estimated_size_bytes_tracks_stored_data() {
		cache.put(new KVKey("req-4", 0), block("req-4", 0, 2048));
		assertThat(cache.estimatedSizeBytes()).isGreaterThanOrEqualTo(2048);
	}

	@Test
	void tier_name_is_cpu() {
		assertThat(cache.tierName()).isEqualTo("cpu");
	}
}
