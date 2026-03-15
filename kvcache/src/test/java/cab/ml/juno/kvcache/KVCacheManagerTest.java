package cab.ml.juno.kvcache;

import static cab.ml.juno.kvcache.KVBlockFactory.block;
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KVKey;
import cab.ml.juno.kvcache.PrefixCache;

class KVCacheManagerTest {

	private static final long MB = 1024 * 1024;

	private KVCacheManager manager() {
		return new KVCacheManager(new GpuKVCache(100 * MB), new CpuKVCache(1000));
	}

	@Test
	void put_makes_block_retrievable() {
		KVCacheManager mgr = manager();
		KVKey key = new KVKey("req-1", 0);

		mgr.put(key, block("req-1", 0));

		assertThat(mgr.get(key)).isPresent();
	}

	@Test
	void get_falls_back_to_cpu_when_not_in_gpu() {
		// Put directly into CPU cache, bypass manager
		CpuKVCache cpu = new CpuKVCache(1000);
		GpuKVCache gpu = new GpuKVCache(100 * MB);
		KVCacheManager mgr = new KVCacheManager(gpu, cpu);

		KVKey key = new KVKey("req-2", 0);
		cpu.put(key, block("req-2", 0));

		assertThat(mgr.get(key)).isPresent();
	}

	@Test
	void evict_clears_both_tiers() {
		KVCacheManager mgr = manager();
		mgr.put(new KVKey("req-3", 0), block("req-3", 0));
		mgr.put(new KVKey("req-3", 1), block("req-3", 1));

		mgr.evict("req-3");

		assertThat(mgr.get(new KVKey("req-3", 0))).isEmpty();
		assertThat(mgr.get(new KVKey("req-3", 1))).isEmpty();
	}

	@Test
	void prefix_cache_hit_detected_after_caching() {
		KVCacheManager mgr = manager();
		int[] tokens = { 1, 2, 3, 4, 5, 6, 7 };

		mgr.cachePrefix(tokens, 5, "prefix-key");
		PrefixCache.PrefixMatch match = mgr.findLongestPrefix(tokens);

		assertThat(match.isHit()).isTrue();
		assertThat(match.matchedTokens()).isEqualTo(5);
	}

	@Test
	void stats_reflect_stored_blocks() {
		KVCacheManager mgr = manager();
		mgr.put(new KVKey("req-4", 0), block("req-4", 0, 4096));

		assertThat(mgr.gpuBytesUsed()).isGreaterThanOrEqualTo(4096);
		assertThat(mgr.cpuBytesUsed()).isGreaterThanOrEqualTo(4096);
		assertThat(mgr.gpuBlockCount()).isGreaterThan(0);
	}
}
