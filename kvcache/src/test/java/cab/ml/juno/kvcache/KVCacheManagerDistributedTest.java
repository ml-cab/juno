package cab.ml.juno.kvcache;

import static cab.ml.juno.kvcache.KVBlockFactory.block;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * Tests for distributed KV cache behaviour — each node owns only its layer
 * range.
 *
 * Topology used in these tests: Node 1 — layers 0..7 (LayerRange.of(0, 8)) Node
 * 2 — layers 8..15 (LayerRange.of(8, 16)) Node 3 — layers 16..21
 * (LayerRange.of(16, 22))
 */
class KVCacheManagerDistributedTest {

	private static final long MB = 1024 * 1024;

	private KVCacheManager managerForRange(LayerRange range) {
		return new KVCacheManager(new GpuKVCache(64 * MB), new CpuKVCache(1000), range);
	}

	// ── Layer ownership ───────────────────────────────────────────────────────

	@Test
	void node1_accepts_blocks_for_its_layers() {
		KVCacheManager node1 = managerForRange(LayerRange.of(0, 8));

		for (int layer = 0; layer < 8; layer++) {
			node1.put(new KVKey("req-1", layer), block("req-1", layer));
		}

		for (int layer = 0; layer < 8; layer++) {
			assertThat(node1.get(new KVKey("req-1", layer))).isPresent();
		}
	}

	@Test
	void node1_rejects_blocks_for_other_nodes_layers() {
		KVCacheManager node1 = managerForRange(LayerRange.of(0, 8));

		assertThatThrownBy(() -> node1.put(new KVKey("req-1", 8), block("req-1", 8)))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("layer 8");

		assertThatThrownBy(() -> node1.put(new KVKey("req-1", 16), block("req-1", 16)))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("layer 16");
	}

	@Test
	void node2_rejects_blocks_from_node1_and_node3_ranges() {
		KVCacheManager node2 = managerForRange(LayerRange.of(8, 16));

		assertThatThrownBy(() -> node2.put(new KVKey("req-1", 0), block("req-1", 0)))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> node2.put(new KVKey("req-1", 16), block("req-1", 16)))
				.isInstanceOf(IllegalArgumentException.class);
	}

	// ── Isolation across nodes ────────────────────────────────────────────────

	@Test
	void blocks_are_isolated_between_nodes() {
		KVCacheManager node1 = managerForRange(LayerRange.of(0, 8));
		KVCacheManager node2 = managerForRange(LayerRange.of(8, 16));
		KVCacheManager node3 = managerForRange(LayerRange.of(16, 22));

		// Each node stores blocks for its own layers
		node1.put(new KVKey("req-1", 0), block("req-1", 0));
		node2.put(new KVKey("req-1", 8), block("req-1", 8));
		node3.put(new KVKey("req-1", 16), block("req-1", 16));

		// Each node can retrieve only its own blocks
		assertThat(node1.get(new KVKey("req-1", 0))).isPresent();
		assertThat(node2.get(new KVKey("req-1", 8))).isPresent();
		assertThat(node3.get(new KVKey("req-1", 16))).isPresent();

		// No cross-node visibility
		assertThat(node1.get(new KVKey("req-1", 8))).isEmpty();
		assertThat(node2.get(new KVKey("req-1", 0))).isEmpty();
		assertThat(node3.get(new KVKey("req-1", 8))).isEmpty();
	}

	@Test
	void evict_on_one_node_does_not_affect_others() {
		KVCacheManager node1 = managerForRange(LayerRange.of(0, 8));
		KVCacheManager node2 = managerForRange(LayerRange.of(8, 16));

		node1.put(new KVKey("req-1", 0), block("req-1", 0));
		node2.put(new KVKey("req-1", 8), block("req-1", 8));

		node1.evict("req-1");

		// node1 cleared
		assertThat(node1.get(new KVKey("req-1", 0))).isEmpty();
		// node2 unaffected — evict is local
		assertThat(node2.get(new KVKey("req-1", 8))).isPresent();
	}

	// ── Backward compat — LayerRange.all() ───────────────────────────────────

	@Test
	void unbounded_manager_accepts_any_layer() {
		// Old-style coordinator manager — no layer restriction
		KVCacheManager coordinator = new KVCacheManager(new GpuKVCache(64 * MB), new CpuKVCache(1000));

		// Should accept blocks for any layer without throwing
		coordinator.put(new KVKey("req-1", 0), block("req-1", 0));
		coordinator.put(new KVKey("req-1", 15), block("req-1", 15));
		coordinator.put(new KVKey("req-1", 99), block("req-1", 99));

		assertThat(coordinator.get(new KVKey("req-1", 0))).isPresent();
		assertThat(coordinator.get(new KVKey("req-1", 99))).isPresent();
	}

	// ── Layer range metadata ──────────────────────────────────────────────────

	@Test
	void layer_range_is_queryable_on_manager() {
		KVCacheManager node2 = managerForRange(LayerRange.of(8, 16));
		assertThat(node2.layerRange().startLayer()).isEqualTo(8);
		assertThat(node2.layerRange().endLayer()).isEqualTo(16);
		assertThat(node2.layerRange().layerCount()).isEqualTo(8);
	}

	@Test
	void owns_layer_reflects_range() {
		KVCacheManager node1 = managerForRange(LayerRange.of(0, 8));
		assertThat(node1.ownsLayer(0)).isTrue();
		assertThat(node1.ownsLayer(7)).isTrue();
		assertThat(node1.ownsLayer(8)).isFalse();
	}
}
