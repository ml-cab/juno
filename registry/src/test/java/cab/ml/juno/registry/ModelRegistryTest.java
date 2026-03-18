package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ModelRegistryTest {

	private static final long GB = 1024L * 1024 * 1024;

	private ModelRegistry registry;
	private ModelDescriptor tinyllama;
	private List<NodeDescriptor> threeNodes;

	@BeforeEach
	void setUp() {
		registry = new ModelRegistry(ShardPlanner.create());
		tinyllama = ModelDescriptor.of("tinyllama", "llama", 22, 2048, 32000, 32, QuantizationType.Q4_K_M,
				"/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf");
		threeNodes = List.of(node("n1", 4 * GB, NodeStatus.READY), node("n2", 4 * GB, NodeStatus.READY),
				node("n3", 4 * GB, NodeStatus.READY));
	}

	// ── register ─────────────────────────────────────────────────────────────

	@Test
	void register_returns_valid_shard_map() {
		ShardMap map = registry.register(tinyllama, threeNodes);

		assertThat(map.modelId()).isEqualTo("tinyllama");
		assertThat(map.nodeCount()).isGreaterThan(0);
		map.validateCoverage();
	}

	@Test
	void register_transitions_model_to_loading_status() {
		registry.register(tinyllama, threeNodes);

		assertThat(registry.getModel("tinyllama")).isPresent()
				.hasValueSatisfying(m -> assertThat(m.status()).isEqualTo(ModelStatus.LOADING));
	}

	@Test
	void register_stores_shard_map_retrievable_by_model_id() {
		registry.register(tinyllama, threeNodes);

		assertThat(registry.getShardMap("tinyllama")).isPresent();
	}

	@Test
	void register_same_model_twice_replaces_old_shard_map() {
		ShardMap first = registry.register(tinyllama, threeNodes);
		ShardMap second = registry.register(tinyllama, threeNodes);

		// second shard map must be computed after (or equal to) the first
		assertThat(second.computedAt()).isAfterOrEqualTo(first.computedAt());
		// registry holds the second map
		assertThat(registry.getShardMap("tinyllama").orElseThrow().computedAt()).isEqualTo(second.computedAt());
		// still one model — no duplicates
		assertThat(registry.listModels()).hasSize(1);
	}

	@Test
	void register_throws_when_no_nodes_available() {
		List<NodeDescriptor> offlineNodes = List.of(node("n1", 4 * GB, NodeStatus.OFFLINE));
		assertThatThrownBy(() -> registry.register(tinyllama, offlineNodes))
				.isInstanceOf(InsufficientClusterVramException.class);
	}

	// ── unregister ────────────────────────────────────────────────────────────

	@Test
	void unregister_removes_model_and_shard_map() {
		registry.register(tinyllama, threeNodes);
		registry.unregister("tinyllama");

		assertThat(registry.getModel("tinyllama")).isEmpty();
		assertThat(registry.getShardMap("tinyllama")).isEmpty();
	}

	@Test
	void unregister_nonexistent_model_is_noop() {
		// should not throw
		registry.unregister("does-not-exist");
	}

	@Test
	void is_loaded_false_after_unregister() {
		registry.register(tinyllama, threeNodes);
		registry.unregister("tinyllama");
		assertThat(registry.isLoaded("tinyllama")).isFalse();
	}

	// ── status lifecycle ─────────────────────────────────────────────────────

	@Test
	void mark_loaded_transitions_to_loaded() {
		registry.register(tinyllama, threeNodes);
		registry.markLoaded("tinyllama");

		assertThat(registry.getModel("tinyllama").orElseThrow().status()).isEqualTo(ModelStatus.LOADED);
		assertThat(registry.isLoaded("tinyllama")).isTrue();
	}

	@Test
	void mark_error_transitions_to_error() {
		registry.register(tinyllama, threeNodes);
		registry.markError("tinyllama", "GPU OOM");

		assertThat(registry.getModel("tinyllama").orElseThrow().status()).isEqualTo(ModelStatus.ERROR);
		assertThat(registry.isLoaded("tinyllama")).isFalse();
	}

	@Test
	void mark_loaded_on_unknown_model_is_noop() {
		// should not throw
		registry.markLoaded("ghost");
	}

	// ── query ─────────────────────────────────────────────────────────────────

	@Test
	void list_models_empty_initially() {
		assertThat(registry.listModels()).isEmpty();
	}

	@Test
	void list_models_returns_all_registered() {
		ModelDescriptor another = ModelDescriptor.of("llama3-8b", "llama", 32, 4096, 32000, 32, QuantizationType.Q4_K_M,
				"/models/llama3-8b.gguf");

		registry.register(tinyllama, threeNodes);
		registry.register(another, threeNodes);

		assertThat(registry.listModels()).hasSize(2);
		assertThat(registry.listModels()).extracting(ModelDescriptor::modelId).containsExactlyInAnyOrder("tinyllama",
				"llama3-8b");
	}

	@Test
	void get_model_returns_empty_for_unknown() {
		assertThat(registry.getModel("unknown")).isEmpty();
	}

	@Test
	void get_shard_map_returns_empty_for_unknown() {
		assertThat(registry.getShardMap("unknown")).isEmpty();
	}

	@Test
	void is_loaded_false_for_loading_state() {
		registry.register(tinyllama, threeNodes); // status = LOADING
		assertThat(registry.isLoaded("tinyllama")).isFalse();
	}

	// ── helpers ───────────────────────────────────────────────────────────────

	private NodeDescriptor node(String id, long vram, NodeStatus status) {
		return new NodeDescriptor(id, "192.168.1." + id.charAt(id.length() - 1), 9092, vram, vram, status, 0.9,
				Instant.now(), Instant.now());
	}
}
