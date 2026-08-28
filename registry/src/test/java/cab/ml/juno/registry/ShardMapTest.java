package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

class ShardMapTest {

	private ShardAssignment shard(String nodeId, int start, int end) {
		return new ShardAssignment(nodeId, "host", 9092, start, end, start == 0, end == 32);
	}

	private ShardMap validMap() {
		return new ShardMap("llama3-8b", 32, List.of(shard("n1", 0, 16), shard("n2", 16, 32)), Instant.now());
	}

	@Test
	void first_node_has_embeddings() {
		ShardMap map = validMap();
		assertThat(map.firstNode().hasEmbeddings()).isTrue();
		assertThat(map.firstNode().nodeId()).isEqualTo("n1");
	}

	@Test
	void last_node_has_output_projection() {
		ShardMap map = validMap();
		assertThat(map.lastNode().hasOutputProjection()).isTrue();
		assertThat(map.lastNode().nodeId()).isEqualTo("n2");
	}

	@Test
	void node_count_is_correct() {
		assertThat(validMap().nodeCount()).isEqualTo(2);
	}

	@Test
	void validate_coverage_passes_for_valid_map() {
		assertThatCode(() -> validMap().validateCoverage()).doesNotThrowAnyException();
	}

	@Test
	void validate_coverage_detects_gap() {
		ShardMap gapped = new ShardMap("model", 32, List.of(shard("n1", 0, 10), shard("n2", 12, 32) // gap at layers
																									// 10-11
		), Instant.now());
		assertThatThrownBy(gapped::validateCoverage).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("gap");
	}

	@Test
	void validate_coverage_detects_incomplete_coverage() {
		ShardMap incomplete = new ShardMap("model", 32, List.of(shard("n1", 0, 16) // only covers half
		), Instant.now());
		assertThatThrownBy(incomplete::validateCoverage).isInstanceOf(IllegalStateException.class);
	}

	@Test
	void assignments_list_is_immutable() {
		ShardMap map = validMap();
		assertThatThrownBy(() -> map.assignments().add(shard("n3", 32, 48)))
				.isInstanceOf(UnsupportedOperationException.class);
	}

	// ── evenSplit: local in-process pipeline-parallelism split ──────────────────
	//
	// Regression net for the bug this replaces: ConsoleMain's local mode used
	// to route through ShardPlanner.plan() with a fabricated per-node VRAM
	// figure large enough to "always fit". Since ShardPlanner is a greedy
	// algorithm (first node takes everything that fits, leaving only the
	// contractually-required 1 layer for each remaining node), that gave node 0
	// nearly the entire model and 1 layer each to every other node — for a
	// 24-layer model split 3 ways, 22/1/1 instead of 8/8/8. These tests assert
	// evenSplit actually splits evenly.

	@Test
	void evenSplit_divides_evenly_when_layers_are_a_multiple_of_nodeCount() {
		ShardMap map = ShardMap.evenSplit("model", 24, 3);

		assertThat(map.nodeCount()).isEqualTo(3);
		assertThat(map.assignments()).extracting(ShardAssignment::layerCount).containsExactly(8, 8, 8);
		map.validateCoverage(); // no gaps/overlaps, covers all 24 layers
	}

	@Test
	void evenSplit_distributes_remainder_one_layer_per_node_from_the_front() {
		// 25 layers / 3 nodes = 8 each with 1 left over — must not all pile onto
		// one node the way the old VRAM-greedy path did.
		ShardMap map = ShardMap.evenSplit("model", 25, 3);

		assertThat(map.assignments()).extracting(ShardAssignment::layerCount).containsExactly(9, 8, 8);
		map.validateCoverage();
	}

	@Test
	void evenSplit_first_node_has_embeddings_last_node_has_output_projection() {
		ShardMap map = ShardMap.evenSplit("model", 24, 3);

		assertThat(map.firstNode().hasEmbeddings()).isTrue();
		assertThat(map.firstNode().hasOutputProjection()).isFalse();
		assertThat(map.lastNode().hasOutputProjection()).isTrue();
		assertThat(map.lastNode().hasEmbeddings()).isFalse();
	}

	@Test
	void evenSplit_single_node_gets_the_whole_model() {
		ShardMap map = ShardMap.evenSplit("model", 24, 1);

		assertThat(map.nodeCount()).isEqualTo(1);
		assertThat(map.firstNode().hasEmbeddings()).isTrue();
		assertThat(map.firstNode().hasOutputProjection()).isTrue();
		assertThat(map.firstNode().layerCount()).isEqualTo(24);
	}

	@Test
	void evenSplit_nodeCount_greater_than_totalLayers_is_capped_to_one_layer_per_node() {
		// Asking for more stages than layers can't give every stage a layer —
		// cap rather than throw or hand out empty/zero-layer assignments.
		ShardMap map = ShardMap.evenSplit("model", 3, 8);

		assertThat(map.nodeCount()).isEqualTo(3);
		assertThat(map.assignments()).extracting(ShardAssignment::layerCount).containsExactly(1, 1, 1);
		map.validateCoverage();
	}

	@Test
	void evenSplit_rejects_nodeCount_below_one() {
		assertThatThrownBy(() -> ShardMap.evenSplit("model", 24, 0)).isInstanceOf(IllegalArgumentException.class);
	}
}