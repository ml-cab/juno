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
}
