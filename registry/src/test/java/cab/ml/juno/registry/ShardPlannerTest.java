package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

class ShardPlannerTest {

	private static final long GB = 1024L * 1024 * 1024;

	private final ShardPlanner planner = ShardPlanner.create();

	private NodeDescriptor node(String id, long vramFreeGb, NodeStatus status) {
		long total = vramFreeGb * GB;
		return new NodeDescriptor(id, "192.168.1." + id.charAt(id.length() - 1), 9092, total, total, status, 0.0,
				Instant.now(), Instant.now());
	}

	@Test
	void single_node_gets_all_layers() {
		List<NodeDescriptor> nodes = List.of(node("n1", 8, NodeStatus.READY));

		ShardMap map = planner.plan("llama3-8b", 32, 200 * 1024 * 1024L, nodes);

		assertThat(map.nodeCount()).isEqualTo(1);
		assertThat(map.firstNode().startLayer()).isEqualTo(0);
		assertThat(map.firstNode().endLayer()).isEqualTo(32);
		assertThat(map.firstNode().hasEmbeddings()).isTrue();
		assertThat(map.firstNode().hasOutputProjection()).isTrue();
	}

	@Test
	void two_equal_nodes_split_layers_evenly() {
		List<NodeDescriptor> nodes = List.of(node("n1", 4, NodeStatus.READY), node("n2", 4, NodeStatus.READY));
		// 4GB usable per node (3.6GB after headroom) → ~18 layers at 200MB each
		// 32 layers total → needs both nodes
		long vramPerLayer = 200 * 1024 * 1024L;

		ShardMap map = planner.plan("model", 32, vramPerLayer, nodes);

		assertThat(map.nodeCount()).isEqualTo(2);
		map.validateCoverage();
	}

	@Test
	void first_assignment_has_embeddings_last_has_output_projection() {
		List<NodeDescriptor> nodes = List.of(node("n1", 4, NodeStatus.READY), node("n2", 4, NodeStatus.READY));
		// 200MB/layer forces split across both nodes
		ShardMap map = planner.plan("model", 32, 200 * 1024 * 1024L, nodes);

		assertThat(map.firstNode().hasEmbeddings()).isTrue();
		assertThat(map.firstNode().hasOutputProjection()).isFalse();
		assertThat(map.lastNode().hasEmbeddings()).isFalse();
		assertThat(map.lastNode().hasOutputProjection()).isTrue();
	}

	@Test
	void offline_nodes_are_excluded_from_plan() {
		List<NodeDescriptor> nodes = List.of(node("n1", 8, NodeStatus.OFFLINE), node("n2", 8, NodeStatus.READY));

		ShardMap map = planner.plan("model", 32, 100 * 1024 * 1024L, nodes);

		assertThat(map.assignments()).noneMatch(a -> a.nodeId().equals("n1"));
	}

	@Test
	void throws_when_no_nodes_available() {
		List<NodeDescriptor> nodes = List.of(node("n1", 4, NodeStatus.OFFLINE));

		assertThatThrownBy(() -> planner.plan("model", 32, 200 * 1024 * 1024L, nodes))
				.isInstanceOf(InsufficientClusterVramException.class);
	}

	@Test
	void throws_when_cluster_vram_insufficient() {
		// Node has 1GB usable, but model needs 32 * 500MB = 16GB
		List<NodeDescriptor> nodes = List.of(node("n1", 1, NodeStatus.READY));
		long vramPerLayer = 500 * 1024 * 1024L;

		assertThatThrownBy(() -> planner.plan("big-model", 32, vramPerLayer, nodes))
				.isInstanceOf(InsufficientClusterVramException.class).hasMessageContaining("big-model");
	}

	@Test
	void plan_coverage_is_always_valid() {
		List<NodeDescriptor> nodes = List.of(node("n1", 3, NodeStatus.READY), node("n2", 4, NodeStatus.READY),
				node("n3", 2, NodeStatus.IDLE));

		ShardMap map = planner.plan("model", 32, 100 * 1024 * 1024L, nodes);
		assertThatCode(map::validateCoverage).doesNotThrowAnyException();
	}
}
