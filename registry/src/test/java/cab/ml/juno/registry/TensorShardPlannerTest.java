package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

class TensorShardPlannerTest {

    private static final int TOTAL_LAYERS = 22;
    private static final int NUM_HEADS    = 32;

    private final TensorShardPlanner planner = TensorShardPlanner.create();

    private NodeDescriptor node(String id, NodeStatus status) {
        long vram = 4L * 1024 * 1024 * 1024;
        return new NodeDescriptor(id, "192.168.1." + id.charAt(id.length() - 1),
                9091, vram, vram, status, 1.0, Instant.now(), Instant.now());
    }

    // ── happy-path ────────────────────────────────────────────────────────────

    @Test
    void plan_assigns_sequential_ranks_to_all_eligible_nodes() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.READY),
                node("n3", NodeStatus.READY));

        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        assertThat(result).hasSize(3);
        assertThat(result.get(0).tensorRank()).isEqualTo(0);
        assertThat(result.get(1).tensorRank()).isEqualTo(1);
        assertThat(result.get(2).tensorRank()).isEqualTo(2);
    }

    @Test
    void plan_all_nodes_get_full_layer_range() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.READY),
                node("n3", NodeStatus.READY));

        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        for (TensorShardAssignment a : result) {
            assertThat(a.startLayer()).isEqualTo(0);
            assertThat(a.endLayer()).isEqualTo(TOTAL_LAYERS);
            assertThat(a.layerCount()).isEqualTo(TOTAL_LAYERS);
        }
    }

    @Test
    void plan_all_nodes_have_embeddings_and_output_projection() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.READY),
                node("n3", NodeStatus.READY));

        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        for (TensorShardAssignment a : result) {
            assertThat(a.hasEmbeddings()).as("hasEmbeddings for " + a.nodeId()).isTrue();
            assertThat(a.hasOutputProjection()).as("hasOutputProjection for " + a.nodeId()).isTrue();
        }
    }

    @Test
    void plan_world_size_equals_eligible_node_count() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.READY));

        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        assertThat(result).hasSize(2);
        for (TensorShardAssignment a : result) {
            assertThat(a.tensorWorldSize()).isEqualTo(2);
        }
    }

    @Test
    void plan_skips_nodes_that_are_not_available() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.DEGRADED),   // excluded
                node("n3", NodeStatus.IDLE));       // included

        // 2 eligible nodes, 32 heads → 16 heads each — divisible
        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        assertThat(result).hasSize(2);
        assertThat(result.stream().map(TensorShardAssignment::tensorWorldSize)).allMatch(ws -> ws == 2);
    }

    @Test
    void plan_single_node_gets_rank_zero_world_size_one() {
        List<NodeDescriptor> nodes = List.of(node("n1", NodeStatus.READY));

        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        assertThat(result).hasSize(1);
        assertThat(result.get(0).tensorRank()).isEqualTo(0);
        assertThat(result.get(0).tensorWorldSize()).isEqualTo(1);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    @Test
    void plan_rejects_numHeads_not_divisible_by_node_count() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.READY),
                node("n2", NodeStatus.READY),
                node("n3", NodeStatus.READY));

        // 30 heads, 3 nodes → 10 each — this is fine (30 % 3 == 0)
        // Use 31 heads, 3 nodes → 31 % 3 != 0 → should throw
        assertThatThrownBy(() -> planner.plan("m", TOTAL_LAYERS, 31, nodes))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("numHeads")
                .hasMessageContaining("divisible");
    }

    @Test
    void plan_rejects_no_eligible_nodes() {
        List<NodeDescriptor> nodes = List.of(
                node("n1", NodeStatus.DEGRADED),
                node("n2", NodeStatus.DEGRADED));

        assertThatThrownBy(() -> planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes))
                .isInstanceOf(InsufficientClusterVramException.class);
    }

    @Test
    void plan_rejects_empty_node_list() {
        assertThatThrownBy(() -> planner.plan("m", TOTAL_LAYERS, NUM_HEADS, List.of()))
                .isInstanceOf(InsufficientClusterVramException.class);
    }

    @Test
    void plan_result_is_immutable() {
        List<NodeDescriptor> nodes = List.of(node("n1", NodeStatus.READY));
        List<TensorShardAssignment> result = planner.plan("m", TOTAL_LAYERS, NUM_HEADS, nodes);

        assertThatThrownBy(() -> result.add(result.get(0)))
                .isInstanceOf(UnsupportedOperationException.class);
    }
}