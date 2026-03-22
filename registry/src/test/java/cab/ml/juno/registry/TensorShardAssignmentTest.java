package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class TensorShardAssignmentTest {

    @Test
    void valid_assignment_constructs_correctly() {
        TensorShardAssignment a = new TensorShardAssignment(
                "node-1", "192.168.1.10", 9091, 0, 22, true, true, 0, 3);

        assertThat(a.nodeId()).isEqualTo("node-1");
        assertThat(a.host()).isEqualTo("192.168.1.10");
        assertThat(a.grpcPort()).isEqualTo(9091);
        assertThat(a.startLayer()).isEqualTo(0);
        assertThat(a.endLayer()).isEqualTo(22);
        assertThat(a.hasEmbeddings()).isTrue();
        assertThat(a.hasOutputProjection()).isTrue();
        assertThat(a.tensorRank()).isEqualTo(0);
        assertThat(a.tensorWorldSize()).isEqualTo(3);
    }

    @Test
    void layer_count_equals_endLayer_minus_startLayer() {
        TensorShardAssignment a = new TensorShardAssignment(
                "node-2", "host", 9091, 0, 22, true, true, 1, 3);

        assertThat(a.layerCount()).isEqualTo(22);
    }

    @Test
    void grpc_target_combines_host_and_port() {
        TensorShardAssignment a = new TensorShardAssignment(
                "node-3", "192.168.1.12", 9092, 0, 22, true, true, 2, 3);

        assertThat(a.grpcTarget()).isEqualTo("192.168.1.12:9092");
    }

    @Test
    void rejects_rank_equal_to_world_size() {
        assertThatThrownBy(() ->
                new TensorShardAssignment("n", "h", 9091, 0, 22, true, true, 3, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tensorRank")
                .hasMessageContaining("tensorWorldSize");
    }

    @Test
    void rejects_rank_greater_than_world_size() {
        assertThatThrownBy(() ->
                new TensorShardAssignment("n", "h", 9091, 0, 22, true, true, 5, 3))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejects_negative_tensor_rank() {
        assertThatThrownBy(() ->
                new TensorShardAssignment("n", "h", 9091, 0, 22, true, true, -1, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tensorRank");
    }

    @Test
    void rejects_zero_world_size() {
        assertThatThrownBy(() ->
                new TensorShardAssignment("n", "h", 9091, 0, 22, true, true, 0, 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tensorWorldSize");
    }

    @Test
    void rejects_blank_node_id() {
        assertThatThrownBy(() ->
                new TensorShardAssignment("", "h", 9091, 0, 22, true, true, 0, 1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("nodeId");
    }

    @Test
    void all_three_ranks_in_world_of_three_are_valid() {
        for (int rank = 0; rank < 3; rank++) {
            final int r = rank;
            TensorShardAssignment a = new TensorShardAssignment(
                    "node-" + rank, "host", 9091, 0, 22, true, true, r, 3);
            assertThat(a.tensorRank()).isEqualTo(r);
        }
    }
}