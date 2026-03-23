package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.TensorShardAssignment;

class TensorShardContextTest {

	private static final int VOCAB_SIZE = 32_000;
	private static final int HIDDEN_DIM = 2_048;
	private static final int NUM_HEADS = 32;
	private static final int TOTAL_LAYERS = 22;

	private TensorShardContext ctx(int rank, int worldSize) {
		return new TensorShardContext("node-" + rank, 0, TOTAL_LAYERS, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, rank,
				worldSize);
	}

	// ── geometry ──────────────────────────────────────────────────────────────

	@Test
	void headsPerNode_divides_numHeads_evenly_by_worldSize() {
		// 32 heads / 4 nodes = 8 heads each
		TensorShardContext c = ctx(0, 4);
		assertThat(c.headsPerNode()).isEqualTo(8);
	}

	@Test
	void headStart_and_headEnd_are_contiguous_across_all_ranks() {
		int worldSize = 4;
		int prevEnd = 0;
		for (int rank = 0; rank < worldSize; rank++) {
			TensorShardContext c = ctx(rank, worldSize);
			assertThat(c.headStart()).isEqualTo(prevEnd);
			assertThat(c.headEnd()).isEqualTo(prevEnd + c.headsPerNode());
			prevEnd = c.headEnd();
		}
		assertThat(prevEnd).isEqualTo(NUM_HEADS);
	}

	@Test
	void last_rank_headEnd_equals_numHeads() {
		int worldSize = 4;
		TensorShardContext last = ctx(worldSize - 1, worldSize);
		assertThat(last.headEnd()).isEqualTo(NUM_HEADS);
	}

	@Test
	void headDim_equals_hiddenDim_divided_by_numHeads() {
		TensorShardContext c = ctx(0, 2);
		// 2048 / 32 = 64
		assertThat(c.headDim()).isEqualTo(64);
	}

	@Test
	void sliceDim_equals_hiddenDim_divided_by_worldSize() {
		// 3 nodes: 2048 / 3 is not integer — but sliceDim uses integer division
		// Use 4 nodes for clean division: 2048 / 4 = 512
		TensorShardContext c = ctx(0, 4);
		assertThat(c.sliceDim()).isEqualTo(512);
	}

	@Test
	void layerCount_equals_endLayer_minus_startLayer() {
		TensorShardContext c = ctx(1, 3);
		assertThat(c.layerCount()).isEqualTo(TOTAL_LAYERS);
	}

	// ── factory ───────────────────────────────────────────────────────────────

	@Test
	void from_TensorShardAssignment_populates_all_fields() {
		TensorShardAssignment assignment = new TensorShardAssignment("node-2", "192.168.1.2", 9091, 0, TOTAL_LAYERS,
				true, true, 2, 4);

		TensorShardContext c = TensorShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

		assertThat(c.nodeId()).isEqualTo("node-2");
		assertThat(c.startLayer()).isEqualTo(0);
		assertThat(c.endLayer()).isEqualTo(TOTAL_LAYERS);
		assertThat(c.tensorRank()).isEqualTo(2);
		assertThat(c.tensorWorldSize()).isEqualTo(4);
		assertThat(c.vocabSize()).isEqualTo(VOCAB_SIZE);
		assertThat(c.hiddenDim()).isEqualTo(HIDDEN_DIM);
		assertThat(c.numHeads()).isEqualTo(NUM_HEADS);
	}

	// ── validation ────────────────────────────────────────────────────────────

	@Test
	void rejects_numHeads_not_divisible_by_worldSize() {
		// 31 heads, 3 nodes: 31 % 3 != 0
		assertThatThrownBy(() -> new TensorShardContext("n", 0, TOTAL_LAYERS, VOCAB_SIZE, HIDDEN_DIM, 31, 0, 3))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("numHeads");
	}

	@Test
	void rejects_rank_equal_to_worldSize() {
		assertThatThrownBy(() -> new TensorShardContext("n", 0, TOTAL_LAYERS, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, 3, 3))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tensorRank");
	}

	@Test
	void rejects_negative_rank() {
		assertThatThrownBy(() -> new TensorShardContext("n", 0, TOTAL_LAYERS, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, -1, 3))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_zero_world_size() {
		assertThatThrownBy(() -> new TensorShardContext("n", 0, TOTAL_LAYERS, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, 0, 0))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tensorWorldSize");
	}
}