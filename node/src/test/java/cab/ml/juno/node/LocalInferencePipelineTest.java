package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

class LocalInferencePipelineTest {

	private static final int VOCAB = 32000;
	private static final int HIDDEN_DIM = 4096;
	private static final int NUM_HEADS = 32;

	private ShardMap twoNodeMap() {
		return new ShardMap("llama3-8b", 32, List.of(new ShardAssignment("n1", "host1", 9091, 0, 16, true, false),
				new ShardAssignment("n2", "host2", 9091, 16, 32, false, true)), Instant.now());
	}

	@Test
	void single_handler_pipeline_returns_logits() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler(55);
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), handler, VOCAB, HIDDEN_DIM,
				NUM_HEADS);

		float[] logits = pipeline.forward("req-1", new int[] { 1, 2, 3 }, 0);

		assertThat(logits).hasSize(VOCAB);
		assertThat(logits[55]).isGreaterThan(0.0f);
	}

	@Test
	void pipeline_calls_each_stage_once_per_forward() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler();
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), handler, VOCAB, HIDDEN_DIM,
				NUM_HEADS);

		pipeline.forward("req-1", new int[] { 1, 2, 3 }, 0);

		// 2 nodes → handler called twice
		assertThat(handler.callCount()).isEqualTo(2);
	}

	@Test
	void pipeline_with_per_stage_handlers() {
		CyclicForwardPassHandler h1 = new CyclicForwardPassHandler();
		CyclicForwardPassHandler h2 = new CyclicForwardPassHandler(88);

		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), List.of(h1, h2), VOCAB, HIDDEN_DIM,
				NUM_HEADS);

		float[] logits = pipeline.forward("req-1", new int[] { 1, 2, 3 }, 0);

		assertThat(logits[88]).isGreaterThan(0.0f);
		assertThat(h1.callCount()).isEqualTo(1);
		assertThat(h2.callCount()).isEqualTo(1);
	}

	@Test
	void stage_count_matches_shard_map_node_count() {
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), new CyclicForwardPassHandler(),
				VOCAB, HIDDEN_DIM, NUM_HEADS);
		assertThat(pipeline.stageCount()).isEqualTo(2);
	}

	@Test
	void vocab_size_matches_context() {
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), new CyclicForwardPassHandler(),
				VOCAB, HIDDEN_DIM, NUM_HEADS);
		assertThat(pipeline.vocabSize()).isEqualTo(VOCAB);
	}

	@Test
	void rejects_handler_count_mismatch() {
		assertThatThrownBy(() -> LocalInferencePipeline.from(twoNodeMap(), List.of(new CyclicForwardPassHandler()),
				VOCAB, HIDDEN_DIM, NUM_HEADS)).isInstanceOf(IllegalArgumentException.class);
	}
}
