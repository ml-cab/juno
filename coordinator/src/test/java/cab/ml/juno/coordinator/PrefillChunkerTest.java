package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.node.InferencePipeline;

class PrefillChunkerTest {

	@Test
	void batched_call_count_splits_remainder() {
		assertThat(PrefillChunker.batchedCallCount(0, 101, 32)).isEqualTo(4); // 100 tokens / 32
		assertThat(PrefillChunker.batchedCallCount(0, 65, 32)).isEqualTo(2); // 64 prefill tokens
		assertThat(PrefillChunker.batchedCallCount(0, 33, 32)).isEqualTo(1);
		assertThat(PrefillChunker.batchedCallCount(10, 15, 32)).isEqualTo(1); // 4 tokens
	}

	@Test
	void batched_mode_chunks_prefill_calls() {
		RecordingPipeline pipeline = new RecordingPipeline();
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

		PrefillChunker.run(pipeline, PrefillMode.BATCHED, 3, "req", prompt, 0);

		assertThat(pipeline.prefillCalls).hasSize(3);
		assertThat(pipeline.prefillCalls.get(0).tokens).containsExactly(1, 2, 3);
		assertThat(pipeline.prefillCalls.get(0).startPos).isZero();
		assertThat(pipeline.prefillCalls.get(1).tokens).containsExactly(4, 5, 6);
		assertThat(pipeline.prefillCalls.get(1).startPos).isEqualTo(3);
		assertThat(pipeline.prefillCalls.get(2).tokens).containsExactly(7, 8, 9);
		assertThat(pipeline.prefillCalls.get(2).startPos).isEqualTo(6);
	}

	@Test
	void single_mode_uses_forward_per_position() {
		RecordingPipeline pipeline = new RecordingPipeline();
		int[] prompt = { 10, 20, 30, 40 };

		PrefillChunker.run(pipeline, PrefillMode.SINGLE, 32, "req", prompt, 0);

		assertThat(pipeline.prefillCalls).isEmpty();
		assertThat(pipeline.forwardPositions).containsExactly(0, 1, 2);
	}

	private static final class RecordingPipeline implements InferencePipeline {
		final List<PrefillCall> prefillCalls = new ArrayList<>();
		final List<Integer> forwardPositions = new ArrayList<>();

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			forwardPositions.add(startPos);
			return new float[16];
		}

		@Override
		public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
			prefillCalls.add(new PrefillCall(newTokens, startPosition));
		}

		@Override
		public int vocabSize() {
			return 16;
		}
	}

	private record PrefillCall(int[] tokens, int startPos) {
	}
}
