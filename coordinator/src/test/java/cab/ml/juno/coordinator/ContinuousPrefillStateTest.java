package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ContinuousPrefillStateTest {

	@Test
	void empty_when_prompt_already_warm() {
		int[] prompt = { 1, 2, 3 };
		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 2);
		assertThat(state.isComplete()).isTrue();
		assertThat(state.remainingTokens()).isZero();
		assertThat(state.nextChunk(32).isEmpty()).isTrue();
	}

	@Test
	void chunk_boundaries_and_remainder() {
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // 9 prefill tokens
		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 0);

		ContinuousPrefillState.Chunk c1 = state.nextChunk(3);
		assertThat(c1.tokens()).containsExactly(1, 2, 3);
		assertThat(c1.startPos()).isZero();
		state.advance(c1.tokenCount());

		ContinuousPrefillState.Chunk c2 = state.nextChunk(3);
		assertThat(c2.tokens()).containsExactly(4, 5, 6);
		assertThat(c2.startPos()).isEqualTo(3);
		state.advance(c2.tokenCount());

		ContinuousPrefillState.Chunk c3 = state.nextChunk(3);
		assertThat(c3.tokens()).containsExactly(7, 8, 9);
		assertThat(c3.startPos()).isEqualTo(6);
		state.advance(c3.tokenCount());

		assertThat(state.isComplete()).isTrue();
		assertThat(state.nextChunk(3).isEmpty()).isTrue();
	}

	@Test
	void chunk_size_one_matches_per_token() {
		int[] prompt = { 10, 20, 30, 40 };
		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 0);

		assertThat(state.nextChunk(1).tokens()).containsExactly(10);
		state.advance(1);
		assertThat(state.nextChunk(1).tokens()).containsExactly(20);
		state.advance(1);
		assertThat(state.nextChunk(1).tokens()).containsExactly(30);
		state.advance(1);
		assertThat(state.isComplete()).isTrue();
	}

	@Test
	void prefix_hit_starts_mid_prompt() {
		int[] prompt = { 1, 2, 3, 4, 5, 6 };
		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 2);

		ContinuousPrefillState.Chunk c = state.nextChunk(32);
		assertThat(c.tokens()).containsExactly(3, 4, 5);
		assertThat(c.startPos()).isEqualTo(2);
		state.advance(c.tokenCount());
		assertThat(state.isComplete()).isTrue();
	}

	@Test
	void remaining_tokens_tracks_progress() {
		int[] prompt = { 1, 2, 3, 4, 5 };
		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 0);
		assertThat(state.remainingTokens()).isEqualTo(4);
		state.advance(2);
		assertThat(state.remainingTokens()).isEqualTo(2);
		state.advance(2);
		assertThat(state.remainingTokens()).isZero();
	}

	@Test
	void windows_match_prefill_chunker_batched_path() {
		RecordingPipeline viaChunker = new RecordingPipeline();
		RecordingPipeline viaState = new RecordingPipeline();
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
		int chunk = 4;

		PrefillChunker.run(viaChunker, PrefillMode.BATCHED, chunk, "req", prompt, 0);

		ContinuousPrefillState state = ContinuousPrefillState.start(prompt, 0);
		while (!state.isComplete()) {
			ContinuousPrefillState.Chunk c = state.nextChunk(chunk);
			viaState.prefillBatch("req", c.tokens(), c.startPos());
			state.advance(c.tokenCount());
		}

		assertThat(viaState.prefillCalls).hasSize(viaChunker.prefillCalls.size());
		for (int i = 0; i < viaChunker.prefillCalls.size(); i++) {
			assertThat(viaState.prefillCalls.get(i).tokens)
					.containsExactly(viaChunker.prefillCalls.get(i).tokens);
			assertThat(viaState.prefillCalls.get(i).startPos)
					.isEqualTo(viaChunker.prefillCalls.get(i).startPos);
		}
	}

	private static final class RecordingPipeline implements cab.ml.juno.node.InferencePipeline {
		final java.util.List<PrefillCall> prefillCalls = new java.util.ArrayList<>();

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
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
