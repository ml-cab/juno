package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import cab.ml.juno.node.InferencePipeline;

/**
 * {@link DraftModelSession} drives its own {@link InferencePipeline} exactly
 * as a plain (non-speculative) decode loop would — one {@code forward} call
 * per position, feeding its own greedy pick back in — then reconciles that
 * tentative continuation against ground truth in {@link DraftModelSession#observe}.
 * These tests exercise that reconciliation directly, independent of
 * {@link GenerationLoop}'s own draft/verify wiring (covered by
 * {@code GenerationLoopSpeculativeDecodeTest}).
 */
class DraftModelSessionTest {

	/** Counts {@link InferencePipeline#forward} calls on top of {@link RecordingVerifyPipeline}. */
	private static final class CountingPipeline implements InferencePipeline {
		private final RecordingVerifyPipeline delegate;
		int forwardCalls = 0;

		CountingPipeline(Map<Integer, Integer> scriptByPosition) {
			this.delegate = new RecordingVerifyPipeline(scriptByPosition);
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			forwardCalls++;
			return delegate.forward(requestId, tokens, startPos);
		}

		@Override
		public int vocabSize() {
			return delegate.vocabSize();
		}
	}

	/** Maps absolute position -> scripted greedy token at that position. */
	private static Map<Integer, Integer> script(int... tokensByPosition) {
		Map<Integer, Integer> m = new HashMap<>();
		for (int p = 0; p < tokensByPosition.length; p++)
			m.put(p, tokensByPosition[p]);
		return m;
	}

	@Test
	void first_propose_primes_from_context_and_returns_greedy_continuation() {
		// allTokens = [1, 2, 3] (positions 0,1,2 already committed). Draft model's
		// own greedy pick for position 3 is scripted as 30, position 4 as 40.
		Map<Integer, Integer> s = script(1, 2, 3, 30, 40);
		CountingPipeline pipeline = new CountingPipeline(s);
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		int[] draft = session.propose(new int[] { 1, 2, 3 }, 2);

		assertThat(draft).containsExactly(30, 40);
	}

	@Test
	void full_acceptance_needs_no_resync_forward_call() {
		Map<Integer, Integer> s = script(1, 2, 3, 30, 40);
		CountingPipeline pipeline = new CountingPipeline(s);
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		session.propose(new int[] { 1, 2, 3 }, 2);
		int callsAfterPropose = pipeline.forwardCalls;

		// Ground truth matches the draft exactly for both positions.
		session.observe(new int[] { 1, 2, 3, 30, 40 });

		assertThat(pipeline.forwardCalls).isEqualTo(callsAfterPropose);
	}

	@Test
	void divergence_triggers_exactly_one_resync_forward_call() {
		Map<Integer, Integer> s = script(1, 2, 3, 30, 40);
		CountingPipeline pipeline = new CountingPipeline(s);
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		session.propose(new int[] { 1, 2, 3 }, 2);
		int callsAfterPropose = pipeline.forwardCalls;

		// Ground truth accepted position 3's draft (30) but overrode position 4
		// with 99 instead of the drafted 40.
		session.observe(new int[] { 1, 2, 3, 30, 99 });

		assertThat(pipeline.forwardCalls).isEqualTo(callsAfterPropose + 1);
	}

	@Test
	void propose_after_resync_continues_from_the_corrected_position() {
		// The script is purely a function of the position being PREDICTED, not of
		// which token was actually fed as input — so index 4 (40) is what the draft
		// model itself proposes for position 4 (a forward call with startPos=3),
		// while index 5 (50) is what ANY forward call with startPos=4 predicts for
		// position 5, whether that call is the resync inside observe() or the next
		// propose() round — both must agree with the model, only the fed-in input
		// token differs (the draft's own stale 40 vs. the real overridden 99).
		Map<Integer, Integer> s = script(1, 2, 3, 30, 40, 50);
		CountingPipeline pipeline = new CountingPipeline(s);
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		int[] firstDraft = session.propose(new int[] { 1, 2, 3 }, 2); // drafts [30, 40] for positions 3,4
		assertThat(firstDraft).containsExactly(30, 40);

		session.observe(new int[] { 1, 2, 3, 30, 99 }); // position 4 overridden to 99, not the drafted 40

		int[] nextDraft = session.propose(new int[] { 1, 2, 3, 30, 99 }, 1);

		assertThat(nextDraft).containsExactly(50);
	}

	@Test
	void empty_propose_round_is_resynced_on_the_next_observe() {
		// Simulates GenerationLoop's plain-fallback branch: no propose() call at
		// all for a round, but the target still emits a real token that the draft
		// session must catch up on before its next propose().
		Map<Integer, Integer> s = script(1, 2, 3, 88, 77);
		CountingPipeline pipeline = new CountingPipeline(s);
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		session.propose(new int[] { 1, 2 }, 1); // primes on [1,2], drafts position 2 -> scripted 3
		session.observe(new int[] { 1, 2, 3 }); // fully accepted, draftPos now at position 2

		// A plain-fallback round emits 88 at position 3 without ever calling propose().
		session.observe(new int[] { 1, 2, 3, 88 });

		int[] nextDraft = session.propose(new int[] { 1, 2, 3, 88 }, 1);

		assertThat(nextDraft).containsExactly(77);
	}

	@Test
	void close_before_priming_does_not_touch_the_pipeline() {
		CountingPipeline pipeline = new CountingPipeline(script());
		DraftModelSession session = new DraftModelSession(pipeline, "req#draft");

		session.close();

		assertThat(pipeline.forwardCalls).isZero();
	}
}
