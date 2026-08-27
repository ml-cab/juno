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

	// ── evict(): must reach every stage's handler ───────────────────────────────
	//
	// GenerationLoop calls pipeline.evict(requestId) after every stateless
	// request to release each handler's in-process KV cache arrays (see
	// ForwardPassHandler.evict javadoc — without this, they leak for the life
	// of the process). These tests are the regression net for that cascade.

	@Test
	void evict_reaches_every_stage_handler() {
		CyclicForwardPassHandler h1 = new CyclicForwardPassHandler();
		CyclicForwardPassHandler h2 = new CyclicForwardPassHandler();
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), List.of(h1, h2), VOCAB,
				HIDDEN_DIM, NUM_HEADS);

		pipeline.evict("req-1");

		assertThat(h1.wasEvicted("req-1")).isTrue();
		assertThat(h2.wasEvicted("req-1")).isTrue();
	}

	@Test
	void evict_only_affects_the_given_requestId() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler();
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), handler, VOCAB, HIDDEN_DIM,
				NUM_HEADS);

		pipeline.evict("req-1");

		assertThat(handler.wasEvicted("req-1")).isTrue();
		assertThat(handler.wasEvicted("req-2")).isFalse();
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

	@Test
	void embedLastToken_returns_hidden_vector_from_final_stage() {
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), new CyclicForwardPassHandler(),
				VOCAB, HIDDEN_DIM, NUM_HEADS);

		float[] emb = pipeline.embedLastToken("embed-req", new int[] { 10, 20, 30 });

		assertThat(emb).hasSize(HIDDEN_DIM);
		assertThat(emb[0]).isEqualTo(0.0f);
		assertThat(emb[1]).isEqualTo(0.01f);
	}

	// ── Handler-list snapshot timing (root cause of the 2026-07-12 vision hang) ─
	//
	// LocalInferencePipeline.from(shardMap, handlers, ...) reads handlers.get(i)
	// once, at construction time, and stores that exact reference into each
	// NodeStage — it never re-reads the list afterwards. ConsoleMain used to call
	// LlavaHandlerFactory.buildFromHandlers(..., handlers, ...) (which replaces
	// handlers.get(0) with a vision-aware wrapper) AFTER building the pipeline
	// from a defensive copy of that same list, so the wrap silently never took
	// effect and vision requests were routed straight to the plain text handler.
	// These tests pin down the exact ordering contract the fix relies on.

	@Test
	void mutating_handlers_list_after_pipeline_construction_has_no_effect() {
		List<ForwardPassHandler> handlers = new java.util.ArrayList<>(
				List.of(new CyclicForwardPassHandler(11), new CyclicForwardPassHandler(22)));

		// Mirrors the buggy ConsoleMain ordering: build from a defensive copy first...
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), new java.util.ArrayList<>(handlers),
				VOCAB, HIDDEN_DIM, NUM_HEADS);

		// ...then swap in a "vision-aware" stand-in for stage 0, too late.
		CyclicForwardPassHandler lateSwap = new CyclicForwardPassHandler(99);
		handlers.set(0, lateSwap);

		pipeline.forward("req-late-swap", new int[] { 1, 2, 3 }, 0);

		// The pipeline never saw the swap: the "vision-aware" stand-in was never invoked.
		assertThat(lateSwap.callCount()).isEqualTo(0);
	}

	@Test
	void mutating_handlers_list_before_pipeline_construction_is_picked_up() {
		List<ForwardPassHandler> handlers = new java.util.ArrayList<>(
				List.of(new CyclicForwardPassHandler(11), new CyclicForwardPassHandler(22)));

		// Fixed ConsoleMain ordering: wrap/swap stage 0 first...
		CyclicForwardPassHandler earlySwap = new CyclicForwardPassHandler(99);
		handlers.set(0, earlySwap);

		// ...then build the pipeline, which now captures the swapped handler.
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(twoNodeMap(), new java.util.ArrayList<>(handlers),
				VOCAB, HIDDEN_DIM, NUM_HEADS);

		pipeline.forward("req-early-swap", new int[] { 1, 2, 3 }, 0);

		assertThat(earlySwap.callCount()).isEqualTo(1);
	}
}