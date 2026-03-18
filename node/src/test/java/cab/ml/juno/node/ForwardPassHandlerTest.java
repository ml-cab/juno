package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ForwardPassHandlerTest {

	private ShardContext intermediateCtx() {
		return new ShardContext("n1", 0, 16, true, false, 32000, 4096, 32);
	}

	private ShardContext lastNodeCtx() {
		return new ShardContext("n2", 16, 32, false, true, 32000, 4096, 32);
	}

	@Test
	void intermediate_node_returns_activations_of_hidden_dim_size() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler();
		ForwardRequest req = ForwardRequest.withTokens("req-1", new int[] { 1, 2, 3 }, 0);

		ForwardResult result = handler.forward(req, intermediateCtx());

		assertThat(result.isFinalNode()).isFalse();
		assertThat(result.activations()).hasSize(4096); // hiddenDim
		assertThat(result.logits()).isNull();
	}

	@Test
	void last_node_returns_logits_of_vocab_size() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler(99);
		ForwardRequest req = ForwardRequest.withActivations("req-1", new float[4096], 0);

		ForwardResult result = handler.forward(req, lastNodeCtx());

		assertThat(result.isFinalNode()).isTrue();
		assertThat(result.logits()).hasSize(32000); // vocabSize
		assertThat(result.activations()).isNull();
	}

	@Test
	void winner_token_has_highest_logit() {
		int winner = 777;
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler(winner);
		ForwardRequest req = ForwardRequest.withActivations("req-1", new float[4096], 0);

		ForwardResult result = handler.forward(req, lastNodeCtx());

		float[] logits = result.logits();
		assertThat(logits[winner]).isGreaterThan(0.0f);
		for (int i = 0; i < logits.length; i++) {
			if (i != winner)
				assertThat(logits[i]).isEqualTo(0.0f);
		}
	}

	@Test
	void call_count_increments_per_forward() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler();
		ForwardRequest req = ForwardRequest.withTokens("req-1", new int[] { 1 }, 0);

		handler.forward(req, intermediateCtx());
		handler.forward(req, intermediateCtx());
		handler.forward(req, intermediateCtx());

		assertThat(handler.callCount()).isEqualTo(3);
	}

	@Test
	void is_ready_returns_true() {
		assertThat(new CyclicForwardPassHandler().isReady()).isTrue();
	}
}
