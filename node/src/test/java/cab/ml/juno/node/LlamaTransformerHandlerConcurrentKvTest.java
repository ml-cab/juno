package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Regression: concurrent forward passes with distinct requestIds must not throw
 * {@link java.util.ConcurrentModificationException} from the in-process KV map.
 */
@DisplayName("LlamaTransformerHandler — concurrent KV cache")
class LlamaTransformerHandlerConcurrentKvTest {

	private static final int VOCAB_SIZE = 256;
	private static final int HIDDEN_DIM = 32;
	private static final int NUM_HEADS = 4;
	private static final int NUM_KV_HEADS = 4;
	private static final int NUM_LAYERS = 2;

	@Test
	@DisplayName("nine concurrent requestIds complete forward passes without map corruption")
	void concurrentDistinctRequestIds() throws Exception {
		LlamaTransformerHandler handler = LlamaTransformerHandler.newTestInstance(
				VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS,
				NUM_LAYERS, 0, NUM_LAYERS, true, true, null);
		ShardContext ctx = new ShardContext("node-1", 0, NUM_LAYERS, true, true, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

		int sessions = 9;
		int steps = 5;
		ExecutorService pool = Executors.newFixedThreadPool(sessions);
		try {
			List<Future<?>> futures = new ArrayList<>(sessions);
			for (int sid = 0; sid < sessions; sid++) {
				final int sessionId = sid;
				futures.add(pool.submit(() -> {
					for (int step = 0; step < steps; step++) {
						ForwardRequest req = ForwardRequest.withTokens(
								"req-" + sessionId, new int[] { 1 + sessionId }, step);
						ForwardResult result = handler.forward(req, ctx);
						assertThat(result.logits()).hasSize(VOCAB_SIZE);
					}
				}));
			}
			for (Future<?> future : futures) {
				future.get(60, TimeUnit.SECONDS);
			}
		} finally {
			pool.shutdownNow();
		}
	}
}
