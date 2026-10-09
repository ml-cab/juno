package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * A request that fails mid-generation must release the KV it holds, as a
 * finished one does: before this, a stateless request that failed (for example at
 * the context limit) kept its KV for the life of the process.
 */
@DisplayName("Generation - a failed request releases its KV")
class GenerationFailureEvictionTest {

	private static final int LIMIT = 24;
	private static final List<ChatMessage> CHAT = List.of(ChatMessage.user("hello there"));

	private static GenerationLoop loop(GenerationLoopContextShiftTest.LimitedPipeline p) {
		return new GenerationLoop(new SimpleTokenizer(), Sampler.create(), p,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(100)));
	}

	private static InferenceRequest request() {
		return InferenceRequest.of("llama3-8b", CHAT, SamplingParams.defaults().withMaxTokens(3 * LIMIT),
				RequestPriority.NORMAL);
	}

	@Test
	@DisplayName("single request: KV released after the failure")
	void singleRequest() {
		var p = new GenerationLoopContextShiftTest.LimitedPipeline(LIMIT);
		InferenceRequest r = request();
		assertThatThrownBy(() -> loop(p).generate(r, TokenConsumer.discard())).isInstanceOf(IllegalStateException.class);
		assertThat(p.kv).doesNotContainKey(r.kvCacheKey());
	}

	@Test
	@DisplayName("static batch: every member's KV released after the failure")
	void staticBatch() {
		var p = new GenerationLoopContextShiftTest.LimitedPipeline(LIMIT);
		InferenceRequest a = request();
		InferenceRequest b = request();
		assertThatThrownBy(() -> loop(p).generateBatch(
				List.of(new BatchEntry(a, TokenConsumer.discard()), new BatchEntry(b, TokenConsumer.discard()))))
				.isInstanceOf(IllegalStateException.class);
		assertThat(p.kv).isEmpty();
	}

	@Test
	@DisplayName("continuous engine: a failed slot's KV released")
	void continuousSlot() throws Exception {
		var p = new GenerationLoopContextShiftTest.LimitedPipeline(LIMIT);
		ContinuousBatchEngine engine = new ContinuousBatchEngine(loop(p), 4, 0);
		engine.start();
		try {
			CompletableFuture<GenerationResult> f = new CompletableFuture<>();
			engine.submit(request(), TokenConsumer.discard(), f);
			assertThatThrownBy(() -> f.get(30, TimeUnit.SECONDS)).isInstanceOf(ExecutionException.class);
		} finally {
			engine.shutdown();
		}
		assertThat(p.kv).isEmpty();
	}
}
