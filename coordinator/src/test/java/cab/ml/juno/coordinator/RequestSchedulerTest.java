package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.StubTokenizer;

class RequestSchedulerTest {

	private RequestScheduler scheduler;

	@BeforeEach
	void setUp() {
		var loop = new GenerationLoop(new StubTokenizer(), Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		scheduler = new RequestScheduler(10, loop);
	}

	private InferenceRequest req(RequestPriority priority) {
		return InferenceRequest.of("model", List.of(ChatMessage.user("hi")), SamplingParams.defaults().withMaxTokens(2),
				priority);
	}

	@Test
	void submit_and_wait_returns_result() {
		GenerationResult result = scheduler.submitAndWait(req(RequestPriority.NORMAL));
		assertThat(result).isNotNull();
		assertThat(result.requestId()).isNotBlank();
	}

	@Test
	void async_submit_calls_consumer() throws InterruptedException {
		CountDownLatch latch = new CountDownLatch(1);
		TokenConsumer consumer = (piece, tokenId, pos) -> {
			if (pos == 0)
				latch.countDown();
		};

		scheduler.submit(req(RequestPriority.HIGH), consumer);

		assertThat(latch.await(5, TimeUnit.SECONDS)).isTrue();
	}

	@Test
	void throws_queue_full_exception_when_saturated() {
		// Fill a tiny scheduler
		var tinyLoop = new GenerationLoop(new StubTokenizer(), Sampler.create(),
				// slow pipeline — blocks long enough to fill queue
				new InferencePipeline() {
					@Override
					public float[] forward(String requestId, int[] tokens, int startPos) {
						try {
							Thread.sleep(200);
						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
						}
						return new float[1000];
					}

					@Override
					public int vocabSize() {
						return 1000;
					}
				}, new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		var tinyScheduler = new RequestScheduler(1, tinyLoop);

		// First request fills the queue
		tinyScheduler.submit(req(RequestPriority.LOW), TokenConsumer.discard());

		// Second should be rejected
		assertThatThrownBy(() -> tinyScheduler.submit(req(RequestPriority.LOW), TokenConsumer.discard()))
				.isInstanceOf(RequestScheduler.QueueFullException.class)
				.satisfies(e -> assertThat(((RequestScheduler.QueueFullException) e).retryAfterSeconds())
						.isGreaterThan(0));
	}

	@Test
	void queue_full_exception_has_positive_retry_after() {
		var ex = new RequestScheduler.QueueFullException("full", 5);
		assertThat(ex.retryAfterSeconds()).isEqualTo(5);
	}
}
