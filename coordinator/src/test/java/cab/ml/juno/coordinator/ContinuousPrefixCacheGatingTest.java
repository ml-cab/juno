package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The continuous engine reads the prefix trie only for session requests, and a
 * stateless slot evicts its pipeline KV when it retires. The trie must therefore
 * never be populated on behalf of a stateless slot: such an entry dangles as soon
 * as it is written, and a later session request with the same prompt would match
 * it and skip prefill against KV it never wrote.
 */
class ContinuousPrefixCacheGatingTest {

	private static final String MODEL = "tinyllama";
	private static final String SYSTEM = "You are a careful assistant that answers briefly and precisely";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(4);

	private SimpleTokenizer tokenizer;
	private KvTrackingPipeline pipeline;
	private RequestScheduler scheduler;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		pipeline = new KvTrackingPipeline();
		scheduler = newScheduler(pipeline);
	}

	@AfterEach
	void tearDown() {
		scheduler.shutdown();
	}

	private RequestScheduler newScheduler(KvTrackingPipeline p) {
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), p,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		return new RequestScheduler(32, loop, BatchConfig.of(4, 20), ServeScheduleOptions.parse("continuous"));
	}

	private static List<ChatMessage> messages(String user) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user));
	}

	private GenerationResult run(InferenceRequest request) throws Exception {
		return scheduler.submit(request, TokenConsumer.discard()).get(30, TimeUnit.SECONDS);
	}

	private List<Integer> reference(List<ChatMessage> messages) throws Exception {
		KvTrackingPipeline isolated = new KvTrackingPipeline();
		RequestScheduler fresh = newScheduler(isolated);
		try {
			return fresh.submit(InferenceRequest.of(MODEL, messages, PARAMS, RequestPriority.NORMAL),
					TokenConsumer.discard()).get(30, TimeUnit.SECONDS).tokenIds();
		} finally {
			fresh.shutdown();
		}
	}

	@Test
	void stateless_slot_does_not_seed_a_hit_for_a_later_session_request() throws Exception {
		run(InferenceRequest.of(MODEL, messages("shared question"), PARAMS, RequestPriority.NORMAL));
		assertThat(pipeline.underruns()).isEmpty();

		GenerationResult result = run(InferenceRequest.ofSession("s-late", MODEL, messages("shared question"),
				PARAMS, RequestPriority.NORMAL));

		assertThat(pipeline.underruns())
				.as("the session request resumed from a trie entry left by a stateless continuous slot").isEmpty();
		assertThat(result.tokenIds()).isEqualTo(reference(messages("shared question")));
	}

	@Test
	void session_reuse_across_turns_still_skips_the_cached_prefix() throws Exception {
		List<ChatMessage> turn1 = messages("first turn");
		run(InferenceRequest.ofSession("s-cont", MODEL, turn1, PARAMS, RequestPriority.NORMAL));
		int cachedPrefixLen = tokenizer.encode(ChatTemplateFormatter.forModelType(MODEL).format(turn1)).length;

		List<ChatMessage> turn2 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("first turn"),
				ChatMessage.assistant("noted"), ChatMessage.user("second turn"));
		int mark = pipeline.mark();
		GenerationResult result = run(
				InferenceRequest.ofSession("s-cont", MODEL, turn2, PARAMS, RequestPriority.NORMAL));

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(pipeline.positionsWrittenSince(mark, "s-cont")).first().isEqualTo(cachedPrefixLen);
		assertThat(result.tokenIds()).isEqualTo(reference(turn2));
	}
}
