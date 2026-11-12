package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.kvcache.SharedPrefixOptions;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * A request whose prompt starts like a finished request's starts from the kept prompt's KV ({@link SharedPrefixPool})
 * on every path that admits requests (one request, a static batch, the continuous engine), and replies exactly as it
 * does when it computes the prompt itself. {@link KvTrackingPipeline}'s logits depend on every token its key holds,
 * so a wrong or missing shared position changes the reply, and a forward past an unwritten position is recorded.
 */
class SharedPrefixPoolTest {

	private static final String MODEL = "tinyllama";
	private static final String SYSTEM = "You are a careful assistant that answers briefly and precisely, checks every"
			+ " figure twice, states what it does not know, and never invents a source or a number it cannot show";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(4);

	private final String prevBudget = System.getProperty(SharedPrefixOptions.ENV);
	private SimpleTokenizer tokenizer;
	private KvTrackingPipeline pipeline;
	private GenerationLoop loop;
	private RequestScheduler continuous;

	@BeforeEach
	void setUp() {
		System.clearProperty(SharedPrefixOptions.ENV);
		tokenizer = new SimpleTokenizer();
		pipeline = new KvTrackingPipeline(true);
		loop = newLoop(pipeline);
	}

	@AfterEach
	void tearDown() {
		if (continuous != null)
			continuous.shutdown();
		if (prevBudget == null)
			System.clearProperty(SharedPrefixOptions.ENV);
		else
			System.setProperty(SharedPrefixOptions.ENV, prevBudget);
	}

	private GenerationLoop newLoop(KvTrackingPipeline p) {
		return new GenerationLoop(tokenizer, Sampler.create(), p,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private static List<ChatMessage> messages(String user) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user));
	}

	private static InferenceRequest stateless(String user) {
		return InferenceRequest.of(MODEL, messages(user), PARAMS, RequestPriority.NORMAL);
	}

	private int[] prompt(List<ChatMessage> messages) {
		return tokenizer.encode(ChatTemplateFormatter.forModelType(MODEL).format(messages));
	}

	private int commonPrefix(String userA, String userB) {
		int[] a = prompt(messages(userA));
		int[] b = prompt(messages(userB));
		int n = 0;
		while (n < Math.min(a.length, b.length) && a[n] == b[n])
			n++;
		return n;
	}

	/** The reply on a pipeline that shares nothing. */
	private List<Integer> unshared(List<ChatMessage> messages) {
		return newLoop(new KvTrackingPipeline())
				.generate(InferenceRequest.of(MODEL, messages, PARAMS, RequestPriority.NORMAL), TokenConsumer.discard())
				.tokenIds();
	}

	private GenerationResult viaContinuous(InferenceRequest request) throws Exception {
		if (continuous == null)
			continuous = new RequestScheduler(32, loop, BatchConfig.of(4, 20), ServeScheduleOptions.parse("continuous"));
		return continuous.submit(request, TokenConsumer.discard()).get(30, TimeUnit.SECONDS);
	}

	@Test
	void single_path_a_stateless_request_starts_from_an_earlier_requests_system_prompt() {
		assertThat(commonPrefix("hello", "what time is it")).isGreaterThan(SharedPrefixPool.MIN_SHARED_TOKENS);
		loop.generate(stateless("hello"), TokenConsumer.discard());
		InferenceRequest b = stateless("what time is it");
		int mark = pipeline.mark();

		GenerationResult result = loop.generate(b, TokenConsumer.discard());

		assertThat(pipeline.positionsWrittenSince(mark, b.requestId())).as("b computes only past the shared prefix")
				.first().isEqualTo(commonPrefix("hello", "what time is it"));
		assertThat(pipeline.underruns()).isEmpty();
		assertThat(result.tokenIds()).isEqualTo(unshared(messages("what time is it")));
		assertThat(loop.sharedPrefixes().hits()).isEqualTo(1);
		assertThat(loop.kvCache().prefixHits()).as("no request resumed from its own key").isZero();
	}

	@Test
	void single_path_the_same_prompt_again_shares_all_but_its_last_token_and_is_kept_once() {
		loop.generate(stateless("hello"), TokenConsumer.discard());
		InferenceRequest again = stateless("hello");
		int mark = pipeline.mark();

		GenerationResult result = loop.generate(again, TokenConsumer.discard());

		int len = prompt(messages("hello")).length;
		assertThat(pipeline.positionsWrittenSince(mark, again.requestId())).first().isEqualTo(len - 1);
		assertThat(result.tokenIds()).isEqualTo(unshared(messages("hello")));
		assertThat(loop.sharedPrefixes().heldPrompts()).isEqualTo(1);
		assertThat(loop.sharedPrefixes().heldTokens()).isEqualTo(len);
	}

	@Test
	void static_batch_requests_start_from_a_kept_prompt() {
		loop.generate(stateless("hello"), TokenConsumer.discard());
		InferenceRequest b = stateless("what time is it");
		InferenceRequest c = stateless("hello");
		int mark = pipeline.mark();

		List<GenerationResult> results = loop.generateBatch(List.of(new BatchEntry(b, TokenConsumer.discard()),
				new BatchEntry(c, TokenConsumer.discard())));

		assertThat(pipeline.positionsWrittenSince(mark, b.requestId())).first()
				.isEqualTo(commonPrefix("hello", "what time is it"));
		assertThat(pipeline.positionsWrittenSince(mark, c.requestId())).first()
				.isEqualTo(prompt(messages("hello")).length - 1);
		assertThat(pipeline.underruns()).isEmpty();
		assertThat(results.get(0).tokenIds()).isEqualTo(unshared(messages("what time is it")));
		assertThat(results.get(1).tokenIds()).isEqualTo(unshared(messages("hello")));
		assertThat(loop.sharedPrefixes().hits()).isEqualTo(2);
	}

	@Test
	void continuous_requests_start_from_a_kept_prompt() throws Exception {
		viaContinuous(stateless("hello"));
		InferenceRequest b = stateless("what time is it");
		int mark = pipeline.mark();

		GenerationResult result = viaContinuous(b);

		assertThat(pipeline.positionsWrittenSince(mark, b.requestId())).first()
				.isEqualTo(commonPrefix("hello", "what time is it"));
		assertThat(pipeline.underruns()).isEmpty();
		assertThat(result.tokenIds()).isEqualTo(unshared(messages("what time is it")));
		assertThat(loop.sharedPrefixes().hits()).isEqualTo(1);
	}

	@Test
	void a_second_session_opening_with_the_same_message_starts_from_the_first_sessions_kept_prompt() throws Exception {
		List<ChatMessage> turn2 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("hello"),
				ChatMessage.assistant("noted"), ChatMessage.user("and then"));
		int len = prompt(messages("hello")).length;
		for (boolean viaEngine : new boolean[] { false, true }) {
			String a = viaEngine ? "c-a" : "s-a";
			String b = viaEngine ? "c-b" : "s-b";
			InferenceRequest first = InferenceRequest.ofSession(a, MODEL, messages("hello"), PARAMS,
					RequestPriority.NORMAL);
			InferenceRequest second = InferenceRequest.ofSession(b, MODEL, messages("hello"), PARAMS,
					RequestPriority.NORMAL);
			InferenceRequest next = InferenceRequest.ofSession(b, MODEL, turn2, PARAMS, RequestPriority.NORMAL);
			if (viaEngine)
				viaContinuous(first);
			else
				loop.generate(first, TokenConsumer.discard());
			int mark = pipeline.mark();
			GenerationResult r = viaEngine ? viaContinuous(second) : loop.generate(second, TokenConsumer.discard());
			assertThat(pipeline.positionsWrittenSince(mark, b)).as(b + " shares " + a + "'s kept prompt").first()
					.isEqualTo(len - 1);
			assertThat(r.tokenIds()).isEqualTo(unshared(messages("hello")));

			mark = pipeline.mark();
			GenerationResult r2 = viaEngine ? viaContinuous(next) : loop.generate(next, TokenConsumer.discard());
			assertThat(pipeline.positionsWrittenSince(mark, b)).as(b + "'s next turn resumes from its own KV").first()
					.isEqualTo(len);
			assertThat(r2.tokenIds()).isEqualTo(unshared(turn2));
		}
		assertThat(pipeline.underruns()).isEmpty();
	}

	@Test
	void kept_prompts_beyond_the_budget_are_released_least_recently_used_first() {
		int len = prompt(messages("hello")).length;
		System.setProperty(SharedPrefixOptions.ENV, Integer.toString(len + 4)); // room for one prompt
		loop = newLoop(pipeline);
		loop.generate(stateless("hello"), TokenConsumer.discard());
		List<String> first = pipeline.keysStartingWith(SharedPrefixPool.KEY_PREFIX);
		assertThat(first).hasSize(1);

		loop.generate(stateless("what time is it"), TokenConsumer.discard());

		assertThat(pipeline.keysStartingWith(SharedPrefixPool.KEY_PREFIX)).as("the earlier prompt is released")
				.hasSize(1).doesNotContainAnyElementsOf(first);
		assertThat(loop.sharedPrefixes().heldTokens()).isLessThanOrEqualTo(len + 4);
		InferenceRequest again = stateless("what time is it");
		int mark = pipeline.mark();
		loop.generate(again, TokenConsumer.discard());
		assertThat(pipeline.positionsWrittenSince(mark, again.requestId())).as("the later prompt is the one kept")
				.first().isEqualTo(prompt(messages("what time is it")).length - 1);
	}

	@Test
	void a_budget_of_zero_turns_sharing_off() {
		System.setProperty(SharedPrefixOptions.ENV, "0");
		loop = newLoop(pipeline);
		loop.generate(stateless("hello"), TokenConsumer.discard());
		InferenceRequest b = stateless("hello");
		int mark = pipeline.mark();
		loop.generate(b, TokenConsumer.discard());

		assertThat(loop.sharedPrefixes()).isNull();
		assertThat(pipeline.positionsWrittenSince(mark, b.requestId())).first().isEqualTo(0);
		assertThat(pipeline.keysStartingWith(SharedPrefixPool.KEY_PREFIX)).isEmpty();
	}

	@Test
	void a_pipeline_that_cannot_share_is_off_by_default_and_refuses_an_explicit_budget() {
		assertThat(newLoop(new KvTrackingPipeline()).sharedPrefixes()).isNull();
		System.setProperty(SharedPrefixOptions.ENV, "4096");
		assertThatThrownBy(() -> newLoop(new KvTrackingPipeline())).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("--prefix-cache-tokens");
	}

	@Test
	void lora_playback_is_never_shared() {
		String prev = System.getProperty(GenerationLoop.LORA_PLAY_PROPERTY);
		System.setProperty(GenerationLoop.LORA_PLAY_PROPERTY, "/some/adapter.lora");
		try {
			assertThat(newLoop(pipeline).sharedPrefixes()).isNull();
		} finally {
			if (prev == null)
				System.clearProperty(GenerationLoop.LORA_PLAY_PROPERTY);
			else
				System.setProperty(GenerationLoop.LORA_PLAY_PROPERTY, prev);
		}
	}
}
