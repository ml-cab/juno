package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * Static micro-batching ({@link GenerationLoop#generateBatch}) keys each
 * request's pipeline KV by its own request id and evicts it when the request
 * finishes. A prefix-cache hit therefore can never be honoured on that path: the
 * matched positions were written under some other key (or under a key that has
 * since been evicted), so skipping prefill would continue from KV that this
 * request never wrote.
 *
 * <p>
 * A hit needs a previously cached prompt to be a full prefix of the new prompt
 * (only trie leaves carry a cache key), so the trigger is a repeated or
 * extended prompt in a later batch, not merely a shared system prompt. Within one
 * batch every lookup happens before any write, so the second round below is
 * required.
 *
 * <p>
 * The documented contract this guards is: prefix KV reuse is session-scoped, and
 * requests that do not share a session never skip prefill.
 */
class StaticBatchPrefixCacheSessionGatingTest {

	private static final String MODEL = "tinyllama";
	private static final String SYSTEM = "You are a careful assistant that answers briefly and precisely";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(4);

	private SimpleTokenizer tokenizer;
	private KvTrackingPipeline pipeline;
	private GenerationLoop loop;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		pipeline = new KvTrackingPipeline();
		loop = newLoop(tokenizer, pipeline);
	}

	private static GenerationLoop newLoop(SimpleTokenizer tokenizer, KvTrackingPipeline pipeline) {
		return new GenerationLoop(tokenizer, Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private static List<ChatMessage> messages(String user) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user));
	}

	private static BatchEntry stateless(String user) {
		return new BatchEntry(InferenceRequest.of(MODEL, messages(user), PARAMS, RequestPriority.NORMAL),
				TokenConsumer.discard());
	}

	private static InferenceRequest session(String sessionId, List<ChatMessage> messages) {
		return InferenceRequest.ofSession(sessionId, MODEL, messages, PARAMS, RequestPriority.NORMAL);
	}

	private int[] promptIds(List<ChatMessage> messages) {
		return tokenizer.encode(ChatTemplateFormatter.forModelType(MODEL).format(messages));
	}

	/**
	 * Output of the same prompt on an isolated pipeline and cache. The tokenizer is
	 * shared with the loop under test because its ids depend on encounter order;
	 * its encoding is idempotent, so sharing it does not carry any cache state.
	 */
	private List<Integer> reference(List<ChatMessage> messages) {
		GenerationLoop fresh = newLoop(tokenizer, new KvTrackingPipeline());
		return fresh.generate(InferenceRequest.of(MODEL, messages, PARAMS, RequestPriority.NORMAL),
				TokenConsumer.discard()).tokenIds();
	}

	private static boolean startsWith(int[] full, int[] prefix) {
		if (prefix.length > full.length)
			return false;
		for (int i = 0; i < prefix.length; i++)
			if (full[i] != prefix[i])
				return false;
		return true;
	}

	@Test
	void repeated_prompt_in_a_later_static_batch_is_fully_prefilled() {
		loop.generateBatch(List.of(stateless("first question"), stateless("second question")));
		assertThat(pipeline.underruns()).isEmpty();

		BatchEntry repeated = stateless("first question");
		BatchEntry other = stateless("third question");
		List<GenerationResult> results = loop.generateBatch(List.of(repeated, other));

		assertThat(pipeline.underruns()).as("a batched request continued from KV it never wrote").isEmpty();
		assertThat(results.get(0).tokenIds()).isEqualTo(reference(messages("first question")));
		assertThat(results.get(1).tokenIds()).isEqualTo(reference(messages("third question")));
	}

	@Test
	void shared_system_prompt_with_distinct_user_turns_is_fully_prefilled() {
		loop.generateBatch(List.of(stateless("alpha"), stateless("beta")));
		List<GenerationResult> results = loop.generateBatch(List.of(stateless("gamma"), stateless("delta")));

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(results.get(0).tokenIds()).isEqualTo(reference(messages("gamma")));
		assertThat(results.get(1).tokenIds()).isEqualTo(reference(messages("delta")));
	}

	@Test
	void stateless_batched_prompt_does_not_seed_a_hit_for_a_later_session_request() {
		loop.generateBatch(List.of(stateless("shared question"), stateless("unrelated question")));
		assertThat(pipeline.underruns()).isEmpty();

		GenerationResult result = loop.generate(session("s-late", messages("shared question")),
				TokenConsumer.discard());

		assertThat(pipeline.underruns())
				.as("the session request resumed from a trie entry left by a stateless batched request").isEmpty();
		assertThat(result.tokenIds()).isEqualTo(reference(messages("shared question")));
	}

	@Test
	void session_kv_is_not_reused_by_a_stateless_batched_request() {
		loop.generate(session("s-owner", messages("shared question")), TokenConsumer.discard());
		assertThat(pipeline.underruns()).isEmpty();

		List<GenerationResult> results = loop
				.generateBatch(List.of(stateless("shared question"), stateless("other question")));

		assertThat(pipeline.underruns())
				.as("a stateless batched request resumed from KV that lives under a session key").isEmpty();
		assertThat(results.get(0).tokenIds()).isEqualTo(reference(messages("shared question")));
	}

	@Test
	void session_request_inside_a_static_batch_is_fully_prefilled() {
		List<ChatMessage> turn1 = messages("first turn");
		loop.generate(session("s-batched", turn1), TokenConsumer.discard());
		assertThat(pipeline.underruns()).isEmpty();

		// Turn 2 extends turn 1's prompt, so the session's own trie leaf is a full
		// prefix of it. Inside a batch this request's KV is keyed by its request id,
		// not by the session id, so the hit must not be honoured there either.
		List<ChatMessage> turn2 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("first turn"),
				ChatMessage.assistant("noted"), ChatMessage.user("second turn"));
		assertThat(startsWith(promptIds(turn2), promptIds(turn1)))
				.as("test precondition: turn 2 must extend turn 1's prompt or this test proves nothing").isTrue();

		BatchEntry sessionTurn2 = new BatchEntry(session("s-batched", turn2), TokenConsumer.discard());
		List<GenerationResult> results = loop.generateBatch(List.of(sessionTurn2, stateless("bystander")));

		assertThat(pipeline.underruns()).as("a batched session request continued from KV it never wrote").isEmpty();
		assertThat(results.get(0).tokenIds()).isEqualTo(reference(turn2));
	}

	@Test
	void session_reuse_on_the_single_request_path_still_skips_the_cached_prefix() {
		List<ChatMessage> turn1 = messages("first turn");
		loop.generate(session("s-single", turn1), TokenConsumer.discard());
		int cachedPrefixLen = promptIds(turn1).length;

		List<ChatMessage> turn2 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("first turn"),
				ChatMessage.assistant("noted"), ChatMessage.user("second turn"));
		assertThat(startsWith(promptIds(turn2), promptIds(turn1))).isTrue();

		int mark = pipeline.mark();
		GenerationResult result = loop.generate(session("s-single", turn2), TokenConsumer.discard());

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(pipeline.positionsWrittenSince(mark, "s-single")).first().isEqualTo(cachedPrefixLen);
		assertThat(result.tokenIds()).isEqualTo(reference(turn2));
	}
}
