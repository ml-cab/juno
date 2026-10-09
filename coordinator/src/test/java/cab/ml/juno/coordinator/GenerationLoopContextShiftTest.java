package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * Context shift in the generation paths (single request, static batch,
 * continuous engine), against a pipeline that, like a real handler, refuses a
 * position at its context limit and compacts its KV on {@code shiftKv}.
 */
@DisplayName("GenerationLoop - context shift")
class GenerationLoopContextShiftTest {

	private static final int LIMIT = 24;
	private static final String MODEL = "llama3-8b";
	private static final List<ChatMessage> CHAT = List.of(ChatMessage.system("you are terse"),
			ChatMessage.user("hello there"));

	private final SimpleTokenizer tokenizer = new SimpleTokenizer();
	private final Sampler sampler = Sampler.create();
	private KVCacheManager kvCache;
	private String savedDefault;

	@BeforeEach
	void setUp() {
		kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
		savedDefault = System.getProperty(ContextShiftOptions.PROPERTY);
		System.clearProperty(ContextShiftOptions.PROPERTY);
	}

	@AfterEach
	void restore() {
		if (savedDefault == null)
			System.clearProperty(ContextShiftOptions.PROPERTY);
		else
			System.setProperty(ContextShiftOptions.PROPERTY, savedDefault);
	}

	/** A KV of token ids by position with a hard context limit, like a handler's. */
	static class LimitedPipeline implements InferencePipeline {
		static final int NEXT = 50;
		final int limit;
		final Map<String, List<Integer>> kv = new ConcurrentHashMap<>();
		final List<String> shifts = Collections.synchronizedList(new ArrayList<>());
		final List<String> errors = Collections.synchronizedList(new ArrayList<>());
		volatile int maxPosition = -1;

		LimitedPipeline(int limit) {
			this.limit = limit;
		}

		private void write(String id, int pos, int token) {
			if (pos >= limit)
				throw new IllegalStateException("KV cache position " + pos + " exceeds the context limit " + limit);
			List<Integer> row = kv.computeIfAbsent(id, k -> Collections.synchronizedList(new ArrayList<>()));
			if (row.size() < pos)
				errors.add(id + ": write at " + pos + " over " + row.size() + " written positions");
			while (row.size() > pos)
				row.remove(row.size() - 1);
			row.add(token);
			maxPosition = Math.max(maxPosition, pos);
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			write(requestId, startPos, tokens[tokens.length - 1]);
			float[] logits = new float[vocabSize()];
			logits[NEXT] = 100f;
			return logits;
		}

		@Override
		public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
			for (int p = 0; p < newTokens.length; p++)
				write(requestId, startPosition + p, newTokens[p]);
		}

		@Override
		public int vocabSize() {
			return 1000;
		}

		@Override
		public void evict(String requestId) {
			kv.remove(requestId);
		}

		@Override
		public int contextLimit() {
			return limit;
		}

		@Override
		public boolean supportsContextShift() {
			return true;
		}

		@Override
		public void shiftKv(String requestId, int seqLen, int keep, int discard) {
			List<Integer> row = kv.get(requestId);
			if (row == null || row.size() != seqLen)
				errors.add(requestId + ": shift of " + seqLen + " over " + (row == null ? 0 : row.size()));
			else
				row.subList(keep, keep + discard).clear();
			shifts.add(requestId + ":" + seqLen + ":" + keep + ":" + discard);
		}
	}

	private GenerationLoop loop(InferencePipeline p) {
		return new GenerationLoop(tokenizer, sampler, p, kvCache);
	}

	private static InferenceRequest request(List<ChatMessage> messages, int maxTokens, Boolean shift) {
		InferenceRequest r = InferenceRequest.of(MODEL, messages, SamplingParams.defaults().withMaxTokens(maxTokens),
				RequestPriority.NORMAL);
		return shift == null ? r : r.withContextShift(shift);
	}

	private int[] prompt(List<ChatMessage> messages) {
		return tokenizer.encode(ChatTemplateFormatter.forModelType(MODEL).format(messages));
	}

	/** Leading tokens the conversation shares with its system message and an empty user turn. */
	private int systemTokens() {
		int[] all = prompt(CHAT);
		int[] sys = prompt(List.of(CHAT.get(0), ChatMessage.user("")));
		int n = 0;
		while (n < all.length && n < sys.length && all[n] == sys[n])
			n++;
		return n;
	}

	private static String words(int n) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++)
			sb.append("w").append(i).append(' ');
		return sb.toString().strip();
	}

	@Test
	@DisplayName("without the opt-in, generation still fails at the context limit")
	void defaultStillFailsAtLimit() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		assertThatThrownBy(() -> loop(p).generate(request(CHAT, 3 * LIMIT, null), TokenConsumer.discard()))
				.isInstanceOf(IllegalStateException.class).hasMessageContaining("context limit");
		assertThat(p.shifts).isEmpty();
	}

	@Test
	@DisplayName("with the opt-in, the KV shifts instead: every position stays inside the limit, the system prompt is kept")
	void optInShiftsAndKeepsSystemPrompt() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		InferenceRequest req = request(CHAT, 3 * LIMIT, true);
		GenerationResult r = loop(p).generate(req, TokenConsumer.discard());

		assertThat(r.generatedTokens()).isEqualTo(3 * LIMIT);
		assertThat(p.errors).isEmpty();
		assertThat(p.maxPosition).isLessThan(LIMIT);
		assertThat(p.shifts).isNotEmpty().allSatisfy(s -> assertThat(s).contains(":" + systemTokens() + ":"));
		assertThat(p.kv).as("stateless request evicted at the end").doesNotContainKey(req.kvCacheKey());
	}

	@Test
	@DisplayName("a prompt longer than the limit is cut after the system prompt before prefill, only with the opt-in")
	void longPromptIsTruncatedOnlyWithOptIn() {
		List<ChatMessage> longChat = List.of(CHAT.get(0), ChatMessage.user(words(3 * LIMIT)));
		LimitedPipeline refused = new LimitedPipeline(LIMIT);
		assertThatThrownBy(() -> loop(refused).generate(request(longChat, 4, null), TokenConsumer.discard()))
				.isInstanceOf(IllegalStateException.class);

		LimitedPipeline p = new LimitedPipeline(LIMIT);
		GenerationResult r = loop(p).generate(request(longChat, 4, true), TokenConsumer.discard());
		assertThat(r.generatedTokens()).isEqualTo(4);
		assertThat(p.errors).isEmpty();
		assertThat(p.maxPosition).isLessThan(LIMIT);
	}

	@Test
	@DisplayName("the server default turns shifting on for a request that does not say")
	void serverDefaultApplies() {
		System.setProperty(ContextShiftOptions.PROPERTY, "on");
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		assertThat(loop(p).generate(request(CHAT, 2 * LIMIT, null), TokenConsumer.discard()).generatedTokens())
				.isEqualTo(2 * LIMIT);
		assertThat(p.shifts).isNotEmpty();

		LimitedPipeline q = new LimitedPipeline(LIMIT);
		assertThatThrownBy(() -> loop(q).generate(request(CHAT, 2 * LIMIT, false), TokenConsumer.discard()))
				.as("an explicit false overrides the server default").isInstanceOf(IllegalStateException.class);
	}

	@Test
	@DisplayName("static batch: only the member that reaches the limit shifts")
	void staticBatchShiftsOneMember() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		InferenceRequest shortReq = request(CHAT, 2, true);
		InferenceRequest longReq = request(CHAT, 3 * LIMIT, true);
		List<GenerationResult> results = loop(p).generateBatch(
				List.of(new BatchEntry(shortReq, TokenConsumer.discard()), new BatchEntry(longReq, TokenConsumer.discard())));
		assertThat(results.get(0).generatedTokens()).isEqualTo(2);
		assertThat(results.get(1).generatedTokens()).isEqualTo(3 * LIMIT);
		assertThat(p.errors).isEmpty();
		assertThat(p.maxPosition).isLessThan(LIMIT);
		assertThat(p.shifts).isNotEmpty().allSatisfy(s -> assertThat(s).startsWith(longReq.kvCacheKey() + ":"));
	}

	@Test
	@DisplayName("continuous engine: a slot shifts and finishes without disturbing the other slot")
	void continuousSlotShifts() throws Exception {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		ContinuousBatchEngine engine = new ContinuousBatchEngine(loop(p), 4, 0);
		engine.start();
		try {
			CompletableFuture<GenerationResult> a = new CompletableFuture<>();
			CompletableFuture<GenerationResult> b = new CompletableFuture<>();
			engine.submit(request(CHAT, 3 * LIMIT, true), TokenConsumer.discard(), a);
			engine.submit(request(CHAT, 3, true), TokenConsumer.discard(), b);
			assertThat(a.get(30, TimeUnit.SECONDS).generatedTokens()).isEqualTo(3 * LIMIT);
			assertThat(b.get(30, TimeUnit.SECONDS).generatedTokens()).isEqualTo(3);
		} finally {
			engine.shutdown();
		}
		assertThat(p.errors).isEmpty();
		assertThat(p.maxPosition).isLessThan(LIMIT);
		assertThat(p.shifts).isNotEmpty();
	}

	@Test
	@DisplayName("session: after a turn that shifted, the next turn starts from a fresh KV instead of a stale prefix")
	void sessionAfterShiftStartsFresh() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		GenerationLoop loop = loop(p);
		InferenceRequest turn1 = InferenceRequest.ofSession("sess", MODEL, CHAT,
				SamplingParams.defaults().withMaxTokens(2 * LIMIT), RequestPriority.NORMAL).withContextShift(true);
		loop.generate(turn1, TokenConsumer.discard());
		assertThat(p.shifts).isNotEmpty();
		assertThat(p.kv).as("shifted session KV dropped").doesNotContainKey("sess");

		List<ChatMessage> turn2Msgs = new ArrayList<>(CHAT);
		turn2Msgs.add(ChatMessage.assistant("ok"));
		turn2Msgs.add(ChatMessage.user("again"));
		InferenceRequest turn2 = InferenceRequest.ofSession("sess", MODEL, turn2Msgs,
				SamplingParams.defaults().withMaxTokens(2), RequestPriority.NORMAL).withContextShift(true);
		loop.generate(turn2, TokenConsumer.discard());
		assertThat(p.errors).as("turn 2 never continued from positions it had not written").isEmpty();
		loop.evictSession("sess");
	}

	@Test
	@DisplayName("fails closed before any forward: a pipeline without context shift")
	void unsupportedPipelineFailsClosed() {
		LimitedPipeline p = new LimitedPipeline(LIMIT) {
			@Override
			public boolean supportsContextShift() {
				return false;
			}
		};
		assertThatThrownBy(() -> loop(p).generate(request(CHAT, 4, true), TokenConsumer.discard()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("context shift");
		assertThat(p.kv).isEmpty();
	}

	@Test
	@DisplayName("fails closed before any forward: draft-model speculation")
	void draftModelSpeculationFailsClosed() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		GenerationLoop spec = new GenerationLoop(tokenizer, sampler, p, kvCache, PrefillMode.BATCHED,
				PrefillBatchOptions.DEFAULT_CHUNK_SIZE,
				SpeculativeDecodeOptions.of(SpeculativeDecodeOptions.SpecType.DRAFT_SIMPLE, 3, 4, "draft.gguf"),
				new LimitedPipeline(LIMIT));
		assertThatThrownBy(() -> spec.generate(request(CHAT, 4, true), TokenConsumer.discard()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("draft");
		assertThat(p.kv).isEmpty();
	}

	@Test
	@DisplayName("fails closed before any forward: a system prompt that leaves no room to shift")
	void systemPromptTooLongFailsClosed() {
		LimitedPipeline p = new LimitedPipeline(LIMIT);
		List<ChatMessage> chat = List.of(ChatMessage.system(words(LIMIT)), ChatMessage.user("hi"));
		assertThatThrownBy(() -> loop(p).generate(request(chat, 4, true), TokenConsumer.discard()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("system prompt");
		assertThat(p.kv).isEmpty();
	}
}
