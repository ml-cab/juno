package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * Ngram-simple speculative decoding must produce the exact same token sequence
 * as plain decoding for the same underlying "target model" — speculation may
 * change how many {@link InferencePipeline#forward}/{@link InferencePipeline#verifyDraft}
 * calls happen and in what groupings, but never which tokens get emitted
 * (PLAN-Infra-Tier9.md exit gate 1: "greedy + speculation == greedy without
 * speculation").
 */
class GenerationLoopSpeculativeDecodeTest {

	private SimpleTokenizer tokenizer;
	private Sampler sampler;
	private KVCacheManager kvCache;

	// "hi" as a single user message tokenizes to exactly 2 tokens under the
	// llama3 template + SimpleTokenizer (see GenerationLoopTest's stops_at_eos_token
	// comment for the same reasoning) -> decode step 0 calls forward/verifyDraft
	// with startPos == 1, predicting absolute position 2.
	private static final List<ChatMessage> TWO_TOKEN_PROMPT = List.of(ChatMessage.user("hi"));
	private static final int FIRST_DECODE_POSITION = 2;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		sampler = Sampler.create();
		kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
	}

	private GenerationLoop specLoop(InferencePipeline pipeline) {
		SpeculativeDecodeOptions spec = SpeculativeDecodeOptions.of(SpeculativeDecodeOptions.SpecType.NGRAM_SIMPLE, 1,
				2);
		return new GenerationLoop(tokenizer, sampler, pipeline, kvCache, PrefillMode.BATCHED,
				PrefillBatchOptions.DEFAULT_CHUNK_SIZE, spec);
	}

	private GenerationLoop disabledSpecLoop(InferencePipeline pipeline) {
		return new GenerationLoop(tokenizer, sampler, pipeline, kvCache, PrefillMode.BATCHED,
				PrefillBatchOptions.DEFAULT_CHUNK_SIZE, SpeculativeDecodeOptions.disabled());
	}

	private GenerationLoop plainLoop(InferencePipeline pipeline) {
		return new GenerationLoop(tokenizer, sampler, pipeline, kvCache);
	}

	private InferenceRequest requestWithMaxTokens(int maxTokens) {
		return InferenceRequest.of("llama3-8b", TWO_TOKEN_PROMPT, SamplingParams.defaults().withMaxTokens(maxTokens),
				RequestPriority.NORMAL);
	}

	/** Maps decode step index (0-based) -> scripted token, converting to absolute positions. */
	private static Map<Integer, Integer> scriptFromStep(int... tokensByStep) {
		Map<Integer, Integer> script = new HashMap<>();
		for (int step = 0; step < tokensByStep.length; step++) {
			script.put(FIRST_DECODE_POSITION + step, tokensByStep[step]);
		}
		return script;
	}

	@Test
	void spec_type_none_matches_plain_decode_exactly() {
		Map<Integer, Integer> script = scriptFromStep(500, 600, 500, 600, 500, 600, 500, 600);

		GenerationResult plain = plainLoop(new RecordingVerifyPipeline(script)).generate(requestWithMaxTokens(8),
				TokenConsumer.discard());
		GenerationResult disabled = disabledSpecLoop(new RecordingVerifyPipeline(script))
				.generate(requestWithMaxTokens(8), TokenConsumer.discard());

		assertThat(disabled.tokenIds()).isEqualTo(plain.tokenIds());
	}

	@Test
	void ngram_simple_matches_plain_decode_and_actually_drafts() {
		Map<Integer, Integer> script = scriptFromStep(500, 600, 500, 600, 500, 600, 500, 600);

		GenerationResult reference = plainLoop(new RecordingVerifyPipeline(script)).generate(requestWithMaxTokens(8),
				TokenConsumer.discard());

		RecordingVerifyPipeline specPipeline = new RecordingVerifyPipeline(script);
		GenerationResult speculative = specLoop(specPipeline).generate(requestWithMaxTokens(8),
				TokenConsumer.discard());

		// Token-identity invariant: same output regardless of drafting.
		assertThat(speculative.tokenIds()).isEqualTo(reference.tokenIds());
		assertThat(speculative.tokenIds()).containsExactly(500, 600, 500, 600, 500, 600, 500, 600);
		assertThat(speculative.generatedTokens()).isEqualTo(8);
		assertThat(speculative.stopReason()).isEqualTo(GenerationResult.StopReason.MAX_TOKENS);

		// The alternating 500/600 pattern repeats enough that the ngram-1 cache
		// picks it up mid-generation and drafts multiple tokens per round —
		// proves the batched-verify path (not just the plain fallback) actually
		// ran, matching the hand-traced round shape [2, 2, 1] for this exact script.
		assertThat(specPipeline.verifyDraftWindowSizes()).isEqualTo(List.of(2, 2, 1));
	}

	@Test
	void divergence_emits_target_prediction_not_the_stale_draft() {
		// Same alternating warm-up as above, but position 5 breaks the pattern
		// right where the ngram cache would otherwise draft another "600" — the
		// target model's own (scripted) prediction must be emitted instead of the
		// stale drafted token, and generation must continue correctly afterward.
		Map<Integer, Integer> script = scriptFromStep(500, 600, 500, /* diverge: */ 700, 800, 900);

		GenerationResult reference = plainLoop(new RecordingVerifyPipeline(script)).generate(requestWithMaxTokens(6),
				TokenConsumer.discard());

		RecordingVerifyPipeline specPipeline = new RecordingVerifyPipeline(script);
		GenerationResult speculative = specLoop(specPipeline).generate(requestWithMaxTokens(6),
				TokenConsumer.discard());

		assertThat(speculative.tokenIds()).isEqualTo(reference.tokenIds());
		assertThat(speculative.tokenIds()).containsExactly(500, 600, 500, 700, 800, 900);
		// Round 1 drafts [600, 500] (from the "500->600->500" warm-up); position 5's
		// real answer (700) diverges from the drafted 600, so only that round's
		// window size is observable here — the accept count itself is JFR-only.
		assertThat(specPipeline.verifyDraftWindowSizes()).isEqualTo(List.of(2));
	}
}
