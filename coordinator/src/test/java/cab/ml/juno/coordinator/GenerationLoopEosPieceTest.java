package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;

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
import cab.ml.juno.tokenizer.StubTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Verifies that EOS marker pieces ("</s>", "<|endoftext|>", etc.) never leak
 * into generated output text or reach the streaming consumer.
 */
@DisplayName("GenerationLoop — EOS piece suppression")
class GenerationLoopEosPieceTest {

	private StubTokenizer stubTokenizer;
	private Sampler sampler;
	private KVCacheManager kvCache;

	@BeforeEach
	void setUp() {
		stubTokenizer = new StubTokenizer();
		sampler = Sampler.create();
		kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
	}

	// ── Delegating tokenizer — StubTokenizer is final so we can't subclass it.
	// Use a delegation wrapper so tests can override decodeToken() per-token-ID.

	private static final class DelegatingTokenizer implements Tokenizer {
		private final StubTokenizer delegate;
		private final java.util.Map<Integer, String> overrides = new java.util.HashMap<>();

		DelegatingTokenizer(StubTokenizer delegate) {
			this.delegate = delegate;
		}

		void override(int id, String piece) {
			overrides.put(id, piece);
		}

		@Override
		public int[] encode(String text) {
			return delegate.encode(text);
		}

		@Override
		public String decode(int[] ids) {
			return delegate.decode(ids);
		}

		@Override
		public String decodeToken(int tokenId) {
			return overrides.getOrDefault(tokenId, delegate.decodeToken(tokenId));
		}

		@Override
		public int bosTokenId() {
			return delegate.bosTokenId();
		}

		@Override
		public int eosTokenId() {
			return delegate.eosTokenId();
		}

		@Override
		public int padTokenId() {
			return delegate.padTokenId();
		}

		@Override
		public int vocabSize() {
			return delegate.vocabSize();
		}

		@Override
		public String modelType() {
			return delegate.modelType();
		}

		@Override
		public boolean isReady() {
			return delegate.isReady();
		}
	}

	// ── helpers ───────────────────────────────────────────────────────────────

	private GenerationLoop loopWith(Tokenizer tok, InferencePipeline pipeline) {
		return new GenerationLoop(tok, sampler, pipeline, kvCache);
	}

	private InferenceRequest req(String prompt) {
		return InferenceRequest.of("llama3-8b", List.of(ChatMessage.user(prompt)),
				SamplingParams.defaults().withMaxTokens(10), RequestPriority.NORMAL);
	}

	// ── Test 1 ────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("EOS token ID stops generation without streaming any piece")
	void eos_token_id_stops_immediately_no_piece_streamed() {
		int eos = stubTokenizer.eosTokenId();

		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, // index 0 —
																										// prefill,
																										// discarded
				eos // index 1 — decode step 0
		);

		List<String> received = new ArrayList<>();
		GenerationResult result = loopWith(stubTokenizer, pipeline).generate(req("hi"),
				(piece, id, step) -> received.add(piece));

		assertThat(received).isEmpty();
		assertThat(result.text()).isEmpty();
		assertThat(result.generatedTokens()).isEqualTo(0);
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
	}

	// ── Test 2 — the actual regression ───────────────────────────────────────

	/**
	 * REGRESSION: a non-EOS token ID decodes to "</s>" — seen in GgufTokenizer when
	 * the vocab stores "</s>" both as a regular token and at the special EOS ID.
	 *
	 * FAILS before the isEosMarker() fix, PASSES after.
	 */
	@Test
	@DisplayName("Non-EOS token that decodes to \"</s>\" must not appear in output")
	void eos_string_piece_from_non_eos_token_suppressed() {
		int suspiciousToken = 100;
		assertThat(suspiciousToken).isNotEqualTo(stubTokenizer.eosTokenId());

		DelegatingTokenizer tok = new DelegatingTokenizer(stubTokenizer);
		tok.override(suspiciousToken, "</s>");

		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, // prefill
				suspiciousToken, // decode step 0 → "</s>"
				StubInferencePipeline.DEFAULT_TOKEN // step 1 — must never reach
		);

		List<String> streamed = new ArrayList<>();
		GenerationResult result = loopWith(tok, pipeline).generate(req("hi"), (piece, id, step) -> streamed.add(piece));

		assertThat(streamed).as("</s> must not be streamed").doesNotContain("</s>");
		assertThat(result.text()).as("</s> must not appear in text()").doesNotContain("</s>");
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
	}

	// ── Test 3 ────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("Non-EOS token that decodes to \"<|endoftext|>\" must not appear in output")
	void endoftext_string_piece_from_non_eos_token_suppressed() {
		int suspiciousToken = 101;
		assertThat(suspiciousToken).isNotEqualTo(stubTokenizer.eosTokenId());

		DelegatingTokenizer tok = new DelegatingTokenizer(stubTokenizer);
		tok.override(suspiciousToken, "<|endoftext|>");

		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, suspiciousToken,
				StubInferencePipeline.DEFAULT_TOKEN);

		List<String> streamed = new ArrayList<>();
		GenerationResult result = loopWith(tok, pipeline).generate(req("hi"), (piece, id, step) -> streamed.add(piece));

		assertThat(streamed).doesNotContain("<|endoftext|>");
		assertThat(result.text()).doesNotContain("<|endoftext|>");
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
	}

	// ── Test 4 ────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("Tokens with angle brackets that are not EOS markers pass through normally")
	void non_eos_angle_bracket_tokens_are_not_filtered() {
		int mathToken = 200;
		assertThat(mathToken).isNotEqualTo(stubTokenizer.eosTokenId());

		DelegatingTokenizer tok = new DelegatingTokenizer(stubTokenizer);
		tok.override(mathToken, "3<x<7");

		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, // prefill
				mathToken, // decode step 0 → "3<x<7"
				stubTokenizer.eosTokenId() // decode step 1 — stop
		);

		List<String> streamed = new ArrayList<>();
		loopWith(tok, pipeline).generate(req("hi"), (piece, id, step) -> streamed.add(piece));

		assertThat(streamed).contains("3<x<7");
	}
}