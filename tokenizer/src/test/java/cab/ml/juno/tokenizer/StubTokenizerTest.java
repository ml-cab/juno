package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.tokenizer.StubTokenizer;

/**
 * Tests StubTokenizer AND validates the Tokenizer contract. Since DJLTokenizer
 * requires a model file, StubTokenizer is our test vehicle for anything that
 * depends on the Tokenizer interface.
 */
class StubTokenizerTest {

	private final StubTokenizer tokenizer = new StubTokenizer();

	@Test
	void encode_returns_non_empty_ids_for_text() {
		int[] ids = tokenizer.encode("hello world");
		assertThat(ids).hasSize(2);
	}

	@Test
	void encode_is_deterministic() {
		int[] first = tokenizer.encode("foo bar");
		int[] second = tokenizer.encode("foo bar");
		assertThat(first).isEqualTo(second);
	}

	@Test
	void decode_roundtrip_preserves_words() {
		int[] ids = tokenizer.encode("the quick brown fox");
		String decoded = tokenizer.decode(ids);
		assertThat(decoded).contains("the").contains("quick").contains("brown").contains("fox");
	}

	@Test
	void decodeToken_returns_empty_for_special_tokens() {
		assertThat(tokenizer.decodeToken(tokenizer.bosTokenId())).isEmpty();
		assertThat(tokenizer.decodeToken(tokenizer.eosTokenId())).isEmpty();
		assertThat(tokenizer.decodeToken(tokenizer.padTokenId())).isEmpty();
	}

	@Test
	void same_word_always_gets_same_id() {
		int id1 = tokenizer.encode("hello")[0];
		int id2 = tokenizer.encode("hello")[0];
		assertThat(id1).isEqualTo(id2);
	}

	@Test
	void different_words_get_different_ids() {
		int idHello = tokenizer.encode("hello")[0];
		int idWorld = tokenizer.encode("world")[0];
		assertThat(idHello).isNotEqualTo(idWorld);
	}

	@Test
	void special_token_ids_are_defined() {
		assertThat(tokenizer.bosTokenId()).isGreaterThanOrEqualTo(0);
		assertThat(tokenizer.eosTokenId()).isGreaterThanOrEqualTo(0);
		assertThat(tokenizer.padTokenId()).isGreaterThanOrEqualTo(0);
	}

	@Test
	void isReady_returns_true() {
		assertThat(tokenizer.isReady()).isTrue();
	}

	@Test
	void encode_empty_string_returns_empty_array() {
		assertThat(tokenizer.encode("")).isEmpty();
		assertThat(tokenizer.encode(null)).isEmpty();
	}
}
