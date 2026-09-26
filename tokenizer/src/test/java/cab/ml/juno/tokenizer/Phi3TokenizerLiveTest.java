package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

/**
 * Live checks for a Phi-3 GGUF's turn-boundary tokens.
 *
 * <p>
 * Phi-3 stores its role headers as control tokens, and a control token decodes
 * to the empty string because it is prompt scaffolding rather than content. A
 * model that emits one has started writing the next speaker's turn, but nothing
 * reaches a filter watching the decoded text — so the boundary has to be
 * recognised by token id.
 */
class Phi3TokenizerLiveTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("tokenizer")
			? Path.of(System.getProperty("user.dir")).getParent()
					.resolve("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
			: Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void role_headers_are_turn_boundaries_by_id_though_they_decode_to_nothing() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			// Control tokens: prompt scaffolding, so decode renders them as nothing.
			assertThat(tok.decodeToken(32010)).as("<|user|>").isEmpty();
			assertThat(tok.decodeToken(32001)).as("<|assistant|>").isEmpty();
			assertThat(tok.decodeToken(32006)).as("<|system|>").isEmpty();
			// Which is exactly why generation has to recognise them by id.
			assertThat(tok.chatTurnTokenIds()).contains(32010, 32001, 32006, 32007);
			// <|end|> does reach the decoded text, and stops generation there too.
			assertThat(tok.decodeToken(32007)).isEqualTo("<|end|>");
		}
	}
}
