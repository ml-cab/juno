package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

class GgufTokenizerBosTest {

	private static boolean phiModelPresent() {
		return Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists()
				|| Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists();
	}

	private static Path phiModelPath() {
		Path p = Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
		return p.toFile().exists() ? p : Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
	}

	@Test
	@EnabledIf("phiModelPresent")
	void phi3_prompt_doesNotPrependBos_whenAddBosTokenFalse() throws Exception {
		String prompt = "<|user|>\nHello<|end|>\n<|assistant|>\n";
		int[] expected = { 32010, 29871, 13, 10994, 32007, 29871, 13, 32001, 29871, 13 };

		try (GgufReader r = GgufReader.open(phiModelPath())) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			int[] ids = tok.encode(prompt);
			assertThat(ids).as("must match llama.cpp tokenization (no leading BOS)").containsExactly(expected);
			assertThat(ids[0]).isNotEqualTo(tok.bosTokenId());
			assertThat(tok.decodeToken(32007)).isEqualTo("<|end|>");
		}
	}
}
