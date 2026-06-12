package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

class Qwen2TokenizerLiveTest {

	private static final String IM_END = "<|" + "im_end|>";
	private static final Path MODEL = Path.of(System.getProperty("user.dir")).getParent() != null
			&& Path.of(System.getProperty("user.dir")).endsWith("tokenizer")
					? Path.of(System.getProperty("user.dir")).getParent().resolve("models/qwen2.5-3b-instruct-q4_k_m.gguf")
					: Path.of("models/qwen2.5-3b-instruct-q4_k_m.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_chatml_prompt_matches_llama_cpp_reference() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			String prompt = "<|im_start|>user\nhello" + IM_END + "\n<|im_start|>assistant\n";
			int[] ids = tok.encode(prompt);
			assertThat(ids).containsExactly(151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198);
			assertThat(tok.eosTokenId()).isEqualTo(151645);
			assertThat(tok.decodeToken(151645)).isEqualTo(IM_END);
		}
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_word_alone_matches_llama_cpp() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			// llama.cpp: 14990 hello, 198 newline
			assertThat(tok.encode("hello\n")).containsExactly(14990, 198);
		}
	}
}
