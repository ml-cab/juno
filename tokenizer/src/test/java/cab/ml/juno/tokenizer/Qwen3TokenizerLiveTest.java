package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

/** Live tokenizer checks for Qwen3 MoE GGUF (enable_thinking=false prompt). */
class Qwen3TokenizerLiveTest {

	private static final String IM_END = "<|" + "im_end|>";
	private static final Path MODEL = Path.of(System.getProperty("user.dir")).getParent() != null
			&& Path.of(System.getProperty("user.dir")).endsWith("tokenizer")
					? Path.of(System.getProperty("user.dir")).getParent()
							.resolve("models/qwen3-moe-6x0.6b-3.6b-writing-on-fire-uncensored-q8_0.gguf")
					: Path.of("models/qwen3-moe-6x0.6b-3.6b-writing-on-fire-uncensored-q8_0.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_qwen3_template_matches_no_think_reference() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			String prompt = ChatTemplate.qwen3().format(List.of(ChatMessage.user("Hello")));
			int[] ids = tok.encode(prompt);
			// user + assistant opener + empty think block (151667,271,151668,271)
			assertThat(ids).containsExactly(151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198, 151667, 271,
					151668, 271);
			assertThat(tok.decodeToken(151667)).isEmpty();
			assertThat(tok.decodeToken(151668)).isEmpty();
		}
	}
}
