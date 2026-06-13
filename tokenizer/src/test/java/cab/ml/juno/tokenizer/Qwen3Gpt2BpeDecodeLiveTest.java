package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

/** GPT-2 BPE Chinese encode/decode roundtrip on Qwen3 MoE GGUF. */
class Qwen3Gpt2BpeDecodeLiveTest {

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
	void chinese_roundtrip_and_known_token() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			GgufTokenizer tok = GgufTokenizer.load(r);
			String text = "你好，用户可以随时告诉我需要什么帮助";
			int[] ids = tok.encode(text);
			assertThat(ids).isNotEmpty();
			assertThat(ids[0]).isNotEqualTo(0);
			assertThat(tok.decode(ids)).isEqualTo(text);

			// token 99692 = "好的" in this vocab (verified via byte decode)
			assertThat(tok.decodeToken(99692)).isEqualTo("好的");

			Tokenizer.StreamContext stream = tok.openStreamContext();
			StringBuilder out = new StringBuilder();
			for (int id : ids)
				out.append(stream.append(id));
			out.append(stream.flush());
			assertThat(out.toString()).isEqualTo(text);
		}
	}
}
