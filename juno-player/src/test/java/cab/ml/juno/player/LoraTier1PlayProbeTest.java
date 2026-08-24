package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.node.GgufReader;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Regression: training must use the same BOS policy as inference encode().
 * Prepending an extra BOS (when add_bos_token=true) poisons adapters into emitting
 * chat-template tokens as "answers".
 */
class LoraTier1PlayProbeTest {

	private static Path modelPath() {
		String prop = System.getProperty("juno.test.model");
		if (prop != null && !prop.isBlank() && Files.exists(Path.of(prop)))
			return Path.of(prop);
		Path p = Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		if (Files.exists(p))
			return p.toAbsolutePath();
		p = Path.of("/home/medion/Repo/juno/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		return Files.exists(p) ? p : null;
	}

	@Test
	void trainTokenizationMatchesInferenceEncodeWithoutDoubleBos() throws Exception {
		Path model = modelPath();
		assumeTrue(model != null, "need TinyLlama gguf");

		try (GgufReader reader = GgufReader.open(model)) {
			Tokenizer tokenizer = GgufTokenizer.load(reader);
			String key = ChatModelType.fromPath(model.toString());
			String text = ChatTrainingFormats.qaTurn("What is my name?", "Futhus", key)
					+ ChatTrainingFormats.qaTurn("what is my name?", "Futhus", key);
			int[] tokens = tokenizer.encode(text);

			assertThat(tokens.length).isGreaterThan(2);
			assertThat(tokens[0]).as("encode() already supplies BOS when add_bos=true")
					.isEqualTo(tokenizer.bosTokenId());
			assertThat(tokens[1]).as("must not double-prepend BOS").isNotEqualTo(tokenizer.bosTokenId());

			// Same path LoraTrainer.trainRawText uses after the fix.
			List<int[]> chunks = LoraTrainer.chunkTokens(tokens, 32);
			assertThat(chunks).isNotEmpty();
			assertThat(chunks.get(0)[0]).isEqualTo(tokenizer.bosTokenId());
			assertThat(chunks.get(0)[1]).isNotEqualTo(tokenizer.bosTokenId());
		}
	}
}
