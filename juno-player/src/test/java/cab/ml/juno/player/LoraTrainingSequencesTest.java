package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import cab.ml.juno.node.GgufReader;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

class LoraTrainingSequencesTest {

	private static Path modelPath() {
		Path p = Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		if (Files.exists(p))
			return p.toAbsolutePath();
		p = Path.of("/home/medion/Repo/juno/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		return Files.exists(p) ? p : null;
	}

	@Test
	void qaMaskTrainsAnswerTokensOnly() throws Exception {
		Path model = modelPath();
		assumeTrue(model != null, "need TinyLlama gguf");

		try (GgufReader reader = GgufReader.open(model)) {
			Tokenizer tok = GgufTokenizer.load(reader);
			var seq = LoraTrainingSequences.buildQa(tok, "What is my name?", "BOFA", "tinyllama");

			assertThat(seq.tokens().length).isGreaterThan(10);
			assertThat(seq.lossMask()).hasSize(seq.tokens().length - 1);
			assertThat(seq.predictionCount()).isGreaterThan(0);
			assertThat(seq.predictionCount()).isLessThan(seq.lossMask().length);

			// Answer token id(s) for "BOFA" must appear as supervised targets.
			int[] answerIds = LoraTrainingSequences.encodeNoBos(tok, "BOFA");
			assertThat(answerIds.length).isGreaterThan(0);
			boolean sawAnswerTarget = false;
			for (int i = 0; i < seq.lossMask().length; i++) {
				if (!seq.lossMask()[i])
					continue;
				int target = seq.tokens()[i + 1];
				for (int aid : answerIds) {
					if (target == aid)
						sawAnswerTarget = true;
				}
			}
			assertThat(sawAnswerTarget).as("masked loss should supervise answer token(s)").isTrue();

			var chunks = LoraTrainingSequences.chunk(seq, 32);
			assertThat(chunks).isNotEmpty();
			for (var c : chunks)
				assertThat(c.predictionCount()).isGreaterThan(0);
		}
	}
}
