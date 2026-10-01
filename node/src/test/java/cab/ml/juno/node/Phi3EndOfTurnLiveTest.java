package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Phi-3.5-mini must end its turn: after "Hello" / "Hello! How can I help you
 * today?" the next token is {@code <|end|>} (32007). Rotating a short sequence
 * with the long-context RoPE factors put that probability near 0.50, so about
 * half of sampled replies ran on to the token limit; the short factors put it
 * near 0.99.
 */
class Phi3EndOfTurnLiveTest {

	/** User "Hello", then the assistant's "Hello! How can I help you today?"; no BOS. */
	private static final int[] IDS = { 32010, 29871, 13, 10994, 32007, 29871, 13, 32001, 29871, 13, 10994, 29991,
			1128, 508, 306, 1371, 366, 9826, 29973 };
	private static final int END_OF_TURN = 32007;

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
	void endOfTurnIsNearlyCertainAfterTheAnswer() throws Exception {
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(phiModelPath())) {
			LlamaConfig cfg = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
		Phi3TransformerHandler handler = Phi3TransformerHandler.load(phiModelPath(), ctx, CpuMatVec.INSTANCE);
		String reqId = "phi3-end-of-turn";

		float[] logits = null;
		for (int p = 0; p < IDS.length; p++)
			logits = handler.forward(ForwardRequest.withTokens(reqId, Arrays.copyOfRange(IDS, 0, p + 1), p), ctx)
					.logits();

		double max = Double.NEGATIVE_INFINITY;
		for (float l : logits)
			max = Math.max(max, l);
		double sum = 0;
		for (float l : logits)
			sum += Math.exp(l - max);
		double pEnd = Math.exp(logits[END_OF_TURN] - max) / sum;
		System.out.printf("P(<|end|>) after the answer = %.4f%n", pEnd);
		assertThat(pEnd).as("P(<|end|>) at the last position").isGreaterThanOrEqualTo(0.95);
	}
}
