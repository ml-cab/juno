package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * End-to-end greedy decode on the real Phi-3.5-mini GGUF — compares against
 * llama.cpp reference for the standard Hello chat prompt.
 */
class Phi3GreedyDecodeIntegrationTest {

	private static final int[] PROMPT_IDS = { 32010, 29871, 13, 10994, 32007, 29871, 13, 32001, 29871, 13 };
	/** llama.cpp greedy continuation after the Hello chat prompt (9 tokens). */
	private static final int[] EXPECTED_LLAMA_IDS = { 15043, 29991, 1128, 508, 306, 6985, 366, 9826, 29973 };
	/** Juno may pick 10994 (prompt "▁Hello") or llama's 15043 (assistant "Hello") at step 0. */
	private static final int HELLO_PROMPT_ID = 10994;
	private static final int HELLO_ASSISTANT_ID = 15043;

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
	void greedyHelloResponse_matchesLlamaCpp() throws Exception {
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(phiModelPath())) {
			LlamaConfig cfg = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}

		Phi3TransformerHandler handler = Phi3TransformerHandler.load(phiModelPath(), ctx, CpuMatVec.INSTANCE);
		String reqId = "phi3-greedy";

		for (int p = 0; p < PROMPT_IDS.length - 1; p++) {
			int[] slice = Arrays.copyOfRange(PROMPT_IDS, 0, p + 1);
			handler.forward(ForwardRequest.withTokens(reqId, slice, p), ctx);
		}

		int[] ids = PROMPT_IDS.clone();
		int pos = PROMPT_IDS.length - 1;
		List<Integer> generated = new ArrayList<>();

		for (int step = 0; step < EXPECTED_LLAMA_IDS.length; step++) {
			float[] logits = handler.forward(ForwardRequest.withTokens(reqId, ids, pos), ctx).logits();
			int tok = argmax(logits);
			generated.add(tok);
			ids = Arrays.copyOf(ids, ids.length + 1);
			ids[ids.length - 1] = tok;
			pos++;
		}

		assertThat(generated.get(0)).as("first token is a Hello variant")
				.isIn(HELLO_PROMPT_ID, HELLO_ASSISTANT_ID);
		assertThat(generated.subList(1, generated.size()))
				.as("tokens after Hello must match llama.cpp exactly")
				.containsExactlyElementsOf(Arrays.stream(EXPECTED_LLAMA_IDS).skip(1).boxed().toList());
	}

	private static int argmax(float[] logits) {
		int best = 0;
		for (int i = 1; i < logits.length; i++) {
			if (logits[i] > logits[best])
				best = i;
		}
		return best;
	}

}
