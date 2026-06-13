package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Greedy decode on a real Qwen3-MoE GGUF when present under {@code models/}.
 */
class Qwen3MoeGreedyDecodeIntegrationTest {

	private static boolean modelPresent() {
		return Path.of("models/qwen3-30b-a3b-q4_k_m.gguf").toFile().exists()
				|| Path.of("../models/qwen3-30b-a3b-q4_k_m.gguf").toFile().exists();
	}

	private static Path modelPath() {
		Path p = Path.of("models/qwen3-30b-a3b-q4_k_m.gguf");
		return p.toFile().exists() ? p : Path.of("../models/qwen3-30b-a3b-q4_k_m.gguf");
	}

	@Test
	@EnabledIf("modelPresent")
	void greedyHelloResponse_producesFiniteTokens() throws Exception {
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(modelPath())) {
			Qwen3Config cfg = Qwen3Config.from(r);
			ctx = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}

		Qwen3MoeTransformerHandler handler = Qwen3MoeTransformerHandler.load(modelPath(), ctx, CpuMatVec.INSTANCE);
		String reqId = "qwen3moe-greedy";

		int[] prompt = { 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
				9707, 0, 151645, 198, 151644, 77091, 198 };
		for (int p = 0; p < prompt.length - 1; p++) {
			handler.forward(ForwardRequest.withTokens(reqId, Arrays.copyOfRange(prompt, 0, p + 1), p), ctx);
		}

		List<Integer> generated = new ArrayList<>();
		int[] ids = prompt.clone();
		int pos = prompt.length - 1;
		for (int step = 0; step < 4; step++) {
			float[] logits = handler.forward(ForwardRequest.withTokens(reqId, ids, pos), ctx).logits();
			int tok = argmax(logits);
			generated.add(tok);
			ids = Arrays.copyOf(ids, ids.length + 1);
			ids[ids.length - 1] = tok;
			pos++;
		}

		assertThat(generated).isNotEmpty();
		for (int tok : generated) {
			assertThat(tok).isGreaterThanOrEqualTo(0);
		}
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
