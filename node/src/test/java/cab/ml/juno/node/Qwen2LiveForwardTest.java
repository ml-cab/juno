package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;

/** Live forward pass on Qwen2.5 GGUF when present in models/. */
class Qwen2LiveForwardTest {

	private static final int IM_END_ID = 151645;

	/** llama.cpp tokenization of Juno ChatML "hello" prompt (--no-bos). */
	private static final int[] HELLO_PROMPT_IDS = {
			151644, 872, 198, 14990, IM_END_ID, 198, 151644, 77091, 198
	};

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).getParent() != null
			&& Path.of(System.getProperty("user.dir")).endsWith("node")
					? Path.of(System.getProperty("user.dir")).getParent().resolve("models/qwen2.5-3b-instruct-q4_k_m.gguf")
					: Path.of("models/qwen2.5-3b-instruct-q4_k_m.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_prompt_first_greedy_token_is_not_im_end() throws Exception {
		Path model = MODEL;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig config = LlamaConfig.from(r);

			ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
					config.hiddenDim(), config.numHeads());
			LlamaTransformerHandler handler = LlamaTransformerHandler.load(model, ctx);

			int[] promptIds = HELLO_PROMPT_IDS;

			for (int p = 0; p < promptIds.length - 1; p++) {
				int[] slice = Arrays.copyOfRange(promptIds, 0, p + 1);
				handler.forward(ForwardRequest.withTokens("kv", slice, p), ctx);
			}

			float[] logits = handler.forward(ForwardRequest.withTokens("kv", promptIds, promptIds.length - 1), ctx)
					.logits();

			int best = 0;
			float bestScore = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < logits.length; i++) {
				if (logits[i] > bestScore) {
					bestScore = logits[i];
					best = i;
				}
			}

			assertThat(best).as("greedy first token id (score=%s)", bestScore).isNotEqualTo(IM_END_ID);
		}
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_prompt_top5_greedy_tokens() throws Exception {
		Path model = MODEL;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig config = LlamaConfig.from(r);

			ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
					config.hiddenDim(), config.numHeads());
			LlamaTransformerHandler handler = LlamaTransformerHandler.load(model, ctx);

			int[] promptIds = HELLO_PROMPT_IDS;
			for (int p = 0; p < promptIds.length - 1; p++) {
				handler.forward(ForwardRequest.withTokens("kv", Arrays.copyOfRange(promptIds, 0, p + 1), p), ctx);
			}
			float[] logits = handler.forward(ForwardRequest.withTokens("kv", promptIds, promptIds.length - 1), ctx)
					.logits();

			int[] top = topK(logits, 5);
			for (int id : top) {
				System.out.println("TOP " + id);
			}
			assertThat(top[0]).isNotEqualTo(IM_END_ID);
		}
	}

	private static int[] topK(float[] logits, int k) {
		Integer[] idx = new Integer[logits.length];
		for (int i = 0; i < idx.length; i++)
			idx[i] = i;
		java.util.Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));
		int[] out = new int[k];
		for (int i = 0; i < k; i++)
			out[i] = idx[i];
		return out;
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_prompt_three_node_pipeline_first_token_is_not_im_end() throws Exception {
		Path model = MODEL;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig config = LlamaConfig.from(r);
			long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
			long nodeVram = config.numLayers() * vramPerLayer * 2;

			List<NodeDescriptor> nodes = new ArrayList<>();
			for (int i = 0; i < 3; i++) {
				nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, nodeVram, nodeVram,
						NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
			}
			ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);

			List<ForwardPassHandler> handlers = new ArrayList<>();
			for (var assignment : shardMap.assignments()) {
				ShardContext ctx = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(),
						config.numHeads());
				handlers.add(ForwardPassHandlerLoader.load(model, ctx));
			}

			LocalInferencePipeline pipeline = LocalInferencePipeline.from(shardMap, handlers, config.vocabSize(),
					config.hiddenDim(), config.numHeads());

			int[] promptIds = HELLO_PROMPT_IDS;
			for (int p = 0; p < promptIds.length - 1; p++) {
				int[] slice = Arrays.copyOfRange(promptIds, 0, p + 1);
				pipeline.forward("kv", slice, p);
			}

			float[] logits = pipeline.forward("kv", promptIds, promptIds.length - 1);
			int best = 0;
			float bestScore = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < logits.length; i++) {
				if (logits[i] > bestScore) {
				 bestScore = logits[i];
				 best = i;
				}
			}

			assertThat(best).as("3-node pipeline greedy first token (score=%s)", bestScore).isNotEqualTo(IM_END_ID);
		}
	}
}
