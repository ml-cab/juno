package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LlamaTransformerHandler;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/** End-to-end GenerationLoop on Qwen2.5 when model is present. */
class Qwen2GenerationLoopLiveTest {

	private static final int HELLO_GREEDY_TOKEN = 9707; // llama.cpp reference for hello ChatML prompt

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).getParent() != null
			&& Path.of(System.getProperty("user.dir")).endsWith("coordinator")
					? Path.of(System.getProperty("user.dir")).getParent().resolve("models/qwen2.5-3b-instruct-q4_k_m.gguf")
					: Path.of("models/qwen2.5-3b-instruct-q4_k_m.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void single_node_pipeline_first_greedy_token_is_hello() throws Exception {
		Path model = MODEL;
		LlamaConfig config;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			config = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}

		String prompt = cab.ml.juno.tokenizer.ChatTemplateFormatter.forModelType("qwen2.5")
				.format(List.of(ChatMessage.user("hello")));
		int[] promptIds = tokenizer.encode(prompt);
		assertThat(promptIds).containsExactly(151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198);

		LocalInferencePipeline pipeline = singleNodePipeline(model, config);
		for (int p = 0; p < promptIds.length - 1; p++) {
			pipeline.forward("sess-1", Arrays.copyOfRange(promptIds, 0, p + 1), p);
		}
		int best = argmax(pipeline.forward("sess-1", promptIds, promptIds.length - 1));
		assertThat(best).isEqualTo(HELLO_GREEDY_TOKEN);
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_greedy_generates_non_empty_reply() throws Exception {
		Path model = MODEL;
		LlamaConfig config;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			config = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}

		LocalInferencePipeline pipeline = singleNodePipeline(model, config);
		KVCacheManager kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		SamplingParams params = SamplingParams.deterministic().withMaxTokens(30);
		InferenceRequest request = InferenceRequest.ofSession("sess-1", "qwen2.5",
				List.of(ChatMessage.user("hello")), params, RequestPriority.NORMAL);

		GenerationResult result = loop.generate(request, TokenConsumer.discard());

		assertThat(result.generatedTokens()).isGreaterThan(0);
		assertThat(result.text().toLowerCase()).containsAnyOf("hello", "hi", "hey");
	}

	@Test
	@EnabledIf("modelPresent")
	void three_node_pipeline_first_greedy_token_matches_single_node() throws Exception {
		Path model = MODEL;
		LlamaConfig config;
		try (GgufReader r = GgufReader.open(model)) {
			config = LlamaConfig.from(r);
		}

		int[] promptIds = { 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198 };
		LocalInferencePipeline single = singleNodePipeline(model, config);
		LocalInferencePipeline triple = threeNodePipeline(model, config);

		for (int p = 0; p < promptIds.length - 1; p++) {
			int[] slice = Arrays.copyOfRange(promptIds, 0, p + 1);
			single.forward("kv", slice, p);
			triple.forward("kv", slice, p);
		}

		int singleBest = argmax(single.forward("kv", promptIds, promptIds.length - 1));
		int tripleBest = argmax(triple.forward("kv", promptIds, promptIds.length - 1));
		System.out.println("single=" + singleBest + " triple=" + tripleBest);
		assertThat(tripleBest).isEqualTo(singleBest).isEqualTo(HELLO_GREEDY_TOKEN);
	}

	private static LocalInferencePipeline singleNodePipeline(Path model, LlamaConfig config) throws Exception {
		ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		ForwardPassHandler handler = LlamaTransformerHandler.load(model, ctx);
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("node-0", "localhost", 9090,
				config.numLayers() * vramPerLayer * 2, config.numLayers() * vramPerLayer * 2, NodeStatus.READY, 1.0,
				Instant.now(), Instant.now()));
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);
		return LocalInferencePipeline.from(shardMap, handler, config.vocabSize(), config.hiddenDim(), config.numHeads());
	}

	private static LocalInferencePipeline threeNodePipeline(Path model, LlamaConfig config) throws Exception {
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		long nodeVram = config.numLayers() * vramPerLayer * 2;
		List<NodeDescriptor> nodes = new ArrayList<>();
		for (int i = 0; i < 3; i++) {
			nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, nodeVram, nodeVram, NodeStatus.READY, 1.0,
					Instant.now(), Instant.now()));
		}
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);
		List<ForwardPassHandler> handlers = new ArrayList<>();
		for (var assignment : shardMap.assignments()) {
			ShardContext ctx = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());
			handlers.add(ForwardPassHandlerLoader.load(model, ctx));
		}
		return LocalInferencePipeline.from(shardMap, handlers, config.vocabSize(), config.hiddenDim(), config.numHeads());
	}

	private static int argmax(float[] logits) {
		int best = 0;
		float bestScore = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < logits.length; i++) {
			if (logits[i] > bestScore) {
				bestScore = logits[i];
				best = i;
			}
		}
		return best;
	}
}
