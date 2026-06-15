package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.Qwen3Config;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/** End-to-end GenerationLoop on Qwen3 when model is present under {@code models/}. */
class Qwen3GenerationLoopLiveTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).getParent() != null
			&& Path.of(System.getProperty("user.dir")).endsWith("coordinator")
					? Path.of(System.getProperty("user.dir")).getParent().resolve("models/qwen3-4b-instruct-q4_k_m.gguf")
					: Path.of("models/qwen3-4b-instruct-q4_k_m.gguf");

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void hello_greedy_generates_non_empty_reply() throws Exception {
		Path model = MODEL;
		Qwen3Config config;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			config = Qwen3Config.from(r);
			tokenizer = GgufTokenizer.load(r);
		}

		LocalInferencePipeline pipeline = singleNodePipeline(model, config);
		KVCacheManager kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		SamplingParams params = SamplingParams.deterministic().withMaxTokens(30);
		InferenceRequest request = InferenceRequest.ofSession("sess-qwen3", "qwen3",
				List.of(ChatMessage.user("hello")), params, RequestPriority.NORMAL);

		GenerationResult result = loop.generate(request, TokenConsumer.discard());

		assertThat(result.generatedTokens()).isGreaterThan(0);
		assertThat(result.text().toLowerCase()).containsAnyOf("hello", "hi", "hey");
	}

	private static LocalInferencePipeline singleNodePipeline(Path model, Qwen3Config config) throws Exception {
		ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx);
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("node-0", "localhost", 9090,
				config.numLayers() * vramPerLayer * 2, config.numLayers() * vramPerLayer * 2, NodeStatus.READY, 1.0,
				Instant.now(), Instant.now()));
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);
		return LocalInferencePipeline.from(shardMap, handler, config.vocabSize(), config.hiddenDim(), config.numHeads());
	}
}
