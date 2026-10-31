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

/**
 * End-to-end check on a real model (CPU path) that a static micro-batch gives
 * every request its single-request greedy output:
 * <ul>
 * <li>it never resumes from KV it did not write. A repeated prompt in a later
 * batch used to match a stale prefix-cache entry, skip prefill, and decode over
 * zeroed KV;</li>
 * <li>its arithmetic does not depend on the batch. The colour answer ends after
 * three tokens, so the spider request then decodes alone; while the batched decode
 * ran another matmul than the single-request decode, that switch flipped the spider
 * answer's near tie at its fifth token ("six" to "eight") on the CPU's default
 * integer kernels.</li>
 * </ul>
 * Runs on those kernels in its own surefire execution (the Vector API module is on);
 * skipped when the model file is not present.
 */
class TinyLlamaStaticBatchLiveTest {

	private static final String MODEL_FILE = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
	private static final String SYSTEM = "You are a helpful assistant. Answer in one short sentence.";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(6);

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("coordinator")
			? Path.of(System.getProperty("user.dir")).getParent().resolve(MODEL_FILE)
			: Path.of(MODEL_FILE);

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	private static BatchEntry stateless(String user) {
		return new BatchEntry(InferenceRequest.of("tinyllama",
				List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user)), PARAMS, RequestPriority.NORMAL),
				TokenConsumer.discard());
	}

	@Test
	@EnabledIf("modelPresent")
	void repeated_prompt_in_a_later_static_batch_matches_single_request_output() throws Exception {
		LlamaConfig config;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(MODEL)) {
			config = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), singleNodePipeline(config),
				new KVCacheManager(new GpuKVCache(256L * 1024 * 1024), new CpuKVCache(4096)));

		BatchEntry a1 = stateless("What is the capital of France?");
		BatchEntry b1 = stateless("Name one primary color.");
		List<GenerationResult> round1 = loop.generateBatch(List.of(a1, b1));

		BatchEntry a2 = stateless("What is the capital of France?");
		BatchEntry c2 = stateless("How many legs does a spider have?");
		List<GenerationResult> round2 = loop.generateBatch(List.of(a2, c2));

		BatchEntry b3 = stateless("Name one primary color.");
		BatchEntry c3 = stateless("How many legs does a spider have?");
		List<GenerationResult> round3 = loop.generateBatch(List.of(b3, c3));

		List<Integer> aSingle = loop.generate(stateless("What is the capital of France?").request(),
				TokenConsumer.discard()).tokenIds();
		List<Integer> bSingle = loop.generate(stateless("Name one primary color.").request(),
				TokenConsumer.discard()).tokenIds();
		List<Integer> cSingle = loop.generate(stateless("How many legs does a spider have?").request(),
				TokenConsumer.discard()).tokenIds();

		assertThat(round1.get(0).tokenIds()).isEqualTo(aSingle);
		assertThat(round2.get(0).tokenIds()).as("repeated prompt in a later batch").isEqualTo(aSingle);
		assertThat(round2.get(1).tokenIds()).isEqualTo(cSingle);
		assertThat(round3.get(0).tokenIds()).isEqualTo(bSingle);
		assertThat(round3.get(1).tokenIds()).as("a request that finishes its decode batch alone").isEqualTo(cSingle);
	}

	private static LocalInferencePipeline singleNodePipeline(LlamaConfig config) throws Exception {
		ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		ForwardPassHandler handler = LlamaTransformerHandler.load(MODEL, ctx);
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("node-0", "localhost", 9090,
				config.numLayers() * vramPerLayer * 2, config.numLayers() * vramPerLayer * 2, NodeStatus.READY, 1.0,
				Instant.now(), Instant.now()));
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);
		return LocalInferencePipeline.from(shardMap, handler, config.vocabSize(), config.hiddenDim(),
				config.numHeads());
	}
}
