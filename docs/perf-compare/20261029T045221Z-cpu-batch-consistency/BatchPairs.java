import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import cab.ml.juno.coordinator.BatchEntry;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
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

/** Static-batch divergence diagnosis: composition alone (sequential batches), then concurrent batches. */
public class BatchPairs {
	static final String SYSTEM = "You are a helpful assistant. Answer in one short sentence.";
	static final List<String> P = List.of("What is the capital of France?", "Name one primary color.",
			"How many legs does a spider have?");
	static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(8);

	static BatchEntry e(int i) {
		return new BatchEntry(InferenceRequest.of("tinyllama",
				List.of(ChatMessage.system(SYSTEM), ChatMessage.user(P.get(i))), PARAMS, RequestPriority.NORMAL),
				TokenConsumer.discard());
	}

	public static void main(String[] a) throws Exception {
		Path model = Path.of(a[0]);
		int reps = Integer.parseInt(a[1]);
		LlamaConfig config;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			config = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		ShardContext ctx = new ShardContext("n0", 0, config.numLayers(), true, true, config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		ForwardPassHandler handler = LlamaTransformerHandler.load(model, ctx);
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("node-0", "localhost", 9090,
				config.numLayers() * vramPerLayer * 2, config.numLayers() * vramPerLayer * 2, NodeStatus.READY, 1.0,
				Instant.now(), Instant.now()));
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayer, nodes);
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(shardMap, handler, config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(256L * 1024 * 1024), new CpuKVCache(4096)));

		List<GenerationResult> single = new ArrayList<>();
		for (int i = 0; i < P.size(); i++) {
			single.add(loop.generate(e(i).request(), TokenConsumer.discard()));
			System.out.println("single " + i + ": '" + single.get(i).text() + "' " + single.get(i).tokenIds());
		}

		int bad = 0;
		for (int i = 0; i < P.size(); i++)
			for (int j = 0; j < P.size(); j++) {
				List<GenerationResult> r = loop.generateBatch(List.of(e(i), e(j)));
				bad += report("seq [" + i + "," + j + "]", new int[] { i, j }, r, single);
			}
		System.out.println("SEQUENTIAL mismatches: " + bad);

		int cbad = 0;
		int[][][] pairs = { { { 0, 1 }, { 2, 0 } }, { { 2, 0 }, { 0, 1 } }, { { 0, 2 }, { 1, 0 } } };
		for (int rep = 0; rep < reps; rep++)
			for (int[][] pr : pairs) {
				AtomicReference<List<GenerationResult>> ra = new AtomicReference<>();
				AtomicReference<List<GenerationResult>> rb = new AtomicReference<>();
				CountDownLatch go = new CountDownLatch(1);
				Thread ta = Thread.startVirtualThread(() -> {
					await(go);
					ra.set(loop.generateBatch(List.of(e(pr[0][0]), e(pr[0][1]))));
				});
				Thread tb = Thread.startVirtualThread(() -> {
					await(go);
					rb.set(loop.generateBatch(List.of(e(pr[1][0]), e(pr[1][1]))));
				});
				go.countDown();
				ta.join();
				tb.join();
				cbad += report("conc rep" + rep + " A", pr[0], ra.get(), single);
				cbad += report("conc rep" + rep + " B", pr[1], rb.get(), single);
			}
		System.out.println("CONCURRENT mismatches: " + cbad);
		System.exit(0);
	}

	static void await(CountDownLatch l) {
		try {
			l.await();
		} catch (InterruptedException x) {
			throw new RuntimeException(x);
		}
	}

	static int report(String label, int[] order, List<GenerationResult> r, List<GenerationResult> single) {
		int bad = 0;
		for (int k = 0; k < order.length; k++) {
			List<Integer> want = single.get(order[k]).tokenIds();
			List<Integer> got = r.get(k).tokenIds();
			if (!got.equals(want)) {
				bad++;
				int at = 0;
				while (at < Math.min(got.size(), want.size()) && got.get(at).equals(want.get(at)))
					at++;
				System.out.println("MISMATCH " + label + " row " + k + " prompt " + order[k] + " at token " + at
						+ ": single '" + single.get(order[k]).text() + "' batched '" + r.get(k).text() + "'");
			}
		}
		return bad;
	}
}
