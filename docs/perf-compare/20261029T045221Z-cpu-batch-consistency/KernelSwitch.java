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
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.MultiDecodeForwardRequest;
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
public class KernelSwitch {
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
		Rec rec = new Rec(pipeline, handler, ctx);
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), rec,
				new KVCacheManager(new GpuKVCache(256L * 1024 * 1024), new CpuKVCache(4096)));
		int prompt = Integer.parseInt(a[1]);
		int[] modes = { 0, -1, 1, 2, 3, 4, 5, 6 };
		for (int m : modes) {
			rec.reset(m);
			GenerationResult g = loop.generate(e(prompt).request(), TokenConsumer.discard());
			String label = m == 0 ? "single-row kernel only" : m < 0 ? "gemm only" : "gemm for first " + m + " steps";
			System.out.println("MODE " + label + ": '" + g.text() + "' " + g.tokenIds());
			for (int step = 0; step < rec.logits.size(); step++) {
				float[] l = rec.logits.get(step);
				int t1 = -1, t2 = -1;
				for (int v = 0; v < l.length; v++) {
					if (t1 < 0 || l[v] > l[t1]) { t2 = t1; t1 = v; }
					else if (t2 < 0 || l[v] > l[t2]) t2 = v;
				}
				System.out.printf("  step %d: top '%s'(%d) %.5f, second '%s'(%d) %.5f, gap %.5f%n", step,
						tokenizer.decodeToken(t1), t1, l[t1], tokenizer.decodeToken(t2), t2, l[t2], l[t1] - l[t2]);
			}
		}
		System.exit(0);
	}

	static final class Rec implements InferencePipeline {
		final LocalInferencePipeline d;
		final ForwardPassHandler h;
		final ShardContext ctx;
		int gemmSteps;
		int calls;
		final List<float[]> logits = new ArrayList<>();

		Rec(LocalInferencePipeline d, ForwardPassHandler h, ShardContext ctx) {
			this.d = d;
			this.h = h;
			this.ctx = ctx;
		}

		void reset(int gemmSteps) {
			this.gemmSteps = gemmSteps;
			this.calls = 0;
			logits.clear();
		}

		@Override
		public float[] forward(String id, int[] toks, int pos) {
			float[] l;
			if (gemmSteps < 0 || calls < gemmSteps)
				l = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of(id),
						new int[] { toks[toks.length - 1] }, new int[] { pos }), ctx).logits()[0];
			else
				l = d.forward(id, toks, pos);
			calls++;
			logits.add(l.clone());
			return l;
		}

		@Override
		public int vocabSize() {
			return d.vocabSize();
		}

		@Override
		public float[][] forwardBatch(List<String> ids, List<int[]> toks, List<Integer> pos) {
			return d.forwardBatch(ids, toks, pos);
		}

		@Override
		public void prefillBatch(String id, int[] toks, int pos) {
			d.prefillBatch(id, toks, pos);
		}

		@Override
		public void evict(String id) {
			d.evict(id);
		}

		@Override
		public int contextLimit() {
			return d.contextLimit();
		}
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
