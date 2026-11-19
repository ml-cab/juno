package cab.ml.juno.kit;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

import javax.management.NotificationEmitter;
import javax.management.openmbean.CompositeData;

import com.sun.management.GarbageCollectionNotificationInfo;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CpuMatVec;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;

/** Scratch: live heap of one handler at a prompt of N tokens plus G decoded tokens. Not part of the repo. */
public final class HeapProbeMain {

	static final AtomicLong PEAK_AFTER_GC = new AtomicLong();

	public static void main(String[] args) throws Exception {
		Path model = Path.of(args[0]);
		boolean gpu = args[1].equals("gpu");
		int tokens = Integer.parseInt(args[2]);
		int generated = Integer.parseInt(args[3]);
		for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
			((NotificationEmitter) gc).addNotificationListener((n, h) -> {
				if (!n.getType().equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION))
					return;
				var info = GarbageCollectionNotificationInfo.from((CompositeData) n.getUserData());
				long after = 0;
				for (MemoryUsage u : info.getGcInfo().getMemoryUsageAfterGc().values())
					after += u.getUsed();
				PEAK_AFTER_GC.accumulateAndGet(after, Math::max);
			}, null, null);
		}
		ShardContext ctx;
		int[] ids;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		ids = Arrays.stream(java.nio.file.Files.readString(Path.of(args[4])).trim().split("\\s+"))
				.mapToInt(Integer::parseInt).toArray();
		int[] prompt = Arrays.copyOf(ids, tokens);
		long base = live();
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx,
				gpu ? new CudaMatVec(GpuContext.shared(0)) : CpuMatVec.INSTANCE);
		long loaded = live();
		if (args.length > 5) {
			System.out.println("HISTO --- after load");
			histogram(Integer.parseInt(args[5]));
			System.out.println("HISTO --- with session");
		}
		PEAK_AFTER_GC.set(0);
		String session = "heap-probe";
		long t0 = System.nanoTime();
		float[] logits = handler.forwardBatch(BatchForwardRequest.withTokens(session, prompt, 0), ctx).lastLogits();
		int next = argmax(logits);
		for (int i = 0; i < generated; i++) {
			logits = handler.forward(ForwardRequest.withTokens(session, new int[] { next }, tokens + i), ctx).logits();
			next = argmax(logits);
		}
		double secs = (System.nanoTime() - t0) / 1e9;
		long peakDuring = PEAK_AFTER_GC.get();
		long withSession = live();
		if (args.length > 5)
			histogram(Integer.parseInt(args[5]));
		handler.evict(session);
		long evicted = live();
		System.out.printf(
				"HEAP-PROBE model=%s backend=%s prompt=%d generated=%d secs=%.1f base_mb=%d loaded_mb=%d "
						+ "peak_after_gc_during_mb=%d live_with_session_mb=%d live_after_evict_mb=%d gpuAttention=%s%n",
				model.getFileName(), args[1], tokens, generated, secs, mb(base), mb(loaded), mb(peakDuring),
				mb(withSession), mb(evicted), handler.gpuAttentionActive());
		System.exit(0);
	}

	static long live() {
		for (int i = 0; i < 2; i++)
			System.gc();
		return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
	}

	static void histogram(int lines) throws Exception {
		var server = ManagementFactory.getPlatformMBeanServer();
		var name = new javax.management.ObjectName("com.sun.management:type=DiagnosticCommand");
		String out = (String) server.invoke(name, "gcClassHistogram", new Object[] { new String[0] },
				new String[] { String[].class.getName() });
		String[] rows = out.split("\n");
		for (int i = 0; i < Math.min(lines + 3, rows.length); i++)
			System.out.println("HISTO " + rows[i]);
	}

	static long mb(long b) {
		return b / (1024 * 1024);
	}

	static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
