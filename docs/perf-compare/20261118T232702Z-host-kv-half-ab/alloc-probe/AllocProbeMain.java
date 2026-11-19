package cab.ml.juno.kit;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Arrays;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CpuMatVec;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;

/** Bytes the calling thread allocates per request: prefill of N tokens, G decode steps, evict; R rounds. */
public final class AllocProbeMain {
	public static void main(String[] a) throws Exception {
		Path model = Path.of(a[0]);
		int n = Integer.parseInt(a[1]), g = Integer.parseInt(a[2]), rounds = Integer.parseInt(a[3]);
		boolean reserve = a.length > 4 && a[4].equals("reserve");
		boolean gpu = a.length > 5 && a[5].equals("gpu");
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, ctx, gpu ? new CudaMatVec(GpuContext.shared(0)) : CpuMatVec.INSTANCE);
		Method rk = null;
		if (reserve)
			rk = ForwardPassHandler.class.getMethod("reserveKv", String.class, int.class);
		int[] prompt = new int[n];
		for (int i = 0; i < n; i++)
			prompt[i] = 100 + (i * 37) % 900;
		var tb = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		for (int round = 0; round < rounds; round++) {
			String id = "req-" + round;
			long t0 = tb.getCurrentThreadAllocatedBytes();
			if (rk != null)
				rk.invoke(h, id, n + g);
			float[] logits = h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), ctx).lastLogits();
			long t1 = tb.getCurrentThreadAllocatedBytes();
			for (int i = 0; i < g - 1; i++)
				logits = h.forward(ForwardRequest.withTokens(id, new int[] { argmax(logits) }, n + i), ctx).logits();
			long t2 = tb.getCurrentThreadAllocatedBytes();
			h.evict(id);
			System.out.printf("ALLOC round=%d prefill=%,d decode=%,d decode_per_token=%,d%n", round, t1 - t0, t2 - t1,
					(t2 - t1) / (g - 1));
		}
	}

	static int argmax(float[] x) {
		int b = 0;
		for (int i = 1; i < x.length; i++)
			if (x[i] > x[b])
				b = i;
		return b;
	}
}
