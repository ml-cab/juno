package cab.ml.juno.node;

import java.util.Arrays;

import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Drives one real handler through a context shift and scores it against
 * {@link ContextShiftOracle}: request {@code s} is prefilled with {@code n}
 * tokens and shifted; request {@code o} is prefilled the same way, its device
 * mirrors (if any) retired, and its host KV overwritten with the oracle KV; both
 * then take the same next token at position {@code n - discard}. With GPU
 * attention active, a control pair without a shift measures how far device and
 * host attention differ on their own (zero on the CPU, so not run there).
 */
final class ContextShiftLiveCheck {

	/** {@code watermarks}: each device mirror's written length right after the shift; null without mirrors. */
	record Result(double shiftDiff, double controlDiff, boolean sameGreedyToken, int[] watermarks) {
	}

	private ContextShiftLiveCheck() {
	}

	static SessionKvTensor[][] hostKv(ForwardPassHandler h, String id) {
		return switch (h) {
		case LlamaTransformerHandler x -> x.hostKv(id);
		case Phi3TransformerHandler x -> x.hostKv(id);
		case Qwen3TransformerHandler x -> x.hostKv(id);
		case Qwen3MoeTransformerHandler x -> x.hostKv(id);
		case Phi2TransformerHandler x -> x.hostKv(id);
		case LoraTrainableHandler x -> x.hostKv(id);
		case Phi3LoraTrainableHandler x -> x.hostKv(id);
		case Qwen3LoraTrainableHandler x -> x.hostKv(id);
		case Qwen2LoraTrainableHandler x -> x.delegate().hostKv(id);
		default -> throw new IllegalArgumentException(h.getClass().getSimpleName());
		};
	}

	static void retireDeviceKv(ForwardPassHandler h, String id) {
		switch (h) {
		case LlamaTransformerHandler x -> x.retireDeviceKv(id);
		case Phi3TransformerHandler x -> x.retireDeviceKv(id);
		case Qwen3TransformerHandler x -> x.retireDeviceKv(id);
		default -> {
		}
		}
	}

	static int[] deviceKvWatermarks(ForwardPassHandler h, String id) {
		return switch (h) {
		case LlamaTransformerHandler x -> x.deviceKvWatermarks(id);
		case Phi3TransformerHandler x -> x.deviceKvWatermarks(id);
		case Qwen3TransformerHandler x -> x.deviceKvWatermarks(id);
		default -> null;
		};
	}

	static int token(int i) {
		return 300 + (i * 37) % 2000;
	}

	static Result run(ForwardPassHandler h, ShardContext shard, int kvDim, int n, int keep, int discard,
			ContextShiftOracle.RowRotation unrotate, ContextShiftOracle.RowRotation rotate) {
		int[] all = new int[n + 1];
		for (int i = 0; i <= n; i++)
			all[i] = token(i);
		int[] prefix = Arrays.copyOf(all, n);
		int newLen = n - discard;
		try {
			h.forwardBatch(BatchForwardRequest.withTokens("s", prefix, 0), shard);
			h.forwardBatch(BatchForwardRequest.withTokens("o", prefix, 0), shard);
			retireDeviceKv(h, "o");
			SessionKvTensor[][] o = hostKv(h, "o");
			ContextShiftOracle.inject(o[0], ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[0], n), kvDim, n,
					keep, discard, true, unrotate, rotate), kvDim, newLen);
			ContextShiftOracle.inject(o[1], ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[1], n), kvDim, n,
					keep, discard, false, null, null), kvDim, newLen);
			float[] expected = h.forward(ForwardRequest.withTokens("o", new int[] { all[n] }, newLen), shard).logits();

			h.shiftKv("s", n, keep, discard);
			int[] watermarks = deviceKvWatermarks(h, "s");
			float[] shifted = h.forward(ForwardRequest.withTokens("s", new int[] { all[n] }, newLen), shard).logits();

			double control = 0;
			if (h.gpuAttentionActive()) {
				h.forwardBatch(BatchForwardRequest.withTokens("gd", prefix, 0), shard);
				h.forwardBatch(BatchForwardRequest.withTokens("gh", prefix, 0), shard);
				retireDeviceKv(h, "gh");
				float[] device = h.forward(ForwardRequest.withTokens("gd", new int[] { all[n] }, n), shard).logits();
				float[] host = h.forward(ForwardRequest.withTokens("gh", new int[] { all[n] }, n), shard).logits();
				control = maxAbsDiff(device, host);
			}

			Result r = new Result(maxAbsDiff(shifted, expected), control, argmax(shifted) == argmax(expected),
					watermarks);
			System.out.printf("%s: context shift max |logit diff| = %.5f, device-vs-host control = %.5f%n",
					h.getClass().getSimpleName(), r.shiftDiff(), r.controlDiff());
			return r;
		} finally {
			for (String id : new String[] { "s", "o", "gd", "gh" })
				h.evict(id);
		}
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
