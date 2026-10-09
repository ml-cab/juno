/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.node;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Times the decode step that performs a context shift against an ordinary decode
 * step at the same depth, on a LLaMA-family model at the context limit.
 *
 * <p>A request at the limit holds {@code depth} positions. The bench prefills a
 * short system prompt ({@code --keep} tokens), fills the rest of the host KV
 * directly with random rows (a cache's contents do not change what a step costs,
 * and prefilling 32,768 tokens on the CPU would take hours), and on a GPU rebuilds
 * the device KV mirror from those rows, as a shift does. Then, per repetition:
 * <ul>
 * <li>five ordinary decode steps ending at position {@code depth - 1}; their
 * median is the decode step;</li>
 * <li>at the full context, the shift the generation loop performs (keep the
 * system prompt, discard half of what follows it: {@code shiftKv}) followed by the
 * decode step at the shifted position, timed together as the shift step.</li>
 * </ul>
 * One warm-up repetition at {@code --warmup-depth} runs first and is not reported.
 * Prints one line per repetition and a {@code RESULT} line with the median ratio.
 *
 * <p>Usage: {@code ContextShiftStepBench --model <gguf> [--gpu] [--depth N] [--keep N] [--reps N]
 * [--warmup-depth N]}. Driven by {@code scripts/performance-tests/context-shift-step-bench.sh}.
 */
public final class ContextShiftStepBench {

	private static final int DECODE_STEPS = 5;

	private ContextShiftStepBench() {
	}

	public static void main(String[] args) throws Exception {
		Path model = null;
		boolean gpu = false;
		int depth = DenseKvTensor.MAX_SEQ_LEN;
		int keep = 32;
		int reps = 3;
		int warmupDepth = 4096;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--model" -> model = Path.of(args[++i]);
			case "--gpu" -> gpu = true;
			case "--depth" -> depth = Integer.parseInt(args[++i]);
			case "--keep" -> keep = Integer.parseInt(args[++i]);
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--warmup-depth" -> warmupDepth = Integer.parseInt(args[++i]);
			default -> throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		if (model == null)
			throw new IllegalArgumentException("--model is required");

		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(model)) {
			cfg = LlamaConfig.from(r);
		}
		ShardContext shard = new ShardContext("bench", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
		GpuContext ctx = gpu ? GpuContext.init(0) : null;
		MatVec backend = gpu ? new CudaMatVec(ctx) : CpuMatVec.INSTANCE;
		LlamaTransformerHandler h = LlamaTransformerHandler.load(model, shard, backend);
		try {
			depth = Math.min(depth, h.contextLimit());
			System.out.printf(Locale.ROOT,
					"model=%s backend=%s depth=%d keep=%d discard=%d layers=%d kvDim=%d gpuAttention=%s residency=%s%n",
					model.getFileName(), gpu ? "gpu" : "cpu", depth, keep, (depth - keep) / 2, cfg.numLayers(),
					cfg.kvDim(), h.gpuAttentionActive(), h.gpuResidencyActive());
			rep(h, shard, cfg, Math.min(warmupDepth, depth), keep, -1);
			double[] ratios = new double[reps];
			for (int i = 0; i < reps; i++)
				ratios[i] = rep(h, shard, cfg, depth, keep, i + 1);
			double[] sorted = ratios.clone();
			Arrays.sort(sorted);
			System.out.printf(Locale.ROOT, "RESULT median_ratio=%.3f min=%.3f max=%.3f reps=%d%n", sorted[reps / 2],
					sorted[0], sorted[reps - 1], reps);
		} finally {
			h.releaseGpuResources();
			if (ctx != null)
				ctx.close();
		}
	}

	/** One repetition at {@code depth}; returns shift step over median decode step. {@code rep} -1 is the warm-up. */
	private static double rep(LlamaTransformerHandler h, ShardContext shard, LlamaConfig cfg, int depth, int keep,
			int rep) {
		String id = "bench-" + rep;
		Random rng = new Random(rep + 7L);
		int[] prompt = new int[keep];
		for (int i = 0; i < keep; i++)
			prompt[i] = 300 + rng.nextInt(2000);
		try {
			h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard);
			int firstDecode = depth - DECODE_STEPS;
			fill(h.hostKv(id), keep, firstDecode, cfg.kvDim(), rng);
			h.rewriteDeviceKv(id, firstDecode);

			long[] decode = new long[DECODE_STEPS];
			for (int s = 0; s < DECODE_STEPS; s++) {
				int pos = firstDecode + s;
				long t0 = System.nanoTime();
				h.forward(ForwardRequest.withTokens(id, new int[] { 300 + rng.nextInt(2000) }, pos), shard);
				decode[s] = System.nanoTime() - t0;
			}
			long[] sortedDecode = decode.clone();
			Arrays.sort(sortedDecode);
			long medianDecode = sortedDecode[DECODE_STEPS / 2];

			int discard = (depth - keep) / 2;
			long t0 = System.nanoTime();
			h.shiftKv(id, depth, keep, discard);
			long tShift = System.nanoTime();
			h.forward(ForwardRequest.withTokens(id, new int[] { 300 + rng.nextInt(2000) }, depth - discard), shard);
			long t1 = System.nanoTime();

			long step = t1 - t0;
			double ratio = (double) step / medianDecode;
			System.out.printf(Locale.ROOT,
					"%s depth=%d decode_ms=%s median_decode_ms=%.2f shift_ms=%.2f shift_step_ms=%.2f ratio=%.3f%n",
					rep < 0 ? "warmup" : "rep " + rep, depth, Arrays.toString(Arrays.stream(decode)
							.mapToObj(n -> String.format(Locale.ROOT, "%.2f", n / 1e6)).toArray()),
					medianDecode / 1e6, (tShift - t0) / 1e6, step / 1e6, ratio);
			return ratio;
		} finally {
			h.evict(id);
		}
	}

	/** Writes random K and V rows at positions {@code [from, to)} on every layer. */
	private static void fill(SessionKvTensor[][] kv, int from, int to, int kvDim, Random rng) {
		float[] row = new float[kvDim];
		for (int layer = 0; layer < kv[0].length; layer++)
			for (int which = 0; which < 2; which++) {
				SessionKvTensor t = kv[which][layer];
				for (int pos = from; pos < to; pos++) {
					for (int i = 0; i < kvDim; i++)
						row[i] = rng.nextFloat() - 0.5f;
					t.ensureCapacity(pos);
					t.writeToken(pos, row);
				}
			}
	}
}
