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
package cab.ml.juno.kvcache;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Gather-tax microbench for dual KV path (dense vs paged gather + GQA).
 *
 * <p>Matrix: ctx ∈ {2k, 8k, 32k} × batch ∈ {1, 8, 32}, optional page sizes.
 * Prints absolute ns and gather overhead as % of (gather + attention).
 * Budget gate: ≤ ~15% at batch 8 / ctx 8k.
 *
 * <p>Shapes match TinyLlama-class GQA (heads=32, kvHeads=4, headDim=64).
 */
public final class GatherTaxMicrobench {

	/** TinyLlama-like attention geometry. */
	public static final int NUM_HEADS = 32;
	public static final int NUM_KV_HEADS = 4;
	public static final int HEAD_DIM = 64;
	public static final int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 256
	public static final int Q_DIM = NUM_HEADS * HEAD_DIM; // 2048

	public static final int[] DEFAULT_CTX = { 2048, 8192, 32768 };
	public static final int[] DEFAULT_BATCH = { 1, 8, 32 };
	public static final int[] DEFAULT_PAGES = { 16, 64, 128 };

	/** Proceed-to-continuous reference cell. */
	public static final int GATE_CTX = 8192;
	public static final int GATE_BATCH = 8;
	public static final double GATE_MAX_GATHER_PCT = 15.0;

	private GatherTaxMicrobench() {
	}

	public record Cell(
			int ctx,
			int batch,
			int pageSize,
			long denseAttnNs,
			long gatherNs,
			long pagedAttnNs,
			long denseTotalNs,
			long pagedTotalNs) {

		/** Gather share of paged (gather + attention), percent. */
		public double gatherPctOfPaged() {
			long den = gatherNs + pagedAttnNs;
			if (den <= 0)
				return 0;
			return 100.0 * gatherNs / den;
		}

		/** Paged total vs dense attention-only (dense has no gather). */
		public double pagedOverDensePct() {
			if (denseTotalNs <= 0)
				return 0;
			return 100.0 * (pagedTotalNs - denseTotalNs) / denseTotalNs;
		}
	}

	public static void main(String[] args) throws IOException {
		int[] ctxs = DEFAULT_CTX;
		int[] batches = DEFAULT_BATCH;
		int[] pages = DEFAULT_PAGES;
		int warmup = 3;
		int iters = 7;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--ctx" -> ctxs = parseInts(args[++i]);
			case "--batch" -> batches = parseInts(args[++i]);
			case "--page-size" -> pages = parseInts(args[++i]);
			case "--warmup" -> warmup = Integer.parseInt(args[++i]);
			case "--iters" -> iters = Integer.parseInt(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			case "--help" -> {
				usage();
				return;
			}
			default -> throw new IllegalArgumentException("unknown arg: " + args[i]);
			}
		}

		List<Cell> cells = runMatrix(ctxs, batches, pages, warmup, iters);
		String report = formatReport(cells);
		System.out.print(report);
		if (out != null) {
			Files.createDirectories(out.getParent() == null ? Path.of(".") : out.getParent());
			try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
				pw.print(report);
			}
			System.err.println("[gather-tax] wrote " + out.toAbsolutePath());
		}
		Cell gate = findCell(cells, GATE_CTX, GATE_BATCH, pages[0]);
		if (gate != null) {
			double pct = gate.gatherPctOfPaged();
			System.err.printf(Locale.ROOT,
					"[gather-tax] gate ctx=%d batch=%d page=%d gatherPct=%.2f (budget ≤ %.0f)%n",
					gate.ctx(), gate.batch(), gate.pageSize(), pct, GATE_MAX_GATHER_PCT);
			if (pct > GATE_MAX_GATHER_PCT)
				System.exit(2);
		}
	}

	private static void usage() {
		System.out.println(
				"Usage: GatherTaxMicrobench [--ctx 2048,8192,32768] [--batch 1,8,32] "
						+ "[--page-size 16,64,128] [--warmup N] [--iters N] [--out path.md]");
	}

	public static List<Cell> runMatrix(int[] ctxs, int[] batches, int[] pages, int warmup, int iters) {
		if (warmup < 0 || iters < 1)
			throw new IllegalArgumentException("warmup>=0 iters>=1");
		List<Cell> out = new ArrayList<>();
		for (int page : pages) {
			for (int ctx : ctxs) {
				if (ctx < 1 || ctx > DenseKvTensor.MAX_SEQ_LEN)
					throw new IllegalArgumentException(
							"ctx " + ctx + " out of range 1.." + DenseKvTensor.MAX_SEQ_LEN);
				for (int batch : batches) {
					out.add(measureCell(ctx, batch, page, warmup, iters));
				}
			}
		}
		return out;
	}

	public static Cell measureCell(int ctx, int batch, int pageSize, int warmup, int iters) {
		DenseKvTensor denseK = new DenseKvTensor(KvElementType.F16, KV_DIM, Math.min(ctx, 64));
		DenseKvTensor denseV = new DenseKvTensor(KvElementType.F16, KV_DIM, Math.min(ctx, 64));
		PagedKvArena arena = new PagedKvArena(pageSize, KV_DIM, KvElementType.F16, KvElementType.F16);
		PagedKvTensor pagedK = arena.newK();
		PagedKvTensor pagedV = arena.newV();

		float[] tok = new float[KV_DIM];
		for (int t = 0; t < ctx; t++) {
			fillToken(tok, t);
			denseK.writeToken(t, tok);
			denseV.writeToken(t, tok);
			pagedK.writeToken(t, tok);
			pagedV.writeToken(t, tok);
		}

		float[] q = new float[Q_DIM];
		for (int i = 0; i < Q_DIM; i++)
			q[i] = (i % 17) * 0.01f;
		float[] out = new float[Q_DIM];
		float[] scores = new float[ctx];
		float[] kScratch = new float[ctx * KV_DIM];
		float[] vScratch = new float[ctx * KV_DIM];

		// Warmup
		for (int w = 0; w < warmup; w++) {
			timeDense(denseK, denseV, q, out, scores, ctx, batch);
			timePaged(pagedK, pagedV, q, out, scores, kScratch, vScratch, ctx, batch);
		}

		long denseAttn = 0;
		long gather = 0;
		long pagedAttn = 0;
		for (int i = 0; i < iters; i++) {
			denseAttn += timeDense(denseK, denseV, q, out, scores, ctx, batch);
			long[] gp = timePagedSplit(pagedK, pagedV, q, out, scores, kScratch, vScratch, ctx, batch);
			gather += gp[0];
			pagedAttn += gp[1];
		}
		denseAttn /= iters;
		gather /= iters;
		pagedAttn /= iters;
		return new Cell(ctx, batch, pageSize, denseAttn, gather, pagedAttn, denseAttn, gather + pagedAttn);
	}

	/** Returns {gatherNs, attnNs} averaged over the batch streams. */
	private static long[] timePagedSplit(PagedKvTensor k, PagedKvTensor v, float[] q, float[] out,
			float[] scores, float[] kScratch, float[] vScratch, int ctx, int batch) {
		long g = 0;
		long a = 0;
		for (int b = 0; b < batch; b++) {
			long t0 = System.nanoTime();
			k.viewForAttention(ctx, kScratch);
			v.viewForAttention(ctx, vScratch);
			long t1 = System.nanoTime();
			gqaInto(q, kScratch, vScratch, ctx, out, scores);
			long t2 = System.nanoTime();
			g += (t1 - t0);
			a += (t2 - t1);
		}
		return new long[] { g, a };
	}

	private static long timePaged(PagedKvTensor k, PagedKvTensor v, float[] q, float[] out,
			float[] scores, float[] kScratch, float[] vScratch, int ctx, int batch) {
		long[] parts = timePagedSplit(k, v, q, out, scores, kScratch, vScratch, ctx, batch);
		return parts[0] + parts[1];
	}

	private static long timeDense(DenseKvTensor k, DenseKvTensor v, float[] q, float[] out,
			float[] scores, int ctx, int batch) {
		long ns = 0;
		for (int b = 0; b < batch; b++) {
			long t0 = System.nanoTime();
			float[] kView = k.viewForAttention(ctx, null);
			float[] vView = v.viewForAttention(ctx, null);
			gqaInto(q, kView, vView, ctx, out, scores);
			ns += System.nanoTime() - t0;
		}
		return ns;
	}

	/**
	 * Same structure as handler {@code gqaInto} (grouped-query attention into
	 * preallocated buffers). Kept here so the microbench does not depend on node.
	 */
	static void gqaInto(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores) {
		float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
		java.util.Arrays.fill(out, 0f);
		int gqaR = NUM_HEADS / NUM_KV_HEADS;
		for (int h = 0; h < NUM_HEADS; h++) {
			int kvHead = h / gqaR;
			int qBase = h * HEAD_DIM;
			int kBase = kvHead * HEAD_DIM;
			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * KV_DIM + kBase;
				for (int d = 0; d < HEAD_DIM; d++)
					dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}
			softmax(scores, seqLen);
			int outBase = h * HEAD_DIM;
			for (int t = 0; t < seqLen; t++) {
				int vOffset = t * KV_DIM + kBase;
				float w = scores[t];
				for (int d = 0; d < HEAD_DIM; d++)
					out[outBase + d] += w * vCache[vOffset + d];
			}
		}
	}

	static void softmax(float[] scores, int n) {
		float max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < n; i++)
			if (scores[i] > max)
				max = scores[i];
		float sum = 0f;
		for (int i = 0; i < n; i++) {
			scores[i] = (float) Math.exp(scores[i] - max);
			sum += scores[i];
		}
		for (int i = 0; i < n; i++)
			scores[i] /= sum;
	}

	private static void fillToken(float[] tok, int t) {
		for (int i = 0; i < tok.length; i++)
			tok[i] = ((t * 31 + i) & 0xff) * 0.001f;
	}

	public static String formatReport(List<Cell> cells) {
		StringBuilder sb = new StringBuilder();
		sb.append("# Gather-tax microbench\n\n");
		sb.append("Geometry: TinyLlama-like GQA heads=").append(NUM_HEADS)
				.append(" kvHeads=").append(NUM_KV_HEADS)
				.append(" headDim=").append(HEAD_DIM)
				.append(" kvDim=").append(KV_DIM).append(".\n");
		sb.append("Dense F16 attention has no gather; paged path gathers K+V then runs the same GQA.\n");
		sb.append("Gate: gather ≤ ~").append((int) GATE_MAX_GATHER_PCT)
				.append("% of (gather+attn) at batch ").append(GATE_BATCH)
				.append(" / ctx ").append(GATE_CTX).append(".\n\n");
		sb.append("| ctx | batch | page | dense_attn_ms | gather_ms | paged_attn_ms | gather% | paged_vs_dense% |\n");
		sb.append("|----:|------:|-----:|-------------:|----------:|--------------:|--------:|----------------:|\n");
		for (Cell c : cells) {
			sb.append(String.format(Locale.ROOT,
					"| %d | %d | %d | %.3f | %.3f | %.3f | %.2f | %.2f |\n",
					c.ctx(), c.batch(), c.pageSize(),
					c.denseAttnNs() / 1e6,
					c.gatherNs() / 1e6,
					c.pagedAttnNs() / 1e6,
					c.gatherPctOfPaged(),
					c.pagedOverDensePct()));
		}
		Cell gate = findCell(cells, GATE_CTX, GATE_BATCH, cells.isEmpty() ? 16 : cells.get(0).pageSize());
		sb.append('\n');
		if (gate != null) {
			boolean ok = gate.gatherPctOfPaged() <= GATE_MAX_GATHER_PCT;
			sb.append("**Budget decision (page=").append(gate.pageSize()).append("):** ")
					.append(ok ? "PASS" : "FAIL")
					.append(String.format(Locale.ROOT, " — gather %.2f%% at ctx=%d batch=%d.%n",
							gate.gatherPctOfPaged(), gate.ctx(), gate.batch()));
		}
		return sb.toString();
	}

	static Cell findCell(List<Cell> cells, int ctx, int batch, int pageSize) {
		for (Cell c : cells) {
			if (c.ctx() == ctx && c.batch() == batch && c.pageSize() == pageSize)
				return c;
		}
		for (Cell c : cells) {
			if (c.ctx() == ctx && c.batch() == batch)
				return c;
		}
		return null;
	}

	private static int[] parseInts(String csv) {
		String[] parts = csv.split(",");
		int[] out = new int[parts.length];
		for (int i = 0; i < parts.length; i++)
			out[i] = Integer.parseInt(parts[i].strip());
		return out;
	}
}
