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
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * How long reclaiming idle KV pages takes, against the budget it may add to a decode step.
 *
 * <p>Each cell builds one model's K and V page pools at ctx {@value #GATE_CTX}, batch {@value #GATE_BATCH}, page
 * {@value #GATE_PAGE}: {@code batch} live requests and {@code batch} finished ones, their tables built interleaved
 * so that the finished requests' idle pages sit among the live ones in every pool, then the finished requests
 * released. The model sets the page count (layers x requests x ctx / page, K and V). A row stores {@code kvDim}
 * floats, small by default: the pass drops the idle pages' storage and touches none of their bytes, so its cost is
 * per page and the row width only sets the heap the bench needs.
 *
 * <p>Columns: {@code pass_ms}, one reclaim pass releasing every idle page of both pools timed on its own, median and
 * max over the repetitions: what a decode step would wait if the pass ran on the decode thread, which the server
 * does not do; {@code hold_us}, the pass divided by its lock holds of at most {@link KvBlockPool#RECLAIM_CHUNK}
 * pages; {@code step_ms}, the KV-write half of a decode step (one row appended to every live table, the part that
 * takes the pools' lock), p50 and max over {@code steps} steps alone, and p50 and max over the steps that overlapped
 * a pass running on another thread, as the server runs it. {@code added_ms} is the longest overlapping step less the
 * median step alone: the most a pass added to one decode step. Gate: {@code added_ms} at most
 * {@value #GATE_MAX_ADDED_MS} ms in every cell.
 */
public final class KvReclaimPauseMicrobench {

	public static final int GATE_CTX = 8192;
	public static final int GATE_BATCH = 8;
	public static final int GATE_PAGE = 16;
	/** The most a reclaim pass may add to a decode step, ms. */
	public static final double GATE_MAX_ADDED_MS = 5.0;

	/** A model's layer count, by name. */
	public record Model(String name, int layers) {
	}

	/** The sweep models: TinyLlama 1.1B, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B. */
	public static final List<Model> DEFAULT_MODELS = List.of(new Model("tinyllama", 22), new Model("qwen2.5-3b", 36),
			new Model("phi-3.5-mini", 32), new Model("mistral-7b", 32));

	public record Cell(String model, int layers, int ctx, int batch, int pageSize, int kvDim, long idlePages,
			int holds, double passMedianMs, double passMaxMs, double stepP50Ms, double stepMaxMs,
			double stepP50ContendedMs, double stepMaxContendedMs, int overlappingSteps) {

		public double holdUs() {
			return holds == 0 ? 0 : passMedianMs * 1000.0 / holds;
		}

		/** The longest step that overlapped a pass, less the median step alone. */
		public double addedMs() {
			return stepMaxContendedMs - stepP50Ms;
		}
	}

	private KvReclaimPauseMicrobench() {
	}

	public static void main(String[] args) throws Exception {
		List<Model> models = DEFAULT_MODELS;
		int ctx = GATE_CTX;
		int batch = GATE_BATCH;
		int page = GATE_PAGE;
		int kvDim = 16;
		int reps = 5;
		int steps = 256;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--models" -> models = parseModels(args[++i]);
			case "--ctx" -> ctx = Integer.parseInt(args[++i]);
			case "--batch" -> batch = Integer.parseInt(args[++i]);
			case "--page-size" -> page = Integer.parseInt(args[++i]);
			case "--kv-dim" -> kvDim = Integer.parseInt(args[++i]);
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--steps" -> steps = Integer.parseInt(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			case "--help" -> {
				System.out.println("KvReclaimPauseMicrobench [--models name:layers,...] [--ctx N] [--batch N]"
						+ " [--page-size N] [--kv-dim N] [--reps N] [--steps N] [--out report.md]");
				return;
			}
			default -> throw new IllegalArgumentException("unknown arg: " + args[i]);
			}
		}
		List<Cell> cells = new ArrayList<>();
		for (Model m : models) {
			Cell c = run(m, ctx, batch, page, kvDim, reps, steps);
			cells.add(c);
			System.err.printf(Locale.ROOT, "[kv-reclaim] %s: %d idle pages, pass max %.3f ms, added to a step %.3f ms%n",
					m.name(), c.idlePages(), c.passMaxMs(), c.addedMs());
		}
		String report = formatReport(cells);
		System.out.print(report);
		if (out != null) {
			Files.createDirectories(out.toAbsolutePath().getParent());
			try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
				pw.print(report);
			}
		}
		double worst = cells.stream().mapToDouble(Cell::addedMs).max().orElse(0);
		System.err.printf(Locale.ROOT, "[kv-reclaim] gate: most added to a decode step %.3f ms (budget <= %.1f ms)%n",
				worst, GATE_MAX_ADDED_MS);
		if (worst > GATE_MAX_ADDED_MS)
			System.exit(2);
	}

	/** One cell; package-private for the test, which runs it small. */
	static Cell run(Model m, int ctx, int batch, int page, int kvDim, int reps, int steps) throws Exception {
		KvBlockPool[] pools = { new KvBlockPool(page, kvDim, KvElementType.F16),
				new KvBlockPool(page, kvDim, KvElementType.F16) };
		int tablesPerRequest = m.layers() * pools.length;
		KvPageTable[][] live = new KvPageTable[batch][tablesPerRequest];
		KvPageTable[][] done = new KvPageTable[batch][tablesPerRequest];
		float[] row = new float[kvDim];
		Arrays.fill(row, 0.5f);
		for (int r = 0; r < batch; r++)
			for (int t = 0; t < tablesPerRequest; t++) {
				live[r][t] = new KvPageTable(pools[t % pools.length]);
				done[r][t] = new KvPageTable(pools[t % pools.length]);
			}
		// Interleaved, so that the finished requests' pages sit among the live ones.
		for (int p = 0; p < ctx; p++)
			for (int r = 0; r < batch; r++)
				for (int t = 0; t < tablesPerRequest; t++) {
					live[r][t].appendToken(row);
					done[r][t].appendToken(row);
				}

		double[] pass = new double[reps];
		long idle = 0;
		int holds = 0;
		for (int rep = 0; rep < reps; rep++) {
			idle = makeIdle(done, pools, row, ctx, rep > 0);
			holds = 0;
			for (KvBlockPool pool : pools)
				holds += (pool.idlePages() + KvBlockPool.RECLAIM_CHUNK - 1) / KvBlockPool.RECLAIM_CHUNK;
			long t0 = System.nanoTime();
			long released = 0;
			for (KvBlockPool pool : pools)
				released += pool.reclaimIdle();
			pass[rep] = (System.nanoTime() - t0) / 1e6;
			if (released != idle)
				throw new IllegalStateException("released " + released + " of " + idle + " idle pages");
		}

		double[] alone = new double[steps];
		for (int s = 0; s < steps; s++)
			alone[s] = step(live, row);
		truncate(live, steps);
		List<Double> overlapping = new ArrayList<>();
		for (int rep = 0; rep < reps; rep++) {
			makeIdle(done, pools, row, ctx, true);
			AtomicBoolean running = new AtomicBoolean(true);
			Thread reclaimer = new Thread(() -> {
				for (KvBlockPool pool : pools)
					pool.reclaimIdle();
				running.set(false);
			}, "kv-reclaim-pass");
			reclaimer.start();
			int n = 0;
			while (running.get() && n < steps) {
				overlapping.add(step(live, row));
				n++;
			}
			reclaimer.join();
			truncate(live, n);
		}
		for (KvPageTable[] tables : live)
			for (KvPageTable t : tables)
				t.release();
		Arrays.sort(pass);
		Arrays.sort(alone);
		double[] over = overlapping.stream().mapToDouble(Double::doubleValue).sorted().toArray();
		return new Cell(m.name(), m.layers(), ctx, batch, page, kvDim, idle, holds, pass[reps / 2], pass[reps - 1],
				percentile(alone, 0.5), alone[alone.length - 1], over.length == 0 ? 0 : percentile(over, 0.5),
				over.length == 0 ? 0 : over[over.length - 1], over.length);
	}

	/**
	 * Releases the finished requests' tables (written again first when {@code refill}) and starts a reclaim period
	 * with every one of their pages idle; returns the idle pages.
	 */
	private static long makeIdle(KvPageTable[][] done, KvBlockPool[] pools, float[] row, int ctx, boolean refill) {
		if (refill)
			for (KvPageTable[] tables : done)
				for (KvPageTable t : tables)
					for (int p = 0; p < ctx; p++)
						t.appendToken(row);
		for (KvPageTable[] tables : done)
			for (KvPageTable t : tables)
				t.release();
		long idle = 0;
		for (KvBlockPool pool : pools) {
			pool.reclaimIdle(); // releases nothing: the pages became idle in this period
			idle += pool.idlePages();
		}
		return idle;
	}

	/** One decode step's KV writes, ms: a row appended to every live table. */
	private static double step(KvPageTable[][] live, float[] row) {
		long t0 = System.nanoTime();
		for (KvPageTable[] tables : live)
			for (KvPageTable t : tables)
				t.appendToken(row);
		return (System.nanoTime() - t0) / 1e6;
	}

	private static void truncate(KvPageTable[][] live, int steps) {
		for (KvPageTable[] tables : live)
			for (KvPageTable t : tables)
				t.truncate(t.seqLen() - steps);
	}

	private static double percentile(double[] sorted, double q) {
		return sorted[Math.min(sorted.length - 1, (int) Math.floor(q * sorted.length))];
	}

	static String formatReport(List<Cell> cells) {
		StringBuilder sb = new StringBuilder();
		sb.append("# KV page reclaim pause\n\n");
		sb.append(String.format(Locale.ROOT,
				"Budget: a reclaim pass adds at most %.1f ms to any decode step it overlaps (added_ms: the longest"
						+ " overlapping step less the median step alone).%n%n",
				GATE_MAX_ADDED_MS));
		sb.append("| model | layers | ctx | batch | page | kv_dim | idle_pages | pass_ms median | pass_ms max |"
				+ " hold_us | step_ms p50 | step_ms max | overlapping steps | step_ms p50, overlapping |"
				+ " step_ms max, overlapping | added_ms |\n");
		sb.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
		for (Cell c : cells)
			sb.append(String.format(Locale.ROOT,
					"| %s | %d | %d | %d | %d | %d | %d | %.3f | %.3f | %.1f | %.3f | %.3f | %d | %.3f | %.3f | %.3f |%n",
					c.model(), c.layers(), c.ctx(), c.batch(), c.pageSize(), c.kvDim(), c.idlePages(),
					c.passMedianMs(), c.passMaxMs(), c.holdUs(), c.stepP50Ms(), c.stepMaxMs(), c.overlappingSteps(),
					c.stepP50ContendedMs(), c.stepMaxContendedMs(), c.addedMs()));
		return sb.toString();
	}

	private static List<Model> parseModels(String s) {
		List<Model> out = new ArrayList<>();
		for (String part : s.split(",")) {
			String[] kv = part.split(":");
			out.add(new Model(kv[0].strip(), Integer.parseInt(kv[1].strip())));
		}
		return out;
	}
}
