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

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Measures what one GPU RMS norm costs today, when every call stages its own
 * activation to the device and reads the result back.
 *
 * <p>{@link CudaRmsNorm} is built, tested and deliberately left unconstructed by
 * {@link LlamaTransformerHandler}, because a live decode comparison found it
 * slower than the scalar CPU path it replaces. That finding is the premise of
 * the activation-residency work, so it is re-established here as a repeatable
 * measurement rather than carried forward as a remembered number.
 *
 * <p>Two lanes run at each width: {@link #LANE_CPU_SCALAR}, the scalar
 * {@code rmsNormInto} the handler actually uses, and
 * {@link #LANE_GPU_ROUND_TRIP}, today's per-call host-to-device upload, kernel
 * launch and device-to-host download. Ratios are reported as CPU-scalar median
 * over lane median, so a lane at least as fast as the CPU path reads at or
 * above {@code 1.00x} and a slower one reads below it.
 *
 * <p>Both widths matter and are reported separately. Decode runs one row at a
 * time, where a fixed per-call launch cost dominates; prefill runs a wide batch,
 * where the cost is the megabytes staged each way. A result at one width says
 * nothing about the other.
 *
 * <p>Repetition dispersion is reported per row, and a row whose repetitions
 * disagree by more than {@link #MAX_SCORABLE_SPREAD_PCT} of their median is
 * marked unscorable rather than quietly averaged into a conclusion.
 */
public final class RmsNormRoundTripMicrobench {

	/** Hidden size of the smallest model in the standing sweep. */
	public static final int DEFAULT_DIM = 2048;

	/** One row per step: the width a decode actually runs at. */
	public static final int DECODE_BATCH = 1;

	/** A wide prompt window: the width the residency primitive's largest consumer runs at. */
	public static final int PREFILL_BATCH = 512;

	/** The scalar CPU path {@code LlamaTransformerHandler} uses today. */
	public static final String LANE_CPU_SCALAR = "cpu-scalar";

	/** One GPU norm per call, staging the activation both ways itself. */
	public static final String LANE_GPU_ROUND_TRIP = "gpu-round-trip";

	/** A row whose repetitions disagree by more than this share of their median is not scorable. */
	public static final double MAX_SCORABLE_SPREAD_PCT = 15.0;

	/** Largest absolute divergence tolerated between the GPU and CPU lanes' outputs. */
	public static final double CORRECTNESS_TOLERANCE = 1e-4;

	/**
	 * Share of the device above which retention cannot be this harness's own
	 * scratch. Far above the few megabytes it allocates, far below the share a
	 * competing process on the device takes.
	 */
	public static final double UNACCOUNTED_RETENTION_SHARE = 0.05;

	/**
	 * Warm-up per lane before calibration. Measured: at 500 ms the decode GPU
	 * lane's repetitions still spanned 37.9% of their median and the row was not
	 * scorable; at this value they span under 3%.
	 */
	public static final long DEFAULT_WARMUP_MS = 3000;

	/** Measurement window per repetition, well clear of timer and scheduler noise. */
	public static final long DEFAULT_TARGET_MS = 800;

	private static final float EPS = 1e-5f;

	/** Consumes lane output so the measured work cannot be optimised away. */
	private static volatile double sink;

	private RmsNormRoundTripMicrobench() {
	}

	/** One lane's per-repetition wall time, in milliseconds per call. */
	public record Reading(String lane, double[] repMs) {

		public Reading {
			if (lane == null || lane.isBlank())
				throw new IllegalArgumentException("lane must not be blank");
			if (repMs == null || repMs.length == 0)
				throw new IllegalArgumentException("lane " + lane + " needs at least one repetition");
			repMs = repMs.clone();
		}

		@Override
		public double[] repMs() {
			return repMs.clone();
		}

		/** Median repetition, which is what every ratio is taken from. */
		public double medianMs() {
			double[] sorted = repMs.clone();
			Arrays.sort(sorted);
			int n = sorted.length;
			return (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
		}

		public double minMs() {
			double lo = repMs[0];
			for (double v : repMs)
				lo = Math.min(lo, v);
			return lo;
		}

		public double maxMs() {
			double hi = repMs[0];
			for (double v : repMs)
				hi = Math.max(hi, v);
			return hi;
		}

		/** Repetition disagreement as a percentage of the median. */
		public double spreadPct() {
			double median = medianMs();
			if (median <= 0)
				return 0;
			return 100.0 * (maxMs() - minMs()) / median;
		}

		/** Whether this row's repetitions agree closely enough to be read as a result. */
		public boolean scorable() {
			return spreadPct() <= MAX_SCORABLE_SPREAD_PCT;
		}
	}

	/** Every lane measured at one width. */
	public record Cell(String label, int batch, int dim, List<Reading> lanes) {

		public Cell {
			if (label == null || label.isBlank())
				throw new IllegalArgumentException("label must not be blank");
			if (batch <= 0 || dim <= 0)
				throw new IllegalArgumentException("batch and dim must be positive");
			if (lanes == null || lanes.isEmpty())
				throw new IllegalArgumentException("width " + label + " needs at least one lane");
			lanes = List.copyOf(lanes);
		}

		/** The named lane, or {@code null} when this cell does not carry it. */
		public Reading lane(String name) {
			for (Reading r : lanes)
				if (r.lane().equals(name))
					return r;
			return null;
		}

		/**
		 * CPU-scalar median over {@code name}'s median: at or above {@code 1.0}
		 * when that lane is at least as fast as the scalar CPU path.
		 */
		public double speedupOverCpu(String name) {
			Reading target = lane(name);
			if (target == null)
				throw new IllegalArgumentException("no lane named " + name + " at width " + label);
			Reading cpu = lane(LANE_CPU_SCALAR);
			if (cpu == null)
				throw new IllegalStateException("width " + label + " has no " + LANE_CPU_SCALAR
						+ " lane to score against");
			return cpu.medianMs() / target.medianMs();
		}
	}

	/**
	 * Free device memory either side of a run, so retention is visible without a crash.
	 *
	 * <p>{@code totalBytes} is carried so a reader can tell whether the device was
	 * shared. This figure is device-wide, not per-process: another process taking
	 * or releasing memory during the run lands in it, and the harness cannot tell
	 * that apart from its own retention. Read it only from an otherwise idle
	 * device, and use {@link #deviceBusy()} to see whether that held.
	 */
	public record DeviceMemoryDelta(long freeBeforeBytes, long freeAfterBytes, long totalBytes) {

		/** Device memory not returned, or zero when the run gave back at least what it took. */
		public long leakedBytes() {
			return Math.max(0L, freeBeforeBytes - freeAfterBytes);
		}

		/**
		 * Whether the device answered the query at all. A failed
		 * {@code memGetInfo} reports zero bytes, which would otherwise read as a
		 * clean run rather than as no reading.
		 */
		public boolean queried() {
			return freeBeforeBytes > 0 || freeAfterBytes > 0;
		}

		/**
		 * Whether more memory went missing than this harness could plausibly be
		 * holding. Its device scratch is two activation buffers and a weight
		 * vector -- single-digit megabytes at the widest batch measured -- so a
		 * retention above {@link #UNACCOUNTED_RETENTION_SHARE} of the device means
		 * something else on it took memory during the run, and the timings shared
		 * the device as well. Observed in practice when a stray test process held
		 * the GPU alongside a run.
		 */
		public boolean retentionUnaccounted() {
			return totalBytes > 0 && leakedBytes() > (long) (totalBytes * UNACCOUNTED_RETENTION_SHARE);
		}
	}

	/** One width's measurement, with the divergence observed against the CPU lane. */
	public record WidthResult(Cell cell, double maxAbsDiff) {
	}

	public static void main(String[] args) throws IOException {
		int dim = DEFAULT_DIM;
		int[] batches = { DECODE_BATCH, PREFILL_BATCH };
		int reps = 3;
		long warmupMs = DEFAULT_WARMUP_MS;
		long targetMs = DEFAULT_TARGET_MS;
		Path out = null;
		String host = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--dim" -> dim = Integer.parseInt(args[++i]);
			case "--batch" -> batches = parseInts(args[++i]);
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--warmup-ms" -> warmupMs = Long.parseLong(args[++i]);
			case "--target-ms" -> targetMs = Long.parseLong(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			case "--host" -> host = args[++i];
			case "--help" -> {
				usage();
				return;
			}
			default -> throw new IllegalArgumentException("unknown arg: " + args[i]);
			}
		}

		if (!CudaAvailability.isAvailable()) {
			System.err.println("[rmsnorm-roundtrip] no CUDA device available - this harness measures a GPU path "
					+ "and will not report a ratio without one");
			System.exit(2);
		}

		boolean wrongBackend = false;
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaRmsNorm gpu = CudaRmsNorm.tryCreate(ctx);
			if (gpu == null) {
				System.err.println("[rmsnorm-roundtrip] backend is " + ctx.backendLabel()
						+ ", not cuda - the GPU norm path exists only on cuda");
				wrongBackend = true;
			} else {
				String report = runAll(ctx, gpu, batches, dim, warmupMs, targetMs, reps,
						host == null ? CudaAvailability.deviceName(0) : host);
				System.out.print(report);
				if (out != null)
					write(out, report);
			}
		}
		if (wrongBackend)
			System.exit(2);
	}

	/** Measures every width and renders the report, with device memory read either side. */
	private static String runAll(GpuContext ctx, CudaRmsNorm gpu, int[] batches, int dim,
			long warmupMs, long targetMs, int reps, String host) {
		long[] before = ctx.bindings().memGetInfo(ctx.deviceIndex());
		List<WidthResult> results = new ArrayList<>();
		List<Cell> cells = new ArrayList<>();
		for (int batch : batches) {
			WidthResult r = measure(gpu, labelFor(batch), batch, dim, warmupMs, targetMs, reps);
			results.add(r);
			cells.add(r.cell());
		}
		long[] after = ctx.bindings().memGetInfo(ctx.deviceIndex());
		return formatReport(cells, new DeviceMemoryDelta(before[0], after[0], before[1]), host)
				+ formatCorrectness(results);
	}

	private static void write(Path out, String report) throws IOException {
		Path parent = out.getParent();
		Files.createDirectories(parent == null ? Path.of(".") : parent);
		try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
			pw.print(report);
		}
		System.err.println("[rmsnorm-roundtrip] wrote " + out.toAbsolutePath());
	}

	/** Runs both lanes at one width and checks the GPU lane against the CPU lane's output. */
	static WidthResult measure(CudaRmsNorm gpu, String label, int batch, int dim,
			long warmupMs, long targetMs, int reps) {
		Random rnd = new Random(1);
		float[][] x = new float[batch][dim];
		for (float[] row : x)
			for (int i = 0; i < dim; i++)
				row[i] = (rnd.nextFloat() * 2f) - 1f;
		float[] weight = new float[dim];
		for (int i = 0; i < dim; i++)
			weight[i] = (rnd.nextFloat() * 2f) - 1f;

		float[][] cpuOut = new float[batch][dim];
		float[][] gpuOut = new float[batch][dim];

		Runnable cpuLane = () -> {
			for (int b = 0; b < batch; b++)
				LlamaTransformerHandler.rmsNormInto(x[b], weight, EPS, cpuOut[b]);
			sink += cpuOut[0][0];
		};
		Runnable gpuLane = () -> {
			if (!gpu.normalizeBatch(x, weight, EPS, gpuOut))
				throw new IllegalStateException("the GPU norm kernel failed to load - no ratio can be reported");
			sink += gpuOut[0][0];
		};

		System.err.printf(Locale.ROOT, "[rmsnorm-roundtrip] %s batch=%d dim=%d%n", label, batch, dim);
		Reading cpu = measureLane(LANE_CPU_SCALAR, cpuLane, warmupMs, targetMs, reps);
		Reading gpuReading = measureLane(LANE_GPU_ROUND_TRIP, gpuLane, warmupMs, targetMs, reps);

		double maxAbsDiff = 0;
		for (int b = 0; b < batch; b++)
			for (int i = 0; i < dim; i++)
				maxAbsDiff = Math.max(maxAbsDiff, Math.abs(cpuOut[b][i] - gpuOut[b][i]));
		if (maxAbsDiff > CORRECTNESS_TOLERANCE)
			throw new IllegalStateException("GPU norm diverged from the scalar CPU path by " + maxAbsDiff
					+ " at width " + label + ", above the " + CORRECTNESS_TOLERANCE
					+ " tolerance - the timing below would be measuring a different computation");

		return new WidthResult(new Cell(label, batch, dim, List.of(cpu, gpuReading)), maxAbsDiff);
	}

	static Reading measureLane(String lane, Runnable op, long warmupMs, long targetMs, int reps) {
		warmUp(op, warmupMs * 1_000_000L);
		int iters = calibrateIters(op, targetMs * 1_000_000L);
		double[] repMs = new double[reps];
		for (int r = 0; r < reps; r++) {
			long t0 = System.nanoTime();
			for (int i = 0; i < iters; i++)
				op.run();
			long elapsed = System.nanoTime() - t0;
			repMs[r] = (elapsed / 1e6) / iters;
		}
		Reading reading = new Reading(lane, repMs);
		System.err.printf(Locale.ROOT, "  %-16s %d iters/rep  median %.4f ms  spread %.1f%%%n",
				lane, iters, reading.medianMs(), reading.spreadPct());
		return reading;
	}

	/** Runs the lane until the JIT has settled and any device context is warm. */
	private static void warmUp(Runnable op, long warmupNanos) {
		long deadline = System.nanoTime() + warmupNanos;
		do {
			op.run();
		} while (System.nanoTime() < deadline);
	}

	/** Sizes each repetition to a measurement window well clear of timer noise. */
	private static int calibrateIters(Runnable op, long targetNanos) {
		long t0 = System.nanoTime();
		op.run();
		long one = Math.max(1L, System.nanoTime() - t0);
		long iters = targetNanos / one;
		return (int) Math.max(1L, Math.min(100_000L, iters));
	}

	static String labelFor(int batch) {
		return batch == DECODE_BATCH ? "decode" : "prefill";
	}

	/**
	 * Renders the per-width lane table. Kept separate from measurement so the
	 * scoring this tier's decision rests on is testable without a device.
	 */
	public static String formatReport(List<Cell> cells, DeviceMemoryDelta memory, String host) {
		StringBuilder sb = new StringBuilder();
		sb.append("# RMS-norm host-round-trip microbench\n\n");
		sb.append("Host: ").append(host).append("\n\n");
		sb.append("`vs CPU scalar` is the scalar CPU median over the lane median: at or above `1.00x`\n");
		sb.append("the lane is at least as fast as the path it would replace. A row whose repetitions\n");
		sb.append("disagree by more than ").append(fmt(MAX_SCORABLE_SPREAD_PCT)).append("% of their median is not scorable.\n\n");
		sb.append("| width | batch | dim | lane | median ms | min ms | max ms | spread % | vs CPU scalar | scorable |\n");
		sb.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n");
		for (Cell c : cells)
			for (Reading r : c.lanes())
				sb.append(String.format(Locale.ROOT,
						"| %s | %d | %d | %s | %.4f | %.4f | %.4f | %.1f | %.2fx | %s |%n",
						c.label(), c.batch(), c.dim(), r.lane(), r.medianMs(), r.minMs(), r.maxMs(),
						r.spreadPct(), c.speedupOverCpu(r.lane()), r.scorable() ? "yes" : "no"));
		if (!memory.queried()) {
			sb.append("\nFree VRAM: the device did not answer the memory query, so this run makes no "
					+ "claim about device-memory retention.\n");
		} else {
			sb.append("\nFree VRAM before: ").append(memory.freeBeforeBytes()).append(" bytes, after: ")
					.append(memory.freeAfterBytes()).append(" bytes, of ").append(memory.totalBytes())
					.append(" total; not returned: ").append(memory.leakedBytes()).append(" bytes.\n");
			if (memory.retentionUnaccounted())
				sb.append("That is far more than this harness allocates, so something else on the device "
						+ "took memory during the run and the timings above shared it. "
						+ "Re-run on an idle device.\n");
		}
		return sb.toString();
	}

	private static String formatCorrectness(List<WidthResult> results) {
		StringBuilder sb = new StringBuilder("\nLargest divergence from the scalar CPU path, per width ");
		sb.append("(tolerance ").append(CORRECTNESS_TOLERANCE).append("):\n\n");
		for (WidthResult r : results)
			sb.append(String.format(Locale.ROOT, "- %s (batch %d): %.3e%n",
					r.cell().label(), r.cell().batch(), r.maxAbsDiff()));
		return sb.toString();
	}

	private static String fmt(double v) {
		return String.format(Locale.ROOT, "%.0f", v);
	}

	static int[] parseInts(String csv) {
		String[] parts = csv.split(",");
		int[] out = new int[parts.length];
		for (int i = 0; i < parts.length; i++)
			out[i] = Integer.parseInt(parts[i].trim());
		return out;
	}

	private static void usage() {
		System.out.println("""
				Usage: RmsNormRoundTripMicrobench [options]
				  --dim N           hidden size (default 2048)
				  --batch A,B       widths to measure (default 1,512)
				  --reps N          repetitions per lane (default 3)
				  --warmup-ms N     warm-up per lane before calibration (default 3000)
				  --target-ms N     measurement window per repetition (default 800)
				  --out PATH        write the report to PATH as well as stdout
				  --host LABEL      host label for the report (default: the CUDA device name)
				""");
	}
}
