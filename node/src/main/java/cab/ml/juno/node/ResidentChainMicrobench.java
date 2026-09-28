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

import cab.ml.juno.node.RmsNormRoundTripMicrobench.Cell;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.DeviceMemoryDelta;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.Reading;
import cab.ml.juno.node.RmsNormRoundTripMicrobench.WidthResult;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static cab.ml.juno.node.RmsNormRoundTripMicrobench.CORRECTNESS_TOLERANCE;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.DECODE_BATCH;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.DEFAULT_DIM;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.DEFAULT_TARGET_MS;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.DEFAULT_WARMUP_MS;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.LANE_CPU_SCALAR;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.MAX_SCORABLE_SPREAD_PCT;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.PREFILL_BATCH;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.labelFor;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.measureLane;
import static cab.ml.juno.node.RmsNormRoundTripMicrobench.parseInts;

/**
 * Measures a two-operation chain - RMS norm, then RoPE on the normalized rows -
 * with the activation kept on the device between the operations, against the
 * same two operations each doing its own host round trip, and against the
 * scalar CPU path.
 *
 * <p>This is the measurement the activation-residency work turns on.
 * {@link RmsNormRoundTripMicrobench} established that a single GPU operation
 * which stages its own activation both ways is slower than the scalar CPU path
 * at every width. The claim under test here is that the cost is the round trip,
 * not the device work, so removing the round trip between two operations is
 * what makes the chain pay. Four lanes at each width:
 * <ul>
 *   <li>{@code cpu-scalar}: {@code rmsNormInto}, then {@code rope} in place, per
 *       row - the path {@link LlamaTransformerHandler} runs today;</li>
 *   <li>{@link #LANE_OP_AT_A_TIME}: upload, norm, download; upload, RoPE,
 *       download. The norm weight and the RoPE table are already on the device,
 *       so the only difference from the resident chain is the host round trip
 *       between the two operations;</li>
 *   <li>{@link #LANE_RESIDENT_CHAIN}: upload once, norm, RoPE, download once -
 *       what a residency region holding exactly these two operations costs;</li>
 *   <li>{@link #LANE_DEVICE_ONLY}: norm and RoPE on an activation already on the
 *       device, then wait for them - the marginal cost of the two operations
 *       inside a longer region whose entry and exit are paid elsewhere.</li>
 * </ul>
 *
 * <p>Two ratios are reported. Speedup against the CPU chain uses
 * {@link RmsNormRoundTripMicrobench}'s orientation (CPU median over lane median,
 * at or above {@code 1.00x} when the lane is at least as fast). Cost against
 * op-at-a-time is the other way round (lane median over op-at-a-time median,
 * below {@code 1.00} when the chain is cheaper), because it answers "what share
 * of the op-at-a-time cost is left once the round trip between the operations
 * is gone".
 *
 * <p>Every GPU lane dispatches through the same {@link ResidentActivation} and
 * {@link KernelParams} path, so the ratio between them isolates the round trip.
 * The ratio against {@link RmsNormRoundTripMicrobench}'s round-trip lane does
 * not: that lane also pays per-call weight upload and argument boxing.
 */
public final class ResidentChainMicrobench {

	/** The head size of every model in the standing sweep except the two with 128-wide heads. */
	public static final int DEFAULT_HEAD_DIM = 64;

	/** The RoPE base of the smallest sweep model. */
	public static final float DEFAULT_THETA = 10000f;

	/** The first decode step after a prefill window of {@link RmsNormRoundTripMicrobench#PREFILL_BATCH} tokens. */
	public static final int DEFAULT_DECODE_POS = 512;

	/** Each operation stages its own activation both ways. */
	public static final String LANE_OP_AT_A_TIME = "gpu-op-at-a-time";

	/** One upload, both operations on the device, one download. */
	public static final String LANE_RESIDENT_CHAIN = "gpu-resident-chain";

	/** Both operations on an activation already on the device, no transfer. */
	public static final String LANE_DEVICE_ONLY = "gpu-device-only";

	private static final float EPS = 1e-5f;

	/** Consumes lane output so the measured work cannot be optimised away. */
	private static volatile double sink;

	private ResidentChainMicrobench() {
	}

	/**
	 * {@code lane}'s median over the op-at-a-time median: below {@code 1.0} when
	 * the lane costs less than the same two operations with a host round trip
	 * between them.
	 */
	public static double costOverOpAtATime(Cell cell, String lane) {
		Reading target = cell.lane(lane);
		if (target == null)
			throw new IllegalArgumentException("no lane named " + lane + " at width " + cell.label());
		Reading opAtATime = cell.lane(LANE_OP_AT_A_TIME);
		if (opAtATime == null)
			throw new IllegalStateException("width " + cell.label() + " has no " + LANE_OP_AT_A_TIME
					+ " lane to score the chain against");
		return target.medianMs() / opAtATime.medianMs();
	}

	public static void main(String[] args) throws IOException {
		int dim = DEFAULT_DIM;
		int headDim = DEFAULT_HEAD_DIM;
		float theta = DEFAULT_THETA;
		int decodePos = DEFAULT_DECODE_POS;
		int[] batches = { DECODE_BATCH, PREFILL_BATCH };
		int reps = 3;
		long warmupMs = DEFAULT_WARMUP_MS;
		long targetMs = DEFAULT_TARGET_MS;
		Path out = null;
		String host = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--dim" -> dim = Integer.parseInt(args[++i]);
			case "--head-dim" -> headDim = Integer.parseInt(args[++i]);
			case "--theta" -> theta = Float.parseFloat(args[++i]);
			case "--decode-pos" -> decodePos = Integer.parseInt(args[++i]);
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
		if (dim % headDim != 0)
			throw new IllegalArgumentException("--dim " + dim + " is not a whole number of --head-dim " + headDim);

		if (!CudaAvailability.isAvailable()) {
			System.err.println("[resident-chain] no CUDA device available - this harness measures a GPU path "
					+ "and will not report a ratio without one");
			System.exit(2);
		}

		boolean unavailable = false;
		try (GpuContext ctx = GpuContext.init(0)) {
			CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
			CudaRope rope = CudaRope.tryCreate(ctx, headDim, theta);
			if (norm == null || rope == null || RmsNormKernel.tryLoad() == null || RopeKernel.tryLoad() == null) {
				System.err.println("[resident-chain] backend is " + ctx.backendLabel()
						+ " or a kernel failed to load - the resident path exists only on cuda");
				unavailable = true;
				if (rope != null)
					rope.close();
			} else {
				try (rope) {
					String report = runAll(ctx, norm, rope, batches, dim, decodePos, warmupMs, targetMs, reps,
							host == null ? CudaAvailability.deviceName(0) : host);
					System.out.print(report);
					if (out != null)
						write(out, report);
				}
			}
		}
		if (unavailable)
			System.exit(2);
	}

	/** Measures every width and renders the report, with device memory read either side. */
	private static String runAll(GpuContext ctx, CudaRmsNorm norm, CudaRope rope, int[] batches, int dim,
			int decodePos, long warmupMs, long targetMs, int reps, String host) {
		// First use of a stream creates driver state that is kept for the process;
		// take it before the reading so the retention figure is this harness's own.
		ResidentChain.open(ctx).close();
		long[] before = ctx.bindings().memGetInfo(ctx.deviceIndex());
		List<WidthResult> results = new ArrayList<>();
		List<Cell> cells = new ArrayList<>();
		for (int batch : batches) {
			int startPos = batch == DECODE_BATCH ? decodePos : 0;
			WidthResult r = measure(ctx, norm, rope, labelFor(batch), batch, dim, startPos, warmupMs, targetMs, reps);
			results.add(r);
			cells.add(r.cell());
		}
		long[] after = ctx.bindings().memGetInfo(ctx.deviceIndex());
		return formatReport(cells, new DeviceMemoryDelta(before[0], after[0], before[1]), host, decodePos,
				rope.headDim(), rope.ropeTheta()) + formatCorrectness(results);
	}

	/**
	 * Runs the four lanes at one width, rows at positions {@code startPos} onwards,
	 * and checks every GPU lane's output against the CPU lane's.
	 */
	static WidthResult measure(GpuContext ctx, CudaRmsNorm norm, CudaRope rope, String label, int batch, int dim,
			int startPos, long warmupMs, long targetMs, int reps) {
		int headDim = rope.headDim();
		float theta = rope.ropeTheta();
		int nHeads = dim / headDim;
		Random rnd = new Random(1);
		float[][] x = new float[batch][dim];
		for (float[] row : x)
			for (int i = 0; i < dim; i++)
				row[i] = (rnd.nextFloat() * 2f) - 1f;
		float[] weight = new float[dim];
		for (int i = 0; i < dim; i++)
			weight[i] = (rnd.nextFloat() * 2f) - 1f;

		float[][] cpuOut = new float[batch][dim];
		float[][] roundTripMid = new float[batch][dim];
		float[][] opOut = new float[batch][dim];
		float[][] chainOut = new float[batch][dim];
		float[][] deviceOut = new float[batch][dim];

		System.err.printf(Locale.ROOT, "[resident-chain] %s batch=%d dim=%d heads=%d startPos=%d%n",
				label, batch, dim, nHeads, startPos);
		Reading cpu;
		Reading op;
		Reading chained;
		Reading deviceOnly;
		try (ResidentChain chain = ResidentChain.open(ctx);
				DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, weight, 1, dim)) {
			ResidentActivation in = chain.allocate(batch, dim);
			ResidentActivation normed = chain.allocate(batch, dim);

			Runnable cpuLane = () -> {
				for (int b = 0; b < batch; b++) {
					LlamaTransformerHandler.rmsNormInto(x[b], weight, EPS, cpuOut[b]);
					LlamaTransformerHandler.rope(cpuOut[b], startPos + b, nHeads, headDim, theta);
				}
				sink += cpuOut[0][0];
			};
			Runnable opAtATimeLane = () -> {
				in.upload(x, batch);
				dispatched(norm.normalizeResident(in, w, EPS, normed));
				normed.materialize(roundTripMid);
				normed.upload(roundTripMid, batch);
				dispatched(rope.applyResident(normed, startPos));
				normed.materialize(opOut);
				sink += opOut[0][0];
			};
			Runnable chainLane = () -> {
				in.upload(x, batch);
				dispatched(norm.normalizeResident(in, w, EPS, normed));
				dispatched(rope.applyResident(normed, startPos));
				normed.materialize(chainOut);
				sink += chainOut[0][0];
			};
			Runnable deviceOnlyLane = () -> {
				dispatched(norm.normalizeResident(in, w, EPS, normed));
				dispatched(rope.applyResident(normed, startPos));
				chain.sync();
			};

			cpu = measureLane(LANE_CPU_SCALAR, cpuLane, warmupMs, targetMs, reps);
			op = measureLane(LANE_OP_AT_A_TIME, opAtATimeLane, warmupMs, targetMs, reps);
			chained = measureLane(LANE_RESIDENT_CHAIN, chainLane, warmupMs, targetMs, reps);
			in.upload(x, batch);
			deviceOnly = measureLane(LANE_DEVICE_ONLY, deviceOnlyLane, warmupMs, targetMs, reps);
			normed.materialize(deviceOut);
		}

		double maxAbsDiff = 0;
		for (float[][] gpuOut : List.of(opOut, chainOut, deviceOut))
			for (int b = 0; b < batch; b++)
				for (int i = 0; i < dim; i++)
					maxAbsDiff = Math.max(maxAbsDiff, Math.abs(cpuOut[b][i] - gpuOut[b][i]));
		if (maxAbsDiff > CORRECTNESS_TOLERANCE)
			throw new IllegalStateException("a GPU lane diverged from the scalar CPU chain by " + maxAbsDiff
					+ " at width " + label + ", above the " + CORRECTNESS_TOLERANCE
					+ " tolerance - the timing would be measuring a different computation");

		return new WidthResult(new Cell(label, batch, dim, List.of(cpu, op, chained, deviceOnly)), maxAbsDiff);
	}

	private static void dispatched(boolean ok) {
		if (!ok)
			throw new IllegalStateException("a GPU kernel failed to load - no ratio can be reported");
	}

	/**
	 * Renders the per-width lane table and the reading per width. Kept separate
	 * from measurement so what the residency result is read as is testable
	 * without a device.
	 */
	public static String formatReport(List<Cell> cells, DeviceMemoryDelta memory, String host, int decodePos,
			int headDim, float theta) {
		StringBuilder sb = new StringBuilder();
		sb.append("# Resident RMS-norm + RoPE chain microbench\n\n");
		sb.append("Host: ").append(host).append("\n\n");
		sb.append(String.format(Locale.ROOT,
				"The chain is RMS norm, then RoPE on the normalized rows, as heads of head size %d (base %.0f): "
						+ "the decode row at position %d, the prefill window at positions 0 onwards.%n%n",
				headDim, theta, decodePos));
		sb.append("`speedup vs CPU scalar` is the scalar CPU chain's median over the lane median: at or above\n");
		sb.append("`1.00x` the lane is at least as fast as the path it would replace. `cost vs op-at-a-time` is the\n");
		sb.append("lane median over the op-at-a-time median: the share of that cost left once the host round trip\n");
		sb.append("between the two operations is removed. A row whose repetitions disagree by more than ")
				.append(String.format(Locale.ROOT, "%.0f", MAX_SCORABLE_SPREAD_PCT))
				.append("% of their\nmedian is not scorable.\n\n");
		sb.append("| width | batch | dim | lane | median ms | min ms | max ms | spread % | speedup vs CPU scalar"
				+ " | cost vs op-at-a-time | scorable |\n");
		sb.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n");
		List<String> unscorable = new ArrayList<>();
		for (Cell c : cells)
			for (Reading r : c.lanes()) {
				sb.append(String.format(Locale.ROOT,
						"| %s | %d | %d | %s | %.4f | %.4f | %.4f | %.1f | %.2fx | %.2f | %s |%n",
						c.label(), c.batch(), c.dim(), r.lane(), r.medianMs(), r.minMs(), r.maxMs(), r.spreadPct(),
						c.speedupOverCpu(r.lane()), costOverOpAtATime(c, r.lane()), r.scorable() ? "yes" : "no"));
				if (!r.scorable())
					unscorable.add(c.label() + " " + r.lane());
			}

		sb.append("\nReading per width:\n\n");
		for (Cell c : cells)
			sb.append(String.format(Locale.ROOT,
					"- %s (batch %d): resident chain %.2fx the CPU scalar chain, %.2f of op-at-a-time; "
							+ "device-only %.2fx the CPU scalar chain, %.2f of op-at-a-time%n",
					c.label(), c.batch(), c.speedupOverCpu(LANE_RESIDENT_CHAIN),
					costOverOpAtATime(c, LANE_RESIDENT_CHAIN), c.speedupOverCpu(LANE_DEVICE_ONLY),
					costOverOpAtATime(c, LANE_DEVICE_ONLY)));
		if (!unscorable.isEmpty())
			sb.append("\nRows not scorable, re-run before reading them: ").append(String.join(", ", unscorable))
					.append('\n');

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
		StringBuilder sb = new StringBuilder("\nLargest divergence of any GPU lane from the scalar CPU chain, per width ");
		sb.append("(tolerance ").append(CORRECTNESS_TOLERANCE).append("):\n\n");
		for (WidthResult r : results)
			sb.append(String.format(Locale.ROOT, "- %s (batch %d): %.3e%n",
					r.cell().label(), r.cell().batch(), r.maxAbsDiff()));
		return sb.toString();
	}

	private static void write(Path out, String report) throws IOException {
		Path parent = out.getParent();
		Files.createDirectories(parent == null ? Path.of(".") : parent);
		try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
			pw.print(report);
		}
		System.err.println("[resident-chain] wrote " + out.toAbsolutePath());
	}

	private static void usage() {
		System.out.println("""
				Usage: ResidentChainMicrobench [options]
				  --dim N           hidden size (default 2048)
				  --head-dim N      RoPE head size (default 64)
				  --theta F         RoPE base (default 10000)
				  --decode-pos N    position of the decode row (default 512)
				  --batch A,B       widths to measure (default 1,512)
				  --reps N          repetitions per lane (default 3)
				  --warmup-ms N     warm-up per lane before calibration (default 3000)
				  --target-ms N     measurement window per repetition (default 800)
				  --out PATH        write the report to PATH as well as stdout
				  --host LABEL      host label for the report (default: the CUDA device name)
				""");
	}
}
