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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Common-pool parallel streams against the persistent worker pool
 * ({@link CpuThreads.Dispatch}), on the integer CPU kernels, alternating the two
 * on identical work.
 *
 * <p>Cases: single decode-width matmuls at TinyLlama and Mistral 7B shapes; one
 * simulated decode layer (its seven projections back to back, as a decode step
 * issues them); and the prefill matmul at B=741 (the vision prefill width).
 * Run via {@code scripts/performance-tests/cpu-dispatch-microbench.sh}.
 */
public final class CpuDispatchMicrobench {

	private CpuDispatchMicrobench() {
	}

	record Mat(String label, int type, int rows, int cols) {
	}

	interface Work {
		void run();
	}

	public static void main(String[] args) throws IOException {
		int reps = 5;
		long targetMs = 600;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--target-ms" -> targetMs = Long.parseLong(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			default -> throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		StringBuilder report = new StringBuilder(String.format(Locale.ROOT,
				"# CPU dispatch: common pool against worker pool (worker threads=%d, common pool parallelism=%d, reps=%d)%n%n",
				CpuThreads.threads(), java.util.concurrent.ForkJoinPool.getCommonPoolParallelism(), reps));
		report.append("| Case | Common pool ms | Worker pool ms | Worker / common | Common min to max | Worker min to max |\n");
		report.append("|---|---|---|---|---|---|\n");

		List<Mat> tiny = List.of(new Mat("q", 12, 2048, 2048), new Mat("k", 12, 256, 2048), new Mat("v", 14, 256, 2048),
				new Mat("o", 12, 2048, 2048), new Mat("gate", 12, 5632, 2048), new Mat("up", 12, 5632, 2048),
				new Mat("down", 14, 2048, 5632));
		List<Mat> mistral = List.of(new Mat("q", 12, 4096, 4096), new Mat("k", 12, 1024, 4096),
				new Mat("v", 14, 1024, 4096), new Mat("o", 12, 4096, 4096), new Mat("gate", 12, 14336, 4096),
				new Mat("up", 12, 14336, 4096), new Mat("down", 14, 4096, 14336));

		for (Mat m : List.of(tiny.get(1), tiny.get(0), tiny.get(6), mistral.get(1), mistral.get(4), mistral.get(6)))
			run(report, "decode " + m.type() + " " + m.rows() + "x" + m.cols(), decode(List.of(m)), reps, targetMs);
		run(report, "decode layer, TinyLlama shapes (7 matmuls)", decode(tiny), reps, targetMs);
		run(report, "decode layer, Mistral 7B shapes (7 matmuls)", decode(mistral), reps, targetMs);
		run(report, "prefill B=741, Q4_K 2048x2048", prefill(12, 2048, 2048, 741), reps, targetMs);
		run(report, "prefill B=32, Q4_K 4096x4096", prefill(12, 4096, 4096, 32), reps, targetMs);
		System.out.print(report);
		if (out != null)
			Files.writeString(out, report.toString());
	}

	private static Work decode(List<Mat> mats) {
		Random rnd = new Random(5);
		GgufReader.QuantizedTensor[] w = new GgufReader.QuantizedTensor[mats.size()];
		float[][] x = new float[mats.size()][];
		float[][] y = new float[mats.size()][];
		for (int i = 0; i < mats.size(); i++) {
			Mat m = mats.get(i);
			w[i] = CpuIntegerKernelsTestSupport.synthetic(m.type(), m.rows(), m.cols(), rnd);
			x[i] = gaussian(m.cols(), rnd);
			y[i] = new float[m.rows()];
		}
		return () -> {
			for (int i = 0; i < w.length; i++)
				CpuIntegerKernels.matVecInto(w[i], x[i], 0, y[i].length, y[i], x[i].length);
		};
	}

	private static Work prefill(int type, int rows, int cols, int batch) {
		Random rnd = new Random(7);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTestSupport.synthetic(type, rows, cols, rnd);
		float[][] X = new float[batch][];
		for (int b = 0; b < batch; b++)
			X[b] = gaussian(cols, rnd);
		float[][] Y = new float[batch][rows];
		return () -> CpuIntegerGemm.sgemmInto(w, X, Y, 0, rows, cols);
	}

	private static float[] gaussian(int n, Random rnd) {
		float[] x = new float[n];
		for (int i = 0; i < n; i++)
			x[i] = (float) rnd.nextGaussian();
		return x;
	}

	private static void run(StringBuilder report, String label, Work work, int reps, long targetMs) {
		CpuThreads.Dispatch was = CpuThreads.setDispatch(CpuThreads.Dispatch.COMMON_POOL);
		try {
			for (CpuThreads.Dispatch d : CpuThreads.Dispatch.values()) {
				CpuThreads.setDispatch(d);
				long end = System.nanoTime() + 1_500_000_000L;
				while (System.nanoTime() < end)
					work.run();
			}
			CpuThreads.setDispatch(CpuThreads.Dispatch.COMMON_POOL);
			long t0 = System.nanoTime();
			work.run();
			double one = (System.nanoTime() - t0) / 1e6;
			int calls = (int) Math.max(3, Math.round(targetMs / Math.max(one, 0.01)));
			double[][] ms = new double[2][reps];
			for (int r = 0; r < reps; r++)
				for (CpuThreads.Dispatch d : CpuThreads.Dispatch.values()) {
					CpuThreads.setDispatch(d);
					long s = System.nanoTime();
					for (int c = 0; c < calls; c++)
						work.run();
					ms[d.ordinal()][r] = (System.nanoTime() - s) / 1e6 / calls;
				}
			double common = median(ms[0]);
			double worker = median(ms[1]);
			double[] c = ms[0].clone();
			double[] w = ms[1].clone();
			Arrays.sort(c);
			Arrays.sort(w);
			String line = String.format(Locale.ROOT, "| %s | %.3f | %.3f | %.3f | %.3f to %.3f | %.3f to %.3f |", label,
					common, worker, worker / common, c[0], c[reps - 1], w[0], w[reps - 1]);
			report.append(line).append('\n');
			System.out.println(line);
		} finally {
			CpuThreads.setDispatch(was);
		}
	}

	private static double median(double[] v) {
		double[] s = v.clone();
		Arrays.sort(s);
		return s[s.length / 2];
	}
}
