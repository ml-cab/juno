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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Decode-batch CPU matmuls timed side by side at small batch widths, per weight shape:
 * <ul>
 * <li>{@code gemm}: {@link CpuIntegerGemm}, the cache-blocked prefill matmul;</li>
 * <li>{@code decode-batch}: {@link CpuIntegerDecodeBatch}, each row bit for bit the single-row kernel's;</li>
 * <li>{@code single-calls}: one {@link CpuIntegerKernels} call per activation row.</li>
 * </ul>
 * Weight copies are cycled so the bytes touched per round exceed the last-level cache several times over, as in a
 * real decode step. All three run on {@link CpuThreads} at the selected integer width.
 *
 * <p>Run via {@code scripts/performance-tests/cpu-decode-batch-kernels.sh}.
 */
public final class CpuDecodeBatchMicrobench {

	private static final long CACHE_DEFEAT_BYTES = 96L << 20;

	private CpuDecodeBatchMicrobench() {
	}

	interface Kernel {
		void run(GgufReader.QuantizedTensor w, float[][] X, float[][] Y, int rows, int cols);
	}

	record Variant(String name, Kernel kernel) {
	}

	record Shape(int type, int rows, int cols) {
		String label() {
			return (type == 12 ? "Q4_K" : type == 13 ? "Q5_K" : type == 14 ? "Q6_K" : "Q8_0") + " " + rows + "x" + cols;
		}
	}

	static final List<Variant> VARIANTS = List.of(
			new Variant("gemm", (w, X, Y, rows, cols) -> CpuIntegerGemm.sgemmInto(w, X, Y, 0, rows, cols)),
			new Variant("decode-batch", (w, X, Y, rows, cols) -> CpuIntegerDecodeBatch.matmulInto(w, X, Y, 0, rows, cols)),
			new Variant("single-calls", (w, X, Y, rows, cols) -> {
				for (int b = 0; b < X.length; b++)
					CpuIntegerKernels.matVecInto(w, X[b], 0, rows, Y[b], cols);
			}));

	public static void main(String[] args) throws IOException {
		List<Shape> shapes = new ArrayList<>(List.of(new Shape(12, 2048, 2048), new Shape(12, 5632, 2048),
				new Shape(14, 2048, 5632), new Shape(12, 11008, 2048), new Shape(14, 2048, 11008)));
		int[] batches = { 1, 2, 4, 8 };
		int reps = 5;
		long targetMs = 600;
		long warmupMs = 2000;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--target-ms" -> targetMs = Long.parseLong(args[++i]);
			case "--warmup-ms" -> warmupMs = Long.parseLong(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			case "--batches" -> batches = Arrays.stream(args[++i].split(",")).mapToInt(Integer::parseInt).toArray();
			case "--shapes" -> {
				shapes.clear();
				for (String s : args[++i].split(",")) {
					String[] p = s.split(":");
					shapes.add(new Shape(Integer.parseInt(p[0]), Integer.parseInt(p[1]), Integer.parseInt(p[2])));
				}
			}
			default -> throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		int threads = CpuThreads.threads();
		StringBuilder report = new StringBuilder();
		report.append(String.format(Locale.ROOT,
				"# CPU decode-batch matmuls (threads=%d, integer width %d bits, reps=%d, target %d ms per rep)%n%n",
				threads, CpuIntegerWidth.BITS, reps, targetMs));
		report.append("| Shape | Batch | Variant | Median ms/call | Min to max | Time vs gemm | Rows bit-identical to single |\n");
		report.append("|---|---|---|---|---|---|---|\n");

		for (Shape shape : shapes) {
			Random rnd = new Random(shape.type() * 31L + shape.rows());
			GgufReader.QuantizedTensor first = CpuIntegerKernelsTestSupport.synthetic(shape.type(), shape.rows(),
					shape.cols(), rnd);
			long bytes = first.data().length;
			int copies = (int) Math.max(2, (CACHE_DEFEAT_BYTES + bytes - 1) / bytes);
			GgufReader.QuantizedTensor[] weights = new GgufReader.QuantizedTensor[copies];
			weights[0] = first;
			for (int c = 1; c < copies; c++)
				weights[c] = CpuIntegerKernelsTestSupport.synthetic(shape.type(), shape.rows(), shape.cols(), rnd);
			for (int batch : batches) {
				float[][] X = new float[batch][shape.cols()];
				for (float[] x : X)
					for (int i = 0; i < x.length; i++)
						x[i] = (float) rnd.nextGaussian();
				float[][][] outs = new float[VARIANTS.size()][batch][shape.rows()];
				for (int v = 0; v < VARIANTS.size(); v++)
					VARIANTS.get(v).kernel().run(weights[0], X, outs[v], shape.rows(), shape.cols());
				for (Variant v : VARIANTS)
					timeFor(v, weights, X, shape, warmupMs);
				int calls = Math.max(copies, callsFor(VARIANTS.get(0), weights, X, shape, targetMs));
				double[][] ms = new double[VARIANTS.size()][reps];
				for (int r = 0; r < reps; r++)
					for (int v = 0; v < VARIANTS.size(); v++)
						ms[v][r] = timeN(VARIANTS.get(v), weights, X, shape, calls);
				double base = median(ms[0]);
				for (int v = 0; v < VARIANTS.size(); v++) {
					double med = median(ms[v]);
					double[] s = ms[v].clone();
					Arrays.sort(s);
					report.append(String.format(Locale.ROOT, "| %s | %d | %s | %.3f | %.3f to %.3f | %.3f | %s |%n",
							shape.label(), batch, VARIANTS.get(v).name(), med, s[0], s[s.length - 1], med / base,
							bitIdentical(outs[v], outs[2]) ? "yes" : "no"));
					System.out.printf(Locale.ROOT,
							"{\"threads\":%d,\"bits\":%d,\"shape\":\"%s\",\"batch\":%d,\"variant\":\"%s\",\"calls_per_rep\":%d,\"ms_per_call\":%s,\"median_ms\":%.4f}%n",
							threads, CpuIntegerWidth.BITS, shape.label(), batch, VARIANTS.get(v).name(), calls,
							Arrays.toString(ms[v]), med);
				}
			}
		}
		System.out.print(report);
		if (out != null)
			Files.writeString(out, report.toString());
	}

	private static boolean bitIdentical(float[][] a, float[][] b) {
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[i].length; j++)
				if (Float.floatToRawIntBits(a[i][j]) != Float.floatToRawIntBits(b[i][j]))
					return false;
		return true;
	}

	private static void timeFor(Variant v, GgufReader.QuantizedTensor[] w, float[][] X, Shape s, long ms) {
		long end = System.nanoTime() + ms * 1_000_000L;
		float[][] Y = new float[X.length][s.rows()];
		for (int c = 0; System.nanoTime() < end; c++)
			v.kernel().run(w[c % w.length], X, Y, s.rows(), s.cols());
	}

	private static int callsFor(Variant v, GgufReader.QuantizedTensor[] w, float[][] X, Shape s, long targetMs) {
		double perCall = timeN(v, w, X, s, w.length);
		return (int) Math.max(1, targetMs / Math.max(1e-3, perCall));
	}

	/** Milliseconds per call over {@code n} calls, cycling the weight copies. */
	private static double timeN(Variant v, GgufReader.QuantizedTensor[] w, float[][] X, Shape s, int n) {
		float[][] Y = new float[X.length][s.rows()];
		long t0 = System.nanoTime();
		for (int c = 0; c < n; c++)
			v.kernel().run(w[c % w.length], X, Y, s.rows(), s.cols());
		return (System.nanoTime() - t0) / 1e6 / n;
	}

	private static double median(double[] a) {
		double[] s = a.clone();
		Arrays.sort(s);
		return s.length % 2 == 1 ? s[s.length / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
	}
}
