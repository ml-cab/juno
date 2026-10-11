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
import java.util.stream.IntStream;

/**
 * Prefill-width (many activation rows) CPU matmul variants, timed side by side.
 *
 * <p>Variants, per format:
 * <ul>
 * <li>{@code a}: the shipped float path: {@code sgemmQ4KWeightStationary} or
 * {@code sgemmQ5KWeightStationary}, or for Q6_K (which has no batched float
 * kernel) one float-activation matVec per activation row;</li>
 * <li>{@code b-float-split4} (Q4_K): the weight-stationary float kernel with
 * the dequantized block hoisted per row and four independent accumulators;</li>
 * <li>{@code c-int-scalar} (Q4_K): integer dots over a once-quantized
 * activation, per row, scalar loops C2 may auto-vectorize;</li>
 * <li>{@code d-int-gemm}: {@link CpuIntegerGemm}, the cache-blocked Vector API
 * integer matmul.</li>
 * </ul>
 * Run via {@code scripts/performance-tests/cpu-prefill-kernel-variants.sh}.
 */
public final class CpuPrefillKernelVariantMicrobench {

	private CpuPrefillKernelVariantMicrobench() {
	}

	interface Kernel {
		void run(GgufReader.QuantizedTensor w, float[][] X, float[][] Y, int rows, int cols);
	}

	record Variant(String name, Kernel kernel) {
	}

	public static void main(String[] args) throws IOException {
		int[] batches = { 128, 741 };
		int[] types = { 12, 13, 14 };
		int rows = 4096;
		int cols = 4096;
		int reps = 3;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--batch" -> batches = Arrays.stream(args[++i].split(",")).mapToInt(Integer::parseInt).toArray();
			case "--types" -> types = Arrays.stream(args[++i].split(",")).mapToInt(Integer::parseInt).toArray();
			case "--rows" -> rows = Integer.parseInt(args[++i]);
			case "--cols" -> cols = Integer.parseInt(args[++i]);
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
			default -> throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		int threads = CpuThreads.threads();
		StringBuilder report = new StringBuilder(String.format(Locale.ROOT,
				"# CPU prefill kernel variants (threads=%d, reps=%d, %dx%d)%n%n", threads, reps, rows, cols));
		report.append("| Type | B | Variant | Median ms/call | Min to max | GMAC/s | Time vs a | Max rel. diff vs a |\n");
		report.append("|---|---|---|---|---|---|---|---|\n");
		for (int type : types) {
			GgufReader.QuantizedTensor w = synthetic(type, rows, cols, new Random(11 + type));
			for (int batch : batches) {
				Random rnd = new Random(batch);
				float[][] X = new float[batch][cols];
				for (float[] x : X)
					for (int i = 0; i < cols; i++)
						x[i] = (float) rnd.nextGaussian();
				List<Variant> vs = variants(type);
				float[][][] outs = new float[vs.size()][][];
				for (int v = 0; v < vs.size(); v++) {
					outs[v] = new float[batch][rows];
					vs.get(v).kernel().run(w, X, outs[v], rows, cols); // warm-up 1
					vs.get(v).kernel().run(w, X, outs[v], rows, cols); // warm-up 2
				}
				double[][] ms = new double[vs.size()][reps];
				for (int r = 0; r < reps; r++)
					for (int v = 0; v < vs.size(); v++) {
						float[][] Y = new float[batch][rows];
						long t0 = System.nanoTime();
						vs.get(v).kernel().run(w, X, Y, rows, cols);
						ms[v][r] = (System.nanoTime() - t0) / 1e6;
					}
				double base = median(ms[0]);
				double macs = (double) rows * cols * batch;
				for (int v = 0; v < vs.size(); v++) {
					double med = median(ms[v]);
					double[] s = ms[v].clone();
					Arrays.sort(s);
					report.append(String.format(Locale.ROOT, "| %s | %d | %s | %.1f | %.1f to %.1f | %.2f | %.3f | %.2e |%n",
							name(type), batch, vs.get(v).name(), med, s[0], s[s.length - 1], macs / (med * 1e6), med / base,
							maxRelDiff(outs[0], outs[v])));
					System.out.printf(Locale.ROOT,
							"{\"threads\":%d,\"type\":\"%s\",\"batch\":%d,\"variant\":\"%s\",\"ms\":%s,\"median_ms\":%.2f}%n",
							threads, name(type), batch, vs.get(v).name(), Arrays.toString(ms[v]), med);
				}
			}
		}
		System.out.print(report);
		if (out != null)
			Files.writeString(out, report.toString());
	}

	private static String name(int type) {
		return switch (type) {
		case 12 -> "Q4_K";
		case 13 -> "Q5_K";
		case 14 -> "Q6_K";
		case 8 -> "Q8_0";
		default -> "type" + type;
		};
	}

	private static List<Variant> variants(int type) {
		List<Variant> v = new ArrayList<>();
		switch (type) {
		case 12 -> {
			v.add(new Variant("a-float-ws", (w, X, Y, rows, cols) -> LlamaTransformerHandler
					.sgemmQ4KWeightStationary(w.data(), X, Y, 0, rows, cols)));
			v.add(new Variant("b-float-split4", CpuPrefillKernelVariantMicrobench::q4kFloatSplit4));
			v.add(new Variant("c-int-scalar", CpuPrefillKernelVariantMicrobench::q4kIntScalar));
		}
		case 13 -> v.add(new Variant("a-float-ws", (w, X, Y, rows, cols) -> LlamaTransformerHandler
				.sgemmQ5KWeightStationary(w.data(), X, Y, 0, rows, cols)));
		case 14 -> v.add(new Variant("a-float-per-row", (w, X, Y, rows, cols) -> {
			for (int b = 0; b < X.length; b++)
				System.arraycopy(LlamaTransformerHandler.matVecFloatActivation(w, X[b], 0, rows, cols), 0, Y[b], 0, rows);
		}));
		default -> throw new IllegalArgumentException("type " + type);
		}
		v.add(new Variant("d-int-gemm", (w, X, Y, rows, cols) -> CpuIntegerGemm.sgemmInto(w, X, Y, 0, rows, cols)));
		return v;
	}

	/** Weight-stationary Q4_K with the block dequantized once per row-block and four accumulators per dot. */
	static void q4kFloatSplit4(GgufReader.QuantizedTensor w, float[][] X, float[][] Y, int rows, int cols) {
		byte[] raw = w.data();
		int blocksPerRow = cols / 256;
		int bytesPerRow = blocksPerRow * 144;
		int batch = X.length;
		IntStream.range(0, rows).parallel().forEach(r -> {
			float[] dq = new float[256];
			float[] acc = new float[batch];
			for (int blk = 0; blk < blocksPerRow; blk++) {
				int bo = r * bytesPerRow + blk * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					float s0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					float m0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					float s1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					float m1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
					int q = bo + 16 + g * 32;
					for (int i = 0; i < 32; i++) {
						dq[g * 64 + i] = s0 * (raw[q + i] & 0x0F) - m0;
						dq[g * 64 + 32 + i] = s1 * ((raw[q + i] >> 4) & 0x0F) - m1;
					}
				}
				int xb = blk * 256;
				for (int p = 0; p < batch; p++) {
					float[] x = X[p];
					float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;
					for (int i = 0; i < 256; i += 4) {
						a0 += dq[i] * x[xb + i];
						a1 += dq[i + 1] * x[xb + i + 1];
						a2 += dq[i + 2] * x[xb + i + 2];
						a3 += dq[i + 3] * x[xb + i + 3];
					}
					acc[p] += (a0 + a1) + (a2 + a3);
				}
			}
			for (int p = 0; p < batch; p++)
				Y[p][r] = acc[p];
		});
	}

	/** Q4_K integer dots per row over a once-quantized activation; scalar loops. */
	static void q4kIntScalar(GgufReader.QuantizedTensor w, float[][] X, float[][] Y, int rows, int cols) {
		byte[] raw = w.data();
		int batch = X.length;
		CpuIntegerKernels.Q8Activation[] a = new CpuIntegerKernels.Q8Activation[batch];
		IntStream.range(0, batch).parallel().forEach(b -> a[b] = CpuIntegerKernels.Q8Activation.quantize(X[b], cols));
		int blocksPerRow = cols / 256;
		int bytesPerRow = blocksPerRow * 144;
		IntStream.range(0, rows).parallel().forEach(r -> {
			byte[] wq = new byte[256];
			float[] sc = new float[8];
			float[] mn = new float[8];
			float[] acc = new float[batch];
			for (int blk = 0; blk < blocksPerRow; blk++) {
				int bo = r * bytesPerRow + blk * 144;
				int scBase = bo + 4;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				for (int g = 0; g < 4; g++) {
					int q = bo + 16 + g * 32;
					for (int i = 0; i < 32; i++) {
						wq[g * 64 + i] = (byte) (raw[q + i] & 0x0F);
						wq[g * 64 + 32 + i] = (byte) ((raw[q + i] >> 4) & 0x0F);
					}
					sc[2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
					sc[2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
					mn[2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
					mn[2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
				}
				int x0 = blk * 256;
				for (int p = 0; p < batch; p++) {
					byte[] xq = a[p].q;
					float[] xd = a[p].d;
					int[] xs = a[p].sum32;
					float s = 0f;
					for (int j = 0; j < 8; j++) {
						int e = j * 32;
						int dot = 0;
						for (int i = 0; i < 32; i++)
							dot += wq[e + i] * xq[x0 + e + i];
						int ab = (x0 + e) >> 5;
						s += xd[ab] * (sc[j] * dot - mn[j] * xs[ab]);
					}
					acc[p] += s;
				}
			}
			for (int p = 0; p < batch; p++)
				Y[p][r] = acc[p];
		});
	}

	private static double median(double[] v) {
		double[] s = v.clone();
		Arrays.sort(s);
		return s[s.length / 2];
	}

	private static double maxRelDiff(float[][] ref, float[][] got) {
		double scale = 0;
		double max = 0;
		for (int b = 0; b < ref.length; b++)
			for (int i = 0; i < ref[b].length; i++) {
				scale = Math.max(scale, Math.abs(ref[b][i]));
				max = Math.max(max, Math.abs(got[b][i] - ref[b][i]));
			}
		return scale == 0 ? max : max / scale;
	}

	private static GgufReader.QuantizedTensor synthetic(int type, int rows, int cols, Random rnd) {
		int blockBytes = switch (type) {
		case 12 -> 144;
		case 13 -> 176;
		case 14 -> 210;
		default -> throw new IllegalArgumentException("type " + type);
		};
		int blocks = rows * (cols / 256);
		byte[] raw = new byte[blocks * blockBytes];
		rnd.nextBytes(raw);
		for (int i = 0; i < blocks; i++) {
			int bo = i * blockBytes;
			if (type == 14) {
				raw[bo + 208] = 0x00;
				raw[bo + 209] = 0x1C; // d = 0.00390625
			} else {
				raw[bo] = 0x00;
				raw[bo + 1] = 0x24; // d = 0.015625
				raw[bo + 2] = 0x00;
				raw[bo + 3] = 0x20; // dmin = 0.0078125
			}
		}
		return new GgufReader.QuantizedTensor("w", type, (long) rows * cols, raw);
	}
}
