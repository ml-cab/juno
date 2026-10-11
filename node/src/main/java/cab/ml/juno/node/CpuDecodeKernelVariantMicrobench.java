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
 * Decode-width (one activation row) CPU kernel variants, timed side by side, to
 * measure how much of a quantized matVec's time is the loop-carried dependency
 * of accumulating a whole row into one float.
 *
 * <p>Variants, per format (Q4_K, Q6_K):
 * <ul>
 * <li>{@code production}: {@link LlamaTransformerHandler#matVec} as shipped, a
 * check that the copy below is faithful;</li>
 * <li>{@code a-single}: a copy of the shipped kernel, one {@code float}
 * accumulator per row;</li>
 * <li>{@code b-split4} and {@code b-split8}: the same dequantize-and-multiply
 * arithmetic with 4 or 8 independent accumulators, scalar, no Vector API;</li>
 * <li>{@code c-int-scalar} and {@code d-int-vector}: integer dots over a
 * once-quantized activation, scalar or Vector API
 * ({@link CpuDecodeIntegerVariants}). Their difference against
 * {@code a-single} includes the activation quantization error.</li>
 * </ul>
 * All variants dispatch rows with {@code IntStream.parallel()} on the common
 * pool, as the shipped kernel does, so the thread count is the pool's
 * parallelism plus the caller. Weight copies are cycled so the bytes touched per
 * round exceed the last-level cache several times over, as in a real decode,
 * where every matrix is evicted before it is read again.
 *
 * <p>Run via {@code scripts/performance-tests/cpu-decode-kernel-variants.sh}.
 */
public final class CpuDecodeKernelVariantMicrobench {

	private static final long CACHE_DEFEAT_BYTES = 96L << 20;

	private CpuDecodeKernelVariantMicrobench() {
	}

	interface Kernel {
		void run(byte[] raw, float[] x, float[] y, int rows, int cols);
	}

	record Variant(String name, Kernel kernel) {
	}

	record Shape(int type, int rows, int cols) {
		String label() {
			return (type == 12 ? "Q4_K" : "Q6_K") + " " + rows + "x" + cols;
		}

		int blockBytes() {
			return type == 12 ? 144 : 210;
		}

		long bytes() {
			return (long) rows * (cols / 256) * blockBytes();
		}
	}

	public static void main(String[] args) throws IOException {
		List<Shape> shapes = new ArrayList<>(List.of(new Shape(12, 14336, 4096), new Shape(14, 4096, 14336)));
		int reps = 5;
		long targetMs = 1000;
		long warmupMs = 3000;
		Path out = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--target-ms" -> targetMs = Long.parseLong(args[++i]);
			case "--warmup-ms" -> warmupMs = Long.parseLong(args[++i]);
			case "--out" -> out = Path.of(args[++i]);
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
		// The shipped kernels run on CpuThreads; the scripts set the common pool to
		// the same count for the frozen variants, which still use parallel streams.
		int threads = CpuThreads.threads();
		StringBuilder report = new StringBuilder();
		report.append(String.format(Locale.ROOT,
				"# CPU decode kernel variants (threads=%d, reps=%d, target %d ms per rep)%n%n", threads, reps, targetMs));
		report.append("| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |\n");
		report.append("|---|---|---|---|---|---|---|\n");

		for (Shape shape : shapes) {
			List<Variant> variants = variants(shape.type());
			int copies = (int) Math.max(2, (CACHE_DEFEAT_BYTES + shape.bytes() - 1) / shape.bytes());
			byte[][] weights = new byte[copies][];
			for (int c = 0; c < copies; c++)
				weights[c] = synthetic(shape, c);
			float[] x = new float[shape.cols()];
			Random rnd = new Random(3);
			for (int i = 0; i < x.length; i++)
				x[i] = (float) rnd.nextGaussian();

			float[][] outs = new float[variants.size()][shape.rows()];
			for (int v = 0; v < variants.size(); v++)
				variants.get(v).kernel().run(weights[0], x, outs[v], shape.rows(), shape.cols());

			for (Variant v : variants)
				timeCalls(v, weights, x, shape, warmupMs);
			int calls = Math.max(copies, callsFor(variants.get(1), weights, x, shape, targetMs));

			double[][] ms = new double[variants.size()][reps];
			for (int r = 0; r < reps; r++)
				for (int v = 0; v < variants.size(); v++)
					ms[v][r] = timeN(variants.get(v), weights, x, shape, calls);

			double base = median(ms[1]);
			for (int v = 0; v < variants.size(); v++) {
				double med = median(ms[v]);
				double[] s = ms[v].clone();
				Arrays.sort(s);
				String line = String.format(Locale.ROOT, "| %s | %s | %.3f | %.3f to %.3f | %.2f | %.3f | %.2e |",
						shape.label(), variants.get(v).name(), med, s[0], s[s.length - 1],
						shape.bytes() / (med * 1e6), med / base, maxRelDiff(outs[1], outs[v]));
				report.append(line).append('\n');
				System.out.printf(Locale.ROOT,
						"{\"threads\":%d,\"shape\":\"%s\",\"variant\":\"%s\",\"calls_per_rep\":%d,\"ms_per_call\":%s,\"median_ms\":%.4f}%n",
						threads, shape.label(), variants.get(v).name(), calls, Arrays.toString(ms[v]), med);
			}
		}
		System.out.print(report);
		if (out != null)
			Files.writeString(out, report.toString());
	}

	private static List<Variant> variants(int type) {
		if (type == 12)
			return List.of(
					new Variant("production", (raw, x, y, rows, cols) -> System.arraycopy(
							LlamaTransformerHandler.matVec(new GgufReader.QuantizedTensor("w", 12, (long) rows * cols, raw),
									x, rows, cols), 0, y, 0, rows)),
					new Variant("a-single", CpuDecodeKernelVariantMicrobench::q4kSingle),
					new Variant("b-split4", CpuDecodeKernelVariantMicrobench::q4kSplit4),
					new Variant("b-split8", CpuDecodeKernelVariantMicrobench::q4kSplit8),
					new Variant("c-int-scalar", CpuDecodeIntegerVariants::q4kScalar),
					new Variant("d-int-vector", CpuDecodeIntegerVariants::q4kVector),
					new Variant("d2-int-vector-lanes", CpuDecodeIntegerVariants::q4kVector2),
					new Variant("d3-int-vector-bytes", CpuDecodeIntegerVariants::q4kVector3));
		return List.of(
				new Variant("production", (raw, x, y, rows, cols) -> System.arraycopy(
						LlamaTransformerHandler.matVec(new GgufReader.QuantizedTensor("w", 14, (long) rows * cols, raw),
								x, rows, cols), 0, y, 0, rows)),
				new Variant("a-single", CpuDecodeKernelVariantMicrobench::q6kSingle),
				new Variant("b-split4", CpuDecodeKernelVariantMicrobench::q6kSplit4),
				new Variant("b-split8", CpuDecodeKernelVariantMicrobench::q6kSplit8),
				new Variant("c-int-scalar", CpuDecodeIntegerVariants::q6kScalar),
				new Variant("d-int-vector", CpuDecodeIntegerVariants::q6kVector),
				new Variant("d2-int-vector-lanes", CpuDecodeIntegerVariants::q6kVector2),
				new Variant("d3-int-vector-bytes", CpuDecodeIntegerVariants::q6kVector3));
	}

	// ── Q4_K ────────────────────────────────────────────────────────────────

	/** Copy of the shipped {@code matVecQ4KrawInto}: one accumulator per row. */
	static void q4kSingle(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float acc = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 256; g += 64) {
					int s0 = g / 32;
					int s1 = s0 + 1;
					float scale0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s0);
					float min0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s0);
					float scale1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s1);
					float min1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s1);
					for (int i = 0; i < 32; i++)
						acc += (scale0 * (raw[qsBase + qi + i] & 0x0F) - min0) * x[xBase + g + i];
					for (int i = 0; i < 32; i++)
						acc += (scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1) * x[xBase + g + 32 + i];
					qi += 32;
				}
				xBase += 256;
			}
			y[r] = acc;
		});
	}

	/** Same arithmetic, four independent accumulators. */
	static void q4kSplit4(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 256; g += 64) {
					int s0 = g / 32;
					int s1 = s0 + 1;
					float scale0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s0);
					float min0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s0);
					float scale1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s1);
					float min1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s1);
					int q = qsBase + qi;
					int xl = xBase + g;
					int xh = xl + 32;
					for (int i = 0; i < 32; i += 4) {
						a0 += (scale0 * (raw[q + i] & 0x0F) - min0) * x[xl + i];
						a1 += (scale0 * (raw[q + i + 1] & 0x0F) - min0) * x[xl + i + 1];
						a2 += (scale0 * (raw[q + i + 2] & 0x0F) - min0) * x[xl + i + 2];
						a3 += (scale0 * (raw[q + i + 3] & 0x0F) - min0) * x[xl + i + 3];
					}
					for (int i = 0; i < 32; i += 4) {
						a0 += (scale1 * ((raw[q + i] >> 4) & 0x0F) - min1) * x[xh + i];
						a1 += (scale1 * ((raw[q + i + 1] >> 4) & 0x0F) - min1) * x[xh + i + 1];
						a2 += (scale1 * ((raw[q + i + 2] >> 4) & 0x0F) - min1) * x[xh + i + 2];
						a3 += (scale1 * ((raw[q + i + 3] >> 4) & 0x0F) - min1) * x[xh + i + 3];
					}
					qi += 32;
				}
				xBase += 256;
			}
			y[r] = (a0 + a1) + (a2 + a3);
		});
	}

	/** Same arithmetic, eight independent accumulators. */
	static void q4kSplit8(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 144;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 144;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 256; g += 64) {
					int s0 = g / 32;
					int s1 = s0 + 1;
					float scale0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s0);
					float min0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s0);
					float scale1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s1);
					float min1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s1);
					int q = qsBase + qi;
					int xl = xBase + g;
					int xh = xl + 32;
					for (int i = 0; i < 32; i += 8) {
						a0 += (scale0 * (raw[q + i] & 0x0F) - min0) * x[xl + i];
						a1 += (scale0 * (raw[q + i + 1] & 0x0F) - min0) * x[xl + i + 1];
						a2 += (scale0 * (raw[q + i + 2] & 0x0F) - min0) * x[xl + i + 2];
						a3 += (scale0 * (raw[q + i + 3] & 0x0F) - min0) * x[xl + i + 3];
						a4 += (scale0 * (raw[q + i + 4] & 0x0F) - min0) * x[xl + i + 4];
						a5 += (scale0 * (raw[q + i + 5] & 0x0F) - min0) * x[xl + i + 5];
						a6 += (scale0 * (raw[q + i + 6] & 0x0F) - min0) * x[xl + i + 6];
						a7 += (scale0 * (raw[q + i + 7] & 0x0F) - min0) * x[xl + i + 7];
					}
					for (int i = 0; i < 32; i += 8) {
						a0 += (scale1 * ((raw[q + i] >> 4) & 0x0F) - min1) * x[xh + i];
						a1 += (scale1 * ((raw[q + i + 1] >> 4) & 0x0F) - min1) * x[xh + i + 1];
						a2 += (scale1 * ((raw[q + i + 2] >> 4) & 0x0F) - min1) * x[xh + i + 2];
						a3 += (scale1 * ((raw[q + i + 3] >> 4) & 0x0F) - min1) * x[xh + i + 3];
						a4 += (scale1 * ((raw[q + i + 4] >> 4) & 0x0F) - min1) * x[xh + i + 4];
						a5 += (scale1 * ((raw[q + i + 5] >> 4) & 0x0F) - min1) * x[xh + i + 5];
						a6 += (scale1 * ((raw[q + i + 6] >> 4) & 0x0F) - min1) * x[xh + i + 6];
						a7 += (scale1 * ((raw[q + i + 7] >> 4) & 0x0F) - min1) * x[xh + i + 7];
					}
					qi += 32;
				}
				xBase += 256;
			}
			y[r] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
		});
	}

	// ── Q6_K ────────────────────────────────────────────────────────────────

	/** Copy of the shipped {@code matVecQ6Kraw}: one accumulator per row. */
	static void q6kSingle(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float acc = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = xBase + half * 128;
					for (int l = 0; l < 32; l++) {
						int is = l / 16;
						int qlL = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL = raw[qhOff + l] & 0xFF;
						int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
						float d1 = d * raw[scOff + is];
						float d2 = d * raw[scOff + is + 2];
						float d3 = d * raw[scOff + is + 4];
						float d4 = d * raw[scOff + is + 6];
						acc += d1 * q1 * x[xOff + l];
						acc += d2 * q2 * x[xOff + l + 32];
						acc += d3 * q3 * x[xOff + l + 64];
						acc += d4 * q4 * x[xOff + l + 96];
					}
				}
				xBase += 256;
			}
			y[r] = acc;
		});
	}

	/** Same arithmetic, one accumulator per sub-row (four). */
	static void q6kSplit4(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = xBase + half * 128;
					for (int l = 0; l < 32; l++) {
						int is = l / 16;
						int qlL = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL = raw[qhOff + l] & 0xFF;
						int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
						a1 += d * raw[scOff + is] * q1 * x[xOff + l];
						a2 += d * raw[scOff + is + 2] * q2 * x[xOff + l + 32];
						a3 += d * raw[scOff + is + 4] * q3 * x[xOff + l + 64];
						a4 += d * raw[scOff + is + 6] * q4 * x[xOff + l + 96];
					}
				}
				xBase += 256;
			}
			y[r] = (a1 + a2) + (a3 + a4);
		});
	}

	/** Same arithmetic, eight accumulators (sub-row by even/odd position). */
	static void q6kSplit8(byte[] raw, float[] x, float[] y, int rows, int cols) {
		final int blocksPerRow = cols / 256;
		final int bytesPerRow = blocksPerRow * 210;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, b1 = 0f, b2 = 0f, b3 = 0f, b4 = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * 210;
				float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = xBase + half * 128;
					for (int l = 0; l < 32; l += 2) {
						int is = l / 16;
						float d1 = d * raw[scOff + is];
						float d2 = d * raw[scOff + is + 2];
						float d3 = d * raw[scOff + is + 4];
						float d4 = d * raw[scOff + is + 6];
						int qlL = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL = raw[qhOff + l] & 0xFF;
						a1 += d1 * (((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32) * x[xOff + l];
						a2 += d2 * (((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32) * x[xOff + l + 32];
						a3 += d3 * (((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32) * x[xOff + l + 64];
						a4 += d4 * (((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32) * x[xOff + l + 96];
						int m = l + 1;
						qlL = raw[qlOff + m] & 0xFF;
						qlL2 = raw[qlOff + m + 32] & 0xFF;
						qhL = raw[qhOff + m] & 0xFF;
						b1 += d1 * (((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32) * x[xOff + m];
						b2 += d2 * (((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32) * x[xOff + m + 32];
						b3 += d3 * (((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32) * x[xOff + m + 64];
						b4 += d4 * (((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32) * x[xOff + m + 96];
					}
				}
				xBase += 256;
			}
			y[r] = ((a1 + a2) + (a3 + a4)) + ((b1 + b2) + (b3 + b4));
		});
	}

	// ── harness ─────────────────────────────────────────────────────────────

	private static void timeCalls(Variant v, byte[][] w, float[] x, Shape s, long ms) {
		float[] y = new float[s.rows()];
		long end = System.nanoTime() + ms * 1_000_000L;
		int c = 0;
		while (System.nanoTime() < end)
			v.kernel().run(w[c++ % w.length], x, y, s.rows(), s.cols());
	}

	private static int callsFor(Variant v, byte[][] w, float[] x, Shape s, long targetMs) {
		double one = timeN(v, w, x, s, w.length);
		return (int) Math.max(1, Math.round(targetMs / one));
	}

	/** Returns milliseconds per call over {@code n} calls cycling the weight copies. */
	private static double timeN(Variant v, byte[][] w, float[] x, Shape s, int n) {
		float[] y = new float[s.rows()];
		long t0 = System.nanoTime();
		for (int c = 0; c < n; c++)
			v.kernel().run(w[c % w.length], x, y, s.rows(), s.cols());
		return (System.nanoTime() - t0) / 1e6 / n;
	}

	private static double median(double[] v) {
		double[] s = v.clone();
		Arrays.sort(s);
		return s[s.length / 2];
	}

	private static double maxRelDiff(float[] ref, float[] got) {
		double scale = 0;
		for (float f : ref)
			scale = Math.max(scale, Math.abs(f));
		double max = 0;
		for (int i = 0; i < ref.length; i++)
			max = Math.max(max, Math.abs(got[i] - ref[i]));
		return scale == 0 ? max : max / scale;
	}

	/** Random packed blocks with every FP16 scale field set to a small finite value. */
	private static byte[] synthetic(Shape s, int seed) {
		int blocks = s.rows() * (s.cols() / 256);
		int bb = s.blockBytes();
		byte[] raw = new byte[blocks * bb];
		new Random(17 + seed).nextBytes(raw);
		for (int i = 0; i < blocks; i++) {
			int bo = i * bb;
			if (s.type() == 12) {
				raw[bo] = 0x00; raw[bo + 1] = 0x24; // d = 0.015625
				raw[bo + 2] = 0x00; raw[bo + 3] = 0x20; // dmin = 0.0078125
			} else {
				raw[bo + 208] = 0x00; raw[bo + 209] = 0x20;
			}
		}
		return raw;
	}
}
