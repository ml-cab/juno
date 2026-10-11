package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

/**
 * Vision-width guard for the CPU matmul kernels: at the batch width of a
 * moondream2 prefill (729 image tokens plus a short prompt, B = 741), the
 * production CPU path must be no slower than the kernels it replaced
 * ({@link FrozenCpuKernels}). A general-purpose SIMD dot product was once taken
 * off the hot path because it ran tens to hundreds of times slower at this
 * width on a 128-bit-vector host; this test keeps any successor kernel, scratch
 * layout or thread pool from repeating that.
 *
 * <p>Method: each side is warmed, then the two are timed alternately five times
 * (baseline, candidate, baseline, ...) so host drift lands on both, and the
 * candidate's median must be within {@link #MAX_RATIO} of the baseline's. The
 * bound is the 15% run-to-run noise floor measured on the baseline host; the
 * regression this guards against is an order of magnitude, not a percentage.
 * A relative-RMS bound against the baseline's output keeps a fast but wrong
 * kernel from passing; it is loose enough for a quantized-activation kernel.
 */
@DisplayName("CPU kernels at vision batch width (B = 741)")
class VisionWidthCpuKernelGuardTest {

	private static final int B = 741;
	private static final int ROWS = 256;
	/** moondream2's text width (Phi-2 hidden size). */
	private static final int QUANT_COLS = 2048;
	/** moondream2's vision encoder width (SigLIP hidden size). */
	private static final int F32_COLS = 1152;

	private static final double MAX_RATIO = 1.15;
	private static final double MAX_RELATIVE_RMS = 0.02;
	private static final int WARMUP = 3;
	private static final int REPS = 5;

	@ParameterizedTest(name = "GGML type {0}")
	@ValueSource(ints = { 12, 13, 14, 8 }) // Q4_K, Q5_K, Q6_K, Q8_0
	@Timeout(600)
	@DisplayName("quantized prefill dispatch no slower than the frozen kernel")
	void quantizedBatch_noSlowerThanFrozenKernel(int type) {
		byte[] raw = syntheticWeights(type, ROWS, QUANT_COLS);
		GgufReader.QuantizedTensor qt = new GgufReader.QuantizedTensor(
				"guard", type, (long) ROWS * QUANT_COLS, raw);
		float[][] X = randomActivations(B, QUANT_COLS);
		float[][] yBase = new float[B][ROWS];
		float[][] yCand = new float[B][ROWS];

		Runnable baseline = () -> FrozenCpuKernels.sgemmQuantBatch(type, raw, X, yBase, ROWS, QUANT_COLS);
		Runnable candidate = () -> Phi2TransformerHandler.sgemmQuantBatch(qt, X, yCand, 0, ROWS, QUANT_COLS);

		assertNoSlower("type " + type, baseline, candidate);
		assertClose("type " + type, yBase, yCand);
	}

	@Test
	@Timeout(600)
	@DisplayName("CpuMatVec float sgemm (vision encoder) no slower than the frozen kernel")
	void f32Sgemm_noSlowerThanFrozenKernel() {
		Random rnd = new Random(7);
		float[] A = new float[ROWS * F32_COLS];
		for (int i = 0; i < A.length; i++)
			A[i] = (float) rnd.nextGaussian() * 0.05f;
		float[][] X = randomActivations(B, F32_COLS);
		float[][] yBase = new float[B][ROWS];
		float[][] yCand = new float[B][ROWS];

		Runnable baseline = () -> FrozenCpuKernels.sgemmF32(A, X, yBase, ROWS, F32_COLS);
		Runnable candidate = () -> CpuMatVec.INSTANCE.sgemmInto(A, X, yCand, ROWS, F32_COLS);

		assertNoSlower("F32", baseline, candidate);
		assertClose("F32", yBase, yCand);
	}

	private static void assertNoSlower(String label, Runnable baseline, Runnable candidate) {
		for (int i = 0; i < WARMUP; i++) {
			baseline.run();
			candidate.run();
		}
		long[] base = new long[REPS];
		long[] cand = new long[REPS];
		for (int i = 0; i < REPS; i++) {
			base[i] = time(baseline);
			cand[i] = time(candidate);
		}
		double baseMs = median(base) / 1e6;
		double candMs = median(cand) / 1e6;
		double ratio = candMs / baseMs;
		System.out.printf("vision-width guard %s: baseline median %.1f ms %s, candidate median %.1f ms %s, ratio %.3f%n",
				label, baseMs, Arrays.toString(toMs(base)), candMs, Arrays.toString(toMs(cand)), ratio);
		assertThat(ratio)
				.as("%s at B=%d: candidate %.1f ms vs frozen baseline %.1f ms", label, B, candMs, baseMs)
				.isLessThanOrEqualTo(MAX_RATIO);
	}

	private static void assertClose(String label, float[][] expected, float[][] actual) {
		double err = 0, ref = 0;
		for (int b = 0; b < expected.length; b++) {
			for (int r = 0; r < expected[b].length; r++) {
				double e = expected[b][r];
				double d = actual[b][r] - e;
				err += d * d;
				ref += e * e;
			}
		}
		assertThat(ref).as("%s baseline output is non-trivial", label).isGreaterThan(0);
		assertThat(Math.sqrt(err / ref)).as("%s relative RMS error against the frozen kernel", label)
				.isLessThanOrEqualTo(MAX_RELATIVE_RMS);
	}

	private static long time(Runnable r) {
		long t0 = System.nanoTime();
		r.run();
		return System.nanoTime() - t0;
	}

	private static double median(long[] v) {
		long[] s = v.clone();
		Arrays.sort(s);
		return s[s.length / 2];
	}

	private static long[] toMs(long[] ns) {
		long[] ms = new long[ns.length];
		for (int i = 0; i < ns.length; i++)
			ms[i] = ns[i] / 1_000_000L;
		return ms;
	}

	private static float[][] randomActivations(int b, int cols) {
		Random rnd = new Random(11);
		float[][] X = new float[b][cols];
		for (float[] x : X)
			for (int i = 0; i < cols; i++)
				x[i] = (float) rnd.nextGaussian();
		return X;
	}

	/** Random packed blocks with every block's FP16 scale fields set to small finite values. */
	private static byte[] syntheticWeights(int type, int rows, int cols) {
		int blockSize = type == 8 ? 32 : 256;
		int blockBytes = switch (type) {
		case 12 -> 144;
		case 13 -> 176;
		case 14 -> 210;
		case 8 -> 34;
		default -> throw new IllegalArgumentException("type " + type);
		};
		int blocks = rows * (cols / blockSize);
		byte[] raw = new byte[blocks * blockBytes];
		new Random(1).nextBytes(raw);
		for (int i = 0; i < blocks; i++) {
			int bo = i * blockBytes;
			switch (type) {
			case 12, 13 -> { // d = 0.25, dmin = 0.0625
				raw[bo] = 0x00; raw[bo + 1] = 0x34;
				raw[bo + 2] = 0x00; raw[bo + 3] = 0x2C;
			}
			case 14 -> { raw[bo + 208] = 0x00; raw[bo + 209] = 0x2C; } // d = 0.0625
			case 8 -> { raw[bo] = 0x00; raw[bo + 1] = 0x24; } // d = 0.015625
			default -> { }
			}
		}
		return raw;
	}
}
