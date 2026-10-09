package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A context shift applied to the device KV mirror in place: the kept prefix stays,
 * rows {@code [keep + discard, seqLen)} move down to {@code keep}, and each moved
 * K row is rotated back by the shift distance on the device. The oracle is the
 * host shift ({@link RopeShift#back}) of the same FP16 rows in float, rounded to
 * FP16: the device rotation uses the same cos/sin table and the same rounding, so
 * K and V must match bit for bit, and the watermark is the new length.
 */
@Tag("gpu")
@DisplayName("DeviceKvCache - context shift in place on the device")
class DeviceKvShiftTest {

	private static GpuContext gpu;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		gpu = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (gpu != null)
			gpu.close();
	}

	private static float fp16(float x) {
		return Float.float16ToFloat(Float.floatToFloat16(x));
	}

	private static void check(RopeShift rope, int kvHeads, int headDim, int seqLen, int keep, int discard) {
		int kvDim = kvHeads * headDim;
		Random rng = new Random(seqLen * 31L + keep * 7L + discard);
		float[] k = new float[seqLen * kvDim];
		float[] v = new float[seqLen * kvDim];
		for (int i = 0; i < k.length; i++) {
			k[i] = fp16((float) rng.nextGaussian());
			v[i] = fp16((float) rng.nextGaussian());
		}
		int newLen = seqLen - discard;
		float[] expectK = new float[newLen * kvDim];
		float[] expectV = new float[newLen * kvDim];
		System.arraycopy(k, 0, expectK, 0, keep * kvDim);
		System.arraycopy(v, 0, expectV, 0, keep * kvDim);
		System.arraycopy(k, (keep + discard) * kvDim, expectK, keep * kvDim, (newLen - keep) * kvDim);
		System.arraycopy(v, (keep + discard) * kvDim, expectV, keep * kvDim, (newLen - keep) * kvDim);
		var rotate = rope.back(discard, kvHeads);
		float[] row = new float[kvDim];
		for (int p = keep; p < newLen; p++) {
			System.arraycopy(expectK, p * kvDim, row, 0, kvDim);
			rotate.rotate(row);
			for (int i = 0; i < kvDim; i++)
				expectK[p * kvDim + i] = fp16(row[i]);
		}

		DeviceKvCache m = new DeviceKvCache(gpu, kvDim);
		try {
			m.replacePrefix(k, v, seqLen);
			assertThat(m.shiftInPlace(keep, discard, seqLen, rope, kvHeads)).as("shifted on the device").isTrue();
			assertThat(m.validTokens()).as("watermark at the new length").isEqualTo(newLen);
			assertThat(m.downloadK(newLen)).as("K: kept prefix, moved and rotated rows").containsExactly(expectK);
			assertThat(m.downloadV(newLen)).as("V: kept prefix and moved rows").containsExactly(expectV);
		} finally {
			m.close();
		}
	}

	@Test
	@DisplayName("adjacent pairs over the whole head (LLaMA): half of what follows the prefix discarded")
	void adjacent() {
		check(RopeShift.standard(64, 10000f, RopePairing.ADJACENT), 4, 64, 300, 8, 146);
	}

	@Test
	@DisplayName("split-half pairs (Qwen2, Phi-3, Qwen3 layout) with per-pair frequencies of any shape")
	void splitHalfAnyFrequencies() {
		double[] freq = new double[64];
		Random rng = new Random(5);
		for (int i = 0; i < freq.length; i++)
			freq[i] = rng.nextDouble() * 0.9 + 1e-4;
		check(RopeShift.of(128, RopePairing.SPLIT_HALF, freq), 2, 128, 517, 31, 243);
	}

	@Test
	@DisplayName("partial rotation (Phi-2): only the first ropeDim dims of each head turn")
	void partial() {
		check(RopeShift.partialSplitHalf(80, 32, 10000f), 4, 80, 200, 4, 98);
	}

	@Test
	@DisplayName("a discard shorter than the moved rows: overlapping ranges move intact")
	void overlappingMove() {
		check(RopeShift.standard(64, 10000f, RopePairing.ADJACENT), 4, 64, 300, 8, 20);
	}
}
