package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Regression net for the moondream vision prefill hang: {@code sgemmQ5KWeightStationary}
 * must stay within a small multiple of sequential {@code matVecInto} at vision-scale
 * batch widths. The Vector-API / nested-ForkJoinPool path was measured ~260× slower
 * than matVec for B≈741 on Ivy Bridge (128-bit SPECIES), so a single Phi-2
 * {@code forwardBatch} never finished.
 */
@DisplayName("Q5_K weight-stationary prefill performance")
class Q5KWeightStationaryBenchTest {

	/** Weight-stationary may lose a little to sequential on tiny B; must not explode. */
	private static final double MAX_SLOWDOWN_VS_SEQUENTIAL = 8.0;

	@Test
	@Timeout(60)
	@DisplayName("sgemmQ5KWeightStationary at B=128 stays within 8× of sequential matVec")
	void weightStationary_visionScaleBatch_notPathologicallySlowerThanMatVec() {
		int rows = 256;
		int cols = 2048;
		int B = 128;
		byte[] raw = syntheticQ5K(rows, cols);

		float[][] X = new float[B][cols];
		float[][] Yws = new float[B][rows];
		float[][] Yseq = new float[B][rows];
		for (float[] x : X)
			java.util.Arrays.fill(x, 0.01f);

		GgufReader.QuantizedTensor qt = new GgufReader.QuantizedTensor(
				"bench", 13, (long) rows * cols, raw);

		// Warmup both paths so JIT is not charged to the timed region.
		LlamaTransformerHandler.sgemmQ5KWeightStationary(raw, X, Yws, 0, rows, cols);
		for (int b = 0; b < B; b++)
			LlamaTransformerHandler.matVecInto(qt, X[b], Yseq[b], 0, rows, cols);

		long t0 = System.nanoTime();
		LlamaTransformerHandler.sgemmQ5KWeightStationary(raw, X, Yws, 0, rows, cols);
		long wsMs = Math.max(1L, (System.nanoTime() - t0) / 1_000_000L);

		t0 = System.nanoTime();
		for (int b = 0; b < B; b++)
			LlamaTransformerHandler.matVecInto(qt, X[b], Yseq[b], 0, rows, cols);
		long seqMs = Math.max(1L, (System.nanoTime() - t0) / 1_000_000L);

		double ratio = (double) wsMs / (double) seqMs;
		assertThat(ratio)
				.as("weightStationary_ms=%d sequentialMatVec_ms=%d ratio=%.2f (SIMD/FJP regression)",
						wsMs, seqMs, ratio)
				.isLessThanOrEqualTo(MAX_SLOWDOWN_VS_SEQUENTIAL);

		// Correctness smoke: same mathematical result within FP tolerance.
		for (int b = 0; b < B; b++) {
			for (int r = 0; r < rows; r++) {
				float tol = Math.max(1e-2f, Math.abs(Yseq[b][r]) * 1e-3f);
				assertThat(Yws[b][r]).as("b=%d r=%d", b, r).isCloseTo(Yseq[b][r],
						org.assertj.core.data.Offset.offset(tol));
			}
		}
	}

	private static byte[] syntheticQ5K(int rows, int cols) {
		int blocks = rows * (cols / 256);
		byte[] raw = new byte[blocks * 176];
		new Random(1).nextBytes(raw);
		for (int i = 0; i < blocks; i++) {
			raw[i * 176] = 0x00;
			raw[i * 176 + 1] = 0x34;
			raw[i * 176 + 2] = 0x00;
			raw[i * 176 + 3] = 0x2C;
		}
		return raw;
	}
}
