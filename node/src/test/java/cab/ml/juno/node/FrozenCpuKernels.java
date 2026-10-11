package cab.ml.juno.node;

import java.util.stream.IntStream;

/**
 * Frozen copies of the CPU matmul kernels as they stood before the CPU hot-path
 * rework (integer kernels, hoisted scratch, owned thread pool), taken from
 * commit {@code aabc42cb}: the float weight-stationary Q4_K / Q5_K / Q8_0
 * prefill kernels, the per-row Q6_K kernel the batched path fell back to, and
 * {@link CpuMatVec}'s float weight-stationary sgemm. Dispatch is
 * {@code IntStream.parallel()} on the common pool, as it was then.
 *
 * <p>The baseline {@link VisionWidthCpuKernelGuardTest} times the production
 * kernels against. Do not update these when the production kernels change;
 * that would turn the guard into a comparison of the new kernel with itself.
 */
final class FrozenCpuKernels {

	private FrozenCpuKernels() {
	}

	/** Batched matmul as the moondream text prefill dispatched it per quant type. */
	static void sgemmQuantBatch(int type, byte[] raw, float[][] X, float[][] Y, int rows, int cols) {
		switch (type) {
		case 12 -> sgemmQ4K(raw, X, Y, rows, cols);
		case 13 -> sgemmQ5K(raw, X, Y, rows, cols);
		case 8 -> sgemmQ8_0(raw, X, Y, rows, cols);
		case 14 -> {
			for (int b = 0; b < X.length; b++)
				System.arraycopy(matVecQ6K(raw, X[b], rows, cols), 0, Y[b], 0, rows);
		}
		default -> throw new IllegalArgumentException("no frozen kernel for type " + type);
		}
	}

	/** {@code CpuMatVec.sgemmInto} without its argument checks. */
	static void sgemmF32(float[] A, float[][] X, float[][] Y, int rows, int cols) {
		int B = X.length;
		IntStream.range(0, rows).parallel().forEach(r -> {
			int base = r * cols;
			for (int b = 0; b < B; b++) {
				float acc = 0f;
				float[] xb = X[b];
				for (int c = 0; c < cols; c++)
					acc += A[base + c] * xb[c];
				Y[b][r] = acc;
			}
		});
	}

	private static void sgemmQ4K(byte[] raw, float[][] X, float[][] Y, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 144;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		final int B = X.length;
		for (int b = 0; b < B; b++)
			java.util.Arrays.fill(Y[b], 0, rows, 0f);
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float[] dq = new float[BLOCK_SIZE];
			for (int blk = 0; blk < blocksPerRow; blk++) {
				int bo = rowByteOffset + blk * BLOCK_BYTES;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < BLOCK_SIZE; g += 64) {
					int s0 = g / 32;
					int s1 = s0 + 1;
					float scale0 = d * q4kScale(raw, scBase, s0);
					float min0 = dmin * q4kMin(raw, scBase, s0);
					float scale1 = d * q4kScale(raw, scBase, s1);
					float min1 = dmin * q4kMin(raw, scBase, s1);
					for (int i = 0; i < 32; i++) {
						dq[g + i] = scale0 * (raw[qsBase + qi + i] & 0x0F) - min0;
						dq[g + 32 + i] = scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1;
					}
					qi += 32;
				}
				int xBase = blk * BLOCK_SIZE;
				for (int p = 0; p < B; p++) {
					float acc = 0f;
					float[] xp = X[p];
					for (int i = 0; i < BLOCK_SIZE; i++)
						acc += dq[i] * xp[xBase + i];
					Y[p][r] += acc;
				}
			}
		});
	}

	private static void sgemmQ5K(byte[] raw, float[][] X, float[][] Y, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 176;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		final int B = X.length;
		for (int b = 0; b < B; b++)
			java.util.Arrays.fill(Y[b], 0, rows, 0f);
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float[] dq = new float[BLOCK_SIZE];
			for (int blk = 0; blk < blocksPerRow; blk++) {
				int bo = rowByteOffset + blk * BLOCK_BYTES;
				int scBase = bo + 4;
				int qhBase = bo + 16;
				int qsBase = bo + 48;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = g * 2;
					int s1 = s0 + 1;
					float scale0 = d * q4kScale(raw, scBase, s0);
					float min0 = dmin * q4kMin(raw, scBase, s0);
					float scale1 = d * q4kScale(raw, scBase, s1);
					float min1 = dmin * q4kMin(raw, scBase, s1);
					int hiBit0 = g * 2;
					int hiBit1 = g * 2 + 1;
					int base = g * 64;
					for (int l = 0; l < 32; l++) {
						int lo = raw[qsBase + qi + l] & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit0) & 1;
						dq[base + l] = scale0 * (lo | (hi << 4)) - min0;
					}
					for (int l = 0; l < 32; l++) {
						int lo = (raw[qsBase + qi + l] >>> 4) & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit1) & 1;
						dq[base + 32 + l] = scale1 * (lo | (hi << 4)) - min1;
					}
					qi += 32;
				}
				int xBase = blk * BLOCK_SIZE;
				for (int p = 0; p < B; p++) {
					float acc = 0f;
					float[] xp = X[p];
					for (int i = 0; i < BLOCK_SIZE; i++)
						acc += dq[i] * xp[xBase + i];
					Y[p][r] += acc;
				}
			}
		});
	}

	/** Scalar dequant only: the frozen baseline does not depend on the Vector API probe. */
	private static void sgemmQ8_0(byte[] raw, float[][] X, float[][] Y, int rows, int cols) {
		final int BLOCK_SIZE = 32;
		final int BLOCK_BYTES = 34;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		final int B = X.length;
		for (int b = 0; b < B; b++)
			java.util.Arrays.fill(Y[b], 0, rows, 0f);
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float[] dq = new float[BLOCK_SIZE];
			for (int blk = 0; blk < blocksPerRow; blk++) {
				int bo = rowByteOffset + blk * BLOCK_BYTES;
				float sc = GgufReader.f16ToF32(readLE16(raw, bo));
				for (int i = 0; i < BLOCK_SIZE; i++)
					dq[i] = sc * raw[bo + 2 + i];
				int xBase = blk * BLOCK_SIZE;
				for (int p = 0; p < B; p++) {
					float acc = 0f;
					float[] xp = X[p];
					for (int i = 0; i < BLOCK_SIZE; i++)
						acc += dq[i] * xp[xBase + i];
					Y[p][r] += acc;
				}
			}
		});
	}

	private static float[] matVecQ6K(byte[] raw, float[] x, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 210;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		float[] y = new float[rows];
		IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = r * bytesPerRow;
			float acc = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));
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
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	private static short readLE16(byte[] raw, int off) {
		return (short) ((raw[off] & 0xFF) | ((raw[off + 1] & 0xFF) << 8));
	}

	private static float q4kScale(byte[] raw, int scBase, int j) {
		return (j < 4) ? raw[scBase + j] & 0x3F
				: ((raw[scBase + j + 4] & 0x0F) | ((raw[scBase + j - 4] & 0xC0) >> 2)) & 0x3F;
	}

	private static float q4kMin(byte[] raw, int scBase, int j) {
		return (j < 4) ? raw[scBase + j + 4] & 0x3F
				: (((raw[scBase + j + 4] & 0xFF) >> 4) | ((raw[scBase + j] & 0xC0) >> 2)) & 0x3F;
	}
}
