package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Host packing / CPU batch oracle tests (no GPU required).
 */
@DisplayName("DeviceActivationBatch + CpuFrozenBatchOps")
class DeviceActivationBatchTest {

	@Test
	@DisplayName("packColumns / unpackColumns round-trip")
	void pack_unpack_round_trip() {
		float[][] cols = {
				{ 1f, 2f, 3f },
				{ 4f, 5f, 6f },
				{ 7f, 8f, 9f }
		};
		float[] packed = DeviceActivationBatch.packColumns(cols, 3, 3);
		assertThat(packed).containsExactly(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f);
		float[][] out = new float[3][];
		DeviceActivationBatch.unpackColumns(packed, out, 3, 3);
		for (int b = 0; b < 3; b++)
			assertThat(out[b]).containsExactly(cols[b]);
	}

	@Test
	@DisplayName("packColumns rejects short column")
	void pack_rejects_bad_length() {
		float[][] cols = { { 1f, 2f }, { 3f } };
		assertThatThrownBy(() -> DeviceActivationBatch.packColumns(cols, 2, 2))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("CpuFrozenBatchOps forward matches sequential matVec")
	void cpu_forward_matches_sequential() {
		Random rng = new Random(7);
		int rows = 5, cols = 4, batch = 8;
		float[] W = random(rng, rows * cols);
		float[][] X = new float[batch][];
		for (int b = 0; b < batch; b++)
			X[b] = random(rng, cols);
		float[][] Y = CpuFrozenBatchOps.forward(W, X, rows, cols);
		for (int b = 0; b < batch; b++) {
			float[] expected = GpuMatVecTransposeContractTest.scalarMatVec(W, X[b], rows, cols);
			assertThat(Y[b]).containsExactly(expected, within(0f));
		}
	}

	@Test
	@DisplayName("CpuFrozenBatchOps transpose matches sequential transpose")
	void cpu_transpose_matches_sequential() {
		Random rng = new Random(11);
		int rows = 6, cols = 3, batch = 4;
		float[] W = random(rng, rows * cols);
		float[][] G = new float[batch][];
		for (int b = 0; b < batch; b++)
			G[b] = random(rng, rows);
		float[][] dX = CpuFrozenBatchOps.transpose(W, G, rows, cols);
		for (int b = 0; b < batch; b++) {
			float[] expected = GpuMatVecTransposeContractTest.scalarTransposeMatVec(W, G[b], rows, cols);
			assertThat(dX[b]).containsExactly(expected, within(0f));
		}
	}

	private static float[] random(Random rng, int n) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (rng.nextFloat() * 2f) - 1f;
		return a;
	}
}
