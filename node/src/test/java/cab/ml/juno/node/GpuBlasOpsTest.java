package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU microbatch GEMM parity vs {@link CpuFrozenBatchOps}.
 * Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=GpuBlasOpsTest}.
 */
@Tag("gpu")
@DisplayName("GpuBlasOps — FP32 microbatch (requires CUDA)")
class GpuBlasOpsTest {

	private static final float TOL = 1e-4f;

	private static GpuContext ctx;
	private static CudaMatVec cuda;
	private static GpuBlasOps ops;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		ctx = GpuContext.init(0);
		cuda = new CudaMatVec(ctx);
		ops = new GpuBlasOps(ctx);
	}

	@AfterAll
	static void destroy() {
		if (ops != null)
			ops.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("forward batch agrees with CpuFrozenBatchOps")
	void forward_parity() {
		Random rng = new Random(42);
		int rows = 17, cols = 13, batch = 8;
		float[] W = random(rng, rows * cols);
		float[][] X = new float[batch][];
		for (int b = 0; b < batch; b++)
			X[b] = random(rng, cols);

		float[][] expected = CpuFrozenBatchOps.forward(W, X, rows, cols);
		try (DeviceFloatMatrix dW = cuda.upload(W, rows, cols)) {
			float[][] actual = ops.forward(dW, X, batch);
			for (int b = 0; b < batch; b++)
				assertThat(actual[b]).containsExactly(expected[b], within(TOL));
		}
	}

	@Test
	@DisplayName("transpose batch agrees with CpuFrozenBatchOps")
	void transpose_parity() {
		Random rng = new Random(99);
		int rows = 11, cols = 19, batch = 8;
		float[] W = random(rng, rows * cols);
		float[][] G = new float[batch][];
		for (int b = 0; b < batch; b++)
			G[b] = random(rng, rows);

		float[][] expected = CpuFrozenBatchOps.transpose(W, G, rows, cols);
		try (DeviceFloatMatrix dW = cuda.upload(W, rows, cols)) {
			float[][] actual = ops.transpose(dW, G, batch);
			for (int b = 0; b < batch; b++)
				assertThat(actual[b]).containsExactly(expected[b], within(TOL));
		}
	}

	@Test
	@DisplayName("ResidentWeightMatrix FP32 uses batched GEMM")
	void resident_fp32_batch() {
		Random rng = new Random(3);
		int rows = 8, cols = 8, batch = 4;
		float[] W = random(rng, rows * cols);
		float[][] X = new float[batch][];
		for (int b = 0; b < batch; b++)
			X[b] = random(rng, cols);
		float[][] expected = CpuFrozenBatchOps.forward(W, X, rows, cols);

		try (ResidentWeightMatrix m = ResidentWeightMatrix.uploadFp32(cuda, W, rows, cols)) {
			assertThat(m.supportsBatchedSgemm()).isTrue();
			float[][] actual = m.sgemmBatch(ops, X, batch);
			for (int b = 0; b < batch; b++)
				assertThat(actual[b]).containsExactly(expected[b], within(TOL));
		}
	}

	private static float[] random(Random rng, int n) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (rng.nextFloat() * 2f) - 1f;
		return a;
	}
}
