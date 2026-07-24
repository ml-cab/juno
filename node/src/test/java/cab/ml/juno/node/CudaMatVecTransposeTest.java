package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * CUDA resident transpose tests. Run: {@code mvn test -Dgroups=gpu -pl node}.
 */
@Tag("gpu")
@DisplayName("CudaMatVec — resident transpose (requires CUDA)")
class CudaMatVecTransposeTest extends GpuMatVecTransposeContractTest {

	private static GpuContext ctx;
	private static CudaMatVec cuda;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		ctx = GpuContext.init(0);
		cuda = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Override
	protected GpuMatVec impl() {
		return cuda;
	}

	@Override
	protected GpuContext ctx() {
		return ctx;
	}

	@Test
	@DisplayName("CudaBindings opNoTranspose=0 and opTranspose=1")
	void cuda_op_constants() {
		CudaBindings b = CudaBindings.instance();
		assertThat(b.opNoTranspose()).isEqualTo(0);
		assertThat(b.opTranspose()).isEqualTo(1);
		assertThat(CudaBindings.CUBLAS_OP_N).isEqualTo(0);
		assertThat(CudaBindings.CUBLAS_OP_T).isEqualTo(1);
	}
}
