package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * ROCm resident transpose tests. Run: {@code mvn test -Dgroups=rocm -pl node}.
 */
@Tag("rocm")
@DisplayName("RocmMatVec — resident transpose (requires ROCm)")
class RocmMatVecTransposeTest extends GpuMatVecTransposeContractTest {

	private static String prevBackend;
	private static GpuContext ctx;
	private static RocmMatVec rocm;

	@BeforeAll
	static void init() {
		assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
		prevBackend = System.getProperty("juno.gpu.backend");
		System.setProperty("juno.gpu.backend", "rocm");
		ctx = GpuContext.init(0);
		rocm = new RocmMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
		if (prevBackend == null)
			System.clearProperty("juno.gpu.backend");
		else
			System.setProperty("juno.gpu.backend", prevBackend);
	}

	@Override
	protected GpuMatVec impl() {
		return rocm;
	}

	@Override
	protected GpuContext ctx() {
		return ctx;
	}

	@Test
	@DisplayName("RocmBindings opNoTranspose=111 and opTranspose=112")
	void rocm_op_constants() {
		RocmBindings b = RocmBindings.instance();
		assertThat(b.opNoTranspose()).isEqualTo(111);
		assertThat(b.opTranspose()).isEqualTo(112);
	}
}
