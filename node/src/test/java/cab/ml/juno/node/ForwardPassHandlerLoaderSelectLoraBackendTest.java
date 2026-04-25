package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link ForwardPassHandlerLoader#selectLoraBackend()} defaults an unset
 * {@code JUNO_USE_GPU} to enabling CUDA when available.
 */
@DisplayName("ForwardPassHandlerLoader — selectLoraBackend()")
class ForwardPassHandlerLoaderSelectLoraBackendTest {

	private String originalGpuFlag;

	@BeforeEach
	void saveFlag() {
		originalGpuFlag = System.getProperty("JUNO_USE_GPU");
	}

	@AfterEach
	void restoreFlag() {
		if (originalGpuFlag == null) {
			System.clearProperty("JUNO_USE_GPU");
		} else {
			System.setProperty("JUNO_USE_GPU", originalGpuFlag);
		}
	}

	@Test
	@DisplayName("JUNO_USE_GPU=false → CpuMatVec")
	void explicit_false_cpu() {
		System.setProperty("JUNO_USE_GPU", "false");
		assertThat(ForwardPassHandlerLoader.selectLoraBackend()).isInstanceOf(CpuMatVec.class);
	}

	@Test
	@DisplayName("JUNO_USE_GPU absent and no CUDA → CpuMatVec")
	void absent_no_cuda_cpu() {
		assumeFalse(CudaAvailability.isAvailable(), "CUDA present — skipping");
		System.clearProperty("JUNO_USE_GPU");
		assertThat(ForwardPassHandlerLoader.selectLoraBackend()).isInstanceOf(CpuMatVec.class);
	}

	@Test
	@Tag("gpu")
	@DisplayName("JUNO_USE_GPU absent and CUDA available → CudaMatVec")
	void absent_cuda_gpu() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		System.clearProperty("JUNO_USE_GPU");
		assertThat(ForwardPassHandlerLoader.selectLoraBackend()).isInstanceOf(CudaMatVec.class);
	}
}
